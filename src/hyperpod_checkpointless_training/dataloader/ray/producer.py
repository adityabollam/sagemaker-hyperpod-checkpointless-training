# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# type: ignore
"""Ray actor that prefetches batches on CPU nodes."""

# Standard Library
import threading
import time
import traceback
from typing import Callable, Iterable

# Third Party
import ray

# First Party
from hyperpod_checkpointless_training.dataloader.encryption.encryption_manager import (  # isort: skip
    KMSEncryptionManager,
)
from hyperpod_checkpointless_training.dataloader.ray.window import (  # isort: skip
    BatchWindow,
    COLD_START,
    WARM_START,
)
from hyperpod_checkpointless_training.inprocess.logger import get_logger  # isort: skip

logger = get_logger(__name__)

CPU_DATALOADER_RESOURCE = "hpct_cpu_dataloader"


def prefetch_actor_name(job_name: str, split: str, dp_rank: int) -> str:
    return f"hpct-pdl-{job_name}-{split}-dp{dp_rank}"


class PrefetchActor:
    """Produce and retain prefetched batches for registered consumers."""

    def __init__(
        self,
        dataloader_factory: Callable[..., Iterable],
        dp_rank: int,
        dp_size: int,
        lookback_length: int,
        prefetch_length: int,
        is_val: bool = False,
        enable_batch_encryption: bool = False,
        poll_interval_s: float = 0.01,
        consumer_timeout_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._dataloader_factory = dataloader_factory
        self._dp_rank = dp_rank
        self._dp_size = dp_size
        self._lookback_length = lookback_length
        self._prefetch_length = prefetch_length
        self._is_val = is_val
        self._enable_batch_encryption = enable_batch_encryption
        self._poll_interval_s = poll_interval_s
        self._consumer_timeout_s = consumer_timeout_s
        self._clock = clock

        self._lock = threading.Lock()
        self._window = BatchWindow(
            lookback_length,
            prefetch_length,
            consumer_timeout_s=consumer_timeout_s,
            clock=clock,
        )
        self._generation = 0
        self._iter = None
        self._length = None
        self._prebuilt = None
        self._error: tuple[int, str] | None = None
        self._last_reseek_token = None
        self._registration_counter = 0
        self._consumer_registration_ids: dict[int, int] = {}
        self._retired_final_steps: dict[int, int] = {}
        self._stop_event = threading.Event()
        self._wakeup = threading.Event()
        self._producer_thread = None
        self._producer_thread_generation = None
        self._encryption_manager = None
        self._shutdown_complete = False

    def register(
        self, consumer_id: int, step: int, reseek_token: str | None = None
    ) -> dict:
        with self._lock:
            self._registration_counter += 1
            registration_id = self._registration_counter
            self._consumer_registration_ids[consumer_id] = registration_id
            reason = None
            if self._error is not None and self._error[0] == self._generation:
                reason = "producer_error"
            elif self._window.producer_step is None:
                reason = "uninitialized"
            elif not self._window.can_resume_from(step):
                reason = "outside_window"
            elif reseek_token is not None and reseek_token != self._last_reseek_token:
                if (
                    self._is_val
                    and reseek_token == f"val-after-gen{self._generation - 1}"
                ):
                    self._last_reseek_token = reseek_token
                else:
                    reason = "reseek_token_changed"

            if reason is not None:
                known_consumers = dict(self._window.last_seen)
                if self._window.final_step is not None:
                    self._retired_final_steps[self._generation] = (
                        self._window.final_step
                    )
                    while len(self._retired_final_steps) > 8:
                        oldest_generation = next(iter(self._retired_final_steps))
                        self._retired_final_steps.pop(oldest_generation)
                for known in known_consumers:
                    if known != consumer_id:
                        self._consumer_registration_ids.pop(known, None)
                dropped = self._window.reset(step)
                del dropped
                for known_consumer_id, seen in known_consumers.items():
                    self._window.register_consumer(
                        known_consumer_id, step, last_seen=seen
                    )
                self._generation += 1
                self._error = None
                self._last_reseek_token = reseek_token
                prebuilt = self._prebuilt
                self._prebuilt = None
                try:
                    if prebuilt is not None and prebuilt[0] == step:
                        dataloader = prebuilt[1]
                    else:
                        dataloader = self._dataloader_factory(
                            dp_rank=self._dp_rank,
                            dp_size=self._dp_size,
                            target_load_step=step,
                            is_val=self._is_val,
                        )
                    self._iter = iter(dataloader)
                    try:
                        self._length = len(dataloader)
                    except TypeError:
                        self._length = None
                except Exception:
                    self._iter = None
                    self._length = None
                    self._error = (self._generation, traceback.format_exc())
                    logger.error(
                        "Failed to initialize dataloader for dp_rank=%s step=%s:\n%s",
                        self._dp_rank,
                        step,
                        self._error[1],
                    )

                logger.info(
                    "Reseek dataloader dp_rank=%s step=%s generation=%s reason=%s",
                    self._dp_rank,
                    step,
                    self._generation,
                    reason,
                )
                self._ensure_producer_thread()

            self._window.register_consumer(consumer_id, step)
            self._wakeup.set()
            mode = COLD_START if reason is not None else WARM_START
            return {
                "mode": mode,
                "length": self._length,
                "registration_id": registration_id,
                "generation": self._generation,
                **self._window.snapshot(),
            }

    def _ensure_producer_thread(self) -> None:
        if self._stop_event.is_set():
            return
        if (
            self._producer_thread is None
            or not self._producer_thread.is_alive()
            or self._producer_thread_generation != self._generation
        ):
            generation = self._generation
            self._producer_thread = threading.Thread(
                target=self._produce_loop,
                args=(generation,),
                name=f"hpct-prefetch-dp{self._dp_rank}",
                daemon=True,
            )
            self._producer_thread_generation = generation
            self._producer_thread.start()

    def get_ref(
        self,
        step: int,
        consumer_id: int | None = None,
        generation: int | None = None,
    ) -> tuple[str, list]:
        with self._lock:
            if consumer_id is not None:
                self._window.touch(consumer_id)
            if generation is not None and generation != self._generation:
                final_step = self._retired_final_steps.get(generation)
                if final_step is not None and step >= final_step:
                    return "final", []
                return "superseded", []
            status, ref = self._window.lookup(step)
            if status == "ready":
                return "ready", [ref]
            if self._error is not None and self._error[0] == self._generation:
                return "error", [self._error[1]]
            return status, []

    def ack(self, consumer_id: int, next_step: int, registration_id: int) -> None:
        with self._lock:
            if self._consumer_registration_ids.get(consumer_id) != registration_id:
                return
            evicted = self._window.ack(consumer_id, next_step)
            del evicted
            self._wakeup.set()

    def length(self) -> int | None:
        with self._lock:
            if self._generation == 0 and self._prebuilt is None:
                try:
                    dataloader = self._dataloader_factory(
                        dp_rank=self._dp_rank,
                        dp_size=self._dp_size,
                        target_load_step=0,
                        is_val=self._is_val,
                    )
                    self._prebuilt = (0, dataloader)
                    try:
                        self._length = len(dataloader)
                    except TypeError:
                        self._length = None
                except Exception:
                    logger.warning(
                        "Failed to prebuild dataloader for length dp_rank=%s:\n%s",
                        self._dp_rank,
                        traceback.format_exc(),
                    )
                    return None
            return self._length

    def stats(self) -> dict:
        with self._lock:
            error = (
                self._error[1]
                if self._error is not None and self._error[0] == self._generation
                else None
            )
            return {
                **self._window.snapshot(),
                "generation": self._generation,
                "error": error,
            }

    def shutdown(self) -> None:
        self._stop_event.set()
        self._wakeup.set()
        producer_thread = self._producer_thread
        if (
            producer_thread is not None
            and producer_thread is not threading.current_thread()
        ):
            producer_thread.join(timeout=5)

        with self._lock:
            if not self._shutdown_complete:
                self._generation += 1
                self._iter = None
                self._prebuilt = None
                self._window = BatchWindow(
                    self._lookback_length,
                    self._prefetch_length,
                    consumer_timeout_s=self._consumer_timeout_s,
                    clock=self._clock,
                )
                self._shutdown_complete = True

    def _get_encryption_manager(self):
        if self._encryption_manager is None:
            self._encryption_manager = KMSEncryptionManager()
        return self._encryption_manager

    def _produce_loop(self, generation: int) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                if generation != self._generation:
                    return
                can_produce = (
                    self._iter is not None
                    and not (
                        self._error is not None and self._error[0] == self._generation
                    )
                    and self._window.should_produce()
                )
                if can_produce:
                    iterator = self._iter
                    step = self._window.producer_step

            if not can_produce:
                self._wakeup.wait(self._poll_interval_s)
                self._wakeup.clear()
                continue

            try:
                batch = next(iterator)
                with self._lock:
                    if generation != self._generation:
                        return
            except StopIteration:
                with self._lock:
                    if generation != self._generation:
                        return
                    if self._window.producer_step == step:
                        self._window.mark_final()
                continue
            except Exception:
                error = traceback.format_exc()
                with self._lock:
                    if generation != self._generation:
                        return
                    self._error = (generation, error)
                logger.error(
                    "Failed to produce batch for dp_rank=%s step=%s:\n%s",
                    self._dp_rank,
                    step,
                    error,
                )
                continue

            try:
                payload = (
                    self._get_encryption_manager().encrypt(batch)
                    if self._enable_batch_encryption
                    else batch
                )
                ref = ray.put(payload)
            except Exception:
                error = traceback.format_exc()
                with self._lock:
                    if generation != self._generation:
                        return
                    self._error = (generation, error)
                logger.error(
                    "Failed to publish batch for dp_rank=%s step=%s:\n%s",
                    self._dp_rank,
                    step,
                    error,
                )
                continue

            with self._lock:
                if generation != self._generation:
                    return
                if self._window.producer_step == step:
                    self._window.add(step, ref)
                    ref = None
            del ref


def get_or_create_prefetch_actor(
    *,
    job_name: str,
    split: str,
    dp_rank: int,
    dp_size: int,
    dataloader_factory: Callable[..., Iterable],
    lookback_length: int,
    prefetch_length: int,
    namespace: str,
    num_cpus: float = 1.0,
    resources: dict | None = None,
    enable_batch_encryption: bool = False,
    consumer_timeout_s: float = 300.0,
):
    if resources is None:
        resources = {CPU_DATALOADER_RESOURCE: 0.001}
    return (
        ray.remote(PrefetchActor)
        .options(
            name=prefetch_actor_name(job_name, split, dp_rank),
            namespace=namespace,
            lifetime="detached",
            get_if_exists=True,
            max_concurrency=8,
            num_cpus=num_cpus,
            resources=resources,
            max_restarts=0,
        )
        .remote(
            dataloader_factory=dataloader_factory,
            dp_rank=dp_rank,
            dp_size=dp_size,
            lookback_length=lookback_length,
            prefetch_length=prefetch_length,
            is_val=split == "val",
            enable_batch_encryption=enable_batch_encryption,
            consumer_timeout_s=consumer_timeout_s,
        )
    )
