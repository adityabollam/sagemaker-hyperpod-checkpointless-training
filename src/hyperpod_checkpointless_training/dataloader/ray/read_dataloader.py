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
"""Ray Read Data Loader: consumer side of the Ray CPU-node dataloader."""

# Standard Library
import queue
import threading
import time
from typing import Any

# Third Party
import ray
from ray.exceptions import GetTimeoutError

# First Party
import hyperpod_checkpointless_training.dataloader.encryption.encryption_manager as encryption_manager
from hyperpod_checkpointless_training.inprocess.logger import get_logger

KMSEncryptionManager = encryption_manager.KMSEncryptionManager

# How often blocking loops wake up to re-check the stop event.
_STOP_CHECK_INTERVAL_S = 0.1
# How long stop() waits for the fetch thread to exit.
_STOP_JOIN_TIMEOUT_S = 5.0

_FINAL = object()


class _StopRequested(Exception):
    """Raised inside the fetch thread when stop() was requested."""


class _IterState:
    """Per-__iter__ state: stop event, prefetch queue and fetch thread."""

    def __init__(self, prefetch_length: int):
        self.stop_event = threading.Event()
        self.queue = queue.Queue(maxsize=max(1, prefetch_length))
        self.thread = None


class RayReadDataLoader:
    """Reads batches for one DP replica from its ``PrefetchActor``.

    Runs on every GPU rank. A background thread resolves the ObjectRefs for steps
    ``step, step + 1, ...`` from the actor ``prefetch_length`` steps ahead into a local
    queue; the training loop pops batches from that queue in order.

    fetch_thread:
    actor.get_ref(s) -> ray.get(ref) -> [decrypt] -> queue [(s, batch), ...]

    Main thread:
    queue -> yield batch -> step += 1 -> actor.ack(consumer_id, step) (fire-and-forget)

    If ``stop()`` is called while an iteration is in progress, the iterator raises
    ``RuntimeError`` on its next ``__next__`` (rather than ending cleanly) so the
    training loop never mistakes an abort for the end of an epoch.
    """

    def __init__(
        self,
        step: int,
        actor,
        consumer_id: int,
        prefetch_length: int,
        is_val: bool = False,
        enable_batch_encryption: bool = False,
        poll_interval_s: float = 0.05,
        rpc_timeout_s: float = 60.0,
        register_timeout_s: float = 900.0,
    ):
        """Initialize the Ray Read Data Loader.

        Args:
            step: The current/resumed training step (ignored for validation, which starts at 0).
            actor: ActorHandle of the ``PrefetchActor`` for this rank's DP replica.
            consumer_id: Unique id of this consumer (global rank).
            prefetch_length: Max number of batches buffered locally ahead of the training loop.
            is_val: Whether this is a validation dataloader (restarts from step 0 on every __iter__).
            enable_batch_encryption: Whether batches are KMS-encrypted by the producer.
                If enabled, assumes AWS_KMS_KEY_ID is set in the environment.
            poll_interval_s: Sleep between ``get_ref`` polls while a step is pending.
            rpc_timeout_s: Timeout for every other ``ray.get``.
            register_timeout_s: Timeout for the register and length RPCs; the first register (and a
                val length()) builds the dataloader on the CPU node, which can take minutes.
        """
        self._logger = get_logger(__name__)
        self._actor = actor
        self._consumer_id = consumer_id
        self._prefetch_length = prefetch_length
        self._is_val = is_val
        self._enable_batch_encryption = enable_batch_encryption
        self._poll_interval_s = poll_interval_s
        self._rpc_timeout_s = rpc_timeout_s
        self._register_timeout_s = register_timeout_s

        self._step = 0 if is_val else step
        self._length = None
        self._num_iters = 0
        self._encryption_manager = None
        self._state = None
        self._state_lock = threading.Lock()
        self.restart_info = None

        if not is_val:
            self.restart_info = self._register(self._step, None)

    @property
    def step(self) -> int:
        """Next step this consumer will yield."""
        return self._step

    def set_length(self, length: int) -> None:
        """Set the length of the dataloader."""
        self._length = length

    def __len__(self) -> int:
        if self._length is not None:
            return self._length
        if (
            self.restart_info is not None
            and self.restart_info.get("length") is not None
        ):
            return self.restart_info["length"]
        raise TypeError(f"{self.__class__.__name__} length is unknown")

    def remote_length(self) -> int | None:
        """Length reported by the producer actor (``len`` of the CPU-node dataloader)."""
        if not self._is_val and self.restart_info is not None:
            return self.restart_info.get("length")
        return ray.get(self._actor.length.remote(), timeout=self._register_timeout_s)

    def _register(self, step: int, reseek_token: str | None) -> dict:
        info = ray.get(
            self._actor.register.remote(self._consumer_id, step, reseek_token),
            timeout=self._register_timeout_s,
        )
        self._logger.info(
            f"[{self._consumer_id}] Registered with prefetch actor at step={step} "
            f"reseek_token={reseek_token}: {info}"
        )
        return info

    def _get_encryption_manager(self):
        if self._encryption_manager is None:
            self._encryption_manager = KMSEncryptionManager()
            self._logger.info(
                f"[{self._consumer_id}] Setup encryption manager with "
                f"AWS_KMS_KEY_ID={self._encryption_manager.kms_key_id}."
            )
        return self._encryption_manager

    def __iter__(self):
        self.stop()
        if self._is_val:
            self._step = 0
            self.restart_info = self._register(0, f"val-{self._num_iters}")
        self._num_iters += 1

        state = _IterState(self._prefetch_length)
        state.thread = threading.Thread(
            target=self._fetch_loop,
            args=(state, self._step),
            name=f"RayReadDataLoader-fetch-{self._consumer_id}",
            daemon=True,
        )
        with self._state_lock:
            self._state = state
        state.thread.start()
        return self._generate(state)

    def _generate(self, state: _IterState):
        try:
            while True:
                item = self._next_item(state)
                if item is _FINAL:
                    self._logger.debug(
                        f"[{self._consumer_id}] Reached final step {self._step}."
                    )
                    return
                if isinstance(item, BaseException):
                    raise item
                step, batch = item
                assert (
                    step == self._step
                ), f"Out-of-order batch: expected step {self._step}, got {step}"
                del item
                yield batch
                del batch
                # A stopped (or superseded) iteration must not touch the shared step/acks.
                self._raise_if_stopped(state)
                self._step += 1
                self._ack()
        finally:
            self._stop_state(state)

    def _raise_if_stopped(self, state: _IterState) -> None:
        if state.stop_event.is_set():
            raise RuntimeError(
                f"[{self._consumer_id}] {self.__class__.__name__} was stopped "
                f"at step {self._step}."
            )

    def _next_item(self, state: _IterState) -> Any:
        """Pop the next item from the queue, never blocking forever."""
        while True:
            self._raise_if_stopped(state)
            try:
                return state.queue.get(timeout=_STOP_CHECK_INTERVAL_S)
            except queue.Empty:
                self._raise_if_stopped(state)
                if not state.thread.is_alive():
                    try:
                        return state.queue.get_nowait()
                    except queue.Empty:
                        raise RuntimeError(
                            f"[{self._consumer_id}] Fetch thread exited without "
                            f"producing step {self._step}."
                        ) from None

    def _ack(self) -> None:
        try:
            self._actor.ack.remote(self._consumer_id, self._step)
        except Exception as e:
            self._logger.warning(
                f"[{self._consumer_id}] Failed to ack step {self._step}: {e}"
            )

    def _rpc_get(self, ref, stop_event: threading.Event) -> Any:
        """ray.get with rpc_timeout_s that wakes up periodically to honour stop()."""
        deadline = time.monotonic() + self._rpc_timeout_s
        while True:
            if stop_event.is_set():
                raise _StopRequested()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GetTimeoutError(
                    f"[{self._consumer_id}] ray.get timed out after {self._rpc_timeout_s}s"
                )
            ready, _ = ray.wait([ref], timeout=min(_STOP_CHECK_INTERVAL_S, remaining))
            if ready:
                return ray.get(ref, timeout=self._rpc_timeout_s)

    def _put(self, state: _IterState, item: Any) -> bool:
        """Put an item on the queue; returns False if stop() was requested."""
        while not state.stop_event.is_set():
            try:
                state.queue.put(item, timeout=_STOP_CHECK_INTERVAL_S)
                return True
            except queue.Full:
                continue
        return False

    def _fetch_loop(self, state: _IterState, step: int) -> None:
        try:
            while not state.stop_event.is_set():
                status, payload = self._rpc_get(
                    self._actor.get_ref.remote(step), state.stop_event
                )
                if status == "pending":
                    state.stop_event.wait(self._poll_interval_s)
                elif status == "ready":
                    batch = self._rpc_get(payload[0], state.stop_event)
                    if self._enable_batch_encryption:
                        batch = self._get_encryption_manager().decrypt(batch)
                    if not self._put(state, (step, batch)):
                        return
                    del batch
                    step += 1
                elif status == "final":
                    self._put(state, _FINAL)
                    return
                elif status == "evicted":
                    raise RuntimeError(
                        f"[{self._consumer_id}] Step {step} was evicted from the prefetch "
                        "actor window; it is older than the window's low watermark "
                        "(consider increasing lookback_length)."
                    )
                elif status == "error":
                    detail = payload[0] if payload else "<no traceback>"
                    raise RuntimeError(
                        f"[{self._consumer_id}] Prefetch actor failed to produce step "
                        f"{step}:\n{detail}"
                    )
                else:
                    raise RuntimeError(
                        f"[{self._consumer_id}] Unknown get_ref status {status!r} "
                        f"for step {step}"
                    )
        except _StopRequested:
            return
        except BaseException as e:
            self._logger.error(
                f"[{self._consumer_id}] Fetch thread failed at step {step}: {e!r}"
            )
            self._put(state, e)

    def _stop_state(self, state: _IterState | None) -> None:
        if state is None:
            return
        state.stop_event.set()
        thread = state.thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
            if thread.is_alive():
                self._logger.warning(
                    f"[{self._consumer_id}] Fetch thread did not exit within "
                    f"{_STOP_JOIN_TIMEOUT_S}s."
                )
        with self._state_lock:
            if self._state is state:
                self._state = None

    def stop(self) -> None:
        """Stop the fetch thread. Idempotent, thread-safe and never raises."""
        try:
            with self._state_lock:
                state = self._state
            self._stop_state(state)
        except Exception as e:
            self._logger.warning(
                f"[{self._consumer_id}] Error while stopping {self.__class__.__name__}: {e}"
            )
