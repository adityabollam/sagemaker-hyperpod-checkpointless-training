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
"""Local-Ray end-to-end coverage for CPU dataloader recovery."""

# Standard Library
import importlib
import queue
import sys
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

ray = pytest.importorskip("ray")

config_module = importlib.import_module(
    "hyperpod_checkpointless_training.dataloader.config"
)
RayCPUNodeConfig = config_module.RayCPUNodeConfig
prefetch_actor_name = importlib.import_module(
    "hyperpod_checkpointless_training.dataloader.ray.producer"
).prefetch_actor_name


@pytest.fixture(scope="module", autouse=True)
def local_ray():
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        namespace=f"e2e-{uuid.uuid4().hex}",
        ignore_reinit_error=True,
    )
    yield
    ray.shutdown()
    ray.cloudpickle.unregister_pickle_by_value(sys.modules[__name__])


class StepIterable:
    def __init__(self, n, dp_rank, target_load_step, raise_at=None, block_at=None):
        self.n = n
        self.dp_rank = dp_rank
        self.target_load_step = target_load_step
        self.raise_at = raise_at
        self.block_at = block_at

    def __len__(self):
        return self.n

    def __iter__(self):
        for step in range(self.target_load_step, self.n):
            if step == self.raise_at:
                raise RuntimeError(f"boom at step {step}")
            if step == self.block_at:
                time.sleep(3600)
            yield {
                "tokens": torch.full((2, 4), self.dp_rank * 1000 + step),
                "step": step,
            }


class StepFactory:
    def __init__(self, n=20, calls_path=None, raise_at=None, block_at=None):
        self.n = n
        self.calls_path = calls_path
        self.raise_at = raise_at
        self.block_at = block_at

    def __call__(self, *, dp_rank, dp_size, target_load_step, is_val):
        with open(self.calls_path, "a", encoding="utf-8") as calls_file:
            calls_file.write(f"{dp_rank},{target_load_step},{is_val}\n")
        return StepIterable(
            self.n,
            dp_rank,
            target_load_step,
            raise_at=self.raise_at,
            block_at=self.block_at,
        )


def wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def next_with_deadline(iterator, timeout):
    result = queue.Queue()

    def read_next():
        try:
            result.put((True, next(iterator)))
        except BaseException as error:
            result.put((False, error))

    thread = threading.Thread(target=read_next, daemon=True)
    thread.start()
    try:
        succeeded, value = result.get(timeout=timeout)
    except queue.Empty:
        pytest.fail(f"next() did not return within {timeout:.2f} seconds")
    if succeeded:
        return value
    raise value


def assert_batch(batch, step, dp_rank=0):
    assert batch["step"] == step
    assert torch.equal(batch["tokens"], torch.full((2, 4), dp_rank * 1000 + step))


@pytest.fixture
def e2e(tmp_path):
    loaders = []
    actors = set()

    def build(*, n=20, raise_at=None, block_at=None):
        job_token = uuid.uuid4().hex
        calls_path = tmp_path / f"{job_token}-factory-calls.txt"
        factory = StepFactory(
            n=n,
            calls_path=str(calls_path),
            raise_at=raise_at,
            block_at=block_at,
        )
        config = RayCPUNodeConfig(
            cpu_dataloader_factory=factory,
            job_name=f"job-{job_token}",
            ray_namespace=f"ns-{job_token}",
            prefetch_length=4,
            val_prefetch_length=4,
            lookback_length=2,
            actor_resources={},
        )

        def create(
            step,
            dp_rank=0,
            dp_size=1,
            consumer_id=0,
            is_val=False,
        ):
            split = "val" if is_val else "train"
            actor_name = prefetch_actor_name(config.job_name, split, dp_rank)
            actors.add((actor_name, config.ray_namespace))
            parallel_state_util = MagicMock()
            parallel_state_util.parallel_state.get_data_parallel_rank.return_value = (
                dp_rank
            )
            parallel_state_util.parallel_state.get_data_parallel_world_size.return_value = (
                dp_size
            )
            with patch.object(config_module.dist, "get_rank", return_value=consumer_id):
                loader = config.create(
                    None,
                    parallel_state_util,
                    step,
                    None,
                    None,
                    name=f"job-{job_token}",
                    is_val=is_val,
                )
            loaders.append(loader)
            return loader

        def read_calls():
            if not calls_path.exists():
                return []
            return [
                (int(rank), int(step), is_val == "True")
                for rank, step, is_val in (
                    line.strip().split(",")
                    for line in calls_path.read_text(encoding="utf-8").splitlines()
                )
            ]

        return SimpleNamespace(
            create=create,
            factory=factory,
            calls=read_calls,
            actor_name=lambda split="train", dp_rank=0: prefetch_actor_name(
                config.job_name, split, dp_rank
            ),
            namespace=config.ray_namespace,
        )

    yield build

    for loader in loaders:
        loader.stop()
    for name, namespace in actors:
        try:
            ray.kill(ray.get_actor(name, namespace=namespace), no_restart=True)
        except ValueError:
            pass


def test_cold_start_yields_steps_and_tokens(e2e):
    harness = e2e()
    loader = harness.create(step=0)

    assert len(loader) == 20
    assert loader.restart_info["mode"] == "cold_start"
    iterator = iter(loader)
    for step in range(6):
        assert_batch(next(iterator), step)
    assert harness.calls() == [(0, 0, False)]


def test_warm_restart_reuses_prefetched_batches(e2e):
    harness = e2e()
    loader = harness.create(step=0)
    iterator = iter(loader)
    for step in range(8):
        assert_batch(next(iterator), step)
    loader.stop()
    del iterator, loader

    resumed = harness.create(step=6)

    assert resumed.restart_info["mode"] == "warm_start"
    assert_batch(next(iter(resumed)), 6)
    assert harness.calls() == [(0, 0, False)]


def test_outside_window_reseeks_from_requested_steps(e2e):
    harness = e2e()
    loader = harness.create(step=0)
    iterator = iter(loader)
    for step in range(11):
        assert_batch(next(iterator), step)
    loader.stop()
    del iterator, loader

    resumed = harness.create(step=2)
    assert resumed.restart_info["mode"] == "cold_start"
    iterator = iter(resumed)
    assert_batch(next(iterator), 2)
    resumed.stop()
    del iterator, resumed

    later = harness.create(step=18)
    assert later.restart_info["mode"] == "cold_start"
    iterator = iter(later)
    assert_batch(next(iterator), 18)
    assert_batch(next(iterator), 19)
    with pytest.raises(StopIteration):
        next(iterator)
    assert harness.calls() == [(0, 0, False), (0, 2, False), (0, 18, False)]


def test_actor_loss_recreates_and_resumes_from_checkpoint(e2e):
    harness = e2e()
    loader = harness.create(step=0)
    iterator = iter(loader)
    before_kill = {}
    for step in range(4):
        batch = next(iterator)
        assert_batch(batch, step)
        before_kill[step] = batch

    actor = ray.get_actor(harness.actor_name(), namespace=harness.namespace)
    ray.kill(actor, no_restart=True)

    expected_step = 4
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            batch = next_with_deadline(iterator, deadline - time.monotonic())
        except (RuntimeError, ray.exceptions.RayError):
            break
        assert_batch(batch, expected_step)
        before_kill[expected_step] = batch
        expected_step += 1
    else:
        pytest.fail("next() did not report actor loss within 5 seconds")

    loader.stop()
    del iterator, loader

    attempts = 0
    retry_deadline = time.monotonic() + 5.0
    while time.monotonic() < retry_deadline:
        attempts += 1
        try:
            resumed = harness.create(step=2)
            break
        except ray.exceptions.RayActorError:
            time.sleep(0.05)
    else:
        pytest.fail("could not recreate the prefetch actor within 5 seconds")

    print(f"actor recreation attempts: {attempts}")
    assert resumed.restart_info["mode"] == "cold_start"
    resumed_iterator = iter(resumed)
    for step in (2, 3):
        batch = next(resumed_iterator)
        assert_batch(batch, step)
        assert torch.equal(batch["tokens"], before_kill[step]["tokens"])
    assert harness.calls()[-1] == (0, 2, False)


def test_tensor_parallel_siblings_share_window_and_progress(e2e):
    harness = e2e()
    fast = harness.create(step=0, consumer_id=0)
    slow = harness.create(step=0, consumer_id=1)
    fast_iterator = iter(fast)
    slow_iterator = iter(slow)

    for step in range(8):
        fast_batch = next(fast_iterator)
        slow_batch = next(slow_iterator)
        assert_batch(fast_batch, step)
        assert_batch(slow_batch, step)
        assert torch.equal(fast_batch["tokens"], slow_batch["tokens"])

    for step in (8, 9):
        assert_batch(next(fast_iterator), step)

    actor = ray.get_actor(harness.actor_name(), namespace=harness.namespace)
    stats = None

    def both_consumers_advanced():
        nonlocal stats
        stats = ray.get(actor.stats.remote())
        return (
            stats["consumer_steps"].get(0) == 9 and stats["consumer_steps"].get(1) == 7
        )

    assert wait_until(both_consumers_advanced, timeout=5.0), stats
    assert stats["low_watermark"] <= max(0, min(stats["consumer_steps"].values()) - 2)
    status, refs = ray.get(actor.get_ref.remote(slow.step))
    assert status == "ready"
    assert_batch(ray.get(refs[0]), slow.step)


def test_data_parallel_replicas_use_separate_actors(e2e):
    harness = e2e()
    rank_zero = harness.create(step=0, dp_rank=0, dp_size=2)
    rank_one = harness.create(step=0, dp_rank=1, dp_size=2)

    name_zero = harness.actor_name(dp_rank=0)
    name_one = harness.actor_name(dp_rank=1)
    assert name_zero != name_one
    ray.get_actor(name_zero, namespace=harness.namespace)
    ray.get_actor(name_one, namespace=harness.namespace)

    iterator_zero = iter(rank_zero)
    iterator_one = iter(rank_one)
    for step in range(6):
        assert_batch(next(iterator_zero), step, dp_rank=0)
        assert_batch(next(iterator_one), step, dp_rank=1)


def test_validation_length_repeats_and_siblings_restart_together(e2e):
    harness = e2e()
    first = harness.create(step=0, consumer_id=0, is_val=True)

    assert len(first) == 20
    for _ in range(2):
        batches = list(first)
        assert [batch["step"] for batch in batches] == list(range(20))
        for step, batch in enumerate(batches):
            assert_batch(batch, step)

    second = harness.create(step=0, consumer_id=1, is_val=True)
    for _ in range(2):
        for step, (batch_first, batch_second) in enumerate(
            zip(iter(first), iter(second))
        ):
            assert_batch(batch_first, step)
            assert_batch(batch_second, step)
            assert torch.equal(batch_first["tokens"], batch_second["tokens"])


def test_producer_error_preserves_already_produced_steps(e2e):
    harness = e2e(raise_at=3)
    loader = harness.create(step=0)
    iterator = iter(loader)

    for step in range(3):
        assert_batch(next(iterator), step)
    with pytest.raises(RuntimeError, match="boom at step 3"):
        next(iterator)


def test_stop_interrupts_blocked_producer_promptly(e2e):
    harness = e2e(block_at=2)
    loader = harness.create(step=0)
    iterator = iter(loader)
    assert_batch(next(iterator), 0)
    assert_batch(next(iterator), 1)
    stop_duration = []

    def stop_after_delay():
        time.sleep(0.3)
        start = time.monotonic()
        loader.stop()
        stop_duration.append(time.monotonic() - start)

    stopper = threading.Thread(target=stop_after_delay)
    stopper.start()
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="stopped"):
        next(iterator)
    assert time.monotonic() - start < 6.0
    stopper.join(timeout=6.0)
    assert not stopper.is_alive()
    assert stop_duration and stop_duration[0] < 6.0
