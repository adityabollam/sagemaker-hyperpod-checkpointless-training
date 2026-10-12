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
"""Tests for Ray dataloader producer actors."""

# Standard Library
import importlib
import sys
import threading
import time
import uuid
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

ray = pytest.importorskip("ray")

# First Party
producer_module = importlib.import_module(
    "hyperpod_checkpointless_training.dataloader.ray.producer"
)
CPU_DATALOADER_RESOURCE = producer_module.CPU_DATALOADER_RESOURCE
PrefetchActor = producer_module.PrefetchActor
get_or_create_prefetch_actor = producer_module.get_or_create_prefetch_actor
prefetch_actor_name = producer_module.prefetch_actor_name


class StepDataLoader:
    def __init__(self, n, start, fail_at=None):
        self.n = n
        self.start = start
        self.fail_at = fail_at

    def __len__(self):
        return self.n

    def __iter__(self):
        for step in range(self.start, self.n):
            if step == self.fail_at:
                raise RuntimeError(f"failure at step {step}")
            yield {"tokens": torch.full((2, 4), step)}


class StepFactory:
    def __init__(self, n, fail_at=None, raise_in_factory=False):
        self.n = n
        self.fail_at = fail_at
        self.raise_in_factory = raise_in_factory
        self.calls = []

    def __call__(self, *, dp_rank, dp_size, target_load_step, is_val):
        self.calls.append(
            {
                "dp_rank": dp_rank,
                "dp_size": dp_size,
                "target_load_step": target_load_step,
                "is_val": is_val,
            }
        )
        if self.raise_in_factory:
            raise RuntimeError("factory failure")
        return StepDataLoader(self.n, target_load_step, self.fail_at)


class FailOnceAtStepFactory:
    def __init__(self, n, fail_at):
        self.n = n
        self.fail_at = fail_at
        self.failed = False

    def __call__(self, *, dp_rank, dp_size, target_load_step, is_val):
        factory = self

        class FailOnceDataLoader:
            def __len__(self):
                return factory.n

            def __iter__(self):
                for step in range(target_load_step, factory.n):
                    if step == factory.fail_at and not factory.failed:
                        factory.failed = True
                        raise RuntimeError(f"failure at step {step}")
                    yield {"tokens": torch.full((2, 4), step)}

        return FailOnceDataLoader()


class NoLengthFactory:
    def __init__(self):
        self.calls = []

    def __call__(self, *, dp_rank, dp_size, target_load_step, is_val):
        self.calls.append(target_load_step)
        return iter([{"tokens": torch.full((2, 4), target_load_step)}])


class BlockingFactory:
    def __init__(self, n):
        self.n = n
        self.started = threading.Event()
        self.release = threading.Event()
        self._block_first = True

    def __call__(self, *, dp_rank, dp_size, target_load_step, is_val):
        factory = self

        class BlockingDataLoader:
            def __len__(self):
                return factory.n

            def __iter__(self):
                for step in range(target_load_step, factory.n):
                    if step == 0 and factory._block_first:
                        factory._block_first = False
                        factory.started.set()
                        factory.release.wait(timeout=5)
                    yield {"tokens": torch.full((2, 4), step)}

        return BlockingDataLoader()


class GenerationBlockingFactory:
    def __init__(self, n, blocked_generation, blocked_step):
        self.n = n
        self.blocked_generation = blocked_generation
        self.blocked_step = blocked_step
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, *, dp_rank, dp_size, target_load_step, is_val):
        self.calls += 1
        generation = self.calls
        factory = self

        class GenerationBlockingDataLoader:
            def __iter__(self):
                for step in range(target_load_step, factory.n):
                    if (
                        generation == factory.blocked_generation
                        and step == factory.blocked_step
                    ):
                        factory.started.set()
                        factory.release.wait()
                    yield {"tokens": torch.full((2, 4), step)}

        return GenerationBlockingDataLoader()


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def actor_factory(monkeypatch):
    monkeypatch.setattr(producer_module.ray, "put", lambda payload: payload)
    actors = []

    def create(factory, **kwargs):
        actor = PrefetchActor(
            dataloader_factory=factory,
            dp_rank=kwargs.pop("dp_rank", 0),
            dp_size=kwargs.pop("dp_size", 1),
            lookback_length=kwargs.pop("lookback_length", 1),
            prefetch_length=kwargs.pop("prefetch_length", 3),
            is_val=kwargs.pop("is_val", False),
            enable_batch_encryption=kwargs.pop("enable_batch_encryption", False),
            poll_interval_s=kwargs.pop("poll_interval_s", 0.005),
            consumer_timeout_s=kwargs.pop("consumer_timeout_s", 300.0),
            clock=kwargs.pop("clock", time.monotonic),
        )
        assert not kwargs
        actors.append(actor)
        return actor

    yield create

    for actor in actors:
        actor.shutdown()


def test_first_register_is_cold_and_builds_dataloader(actor_factory):
    factory = StepFactory(5)
    actor = actor_factory(factory, dp_rank=2, dp_size=4, prefetch_length=8)

    result = actor.register(11, 1)

    assert result["mode"] == "cold_start"
    assert result["length"] == 5
    assert factory.calls == [
        {
            "dp_rank": 2,
            "dp_size": 4,
            "target_load_step": 1,
            "is_val": False,
        }
    ]
    assert wait_until(lambda: actor.stats()["producer_step"] == 5)


def test_length_prebuild_is_reused_by_first_register(actor_factory):
    factory = StepFactory(4)
    actor = actor_factory(factory)

    assert actor.length() == 4
    assert factory.calls[0]["target_load_step"] == 0
    assert actor._producer_thread is None

    result = actor.register(2, 0)

    assert result["mode"] == "cold_start"
    assert result["length"] == 4
    assert len(factory.calls) == 1
    assert wait_until(lambda: actor.get_ref(0)[0] == "ready")
    assert torch.equal(actor.get_ref(0)[1][0]["tokens"], torch.full((2, 4), 0))


def test_length_prebuild_is_discarded_when_register_reseeks_elsewhere(
    actor_factory,
):
    factory = StepFactory(8)
    actor = actor_factory(factory)

    assert actor.length() == 8
    result = actor.register(2, 5)

    assert result["mode"] == "cold_start"
    assert [call["target_load_step"] for call in factory.calls] == [0, 5]
    assert wait_until(lambda: actor.get_ref(5)[0] == "ready")
    assert torch.equal(actor.get_ref(5)[1][0]["tokens"], torch.full((2, 4), 5))


def test_length_prebuild_handles_loader_without_len(actor_factory):
    factory = NoLengthFactory()
    actor = actor_factory(factory)

    assert actor.length() is None
    assert factory.calls == [0]
    assert actor._producer_thread is None


def test_length_after_register_uses_cached_length(actor_factory):
    factory = StepFactory(6)
    actor = actor_factory(factory)

    actor.register(0, 0)

    assert actor.length() == 6
    assert len(factory.calls) == 1


def test_warm_register_keeps_refs_and_does_not_rebuild(actor_factory):
    factory = StepFactory(6)
    actor = actor_factory(factory, prefetch_length=4)
    actor.register(1, 0)
    assert wait_until(lambda: actor.stats()["producer_step"] == 4)
    refs_before = dict(actor._window.refs)

    result = actor.register(1, 1)

    assert result["mode"] == "warm_start"
    assert len(factory.calls) == 1
    assert actor._window.refs == refs_before
    assert all(actor._window.refs[step] is refs_before[step] for step in refs_before)


def test_register_outside_window_reseeks(actor_factory):
    factory = StepFactory(20)
    actor = actor_factory(factory, prefetch_length=4)
    actor.register(1, 0)
    assert wait_until(lambda: actor.stats()["producer_step"] == 4)
    generation = actor.stats()["generation"]

    result = actor.register(1, 10)

    assert result["mode"] == "cold_start"
    assert actor.stats()["generation"] == generation + 1
    assert factory.calls[-1]["target_load_step"] == 10


def test_reseek_token_controls_validation_reseeks(actor_factory):
    factory = StepFactory(4)
    actor = actor_factory(factory, is_val=True)

    first = actor.register(3, 0, "val-0")
    first_generation = actor.stats()["generation"]
    second = actor.register(3, 0, "val-0")
    third = actor.register(3, 0, "val-1")

    assert first["mode"] == "cold_start"
    assert second["mode"] == "warm_start"
    assert third["mode"] == "cold_start"
    assert actor.stats()["generation"] == first_generation + 1
    assert all(call["is_val"] for call in factory.calls)
    assert [call["target_load_step"] for call in factory.calls] == [0, 0]


def test_validation_reseek_preserves_sibling_consumers(actor_factory):
    factory = StepFactory(12)
    actor = actor_factory(factory, is_val=True, lookback_length=2, prefetch_length=10)

    registration_a = actor.register("A", 0, "val-0")
    assert registration_a["mode"] == "cold_start"
    assert actor.register("B", 0, "val-0")["mode"] == "warm_start"
    factory_calls = len(factory.calls)
    result = actor.register("A", 0, "val-1")

    assert result["mode"] == "cold_start"
    assert len(factory.calls) == factory_calls + 1
    assert actor._window.consumer_steps == {"A": 0, "B": 0}
    assert wait_until(lambda: actor.stats()["producer_step"] >= 5)

    actor.ack("A", 4, registration_a["registration_id"])
    actor.ack("A", 5, registration_a["registration_id"])

    assert actor.get_ref(0)[0] == "ready"
    assert actor.register("B", 0, "val-1")["mode"] == "warm_start"
    assert len(factory.calls) == factory_calls + 1
    assert actor.get_ref(0)[0] == "ready"


def test_train_out_of_window_reseek_resets_sibling_consumers(actor_factory):
    actor = actor_factory(StepFactory(30), lookback_length=2, prefetch_length=10)
    actor.register("A", 0)
    actor.register("B", 0)
    assert wait_until(lambda: actor.stats()["producer_step"] >= 5)

    result = actor.register("A", 20)

    assert result["mode"] == "cold_start"
    assert actor._window.consumer_steps == {"A": 20, "B": 20}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_expired_consumer_does_not_block_production(actor_factory):
    clock = FakeClock()
    actor = actor_factory(
        StepFactory(50),
        lookback_length=1,
        prefetch_length=3,
        consumer_timeout_s=10.0,
        clock=clock,
    )
    registration_a = actor.register("A", 0)
    actor.register("B", 0)
    assert wait_until(lambda: actor.stats()["producer_step"] == 3)

    # B polls get_ref (refreshing its lease); production stays capped at B's step.
    for _ in range(3):
        clock.advance(6)
        assert actor.get_ref(0, consumer_id="B")[0] == "ready"
        actor.ack("A", 3, registration_a["registration_id"])

    time.sleep(0.1)
    assert actor.stats()["producer_step"] == 3

    # B stops polling and its lease expires; production now follows A alone.
    clock.advance(11)
    actor.ack("A", 3, registration_a["registration_id"])
    assert wait_until(lambda: actor.stats()["producer_step"] == 6)

    # Positional get_ref still works and unknown consumer ids add nothing.
    assert actor.get_ref(0)[0] in ("ready", "evicted")
    consumers_before = dict(actor._window.consumer_steps)
    actor.get_ref(0, consumer_id="unknown")
    assert actor._window.consumer_steps == consumers_before


def test_reseek_does_not_revive_expired_consumer(actor_factory):
    clock = FakeClock()
    actor = actor_factory(
        StepFactory(50),
        lookback_length=1,
        prefetch_length=3,
        consumer_timeout_s=10.0,
        clock=clock,
    )
    actor.register("A", 0)
    actor.register("B", 0)
    assert wait_until(lambda: actor.stats()["producer_step"] == 3)

    clock.advance(11)
    result = actor.register("A", 40)

    assert result["mode"] == "cold_start"
    assert actor._window.consumer_steps == {"A": 40, "B": 40}
    assert actor._window.live_consumer_steps() == [40]


def test_backpressure_ack_and_eviction(actor_factory):
    factory = StepFactory(20)
    actor = actor_factory(factory, lookback_length=1, prefetch_length=3)
    registration = actor.register(1, 0)
    assert wait_until(lambda: actor.stats()["producer_step"] == 3)
    time.sleep(0.1)
    assert actor.stats()["producer_step"] == 3

    actor.ack(1, 3, registration["registration_id"])

    assert wait_until(lambda: actor.stats()["producer_step"] == 6)
    assert actor.stats()["low_watermark"] == 2


def test_ack_from_previous_registration_does_not_evict_window(actor_factory):
    actor = actor_factory(StepFactory(120), lookback_length=2, prefetch_length=30)
    actor.register("A", 88)
    assert wait_until(lambda: actor.stats()["producer_step"] >= 118)

    old_registration = actor.register("A", 100)
    new_registration = actor.register("A", 90)

    assert old_registration["registration_id"] != new_registration["registration_id"]
    actor.ack("A", 101, old_registration["registration_id"])

    assert actor._window.consumer_steps["A"] == 90
    assert [actor.get_ref(step)[0] for step in range(88, 91)] == [
        "ready",
        "ready",
        "ready",
    ]


def test_ready_refs_and_exhaustion(actor_factory):
    actor = actor_factory(StepFactory(3), prefetch_length=4)
    actor.register(0, 0)

    assert wait_until(lambda: actor.stats()["final_step"] == 3)
    for step in range(3):
        status, refs = actor.get_ref(step)
        assert status == "ready"
        assert torch.equal(refs[0]["tokens"], torch.full((2, 4), step))
    assert actor.get_ref(3) == ("final", [])


def test_factory_exception_is_reported_as_error(actor_factory):
    actor = actor_factory(StepFactory(4, raise_in_factory=True))

    actor.register(0, 0)

    status, payload = actor.get_ref(0)
    assert status == "error"
    assert "factory failure" in payload[0]
    assert actor.stats()["error"] == payload[0]


def test_iterator_exception_is_reported_and_reseek_clears_error(actor_factory):
    factory = StepFactory(5, fail_at=1)
    actor = actor_factory(factory, prefetch_length=4)
    actor.register(0, 0)

    assert wait_until(lambda: actor.get_ref(1)[0] == "error")
    status, payload = actor.get_ref(1)
    assert status == "error"
    assert "failure at step 1" in payload[0]

    factory.fail_at = None
    result = actor.register(0, 0, reseek_token="recovered")

    assert result["mode"] == "cold_start"
    assert wait_until(lambda: actor.get_ref(0)[0] == "ready")


def test_register_reseeks_after_current_generation_producer_error(actor_factory):
    factory = FailOnceAtStepFactory(8, fail_at=4)
    actor = actor_factory(factory, prefetch_length=5)
    assert actor.register(0, 3)["mode"] == "cold_start"

    assert wait_until(lambda: actor.get_ref(4)[0] == "error")
    assert actor.get_ref(3)[0] == "ready"

    result = actor.register(0, 3)

    assert result["mode"] == "cold_start"
    for step in range(3, 7):
        assert wait_until(lambda step=step: actor.get_ref(step)[0] == "ready")
        status, payload = actor.get_ref(step)
        assert status == "ready"
        assert torch.equal(payload[0]["tokens"], torch.full((2, 4), step))


def test_produced_steps_stay_ready_after_iterator_exception(actor_factory):
    actor = actor_factory(StepFactory(5, fail_at=3), prefetch_length=4)
    actor.register(0, 0)

    assert wait_until(lambda: actor.get_ref(3)[0] == "error")
    for step in range(3):
        status, payload = actor.get_ref(step)
        assert status == "ready"
        assert torch.equal(payload[0]["tokens"], torch.full((2, 4), step))

    status, payload = actor.get_ref(3)
    assert status == "error"
    assert "failure at step 3" in payload[0]


def test_encryption_manager_is_lazy_and_encrypts_once(actor_factory, monkeypatch):
    class FakeEncryptionManager:
        instances = 0

        def __init__(self):
            type(self).instances += 1

        def encrypt(self, batch):
            return ("enc", batch)

    monkeypatch.setattr(producer_module, "KMSEncryptionManager", FakeEncryptionManager)
    actor = actor_factory(StepFactory(2), enable_batch_encryption=True)
    assert FakeEncryptionManager.instances == 0

    actor.register(0, 0)

    assert wait_until(lambda: actor.get_ref(0)[0] == "ready")
    assert FakeEncryptionManager.instances == 1
    assert actor.get_ref(0)[1][0][0] == "enc"


def test_stale_in_flight_batch_is_dropped_on_reseek(actor_factory):
    factory = BlockingFactory(5)
    actor = actor_factory(factory, prefetch_length=3)
    actor.register(0, 0)
    assert factory.started.wait(timeout=2)

    actor.register(0, 2)
    factory.release.set()

    assert wait_until(lambda: actor.get_ref(2)[0] == "ready")
    assert list(actor._window.refs) == [2, 3, 4]
    assert actor.get_ref(0)[0] == "evicted"


def test_reseek_starts_a_new_producer_when_old_next_is_blocked(actor_factory):
    factory = GenerationBlockingFactory(10, blocked_generation=1, blocked_step=5)
    actor = actor_factory(factory, lookback_length=0, prefetch_length=6)
    registration = actor.register(0, 0)
    assert factory.started.wait(timeout=2)
    assert actor.stats()["producer_step"] == 5

    actor.ack(0, 5, registration["registration_id"])
    assert actor.stats()["low_watermark"] == 5
    assert actor.register(0, 2)["mode"] == "cold_start"

    for step in range(2, 5):
        assert wait_until(lambda step=step: actor.get_ref(step)[0] == "ready")
        status, payload = actor.get_ref(step)
        assert status == "ready"
        assert torch.equal(payload[0]["tokens"], torch.full((2, 4), step))

    assert factory.calls == 2
    factory.release.set()


def test_shutdown_stops_producer_thread_and_is_idempotent(actor_factory):
    actor = actor_factory(StepFactory(100), prefetch_length=2)
    actor.register(0, 0)
    producer_thread = actor._producer_thread
    assert producer_thread is not None

    actor.shutdown()
    generation = actor.stats()["generation"]
    actor.shutdown()

    assert not producer_thread.is_alive()
    assert actor.stats()["generation"] == generation
    assert actor.stats()["num_refs"] == 0


@pytest.fixture(scope="module")
def local_ray():
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        ignore_reinit_error=True,
    )
    yield
    ray.shutdown()
    ray.cloudpickle.unregister_pickle_by_value(sys.modules[__name__])


@pytest.fixture
def remote_actor_factory(local_ray):
    actors = []

    def create(factory, **kwargs):
        namespace = kwargs.pop("namespace", f"test-{uuid.uuid4().hex}")
        actor = get_or_create_prefetch_actor(
            job_name=kwargs.pop("job_name", f"job-{uuid.uuid4().hex}"),
            split=kwargs.pop("split", "train"),
            dp_rank=kwargs.pop("dp_rank", 0),
            dp_size=kwargs.pop("dp_size", 1),
            dataloader_factory=factory,
            lookback_length=kwargs.pop("lookback_length", 1),
            prefetch_length=kwargs.pop("prefetch_length", 3),
            namespace=namespace,
            num_cpus=0,
            resources={},
        )
        assert not kwargs
        actors.append(actor)
        return actor, namespace

    yield create

    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass


def test_remote_actor_get_if_exists_returns_same_actor(remote_actor_factory):
    factory = StepFactory(4)
    first, namespace = remote_actor_factory(
        factory, job_name="shared-name", namespace=f"same-{uuid.uuid4().hex}"
    )
    first_result = ray.get(first.register.remote(1, 0))
    first_generation = ray.get(first.stats.remote())["generation"]
    second, _ = remote_actor_factory(
        factory, job_name="shared-name", namespace=namespace
    )
    second_result = ray.get(second.register.remote(2, 0))
    second_generation = ray.get(second.stats.remote())["generation"]

    assert first_result["mode"] == "cold_start"
    assert second_result["mode"] == "warm_start"
    assert second_generation == first_generation


def test_remote_actor_returns_ray_put_batches(remote_actor_factory):
    actor, _ = remote_actor_factory(StepFactory(3), prefetch_length=3)
    ray.get(actor.register.remote(0, 0))

    def has_batch():
        status, _ = ray.get(actor.get_ref.remote(1))
        return status == "ready"

    assert wait_until(has_batch, timeout=10)
    status, payload = ray.get(actor.get_ref.remote(1))
    assert status == "ready"
    batch = ray.get(payload[0])
    assert torch.equal(batch["tokens"], torch.full((2, 4), 1))


def test_remote_actor_name_matches_contract():
    assert prefetch_actor_name("job", "train", 3) == "hpct-pdl-job-train-dp3"


def test_remote_actor_can_be_recreated_after_kill(remote_actor_factory):
    actor, namespace = remote_actor_factory(
        StepFactory(3), job_name="restart", namespace=f"restart-{uuid.uuid4().hex}"
    )
    ray.get(actor.register.remote(0, 0))
    ray.kill(actor, no_restart=True)

    deadline = time.monotonic() + 10
    fresh_actor = None
    while time.monotonic() < deadline:
        candidate, _ = remote_actor_factory(
            StepFactory(3), job_name="restart", namespace=namespace
        )
        try:
            result = ray.get(candidate.register.remote(0, 0), timeout=2)
        except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
            time.sleep(0.05)
            continue
        if result["mode"] == "cold_start":
            fresh_actor = candidate
            break
    assert fresh_actor is not None


def test_actor_factory_sets_default_resources(monkeypatch):
    remote_class = MagicMock()
    remote_class.options.return_value.remote.return_value = "actor-handle"
    remote = MagicMock(return_value=remote_class)
    monkeypatch.setattr(producer_module.ray, "remote", remote)
    factory = StepFactory(4)

    result = get_or_create_prefetch_actor(
        job_name="job",
        split="val",
        dp_rank=3,
        dp_size=4,
        dataloader_factory=factory,
        lookback_length=2,
        prefetch_length=5,
        namespace="ns",
    )

    assert result == "actor-handle"
    remote.assert_called_once_with(PrefetchActor)
    remote_class.options.assert_called_once_with(
        name="hpct-pdl-job-val-dp3",
        namespace="ns",
        lifetime="detached",
        get_if_exists=True,
        max_concurrency=8,
        num_cpus=1.0,
        resources={CPU_DATALOADER_RESOURCE: 0.001},
        max_restarts=0,
    )
    assert remote_class.options.return_value.remote.call_args.kwargs == {
        "dataloader_factory": factory,
        "dp_rank": 3,
        "dp_size": 4,
        "lookback_length": 2,
        "prefetch_length": 5,
        "is_val": True,
        "enable_batch_encryption": False,
        "consumer_timeout_s": 300.0,
    }
