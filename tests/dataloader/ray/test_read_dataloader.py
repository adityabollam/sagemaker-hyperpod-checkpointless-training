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
"""Unit tests for RayReadDataLoader against a stub prefetch actor."""

import threading
import time
from unittest.mock import patch

import pytest

ray = pytest.importorskip("ray")

import hyperpod_checkpointless_training.dataloader.ray.read_dataloader as rdl  # noqa: E402

RayReadDataLoader = rdl.RayReadDataLoader

MODULE = "hyperpod_checkpointless_training.dataloader.ray.read_dataloader"


@pytest.fixture(scope="module", autouse=True)
def ray_local():
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


@ray.remote
class StubPrefetchActor:
    """Implements the PrefetchActor register/get_ref/ack/length protocol over ray.put batches."""

    def __init__(
        self,
        final_step=None,
        length=None,
        pending_polls=0,
        evicted_below=None,
        error_step=None,
        payload_fn=None,
        get_ref_delay_s=0.0,
    ):
        self.final_step = final_step
        self._length = length
        self.pending_polls = pending_polls
        self.evicted_below = evicted_below
        self.error_step = error_step
        self.payload_fn = payload_fn or (lambda s: {"step": s})
        self.get_ref_delay_s = get_ref_delay_s
        self.refs = {}
        self.polls = 0
        self.registrations = []
        self.acks = []

    def register(self, consumer_id, step, reseek_token=None):
        self.registrations.append((consumer_id, step, reseek_token))
        return {
            "mode": "warm_start" if step else "cold_start",
            "length": self._length,
            "producer_step": step,
        }

    def get_ref(self, step):
        if self.get_ref_delay_s:
            time.sleep(self.get_ref_delay_s)
        if self.error_step is not None and step >= self.error_step:
            return "error", ["Traceback: boom"]
        if self.evicted_below is not None and step < self.evicted_below:
            return "evicted", []
        if self.final_step is not None and step >= self.final_step:
            return "final", []
        if self.polls < self.pending_polls:
            self.polls += 1
            return "pending", []
        if step not in self.refs:
            self.refs[step] = ray.put(self.payload_fn(step))
        return "ready", [self.refs[step]]

    def ack(self, consumer_id, next_step):
        self.acks.append((consumer_id, next_step))

    def length(self):
        return self._length

    def get_registrations(self):
        return self.registrations

    def get_acks(self):
        return self.acks

    def get_polls(self):
        return self.polls


def _make(actor, step=0, **kwargs):
    kwargs.setdefault("consumer_id", 3)
    kwargs.setdefault("prefetch_length", 2)
    kwargs.setdefault("poll_interval_s", 0.01)
    kwargs.setdefault("rpc_timeout_s", 10.0)
    return RayReadDataLoader(step, actor, **kwargs)


def _wait_for(fn, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = fn()
        if result:
            return result
        time.sleep(0.02)
    return fn()


def test_train_registers_in_init_and_iterates_from_resumed_step():
    actor = StubPrefetchActor.remote(final_step=8, length=8)
    dl = _make(actor, step=5)
    assert ray.get(actor.get_registrations.remote()) == [(3, 5, None)]
    assert dl.restart_info["mode"] == "warm_start"
    assert dl.step == 5

    assert [b["step"] for b in dl] == [5, 6, 7]
    assert dl.step == 8
    dl.stop()


def test_pending_then_ready():
    actor = StubPrefetchActor.remote(final_step=3, pending_polls=5)
    dl = _make(actor)
    assert [b["step"] for b in dl] == [0, 1, 2]
    assert ray.get(actor.get_polls.remote()) == 5


def test_final_stops_iteration_and_partial_consumption():
    actor = StubPrefetchActor.remote(final_step=4)
    dl = _make(actor)
    it = iter(dl)
    assert next(it)["step"] == 0
    assert next(it)["step"] == 1
    assert dl.step == 1  # advances only once the loop asks for the next element
    assert [b["step"] for b in it] == [2, 3]
    with pytest.raises(StopIteration):
        next(it)


def test_evicted_raises():
    actor = StubPrefetchActor.remote(evicted_below=10)
    dl = _make(actor, step=2)
    with pytest.raises(RuntimeError, match="evicted"):
        next(iter(dl))


def test_error_raises_with_producer_traceback():
    actor = StubPrefetchActor.remote(error_step=1)
    dl = _make(actor)
    it = iter(dl)
    assert next(it)["step"] == 0
    with pytest.raises(RuntimeError, match="boom"):
        next(it)


def test_rpc_timeout_is_raised_in_main_thread():
    actor = StubPrefetchActor.remote(get_ref_delay_s=2.0)
    ray.get(
        actor.get_polls.remote()
    )  # wait for actor startup before using a short timeout
    dl = _make(actor, rpc_timeout_s=0.3)
    start = time.monotonic()
    with pytest.raises(ray.exceptions.GetTimeoutError):
        next(iter(dl))
    assert time.monotonic() - start < 1.5


def test_stop_from_other_thread_while_blocked_on_pending():
    """stop() returns promptly and the blocked __next__ raises RuntimeError (not StopIteration)."""
    actor = StubPrefetchActor.remote(pending_polls=10**9)
    dl = _make(actor)
    it = iter(dl)
    stop_duration = []

    def _stopper():
        time.sleep(0.3)
        start = time.monotonic()
        dl.stop()
        stop_duration.append(time.monotonic() - start)

    stopper = threading.Thread(target=_stopper)
    stopper.start()
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="stopped"):
        next(it)
    assert time.monotonic() - start < 3.0
    stopper.join(timeout=5)
    assert stop_duration and stop_duration[0] < 2.0
    # idempotent, never raises
    dl.stop()
    dl.stop()


def test_stop_without_iteration_is_noop():
    actor = StubPrefetchActor.remote(final_step=1)
    dl = _make(actor)
    dl.stop()
    dl.stop()


def test_acks_reach_actor():
    actor = StubPrefetchActor.remote(final_step=13)
    dl = _make(actor, step=10)
    assert [b["step"] for b in dl] == [10, 11, 12]
    acks = _wait_for(
        lambda: ray.get(actor.get_acks.remote()) == [(3, 11), (3, 12), (3, 13)]
    )
    assert acks, ray.get(actor.get_acks.remote())


def test_len_semantics():
    actor = StubPrefetchActor.remote(length=7)
    dl = _make(actor)
    assert len(dl) == 7
    assert dl.remote_length() == 7
    dl.set_length(11)
    assert len(dl) == 11

    unknown = _make(StubPrefetchActor.remote(length=None))
    with pytest.raises(TypeError):
        len(unknown)
    assert unknown.remote_length() is None
    unknown.set_length(4)
    assert len(unknown) == 4


def test_val_reregisters_each_iter_from_step_zero():
    actor = StubPrefetchActor.remote(final_step=3, length=3)
    dl = _make(actor, step=42, is_val=True)
    assert ray.get(actor.get_registrations.remote()) == []
    with pytest.raises(TypeError):
        len(dl)
    assert dl.remote_length() == 3

    assert [b["step"] for b in dl] == [0, 1, 2]
    assert len(dl) == 3
    assert [b["step"] for b in dl] == [0, 1, 2]
    assert ray.get(actor.get_registrations.remote()) == [
        (3, 0, "val-0"),
        (3, 0, "val-1"),
    ]


def test_new_iter_stops_previous_iteration():
    actor = StubPrefetchActor.remote(final_step=5)
    dl = _make(actor, is_val=True)
    first = iter(dl)
    assert next(first)["step"] == 0
    second = iter(dl)
    with pytest.raises(RuntimeError, match="stopped"):
        next(first)
    assert dl.step == 0  # the stale iterator did not advance the shared step
    assert [b["step"] for b in second] == [0, 1, 2, 3, 4]


def test_encryption_decrypts_with_lazily_created_manager():
    actor = StubPrefetchActor.remote(
        final_step=2, payload_fn=lambda s: f"cipher-{s}".encode()
    )
    with patch(f"{MODULE}.KMSEncryptionManager") as manager_cls:
        manager_cls.return_value.decrypt.side_effect = lambda c: {"plain": c.decode()}
        dl = _make(actor, enable_batch_encryption=True)
        manager_cls.assert_not_called()
        assert list(dl) == [{"plain": "cipher-0"}, {"plain": "cipher-1"}]
    manager_cls.assert_called_once_with()


def test_encryption_disabled_does_not_create_manager():
    actor = StubPrefetchActor.remote(final_step=1)
    with patch(f"{MODULE}.KMSEncryptionManager") as manager_cls:
        assert list(_make(actor)) == [{"step": 0}]
    manager_cls.assert_not_called()
