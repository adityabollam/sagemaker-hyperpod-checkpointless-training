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
"""Tests for Ray dataloader batch windows."""

# Third Party
import pytest

# First Party
from hyperpod_checkpointless_training.dataloader.ray.window import (  # isort: skip
    BatchWindow,
    COLD_START,
    WARM_START,
)


def test_reset_and_watermarks():
    window = BatchWindow(lookback_length=2, prefetch_length=3)

    assert window.low_watermark is None
    assert window.high_watermark is None

    assert window.reset(5) == []
    assert window.low_watermark == 5
    assert window.high_watermark == 4
    window.add(5, "ref-5")
    window.add(6, "ref-6")
    assert window.low_watermark == 5
    assert window.high_watermark == 6


def test_reset_returns_refs_and_clears_state():
    window = BatchWindow(lookback_length=1, prefetch_length=2)
    window.reset(0)
    window.register_consumer(7, 0)
    window.add(0, "ref-0")
    window.mark_final()

    assert window.reset(4) == ["ref-0"]
    assert not window.refs
    assert not window.consumer_steps
    assert window.final_step is None
    assert window.producer_step == 4


def test_add_requires_contiguous_producer_step():
    window = BatchWindow(lookback_length=0, prefetch_length=2)
    window.reset(3)

    with pytest.raises(AssertionError):
        window.add(4, "ref-4")


def test_lookup_ready_pending_final_and_evicted():
    window = BatchWindow(lookback_length=0, prefetch_length=2)
    assert window.lookup(0) == ("evicted", None)

    window.reset(2)
    assert window.lookup(2) == ("pending", None)
    window.add(2, "ref-2")
    assert window.lookup(2) == ("ready", "ref-2")
    window.mark_final()
    assert window.lookup(3) == ("final", None)
    window.register_consumer(1, 2)
    assert window.ack(1, 3) == ["ref-2"]
    assert window.lookup(2) == ("evicted", None)


def test_register_consumer_overwrites_with_rollback_step():
    window = BatchWindow(lookback_length=0, prefetch_length=2)
    window.reset(0)
    window.register_consumer(4, 5)
    window.register_consumer(4, 2)

    assert window.consumer_steps == {4: 2}


def test_ack_is_monotonic_and_ignores_unknown_consumer():
    window = BatchWindow(lookback_length=0, prefetch_length=2)
    window.reset(0)
    window.register_consumer(3, 4)

    assert window.ack(3, 2) == []
    assert window.consumer_steps[3] == 4
    assert window.ack(8, 9) == []
    assert window.consumer_steps == {3: 4}


def test_evict_uses_minimum_consumer_step_and_lookback():
    window = BatchWindow(lookback_length=1, prefetch_length=6)
    window.reset(0)
    for step in range(6):
        window.add(step, f"ref-{step}")
    window.register_consumer(1, 2)
    window.register_consumer(2, 4)

    assert window.evict() == ["ref-0"]
    assert window.ack(1, 3) == ["ref-1"]
    assert list(window.refs) == [2, 3, 4, 5]


def test_evict_without_consumers_does_nothing():
    window = BatchWindow(lookback_length=0, prefetch_length=3)
    window.reset(0)
    window.add(0, "ref-0")
    window.add(1, "ref-1")

    assert window.evict() == []
    assert list(window.refs.values()) == ["ref-0", "ref-1"]


def test_should_produce_applies_consumer_and_no_consumer_backpressure():
    window = BatchWindow(lookback_length=0, prefetch_length=3)
    assert not window.should_produce()
    window.reset(0)
    window.register_consumer(2, 0)

    assert window.should_produce()
    window.add(0, "ref-0")
    window.add(1, "ref-1")
    assert window.should_produce()
    window.add(2, "ref-2")
    assert not window.should_produce()
    window.ack(2, 1)
    assert window.should_produce()

    no_consumer_window = BatchWindow(lookback_length=0, prefetch_length=2)
    no_consumer_window.reset(10)
    assert no_consumer_window.should_produce()
    no_consumer_window.add(10, "ref-10")
    assert no_consumer_window.should_produce()
    no_consumer_window.add(11, "ref-11")
    assert not no_consumer_window.should_produce()
    no_consumer_window.mark_final()
    assert not no_consumer_window.should_produce()


def test_can_resume_from_window_edges():
    window = BatchWindow(lookback_length=0, prefetch_length=3)
    assert not window.can_resume_from(0)

    window.reset(5)
    assert window.can_resume_from(5)
    window.add(5, "ref-5")
    window.add(6, "ref-6")
    assert window.can_resume_from(5)
    assert window.can_resume_from(7)
    assert not window.can_resume_from(4)
    assert not window.can_resume_from(8)


def test_snapshot_has_exact_keys_and_copies_consumer_steps():
    window = BatchWindow(lookback_length=1, prefetch_length=2)
    window.reset(4)
    window.register_consumer(9, 4)

    snapshot = window.snapshot()

    assert set(snapshot) == {
        "low_watermark",
        "high_watermark",
        "producer_step",
        "final_step",
        "num_refs",
        "consumer_steps",
    }
    snapshot["consumer_steps"][9] = 100
    assert window.consumer_steps == {9: 4}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _produce(window):
    while window.should_produce():
        window.add(window.producer_step, f"ref-{window.producer_step}")


def test_expired_consumer_no_longer_blocks_production_or_eviction():
    clock = FakeClock()
    window = BatchWindow(
        lookback_length=2,
        prefetch_length=5,
        consumer_timeout_s=10.0,
        clock=clock,
    )
    window.reset(0)
    window.register_consumer("A", 0)
    window.register_consumer("B", 0)

    # Drive A to 20 and B to 30 interleaved with producing.
    for step in range(1, 21):
        window.ack("A", step)
        window.ack("B", step)
        _produce(window)
    for step in range(21, 31):
        window.ack("B", step)
    _produce(window)

    assert window.producer_step == 25  # 20 + 5
    assert window.low_watermark == 18  # evicted by A (20 - lookback 2)
    assert window.live_consumer_steps() == [20, 30]

    # B keeps acking (refreshing its lease) while A stays silent past the timeout.
    clock.advance(6)
    window.ack("B", 30)
    clock.advance(6)
    _produce(window)

    assert window.live_consumer_steps() == [30]
    assert window.producer_step == 35  # 30 + 5
    window.ack("B", 30)
    assert window.low_watermark == 28  # evicted up to 30 - lookback 2

    # A re-registers and is counted again.
    window.register_consumer("A", 29)
    assert 29 in window.live_consumer_steps()
    assert not window.should_produce()  # cap is 29 + 5 = 34, producer at 35
    window.ack("B", 30)
    assert window.low_watermark == 28
    assert min(window.live_consumer_steps()) == 29


def test_all_consumers_expired_falls_back_to_low_watermark():
    clock = FakeClock()
    window = BatchWindow(
        lookback_length=2,
        prefetch_length=5,
        consumer_timeout_s=10.0,
        clock=clock,
    )
    window.reset(0)
    window.register_consumer("A", 0)
    _produce(window)
    window.ack("A", 3)
    assert window.low_watermark == 1

    clock.advance(11)
    assert window.live_consumer_steps() == []

    # Production is bounded by low_watermark + prefetch_length.
    _produce(window)
    assert window.producer_step == 6
    assert not window.should_produce()

    # With no live consumers, evict() evicts nothing.
    assert window.evict() == []
    assert window.low_watermark == 1

    # An ack from an expired consumer refreshes it (it becomes live again).
    window.ack("A", 4)
    assert window.live_consumer_steps() == [4]


def test_touch_reset_and_explicit_last_seen():
    clock = FakeClock()
    window = BatchWindow(
        lookback_length=1,
        prefetch_length=2,
        consumer_timeout_s=10.0,
        clock=clock,
    )
    window.reset(0)

    # touch on an unknown id creates nothing.
    window.touch("ghost")
    assert window.consumer_steps == {}
    assert window.last_seen == {}

    window.register_consumer("A", 0)
    assert window.last_seen["A"] == 0.0
    clock.advance(5)
    window.touch("A")
    assert window.last_seen["A"] == 5.0

    # register_consumer with an explicit last_seen keeps it.
    window.register_consumer("B", 0, last_seen=-100.0)
    assert window.last_seen["B"] == -100.0
    assert window.live_consumer_steps() == [0]

    # reset clears last_seen.
    window.reset(0)
    assert window.last_seen == {}

    with pytest.raises(ValueError):
        BatchWindow(1, 2, consumer_timeout_s=0)
    with pytest.raises(ValueError):
        BatchWindow(1, 2, consumer_timeout_s=-1)


def test_restart_mode_values():
    assert COLD_START == "cold_start"
    assert WARM_START == "warm_start"
