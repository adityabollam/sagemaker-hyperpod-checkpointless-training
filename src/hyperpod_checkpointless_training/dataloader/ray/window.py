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
"""Bookkeeping for a Ray dataloader's prefetched batch window."""

# Standard Library
import time
from collections import OrderedDict
from typing import Any, Callable

# First Party
from hyperpod_checkpointless_training.dataloader.mmap.utils import RestartMode

WARM_START = RestartMode.WARM_START.value
COLD_START = RestartMode.COLD_START.value


class BatchWindow:
    """Track produced refs, active consumers, and the producer's range."""

    def __init__(
        self,
        lookback_length: int,
        prefetch_length: int,
        consumer_timeout_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if consumer_timeout_s <= 0:
            raise ValueError("consumer_timeout_s must be > 0")
        self.lookback_length = lookback_length
        self.prefetch_length = prefetch_length
        # math.inf disables consumer expiry.
        self.consumer_timeout_s = consumer_timeout_s
        self.clock = clock
        self.refs: OrderedDict[int, Any] = OrderedDict()
        self.producer_step: int | None = None
        self.final_step: int | None = None
        self.consumer_steps: dict[int, int] = {}
        self.last_seen: dict[int, float] = {}

    @property
    def low_watermark(self) -> int | None:
        return next(iter(self.refs)) if self.refs else self.producer_step

    @property
    def high_watermark(self) -> int | None:
        return None if self.producer_step is None else self.producer_step - 1

    def reset(self, step: int) -> list:
        dropped = list(self.refs.values())
        self.refs.clear()
        self.consumer_steps.clear()
        self.last_seen.clear()
        self.final_step = None
        self.producer_step = step
        return dropped

    def can_resume_from(self, step: int) -> bool:
        low_watermark = self.low_watermark
        return (
            self.producer_step is not None
            and low_watermark is not None
            and low_watermark <= step <= self.producer_step
        )

    def register_consumer(
        self, consumer_id: int, step: int, last_seen: float | None = None
    ) -> None:
        self.consumer_steps[consumer_id] = step
        self.last_seen[consumer_id] = self.clock() if last_seen is None else last_seen

    def touch(self, consumer_id: int) -> None:
        if consumer_id in self.consumer_steps:
            self.last_seen[consumer_id] = self.clock()

    def ack(self, consumer_id: int, next_step: int) -> list:
        if consumer_id not in self.consumer_steps:
            return []
        self.touch(consumer_id)
        self.consumer_steps[consumer_id] = max(
            self.consumer_steps[consumer_id], next_step
        )
        return self.evict()

    def live_consumer_steps(self) -> list[int]:
        now = self.clock()
        return [
            step
            for consumer_id, step in self.consumer_steps.items()
            if now - self.last_seen[consumer_id] <= self.consumer_timeout_s
        ]

    def evict(self) -> list:
        if not self.consumer_steps:
            return []
        live = self.live_consumer_steps()
        if live:
            threshold = min(live) - self.lookback_length
        else:
            threshold = self.low_watermark - self.lookback_length
        evicted = []
        while self.refs and next(iter(self.refs)) < threshold:
            _, ref = self.refs.popitem(last=False)
            evicted.append(ref)
        return evicted

    def should_produce(self) -> bool:
        if self.producer_step is None or self.final_step is not None:
            return False
        live = self.live_consumer_steps()
        if live:
            base = min(live)
        else:
            base = self.low_watermark
        return self.producer_step < base + self.prefetch_length

    def add(self, step: int, ref: Any) -> None:
        assert step == self.producer_step
        self.refs[step] = ref
        self.producer_step += 1

    def mark_final(self) -> None:
        self.final_step = self.producer_step

    def lookup(self, step: int) -> tuple[str, Any]:
        if step in self.refs:
            return "ready", self.refs[step]
        if self.final_step is not None and step >= self.final_step:
            return "final", None
        low_watermark = self.low_watermark
        if self.producer_step is None or (
            low_watermark is not None and step < low_watermark
        ):
            return "evicted", None
        return "pending", None

    def snapshot(self) -> dict:
        return {
            "low_watermark": self.low_watermark,
            "high_watermark": self.high_watermark,
            "producer_step": self.producer_step,
            "final_step": self.final_step,
            "num_refs": len(self.refs),
            "consumer_steps": dict(self.consumer_steps),
        }
