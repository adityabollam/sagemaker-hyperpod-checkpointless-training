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
"""Local-Ray coverage for Ray dataloader lengths and epoch rollovers."""

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
from torch.utils.data import Dataset

# First Party
import hyperpod_checkpointless_training.dataloader.skip_dataloader as skip_dataloader

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
        namespace=f"lengths-{uuid.uuid4().hex}",
        ignore_reinit_error=True,
    )
    yield
    ray.shutdown()
    ray.cloudpickle.unregister_pickle_by_value(sys.modules[__name__])


class IndexDataset(Dataset):
    def __init__(self, n_items):
        self.n_items = n_items

    def __len__(self):
        return self.n_items

    def __getitem__(self, index):
        return {"idx": index}


class IndexFactory:
    def __init__(self, n_items, batch_size=1, n_val_items=None, calls_path=None):
        self.n_items = n_items
        self.batch_size = batch_size
        self.n_val_items = n_val_items
        self.calls_path = calls_path

    def __call__(self, *, dp_rank, dp_size, target_load_step, is_val):
        with open(self.calls_path, "a", encoding="utf-8") as calls_file:
            calls_file.write(f"{dp_rank},{target_load_step},{is_val}\n")
        n_items = (
            self.n_val_items
            if is_val and self.n_val_items is not None
            else self.n_items
        )
        return skip_dataloader.SkipDataLoader(
            IndexDataset(n_items),
            batch_size=self.batch_size,
            _skip_batches=target_load_step,
        )


def collect_batches(iterable, timeout=15):
    result = queue.Queue()

    def collect():
        try:
            result.put((True, list(iterable)))
        except BaseException as error:
            result.put((False, error))

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    try:
        succeeded, value = result.get(timeout=timeout)
    except queue.Empty:
        pytest.fail(f"iteration did not finish within {timeout:.2f} seconds")
    if succeeded:
        return value
    raise value


def batch_index(batch, batch_size=1):
    return int(batch["idx"][0]) // batch_size


def drain(loader, timeout=15, batch_size=1):
    return [
        batch_index(batch, batch_size)
        for batch in collect_batches(loader, timeout=timeout)
    ]


def drain_lockstep(loaders, epochs=1, timeout=30, batch_size=1):
    barrier = threading.Barrier(len(loaders), timeout=10)
    batches_by_loader = [[] for _ in loaders]
    lengths_by_loader = [[] for _ in loaders]
    errors = []

    def drain_loader(index, loader):
        try:
            for _ in range(epochs):
                epoch_batches = []
                for batch in loader:
                    epoch_batches.append(batch_index(batch, batch_size))
                    barrier.wait()
                batches_by_loader[index].append(epoch_batches)
                lengths_by_loader[index].append(len(loader))
        except BaseException as error:
            errors.append((index, error))
            barrier.abort()

    threads = [
        threading.Thread(target=drain_loader, args=(index, loader), daemon=True)
        for index, loader in enumerate(loaders)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
    if any(thread.is_alive() for thread in threads):
        pytest.fail("lockstep iteration did not finish before the timeout")
    if errors:
        pytest.fail("; ".join(f"loader {index}: {error!r}" for index, error in errors))
    return batches_by_loader, lengths_by_loader


def next_with_deadline(iterator, timeout=15):
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


def wait_until(predicate, timeout=10, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


@pytest.fixture
def length_harness(tmp_path):
    loaders = []
    actors = set()

    def build(
        *,
        n_items=12,
        batch_size=1,
        n_val_items=None,
        prefetch_length=4,
        lookback_length=2,
    ):
        job_token = uuid.uuid4().hex
        calls_path = tmp_path / f"{job_token}-factory-calls.txt"
        factory = IndexFactory(
            n_items=n_items,
            batch_size=batch_size,
            n_val_items=n_val_items,
            calls_path=str(calls_path),
        )
        config = RayCPUNodeConfig(
            cpu_dataloader_factory=factory,
            job_name=f"job-{job_token}",
            ray_namespace=f"ns-{job_token}",
            prefetch_length=prefetch_length,
            val_prefetch_length=prefetch_length,
            lookback_length=lookback_length,
            actor_resources={},
        )

        def create(
            step,
            dp_rank=0,
            dp_size=1,
            consumer_id=0,
            is_val=False,
            cached_len=0,
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
            with (
                patch.object(config_module.dist, "get_rank", return_value=consumer_id),
                patch.object(config_module, "get_num_microbatches", return_value=1),
            ):
                loader = config.create(
                    None,
                    parallel_state_util,
                    step,
                    None,
                    None,
                    name=f"job-{job_token}",
                    is_val=is_val,
                    cached_len=cached_len,
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


def test_fresh_train_loader_yields_exactly_len_batches(length_harness):
    harness = length_harness(n_items=10)
    loader = harness.create(step=0)

    batches = drain(loader)

    assert len(loader) == len(batches) == 10
    assert batches == list(range(10))


def test_mid_epoch_create_reports_remaining_batches(length_harness):
    harness = length_harness(n_items=100)
    loader = harness.create(step=25, cached_len=100)

    batches = drain(loader)

    assert len(loader) == len(batches) == 75
    assert batches == list(range(25, 100))
    assert harness.calls() == [(0, 25, False)]


def test_reiterating_same_loader_rolls_over_to_next_epoch(length_harness):
    harness = length_harness()
    loader = harness.create(step=0)

    for _ in range(3):
        batches = drain(loader)
        assert len(loader) == len(batches) == 12
        assert batches == list(range(12))

    assert [call for call in harness.calls() if not call[2]] == [
        (0, 0, False),
        (0, 0, False),
        (0, 0, False),
    ]


def test_rollover_after_mid_epoch_start_reports_full_epoch(length_harness):
    harness = length_harness()
    loader = harness.create(step=5, cached_len=12)

    first_pass = drain(loader)
    assert len(loader) == len(first_pass) == 7
    assert first_pass == list(range(5, 12))

    second_pass = drain(loader)
    assert len(loader) == len(second_pass) == 12
    assert second_pass == list(range(12))


def test_new_create_at_global_epoch_step_starts_next_epoch(length_harness):
    harness = length_harness()

    first = harness.create(step=0)
    first_batches = drain(first)
    assert len(first) == len(first_batches) == 12
    assert first_batches == list(range(12))

    second = harness.create(step=12, cached_len=12)
    second_batches = drain(second)
    assert len(second) == len(second_batches) == 12
    assert second_batches == list(range(12))

    third = harness.create(step=24, cached_len=12)
    third_batches = drain(third)
    assert len(third) == len(third_batches) == 12
    assert third_batches == list(range(12))

    fourth = harness.create(step=29, cached_len=12)
    fourth_batches = drain(fourth)
    assert len(fourth) == len(fourth_batches) == 7
    assert fourth_batches == list(range(5, 12))


def test_partial_last_batch_stops_at_final_step(length_harness):
    harness = length_harness(n_items=10, batch_size=3, prefetch_length=10)
    loader = harness.create(step=0)

    batches = collect_batches(loader)
    indices = [batch_index(batch, 3) for batch in batches]

    assert len(loader) == len(indices) == 4
    assert indices == [0, 1, 2, 3]
    assert len(batches[-1]["idx"]) == 1
    actor = ray.get_actor(harness.actor_name(), namespace=harness.namespace)
    stats = wait_until(
        lambda: (
            state
            if (state := ray.get(actor.stats.remote()))["final_step"] == 4
            else None
        )
    )
    assert stats["final_step"] == 4

    resumed = harness.create(step=6, cached_len=4)
    resumed_batches = drain(resumed, batch_size=3)
    assert len(resumed) == len(resumed_batches) == 2
    assert resumed_batches == [2, 3]


def test_val_length_before_iter_and_each_pass_full(length_harness):
    harness = length_harness(n_items=12, n_val_items=5)
    loader = harness.create(step=7, cached_len=99, is_val=True)

    assert len(loader) == 5
    for _ in range(2):
        batches = drain(loader)
        assert len(loader) == len(batches) == 5
        assert batches == list(range(5))
    assert all(step == 0 for _, step, is_val in harness.calls() if is_val)


def test_val_sibling_with_fewer_passes_joins_current_pass(length_harness):
    harness = length_harness(n_items=12, n_val_items=5)
    first = harness.create(step=0, consumer_id=0, is_val=True)

    for _ in range(2):
        batches = drain(first)
        assert len(first) == len(batches) == 5
        assert batches == list(range(5))

    second = harness.create(step=0, consumer_id=1, is_val=True)
    assert len(second) == 5
    for _ in range(2):
        batches, lengths = drain_lockstep([first, second])
        assert batches == [[list(range(5))], [list(range(5))]]
        assert lengths == [[5], [5]]
    assert all(step == 0 for _, step, is_val in harness.calls() if is_val)


def test_val_created_mid_train_epoch_is_full_length(length_harness):
    harness = length_harness(n_items=12, n_val_items=5)
    train_loader = harness.create(step=0)
    train_iterator = iter(train_loader)
    train_indices = [batch_index(next_with_deadline(train_iterator)) for _ in range(5)]

    val_loader = harness.create(step=5, is_val=True)
    assert len(val_loader) == 5
    val_batches = drain(val_loader)
    assert len(val_loader) == len(val_batches) == 5
    assert val_batches == list(range(5))

    train_indices.extend(
        batch_index(batch) for batch in collect_batches(train_iterator)
    )
    assert len(train_loader) == len(train_indices) == 12
    assert train_indices == list(range(12))


def test_tp_siblings_mid_epoch_lengths_match(length_harness):
    harness = length_harness(n_items=100)
    loaders = [
        harness.create(step=25, cached_len=100, consumer_id=consumer_id)
        for consumer_id in (0, 1)
    ]

    assert [len(loader) for loader in loaders] == [75, 75]
    batches, lengths = drain_lockstep(loaders)
    assert batches == [[list(range(25, 100))], [list(range(25, 100))]]
    assert lengths == [[75], [75]]


def test_tp_siblings_rollover_same_loader(length_harness):
    harness = length_harness()
    loaders = [
        harness.create(step=0, consumer_id=consumer_id) for consumer_id in (0, 1)
    ]

    batches, lengths = drain_lockstep(loaders, epochs=3)

    assert batches == [[list(range(12)) for _ in range(3)] for _ in loaders]
    assert lengths == [[12, 12, 12] for _ in loaders]
    assert [call for call in harness.calls() if not call[2]] == [
        (0, 0, False),
        (0, 0, False),
        (0, 0, False),
    ]


def test_tp_siblings_rollover_via_new_create(length_harness):
    harness = length_harness()
    first_loaders = [
        harness.create(step=0, consumer_id=consumer_id) for consumer_id in (0, 1)
    ]
    first_batches, first_lengths = drain_lockstep(first_loaders)
    assert first_batches == [[list(range(12))], [list(range(12))]]
    assert first_lengths == [[12], [12]]

    next_loaders = [
        harness.create(step=12, cached_len=12, consumer_id=consumer_id)
        for consumer_id in (0, 1)
    ]
    assert [len(loader) for loader in next_loaders] == [12, 12]
    next_batches, next_lengths = drain_lockstep(next_loaders)
    assert next_batches == [[list(range(12))], [list(range(12))]]
    assert next_lengths == [[12], [12]]


def test_warm_restart_inside_window_reports_remaining(length_harness):
    harness = length_harness(n_items=40, prefetch_length=10, lookback_length=4)
    initial_loaders = [
        harness.create(step=0, consumer_id=consumer_id) for consumer_id in (0, 1)
    ]
    iterators = [iter(loader) for loader in initial_loaders]
    for _ in range(20):
        for iterator in iterators:
            next_with_deadline(iterator)
    for loader in initial_loaders:
        loader.stop()

    resumed_loaders = [
        harness.create(step=18, cached_len=40, consumer_id=consumer_id)
        for consumer_id in (0, 1)
    ]
    assert [loader.restart_info["mode"] for loader in resumed_loaders] == [
        "warm_start",
        "warm_start",
    ]
    assert [len(loader) for loader in resumed_loaders] == [22, 22]

    batches, lengths = drain_lockstep(resumed_loaders)
    assert batches == [[list(range(18, 40))], [list(range(18, 40))]]
    assert lengths == [[22], [22]]
    assert harness.calls() == [(0, 0, False)]


def test_stale_generation_after_sibling_rollover(length_harness):
    harness = length_harness(n_items=6)
    loader_a = harness.create(step=0, consumer_id=0)
    loader_b = harness.create(step=0, consumer_id=1)
    actor = ray.get_actor(harness.actor_name(), namespace=harness.namespace)
    reg_a = loader_a.restart_info
    reg_b = loader_b.restart_info

    ray.get(actor.ack.remote(0, 6, reg_a["registration_id"]))
    ray.get(actor.ack.remote(1, 6, reg_b["registration_id"]))
    stats = wait_until(
        lambda: (
            state
            if (state := ray.get(actor.stats.remote()))["final_step"] == 6
            else None
        )
    )
    assert stats["final_step"] == 6

    generation = reg_a["generation"]
    token = f"train-rollover-gen{generation}"
    reg_a2 = ray.get(actor.register.remote(0, 0, token))
    assert reg_a2["mode"] == "cold_start"
    assert reg_a2["generation"] == generation + 1
    assert ray.get(actor.get_ref.remote(6, generation=generation)) == ("final", [])
    assert ray.get(actor.get_ref.remote(5, generation=generation))[0] == "superseded"

    ray.get(actor.ack.remote(1, 6, reg_b["registration_id"]))
    stats = ray.get(actor.stats.remote())
    assert stats["consumer_steps"][1] == 0

    reg_b2 = ray.get(actor.register.remote(1, 0, token))
    assert reg_b2["mode"] == "warm_start"
