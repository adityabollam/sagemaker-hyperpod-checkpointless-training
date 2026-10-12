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
"""End-to-end measurement of the resume step chain across microbatch counts.

Chain under test:
    CheckpointlessCompatibleConnector.resume_end()
      -> MMAPDataModule.load_checkpoint(global_step)   # optimizer steps
      -> MMAPDataModule.train_dataloader()
      -> RayCPUNodeConfig.create(step, cached_len)
      -> PrefetchActor.register -> factory(target_load_step=...)

These are measurements: several cases are expected to expose the unit
mismatch between global_step (optimizer steps) and target_load_step
(micro-batches).
"""

# Standard Library
import importlib
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch
from torch.utils.data import Dataset

ray = pytest.importorskip("ray")

config_module = importlib.import_module(
    "hyperpod_checkpointless_training.dataloader.config"
)
RayCPUNodeConfig = config_module.RayCPUNodeConfig
prefetch_actor_name = importlib.import_module(
    "hyperpod_checkpointless_training.dataloader.ray.producer"
).prefetch_actor_name
MMAPDataModule = importlib.import_module(
    "hyperpod_checkpointless_training.dataloader.mmap_data_module"
).MMAPDataModule
CheckpointlessCompatibleConnector = importlib.import_module(
    "hyperpod_checkpointless_training.nemo_plugins.checkpoint_connector"
).CheckpointlessCompatibleConnector
SkipDataLoader = importlib.import_module(
    "hyperpod_checkpointless_training.dataloader.skip_dataloader"
).SkipDataLoader
_CheckpointConnector = importlib.import_module(
    "lightning.pytorch.trainer.connectors.checkpoint_connector"
)._CheckpointConnector

DATASET_LEN = 64
RESUME_STEP = 6


@pytest.fixture(scope="module")
def local_ray():
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        namespace=f"stepchain-{uuid.uuid4().hex}",
        ignore_reinit_error=True,
    )
    yield
    ray.shutdown()
    ray.cloudpickle.unregister_pickle_by_value(sys.modules[__name__])


class _IndexDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"index": i, "tokens": torch.full((2,), i)}


class _Recorder:
    """Named Ray actor that records the factory's target_load_step calls."""

    def __init__(self):
        self._calls = []

    def record(self, target_load_step):
        self._calls.append(target_load_step)
        return target_load_step

    def calls(self):
        return list(self._calls)


class IndexFactory:
    """Picklable factory mirroring examples/dataloader/data_module.py."""

    def __init__(self, n, recorder_name, namespace):
        self.n = n
        self.recorder_name = recorder_name
        self.namespace = namespace

    def __call__(self, *, dp_rank, dp_size, target_load_step, is_val):
        recorder = ray.get_actor(self.recorder_name, namespace=self.namespace)
        ray.get(recorder.record.remote(target_load_step))
        return SkipDataLoader(
            dataset=_IndexDataset(self.n),
            batch_size=1,
            _skip_batches=target_load_step,
        )


@contextmanager
def patched_num_microbatches(k):
    original = config_module.get_num_microbatches
    with ExitStack() as stack:
        for module in list(sys.modules.values()):
            if getattr(module, "get_num_microbatches", None) is original:
                stack.enter_context(
                    patch.object(module, "get_num_microbatches", lambda: k)
                )
        yield


def resume_to_step(module, step):
    """Drive the real connector so module.global_step is set like a resume."""
    trainer = MagicMock()
    trainer.global_step = step
    trainer.datamodule = module
    with patch.object(_CheckpointConnector, "resume_end", lambda self: None):
        connector = CheckpointlessCompatibleConnector(trainer)
        connector.start_time = time.perf_counter()
        connector.resume_end()
    assert module.global_step == step


@pytest.fixture
def harness(local_ray):
    job_token = uuid.uuid4().hex
    job_name = f"job-{job_token}"
    namespace = f"ns-{job_token}"
    recorder_name = f"rec-{job_token}"
    recorder = (
        ray.remote(_Recorder)
        .options(name=recorder_name, namespace=namespace, num_cpus=0)
        .remote()
    )
    loaders = []
    actors = {prefetch_actor_name(job_name, "train", 0)}

    def make_module():
        factory = IndexFactory(DATASET_LEN, recorder_name, namespace)
        config = RayCPUNodeConfig(
            cpu_dataloader_factory=factory,
            job_name=job_name,
            ray_namespace=namespace,
            prefetch_length=4,
            val_prefetch_length=4,
            lookback_length=2,
            actor_resources={},
        )
        parallel_state_util = MagicMock()
        parallel_state_util.parallel_state.get_data_parallel_rank.return_value = 0
        parallel_state_util.parallel_state.get_data_parallel_world_size.return_value = 1
        return MMAPDataModule(
            data_module=MagicMock(),
            mmap_config=config,
            parallel_state_util=parallel_state_util,
            is_data_loading_rank=lambda: True,
        )

    def train_dl(module):
        with patch.object(config_module.dist, "get_rank", return_value=0):
            dataloader = module.train_dataloader()
        loaders.append(dataloader)
        return dataloader

    def recorder_calls():
        return ray.get(recorder.calls.remote())

    yield SimpleNamespace(
        make_module=make_module,
        train_dl=train_dl,
        calls=recorder_calls,
        actor_name=prefetch_actor_name(job_name, "train", 0),
        namespace=namespace,
    )

    for dataloader in loaders:
        dataloader.stop()
    for name in actors:
        try:
            ray.kill(ray.get_actor(name, namespace=namespace), no_restart=True)
        except ValueError:
            pass
    ray.kill(recorder)


def phase_one(harness, k):
    """Build module A, consume exactly k*6 batches, stop the loader."""
    module_a = harness.make_module()
    module_a.load_checkpoint({"global_step": 0})
    dataloader = harness.train_dl(module_a)
    iterator = iter(dataloader)
    for expected in range(k * RESUME_STEP):
        batch = next(iterator)
        assert int(batch["index"]) == expected
    dataloader.stop()
    del iterator
    return module_a


@pytest.mark.parametrize(
    "k",
    [
        1,
        pytest.param(
            4,
            marks=pytest.mark.xfail(
                strict=False,
                reason=(
                    "observed first index 6, mode cold_start, "
                    "target_load_step [6]; global_step (optimizer steps) is "
                    "passed unscaled when cached_train_dl_len == 0"
                ),
            ),
        ),
    ],
)
def test_warm(harness, k):
    with patched_num_microbatches(k):
        assert config_module.get_num_microbatches() == k
        phase_one(harness, k)
        calls_before = len(harness.calls())

        module_b = harness.make_module()
        assert module_b.cached_train_dl_len == 0
        resume_to_step(module_b, RESUME_STEP)
        dl2 = harness.train_dl(module_b)
        first = int(next(iter(dl2))["index"])
        mode = dl2.restart_info["mode"]
        new_calls = harness.calls()[calls_before:]
        dl2.stop()

    expected = RESUME_STEP * k
    print(
        f"RESULT path=ray variant=warm K={k} first={first} mode={mode} "
        f"target_load_steps={new_calls} expected={expected}"
    )
    assert mode == "warm_start"
    assert not new_calls
    assert first == expected


@pytest.mark.parametrize(
    "k",
    [
        1,
        pytest.param(
            4,
            marks=pytest.mark.xfail(
                strict=False,
                reason=(
                    "observed first index 6, mode cold_start, "
                    "target_load_step [6]; global_step (optimizer steps) is "
                    "passed unscaled when cached_train_dl_len == 0"
                ),
            ),
        ),
    ],
)
def test_cold(harness, k):
    with patched_num_microbatches(k):
        assert config_module.get_num_microbatches() == k
        phase_one(harness, k)
        calls_before = len(harness.calls())

        ray.kill(
            ray.get_actor(harness.actor_name, namespace=harness.namespace),
            no_restart=True,
        )

        module_b = harness.make_module()
        assert module_b.cached_train_dl_len == 0
        resume_to_step(module_b, RESUME_STEP)

        retry_deadline = time.monotonic() + 5.0
        while True:
            try:
                dl2 = harness.train_dl(module_b)
                break
            except ray.exceptions.RayActorError:
                if time.monotonic() > retry_deadline:
                    raise
                time.sleep(0.05)

        first = int(next(iter(dl2))["index"])
        mode = dl2.restart_info["mode"]
        new_calls = harness.calls()[calls_before:]
        dl2.stop()

    expected = RESUME_STEP * k
    print(
        f"RESULT path=ray variant=cold K={k} first={first} mode={mode} "
        f"target_load_steps={new_calls} expected={expected}"
    )
    assert mode == "cold_start"
    assert len(new_calls) == 1
    assert new_calls[0] == expected
    assert first == expected


@pytest.mark.parametrize("k", [1, 4])
def test_restored_state(harness, k):
    with patched_num_microbatches(k):
        assert config_module.get_num_microbatches() == k
        module_a = phase_one(harness, k)
        calls_before = len(harness.calls())

        module_b = harness.make_module()
        assert module_b.cached_train_dl_len == 0
        module_b.load_state_dict(module_a.state_dict())
        resume_to_step(module_b, RESUME_STEP)
        dl2 = harness.train_dl(module_b)
        first = int(next(iter(dl2))["index"])
        mode = dl2.restart_info["mode"]
        new_calls = harness.calls()[calls_before:]
        dl2.stop()

    expected = RESUME_STEP * k
    print(
        f"RESULT path=ray variant=restored_state K={k} first={first} "
        f"mode={mode} target_load_steps={new_calls} expected={expected}"
    )
    assert first == expected
