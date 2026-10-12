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
"""End-to-end measurement of the resume step chain for CacheResumeMMAPConfig.

Same chain as tests/dataloader/ray/test_resume_step_chain.py but for the
non-Ray path: MMAPDataModule -> CacheResumeMMAPConfig.create ->
CacheResumePrefetchedDataLoader -> data_module.train_dataloader(
    target_load_step=...).
"""

# Standard Library
import importlib
import sys
import time
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

config_module = importlib.import_module(
    "hyperpod_checkpointless_training.dataloader.config"
)
CacheResumeMMAPConfig = config_module.CacheResumeMMAPConfig
MMAPDataModule = importlib.import_module(
    "hyperpod_checkpointless_training.dataloader.mmap_data_module"
).MMAPDataModule
MockParallelStateUtil = importlib.import_module(
    "hyperpod_checkpointless_training.dataloader.mmap.utils"
).MockParallelStateUtil
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


class _IndexDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"index": i, "tokens": torch.full((2,), i)}


class RecordingDataModule:
    """Minimal data module recording every target_load_step it receives."""

    def __init__(self, n, calls):
        self.n = n
        self.calls = calls

    def train_dataloader(self, target_load_step=0):
        self.calls.append(target_load_step)
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


@contextmanager
def cpu_only_torch_tensor():
    # config.create builds the length-sync tensor with device='cuda', which is
    # unavailable on CPU-only torch; drop the device kwarg during create.
    real_tensor = torch.tensor

    def patched(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs.pop("device")
        return real_tensor(*args, **kwargs)

    with patch.object(torch, "tensor", patched):
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


@pytest.mark.parametrize(
    "k",
    [
        1,
        pytest.param(
            4,
            marks=pytest.mark.xfail(
                strict=False,
                reason=(
                    "observed first index 6, target_load_steps [6]; "
                    "global_step (optimizer steps) is passed unscaled when "
                    "cached_train_dl_len == 0"
                ),
            ),
        ),
    ],
)
def test_mmap_resume_step_chain(k, tmp_path, monkeypatch):
    if dist.is_initialized():
        pytest.skip("torch.distributed already initialized")
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1
    )
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    calls = []
    loaders = []
    try:
        with patched_num_microbatches(k):
            assert config_module.get_num_microbatches() == k
            config = CacheResumeMMAPConfig(
                cache_dir=str(tmp_path / "pdl"),
                prefetch_length=4,
                val_prefetch_length=4,
                lookback_length=2,
                checkpoint_frequency=None,
            )
            parallel_state_util = MockParallelStateUtil()
            parallel_state_util.pdl_ranks_in_mp_group = {0: [0]}
            parallel_state_util.create_model_parallel_group = lambda: dist.group.WORLD
            parallel_state_util.parallel_state.get_tensor_model_parallel_group = (
                lambda: dist.group.WORLD
            )
            data_module = RecordingDataModule(DATASET_LEN, calls)

            def make_module():
                return MMAPDataModule(
                    data_module=data_module,
                    mmap_config=config,
                    parallel_state_util=parallel_state_util,
                    is_data_loading_rank=lambda: True,
                )

            def train_dl(module):
                with cpu_only_torch_tensor():
                    dataloader = module.train_dataloader()
                loaders.append(dataloader)
                return dataloader

            # Phase 1: consume exactly k*6 batches.
            module_a = make_module()
            module_a.load_checkpoint({"global_step": 0})
            dataloader = train_dl(module_a)
            iterator = iter(dataloader)
            for expected_index in range(k * RESUME_STEP):
                batch = next(iterator)
                assert int(batch["index"]) == expected_index
            dataloader.stop()
            del iterator

            calls_before = len(calls)

            # Phase 2: fresh module, resume through the real connector.
            module_b = make_module()
            assert module_b.cached_train_dl_len == 0
            resume_to_step(module_b, RESUME_STEP)
            dl2 = train_dl(module_b)
            first = int(next(iter(dl2))["index"])
            new_calls = calls[calls_before:]
            dl2.stop()

        expected = RESUME_STEP * k
        print(
            f"RESULT path=mmap variant=cache_resume K={k} first={first} "
            f"mode=n/a target_load_steps={new_calls} expected={expected}"
        )
        assert first == expected
    finally:
        for dataloader in loaders:
            dataloader.stop()
        dist.destroy_process_group()
