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
"""CPU-node dataloader factory for RayCPUNodeConfig.

The factory is created on every GPU rank, pickled by Ray and called inside the PrefetchActor on the
CPU node, so it must stay picklable and must not depend on Megatron parallel state: the DP rank and
size are passed in explicitly. The CPU node must run the same image as training so this module and
the dataset path are importable/readable there.
"""
# Standard Library
from dataclasses import dataclass, field
from typing import Optional

# Third Party
import torch.utils.data

# First Party
from hyperpod_checkpointless_training.dataloader import skip_dataloader
from hyperpod_checkpointless_training.inprocess.logger import get_logger

from .hf_dataset import DEFAULT_SEED, HuggingFaceDataset

logger = get_logger(__name__)


@dataclass
class HFDatasetCPUDataLoaderFactory:
    """Builds the HuggingFace-dataset SkipDataLoader for one DP replica on the CPU node.

    Mirrors ``examples/dataloader/data_module.py::LLMDataModule.train_dataloader/val_dataloader``.
    """

    dataset_path: str = "/fsx/datasets/c4/en/hf-tokenized/llama3/train"
    val_dataset_path: Optional[str] = None
    partition: str = "train"
    seq_length: int = 4096
    micro_batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = False
    shuffle: bool = False
    drop_last: bool = False
    keep_in_memory: bool = True
    _train_dataset: Optional[HuggingFaceDataset] = field(
        default=None, init=False, repr=False, compare=False
    )
    _val_dataset: Optional[HuggingFaceDataset] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __getstate__(self):
        # Never ship loaded datasets over the wire; they are rebuilt lazily on the CPU node.
        state = self.__dict__.copy()
        state["_train_dataset"] = None
        state["_val_dataset"] = None
        return state

    def _load_dataset(self, path: str) -> HuggingFaceDataset:
        return HuggingFaceDataset(
            input_path=path,
            partition=self.partition,
            seq_length=self.seq_length,
            keep_in_memory=self.keep_in_memory,
        )

    def _get_train_dataset(self) -> HuggingFaceDataset:
        if self._train_dataset is None:
            self._train_dataset = self._load_dataset(self.dataset_path)
            logger.info(f"Loaded train dataset with {len(self._train_dataset)} samples")
        return self._train_dataset

    def _get_val_dataset(self) -> HuggingFaceDataset:
        if self._val_dataset is None:
            val_path = self.val_dataset_path or self.dataset_path
            if val_path == self.dataset_path:
                self._val_dataset = self._get_train_dataset()
            else:
                self._val_dataset = self._load_dataset(val_path)
            logger.info(f"Loaded val dataset with {len(self._val_dataset)} samples")
        return self._val_dataset

    def __call__(
        self, *, dp_rank: int, dp_size: int, target_load_step: int, is_val: bool
    ) -> skip_dataloader.SkipDataLoader:
        """
        Build the dataloader for one DP replica.

        Args:
            dp_rank: Data-parallel rank of the replica.
            dp_size: Data-parallel world size.
            target_load_step: Number of batches to skip (resume step).
            is_val: Build the validation dataloader instead of the training one.

        Returns:
            SkipDataLoader positioned at target_load_step.
        """
        if is_val:
            dataset = self._get_val_dataset()
            return skip_dataloader.SkipDataLoader(
                dataset=dataset,
                batch_size=self.micro_batch_size,
                num_workers=self.num_workers,
                collate_fn=dataset.collate_fn,
                pin_memory=self.pin_memory,
                shuffle=False,
                drop_last=False,
                _skip_batches=target_load_step,
            )

        dataset = self._get_train_dataset()
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=self.shuffle,
            seed=DEFAULT_SEED,
            rank=dp_rank,
            num_replicas=dp_size,
        )
        logger.debug(
            f"Created DistributedSampler: dp_rank={dp_rank}, dp_size={dp_size}, shuffle={self.shuffle}"
        )
        # A sampler is mutually exclusive with DataLoader(shuffle=True); shuffling is done by the sampler.
        return skip_dataloader.SkipDataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            _skip_batches=target_load_step,
        )
