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
"""Memory-map Config."""

# Standard Library
import os
from dataclasses import dataclass
from typing import Callable

# Third Party
import torch
from torch import distributed as dist
from megatron.core.num_microbatches_calculator import get_num_microbatches


# First Party
from hyperpod_checkpointless_training.dataloader.mmap.cache_read_dataloader import (
    CacheResumeReadDataLoader,
)
from hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader import (
    CacheResumePrefetchedDataLoader,
)
from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg

logger = get_logger()


@dataclass
class MMAPConfig:
    """Base Config."""

    def _adjust_for_epoch_resume(self, cached_len: int, step: int) -> tuple[int, int]:
        """Adjust cached_len and step when resuming within an epoch or after an epoch."""
        num_microbatches = get_num_microbatches()
        epoch_length = cached_len
        # we need to adjust step in epoch so that DL skips to correct batch after new epoch
        batch_in_epoch = (step * num_microbatches) % epoch_length
        # remaining_batches is normally handled by megatron data sampler but we override the sampler in our example
        # https://github.com/NVIDIA-NeMo/NeMo/blob/55635b9a353daa75f92c329cf71c9017f59ca22b/nemo/collections/common/data/data_samplers.py#L90-L101   
        remaining_batches = epoch_length - batch_in_epoch
        logger.info(debug_msg(f"Original Cached Len: {cached_len} Overriding cached_len to {remaining_batches}, and starting from batch {batch_in_epoch}"))
        return remaining_batches, batch_in_epoch


@dataclass
class CacheResumeMMAPConfig(MMAPConfig):
    """Cache-resume MMAP Config.

    The dataloader factory's ``len()`` must report the full epoch length regardless of
    ``target_load_step``.
    """

    cache_dir: str = "/dev/shm/pdl_cache"
    prefetch_length: int = 10
    val_prefetch_length: int = 10
    lookback_length: int = 2
    checkpoint_frequency: int = None
    model_parallel_group: object = None
    enable_batch_encryption: bool = False

    def create(
        self,
        dataloader_init_callable: Callable,
        parallel_state_util,
        step: int,
        is_data_loading_rank: Callable,
        create_model_parallel_group_callable: Callable,
        name: str = "Train",
        is_val: bool = False,
        cached_len: int = 0,
    ):
        """
        Create and return the configured dataloader.

        Args:
            dataloader_init_callable: The dataloader initialization function.
            parallel_state_util: The parallel state utility instance.
            step: The data step to resume from.
            create_model_parallel_group_callable: A callable to the create_model_parallel_group
            name: The name of the dataloader.
            is_val: A boolean used to toogle between train and val. Returns true for val
        Returns:
            An MMAP wrapped dataloader.
        """
        # Validate required parameters
        if step is None:
            raise ValueError("step must be provided.")
        if self.checkpoint_frequency is None:
            logger.warning("Cache performance degraded since checkpoint frequency is not set")

        if self.model_parallel_group is None:
            self.model_parallel_group = create_model_parallel_group_callable()

        # Set up cache directory
        cache_dir_suffix = "val" if is_val else "train"
        cache_dir = os.path.join(self.cache_dir, cache_dir_suffix)
        step = 0 if is_val else step
        force_cold_start = True if is_val else False
        prefetch_length = self.val_prefetch_length if is_val else self.prefetch_length
        synchronized_length = 0
        stateless_resume = not is_val and not cached_len and step > 0
        epoch_length = None
        remaining_length = None
        data_loading_rank = None

        if cached_len:
            epoch_length = cached_len
        elif stateless_resume:
            data_loading_rank = is_data_loading_rank()
            local_length = (
                len(dataloader_init_callable(target_load_step=0))
                if data_loading_rank
                else 0
            )
            length_tensor = torch.tensor(local_length, dtype=torch.long, device="cuda")
            tp_group = parallel_state_util.parallel_state.get_tensor_model_parallel_group()
            dist.all_reduce(length_tensor, op=dist.ReduceOp.MAX, group=tp_group)
            epoch_length = length_tensor.item()
            if epoch_length > 0:
                remaining_length, step = self._adjust_for_epoch_resume(
                    epoch_length, step
                )
            else:
                remaining_length = 0

        if cached_len and not is_val:
            cached_len, step = self._adjust_for_epoch_resume(cached_len, step)

        # Create the appropriate dataloader based on parallel state
        if data_loading_rank is None:
            data_loading_rank = is_data_loading_rank()

        if data_loading_rank:
            data_loader = CacheResumePrefetchedDataLoader(
                step=step,
                dataloader_init_callable=dataloader_init_callable,
                model_parallel_group=self.model_parallel_group,
                parallel_state_util=parallel_state_util,
                force_cold_start=force_cold_start,
                model_checkpoint_frequency=self.checkpoint_frequency,
                lookback_length=self.lookback_length,
                prefetch_length=prefetch_length,
                cache_dir=cache_dir,
                enable_batch_encryption=self.enable_batch_encryption,
            )
            if not cached_len:
                data_loader.initialize_data_loader()
                if not stateless_resume:
                    synchronized_length = len(data_loader.data_loader)

        else:
            data_loader = CacheResumeReadDataLoader(
                dataloader_init_callable=dataloader_init_callable,
                model_parallel_group=self.model_parallel_group,
                step=step,
                prefetch_length=prefetch_length,
                parallel_state_util=parallel_state_util,
                force_cold_start=force_cold_start,
                cache_dir=cache_dir,
                enable_batch_encryption=self.enable_batch_encryption,
            )

        if not cached_len and not stateless_resume:
            length_tensor = torch.tensor(synchronized_length, dtype=torch.long, device='cuda')
            tp_group = parallel_state_util.parallel_state.get_tensor_model_parallel_group()
            dist.all_reduce(length_tensor, op=dist.ReduceOp.MAX, group=tp_group)
            synchronized_length = length_tensor.item()
    
        # Set data_loader length
        if stateless_resume:
            length = remaining_length
        else:
            length = max(cached_len, synchronized_length)
        data_loader.set_length(length)
        if not is_val:
            data_loader.epoch_length = (
                epoch_length if epoch_length is not None else length
            )

        logger.info(
            debug_msg(f"Created {name} {data_loader.__class__.__name__} with length: {len(data_loader)}")
        )
        return data_loader


@dataclass
class RayCPUNodeConfig(MMAPConfig):
    """Ray CPU-node dataloader Config.

    Batches are produced by one detached Ray ``PrefetchActor`` per (job, split, dp_rank) running
    on a dedicated CPU node, and every GPU rank reads them through a ``RayReadDataLoader``.
    See docs/ray_cpu_dataloader.md.

    For stateless train resumes, the factory's ``len()`` must report the full epoch length
    regardless of ``target_load_step``.
    """

    cpu_dataloader_factory: Callable = None
    job_name: str = "default"
    ray_address: str = "auto"
    ray_namespace: str = "hyperpod_checkpointless"
    prefetch_length: int = 10
    val_prefetch_length: int = 10
    lookback_length: int = 2
    actor_num_cpus: float = 1.0
    actor_resources: dict | None = None
    enable_batch_encryption: bool = False
    register_timeout_s: float = 900.0
    consumer_timeout_s: float = 300.0

    def create(
        self,
        dataloader_init_callable: Callable,
        parallel_state_util,
        step: int,
        is_data_loading_rank: Callable,
        create_model_parallel_group_callable: Callable,
        name: str = "Train",
        is_val: bool = False,
        cached_len: int = 0,
    ):
        """
        Create and return a RayReadDataLoader backed by a PrefetchActor on the CPU node.

        Same signature as CacheResumeMMAPConfig.create. dataloader_init_callable,
        is_data_loading_rank and create_model_parallel_group_callable are unused: the real
        dataloader is built on the CPU node by cpu_dataloader_factory.

        Args:
            dataloader_init_callable: Unused.
            parallel_state_util: The parallel state utility instance.
            step: The data step to resume from.
            is_data_loading_rank: Unused.
            create_model_parallel_group_callable: Unused.
            name: The name of the dataloader.
            is_val: A boolean used to toogle between train and val. Returns true for val
            cached_len: Dataloader length cached from a previous create call.
        Returns:
            A RayReadDataLoader.
        """
        if step is None:
            raise ValueError("step must be provided.")
        if self.cpu_dataloader_factory is None:
            raise ValueError("cpu_dataloader_factory must be provided.")
        logger.debug(
            debug_msg(
                f"{self.__class__.__name__} ignores dataloader_init_callable, is_data_loading_rank "
                "and create_model_parallel_group_callable"
            )
        )

        # Lazy imports: ray is an optional dependency
        import ray

        from hyperpod_checkpointless_training.dataloader.ray.producer import (
            get_or_create_prefetch_actor,
        )
        from hyperpod_checkpointless_training.dataloader.ray.read_dataloader import (
            RayReadDataLoader,
        )

        if not ray.is_initialized():
            ray.init(address=self.ray_address, namespace=self.ray_namespace, ignore_reinit_error=True)

        step = 0 if is_val else step
        prefetch_length = self.val_prefetch_length if is_val else self.prefetch_length

        parallel_state = parallel_state_util.parallel_state
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()

        actor = get_or_create_prefetch_actor(
            job_name=self.job_name,
            split="val" if is_val else "train",
            dp_rank=dp_rank,
            dp_size=dp_size,
            dataloader_factory=self.cpu_dataloader_factory,
            lookback_length=self.lookback_length,
            prefetch_length=prefetch_length,
            namespace=self.ray_namespace,
            num_cpus=self.actor_num_cpus,
            resources=self.actor_resources,
            enable_batch_encryption=self.enable_batch_encryption,
            consumer_timeout_s=self.consumer_timeout_s,
        )

        stateless_resume = not is_val and not cached_len and step > 0
        epoch_length = None
        if cached_len and not is_val:
            epoch_length = cached_len
            length, step = self._adjust_for_epoch_resume(cached_len, step)

        consumer_id = dist.get_rank()
        if stateless_resume:
            epoch_length = RayReadDataLoader.remote_length_from_actor(
                actor, consumer_id, self.register_timeout_s
            )
            if epoch_length:
                length, step = self._adjust_for_epoch_resume(epoch_length, step)
            else:
                step *= get_num_microbatches()
                length = 0
                epoch_length = 0

        data_loader = RayReadDataLoader(
            step,
            actor,
            consumer_id=consumer_id,
            prefetch_length=prefetch_length,
            is_val=is_val,
            enable_batch_encryption=self.enable_batch_encryption,
            register_timeout_s=self.register_timeout_s,
        )

        if is_val:
            length = data_loader.remote_length() or cached_len
        elif not cached_len and not stateless_resume:
            remote = data_loader.remote_length()
            length = remote or 0
        data_loader.set_length(length)
        if not is_val:
            data_loader.epoch_length = (
                epoch_length if epoch_length is not None else length
            )
        if len(data_loader) == 0:
            logger.warning(
                debug_msg(
                    f"{name} {data_loader.__class__.__name__} length is 0: no cached length and the "
                    "PrefetchActor has not reported a dataloader length yet"
                )
            )

        logger.info(
            debug_msg(
                f"Created {name} {data_loader.__class__.__name__} (dp_rank={dp_rank}, dp_size={dp_size}, "
                f"step={step}) with length: {len(data_loader)}"
            )
        )
        return data_loader
