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
"""Unit tests for config.py module."""
import ast
import inspect
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from hyperpod_checkpointless_training.dataloader import config as config_module
from hyperpod_checkpointless_training.dataloader.config import (
    MMAPConfig,
    CacheResumeMMAPConfig,
    RayCPUNodeConfig,
)


class TestMMAPConfig(unittest.TestCase):
    """Test cases for MMAPConfig base class."""

    def test_mmap_config_initialization(self):
        """Test that MMAPConfig can be initialized successfully."""
        config = MMAPConfig()
        self.assertIsInstance(config, MMAPConfig)


class TestCacheResumeMMAPConfig(unittest.TestCase):

    def setUp(self):
        """Setup test environment."""
        self.cache_dir = "/test/cache"
        self.config = CacheResumeMMAPConfig(
            cache_dir=self.cache_dir,
            prefetch_length=5,
            val_prefetch_length=5,
            lookback_length=5,
            checkpoint_frequency=10,
            enable_batch_encryption=False,
        )

        self.mock_dataloader_init = Mock()
        self.mock_parallel_state_util = Mock()
        self.mock_is_data_loading_rank = Mock()
        self.mock_create_model_parallel_group = Mock()

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch("hyperpod_checkpointless_training.dataloader.config.CacheResumePrefetchedDataLoader")
    @patch("torch.distributed.get_rank")
    def test_create_prefetched_dataloader_training(
        self, mock_get_rank, mock_prefetched_loader, mock_tensor, mock_all_reduce
    ):
        """Test creates CacheResumePrefetchedDataLoader for training when is_data_loading_rank=True."""
        mock_get_rank.return_value = 0
        self.mock_is_data_loading_rank.return_value = True
        
        mock_pdl_instance = Mock()
        mock_pdl_instance.data_loader = Mock(__len__=Mock(return_value=100))
        mock_pdl_instance.__len__ = Mock(return_value=100)
        mock_prefetched_loader.return_value = mock_pdl_instance
        
        
        mock_tensor.return_value = Mock(item=Mock(return_value=100))
        self.mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = Mock()

        self.config.create(
            self.mock_dataloader_init,
            self.mock_parallel_state_util,
            0,
            self.mock_is_data_loading_rank,
            self.mock_create_model_parallel_group,
            is_val=False,
        )

        mock_prefetched_loader.assert_called_once()
        self.mock_dataloader_init.assert_not_called()

    @patch("torch.distributed.get_rank")
    def test_create_method_missing_data_step(self, mock_get_rank):
        """Test create method raises ValueError when step is None."""
        mock_get_rank.return_value = 0
        self.mock_is_data_loading_rank.return_value = True
        config = CacheResumeMMAPConfig(checkpoint_frequency=50)

        mock_dataloader_callable = Mock()
        mock_parallel_state_util = Mock()
        mock_is_data_loading_rank = Mock(return_value=True)
        create_model_parallel_group_callable = Mock()

        with self.assertRaises(ValueError) as context:
            config.create(
                mock_dataloader_callable,
                mock_parallel_state_util,
                None,
                mock_is_data_loading_rank,
                create_model_parallel_group_callable,
            )

        self.assertIn("step must be provided", str(context.exception))

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumeReadDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_read_dataloader_training(self, mock_get_rank, mock_read_loader, mock_tensor, mock_all_reduce):
        """Test creates CacheResumeReadDataLoader for training when is_data_loading_rank=False."""
        mock_get_rank.return_value = 0
        self.mock_is_data_loading_rank.return_value = False
        
        mock_cdl_instance = Mock()
        mock_cdl_instance.set_length = Mock()
        mock_cdl_instance.__len__ = Mock(return_value=100)
        mock_read_loader.return_value = mock_cdl_instance
        
        # Mock tensor operations for length synchronization
        mock_length_tensor = Mock()
        mock_length_tensor.item.return_value = 100
        mock_tensor.return_value = mock_length_tensor

        # Mock parallel state
        mock_tp_group = Mock()
        self.mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group
        
        self.config.create(
            self.mock_dataloader_init,
            self.mock_parallel_state_util,
            0,
            self.mock_is_data_loading_rank,
            self.mock_create_model_parallel_group,
            is_val=False,
        )

        mock_read_loader.assert_called_once_with(
            dataloader_init_callable=self.mock_dataloader_init,
            model_parallel_group=self.mock_create_model_parallel_group.return_value,
            step=0,
            prefetch_length=5,
            parallel_state_util=self.mock_parallel_state_util,
            force_cold_start=False,
            cache_dir=os.path.join(self.cache_dir, "train"),
            enable_batch_encryption=False,
        )
        self.mock_dataloader_init.assert_not_called()

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumePrefetchedDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_prefetched_dataloader_validation(
        self, mock_get_rank, mock_prefetched_loader, mock_tensor, mock_all_reduce
    ):
        """Test creates CacheResumePrefetchedDataLoader for validation when is_data_loading_rank=True."""
        mock_get_rank.return_value = 0
        self.mock_is_data_loading_rank.return_value = True
        
        # Mock the dataloader and its length
        mock_underlying_dl = Mock()
        mock_underlying_dl.__len__ = Mock(return_value=100)
        
        mock_pdl_instance = Mock()
        mock_pdl_instance.data_loader = mock_underlying_dl
        mock_pdl_instance.__len__ = Mock(return_value=100)
        mock_pdl_instance.set_length = Mock()
        mock_prefetched_loader.return_value = mock_pdl_instance
        
        # Mock tensor operations for length synchronization
        mock_length_tensor = Mock()
        mock_length_tensor.item.return_value = 100
        mock_tensor.return_value = mock_length_tensor

        # Mock parallel state
        mock_tp_group = Mock()
        self.mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group

        self.config.create(
            self.mock_dataloader_init,
            self.mock_parallel_state_util,
            100,
            self.mock_is_data_loading_rank,
            self.mock_create_model_parallel_group,
            is_val=True,
        )

        mock_prefetched_loader.assert_called_once_with(
            step=0,
            dataloader_init_callable=self.mock_dataloader_init,
            model_parallel_group=self.mock_create_model_parallel_group.return_value,
            parallel_state_util=self.mock_parallel_state_util,
            force_cold_start=True,
            model_checkpoint_frequency=10,
            lookback_length=5,
            prefetch_length=5,
            cache_dir=os.path.join(self.cache_dir, "val"),
            enable_batch_encryption=False,
        )

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumeReadDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_read_dataloader_validation(self, mock_get_rank, mock_read_loader, mock_tensor, mock_all_reduce):
        """Test creates CacheResumeReadDataLoader for validation when is_data_loading_rank=False."""
        mock_get_rank.return_value = 0
        self.mock_is_data_loading_rank.return_value = False
        
        mock_cdl_instance = Mock()
        mock_cdl_instance.set_length = Mock()
        mock_cdl_instance.__len__ = Mock(return_value=100)
        mock_read_loader.return_value = mock_cdl_instance
        
        # Mock tensor operations for length synchronization
        mock_length_tensor = Mock()
        mock_length_tensor.item.return_value = 100
        mock_tensor.return_value = mock_length_tensor

        # Mock parallel state
        mock_tp_group = Mock()
        self.mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group
        
        self.config.create(
            self.mock_dataloader_init,
            self.mock_parallel_state_util,
            100,
            self.mock_is_data_loading_rank,
            self.mock_create_model_parallel_group,
            is_val=True,
        )

        mock_read_loader.assert_called_once_with(
            dataloader_init_callable=self.mock_dataloader_init,
            model_parallel_group=self.mock_create_model_parallel_group.return_value,
            step=0,
            prefetch_length=5,
            parallel_state_util=self.mock_parallel_state_util,
            force_cold_start=True,
            cache_dir=os.path.join(self.cache_dir, "val"),
            enable_batch_encryption=False,
        )


class TestCacheResumeMMAPConfigWithLengthSetting(unittest.TestCase):
    """Test cases for CacheResumeMMAPConfig with length setting functionality."""

    def setUp(self):
        """Setup test environment."""
        self.cache_dir = "/test/cache"
        self.config = CacheResumeMMAPConfig(
            cache_dir=self.cache_dir,
            prefetch_length=5,
            val_prefetch_length=5,
            lookback_length=5,
            checkpoint_frequency=10,
            enable_batch_encryption=False,
        )

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumePrefetchedDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_method_sets_length_for_pdl(
        self, mock_get_rank, mock_pdl_class, mock_tensor, mock_all_reduce
    ):
        """Test create method sets length on PrefetchedDataLoader."""
        mock_get_rank.return_value = 0
        
        # Mock the dataloader and its length
        mock_underlying_dl = Mock()
        mock_underlying_dl.__len__ = Mock(return_value=100)
        
        mock_pdl_instance = Mock()
        mock_pdl_instance.data_loader = mock_underlying_dl
        mock_pdl_instance.__len__ = Mock(return_value=100)
        mock_pdl_instance.set_length = Mock()
        mock_pdl_class.return_value = mock_pdl_instance

        # Mock tensor operations for length synchronization
        mock_length_tensor = Mock()
        mock_length_tensor.item.return_value = 100
        mock_tensor.return_value = mock_length_tensor

        # Mock parallel state
        mock_parallel_state_util = Mock()
        mock_tp_group = Mock()
        mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group

        mock_dataloader_callable = Mock()
        probe_dataloader = Mock()
        probe_dataloader.__len__ = Mock(return_value=100)
        mock_dataloader_callable.return_value = probe_dataloader
        mock_is_data_loading_rank = Mock(return_value=True)
        mock_create_model_parallel_group_callable = Mock()

        with patch(
            "hyperpod_checkpointless_training.dataloader.config.get_num_microbatches",
            return_value=4,
        ):
            result = self.config.create(
                mock_dataloader_callable,
                mock_parallel_state_util,
                30,
                mock_is_data_loading_rank,
                mock_create_model_parallel_group_callable,
                cached_len=0,
            )

        # Verify the dataloader was created and length was set
        self.assertEqual(result, mock_pdl_instance)
        mock_pdl_instance.initialize_data_loader.assert_called_once()
        self.assertEqual(mock_pdl_class.call_args.kwargs["step"], 20)
        mock_pdl_instance.set_length.assert_called_once_with(80)
        self.assertEqual(mock_pdl_instance.epoch_length, 100)
        mock_dataloader_callable.assert_called_once_with(target_load_step=0)
        
        # The full epoch length is probed and reduced once before loader creation.
        mock_tensor.assert_called_once_with(100, dtype=torch.long, device='cuda')
        mock_all_reduce.assert_called_once_with(
            mock_length_tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=mock_tp_group,
        )

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumeReadDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_method_sets_length_for_cdl(
        self, mock_get_rank, mock_cdl_class, mock_tensor, mock_all_reduce
    ):
        """Test create method sets length on CacheResumeReadDataLoader."""
        mock_get_rank.return_value = 1
        
        mock_cdl_instance = Mock()
        mock_cdl_instance.set_length = Mock()
        mock_cdl_instance.__len__ = Mock(return_value=100)
        mock_cdl_class.return_value = mock_cdl_instance

        # Mock tensor operations for length synchronization
        mock_length_tensor = Mock()
        mock_length_tensor.item.return_value = 100
        mock_tensor.return_value = mock_length_tensor

        # Mock parallel state
        mock_parallel_state_util = Mock()
        mock_tp_group = Mock()
        mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group

        mock_dataloader_callable = Mock()
        mock_is_data_loading_rank = Mock(return_value=False)
        mock_create_model_parallel_group_callable = Mock()

        with patch(
            "hyperpod_checkpointless_training.dataloader.config.get_num_microbatches",
            return_value=4,
        ):
            result = self.config.create(
                mock_dataloader_callable,
                mock_parallel_state_util,
                30,
                mock_is_data_loading_rank,
                mock_create_model_parallel_group_callable,
                cached_len=0,
            )

        # Verify the dataloader was created and length was set
        self.assertEqual(result, mock_cdl_instance)
        self.assertEqual(mock_cdl_class.call_args.kwargs["step"], 20)
        mock_cdl_instance.set_length.assert_called_once_with(80)
        self.assertEqual(mock_cdl_instance.epoch_length, 100)
        mock_dataloader_callable.assert_not_called()
        
        # The non-loading rank contributes zero to the sole length reduction.
        mock_tensor.assert_called_once_with(0, dtype=torch.long, device='cuda')
        mock_all_reduce.assert_called_once_with(
            mock_length_tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=mock_tp_group,
        )

    @patch("hyperpod_checkpointless_training.dataloader.config.get_num_microbatches")
    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumePrefetchedDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_method_uses_cached_length(
        self, mock_get_rank, mock_pdl_class, mock_tensor, mock_all_reduce, mock_num_mbs
    ):
        """Test create method uses cached_len when provided."""
        mock_get_rank.return_value = 0
        
        mock_pdl_instance = Mock()
        mock_pdl_instance.set_length = Mock()
        mock_pdl_instance.__len__ = Mock(return_value=50)
        mock_pdl_class.return_value = mock_pdl_instance

        mock_parallel_state_util = Mock()
        mock_dataloader_callable = Mock()
        mock_is_data_loading_rank = Mock(return_value=True)
        mock_create_model_parallel_group_callable = Mock()

        mock_num_mbs.return_value = 1

        # Call with cached_len provided
        self.config.create(
            mock_dataloader_callable,
            mock_parallel_state_util,
            100,
            mock_is_data_loading_rank,
            mock_create_model_parallel_group_callable,
            cached_len=150,
        )

        # step 100 for cached_len 150 should be 50 samples remaining (set length to samples remaining)
        mock_pdl_instance.set_length.assert_called_once_with(50)
        mock_tensor.assert_not_called()  # No tensor operations when cached_len is provided
        mock_all_reduce.assert_not_called()  # No all_reduce when cached_len is provided

    @patch("hyperpod_checkpointless_training.dataloader.config.get_num_microbatches")
    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumePrefetchedDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_method_max_cached_and_synchronized_length(
        self, mock_get_rank, mock_pdl_class, mock_tensor, mock_all_reduce, mock_num_mbs
    ):
        """Test create method uses cached_len when provided (no synchronization occurs)."""
        mock_get_rank.return_value = 0
        
        # Mock the dataloader and its length
        mock_underlying_dl = Mock()
        mock_underlying_dl.__len__ = Mock(return_value=200)  # This won't be used since cached_len is provided
        
        mock_pdl_instance = Mock()
        mock_pdl_instance.data_loader = mock_underlying_dl
        mock_pdl_instance.__len__ = Mock(return_value=50)
        mock_pdl_instance.set_length = Mock()
        mock_pdl_class.return_value = mock_pdl_instance

        # Mock parallel state
        mock_parallel_state_util = Mock()
        mock_tp_group = Mock()
        mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group

        mock_dataloader_callable = Mock()
        mock_is_data_loading_rank = Mock(return_value=True)
        mock_create_model_parallel_group_callable = Mock()

        mock_num_mbs.return_value = 1

        # Call with cached_len=150 - no synchronization should occur
        self.config.create(
            mock_dataloader_callable,
            mock_parallel_state_util,
            100,
            mock_is_data_loading_rank,
            mock_create_model_parallel_group_callable,
            cached_len=150,
        )

        # step 100 for cached_len 150 should be 50 samples remaining (set length to samples remaining)
        mock_pdl_instance.set_length.assert_called_once_with(50)
        # Verify no tensor operations occurred since cached_len was provided
        mock_tensor.assert_not_called()
        mock_all_reduce.assert_not_called()

    def test_create_method_validation_step_none(self):
        """Test create method raises ValueError when step is None."""
        config = CacheResumeMMAPConfig()

        mock_dataloader_callable = Mock()
        mock_parallel_state_util = Mock()
        mock_is_data_loading_rank = Mock(return_value=True)
        mock_create_model_parallel_group_callable = Mock()

        with self.assertRaises(ValueError) as context:
            config.create(
                mock_dataloader_callable,
                mock_parallel_state_util,
                None,
                mock_is_data_loading_rank,
                mock_create_model_parallel_group_callable,
            )

        self.assertIn("step must be provided", str(context.exception))


RAY_PKG = "hyperpod_checkpointless_training.dataloader.ray"
RAY_PRODUCER = f"{RAY_PKG}.producer"
RAY_READ_DATALOADER = f"{RAY_PKG}.read_dataloader"


class TestRayCPUNodeConfig(unittest.TestCase):
    """Test cases for RayCPUNodeConfig with ray and the ray dataloader modules faked."""

    def setUp(self):
        self.factory = Mock(name="cpu_dataloader_factory")
        self.config = RayCPUNodeConfig(
            cpu_dataloader_factory=self.factory,
            job_name="job",
            ray_address="ray://head:10001",
            ray_namespace="ns",
            prefetch_length=7,
            val_prefetch_length=3,
            lookback_length=4,
            actor_num_cpus=2.0,
        )

        self.fake_ray = types.ModuleType("ray")
        self.fake_ray.is_initialized = Mock(return_value=False)
        self.fake_ray.init = Mock()

        self.actor = Mock(name="actor")
        self.fake_producer = types.ModuleType(RAY_PRODUCER)
        self.fake_producer.CPU_DATALOADER_RESOURCE = "hpct_cpu_dataloader"
        self.fake_producer.get_or_create_prefetch_actor = Mock(return_value=self.actor)

        self.read_dl = MagicMock(name="read_dl")
        self.read_dl.remote_length.return_value = 100
        self.read_dl.__len__.return_value = 100
        self.fake_read_dataloader = types.ModuleType(RAY_READ_DATALOADER)
        self.fake_read_dataloader.RayReadDataLoader = Mock(return_value=self.read_dl)
        self.fake_read_dataloader.RayReadDataLoader.remote_length_from_actor = Mock(
            return_value=100
        )

        fake_pkg = types.ModuleType(RAY_PKG)
        fake_pkg.__path__ = []
        fake_pkg.producer = self.fake_producer
        fake_pkg.read_dataloader = self.fake_read_dataloader

        self.parallel_state_util = Mock()
        self.parallel_state_util.parallel_state.get_data_parallel_rank.return_value = 3
        self.parallel_state_util.parallel_state.get_data_parallel_world_size.return_value = 8

        self.dataloader_init = Mock()
        self.is_data_loading_rank = Mock()
        self.create_model_parallel_group = Mock()

        num_microbatches_patcher = patch(
            "hyperpod_checkpointless_training.dataloader.config.get_num_microbatches",
            return_value=1,
        )
        num_microbatches_patcher.start()
        self.addCleanup(num_microbatches_patcher.stop)

        patchers = [
            patch.dict(
                sys.modules,
                {
                    "ray": self.fake_ray,
                    RAY_PKG: fake_pkg,
                    RAY_PRODUCER: self.fake_producer,
                    RAY_READ_DATALOADER: self.fake_read_dataloader,
                },
            ),
            patch("torch.distributed.get_rank", return_value=11),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _create(self, config=None, step=100, **kwargs):
        return (config or self.config).create(
            self.dataloader_init,
            self.parallel_state_util,
            step,
            self.is_data_loading_rank,
            self.create_model_parallel_group,
            **kwargs,
        )

    def test_defaults(self):
        config = RayCPUNodeConfig()
        self.assertIsInstance(config, MMAPConfig)
        self.assertIsNone(config.cpu_dataloader_factory)
        self.assertEqual(config.job_name, "default")
        self.assertEqual(config.ray_address, "auto")
        self.assertEqual(config.ray_namespace, "hyperpod_checkpointless")
        self.assertEqual(config.prefetch_length, 10)
        self.assertEqual(config.val_prefetch_length, 10)
        self.assertEqual(config.lookback_length, 2)
        self.assertEqual(config.actor_num_cpus, 1.0)
        self.assertIsNone(config.actor_resources)
        self.assertFalse(config.enable_batch_encryption)
        self.assertEqual(config.register_timeout_s, 900.0)

    def test_config_module_has_no_top_level_ray_import(self):
        tree = ast.parse(inspect.getsource(config_module))
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                self.assertFalse(name == "ray" or name.startswith("ray."), name)
                self.assertFalse(name.startswith(RAY_PKG), name)

    def test_create_step_none_raises(self):
        with self.assertRaises(ValueError) as context:
            self._create(step=None)
        self.assertIn("step must be provided", str(context.exception))
        self.fake_ray.init.assert_not_called()
        self.fake_producer.get_or_create_prefetch_actor.assert_not_called()

    def test_create_missing_factory_raises(self):
        with self.assertRaises(ValueError) as context:
            self._create(config=RayCPUNodeConfig())
        self.assertIn("cpu_dataloader_factory", str(context.exception))
        self.fake_ray.init.assert_not_called()
        self.fake_producer.get_or_create_prefetch_actor.assert_not_called()

    def test_create_inits_ray_when_not_initialized(self):
        self._create()
        self.fake_ray.init.assert_called_once_with(
            address="ray://head:10001", namespace="ns", ignore_reinit_error=True
        )

    def test_create_skips_ray_init_when_initialized(self):
        self.fake_ray.is_initialized.return_value = True
        self._create()
        self.fake_ray.init.assert_not_called()

    @patch("torch.distributed.all_reduce")
    def test_create_train(self, mock_all_reduce):
        result = self._create(step=100)

        self.assertIs(result, self.read_dl)
        self.fake_producer.get_or_create_prefetch_actor.assert_called_once_with(
            job_name="job",
            split="train",
            dp_rank=3,
            dp_size=8,
            dataloader_factory=self.factory,
            lookback_length=4,
            prefetch_length=7,
            namespace="ns",
            num_cpus=2.0,
            resources=None,
            enable_batch_encryption=False,
            consumer_timeout_s=300.0,
        )
        self.fake_read_dataloader.RayReadDataLoader.assert_called_once_with(
            0,
            self.actor,
            consumer_id=11,
            prefetch_length=7,
            is_val=False,
            enable_batch_encryption=False,
            register_timeout_s=900.0,
        )
        self.fake_read_dataloader.RayReadDataLoader.remote_length_from_actor.assert_called_once_with(
            self.actor, 11, 900.0
        )
        self.read_dl.set_length.assert_called_once_with(100)
        self.assertEqual(self.read_dl.epoch_length, 100)
        mock_all_reduce.assert_not_called()
        self.dataloader_init.assert_not_called()
        self.is_data_loading_rank.assert_not_called()
        self.create_model_parallel_group.assert_not_called()

    def test_create_passes_resources_and_encryption(self):
        config = RayCPUNodeConfig(
            cpu_dataloader_factory=self.factory,
            actor_resources={"custom": 1},
            enable_batch_encryption=True,
            register_timeout_s=5.0,
            consumer_timeout_s=42.0,
        )
        self._create(config=config)

        actor_kwargs = self.fake_producer.get_or_create_prefetch_actor.call_args.kwargs
        self.assertEqual(actor_kwargs["resources"], {"custom": 1})
        self.assertTrue(actor_kwargs["enable_batch_encryption"])
        self.assertEqual(actor_kwargs["consumer_timeout_s"], 42.0)
        self.assertEqual(actor_kwargs["job_name"], "default")
        self.assertEqual(actor_kwargs["namespace"], "hyperpod_checkpointless")
        self.assertTrue(self.fake_read_dataloader.RayReadDataLoader.call_args.kwargs["enable_batch_encryption"])
        self.assertEqual(self.fake_read_dataloader.RayReadDataLoader.call_args.kwargs["register_timeout_s"], 5.0)
        self.fake_ray.init.assert_called_once_with(
            address="auto", namespace="hyperpod_checkpointless", ignore_reinit_error=True
        )

    @patch(
        "hyperpod_checkpointless_training.dataloader.config.get_num_microbatches",
        return_value=4,
    )
    def test_create_unknown_remote_length_sets_zero(self, mock_num_mbs):
        self.fake_read_dataloader.RayReadDataLoader.remote_length_from_actor.return_value = None
        self._create(step=30)
        mock_num_mbs.assert_called_once_with()
        self.assertEqual(
            self.fake_read_dataloader.RayReadDataLoader.call_args.args[0], 120
        )
        self.read_dl.set_length.assert_called_once_with(0)
        self.assertEqual(self.read_dl.epoch_length, 0)

    @patch(
        "hyperpod_checkpointless_training.dataloader.config.get_num_microbatches",
        return_value=4,
    )
    def test_create_stateless_resume_scales_and_wraps(self, mock_num_mbs):
        self.fake_read_dataloader.RayReadDataLoader.remote_length_from_actor.return_value = 100

        self._create(step=30)
        mock_num_mbs.assert_called_once_with()

        self.assertEqual(
            self.fake_read_dataloader.RayReadDataLoader.call_args.args[0], 20
        )
        self.read_dl.set_length.assert_called_once_with(80)
        self.assertEqual(self.read_dl.epoch_length, 100)

    @patch("torch.distributed.all_reduce")
    def test_create_val(self, mock_all_reduce):
        self._create(step=100, name="Val", is_val=True)

        actor_kwargs = self.fake_producer.get_or_create_prefetch_actor.call_args.kwargs
        self.assertEqual(actor_kwargs["split"], "val")
        self.assertEqual(actor_kwargs["prefetch_length"], 3)
        self.fake_read_dataloader.RayReadDataLoader.assert_called_once_with(
            0,
            self.actor,
            consumer_id=11,
            prefetch_length=3,
            is_val=True,
            enable_batch_encryption=False,
            register_timeout_s=900.0,
        )
        mock_all_reduce.assert_not_called()

    @patch("hyperpod_checkpointless_training.dataloader.config.get_num_microbatches", return_value=1)
    def test_create_cached_len_adjusts_for_epoch_resume(self, mock_num_mbs):
        self.read_dl.remote_length.return_value = 40
        self._create(step=100, cached_len=150)

        # batch_in_epoch = 100 % 150 = 100, remaining = 50
        self.assertEqual(self.fake_read_dataloader.RayReadDataLoader.call_args.args[0], 100)
        self.read_dl.set_length.assert_called_once_with(50)

    @patch("hyperpod_checkpointless_training.dataloader.config.get_num_microbatches", return_value=2)
    def test_create_cached_len_uses_remaining_not_remote_length(self, mock_num_mbs):
        self.read_dl.remote_length.return_value = 200
        self._create(step=100, cached_len=150)

        # batch_in_epoch = (100 * 2) % 150 = 50, remaining = 100 < remote_length 200
        self.assertEqual(self.fake_read_dataloader.RayReadDataLoader.call_args.args[0], 50)
        self.read_dl.set_length.assert_called_once_with(100)

    @patch("hyperpod_checkpointless_training.dataloader.config.get_num_microbatches", return_value=2)
    def test_create_val_cached_len_uses_remote_length_without_epoch_adjustment(
        self, mock_num_mbs
    ):
        self.read_dl.remote_length.return_value = 30
        with patch.object(self.config, "_adjust_for_epoch_resume") as adjust:
            self._create(step=7, cached_len=99, is_val=True)

        adjust.assert_not_called()
        mock_num_mbs.assert_not_called()
        self.assertEqual(self.fake_read_dataloader.RayReadDataLoader.call_args.args[0], 0)
        self.read_dl.set_length.assert_called_once_with(30)

    def test_create_val_uses_cached_length_when_remote_length_is_unknown(self):
        self.read_dl.remote_length.return_value = None
        self._create(step=7, cached_len=30, is_val=True)

        self.assertEqual(self.fake_read_dataloader.RayReadDataLoader.call_args.args[0], 0)
        self.read_dl.set_length.assert_called_once_with(30)

    def test_adjust_for_epoch_resume_shared_with_cache_resume_config(self):
        self.assertIs(
            RayCPUNodeConfig._adjust_for_epoch_resume, MMAPConfig._adjust_for_epoch_resume
        )
        self.assertIs(
            CacheResumeMMAPConfig._adjust_for_epoch_resume, MMAPConfig._adjust_for_epoch_resume
        )
