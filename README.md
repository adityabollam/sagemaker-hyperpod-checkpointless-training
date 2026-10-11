# Checkpointless training on Amazon SageMaker HyperPod

Checkpointless training on Amazon SageMaker HyperPod eliminates disruptive checkpoint-restart cycles, maintaining forward training momentum despite failures, reducing recovery time from hours to minutes.

## Key Features

- **In-Process Recovery**: Recover from node failures in minutes without losing training progress by using redundant model copies stored in GPU memory
- **Fast Initialization**: Accelerate training restarts by bypassing expensive communication (NCCL/Gloo) setup processes
- **Smart Data Caching**: Pre-load and cache training data batches to eliminate delays when resuming training after failures
- **Built-in Redundancy**: Leverage distributed optimizer instances for checkpointless recovery
- **NeMo Integration**: Works seamlessly with PyTorch Lightning and NVIDIA NeMo toolkit for large language model training

## Getting Started Examples

| Model | Method | Size | Nodes | Instance | Accelerator | Recipe | Script |
|-------|--------|------|-------|----------|-------------|--------|--------|
| GPT OSS | Full finetune example | 120b | 16 | p5.48xlarge | GPU H100 | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/recipes_collection/recipes/fine-tuning/gpt_oss/checkpointless_gpt_oss_120b_full_fine_tuning.yaml) | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/launcher_scripts/gpt_oss/run_checkpointless_gpt_oss_120b_full_fine_tuning.sh) |
| GPT OSS | LoRA-example | 120b | 2 | p5.48xlarge | GPU H100 | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/recipes_collection/recipes/fine-tuning/gpt_oss/checkpointless_gpt_oss_120b_lora.yaml) | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/launcher_scripts/gpt_oss/run_checkpointless_gpt_oss_120b_lora.sh) |
| Llama3 | Pretrain example | 70b | 16 | p5.48xlarge | GPU H100 | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/recipes_collection/recipes/training/llama/checkpointless_llama3_70b_pretrain.yaml) | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/launcher_scripts/llama/run_checkpointless_llama3_70b_pretrain.sh) |
| Llama3 | LoRA-example | 70b | 2 | p5.48xlarge | GPU H100 | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/recipes_collection/recipes/fine-tuning/llama/checkpointless_llama3_70b_lora.yaml) | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/launcher_scripts/llama/run_checkpointless_llama3_70b_lora.sh) |

## User Guide

For comprehensive documentation including installation steps, environment setup, configuration options, and detailed usage examples, see the tutorials at [Amazon SageMaker HyperPod Checkpointless training](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-eks-checkpointless.html)..

## Quick Start Guide

### Launch Training

#### Hyperpod Recipe Launcher

You can use the SageMaker HyperPod recipes to submit your training job. Using the recipes involves updating k8s.yaml, config.yaml and running the launch script.

```bash
bash launcher_scripts/gpt_oss/run_checkpointless_nemo_gpt_oss_120b_fine_tuning.sh
```

#### Launch Using kubectl

Alternatively, you can deploy the training job directly using kubectl:

```bash
kubectl apply -f <path_to_config>.yaml
```

#### Monitor Job Status

```bash
kubectl get pods
kubectl logs <pod-name>
```

For detailed installation steps, environment setup, and configuration options, see the tutorials at [Amazon SageMaker HyperPod Checkpointless training](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-eks-checkpointless.html).

## Dedicated CPU-node dataloading with Ray

By default every GPU node runs a prefetching dataloader (`CacheResumeMMAPConfig`) that iterates the dataset on the GPU host and caches batches in `/dev/shm`. `RayCPUNodeConfig` moves dataset iteration to a dedicated CPU instance instead:

- A Ray head on the CPU node runs one detached `PrefetchActor` per (job, split, DP rank). Each actor builds the real dataloader (with its CPU workers), stores batches in the Ray object store and keeps a window of recent and upcoming steps.
- Every GPU rank reads its DP replica's batches through a `RayReadDataLoader`, which prefetches a few steps ahead. Ranks on the same GPU node share one local copy of each batch.
- Restarts need no collective: the window lives on the CPU node, so ranks just re-register at the resumed step.

See [docs/ray_cpu_dataloader.md](docs/ray_cpu_dataloader.md) for the design and interface contract.

### Install

```bash
pip install .[ray]   # ray[default]>=2.40,<3
```

### Cluster setup

1. Deploy the Ray head on a CPU instance: [examples/llama3/launch/ray_cpu_dataloader_head.yaml](examples/llama3/launch/ray_cpu_dataloader_head.yaml) (set the image to your training image and the `nodeSelector` to your CPU instance type). It runs `ray start --head ... --resources='{"hpct_cpu_dataloader": 1000}'` and exposes it through the `hpct-ray-cpu-dataloader-head` Service.
2. In the training job, join each GPU pod to the Ray cluster before launching training:

   ```bash
   ray start --address=hpct-ray-cpu-dataloader-head:6379 --num-cpus=0 --block &
   # wait for the raylet, then: hyperpodrun ... / torchrun ...
   ```

   Training processes connect with `ray_address="auto"` (the default).

### Configuration

Replace `CacheResumeMMAPConfig` with `RayCPUNodeConfig` and pass a picklable dataloader factory that runs on the CPU node. For example, in `examples/llama3/llama3_70b_pretrain_checkpointless.py`:

```python
# from hyperpod_checkpointless_training.dataloader.config import CacheResumeMMAPConfig
from hyperpod_checkpointless_training.dataloader.config import RayCPUNodeConfig
from examples.dataloader.ray_cpu_dataloader_factory import HFDatasetCPUDataLoaderFactory

        return CheckpointlessDataModule(
            cfg=self.cfg,
            data_module=MMAPDataModule(
                data_module=base_data_module,
                # mmap_config=CacheResumeMMAPConfig(
                #     cache_dir=self.cfg.mmap.cache_dir, checkpoint_frequency=self.cfg.mmap.checkpoint_frequency
                # ),
                mmap_config=RayCPUNodeConfig(
                    cpu_dataloader_factory=HFDatasetCPUDataLoaderFactory(
                        dataset_path=self.cfg.dataset.dataset_path,
                        val_dataset_path=self.cfg.dataset.get("val_dataset_path", None),
                        seq_length=self.cfg.data.seq_length,
                        micro_batch_size=self.cfg.data.micro_batch_size,
                        num_workers=self.cfg.dataset.get("num_workers", 4),
                        shuffle=self.cfg.dataset.get("shuffle", False),
                        drop_last=self.cfg.dataset.get("drop_last", True),
                        keep_in_memory=self.cfg.dataset.get("keep_in_memory", False),
                    ),
                    job_name=os.environ["JOB_NAME"],  # unique per job: actor names are derived from it
                ),
            ),
        )
```

Other options: `prefetch_length` / `val_prefetch_length` (steps produced and fetched ahead), `lookback_length` (already-consumed steps kept for rollback on restart), `actor_num_cpus`, `actor_resources` (defaults to the `hpct_cpu_dataloader` resource), `ray_namespace`, and `enable_batch_encryption`.

### Fault and resume behaviour

- **GPU fault, CPU node alive (warm start)**: on in-process restart the ranks re-register at the resumed step. If that step is still in the actor's window (within `lookback_length` behind the slowest consumer, or already prefetched), reading continues immediately from the cached batches.
- **Resume step outside the window**: the actor drops its window and rebuilds the dataloader at the resumed step (cold start).
- **CPU node or actor loss (cold start)**: in-flight reads fail with a Ray error, which triggers a restart; on restart a new actor is created and rebuilds the dataloader at the resumed step. This relies on the dataloader factory being seekable via `target_load_step`.

### Current limitations

- The CPU node is a single failure domain: losing it loses all prefetched batches and forces a cold start for every DP replica.
- No replicated producers or persisted window yet; the CPU tier itself cannot warm start.

## Recommended Requirements

| Component | Version |
|-----------|---------|
| Python | >=3.12 |
| PyTorch | >=2.6.0 |
| NeMo Toolkit | 2.6.0rc0 |
| CUDA | 12.5+ |
| Infrastructure | AWS HyperPod Kubernetes cluster |
| Storage | Shared storage (FSx/NFS) |

---

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
**Note**: This repository is temporarily not accepting pull requests.

## License

This project is licensed under the Apache-2.0 License.