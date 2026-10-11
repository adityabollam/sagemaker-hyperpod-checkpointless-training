# Ray CPU-node DataLoader (design + interface contract)

Status: implementation in progress. This document is the **contract** every piece of the
`hyperpod_checkpointless_training.dataloader.ray` package is built against. If an
implementation needs to deviate from a signature below, change this document in the same commit.

## Motivation

Today every data-parallel (DP) replica runs a `CacheResumePrefetchedDataLoader` (PDL) on the tp0
ranks of each GPU node. The PDL owns the real dataloader, `torch.save`s batches into mmap files under
`/dev/shm/pdl_cache`, and the other ranks (`CacheResumeReadDataLoader`, CDL) read those files. On
restart the PDLs `all_gather` their cache sizes over the model-parallel group and prune to the minimum.

Problems this solves:

* GPU hosts spend CPU cores, host RAM and `/dev/shm` on dataset iteration, collation and the
  `save_worker` pool, competing with NCCL host threads and checkpoint I/O.
* A GPU-node failure destroys that node's `/dev/shm` cache, shrinking the warm-start window for the
  whole DP group (min-prune) or forcing a COLD_START.
* Restart requires a collective (`all_gather_cache_size`) before data can flow.

## Design

```
            CPU node(s)  (Ray head, custom resource "hpct_cpu_dataloader")
  +-------------------------------------------------------------------+
  | PrefetchActor "<job>-train-dp0"   PrefetchActor "<job>-train-dp1" |  one detached actor per
  |   torch DataLoader(num_workers=N)   ...                           |  (job, split, dp_rank)
  |   BatchWindow: step -> ObjectRef                                  |
  |   ray.put(batch) -> Plasma                                        |
  +-------------------------------------------------------------------+
                 |  ray.get(ObjectRef)  (pulled once per GPU node into its local Plasma)
  +--------------v--------------+   +-----------------------------+
  | GPU node: every rank runs   |   | GPU node ...                |
  | RayReadDataLoader(dp_rank)  |   |                             |
  +-----------------------------+   +-----------------------------+
```

* **Producer** – `PrefetchActor` (one per `(job_name, split, dp_rank)`), a detached, named Ray
  actor scheduled onto the CPU node via a custom resource. It builds the real dataloader with
  `cpu_dataloader_factory(dp_rank=..., dp_size=..., target_load_step=..., is_val=...)` (CPU workers
  live here), `ray.put`s each batch and records `step -> ObjectRef` in its `BatchWindow`.
* **Index** – the `BatchWindow` inside each actor *is* the metadata index for that DP replica (the
  replacement for the mmap directory listing): ordered `step -> ObjectRef`, producer cursor,
  per-consumer progress, final step. Sharding the index per DP replica means there is no single
  metadata actor on the hot path and the design extends to a sharded CPU tier (actors for different
  DP ranks can live on different CPU nodes).
* **Consumer** – every GPU rank (tp/pp/cp included) builds a `RayReadDataLoader` keyed by its DP
  rank. All ranks of one DP replica read the same ObjectRef; ranks on the same GPU node share one
  local Plasma copy, so cross-node traffic is one pull per (GPU node, DP replica) per step.
  A background thread prefetches `ray.get`s `prefetch_length` steps ahead into a local queue so the
  training loop never waits on the network in steady state.
* Batches are kept on CPU (same as today's PDL/CDL); Lightning/Megatron moves them to GPU.

### Cold start

1. CPU node runs `ray start --head --resources='{"hpct_cpu_dataloader": 1000}'`.
   GPU nodes run `ray start --address=<head>:6379 --num-cpus=0` before `torchrun`, so every GPU
   node has a local raylet + Plasma store. Training processes call `ray.init(address="auto")`.
2. `MMAPDataModule.train_dataloader()` calls `RayCPUNodeConfig.create(...)` on every rank.
3. `create` gets-or-creates the actor for its DP rank and builds a `RayReadDataLoader`, whose
   constructor calls `actor.register(consumer_id=global_rank, step=global_step)`. The first register
   on a fresh actor reseeks: it builds the dataloader at `target_load_step=step` and starts the
   producer thread. Response mode is `COLD_START`.
4. Length comes from `actor.length()` (= `len(dataloader)`), so no TP `all_reduce` is needed.

### Fault -> in-process restart, CPU node alive (WARM_START)

1. `HPDataLoaderAbort` calls `dataloader.stop()` -> consumer threads stop; the detached actors keep
   their windows.
2. Ranks rebuild dataloaders at the resumed `global_step` (which may be *behind* steps already
   consumed; the `lookback_length` window exists for exactly this).
3. `register(consumer_id, step)`: if `window.can_resume_from(step)` the actor keeps its window and
   cursor (`WARM_START`); the consumer starts reading at `step` immediately. Otherwise the actor
   resets the window and reseeks the dataloader to `step` (`COLD_START`).
4. No collective is needed: the window is central, so the old min-prune across PDLs disappears.

### CPU node failure

Objects created by `ray.put` are owned by the actor; if the actor/CPU node dies, its objects are
gone. Consumers see `ray.exceptions.RayActorError`/`ObjectLostError`. On the next in-process restart
`create` calls get-or-create, a new actor is created and the first `register` reseeks to
`global_step` (`COLD_START`). Determinism relies on the existing contract that the dataloader is
seekable via `target_load_step`.

Follow-ups (not in the first PR): replicated producers on 2+ CPU nodes, a persisted manifest for
warm restart of the CPU tier itself, hierarchical relay for very large fan-out.

### Validation dataloader

`is_val=True` -> step is always 0 and every `__iter__` must restart from 0 (same as
`force_cold_start` today). Each consumer counts its own `__iter__` calls and passes
`reseek_token=f"val-{n}"`; the actor reseeks only when the token differs from the last one it saw,
so the N consumers of one DP replica trigger exactly one reseek per validation epoch.

## Package layout

```
src/hyperpod_checkpointless_training/dataloader/ray/
    __init__.py          # no top-level `import ray` (ray is an optional dependency)
    window.py            # BatchWindow  (pure python, no ray import)
    producer.py          # PrefetchActor, get_or_create_prefetch_actor, prefetch_actor_name
    read_dataloader.py   # RayReadDataLoader
src/hyperpod_checkpointless_training/dataloader/config.py   # + RayCPUNodeConfig
tests/dataloader/ray/...
```

`ray` is an optional dependency: `[project.optional-dependencies] ray = ["ray[default]>=2.40,<3"]`.
Modules that need ray import it inside functions or at module top of `producer.py` /
`read_dataloader.py` only; `config.py` must stay importable without ray (import lazily in
`RayCPUNodeConfig.create`).

Constants (in `window.py`):

```python
WARM_START = RestartMode.WARM_START.value   # "warm_start"  (RestartMode from dataloader.mmap.utils)
COLD_START = RestartMode.COLD_START.value   # "cold_start"
```

## `window.py` – `BatchWindow`

Pure bookkeeping, not thread-safe (the actor holds a lock around it). Refs are typed `Any` so this
module never imports ray.

```python
class BatchWindow:
    def __init__(self, lookback_length: int, prefetch_length: int): ...
    refs: "OrderedDict[int, Any]"     # step -> ObjectRef, ascending, contiguous
    producer_step: int | None         # next step the producer will add; None until first reset()
    final_step: int | None            # first step that does not exist (dataset exhausted)
    consumer_steps: dict[int, int]    # consumer_id -> next step that consumer will read

    def reset(self, step: int) -> list:            # clear everything; producer_step = step; returns dropped refs
    @property
    def low_watermark(self) -> int | None:         # min(refs) if refs else producer_step
    @property
    def high_watermark(self) -> int | None:        # producer_step - 1 (None if producer_step is None)
    def can_resume_from(self, step: int) -> bool:  # producer_step is not None and low_watermark <= step <= producer_step
    def register_consumer(self, consumer_id: int, step: int) -> None:  # overwrite (NOT max): restarts roll back
    def ack(self, consumer_id: int, next_step: int) -> list:           # consumer_steps[id] = max(old, next_step); return evict()
    def evict(self) -> list:
        # if no consumers: evict nothing. else m = min(consumer_steps.values());
        # pop refs with step < m - lookback_length; return popped refs
    def should_produce(self) -> bool:
        # producer_step is not None and final_step is None and producer_step < base + prefetch_length,
        # base = min(consumer_steps.values()) if consumer_steps else low_watermark
    def add(self, step: int, ref) -> None:          # assert step == producer_step; refs[step] = ref; producer_step += 1
    def mark_final(self) -> None:                   # final_step = producer_step
    def lookup(self, step: int) -> tuple[str, Any]:
        # ("ready", ref)      step in refs
        # ("final", None)     final_step is not None and step >= final_step
        # ("evicted", None)   producer_step is None or step < low_watermark
        # ("pending", None)   otherwise (step >= producer_step, not yet produced)
    def snapshot(self) -> dict:  # low_watermark, high_watermark, producer_step, final_step, num_refs, consumer_steps
```

## `producer.py` – `PrefetchActor`

Plain Python class (unit-testable without Ray), wrapped with `ray.remote` at creation time.
Threaded actor (`max_concurrency=8`); all `BatchWindow` access under one `threading.Lock`.

```python
CPU_DATALOADER_RESOURCE = "hpct_cpu_dataloader"

def prefetch_actor_name(job_name: str, split: str, dp_rank: int) -> str:
    return f"hpct-pdl-{job_name}-{split}-dp{dp_rank}"

class PrefetchActor:
    def __init__(
        self,
        dataloader_factory: Callable[..., Iterable],  # factory(*, dp_rank, dp_size, target_load_step, is_val)
        dp_rank: int,
        dp_size: int,
        lookback_length: int,
        prefetch_length: int,
        is_val: bool = False,
        enable_batch_encryption: bool = False,
        poll_interval_s: float = 0.01,
    ): ...
        # Does NOT build the dataloader; first register() does.

    def register(self, consumer_id: int, step: int, reseek_token: str | None = None) -> dict:
        # Under lock:
        #   reseek if: window.producer_step is None, or not window.can_resume_from(step),
        #              or (reseek_token is not None and reseek_token != self._last_reseek_token)
        #   reseek = window.reset(step); bump self._generation; build dataloader via factory at
        #            target_load_step=step; self._iter = iter(dl); self._length = len(dl) if possible;
        #            clear stored producer error; remember reseek_token; ensure producer thread running.
        #   window.register_consumer(consumer_id, step)
        # Returns {"mode": COLD_START if reseeked else WARM_START, "length": self._length, **window.snapshot()}

    def get_ref(self, step: int) -> tuple[str, list]:
        # status from window.lookup(step); refs wrapped in a list so Ray returns the ObjectRef itself
        # ("ready", [ref]) / ("pending", []) / ("final", []) / ("evicted", []) / ("error", [traceback_str])
        # "error" takes precedence over pending/final/evicted, but already-produced ready steps are still served.

    def ack(self, consumer_id: int, next_step: int) -> None   # window.ack(...); drop evicted refs (del)
    def length(self) -> int | None
    def stats(self) -> dict                                   # window.snapshot() + generation, error
    def shutdown(self) -> None                                # stop producer thread, drop refs

    # producer thread (daemon), loops until shutdown:
    #   with lock: if not window.should_produce(): wait poll_interval_s; continue
    #              gen = self._generation; it = self._iter; step = window.producer_step
    #   batch = next(it)  (OUTSIDE the lock; StopIteration -> under lock, if gen unchanged: window.mark_final())
    #   payload = encryption_manager.encrypt(batch) if enable_batch_encryption else batch
    #   ref = ray.put(payload)
    #   with lock: if gen == self._generation and window.producer_step == step: window.add(step, ref)
    #              else: drop ref (a reseek happened while we were fetching)
    #   any other exception: store (gen, traceback) so get_ref returns "error"; stop producing for that gen.

def get_or_create_prefetch_actor(
    *,
    job_name: str,
    split: str,                 # "train" | "val"
    dp_rank: int,
    dp_size: int,
    dataloader_factory: Callable[..., Iterable],
    lookback_length: int,
    prefetch_length: int,
    namespace: str,
    num_cpus: float = 1.0,
    resources: dict | None = None,   # default {CPU_DATALOADER_RESOURCE: 0.001}
    enable_batch_encryption: bool = False,
):
    # ray.remote(PrefetchActor).options(name=prefetch_actor_name(...), namespace=namespace,
    #     lifetime="detached", get_if_exists=True, max_concurrency=8, num_cpus=num_cpus,
    #     resources=resources, max_restarts=0).remote(...)
    # returns the ActorHandle
```

## `read_dataloader.py` – `RayReadDataLoader`

Runs on every GPU rank. Same public surface the trainer/abort path uses today:
`__iter__`, `__len__`, `set_length`, `stop()`.

```python
class RayReadDataLoader:
    def __init__(
        self,
        step: int,
        actor,                       # ActorHandle (or any object exposing .register/.get_ref/.ack/.length .remote())
        consumer_id: int,            # global rank
        prefetch_length: int,
        is_val: bool = False,
        enable_batch_encryption: bool = False,
        poll_interval_s: float = 0.05,
        rpc_timeout_s: float = 60.0,  # Timeout for every other ray.get.
        register_timeout_s: float = 900.0,  # ray.get timeout for register/length (first register builds the dataloader)
    ): ...
        # train (is_val=False): calls actor.register(consumer_id, step, None) here and stores the
        #   response as self.restart_info (mode/length/snapshot) – mirrors CacheResume* doing init in __init__.
        # val: registration happens in __iter__ with reseek_token=f"val-{self._num_iters}" and step 0.

    def set_length(self, length: int) -> None
    def __len__(self) -> int          # self._length if set, else restart_info["length"] (TypeError if unknown)
    def remote_length(self) -> int | None   # restart_info["length"] (train) or ray.get(actor.length.remote())
    def __iter__(self):
        # (val: register first). Start a fetch thread for steps self._step, self._step+1, ...:
        #   loop: status, payload = ray.get(actor.get_ref.remote(s), timeout=rpc_timeout_s)
        #     "pending" -> sleep poll_interval_s, retry (check stop event)
        #     "ready"   -> batch = ray.get(payload[0], timeout=rpc_timeout_s); decrypt if encryption;
        #                  queue.put((s, batch)) on a queue.Queue(maxsize=prefetch_length) (put with timeout,
        #                  re-checking stop event)
        #     "final"   -> put a sentinel; exit
        #     "evicted" -> put RuntimeError(f"step {s} evicted from window ...")
        #     "error"   -> put RuntimeError(payload[0])
        #   any exception (incl. RayActorError/GetTimeoutError) -> put it on the queue and exit
        # Main generator: pop items in order (assert step == self._step); exceptions are re-raised;
        #   sentinel -> return; yield batch; after the consumer asks for the next item (i.e. after yield
        #   resumes) self._step += 1 and fire-and-forget actor.ack.remote(consumer_id, self._step).
        # finally: self.stop()
    def stop(self) -> None            # set stop event, join fetch thread (timeout ~5s); idempotent; never raises
    @property
    def step(self) -> int
```

`stop()` is called from `HPDataLoaderManager.abort()` on the fault-handling thread while the main
thread may be blocked in `__next__`; it must return promptly (no unbounded `ray.get`).

## `config.py` – `RayCPUNodeConfig`

```python
@dataclass
class RayCPUNodeConfig(MMAPConfig):
    cpu_dataloader_factory: Callable = None   # required; picklable (cloudpickle) callable,
                                              # factory(*, dp_rank, dp_size, target_load_step, is_val) -> Iterable
    job_name: str = "default"
    ray_address: str = "auto"
    ray_namespace: str = "hyperpod_checkpointless"
    prefetch_length: int = 10
    val_prefetch_length: int = 10
    lookback_length: int = 2
    actor_num_cpus: float = 1.0
    actor_resources: dict | None = None       # None -> {CPU_DATALOADER_RESOURCE: 0.001}
    enable_batch_encryption: bool = False
    register_timeout_s: float = 900.0

    def create(self, dataloader_init_callable, parallel_state_util, step, is_data_loading_rank,
               create_model_parallel_group_callable, name="Train", is_val=False, cached_len=0):
        # same signature as CacheResumeMMAPConfig.create; dataloader_init_callable /
        # is_data_loading_rank / create_model_parallel_group_callable are unused (log at debug).
        # 1. validate step is not None and cpu_dataloader_factory is set (ValueError otherwise)
        # 2. import ray lazily; if not ray.is_initialized(): ray.init(address=ray_address,
        #    namespace=ray_namespace, ignore_reinit_error=True)
        # 3. step = 0 if is_val; if cached_len: cached_len, step = self._adjust_for_epoch_resume(cached_len, step)
        #    (move _adjust_for_epoch_resume from CacheResumeMMAPConfig to MMAPConfig so both share it)
        # 4. ps = parallel_state_util.parallel_state; dp_rank = ps.get_data_parallel_rank();
        #    dp_size = ps.get_data_parallel_world_size()
        # 5. actor = get_or_create_prefetch_actor(job_name, split="val" if is_val else "train", ...)
        # 6. dl = RayReadDataLoader(step, actor, consumer_id=dist.get_rank(),
        #        prefetch_length=val_prefetch_length if is_val else prefetch_length, is_val=is_val,
        #        register_timeout_s=register_timeout_s, ...)
        # 7. dl.set_length(max(cached_len, dl.remote_length() or 0)); log like the existing config; return dl
```

`MMAPDataModule` needs no change: it calls `mmap_config.create(...)` with the same signature.

## Examples / deployment

* `examples/dataloader/ray_cpu_dataloader_factory.py` – picklable factory class
  (`HFDatasetCPUDataLoaderFactory`) that builds the HF dataset + `DistributedSampler(rank=dp_rank,
  num_replicas=dp_size)` + `SkipDataLoader(_skip_batches=target_load_step, num_workers=N)` on the
  CPU node, mirroring `examples/dataloader/data_module.py::train_dataloader/val_dataloader`.
* `examples/llama3/launch/ray_cpu_dataloader_head.yaml` – CPU-instance pod running the Ray head with
  the `hpct_cpu_dataloader` resource + a Service exposing 6379/8265/10001.
* README section "Dedicated CPU-node dataloading with Ray".

## Tests

* `tests/dataloader/ray/test_window.py` – pure `BatchWindow` unit tests.
* `tests/dataloader/ray/test_producer.py` – `PrefetchActor` logic directly (monkeypatch `ray.put`)
  and through a local Ray (`ray.init(num_cpus=4, include_dashboard=False)` module-scoped fixture).
* `tests/dataloader/ray/test_read_dataloader.py` – consumer against a stub Ray actor implementing
  the `register/get_ref/ack/length` protocol.
* `tests/dataloader/test_config.py` – `RayCPUNodeConfig.create` with ray + actor helpers mocked.
* `tests/dataloader/ray/test_e2e_fault_resume.py` – local Ray end-to-end: start, consume, in-process
  "fault" with rollback inside lookback (WARM_START, identical batches), rollback outside the window
  (COLD_START reseek), killed actor (CPU node loss -> new actor, COLD_START), val reseek per iter.

Ray tests are skipped when ray is not installed (`pytest.importorskip("ray")`).
