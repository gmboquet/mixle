Distributed Transformer Training
================================

Mixle has two distributed-gradient backends. They share an explicit
``ParallelPlan`` and fail before fitting when a backend cannot execute a
requested axis. This is separate from the Spark, Dask, Ray, and MPI encoded-data
backends, which merge sufficient statistics rather than gradients.

Backend split
-------------

``torch_native``
    Keeps an arbitrary Mixle ``nn.Module`` in PyTorch. It supports replicated
    data parallelism (DDP), sharded or hybrid data parallelism (FSDP2/HSDP),
    MLP tensor parallelism through DTensor, and PyTorch's CUDA context-parallel
    SDPA path. TP+CP composition, pipeline parallelism, and expert parallelism
    are capability errors on this backend.

``megatron``
    Maps the plan to a Megatron Bridge model provider. Megatron owns full
    transformer/MoE TP, PP, CP, EP, ETP, distributed optimizer overlap, and its
    pretraining schedule. Mixle does not replace those kernels with local
    simulations.

``EP`` and ``ETP`` describe overlapping Megatron process groups. They do not
multiply the physical world size as independent DeviceMesh dimensions. EP must
divide the data-parallel domain.

Native packed training
----------------------

The compact ``LM`` surface uses packed all-position teacher forcing:

.. code-block:: python

   lm.fit(
       token_ids,
       dense=True,
       distributed=True,
       distributed_backend="torch_native",
       dp_shard=8,
       precision="bf16",
       microbatches=4,
       gradient_accumulation_steps=8,
   )

Rows are shuffled globally and deterministically, then partitioned over the
data-parallel domain. Model-parallel ranks see the same rows. Step receipts
distinguish local examples, global examples, microbatches, and accumulation.
The legacy streaming one-target objective accepts data parallelism only; it
refuses TP/PP/CP/EP/ETP rather than running a different plan.

Megatron Bridge
---------------

For a Bridge provider, resolve and configure the backend directly:

.. code-block:: python

   from mixle.utils.parallel import ParallelPlan, get_training_backend

   plan = ParallelPlan(
       dp_replicate=8,
       tp=8,
       pp=4,
       cp=2,
       ep=8,
       etp=2,
       microbatches=16,
   )
   backend = get_training_backend("megatron")
   backend.configure_provider(provider, plan)
   backend.run_pretrain(config, forward_step, plan=plan)

The dependency is optional and imported only when selected. A cluster image
must provide Megatron Bridge, CUDA, and NCCL.

Cluster launch
--------------

``RayTrainLauncher`` starts one worker per physical rank and passes the complete
plan to the worker configuration. ``LightningFabricLauncher`` is intentionally
limited to DDP/FSDP launches; model axes are capability errors. These launchers
manage processes only. The selected training backend still owns parameter
layout, collectives, optimizer state, and checkpoints.

Optimizer and checkpoint semantics
----------------------------------

Optimizer routes are planned from stable logical parameter names before
wrapping. Global matrix geometries such as Muon are not applied independently
to parameter shards: the native backend records a momentum fallback until it
has a declared global factor/gather implementation. Adam and AdamW remain
explicit options, not the automatic default.

``save_training_state`` stores DCP model and optimizer shards plus scheduler,
gradient scaler, Python/NumPy/Torch RNGs, loader position, optimizer step,
parallel geometry, typed-scheduler state, and caller metadata. A checkpoint is
loadable only after its ``_SUCCESS`` manifest is committed. Native asynchronous
saves use DCP's future and report write failures from ``wait()``.

Evidence boundary
-----------------

CPU tests establish objective parity, accumulation semantics, provider mapping,
checkpoint completeness, and capability failures. They do not establish GPU
throughput, MFU, overlap efficiency, or multi-node recovery. Those claims remain
unverified until retained hardware runs are attached to the GPU claims ledger.
