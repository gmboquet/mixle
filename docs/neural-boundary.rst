The Neural Product Boundary
===========================

What mixle owns in the neural space, and where it hands off to dedicated trainers (worklist N9.1). The
short version: **mixle owns neural leaves, their composition into probabilistic models, small/medium fitting
convenience, calibration, and distillation. It does not own frontier-scale foundation-model training** —
that belongs to mature external trainers, and mixle's job there is to *wrap the result*, not to reproduce
the trainer.

What mixle owns
---------------

* **Neural leaves.** Any torch module that scores a batch to per-row log-densities is a distribution leaf
  via :class:`~mixle.models.grad_leaf.GradLeaf` (see :doc:`neural-llm`, :doc:`torch-modules`).
* **Composition.** A neural leaf composes with classical components in the same model — a mixture of a
  torch density and a Gaussian, a neural leaf inside a record — fit by the same ``optimize``
  (``byo_model_contract_test`` pins this).
* **Small/medium fitting convenience.** ``optimize(module)`` and ``LM.fit`` train modest models on a laptop
  or a single node, single- or (via the FSDP2/DDP path) small-multi-process. This is a convenience for
  getting a working model, not a scaling platform.
* **Calibration and distillation.** Conformal calibration, density/OOD gates, and teacher→student
  distillation over the task spine — the parts of the neural workflow that turn a model into a *decision*
  with bounded risk.

The supported neural module contract
------------------------------------

A module is usable as a mixle neural leaf when it:

* exposes ``log_density(batch) -> per-row log-densities`` (detected by
  :func:`~mixle.models.grad_leaf.looks_like_torch_module`), and
* has parameters trainable by gradient descent through the standard torch optimizer loop mixle drives.

Given that, ``GradLeaf(module).estimator()`` makes it fittable and composable like any estimator.

Expected scale for ``optimize(module)`` / ``LM.fit``
----------------------------------------------------

These are sized for **small-to-medium models on available local hardware** — thousands to low-millions of
parameters, corpora that stream from a single host, on CPU, one GPU, or a small multi-GPU node via the FSDP2
path. The mechanics of larger-scale training (tensor/pipeline/context parallelism, sharded checkpointing,
fault tolerance, muP transfer, scaling-law planning) exist in the tree as **prototypes**, explicitly labeled
as such (see the frontier-prototype warnings; worklist A1.4) — they are exact at small scale but are not a
production frontier trainer.

When to switch to an external trainer
-------------------------------------

Reach for a dedicated trainer (TorchTitan, Megatron, DeepSpeed, or a managed service) when the training run
needs any of: a model too large for one node's memory, real multi-node NCCL collectives at scale, a mature
data/checkpoint/elastic-restart stack, or throughput/MFU as a first-class goal. That is the frontier-scale
foundation-model training mixle deliberately does **not** own. The post-0.8 design note (worklist N9.7)
sketches how such a trainer would integrate.

Wrapping externally trained checkpoints
---------------------------------------

The handoff is one-directional and supported: an externally trained checkpoint comes **back** into mixle as
a leaf. Wrap it through the ``GradLeaf`` adapter (or a callable-teacher adapter for API models; see
:doc:`bring_your_own_model`), then compose, calibrate, and distill it exactly like a mixle-trained leaf.
mixle's neural value is the *composition, calibration, and distillation* around a model — not the
frontier-scale training of it.
