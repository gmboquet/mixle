The Neural Product Boundary
===========================

What mixle owns in the neural space, and where it hands off to dedicated trainers (worklist N9.1). The
short version: **mixle owns neural leaves, their composition into probabilistic models, fitting contracts,
calibration, and distillation**. For frontier transformer training it integrates with a mature external
engine rather than reimplementing its kernels: the Megatron adapter maps Mixle's typed plan into Bridge.

What mixle owns
---------------

* **Neural leaves.** Any torch module that scores a batch to per-row log-densities is a distribution leaf
  via :class:`~mixle.models.grad_leaf.GradLeaf` (see :doc:`neural-llm`, :doc:`torch-modules`).
* **Composition.** A neural leaf composes with classical components in the same model — a mixture of a
  torch density and a Gaussian, a neural leaf inside a record — fit by the same ``optimize``
  (``byo_model_contract_test`` pins this).
* **Native distributed fitting.** Packed ``LM.fit`` can execute DDP, FSDP2/HSDP, MLP TP, and CUDA CP through
  an explicit PyTorch DeviceMesh. The backend validates unsupported combinations before fitting.
* **Frontier-engine integration.** A Megatron Bridge provider can receive the same plan for full transformer
  and MoE TP/PP/CP/EP/ETP training while Megatron retains ownership of schedules and optimized kernels.
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

``optimize(module)`` remains a small/medium fitting convenience. ``LM.fit(dense=True, distributed=True)``
is the compact native trainer surface. At larger scale, use the Megatron adapter rather than treating the
compact LM wrapper as a replacement for Megatron's data loader, fused kernels, or topology tuning. See
:doc:`frontier-integration-note` for the exact capability matrix and checkpoint contract.

When to switch to an external trainer
-------------------------------------

Reach for the Megatron backend or another dedicated trainer when the run needs pipeline/expert parallelism,
fused transformer kernels, mature multi-node data ingestion, or throughput/MFU as a first-class goal.
Mixle supplies the model/update plan, receipts, and complete-state handoff; the selected engine executes it.

Wrapping externally trained checkpoints
---------------------------------------

The handoff is one-directional and supported: an externally trained checkpoint comes **back** into mixle as
a leaf. Wrap it through the ``GradLeaf`` adapter (or a callable-teacher adapter for API models; see
:doc:`bring_your_own_model`), then compose, calibrate, and distill it exactly like a mixle-trained leaf.
mixle's neural value is the *composition, calibration, and distillation* around a model — not the
frontier-scale training of it.
