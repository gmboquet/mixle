Post-0.8 Design Note: Real Frontier Integration
===============================================

A design note, **not** an implementation (worklist N9.7). It sketches how mixle would integrate a real
frontier-scale trainer after 0.8.0, so the boundary drawn in :doc:`neural-boundary` has a concrete
follow-through. Nothing here ships in 0.8.0; the frontier-training surfaces that exist today are labeled
prototypes (worklist A1.4).

Integration choice
------------------

Prefer **TorchTitan** as the primary integration target, with DeepSpeed as a secondary path:

* **TorchTitan** is torch-native and FSDP2-first, which matches mixle's existing single-node FSDP2 path and
  the ``CausalLM`` module structure — the smallest conceptual jump, and the checkpoints are plain
  ``torch.distributed.checkpoint`` (DCP), which mixle already round-trips at small scale
  (``mixle.utils.parallel.dcp_checkpoint``).
* **DeepSpeed** (ZeRO-3, pipeline) as a secondary path for users already on that stack; its checkpoints need
  a conversion step to the same DCP/state-dict shape.
* **Megatron-LM** is powerful but its tensor/pipeline-parallel checkpoint format and NVIDIA-specific
  assumptions make it the highest-effort adapter; treat it as a later, optional target.

The choice is deliberately *one primary* so the checkpoint-adapter contract below has a single reference
format to target first.

Ownership split
--------------

* **The external trainer owns** the frontier-scale training loop: data pipeline at scale, N-D parallelism
  (TP/PP/CP), sharded optimizer state, elastic/fault-tolerant restart, throughput/MFU, and the training-time
  checkpoint contract (optimizer + scheduler + step + RNG + loader position).
* **mixle owns** everything *around* the trained model: wrapping the resulting inference artifact as a leaf,
  composing it with classical components, calibration, OOD gating, distillation, and the provenance/evidence
  trail. This is the same split 0.8.0 already draws — mixle's neural value is the workflow around a model,
  not its frontier-scale training.

Checkpoint adapter contract
--------------------------

The handoff is a **checkpoint → leaf** adapter, one-directional:

1. The trainer emits an *inference artifact* — architecture config (a registered builder name + its kwargs)
   plus consolidated weights (a DCP/state-dict save), distinct from its *training checkpoint* (worklist
   M11.3 already draws this line for ``LM``).
2. mixle reconstructs the module via the artifact builder registry (``get_builder(name)(**config)``) and
   loads the consolidated weights, then wraps it as a :class:`~mixle.models.grad_leaf.GradLeaf` — the same
   adapter the bring-your-own-model contract (worklist N9.6) already exercises for smaller checkpoints.
3. The adapter validates the parameter shapes against the config on load (fail-fast on a mismatch), so a
   wrong-config or truncated checkpoint is a clear error, not a silent corruption.

Evaluation and provenance hooks
-------------------------------

* **Provenance in.** The trainer records its config, data hash, step count, and framework/version into the
  artifact's ``meta`` block; mixle's provenance header carries that through so a deployed model's lineage
  names the trainer, run, and data it came from.
* **Evaluation.** The wrapped leaf is evaluated through mixle's existing task scorecard (held-out score,
  calibration, escalation) — the frontier model is scored by the *same* honest receipts as any other leaf,
  so a big external checkpoint earns no exemption from the evidence bar.
* **Tamper/replay.** The provenance/replay tests that already guard mixle artifacts apply unchanged, so a
  frontier-trained artifact's lineage is verifiable after the fact.

Out of scope for this note
--------------------------

Choosing exact library versions, writing the DeepSpeed/Megatron conversion shims, and any performance target
are deferred to the implementing PR. This note fixes the *shape* — one primary trainer, a clean
checkpoint→leaf adapter, and the ownership split — so that work has a target.
