Claim-Evidence Ledger
=====================

Every material public claim about mixle, with the strongest evidence that currently supports it. The
point of a credibility release is that claims are graded, not asserted: a reader can see exactly how
far each statement has been verified and decide whether that is enough for their use.

Evidence grades (from the 0.8.0 release contract):

- **E0 — assertion:** prose, design intent, or a unit test with a stand-in only.
- **E1 — local correctness:** a deterministic unit/integration test on one machine.
- **E2 — artifact correctness:** a test from a clean installation of the built wheel.
- **E3 — realistic workload:** real public data, or a real external model/backend, at bounded scale.
- **E4 — independent reproduction:** another environment or person reproduces it from documented steps.
- **E5 — production/scale:** sustained real deployment, or representative multi-node/multi-GPU runs.

Stable-core claims target at least **E2**; performance and backend claims target at least **E3**.
"Production-ready", "safe", or "frontier" claims require **E5** and are **not** made in 0.8.0. Grades
here are the release owner's current assessment and are revised as evidence lands (several are pending
the 0.8.0 re-run, marked below).

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Claim
     - Grade
     - Evidence / caveat
   * - ~90 distribution families with a common estimator/sampler/encoder contract
     - E2
     - Family, sampler-seed, and scipy-golden density tests across the catalog; imported from the
       clean wheel by the ``clean-wheel`` CI job.
   * - "5000+ tests"
     - E1
     - The full suite runs in CI (``fast`` on 3.11–3.12, ``full`` on 3.12). Not yet run from the wheel
       for the whole suite (only the import sweep is) — that is a 0.8.0 exit criterion.
   * - One ``optimize(...)`` call fits a composed model (distribution + neural + latent) together
     - E2
     - Composite/mixture/HMM estimation tests plus the nested-estimator examples; deterministic-by-default.
   * - A PyTorch module fits in one line with parity to a hand-written loop
     - E1
     - ``torch_parity_test`` / ``grad_control_test`` check parity, freezing, and checkpointing (torch extra).
   * - Distillation into a small local model with calibrated deferral and a cost/quality receipt
     - E1 (E3 pending)
     - Task/calibration/cascade tests are E1. A real Banking77 teacher/student workflow exists, but its
       exact-candidate hosted run must complete before it counts as 0.8.0 E3 evidence.
   * - HMM / GMM performance and crossover behavior
     - No 0.8.0 claim
     - The harness is tracked in ``benchmarks/``. Older measurements are archived and are not release
       evidence; numbers may be published only after an exact-candidate retained run.
   * - Distributed estimation over Spark / Dask / Ray / MPI by switching one argument
     - E1
     - Per-backend encoded-data tests exist. Dask and torchrun are exercised in the scheduled optional lane;
       Spark, Ray, MPI, and Lightning remain tested locally but not CI-gated. Wording is qualified in the
       backend support matrix.
   * - Distributed transformer and MoE training
     - E1
     - The native backend executes DDP/FSDP2/HSDP and selected TP/CP paths; Megatron Bridge owns full
       TP/PP/CP/EP/ETP. CPU objective/checkpoint and provider-delegation tests establish local correctness,
       not a realistic-scale run. GPU throughput, MFU, and multi-node recovery remain unverified until
       retained hardware receipts exist. Unsupported axes fail capability validation.
   * - Serialization round-trips and provenance / replay
     - E1
     - Fresh-process serialization tests; lineage/receipt tamper-and-replay tests. Cross-version
       (load a prior release's artifact) fixtures are a 0.8.0 gate in progress (worklist M11.2).
   * - Calibrated abstention (conformal / cascade / escalation)
     - E1
     - Conformal-coverage and calibration-diagnostic tests. This is a deferral mechanism, not a safety
       guarantee — no "safe to deploy" claim is made (needs E5).

This ledger is maintained alongside the release checklist; when a claim's evidence changes, update the
grade in the same change.
