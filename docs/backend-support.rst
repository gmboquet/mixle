Backend Support Matrix
======================

Mixle separates two orthogonal axes: the **compute engine** (which array library runs the likelihood
and sufficient-statistic math, and on what device) and the **distributed backend** (how sufficient
statistics are computed across workers). Both are selected by an argument to ``optimize`` — but they
are not all equally validated. This page states, per backend, what it does, how it is exercised, and
the evidence grade behind that (E0–E5, the release contract's evidence grades, from assertion up to
sustained production/scale), so a claim about "which backends work" is grounded rather than universal.

Support levels:

- **Supported** — exercised in CI on every run (or a scheduled run) and expected to work on the base
  path.
- **Optional (CI)** — exercised in the scheduled/optional CI job with the extra installed.
- **Tested, not CI-gated** — has tests in the suite, but the backend is not installed in any CI lane,
  so those tests *skip* in CI; correctness rests on local/ad-hoc runs.
- **Hardware-gated** — needs accelerators not present in CI; validated by a retained ad-hoc run.

Compute engines
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 14 26 22 20

   * - Engine
     - Extra
     - Role
     - Support level
     - Evidence
   * - NumPy
     - (base)
     - Default CPU engine; every distribution fits here.
     - Supported
     - E2 — every PR, incl. clean-wheel import.
   * - Numba
     - ``numba``
     - JIT-compiled hot paths; falls back to NumPy when absent.
     - Optional (CI)
     - E1 — scheduled/optional job.
   * - Torch (CPU)
     - ``torch``
     - Autograd + neural leaves; GPU via device argument (see the note below on
       device-dependent numerics).
     - Optional (CI)
     - E1 — optional job (CPU).
   * - Torch (CUDA / GPU)
     - ``torch``
     - GPU arrays and training.
     - Hardware-gated
     - Executed once for 0.8.0 on a rented RTX 3060 (receipt:
       ``release-checklists/0.8.0-cuda-receipt.json``); not CI-gated, so regressions between
       receipts go undetected. Historical runs are not candidate evidence.
   * - JAX
     - ``jax``
     - XLA arrays + the NumPyro NUTS backend.
     - Optional (CI)
     - E1 — optional job.

Distributed backends
--------------------

Selected with ``optimize(..., backend=...)``. Each computes and reduces sufficient statistics across
workers under the same estimation contract.

.. list-table::
   :header-rows: 1
   :widths: 16 14 30 22 18

   * - Backend
     - Extra
     - Role
     - Support level
     - Evidence
   * - multiprocessing
     - (base)
     - Local multi-process sufficient-statistic map/fold.
     - Supported
     - E1.
   * - torchrun (DDP)
     - ``torch``
     - SPMD data-parallel neural training; in-backward all-reduce.
     - Optional (CI)
     - E1 — gated two-rank gloo smoke in the optional job.
   * - MPI
     - ``mpi``
     - Tree-fold reduction of sufficient statistics across ranks.
     - Tested, not CI-gated
     - E1 — ``parallel_test.MPIBackendTestCase`` exists; mpi4py not installed in CI. Retained local execution evidence: the backend-execution-evidence appendix of ``release-checklists/0.8.0.md``.
   * - Spark
     - ``spark``
     - Map/fold over an RDD.
     - Tested, not CI-gated
     - E1 — backend test skips in CI (pyspark not installed). Retained local execution evidence: the backend-execution-evidence appendix of ``release-checklists/0.8.0.md``.
   * - Dask
     - ``dask``
     - Map/fold over a Dask cluster.
     - Optional (CI)
     - E1 — installed and exercised in the scheduled optional job.
   * - Ray
     - ``ray``
     - Map/fold over a Ray cluster.
     - Tested, not CI-gated
     - E1 — backend test skips in CI. Retained local execution evidence: the backend-execution-evidence appendix of ``release-checklists/0.8.0.md``. Retained local execution evidence: the backend-execution-evidence appendix of ``release-checklists/0.8.0.md``.
   * - Lightning
     - ``lightning``
     - Mini-batch iteration driving stochastic/mini-batch EM.
     - Tested, not CI-gated
     - E1 — backend test skips in CI.

Reading this honestly
---------------------

"Tested, not CI-gated" is deliberate wording: the code and its tests exist, but because the backend is
not installed in any CI lane, a regression would not be caught automatically today. Dask is installed in
a scheduled CI lane; the other rows with this label still skip there.

Every one of these backends HAS now been executed against the release candidate, and the commands,
versions and results are retained in the backend-execution-evidence appendix of ``release-checklists/0.8.0.md`` --
including a two-rank ``mpiexec`` fit agreeing with the serial fit to 1e-10. That evidence is
deliberately not treated as promotion to "Supported", for three reasons stated in the file itself: it
is single-machine (macOS arm64, multi-process on one host -- no multi-node, no network transport, no
GPU), it does not cover interruption/recovery, corruption rejection or rollback, and it was produced
by the implementer rather than an independent reviewer. A backend is CI-gated or it is not, and none
of these are.

So the guidance is unchanged: prefer the Supported and Optional-CI rows for anything you depend on,
and validate a "Tested, not CI-gated" backend in your own environment before relying on it. What the
evidence file changes is that you can now see exactly what was run, and repeat it, instead of taking
"tested" on trust.
Multi-node/multi-GPU *frontier-scale* training is out of scope for this release: mixle sits above the
trainer, not as a replacement for a dedicated large-scale training system.

``device=`` is a numerical choice, not only a performance one
-------------------------------------------------------------

A fit on a GPU device does **not** reproduce the same fit on CPU. Measured on Apple M4 / Metal
(MPS) against the candidate, same seed and architecture, the per-observation log-density diverges by
about **2% relative after a single optimizer iteration**, rising to roughly 4% by ten. The gap is
present from the first step rather than accumulating from a tiny one, so it is a float32
kernel difference between the device backends, not drift.

Practically: a log-density, fitted parameter, or model digest recorded on CPU will not match the
same computation on GPU, and a :mod:`mixle.inference.reproduce` receipt will not verify across
devices. That is honest behaviour -- the two fits really are different -- but it is a trap if
``device=`` is read as a speed knob. Pin the device alongside the seed when a result has to
reproduce.

Measurements and commands: the backend-execution-evidence appendix of ``release-checklists/0.8.0.md``. CUDA was
executed once for 0.8.0 on a rented RTX 3060 — the receipt (GPU, driver, torch/CUDA versions, float32 CPU-vs-CUDA
agreement on a GMM fit, an HMM float32 fit, quantized parity) lives at ``release-checklists/0.8.0-cuda-receipt.json``.
That run also exposed two CUDA-only dtype-promotion defects in ``TorchEngine`` (fixed at the same commit): an
existence proof of execution, not a CI gate. Metal is a different backend and says nothing about CUDA.
