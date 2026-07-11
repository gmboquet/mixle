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
     - Autograd + neural leaves; GPU via device argument.
     - Optional (CI)
     - E1 — optional job (CPU).
   * - Torch (CUDA / GPU)
     - ``torch``
     - GPU arrays and training.
     - Hardware-gated
     - E3 (single run) — retained ad-hoc GPU run; not CI-gated.
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
     - E1 — ``mpi_executor_test`` exists; mpi4py not installed in CI.
   * - Spark
     - ``spark``
     - Map/fold over an RDD.
     - Tested, not CI-gated
     - E1 — backend test skips in CI (pyspark not installed).
   * - Dask
     - ``dask``
     - Map/fold over a Dask cluster.
     - Tested, not CI-gated
     - E1 — backend test skips in CI.
   * - Ray
     - ``ray``
     - Map/fold over a Ray cluster.
     - Tested, not CI-gated
     - E1 — backend test skips in CI.
   * - Lightning
     - ``lightning``
     - Mini-batch iteration driving stochastic/mini-batch EM.
     - Tested, not CI-gated
     - E1 — backend test skips in CI.

Reading this honestly
---------------------

"Tested, not CI-gated" is deliberate wording: the code and its tests exist, but because the backend is
not installed in any CI lane, a regression would not be caught automatically today. Installing at least
one such backend in a scheduled CI or hardware job — so its evidence rises to E3 — is tracked in the
0.8.0 worklist (workstream D). Until then, prefer the Supported and Optional-CI rows for anything you
depend on, and validate a "Tested, not CI-gated" backend in your own environment before relying on it.
Multi-node/multi-GPU *frontier-scale* training is out of scope for this release: mixle sits above the
trainer, not as a replacement for a dedicated large-scale training system.
