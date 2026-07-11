Benchmark Methodology
=====================

The rules a mixle benchmark must follow so a timing measures what its prose claims (worklist B7.2). A speed
number that mixes phases, hides a warm-up, or reports alongside a parity *warning* is worse than no number:
it invites a wrong comparison. These rules make the harness's timings honest; the harness itself is tracked
in ``benchmarks/`` (worklist B7.1) and the absolute import/smoke budgets are gated in CI (worklist B7.4).

Separate the phases
-------------------

A "fit" number that silently includes encoding, or a "score" number that includes a one-time JIT compile, is
a category error. Measure and report each phase on its own:

* **initialization** -- building the estimator/model object and any prior/seed setup;
* **encode** -- ``dist_to_encoder().seq_encode(data)`` (the raw-to-vectorized conversion, paid once);
* **fit** -- the EM / gradient / closed-form loop over already-encoded data;
* **score** -- ``seq_log_density`` over encoded data;
* **compilation** -- the first-call cost of a compiled backend (Numba/JAX), which is amortized, not per-call.

A benchmark states which phase(s) its headline number covers, and never folds compilation into a per-call
figure.

Warm vs cold
-----------

Compiled kernels have a large first-call cost and a small steady-state cost. Report **warm** and **cold** as
separate panels, never averaged together:

* **cold** -- first call, includes compilation; the number a user pays once per process.
* **warm** -- steady state after the kernel is compiled; the number a user pays per batch thereafter.

Averaging one cold call into many warm calls produces a figure that describes neither.

Parity is a hard gate, not a footnote
--------------------------------------

Before a timing counts, the benchmarked fit must **pass** a likelihood/parameter parity check against a
reference (the serial path, or SciPy / scikit-learn / hmmlearn where applicable) -- as an assertion that
fails the benchmark, not a printed warning that scrolls by. A fast number for a wrong answer is not a
result. Speed is never bought by weakening correctness or parity (worklist B7.4).

Record what actually happened
-----------------------------

Every benchmarked fit records, alongside the timing:

* the **actual iterations** run and the **convergence status** (converged / hit ``max_its`` / diverged);
* any **failure** (a non-finite objective, a raised exception) -- reported, never silently dropped from the
  aggregate;
* the environment (see :doc:`reproduction`) so a number is attributable to a machine and version.

A timing without its iteration count and convergence status is unreproducible and does not belong in a
release claim.
