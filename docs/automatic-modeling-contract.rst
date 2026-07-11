The Automatic-Modeling Contract
===============================

An explicit statement of what ``optimize(data)`` / ``get_estimator(data)`` **may** infer and what it **may
not** (worklist I6.1). The automatic path is a convenience for getting a defensible first model from raw
data; this page pins its promises so a caller knows exactly what it will and will not do.

Supported input containers
--------------------------

* A **sequence of observations** — a ``list`` (or any re-iterable sequence) of items. Each item may be a
  scalar, a fixed-length ``tuple`` (a *record*), or a numeric vector.
* **Re-iterability is required.** The detector reads the data more than once (to profile field types, then to
  fit and score). A one-shot iterator/generator is **materialized to a list first** rather than being
  silently half-consumed, so a generator input yields a determinate model, not a corrupted one.
* Pre-encoded ``enc_data`` is **not** a valid input for inference — automatic modeling needs the raw values
  to read their types.

Per-field schema inference
--------------------------

For each field (a scalar sequence is a one-field record), the detector picks a family from the values it
sees. Measured behavior:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Field values
     - Inferred family
   * - floats
     - ``Gaussian`` (positive/​skewed floats may resolve to ``Gamma`` / ``LogGaussian`` by shape)
   * - non-negative integer counts
     - ``Poisson``
   * - booleans
     - ``Categorical``
   * - low-cardinality strings/labels
     - ``Categorical``
   * - **high-cardinality identifiers** (≈ unique per row)
     - ``Ignored`` — not modeled (see below)
   * - numeric vectors (fixed length)
     - ``MultivariateGaussian``
   * - fixed-length record (tuple)
     - ``Composite`` of the per-field families
   * - sequences / sets / graphs
     - the corresponding structured family, when the values match its shape

**High-cardinality identifiers are ignored on purpose.** A field whose values are (almost) all distinct
carries no distributional signal a density model can use; modeling it would overfit one parameter per row.
The detector routes such a field to an ``Ignored`` leaf rather than inventing a distribution for it.

Missing values
--------------

Missing entries are handled per the family's missing-data contract (see :doc:`stability-and-missing-data`);
caller-owned data is not rewritten in place to hide missing or non-finite values.

Dependence between fields
-------------------------

The default record model is **independent** per field (a ``Composite``). ``optimize(data)`` will upgrade to a
dependence model — a learned Bayesian network or a copula — **only when it beats the independent baseline by
BIC on the same data**; otherwise it keeps independence. It does not assume dependencies it cannot evidence.

Selection and validation
------------------------

* ``get_estimator(data)`` returns a single inferred estimator (structure only; no held-out split).
* ``propose(data)`` builds a **verified frontier**: it fits each candidate on a train split and scores it on
  a held-out split (``holdout=0.25`` by default), ranking by held-out mean log-density — the ranking is
  out-of-sample, not a guess. The frontier search is bounded by ``max_candidates`` / ``timeout`` (worklist
  I6.5).

What it will *not* do
---------------------

* It will not claim universal correct family recovery. Genuinely nested/overlapping families (Exponential
  vs Gamma, LogNormal vs Inverse-Gaussian) are documented ambiguity regions — see the measured confusion
  matrix in ``model_selection_benchmark_test`` (worklist I6.4).
* It will not model a high-cardinality identifier, fabricate a family without evidence, assume cross-field
  dependence it cannot justify by BIC, or silently consume a one-shot iterator.
