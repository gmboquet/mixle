Numerical Stability and Missing Data
====================================

This page is the stability contract for the core Mixle package. It covers
behavior that users should be able to rely on when fitting mixtures, PPL
models, latent models, and DOE distillation batches.

The short version is:

* impossible observations should score as ``-inf``, not ``NaN``;
* fitted parameters, responsibilities, selected DOE scores, and predictive
  summaries should be finite unless a page documents a special diagnostic
  sentinel;
* ``NaN`` in user data is not a value to repair in place;
* missing-data support must be explicit, either through marginalization or a
  missingness model;
* default fitting routes should reject non-finite observations instead of
  silently changing the data.

Non-Finite Values
-----------------

Mixle distinguishes three cases that are often blurred together:

``NaN`` in caller data
    A missing-data marker only on routes that opt in to missing-data handling.
    It must not be silently converted to a number in the caller's input.

``-inf`` log-density
    A legitimate probability result meaning "this observation is impossible
    under this model" or "no component can explain this row".

``NaN`` in outputs
    A defect for likelihoods, responsibilities, fitted parameters, DOE selected
    scores, and calibration thresholds unless that exact return value is
    documented as a diagnostic sentinel.

Use this rule when reviewing a new implementation: a route may reject
non-finite data, marginalize it, or model missingness. It should not mutate the
input buffer and pretend the observation was clean.

Caller Data Ownership
---------------------

Mixle may encode, copy, standardize, or mask data inside a fitting route. Those
working representations are implementation details. The caller-owned Python
objects, NumPy arrays, pandas frames, or source records remain the user's data
and must keep their original missing sentinels.

This distinction matters in release reviews:

* a copied feature matrix may replace non-finite coordinates with a column
  statistic so a distance calculation can run;
* a marginalized likelihood may ignore a missing field when accumulating
  sufficient statistics;
* a cross-modal selector may mark one modality as unavailable for a row; and
* none of those routes should write replacement values back into the caller's
  input object.

When validating a route that accepts mutable arrays, keep a copy of the input
before the call and compare it after the call. If the route needs to return
cleaned data, that data should be a separate artifact with its own provenance,
not a silent in-place repair.

Outcome Matrix
--------------

Use this matrix when deciding whether a result is acceptable:

.. list-table::
   :header-rows: 1

   * - Situation
     - Acceptable behavior
     - Unacceptable behavior
   * - Observation outside support
     - ``-inf`` log-density or a documented validation error.
     - ``NaN`` score, clipped score, or a fabricated finite likelihood.
   * - ``NaN`` in caller data on a strict route
     - Clear error before fitting or scoring.
     - Treating the value as zero, mean-filled, or ordinary numeric input.
   * - ``NaN`` in caller data on an explicit missing-data route
     - Marginalize, mask, or model missingness without mutating the input.
     - In-place imputation without provenance.
   * - All mixture components impossible for a row
     - ``-inf`` marginal score and finite responsibility fallback.
     - A row of ``NaN`` responsibilities.
   * - Non-finite ranking vector in DOE
     - ``ValueError`` because the objective is undefined.
     - Dropping, filling, or reweighting the row silently.

Missing-Data APIs
-----------------

The stats layer has first-class missing-data helpers:

.. code-block:: python

   from mixle.stats.missing import MISSING, composite_with_missing, marginalized

   maybe_x = marginalized(base_distribution)
   record = composite_with_missing([temperature_dist, pressure_dist])

``marginalized(dist)`` wraps a distribution so a missing field contributes
log-density ``0`` and no sufficient statistics. Estimation then uses the
present rows only. This is missing-at-random handling, not imputation.

Use ``OptionalDistribution`` or ``OptionalEstimator`` when missingness itself is
part of the phenomenon and the probability of absence should be modeled. Use
``marginalized`` when absence is a nuisance and should be integrated out.

PPL Missing Data
----------------

``mixle.ppl`` keeps missing-data behavior explicit. The default is strict:

.. code-block:: python

   from mixle.ppl import Normal, free

   Normal(free, free).fit([1.0, float("nan"), 2.0])
   # raises by default

Pass ``missing="marginalize"`` when ``NaN`` entries should be integrated out
of the likelihood:

.. code-block:: python

   fit = Normal(free, free).fit(
       [1.0, float("nan"), 2.0, 3.0],
       missing="marginalize",
   )

For EM/MLE this wraps the estimator leaves with marginalizing optional leaves,
so the fit matches the MLE over present observations. For MAP, VI, MCMC, HMC,
and NUTS, marginalization is supported on the flat autograd target when the
required backend is available. Closed-form routes that are not wired for
missing data raise instead of guessing.

If you need posterior imputation, do it after fitting through conditionals,
latent posteriors, or posterior predictive queries. Do not replace ``NaN`` in
the training data before the model sees it unless that imputation step is an
explicit part of your data pipeline and provenance.

Mixture Stability
-----------------

Finite mixtures are the highest-risk core path because EM can collapse a
component, underflow responsibilities, or hit singular covariance estimates.
The current implementation documents and tests the following behavior:

* scalar and vectorized mixture scoring use log-sum-exp over component
  log-densities and log-weights;
* vectorized ``seq_log_density`` returns ``-inf`` for rows where no active
  component has a finite score;
* ``seq_posterior`` falls back to the prior mixture weights for all-impossible
  rows, avoiding a row of ``NaN`` responsibilities;
* ``MixtureEstimator(..., robust=True)`` enables k-means++ initialization and a
  small positive weight floor;
* the weight floor clamps plain-MLE weights away from exact zero and
  renormalizes them;
* numeric k-means++ initialization falls back to the Dirichlet path for ragged,
  object, nonnumeric, or non-finite encoded feature matrices;
* high-dimensional Gaussian mixture fits rely on variance/covariance floors in
  the Gaussian estimators, so rank-deficient component assignments do not
  produce singular fitted covariance matrices.

For difficult Gaussian mixtures, prefer a robust estimator and a monotonic EM
driver:

.. code-block:: python

   import numpy as np

   from mixle.inference import optimize
   from mixle.inference.em import MonotonicEM
   from mixle.stats import (
       DiagonalGaussianDistribution,
       DiagonalGaussianEstimator,
       MixtureDistribution,
       MixtureEstimator,
   )

   k = 4
   dim = 64
   init = MixtureDistribution(
       [DiagonalGaussianDistribution(np.zeros(dim), np.ones(dim)) for _ in range(k)],
       np.ones(k) / k,
   )
   estimator = MixtureEstimator(
       [DiagonalGaussianEstimator(dim=dim) for _ in range(k)],
       robust=True,
   )

   model = optimize(
       data,
       estimator,
       prev_estimate=init,
       strategy=MonotonicEM(),
       out=None,
   )

``MonotonicEM`` keeps the last good model when a proposed EM step produces a
non-finite objective. That is a guardrail, not a substitute for validating the
fitted model. Still inspect held-out score, responsibilities, and component
parameters.

DOE Distillation Stability
--------------------------

The DOE distillation selectors are designed for expensive label acquisition,
not for repairing invalid feature matrices.

``distillation_design`` and ``multitask_distillation_design`` require finite
``uncertainty``, ``preference``, ``cost``, and coverage weights. Non-finite
entries in those control vectors raise ``ValueError`` because they would change
the ranking objective.

Feature matrices are copied into standardized working arrays for distance and
diversity calculations. Non-finite feature coordinates are filled in that
working representation so candidate scoring can proceed, but the caller's
source data is not the place where missingness is repaired.

``cross_modal_distillation_design`` treats a row with non-finite coordinates in
one modality as missing for that modality. Eligibility is then based on
available modality count, ``min_modalities``, and ``required_modalities``. The
returned selected scores should be finite; rows without enough modalities are
not silently converted into paired examples.

PPL Numerical Surface
---------------------

Use :doc:`ppl` for the modeling syntax. The stability expectations are:

* ``how="auto"`` should explain and choose a concrete route through
  ``explain_fit``;
* constraints and custom potentials route to numerical posterior objectives
  that can honor them;
* impossible likelihood evaluations inside MAP/MCMC/HMC/NUTS are mapped to a
  large negative finite sentinel for optimizer and sampler safety;
* Cholesky and simplex parameter slots should be rebuilt through structured
  parameter handles rather than unconstrained one-off arrays;
* state-space PPL families expose a real fitted distribution for log-probability
  and simulation after fitting;
* unsupported combinations should raise clear errors instead of returning a
  half-bound object.

Validation Checklist
--------------------

Every new model, estimator, latent model, or capability that touches numeric
data should have focused checks for:

1. scalar/vectorized scoring parity where both paths exist;
2. finite fitted parameters on representative and edge-case data;
3. no ``NaN`` responsibilities, posterior summaries, or selected design
   scores;
4. impossible rows returning ``-inf`` or a documented error, not ``NaN``;
5. explicit behavior for ``NaN`` inputs: reject, marginalize, or model
   missingness;
6. preservation of caller-owned missing sentinels when the route uses a copied
   or encoded working representation;
7. deterministic output when a seed is supplied;
8. a strict Sphinx page or API reference entry documenting any non-default
   missing-data behavior.

Existing regression coverage exercises missing-data preservation, mixture
stability, DOE distillation, and route-specific PPL behavior. Treat those as
minimum coverage, not proof that a new route is safe.
