Univariate Families
===================

This page is the practical catalog for scalar distributions in ``mixle.stats``.
The generated API reference lists every class and method; this page explains
which family to choose, what support it assumes, and how scalar families
compose into larger Mixle models.

The naming pattern is consistent:

* ``FooDistribution`` is a fitted probability model.
* ``FooEstimator`` is the object passed to ``optimize`` to fit that family.
* ``FooEnumerator`` exists only for families with finite or countable support
  where exact support traversal is implemented.

Most scalar families are re-exported from ``mixle.stats``:

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import GammaEstimator, GaussianEstimator, PoissonEstimator

   duration = optimize(durations, GammaEstimator(), out=None)
   residual = optimize(residuals, GaussianEstimator(), out=None)
   counts = optimize(event_counts, PoissonEstimator(), out=None)

Treat scalar families as modeling assumptions, not defaults to apply blindly.
Support, tails, skew, discreteness, and impossible observations should be
checked before the fitted leaf becomes part of a larger record, mixture, or
HMM.

Continuous Families
-------------------

.. list-table::
   :header-rows: 1

   * - Family
     - Support
     - Use when
   * - ``GaussianDistribution`` / ``GaussianEstimator``
     - real line
     - symmetric residuals, measurement noise, simple continuous baselines.
   * - ``StudentTDistribution`` / ``StudentTEstimator``
     - real line
     - residuals have heavier tails than a Gaussian.
   * - ``SkewNormalDistribution`` / ``SkewNormalEstimator``
     - real line
     - residuals are continuous but asymmetric.
   * - ``LaplaceDistribution`` / ``LaplaceEstimator``
     - real line
     - sharper center and heavier tails than Gaussian.
   * - ``LogisticDistribution`` / ``LogisticEstimator``
     - real line
     - symmetric real-valued data with logistic tails.
   * - ``UniformDistribution`` / ``UniformEstimator``
     - bounded interval
     - only a finite range is known or a flat baseline is needed.
   * - ``BetaDistribution`` / ``BetaEstimator``
     - ``[0, 1]``
     - proportions, probabilities, or bounded rates.
   * - ``GammaDistribution`` / ``GammaEstimator``
     - positive real
     - durations, magnitudes, waiting times, positive skew.
   * - ``ExponentialDistribution`` / ``ExponentialEstimator``
     - non-negative real
     - memoryless waiting-time baseline.
   * - ``WeibullDistribution`` / ``WeibullEstimator``
     - non-negative real
     - failure times and hazards that rise or fall with age.
   * - ``LogGaussianDistribution`` / ``LogGaussianEstimator``
     - positive real
     - multiplicative noise or log-normal-like magnitudes.
   * - ``InverseGammaDistribution`` / ``InverseGammaEstimator``
     - positive real
     - variance-like positive quantities and Bayesian scale components.
   * - ``InverseGaussianDistribution`` / ``InverseGaussianEstimator``
     - positive real
     - first-passage-time-like positive data.
   * - ``HalfNormalDistribution`` / ``HalfNormalEstimator``
     - non-negative real
     - magnitudes of zero-centered Gaussian errors.
   * - ``RayleighDistribution`` / ``RayleighEstimator``
     - non-negative real
     - radial magnitudes from two Gaussian components.
   * - ``RicianDistribution`` / ``RicianEstimator``
     - non-negative real
     - magnitude with nonzero signal plus Gaussian noise.
   * - ``NakagamiDistribution`` / ``NakagamiEstimator``
     - non-negative real
     - flexible fading or positive magnitude data.
   * - ``ParetoDistribution`` / ``ParetoEstimator``
     - tail above a threshold
     - heavy-tailed size, wealth, severity, or frequency data.
   * - ``GeneralizedParetoDistribution`` / ``GeneralizedParetoEstimator``
     - threshold exceedances
     - peaks-over-threshold extreme-value modeling.
   * - ``GeneralizedExtremeValueDistribution`` / ``GeneralizedExtremeValueEstimator``
     - real line with shape-dependent tail
     - block maxima or minima.
   * - ``GumbelDistribution`` / ``GumbelEstimator``
     - real line
     - light-tailed extreme-value baseline.
   * - ``GeneralizedGaussianDistribution`` / ``GeneralizedGaussianEstimator``
     - real line
     - symmetric residuals with tunable tail shape.
   * - ``ExponentiallyModifiedGaussianDistribution`` / ``ExponentiallyModifiedGaussianEstimator``
     - real line with right skew
     - Gaussian noise plus exponential delay.
   * - ``TweedieDistribution`` / ``TweedieEstimator``
     - power-variance support
     - compound-like continuous/count mass patterns.

Discrete Families
-----------------

.. list-table::
   :header-rows: 1

   * - Family
     - Support
     - Use when
   * - ``CategoricalDistribution`` / ``CategoricalEstimator``
     - arbitrary labels
     - labels, tokens, classes, states, finite outcomes.
   * - ``IntegerCategoricalDistribution`` / ``IntegerCategoricalEstimator``
     - integer labels
     - dense integer-valued categories where integer encoders matter.
   * - ``BernoulliDistribution`` / ``BernoulliEstimator``
     - ``{0, 1}``
     - binary events.
   * - ``BinomialDistribution`` / ``BinomialEstimator``
     - bounded counts
     - successes out of a fixed number of trials.
   * - ``BetaBinomialDistribution`` / ``BetaBinomialEstimator``
     - bounded counts
     - over-dispersed binomial counts.
   * - ``PoissonDistribution`` / ``PoissonEstimator``
     - non-negative integers
     - count data with mean close to variance.
   * - ``NegativeBinomialDistribution`` / ``NegativeBinomialEstimator``
     - non-negative integers
     - over-dispersed count data.
   * - ``GeometricDistribution`` / ``GeometricEstimator``
     - positive or non-negative waiting count
     - trials until success.
   * - ``LogSeriesDistribution`` / ``LogSeriesEstimator``
     - positive integers
     - rare-species or heavily right-skewed counts.
   * - ``SkellamDistribution`` / ``SkellamEstimator``
     - all integers
     - difference of two Poisson counts.
   * - ``PointMassDistribution`` / ``PointMassEstimator``
     - one value
     - deterministic fields, constants, or degenerate baselines.
   * - ``IntegerUniformSpikeDistribution`` / ``IntegerUniformSpikeEstimator``
     - integer support with spike behavior
     - integer-valued data with a prominent preferred value.

Choosing Between Similar Families
---------------------------------

For real-valued residuals:

* start with ``GaussianEstimator``;
* use ``StudentTEstimator`` when a few large residuals should not dominate;
* use ``SkewNormalEstimator`` when positive and negative deviations behave
  differently;
* use ``LaplaceEstimator`` when absolute-error-like behavior is more natural
  than squared-error-like behavior.

For positive durations:

* start with ``GammaEstimator``;
* use ``ExponentialEstimator`` as a simple memoryless baseline;
* use ``WeibullEstimator`` when the hazard changes with age;
* use ``LogGaussianEstimator`` when multiplicative effects dominate.

For counts:

* start with ``PoissonEstimator``;
* use ``NegativeBinomialEstimator`` when variance exceeds the mean;
* use ``BetaBinomialEstimator`` for bounded over-dispersed counts;
* use ``SkellamEstimator`` when the observation is a difference of counts.

For extremes:

* use ``GeneralizedExtremeValueEstimator`` for block maxima;
* use ``GeneralizedParetoEstimator`` for threshold exceedances;
* use :doc:`analysis` for tail diagnostics before committing to a tail family.

For all scalar choices, keep a simple baseline and a held-out score when the
leaf affects a promoted model. A richer scalar family should earn its place
with validation evidence, not only a better in-sample fit.

Composition Example
-------------------

Scalar families are often leaves in a larger record model:

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import (
       CategoricalEstimator,
       CompositeEstimator,
       GammaEstimator,
       NegativeBinomialEstimator,
   )

   rows = [
       ("login", 0.8, 2),
       ("purchase", 4.2, 5),
       ("login", 1.1, 1),
   ]

   estimator = CompositeEstimator(
       (
           CategoricalEstimator(),       # event type
           GammaEstimator(),             # latency
           NegativeBinomialEstimator(),  # repeated attempts
       )
   )

   model = optimize(rows, estimator, out=None)

The same scalar leaf can appear inside a mixture, HMM emission, survival
wrapper, transform, record field, or process model.

When scalar leaves are nested, inspect field-level likelihoods before
interpreting the whole model. A single poorly matched scalar field can dominate
a composite score.

Enumeration
-----------

Finite and countable discrete families may expose enumerators. Use
:doc:`enumeration` for top-k and support traversal. Use :doc:`operations` when
a continuous family must be quantized into finite support for downstream
enumeration.

Release Evidence
----------------

For scalar-family work, preserve:

* support and unit assumptions;
* baseline comparison for similar families;
* held-out score or task metric when the leaf affects a promoted model;
* non-finite, out-of-support, and impossible-observation behavior;
* tail or dispersion diagnostics when using heavy-tail or count families; and
* field-level diagnostics when scalar leaves are nested in structured models.

API Reference
-------------

The generated scalar-family modules live under:

* :doc:`api/mixle.stats.univariate.continuous`;
* :doc:`api/mixle.stats.univariate.discrete`.
