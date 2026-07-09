Analysis Utilities
==================

``mixle.analysis`` contains applied statistical routines that are not
probability-distribution families themselves. They operate on data, diagnostics,
or fitted summaries and complement the core modeling layer.

The namespace covers:

* extreme-value analysis;
* kernel density estimation;
* species and coverage estimation;
* variograms and kriging;
* rank aggregation;
* spatial mixtures and max-stable processes;
* covariance shrinkage.

Use these tools when you need to understand a dataset, build diagnostics around
a fitted model, or create an analysis component that feeds a larger Mixle
workflow.

Extreme Values
--------------

The extreme-value helpers support peaks-over-threshold analysis, tail index
estimation, return levels, record statistics, and finite-endpoint estimates.

Public functions include:

* ``peaks_over_threshold`` and ``gpd_fit``;
* ``GPDFit``;
* ``return_level``;
* ``hill_estimator`` and ``moment_estimator``;
* ``mean_residual_life``;
* ``endpoint_estimator``;
* ``record_times`` and ``n_records``.

.. code-block:: python

   from mixle.analysis import peaks_over_threshold, return_level

   fit = peaks_over_threshold(losses, threshold=1_000.0)
   hundred_event_level = return_level(fit, period=100)

Use these when the tail behavior is operationally important: loss events,
latency spikes, claims, safety margins, queue overload, or anomaly severity.

Choose thresholds and periods from the application, not from whichever value
produces the most dramatic plot. For release evidence, record the threshold,
sample window, fitted tail parameters, and the sensitivity of the result to a
nearby threshold.

Kernel Density Estimation
-------------------------

``KDE`` and ``kde`` provide one-dimensional kernel density estimation with
bandwidth helpers:

* ``silverman_bandwidth``;
* ``scott_bandwidth``;
* ``kde_mode``;
* ``intensity``.

.. code-block:: python

   from mixle.analysis import kde, kde_mode

   density = kde(samples, bandwidth="silverman")
   mode = kde_mode(samples)

KDE is useful for exploratory analysis, visualization, mode finding, and
building nonparametric baselines before committing to a parametric family.

Bandwidth choice is part of the analysis result. If a KDE is used to justify a
modeling decision, record the selected bandwidth rule or numeric value and
compare it with at least one simpler diagnostic such as a histogram, empirical
quantile table, or held-out score under the candidate parametric model.

Coverage and Diversity
----------------------

Coverage estimators help quantify how much unseen mass remains in discrete
samples.

Public functions include:

* ``turing_coverage`` and ``good_turing``;
* ``chao1`` and ``chao2``;
* ``ace`` and ``ice``;
* ``hill_numbers``;
* ``rarefaction_curve``.

These are useful for species counts, vocabulary coverage, unique error
patterns, rare event types, ontology categories, or any setting where observed
categories are only a sample from a larger support.

.. code-block:: python

   from mixle.analysis import chao1, hill_numbers

   richness = chao1(category_counts)
   diversity = hill_numbers(category_counts, q=[0.0, 1.0, 2.0])

Kriging and Variograms
----------------------

Geostatistical helpers include:

* ``empirical_variogram``;
* ``fit_variogram``;
* ``Variogram``;
* ``ordinary_kriging``;
* ``universal_kriging``;
* ``calibrate_variance``.

.. code-block:: python

   from mixle.analysis import empirical_variogram, fit_variogram, ordinary_kriging

   empirical = empirical_variogram(coords, values)
   variogram = fit_variogram(empirical["distance"], empirical["semivariance"])
   pred, var = ordinary_kriging(coords, values, query_coords, variogram)

Use kriging for spatial interpolation and calibrated uncertainty over
locations. The results can feed downstream distributions, decision objectives,
or design-of-experiments loops.

Spatial diagnostics are sensitive to coordinate systems and duplicated
locations. Normalize coordinate units before fitting variograms, keep the
coordinate reference system with artifacts, and check whether the fitted
variance calibration is credible on held-out locations.

Rank Aggregation
----------------

Rank aggregation tools combine multiple orderings into a consensus:

* ``borda_count``;
* ``copeland``;
* ``kemeny_consensus``;
* ``mallows_fit``;
* ``kendall_distance``;
* ``spearman_footrule``;
* ``cayley_distance``.

Use these for model ranking, human preference aggregation, evaluation
leaderboards, or distillation datasets where several judges provide partial
orders.

Rank aggregation should preserve judge provenance. A consensus ranking is not
evidence that the judges were interchangeable; ties, missing items, and
systematic judge disagreement should be reported when they affect a downstream
promotion decision.

Spatial Mixtures and Max-Stable Models
--------------------------------------

``SpatialMixture`` models spatially structured mixture assignments.
``SmithMaxStable`` and ``fit_smith_maxstable`` support max-stable spatial
extreme-value modeling.
``SmithMaxStableSampler`` is the sampler returned by a fitted Smith process.

Use these when nearby locations should share structure or when spatial extremes
are more important than average behavior.

For max-stable or spatial-mixture work, keep simulated checks separate from
observed-data evidence. A fitted process should reproduce the operational tail
or dependence diagnostic that motivated the model, not only an aggregate loss.

Covariance Shrinkage
--------------------

``LedoitWolfEstimator`` provides covariance shrinkage as a Mixle estimator. It
is useful when covariance matrices are high-dimensional, noisy, or estimated
from limited samples.

This can be used as a preprocessing diagnostic, a fitted covariance component,
or a stabilized input to downstream Gaussian models.

When shrinkage affects a downstream model, record the empirical covariance
diagnostic and the shrinkage estimator output together. The downstream model
should not be the only place where covariance stabilization is visible.

How Analysis Fits With Modeling
-------------------------------

Analysis routines are often upstream or downstream of a model:

* upstream, they reveal tail behavior, dependence, coverage gaps, or spatial
  structure before model design;
* downstream, they validate residuals, calibration, drift, and rare-event
  behavior after fitting;
* alongside inference, they supply objectives and diagnostics for
  anti-regression gates.

They are intentionally separated from ``mixle.stats``. A KDE diagnostic or
rank aggregation routine may be essential to an application, but it is not the
same thing as a distribution family with an estimator and sampler.

Release Evidence
----------------

Analysis outputs often become evidence for model design, promotion, or
monitoring. For those uses, keep enough context for a reviewer to reproduce the
diagnostic:

* the input data window, filtering policy, and any grouping keys;
* the parameter choices such as thresholds, bandwidths, variogram families, or
  rank-distance metrics;
* the fitted summary or diagnostic object that was interpreted;
* the comparison baseline or held-out check; and
* the decision that consumed the diagnostic.

This record matters because analysis routines can be correct while their
interpretation is not. The documentation, artifact, or release note should make
that interpretation explicit.
