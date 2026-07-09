Inference Toolkit
=================

The main :doc:`inference` page explains fitting: ``optimize``, ``fit``,
initialization, EM, streaming, backends, and objectives. The
``mixle.inference`` namespace is broader than fitting. It also contains the
toolkit for evaluating, calibrating, comparing, explaining, and operationally
using probabilistic models.

Use this page as the map for those utilities.

Scoring Rules
-------------

Proper scoring rules are the currency for comparing probabilistic predictions.

Imports:

.. code-block:: python

   from mixle.inference import (
       brier_decomposition,
       brier_score,
       crps_ensemble,
       crps_gaussian,
       energy_score,
       interval_score,
       log_score,
       pinball_loss,
       skill_score,
       winkler_score,
   )

Use log score for full density forecasts, CRPS for continuous predictive
distributions, Brier score for probabilities, interval/Winkler scores for
prediction intervals, and skill scores when comparing against a reference
forecast.

``brier_decomposition`` separates calibration, refinement, and uncertainty
terms for binary probability forecasts.

Calibration
-----------

Calibration tools check whether predicted probabilities and intervals behave
empirically as stated.

Key imports:

* ``ProbabilityCalibrator``;
* ``calibrate_probabilities``;
* ``reliability_curve``;
* ``expected_calibration_error`` and ``maximum_calibration_error``;
* ``coverage_curve`` and ``interval_coverage``;
* ``pit_values``, ``pit_histogram``, ``pit_calibration_error``, and
  ``pit_ensemble``;
* ``top_label_confidence``.

Use these before allowing probabilities to drive decisions, escalation, or
claims about uncertainty quality.

Calibration evidence should name the split, target coverage, realized
coverage, and any segment where coverage is materially worse. A single global
ECE or PIT number is useful, but it is rarely enough for a decision-facing
model.

Conformal Prediction
--------------------

Conformal helpers provide finite-sample coverage wrappers:

* ``split_conformal``;
* ``weighted_conformal``;
* ``mondrian_conformal``;
* ``jackknife_plus``;
* ``cv_plus``;
* ``conformal_label_sets``;
* ``conformal_label_threshold``.

Use conformal methods when the model score is useful but raw probabilities are
not trusted enough to set an answer threshold directly.

Cross-Validation Splitters
--------------------------

The cross-validation surface includes:

* ``kfold`` and ``stratified_kfold``;
* ``group_kfold`` and ``leave_one_group_out``;
* ``leave_one_out``;
* ``blocked_kfold`` and ``spatial_block_kfold``;
* ``time_series_split``;
* ``purged_kfold``;
* ``nested_kfold`` and ``NestedFold``.

Choose a splitter that respects the data-generating structure. Do not use an
iid splitter for grouped, spatial, or temporal data just because it is
convenient.

Model Comparison
----------------

Model-comparison utilities include:

* ``paired_score_difference``;
* ``compare_elpd``;
* ``vuong_test``;
* ``clarke_test``.

Use paired comparisons when two models score the same held-out cases. Use
non-nested tests when a family swap is being considered. For promotion gates,
see :doc:`evolution`.

For release or deployment decisions, store the per-case scores as well as the
summary statistic. They let reviewers inspect whether an apparent improvement
comes from broad gains or from a few extreme examples.

Multiple Testing
----------------

When many hypotheses, alerts, or candidate models are tested together, use the
multiple-testing helpers:

* ``bonferroni``;
* ``holm``;
* ``hochberg``;
* ``benjamini_hochberg``;
* ``benjamini_yekutieli``;
* ``adjust_pvalues``;
* ``fisher_combine``, ``stouffer_combine``, and ``tippett_combine``.

These tools belong near monitoring, model selection, and large diagnostic
reports where isolated p-values would be misleading.

Regression and Classical Inference
----------------------------------

``mixle.inference`` includes plain-array regression tools for cases where a
full distribution family is not the right interface:

* ``glm`` and ``Family``;
* ``lasso``, ``ridge_regression``, and ``elastic_net``;
* ``robust_regression``;
* ``quantile_regression``;
* ``RegressionFit``, ``GLMResult``, and ``PenalizedResult``.

Errors-in-variables tools include:

* ``deming_regression`` and ``DemingFit``;
* ``simex``;
* ``propagate_uncertainty``.

Robust uncertainty tools include:

* ``sandwich_covariance``;
* ``ols_robust_covariance``;
* ``cluster_robust_covariance``;
* ``newey_west_covariance``;
* ``robust_standard_errors``.

Nonparametric Tests
-------------------

Rank-based and distribution-free tests include:

* ``mann_whitney_u`` and ``MannWhitneyResult``;
* ``wilcoxon_signed_rank`` and ``WilcoxonResult``;
* ``kruskal_wallis``;
* ``friedman_test``;
* ``dunn_test`` and ``DunnResult``;
* ``brunner_munzel``;
* ``ks_1samp`` and ``ks_2samp``;
* ``runs_test``;
* ``TestResult``;
* ``sign_test``;
* ``mood_median_test``;
* ``jonckheere_terpstra``;
* ``page_trend_test``;
* ``cliffs_delta``.

Use these as diagnostics and scientific-analysis tools, not as substitutes for
a fitted generative model when the downstream workflow needs prediction,
sampling, or composition.

Ordinal and Survival Models
---------------------------

Ordinal tools:

* ``ordinal_regression`` and ``OrdinalResult``;
* ``kendall_tau``;
* ``somers_d``;
* ``goodman_kruskal_gamma``;
* ``concordance_summary``.

Survival tools:

* ``kaplan_meier``;
* ``nelson_aalen``;
* ``cox_ph`` and ``CoxResult``;
* ``frailty_cox`` and ``FrailtyCoxResult``;
* ``aalen_additive``;
* ``aalen_johansen``;
* ``discrete_time_hazard``;
* ``to_person_period``.

Use these for ordered outcomes and time-to-event data. Use
``SurvivalDistribution`` from :doc:`stats-structured` when survival behavior is
part of a larger distribution composition.

Bayesian Networks, Causal Interventions, and Structure
------------------------------------------------------

Structure helpers include:

* ``learn_bayesian_network``;
* ``HeterogeneousBayesianNetwork``;
* ``MixtureOfBayesianNetworks``;
* ``learn_mixture_bayesian_network``;
* ``learn_structure`` and ``learn_mixture_structure``;
* ``DependencyTreeDistribution`` and ``MixtureOfDependencyTrees``;
* ``dependency_gain``.

Causal helpers include:

* ``InterventionalNetwork``;
* ``do``;
* ``counterfactual``;
* ``average_causal_effect``.

Use these tools when the model structure itself is the object of inference.
Treat learned causal structure as a hypothesis that requires domain review and
held-out checks.

Posterior, Belief, and Explanation
----------------------------------

Posterior and belief helpers include:

* ``posterior``;
* ``ParameterPosterior``;
* ``PredictivePosterior``;
* ``laplace_posterior`` and ``LaplacePosterior``;
* ``BeliefState``;
* ``GaussianBelief``;
* ``as_belief``;
* ``explain`` and ``Explanation``;
* ``forecast`` and ``Forecast``.

These are the bridge from fitted models to uncertainty-aware behavior:
posterior predictive queries, latent belief updates, explanations, and
forecasts.

``laplace_posterior`` is the black-box route when a fitted model can be
flattened into unconstrained parameters but has no conjugate posterior. It
builds a Gaussian approximation from the model's own density scorer and a
finite-difference Hessian. Unsupported parameter structures fail explicitly
rather than pretending to be Bayesian.

Projection and Compression
--------------------------

``mixle.inference.project`` contains closed-form projections for cases where a
rich probabilistic object can be compressed without sampling or optimization.

Key imports:

* ``collapse_mixture``;
* ``reduce_mixture``;
* ``moment_project``;
* ``gaussian_kl``;
* ``fisher_merge``.

Use ``collapse_mixture`` when a Gaussian mixture should become one Gaussian.
The result matches the mixture's first two moments exactly: the mean is the
weighted component mean, and the covariance uses the law of total variance.

Use ``reduce_mixture`` when a Gaussian mixture should keep several components
but fewer than it currently has. The current route uses Runnalls-style greedy
merges: each merge preserves the global first two moments and chooses the pair
with the smallest analytic merge cost.

.. code-block:: python

   from mixle.inference import collapse_mixture, reduce_mixture

   compact = collapse_mixture(large_gaussian_mixture)
   smaller = reduce_mixture(large_gaussian_mixture, n_components=4)

``moment_project`` is the dispatch helper: it takes the exact path for
supported Gaussian mixtures and delegates to the sampling projection in
``mixle.ops.project`` when you provide a target family and request the
approximate path.

``fisher_merge`` merges flat parameter estimates with scalar, diagonal, or
full Fisher information. It is the precision-weighted mean behind Laplace or
Fisher summaries, and it matches Gaussian product-of-experts mean pooling in
the one-dimensional Gaussian case.

For observed-data Fisher geometry, use ``to_fisher`` to obtain a
``FisherView`` or ``FixedFisherView``. These views flatten sufficient
statistics, expose observed Fisher vectors, and give latent models a common
way to report posterior-expected complete-data statistics.

These functions are inference utilities rather than ordinary distribution
operations because their contract is about how one fitted or posterior object
is approximated by another. They should still be recorded in provenance when
used for production compression.

For end-to-end projection examples, see
``examples/mixture_reduction_benchmark.py`` for Gaussian-mixture reduction and
``examples/project_neural_to_structured.py`` for projecting a trained neural
density onto a structured mixture student.

Sampling-Based Inference
------------------------

The target interface exposes sampling-based inference and diagnostics:

* ``nuts``, ``nuts_torch``, and ``NutsResult``;
* ``advi`` and ``AdviResult``;
* ``InferenceBackend`` and ``available_backends``;
* ``register_inference_backend``;
* ``rhat``, ``split_rhat``, ``folded_split_rhat``, and ``rhat_max``;
* ``ess``, ``ess_bulk``, and ``ess_tail``;
* ``mcse_mean``;
* ``geweke_z``;
* ``mcmc_summary``.

Use these when the target is differentiable or sampleable but closed-form
updates are unavailable.

Sampling routes need diagnostics in the artifact record. Keep chain count,
warmup, draws, acceptance or step-size summaries when available, R-hat, ESS,
and the reason a sampling route was chosen over a simpler point estimate.

JIT Scoring
-----------

``jit_seq_log_density`` returns a ``JittedScorer`` that compiles a fixed
model's whole-tree sequence log-density through the JAX engine. Use it for
repeated held-out scoring, bootstrap scoring, or selection loops where the
model parameters are fixed and the same compiled program can be reused.

``jit_em_mixture`` is the specialized route for finite mixtures whose E-step
and closed-form M-step can be lowered to one compiled JAX program. Treat it as
an acceleration tool for supported mixture families, not as a replacement for
the general EM driver.

Resampling
----------

Resampling utilities include:

* ``bootstrap`` and ``BootstrapResult``;
* ``block_bootstrap``;
* ``wild_bootstrap``;
* ``permutation_test`` and ``PermutationResult``.

Use block bootstrap for dependent data, wild bootstrap for heteroscedastic
regression-like settings, and permutation tests for label or treatment
exchangeability questions.

Decision Theory
---------------

``bayes_action`` and ``RiskProfile`` turn posterior beliefs into actions under
loss. Use them when the end product is a decision, not only a probability.

Verifier Selection
------------------

``select_best`` implements the generic best-of-N pattern used by verifier and
test-time-compute systems: score several candidates and keep the winner.

.. code-block:: python

   from mixle.inference import select_best

   result = select_best(candidates, score=verifier_score, conformal_alpha=0.1)
   winner = result.best

The returned ``SelectionResult`` records the winning index, all scores, the
margin over the runner-up, and an optional confidence flag when the margin
clears the conformal/bootstrap band. The candidates can be strings, plans,
models, samples, or any object accepted by the verifier.

Workflow
--------

A robust applied workflow usually has this order:

1. Fit with :doc:`inference`.
2. Score with proper scoring rules.
3. Check calibration and coverage.
4. Compare models on paired held-out cases.
5. Add conformal or decision rules if behavior must change under uncertainty.
6. Record the verification result in :doc:`production` or :doc:`evolution`.

If any step is skipped, record that explicitly. Silent omissions are the
hardest release defects to audit later.
