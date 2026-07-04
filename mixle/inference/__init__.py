"""The inference concern — fit a model and quantify its parameters.

One home for turning data into a fitted/posterior model. Every entry point is the same idea —
*infer parameters from data* — and differs only in what it **requires of its input**:

* **closed-form conjugate Bayes** (``conjugate_posterior``) — needs a ``ConjugateUpdatable`` family;
* **MLE / EM / MAP** (``fit`` / ``optimize`` / ``run_em``) — needs an ``Estimator`` for the model;
* **sampling-based inference** (``nuts`` / ``advi``) — needs a *sampleable / differentiable target*
  (a log-density callable, or a model that can be sampled). This is exactly why it belongs here and
  not in a separate package: it is inference under a capability precondition on the target, nothing
  more.

Everything physically lives in this package: the estimation / EM / fit / objectives / Fisher machinery
(``mixle.inference.{estimation,em,fit,objectives,fisher}``), the MCMC samplers (``mixle.inference.mcmc``),
the engine-agnostic NUTS/ADVI target facade (``mixle.inference.target`` + ``.backends`` + ``.diagnostics``).
Conjugate Bayes is re-exported from its canonical home ``mixle.stats.bayes``. ``mixle.infer`` remains as
a deprecated shim onto this package.

These imports are eager and cycle-free: the machinery's only ``mixle.stats`` dependency is the compute
layer (``mixle.stats.compute.{pdist,sequence}``), never the ``mixle.stats`` package surface — the
vectorized ``seq_*`` drivers were moved out of ``mixle.stats.__init__`` into ``compute.sequence`` for
exactly this reason.
"""

from __future__ import annotations

from mixle.capability import ConjugateUpdatable
from mixle.inference import production
from mixle.inference.bayesian_network import (
    HeterogeneousBayesianNetwork,
    MixtureOfBayesianNetworks,
    bayesian_network_bic,
    learn_bayesian_network,
    learn_mixture_bayesian_network,
    select_mixture_components,
)

# belief states: a distribution over a latent, updated by evidence (the assimilation step)
from mixle.inference.belief import BeliefState, GaussianBelief, as_belief
from mixle.inference.blackbox import LaplacePosterior, laplace_posterior
from mixle.inference.calibrate_fit import CalibrationReport, calibration_report

# calibration diagnostics — "is my probability / interval actually calibrated?"
from mixle.inference.calibration import (
    ProbabilityCalibrator,
    calibrate_probabilities,
    coverage_curve,
    expected_calibration_error,
    interval_coverage,
    maximum_calibration_error,
    pit_calibration_error,
    pit_ensemble,
    pit_histogram,
    pit_values,
    reliability_curve,
    top_label_confidence,
)
from mixle.inference.causal import InterventionalNetwork, average_causal_effect, counterfactual, do
from mixle.inference.conformal import (
    conformal_label_sets,
    conformal_label_threshold,
    cv_plus,
    jackknife_plus,
    mondrian_conformal,
    split_conformal,
    weighted_conformal,
)
from mixle.inference.create import CreatedModel, create
from mixle.inference.cross_validation import (
    NestedFold,
    blocked_kfold,
    group_kfold,
    kfold,
    leave_one_group_out,
    leave_one_out,
    nested_kfold,
    purged_kfold,
    spatial_block_kfold,
    stratified_kfold,
    time_series_split,
)

# Bayes-optimal decisions under a fitted posterior (decision-theoretic action + tail risk)
from mixle.inference.decision import RiskProfile, bayes_action
from mixle.inference.em import EMStrategy, run_em
from mixle.inference.errors_in_variables import DemingFit, deming_regression, propagate_uncertainty, simex
from mixle.inference.estimation import BayesianStreamingEstimator, EMStep, best_of, fit, optimize
from mixle.inference.explain import Explanation, explain
from mixle.inference.fisher import FisherView, FixedFisherView, to_fisher
from mixle.inference.forecast import Forecast, forecast

# generalized linear models + penalized / robust / quantile regression on plain arrays
from mixle.inference.glm import (
    Family,
    GLMResult,
    PenalizedResult,
    RegressionFit,
    elastic_net,
    glm,
    lasso,
    quantile_regression,
    ridge_regression,
    robust_regression,
)
from mixle.inference.jit import JittedScorer, jit_em_mixture, jit_seq_log_density

# model comparison: paired score differences + non-nested (Vuong/Clarke) tests
from mixle.inference.model_comparison import (
    clarke_test,
    compare_elpd,
    paired_score_difference,
    vuong_test,
)

# multiple-testing correction (FWER / FDR) and evidence combination
from mixle.inference.multiple_testing import (
    adjust_pvalues,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    fisher_combine,
    hochberg,
    holm,
    stouffer_combine,
    tippett_combine,
)

# classical nonparametric (rank-based) hypothesis tests
from mixle.inference.nonparametric import (
    DunnResult,
    MannWhitneyResult,
    TestResult,
    WilcoxonResult,
    brunner_munzel,
    cliffs_delta,
    dunn_test,
    friedman_test,
    jonckheere_terpstra,
    kruskal_wallis,
    ks_1samp,
    ks_2samp,
    mann_whitney_u,
    mood_median_test,
    page_trend_test,
    runs_test,
    sign_test,
    wilcoxon_signed_rank,
)
from mixle.inference.orchestration import (
    LearnedAcquisition,
    LearnedPolicy,
    learn_action_policy,
    learn_placement_policy,
    learn_schedule_policy,
    meta_improve,
)

# ordinal (cumulative-link) regression + rank-concordance measures
from mixle.inference.ordinal import (
    OrdinalResult,
    concordance_summary,
    goodman_kruskal_gamma,
    kendall_tau,
    ordinal_regression,
    somers_d,
)
from mixle.inference.placement import BlockPlacement, PlacementPlan, PoolSpec, plan_placement
from mixle.inference.planning import (
    BlockPlan,
    EstimationCertificate,
    Guarantee,
    certify,
    plan_estimation,
)
from mixle.inference.posterior import ParameterPosterior, PredictivePosterior, posterior

# closed-form variational projections — compress a structured teacher onto a smaller student exactly
from mixle.inference.project import collapse_mixture, fisher_merge, gaussian_kl, moment_project, reduce_mixture
from mixle.inference.reproduce import (
    ReproReceipt,
    data_fingerprint,
    param_fingerprint,
    record_fit,
    verify_reproducible,
)

# bootstrap / permutation inference for arbitrary statistics (distribution-free uncertainty)
from mixle.inference.resampling import (
    BootstrapResult,
    PermutationResult,
    block_bootstrap,
    bootstrap,
    permutation_test,
    wild_bootstrap,
)

# robust / sandwich covariance for M-estimators and regression (honest SEs under misspecification)
from mixle.inference.robust import (
    cluster_robust_covariance,
    newey_west_covariance,
    ols_robust_covariance,
    robust_standard_errors,
    sandwich_covariance,
)

# proper scoring rules — fair currency for comparing probabilistic forecasts / interval methods
from mixle.inference.scoring import (
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

# verifier-based selection — the generic best-of-N test-time-compute selector
from mixle.inference.select import SelectionResult, select_best
from mixle.inference.simulate import Scenario, Simulator, simulate
from mixle.inference.skill import Skill, SkillRegistry, default_registry, skill

# online / streaming estimators (single discoverable surface for the streaming drivers)
from mixle.inference.streaming import IncrementalEstimator, StreamingEstimator
from mixle.inference.structure import (
    DependencyTreeDistribution,
    MixtureOfDependencyTrees,
    dependency_gain,
    learn_mixture_structure,
    learn_structure,
)
from mixle.inference.structure_embedded import EmbeddedStructureModel, learn_structure_embedded

# survival / time-to-event estimators and hazard regression
from mixle.inference.survival import (
    CoxResult,
    FrailtyCoxResult,
    aalen_additive,
    aalen_johansen,
    cox_ph,
    discrete_time_hazard,
    frailty_cox,
    kaplan_meier,
    nelson_aalen,
    to_person_period,
)
from mixle.inference.synthesize import Dataset, synthesize

# sampling-based inference — the engine-agnostic NUTS/ADVI facade (target must be sampleable/differentiable)
from mixle.inference.target import (
    AdviResult,
    InferenceBackend,
    NutsResult,
    advi,
    available_backends,
    ess,
    ess_bulk,
    ess_tail,
    folded_split_rhat,
    geweke_z,
    mcmc_summary,
    mcse_mean,
    nuts,
    nuts_torch,
    register_inference_backend,
    rhat,
    rhat_max,
    split_rhat,
)

# epistemic / aleatoric uncertainty decomposition for any predictive (generalizes KG BALD)
from mixle.inference.uncertainty import (
    Clustering,
    UncertaintyDecomposition,
    cluster_samples,
    decompose_entropy,
    decompose_uncertainty,
    decompose_variance,
    marginalize_meaning,
    posterior_ensemble,
    predictive_distribution,
    semantic_entropy,
)
from mixle.inference.uq import UQResult, uq
from mixle.stats.bayes.conjugate import (
    ConjugatePosterior,
    MixtureConjugatePosterior,
    conjugate_posterior,
    is_conjugate_family,
    mixture_conjugate_posterior,
)
from mixle.stats.compute.pdist import ParameterEstimator

# the functional estimation drivers (moved off the mixle.stats object namespace)
from mixle.stats.compute.sequence import estimate, initialize, seq_estimate, seq_initialize

__all__ = [
    "Explanation",
    "explain",
    "InterventionalNetwork",
    "average_causal_effect",
    "counterfactual",
    "do",
    "Forecast",
    "forecast",
    # the estimator contract + MLE/EM/MAP drivers
    "ParameterEstimator",
    "estimate",
    "initialize",
    "seq_estimate",
    "seq_initialize",
    "optimize",
    "jit_seq_log_density",
    "jit_em_mixture",
    "JittedScorer",
    "laplace_posterior",
    "LaplacePosterior",
    "collapse_mixture",
    "reduce_mixture",
    "moment_project",
    "gaussian_kl",
    "fisher_merge",
    "fit",
    "EMStep",
    "best_of",
    "run_em",
    "EMStrategy",
    # online / streaming estimators (single discoverable surface for the streaming drivers)
    "StreamingEstimator",
    "IncrementalEstimator",
    "BayesianStreamingEstimator",
    # Bayes-optimal decisions under a fitted posterior (action + tail-risk profile)
    "bayes_action",
    "RiskProfile",
    # MLOps / production layer (provenance, drift, registry, serving, monitor) lives in the
    # mixle.inference.production subpackage -- imported as `from mixle.inference.production import ...`.
    "production",
    # estimation planning + certificates (the right-method-provably keystone)
    "certify",
    "plan_estimation",
    "EstimationCertificate",
    "BlockPlan",
    "Guarantee",
    # uq() -- one verb, method auto-selected (Laplace / conformal / semantic entropy)
    "uq",
    "UQResult",
    # calibration as a post-condition of fitting (is the model's uncertainty honest on holdout?)
    "calibration_report",
    "CalibrationReport",
    # placement planning -- the local-vs-pool axis of the estimation plan (99/1 topology)
    "plan_placement",
    "PlacementPlan",
    "BlockPlacement",
    "PoolSpec",
    # learned orchestration -- the platform's own decisions as models trained on telemetry (never-worse)
    "learn_placement_policy",
    "LearnedPolicy",
    "learn_action_policy",
    "LearnedAcquisition",
    "learn_schedule_policy",
    "meta_improve",
    # simulate() -- a fitted model as a runtime data generator with causal intervention scenarios
    "simulate",
    "Simulator",
    "Scenario",
    # synthesize() -- a dataset factory: sample, label, keep only what verifies
    "synthesize",
    "Dataset",
    # create() -- data (+ budget/device) to a certified model artifact
    "create",
    "CreatedModel",
    # skill() -- package a fitted model / function as a named, reusable, indexed verb
    "skill",
    "Skill",
    "SkillRegistry",
    "default_registry",
    # the Posterior algebra (q(z|x) / q(theta|x) / posterior-predictive behind one interface)
    "posterior",
    "ParameterPosterior",
    "PredictivePosterior",
    # reproducibility receipts -- record a fit, replay it, check it comes out bit-for-bit
    "record_fit",
    "verify_reproducible",
    "ReproReceipt",
    "data_fingerprint",
    "param_fingerprint",
    # belief states (distribution over a latent, updated by evidence)
    "BeliefState",
    "GaussianBelief",
    "as_belief",
    # epistemic / aleatoric uncertainty decomposition (BALD entropy + law-of-total-variance)
    "UncertaintyDecomposition",
    "decompose_uncertainty",
    "decompose_entropy",
    "decompose_variance",
    "predictive_distribution",
    "posterior_ensemble",
    # semantic clustering of stochastic samples (LLM meaning-clusters -> semantic entropy)
    "Clustering",
    "cluster_samples",
    "marginalize_meaning",
    "semantic_entropy",
    # calibration diagnostics (reliability diagrams, ECE/MCE, PIT, coverage curves)
    "reliability_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "top_label_confidence",
    # recalibration: map raw scores -> calibrated probabilities (isotonic / Platt)
    "ProbabilityCalibrator",
    "calibrate_probabilities",
    "pit_values",
    "pit_ensemble",
    "pit_histogram",
    "pit_calibration_error",
    "interval_coverage",
    "coverage_curve",
    # multiple-testing correction (FWER/FDR) and evidence combination
    "bonferroni",
    "holm",
    "hochberg",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "adjust_pvalues",
    "fisher_combine",
    "stouffer_combine",
    "tippett_combine",
    # bootstrap / permutation inference (distribution-free uncertainty for any statistic)
    "bootstrap",
    "BootstrapResult",
    "block_bootstrap",
    "wild_bootstrap",
    "permutation_test",
    "PermutationResult",
    # robust / sandwich covariance (HC0-3, cluster-robust, Newey-West HAC, generic M-estimator)
    "sandwich_covariance",
    "ols_robust_covariance",
    "cluster_robust_covariance",
    "newey_west_covariance",
    "robust_standard_errors",
    # GLM + penalized / robust / quantile regression (array-level)
    "glm",
    "GLMResult",
    "Family",
    "ridge_regression",
    "elastic_net",
    "lasso",
    "PenalizedResult",
    "robust_regression",
    "quantile_regression",
    "RegressionFit",
    # survival / time-to-event estimators and hazard regression
    "kaplan_meier",
    "nelson_aalen",
    "cox_ph",
    "CoxResult",
    "to_person_period",
    "discrete_time_hazard",
    "aalen_johansen",
    "aalen_additive",
    "frailty_cox",
    "FrailtyCoxResult",
    # ordinal regression + rank concordance
    "ordinal_regression",
    "OrdinalResult",
    "concordance_summary",
    "kendall_tau",
    "goodman_kruskal_gamma",
    "somers_d",
    # classical nonparametric (rank-based) hypothesis tests
    "mann_whitney_u",
    "MannWhitneyResult",
    "wilcoxon_signed_rank",
    "WilcoxonResult",
    "sign_test",
    "kruskal_wallis",
    "friedman_test",
    "brunner_munzel",
    "mood_median_test",
    "dunn_test",
    "DunnResult",
    "jonckheere_terpstra",
    "page_trend_test",
    "ks_1samp",
    "ks_2samp",
    "runs_test",
    "cliffs_delta",
    "TestResult",
    # measurement error: errors-in-variables, SIMEX, Monte-Carlo uncertainty propagation
    "deming_regression",
    "DemingFit",
    "simex",
    "propagate_uncertainty",
    # model comparison (paired score diffs, Vuong/Clarke non-nested, elpd comparison)
    "paired_score_difference",
    "vuong_test",
    "clarke_test",
    "compare_elpd",
    # conformal prediction (distribution-free intervals; split/J+/CV+/Mondrian/weighted)
    "split_conformal",
    "jackknife_plus",
    "cv_plus",
    "mondrian_conformal",
    "weighted_conformal",
    "conformal_label_threshold",
    "conformal_label_sets",
    # automatic dependency-structure learning for heterogeneous records
    "learn_structure",
    "learn_mixture_structure",
    "learn_bayesian_network",
    "learn_mixture_bayesian_network",
    "select_mixture_components",
    "bayesian_network_bic",
    "learn_structure_embedded",
    "EmbeddedStructureModel",
    "HeterogeneousBayesianNetwork",
    "MixtureOfBayesianNetworks",
    "dependency_gain",
    "DependencyTreeDistribution",
    "MixtureOfDependencyTrees",
    # cross-validation fold generators (i.i.d., grouped, temporal, spatial-block, nested)
    "kfold",
    "blocked_kfold",
    "leave_one_out",
    "stratified_kfold",
    "leave_one_group_out",
    "group_kfold",
    "time_series_split",
    "purged_kfold",
    "spatial_block_kfold",
    "nested_kfold",
    "NestedFold",
    # proper scoring rules (lower is better; pair with resampling for score-difference CIs)
    "log_score",
    "brier_score",
    "brier_decomposition",
    "crps_ensemble",
    "crps_gaussian",
    "interval_score",
    "winkler_score",
    "pinball_loss",
    "energy_score",
    "skill_score",
    # verifier-based selection (best-of-N test-time-compute selector + conformal confidence)
    "select_best",
    "SelectionResult",
    # closed-form conjugate Bayes
    "ConjugateUpdatable",
    "conjugate_posterior",
    "mixture_conjugate_posterior",
    "is_conjugate_family",
    "ConjugatePosterior",
    "MixtureConjugatePosterior",
    # Fisher geometry
    "FisherView",
    "FixedFisherView",
    "to_fisher",
    # sampling-based inference (NUTS / ADVI on a sampleable/differentiable target)
    "nuts",
    "nuts_torch",
    "advi",
    "NutsResult",
    "AdviResult",
    "rhat",
    "ess",
    "split_rhat",
    "folded_split_rhat",
    "rhat_max",
    "mcse_mean",
    "geweke_z",
    "mcmc_summary",
    "ess_bulk",
    "ess_tail",
    "available_backends",
    "InferenceBackend",
    "register_inference_backend",
]
