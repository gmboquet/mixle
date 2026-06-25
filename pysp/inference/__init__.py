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
(``pysp.inference.{estimation,em,fit,objectives,fisher}``), the MCMC samplers (``pysp.inference.mcmc``),
the engine-agnostic NUTS/ADVI target facade (``pysp.inference.target`` + ``.backends`` + ``.diagnostics``).
Conjugate Bayes is re-exported from its canonical home ``pysp.stats.bayes``. ``pysp.infer`` remains as
a deprecated shim onto this package.

These imports are eager and cycle-free: the machinery's only ``pysp.stats`` dependency is the compute
layer (``pysp.stats.compute.{pdist,sequence}``), never the ``pysp.stats`` package surface — the
vectorized ``seq_*`` drivers were moved out of ``pysp.stats.__init__`` into ``compute.sequence`` for
exactly this reason.
"""

from __future__ import annotations

from pysp.capability import ConjugateUpdatable

# calibration diagnostics — "is my probability / interval actually calibrated?"
from pysp.inference.calibration import (
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
from pysp.inference.em import EMStrategy, run_em
from pysp.inference.estimation import best_of, fit, optimize
from pysp.inference.fisher import FisherView, FixedFisherView, to_fisher

# multiple-testing correction (FWER / FDR) and evidence combination
from pysp.inference.multiple_testing import (
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

# the Posterior algebra — inference produces posteriors; you draw from them through one interface
from pysp.inference.posterior import ParameterPosterior, PredictivePosterior, posterior

# bootstrap / permutation inference for arbitrary statistics (distribution-free uncertainty)
from pysp.inference.resampling import (
    BootstrapResult,
    PermutationResult,
    block_bootstrap,
    bootstrap,
    permutation_test,
    wild_bootstrap,
)

# robust / sandwich covariance for M-estimators and regression (honest SEs under misspecification)
from pysp.inference.robust import (
    cluster_robust_covariance,
    newey_west_covariance,
    ols_robust_covariance,
    robust_standard_errors,
    sandwich_covariance,
)

# proper scoring rules — fair currency for comparing probabilistic forecasts / interval methods
from pysp.inference.scoring import (
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

# sampling-based inference — the engine-agnostic NUTS/ADVI facade (target must be sampleable/differentiable)
from pysp.inference.target import (
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
from pysp.stats.bayes.conjugate import (
    ConjugatePosterior,
    MixtureConjugatePosterior,
    conjugate_posterior,
    is_conjugate_family,
    mixture_conjugate_posterior,
)
from pysp.stats.compute.pdist import ParameterEstimator

# the functional estimation drivers (moved off the pysp.stats object namespace)
from pysp.stats.compute.sequence import estimate, initialize, seq_estimate, seq_initialize

__all__ = [
    # the estimator contract + MLE/EM/MAP drivers
    "ParameterEstimator",
    "estimate",
    "initialize",
    "seq_estimate",
    "seq_initialize",
    "optimize",
    "fit",
    "best_of",
    "run_em",
    "EMStrategy",
    # the Posterior algebra (q(z|x) / q(theta|x) / posterior-predictive behind one interface)
    "posterior",
    "ParameterPosterior",
    "PredictivePosterior",
    # calibration diagnostics (reliability diagrams, ECE/MCE, PIT, coverage curves)
    "reliability_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "top_label_confidence",
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
