"""The inference concern — fit a model and quantify its parameters.

One home for everything that turns data into a fitted/posterior model: the estimator contract, the
MLE/EM drivers, MAP, closed-form conjugate Bayes, MCMC (NUTS), the EM-strategy protocol, and Fisher
geometry. The capability ``ConjugateUpdatable`` is the top tier (closed-form); everything else routes
through the numerical drivers. Re-export shim today (nothing moves); the concern-oriented layout is in
``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from pysp.capability import ConjugateUpdatable

# --- numerical sampling (MCMC / NUTS) ---
from pysp.infer import nuts

# --- closed-form conjugate Bayes (the top tier) ---
from pysp.stats.bayes.conjugate import (
    ConjugatePosterior,
    MixtureConjugatePosterior,
    conjugate_posterior,
    is_conjugate_family,
    mixture_conjugate_posterior,
)

# --- the estimator contract + the MLE/EM drivers ---
from pysp.stats.compute.pdist import ParameterEstimator
from pysp.utils.em import EMStrategy, run_em
from pysp.utils.estimation import best_of, fit, optimize

# --- Fisher geometry ---
from pysp.utils.fisher import FisherView, FixedFisherView, to_fisher

__all__ = [
    "ParameterEstimator",
    "optimize",
    "fit",
    "best_of",
    "run_em",
    "EMStrategy",
    "ConjugateUpdatable",
    "conjugate_posterior",
    "mixture_conjugate_posterior",
    "is_conjugate_family",
    "ConjugatePosterior",
    "MixtureConjugatePosterior",
    "FisherView",
    "FixedFisherView",
    "to_fisher",
    "nuts",
]
