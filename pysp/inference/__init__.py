"""The inference concern — fit a model and quantify its parameters.

One home for turning data into a fitted/posterior model: the estimator contract, the MLE/EM drivers,
MAP, closed-form conjugate Bayes, MCMC/NUTS, the EM-strategy protocol, and Fisher geometry. The
estimation/EM/fit/objectives/Fisher machinery physically lives in this package
(``pysp.inference.{estimation,em,fit,objectives,fisher}``); conjugate Bayes is re-exported from
``pysp.stats.bayes`` and NUTS from ``pysp.infer`` (their canonical homes). Thin shims remain at the
old ``pysp.utils.*`` paths.

These imports are eager and cycle-free: the machinery's only ``pysp.stats`` dependency is the compute
layer (``pysp.stats.compute.{pdist,sequence}``), never the ``pysp.stats`` package surface — the
vectorized ``seq_*`` drivers were moved out of ``pysp.stats.__init__`` into ``compute.sequence`` for
exactly this reason.
"""

from __future__ import annotations

from pysp.capability import ConjugateUpdatable
from pysp.infer import nuts
from pysp.inference.em import EMStrategy, run_em
from pysp.inference.estimation import best_of, fit, optimize
from pysp.inference.fisher import FisherView, FixedFisherView, to_fisher
from pysp.stats.bayes.conjugate import (
    ConjugatePosterior,
    MixtureConjugatePosterior,
    conjugate_posterior,
    is_conjugate_family,
    mixture_conjugate_posterior,
)
from pysp.stats.compute.pdist import ParameterEstimator

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
