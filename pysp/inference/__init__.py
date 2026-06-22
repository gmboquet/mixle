"""The inference concern — fit a model and quantify its parameters.

One home for turning data into a fitted/posterior model: the estimator contract, the MLE/EM drivers,
MAP, closed-form conjugate Bayes, MCMC/NUTS, the EM-strategy protocol, and Fisher geometry. The
estimation/EM/fit/objectives/Fisher machinery physically lives in this package
(``pysp.inference.{estimation,em,fit,objectives,fisher}``); conjugate Bayes is re-exported from
``pysp.stats.bayes`` and NUTS from ``pysp.infer`` (their canonical homes). Thin shims remain at the
old ``pysp.utils.*`` paths.

Names resolve **lazily** (``__getattr__``): the stats leaves import this machinery *during*
``pysp.stats`` import, so eager imports here would re-enter a half-initialized ``pysp.stats``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY = {
    "EMStrategy": "pysp.inference.em",
    "run_em": "pysp.inference.em",
    "best_of": "pysp.inference.estimation",
    "fit": "pysp.inference.estimation",
    "optimize": "pysp.inference.estimation",
    "FisherView": "pysp.inference.fisher",
    "FixedFisherView": "pysp.inference.fisher",
    "to_fisher": "pysp.inference.fisher",
    "ConjugatePosterior": "pysp.stats.bayes.conjugate",
    "MixtureConjugatePosterior": "pysp.stats.bayes.conjugate",
    "conjugate_posterior": "pysp.stats.bayes.conjugate",
    "is_conjugate_family": "pysp.stats.bayes.conjugate",
    "mixture_conjugate_posterior": "pysp.stats.bayes.conjugate",
    "ParameterEstimator": "pysp.stats.compute.pdist",
    "ConjugateUpdatable": "pysp.capability",
    "nuts": "pysp.infer",
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError("module 'pysp.inference' has no attribute %r" % name)
    import importlib

    return getattr(importlib.import_module(target), name)


def __dir__() -> list[str]:
    return sorted(__all__)


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

if TYPE_CHECKING:  # static-analysis view of the lazily-exported surface
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
