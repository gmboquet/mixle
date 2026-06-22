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
from pysp.inference.em import EMStrategy, run_em
from pysp.inference.estimation import best_of, fit, optimize
from pysp.inference.fisher import FisherView, FixedFisherView, to_fisher

# sampling-based inference — the engine-agnostic NUTS/ADVI facade (target must be sampleable/differentiable)
from pysp.inference.target import (
    AdviResult,
    InferenceBackend,
    NutsResult,
    advi,
    available_backends,
    ess,
    nuts,
    nuts_torch,
    register_inference_backend,
    rhat,
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
    "available_backends",
    "InferenceBackend",
    "register_inference_backend",
]
