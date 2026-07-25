"""Capability-specific contracts for objects attached to ``RandomVariable.result``.

The fitters (``inference.py``, ``vmp.py``, ``regression.py``) each attach a *result* object to a
fitted RV via ``RandomVariable._bound(..., result=...)``. There are several concrete result
classes — ``Posterior``, ``ConjugatePosterior``, ``ConjugateMixturePosterior``,
``HierarchicalPosterior`` (inference.py), ``MixtureVMPResult`` / ``GraphResult`` / ``_VMPFit``
(vmp.py), and ``RegressionResult`` / ``LMMResult`` / ``LocationScaleResult`` (regression.py) — and
they are consumed *duck-typed* in :mod:`mixle.ppl.core` (``getattr(r, "summary", None)``,
``hasattr(r, "samples")``, ``getattr(r, "predictive", None)``, ``getattr(r, "build", None)``,
``r.pointwise_log_likelihood(...)``).

There is deliberately no "everything a posterior might do" protocol. Public fit routes implement
different subsets; each consumer checks one narrow runtime-checkable protocol, or inspects the
normalized :func:`result_capabilities` record.

This module imports only ``typing`` (+ ``numpy`` for an annotation), so ``core.py`` can import it at
module load without an import cycle through inference/regression/vmp (which import ``core`` at module
level).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Summarizable(Protocol):
    """A result that can report a posterior / fit ``summary()`` (read by ``RandomVariable.summary``).

    ``supports(r, Summarizable)`` is the runtime capability check.
    """

    def summary(self) -> dict:
        """Return a dictionary summary of the fitted result."""
        ...


@runtime_checkable
class Sampleable(Protocol):
    """A result that can return parameter / latent ``samples(...)`` (read by ``RandomVariable.posterior``).

    ``supports(r, Sampleable)`` is the runtime capability check.
    """

    def samples(self, param: Any = ..., *args: Any, **kwargs: Any) -> np.ndarray:
        """Return posterior samples for a parameter, latent, or default result target."""
        ...


@runtime_checkable
class PointwiseLogLikelihood(Protocol):
    """A result that can score observations under each posterior draw."""

    def pointwise_log_likelihood(self, data: Any) -> np.ndarray:
        """Return a ``(n_draws, n_observations)`` log-likelihood matrix."""
        ...


@runtime_checkable
class Predictive(Protocol):
    """A result carrying an executable posterior-predictive callable."""

    predictive: Callable[[int, Any], Any]


@dataclass(frozen=True)
class ResultCapabilities:
    """Normalized runtime capability view for any public fit result."""

    summarizable: bool
    sampleable: bool
    predictive: bool
    pointwise_log_likelihood: bool


def result_capabilities(result: Any) -> ResultCapabilities:
    """Return capability flags without assuming a particular fitter result class."""
    predictive = getattr(result, "predictive", None)
    return ResultCapabilities(
        summarizable=isinstance(result, Summarizable),
        sampleable=isinstance(result, Sampleable),
        predictive=callable(predictive),
        pointwise_log_likelihood=isinstance(result, PointwiseLogLikelihood),
    )


__all__ = [
    "Summarizable",
    "Sampleable",
    "PointwiseLogLikelihood",
    "Predictive",
    "ResultCapabilities",
    "result_capabilities",
]
