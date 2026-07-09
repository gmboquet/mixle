"""Shared typing contract for the inference result objects attached to ``RandomVariable.result``.

The fitters (``inference.py``, ``vmp.py``, ``regression.py``) each attach a *result* object to a
fitted RV via ``RandomVariable._bound(..., result=...)``. There are several concrete result
classes — ``Posterior``, ``ConjugatePosterior``, ``ConjugateMixturePosterior``,
``HierarchicalPosterior`` (inference.py), ``MixtureVMPResult`` / ``GraphResult`` / ``_VMPFit``
(vmp.py), and ``RegressionResult`` / ``LMMResult`` / ``LocationScaleResult`` (regression.py) — and
they are consumed *duck-typed* in :mod:`mixle.ppl.core` (``getattr(r, "summary", None)``,
``hasattr(r, "samples")``, ``getattr(r, "predictive", None)``, ``getattr(r, "build", None)``,
``r.pointwise_log_likelihood(...)``).

``PosteriorResult`` is the structural (``typing.Protocol``) capture of that *optional* common
surface. It is intentionally a contract, not a base class: the concrete result classes keep their
distinct bodies and need not inherit it; it exists so that ``RandomVariable.result`` and the
fitters' return annotations can name the shared surface instead of bare ``Any``. Every member is
optional at the call site (the consumers probe with ``hasattr``/``getattr``/``callable``), which is
why the Protocol carries the union of probed members rather than a strict required set.

This module imports only ``typing`` (+ ``numpy`` for an annotation), so ``core.py`` can import it at
module load without an import cycle through inference/regression/vmp (which import ``core`` at module
level).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PosteriorResult(Protocol):
    """Structural contract for a fitted RV's ``.result``.

    Captures the *union* surface that :mod:`mixle.ppl.core` (and the diagnostics) probe on a result
    object. All members are *optional* in practice — consumers guard every access with ``hasattr`` /
    ``getattr`` / ``callable`` — so a concrete result need only implement the slice it supports
    (e.g. a point-estimate ``RegressionResult`` has ``summary``/``predict`` but no ``samples``;
    a ``_VIResult`` is a bare raw-result holder). ``acceptance_rate`` and ``predictive`` are the two
    attributes every concrete result sets (``None`` when not applicable).

    Because the members are independently optional, this *union* protocol is intentionally **not**
    the right gate for an individual probe site: a ``runtime_checkable`` ``isinstance`` against it
    requires *every* member to be present, which only ``Posterior`` satisfies (the conjugate /
    hierarchical / regression / vmp results all lack ``pointwise_log_likelihood`` and several lack
    ``samples`` or ``predictive``). Per-capability probes therefore dispatch on the narrow
    single-method protocols below, each of which is exactly equivalent to the lenient
    ``hasattr``/``callable`` probe it replaces.
    """

    # Set by every concrete result (``None`` when not applicable).
    acceptance_rate: Any
    predictive: Any

    def summary(self) -> dict:
        """Posterior / fit summary (read by ``RandomVariable.summary``)."""
        ...

    def samples(self, param: Any = ..., *args: Any, **kwargs: Any) -> np.ndarray:
        """Parameter / latent draws (read by ``RandomVariable.posterior``)."""
        ...

    def pointwise_log_likelihood(self, data: Any) -> np.ndarray:
        """``(n_draws, n_obs)`` log-likelihood for WAIC / PSIS-LOO."""
        ...


@runtime_checkable
class Summarizable(Protocol):
    """A result that can report a posterior / fit ``summary()`` (read by ``RandomVariable.summary``).

    Narrow, single-method facet of :class:`PosteriorResult`: ``supports(r, Summarizable)`` is exactly
    the old ``callable(getattr(r, "summary", None))`` probe for any result whose ``summary`` is a
    method (every concrete result class is).
    """

    def summary(self) -> dict:
        """Return a dictionary summary of the fitted result."""
        ...


@runtime_checkable
class Sampleable(Protocol):
    """A result that can return parameter / latent ``samples(...)`` (read by ``RandomVariable.posterior``).

    Narrow, single-method facet of :class:`PosteriorResult`: ``supports(r, Sampleable)`` is exactly
    the old ``hasattr(r, "samples")`` probe.
    """

    def samples(self, param: Any = ..., *args: Any, **kwargs: Any) -> np.ndarray:
        """Return posterior samples for a parameter, latent, or default result target."""
        ...
