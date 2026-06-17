"""Uniform exponential-family canonical map for pysp distributions.

Every exponential-family (and conditional-exponential-family) model is expressed in
the canonical form

    p(x)   = h(x) * exp( <eta, T(x)> - A(eta) )            # unconditional
    p(y|x) = h(y) * exp( <eta(x), T(y)> - A(eta(x)) )      # conditional

This module surfaces that form as a first-class object.  The per-family math already
lives in each distribution's :class:`~pysp.stats.compute.declarations.ExponentialFamilySpec`
(``sufficient_statistics`` T, ``natural_parameters`` eta, ``log_partition`` A,
``base_measure`` h); :func:`to_exponential_family` reads that declaration and wraps it,
so adding a family is a matter of providing its spec -- there is no type switch here.

The container threads a compute engine (numpy by default) so the map works under numpy
and torch and stays autograd-friendly for :meth:`ExponentialFamilyForm.mean_parameters`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pysp.engines import NUMPY_ENGINE
from pysp.stats.compute.declarations import (
    ExponentialFamilySpec,
    _generated_exp_family_scalar_expression,
    declaration_for,
)
from pysp.stats.compute.pdist import ProbabilityDistribution


def _flatten_statistics(statistics: tuple[Any, ...], engine: Any) -> Any:
    """Stack a tuple of per-row statistic blocks into an ``(n, dim)`` matrix.

    Each entry is either a length-``n`` row vector or an ``(n, k)`` block; they are
    reshaped to ``(n, -1)`` and concatenated along the trailing (statistic) axis.
    """
    columns = []
    for stat in statistics:
        arr = engine.asarray(stat)
        shape = tuple(getattr(arr, "shape", ()))
        if len(shape) == 0:
            arr = engine.asarray(np.reshape(engine.to_numpy(arr), (1, 1)))
        elif len(shape) == 1:
            arr = arr[:, None]
        else:
            arr = arr.reshape((shape[0], -1))
        columns.append(arr)
    if len(columns) == 1:
        return columns[0]
    return np.concatenate([engine.to_numpy(c) for c in columns], axis=1)


def _flatten_natural(natural: tuple[Any, ...], engine: Any) -> np.ndarray:
    """Flatten a tuple of natural-parameter blocks into a 1-D vector eta."""
    parts = []
    for value in natural:
        arr = np.asarray(engine.to_numpy(engine.asarray(value)), dtype=np.float64)
        parts.append(arr.reshape(-1))
    if not parts:
        raise ValueError("exponential family requires at least one natural parameter.")
    return np.concatenate(parts)


@dataclass(frozen=True)
class ExponentialFamilyForm:
    """Canonical exponential-family view of a single distribution.

    Holds the source distribution and its :class:`ExponentialFamilySpec`, and exposes
    the canonical pieces (eta, T(x), A(eta), log h(x)) plus derived quantities.  All
    array methods are vectorized over a leading sample axis and routed through
    ``engine`` (numpy by default).
    """

    distribution: ProbabilityDistribution
    spec: ExponentialFamilySpec
    engine: Any = NUMPY_ENGINE

    # -- internal helpers --------------------------------------------------

    def _params(self) -> dict[str, Any]:
        declaration = declaration_for(self.distribution)
        params: dict[str, Any] = {}
        for pspec in declaration.parameters:
            value = getattr(self.distribution, pspec.name)
            if value is None or isinstance(value, (str, bytes, bool, int, float, np.number, type)):
                params[pspec.name] = value
            else:
                params[pspec.name] = self.engine.asarray(value)
        return params

    def _encode(self, x: Any) -> Any:
        """Encode raw observations into the form the spec callables consume.

        The spec's ``sufficient_statistics``/``base_measure`` operate on the
        distribution's *encoded* data (e.g. ``(log x, 1/x)`` for inverse-gamma),
        exactly as the generated scalar scorer does, so we route raw observations
        through the distribution encoder first.
        """
        encoder = self.distribution.dist_to_encoder()
        return encoder.seq_encode(list(x))

    # -- canonical pieces --------------------------------------------------

    @property
    def dim(self) -> int:
        """Length of the natural-parameter / sufficient-statistic vector."""
        return int(self.natural_parameters().shape[0])

    def natural_parameters(self) -> np.ndarray:
        """Return the natural parameters ``eta(theta)`` for the current parameters."""
        natural = tuple(self.spec.natural_parameters(self._params(), self.engine))
        return _flatten_natural(natural, self.engine)

    def sufficient_statistics(self, x: Any) -> Any:
        """Return the sufficient statistics ``T(x)`` with shape ``(n, dim)``."""
        enc = self._encode(x)
        if self.spec.sufficient_statistics_from_params is not None:
            statistics = tuple(self.spec.sufficient_statistics_from_params(enc, self._params(), self.engine))
        else:
            statistics = tuple(self.spec.sufficient_statistics(enc, self.engine))
        return _flatten_statistics(statistics, self.engine)

    def log_partition(self, eta: Any = None) -> Any:
        """Return the log-partition ``A``.

        With ``eta=None`` (default) this is ``A(eta(theta))`` for the current
        parameters.  Supplying ``eta`` requires the dual map ``theta(eta)``; it is
        evaluated by reconstructing a distribution via :meth:`from_natural` and is
        only available where ``from_natural`` has a closed form.
        """
        if eta is None:
            return self.spec.log_partition(self._params(), self.engine)
        dist = self.from_natural(eta)
        if dist is None:
            raise NotImplementedError(
                "%s has no closed-form dual map; log_partition(eta) is unavailable." % type(self.distribution).__name__
            )
        return dist.to_exponential_family(engine=self.engine).log_partition()

    def log_base_measure(self, x: Any) -> Any:
        """Return ``log h(x)`` (in log-space to avoid e.g. ``1/x!`` overflow)."""
        enc = self._encode(x)
        if self.spec.base_measure_from_params is not None:
            base = self.spec.base_measure_from_params(enc, self._params(), self.engine)
        elif self.spec.base_measure is not None:
            base = self.spec.base_measure(enc, self.engine)
        else:
            stats = self.sufficient_statistics(x)
            n = int(self.engine.to_numpy(stats).shape[0])
            return self.engine.asarray(np.zeros(n, dtype=np.float64))
        return base

    def log_density(self, x: Any) -> Any:
        """Return ``<eta, T(x)> - A(eta) + log h(x)`` -- the reconstructed log-density."""
        enc = self._encode(x)
        return _generated_exp_family_scalar_expression(enc, self._params(), self.spec, self.engine)

    # -- derived / conveniences -------------------------------------------

    def mean_parameters(self, eps: float = 1.0e-6, n_samples: int = 200000, seed: int | None = 0) -> np.ndarray:
        """Return the mean (expectation) parameters ``grad A(eta) = E[T(x)]``.

        When the family exposes a closed-form dual map (``exp_family_from_natural``)
        this is the exact gradient of ``A`` by central finite differences in natural
        coordinates.  Otherwise it falls back to a Monte-Carlo estimate of ``E[T(x)]``
        over ``n_samples`` draws -- approximate, but universal for any samplable family.
        """
        if self.from_natural(self.natural_parameters()) is not None:
            eta = self.natural_parameters()
            grad = np.empty_like(eta)
            for i in range(eta.shape[0]):
                step = eps * (abs(float(eta[i])) + 1.0)
                up = eta.copy()
                up[i] += step
                down = eta.copy()
                down[i] -= step
                a_up = float(self.engine.to_numpy(self.log_partition(up)))
                a_down = float(self.engine.to_numpy(self.log_partition(down)))
                grad[i] = (a_up - a_down) / (2.0 * step)
            return grad
        samples = self.distribution.sampler(seed).sample(int(n_samples))
        stats = np.asarray(self.engine.to_numpy(self.sufficient_statistics(samples)), dtype=np.float64)
        return stats.mean(axis=0)

    def from_natural(self, eta: Any) -> ProbabilityDistribution | None:
        """Return ``theta(eta)`` as a reconstructed distribution, or ``None``.

        The default has no generic inverse link; families with a closed form attach
        ``exp_family_from_natural(eta) -> ProbabilityDistribution`` to their class.
        """
        fn = getattr(type(self.distribution), "exp_family_from_natural", None)
        if not callable(fn):
            return None
        return fn(np.asarray(eta, dtype=np.float64))


@dataclass(frozen=True)
class ConditionalExponentialFamilyForm:
    """Canonical exponential-family view of a conditional model ``p(y | x)``.

    The response family fixes ``T``, ``A``, and ``h``; ``natural_parameters(x)``
    supplies the per-row natural parameters ``eta(x)`` (the linear predictor for a
    canonical link).  ``dispersion`` carries a nuisance/dispersion parameter when the
    response family has one (e.g. a Gaussian variance).
    """

    response_family: ProbabilityDistribution
    natural_fn: Any
    dispersion: Any = None
    engine: Any = NUMPY_ENGINE

    def _spec(self) -> ExponentialFamilySpec:
        declaration = declaration_for(self.response_family)
        if declaration is None or declaration.exponential_family is None:
            raise TypeError("%s is not an exponential family." % type(self.response_family).__name__)
        return declaration.exponential_family

    def natural_parameters(self, x: Any) -> np.ndarray:
        """Return the per-row natural parameters ``eta(x)``."""
        return np.asarray(self.engine.to_numpy(self.natural_fn(x)), dtype=np.float64)

    def sufficient_statistics(self, y: Any) -> Any:
        """Return the response sufficient statistics ``T(y)``."""
        return self.response_family.to_exponential_family(engine=self.engine).sufficient_statistics(y)

    def log_base_measure(self, y: Any) -> Any:
        """Return ``log h(y)`` for the response family."""
        return self.response_family.to_exponential_family(engine=self.engine).log_base_measure(y)

    def log_partition(self, eta: Any) -> np.ndarray:
        """Return ``A(eta)`` row-wise for a stack of natural parameters."""
        eta_arr = np.atleast_2d(np.asarray(eta, dtype=np.float64))
        out = np.empty(eta_arr.shape[0], dtype=np.float64)
        form = self.response_family.to_exponential_family(engine=self.engine)
        for i in range(eta_arr.shape[0]):
            out[i] = float(self.engine.to_numpy(form.log_partition(eta_arr[i])))
        return out


def to_exponential_family(dist: ProbabilityDistribution, engine: Any = NUMPY_ENGINE) -> ExponentialFamilyForm | None:
    """Return the canonical exponential-family view of ``dist`` or ``None``.

    Mirrors :func:`pysp.utils.fisher.to_fisher`: a thin top-level helper that defers
    to :meth:`ProbabilityDistribution.to_exponential_family`.  Returns ``None`` when
    ``dist`` is not a (single) exponential family.
    """
    hook = getattr(dist, "to_exponential_family", None)
    if callable(hook):
        return hook(engine=engine)
    declaration = declaration_for(dist)
    if declaration is None or declaration.exponential_family is None:
        return None
    return ExponentialFamilyForm(distribution=dist, spec=declaration.exponential_family, engine=engine)


def is_exponential_family(dist: ProbabilityDistribution) -> bool:
    """Return whether ``dist`` exposes a (single) exponential-family canonical form."""
    return to_exponential_family(dist) is not None
