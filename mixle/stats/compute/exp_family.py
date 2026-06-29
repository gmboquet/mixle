"""Uniform exponential-family canonical map for mixle distributions.

Every exponential-family (and conditional-exponential-family) model is expressed in
the canonical form

    p(x)   = h(x) * exp( <eta, T(x)> - A(eta) )            # unconditional
    p(y|x) = h(y) * exp( <eta(x), T(y)> - A(eta(x)) )      # conditional

This module surfaces that form as a first-class object.  The per-family math already
lives in each distribution's :class:`~mixle.stats.compute.declarations.ExponentialFamilySpec`
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

from mixle.engines import NUMPY_ENGINE
from mixle.stats.compute.declarations import (
    ExponentialFamilySpec,
    _generated_exp_family_scalar_expression,
    declaration_for,
)
from mixle.stats.compute.pdist import ProbabilityDistribution


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

    def fisher_information(self, n_samples: int = 200000, seed: int | None = 0) -> np.ndarray:
        """Return the Fisher information in natural coordinates, ``I(eta) = Cov[T(x)] = grad^2 A(eta)``.

        For an exponential family the Fisher information with respect to the natural parameters is
        exactly the covariance of the sufficient statistic (equivalently the Hessian of the
        log-partition). This is the second-order companion to :meth:`mean_parameters` (``grad A =
        E[T]``); it is estimated by the sample covariance of ``T(x)`` over ``n_samples`` draws --
        approximate, but universal for any samplable family -- and returned as a ``(dim, dim)``
        symmetric positive-semidefinite matrix.
        """
        samples = self.distribution.sampler(seed).sample(int(n_samples))
        stats = np.asarray(self.engine.to_numpy(self.sufficient_statistics(samples)), dtype=np.float64)
        cov = np.cov(stats, rowvar=False)
        return np.asarray(cov, dtype=np.float64).reshape(self.dim, self.dim)

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
class ProductExponentialFamilyForm:
    """Canonical exponential-family view of an independent product of distributions.

    A product of exponential families is itself an exponential family with
    ``eta = concat(eta_i)``, ``T(x) = concat(T_i(x_i))``, ``A = sum A_i``, and
    ``log h(x) = sum log h_i(x_i)`` (the closure rule for an independent product).
    Used for :class:`~mixle.stats.combinator.composite.CompositeDistribution`.
    """

    distribution: ProbabilityDistribution
    components: tuple[ExponentialFamilyForm, ...]
    engine: Any = NUMPY_ENGINE
    extract: Any = None  # callable mapping batch x -> tuple(per-component batch)

    def _split(self, x: Any) -> tuple[Any, ...]:
        if self.extract is not None:
            return tuple(self.extract(x))
        rows = list(x)
        return tuple([row[i] for row in rows] for i in range(len(self.components)))

    @property
    def dim(self) -> int:
        """Total natural-parameter dimension (sum over components)."""
        return int(sum(c.dim for c in self.components))

    def natural_parameters(self) -> np.ndarray:
        """Return the concatenated natural parameters of all components."""
        return np.concatenate([c.natural_parameters() for c in self.components])

    def sufficient_statistics(self, x: Any) -> np.ndarray:
        """Return concatenated per-component sufficient statistics ``(n, dim)``."""
        parts = self._split(x)
        blocks = [
            np.asarray(self.engine.to_numpy(c.sufficient_statistics(part)), dtype=np.float64)
            for c, part in zip(self.components, parts)
        ]
        return np.concatenate(blocks, axis=1)

    def log_partition(self, eta: Any = None) -> Any:
        """Return ``A = sum_i A_i`` (current parameters only; ``eta`` override unsupported)."""
        if eta is not None:
            raise NotImplementedError("ProductExponentialFamilyForm.log_partition(eta) is unsupported.")
        return sum(float(self.engine.to_numpy(c.log_partition())) for c in self.components)

    def log_base_measure(self, x: Any) -> np.ndarray:
        """Return ``log h(x) = sum_i log h_i(x_i)`` row-wise."""
        parts = self._split(x)
        total = None
        for c, part in zip(self.components, parts):
            lh = np.asarray(self.engine.to_numpy(c.log_base_measure(part)), dtype=np.float64)
            total = lh if total is None else total + lh
        return total

    def log_density(self, x: Any) -> np.ndarray:
        """Return the reconstructed log-density ``sum_i log p_i(x_i)`` row-wise."""
        parts = self._split(x)
        total = None
        for c, part in zip(self.components, parts):
            lp = np.asarray(self.engine.to_numpy(c.log_density(part)), dtype=np.float64)
            total = lp if total is None else total + lp
        return total

    def mean_parameters(self, **kwargs: Any) -> np.ndarray:
        """Return the concatenated mean parameters of all components."""
        return np.concatenate([c.mean_parameters(**kwargs) for c in self.components])


@dataclass(frozen=True)
class IIDExponentialFamilyForm:
    """Canonical exponential-family view of an iid sequence of a fixed leaf family.

    For a fixed-length-agnostic iid sequence the joint sufficient statistic is the
    sum of the per-element statistics, ``T(x) = sum_t T_0(x_t)``, the natural
    parameters are shared (``eta = eta_0``), and ``A`` / ``log h`` scale with the
    element count.  Used for :class:`~mixle.stats.combinator.sequence.SequenceDistribution`
    when the length is not separately modeled.
    """

    distribution: ProbabilityDistribution
    element: ExponentialFamilyForm
    engine: Any = NUMPY_ENGINE

    @property
    def dim(self) -> int:
        """Natural-parameter dimension (same as the element family)."""
        return self.element.dim

    def natural_parameters(self) -> np.ndarray:
        """Return the shared element natural parameters."""
        return self.element.natural_parameters()

    def sufficient_statistics(self, x: Any) -> np.ndarray:
        """Return per-sequence summed element statistics ``(n, dim)``."""
        rows = []
        for seq in x:
            t = np.asarray(self.engine.to_numpy(self.element.sufficient_statistics(list(seq))), dtype=np.float64)
            rows.append(t.sum(axis=0) if t.shape[0] else np.zeros(self.element.dim))
        return np.asarray(rows, dtype=np.float64)

    def log_partition(self, eta: Any = None) -> Any:
        """Return the per-element ``A`` (the joint scales by the element count)."""
        if eta is not None:
            raise NotImplementedError("IIDExponentialFamilyForm.log_partition(eta) is unsupported.")
        return self.element.log_partition()

    def log_density(self, x: Any) -> np.ndarray:
        """Return the reconstructed iid log-density ``sum_t log p_0(x_t)`` per sequence."""
        out = []
        for seq in x:
            seq = list(seq)
            if not seq:
                out.append(0.0)
                continue
            out.append(float(np.sum(self.engine.to_numpy(self.element.log_density(seq)))))
        return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class MultinomialExponentialFamilyForm:
    """Canonical exponential-family view of a multinomial over an exp-family element.

    A multinomial observation is a bag ``{(v_j, c_j)}`` of values with counts, and the (non
    length-normalized, no separate trial distribution) log-density is the count-weighted sum of the
    element log-densities, ``sum_j c_j log p_0(v_j)``.  So the natural parameters are the element's
    (``eta = eta_0``), the sufficient statistic is the count-weighted sum ``T(x) = sum_j c_j T_0(v_j)``,
    ``log h(x) = sum_j c_j log h_0(v_j)``, and ``A`` is the element's per-trial partition (the joint
    scales by the total count ``n = sum_j c_j``).  Built by
    :meth:`~mixle.stats.multivariate.categorical_multinomial.MultinomialDistribution.to_exponential_family`.
    """

    distribution: ProbabilityDistribution
    element: ExponentialFamilyForm
    engine: Any = NUMPY_ENGINE

    @property
    def dim(self) -> int:
        """Natural-parameter dimension (same as the element family)."""
        return self.element.dim

    def natural_parameters(self) -> np.ndarray:
        """Return the shared element natural parameters."""
        return self.element.natural_parameters()

    @staticmethod
    def _values_counts(obs: Any) -> tuple[list, np.ndarray]:
        pairs = list(obs)
        values = [vc[0] for vc in pairs]
        counts = np.asarray([float(vc[1]) for vc in pairs], dtype=np.float64)
        return values, counts

    def sufficient_statistics(self, x: Any) -> np.ndarray:
        """Return the per-observation count-weighted element statistics ``(n, dim)``."""
        rows = []
        for obs in x:
            values, counts = self._values_counts(obs)
            if not values:
                rows.append(np.zeros(self.element.dim, dtype=np.float64))
                continue
            t = np.asarray(self.engine.to_numpy(self.element.sufficient_statistics(values)), dtype=np.float64)
            rows.append((t * counts[:, None]).sum(axis=0))
        return np.asarray(rows, dtype=np.float64)

    def log_partition(self, eta: Any = None) -> Any:
        """Return the per-trial ``A`` (the joint scales by the total count)."""
        if eta is not None:
            raise NotImplementedError("MultinomialExponentialFamilyForm.log_partition(eta) is unsupported.")
        return self.element.log_partition()

    def log_base_measure(self, x: Any) -> np.ndarray:
        """Return ``log h(x) = sum_j c_j log h_0(v_j)`` per observation."""
        out = []
        for obs in x:
            values, counts = self._values_counts(obs)
            if not values:
                out.append(0.0)
                continue
            h = np.asarray(self.engine.to_numpy(self.element.log_base_measure(values)), dtype=np.float64)
            out.append(float(np.dot(counts, h)))
        return np.asarray(out, dtype=np.float64)

    def log_density(self, x: Any) -> np.ndarray:
        """Return the reconstructed log-density ``sum_j c_j log p_0(v_j)`` per observation."""
        out = []
        for obs in x:
            values, counts = self._values_counts(obs)
            if not values:
                out.append(0.0)
                continue
            lp = np.asarray(self.engine.to_numpy(self.element.log_density(values)), dtype=np.float64)
            out.append(float(np.dot(counts, lp)))
        return np.asarray(out, dtype=np.float64)

    def mean_parameters(self, **kwargs: Any) -> np.ndarray:
        """Return the element mean parameters (the per-trial expectation of ``T``)."""
        return self.element.mean_parameters(**kwargs)


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
    log_partition_fn: Any = None  # callable eta -> A(eta) row-wise
    mean_fn: Any = None  # callable x -> E[y|x]
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
        """Return ``A(eta)`` row-wise for a stack of natural parameters.

        Uses the family-specific ``log_partition_fn`` (supplied by the GLM wiring,
        which knows the response family) when present; otherwise falls back to the
        response family's dual map (only available where ``from_natural`` is closed-form).
        """
        eta_arr = np.asarray(eta, dtype=np.float64)
        if self.log_partition_fn is not None:
            return np.asarray(self.engine.to_numpy(self.log_partition_fn(eta_arr)), dtype=np.float64)
        eta_2d = np.atleast_2d(eta_arr)
        out = np.empty(eta_2d.shape[0], dtype=np.float64)
        form = self.response_family.to_exponential_family(engine=self.engine)
        for i in range(eta_2d.shape[0]):
            out[i] = float(self.engine.to_numpy(form.log_partition(eta_2d[i])))
        return out

    def mean(self, x: Any) -> np.ndarray:
        """Return the conditional mean ``E[y|x] = link_inv(eta(x))`` (the inverse link)."""
        if self.mean_fn is None:
            raise NotImplementedError("conditional mean(x) requires a mean_fn.")
        return np.asarray(self.engine.to_numpy(self.mean_fn(x)), dtype=np.float64)

    def log_density(self, y: Any, x: Any) -> np.ndarray:
        """Return ``log h(y) + <eta(x), T(y)> - A(eta(x))`` row-wise."""
        eta = self.natural_parameters(x)
        eta = np.atleast_2d(eta)
        ty = np.asarray(self.engine.to_numpy(self.sufficient_statistics(y)), dtype=np.float64)
        h = np.asarray(self.engine.to_numpy(self.log_base_measure(y)), dtype=np.float64)
        a = self.log_partition(eta)
        inner = np.einsum("ij,ij->i", ty, eta)
        return h + inner - a


def to_exponential_family(dist: ProbabilityDistribution, engine: Any = NUMPY_ENGINE) -> ExponentialFamilyForm | None:
    """Return the canonical exponential-family view of ``dist`` or ``None``.

    Mirrors :func:`mixle.inference.fisher.to_fisher`: a thin top-level helper that defers
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
