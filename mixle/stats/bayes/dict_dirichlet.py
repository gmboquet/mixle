"""Dirichlet distribution on dictionary-valued probability maps.

Observations are dicts {value: probability} whose probabilities are non-negative and sum to one
(points on the simplex indexed by the dict keys). A DictDirichletDistribution with concentration
parameters alpha = {k: a_k} has log-density

    log f(x; alpha) = gammaln(sum_k a_k) + sum_k [(a_k - 1)*log(x_k) - gammaln(a_k)].

A single scalar alpha is treated as a symmetric Dirichlet whose dimension is inferred from each
observation (is_unbounded).

This is the conjugate prior used by :class:`~mixle.stats.univariate.discrete.categorical.CategoricalDistribution` (see its
``prior=`` argument). It is a parameter prior: it is scored on probability maps, not fit from data by
EM. Ported from mixle.bstats.catdirichlet.
"""

import operator
from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.utils.special import digamma, gammaln

# Tolerance for the "do these probabilities sum to one" simplex check -- numpy's own np.isclose
# default (rtol=1e-05, atol=1e-08), not a bespoke float64-tuned bound copied from a different
# module. This is CategoricalDistribution's conjugate prior (see its ``prior=`` argument), and a
# fitted pmap reaching it is not guaranteed to be float64 precision (e.g. a gradient-fit categorical
# routed through a lower-precision torch dtype); a naive 1e-10/1e-12 bound would reject a
# legitimately fitted, merely float32-precision map as off-simplex (confirmed by the identical
# failure mode against SymmetricDirichletDistribution's own row prior in
# integer_markov_chain_prior_test.py, whose float32 cond_dist rows are exactly this case), while
# this tolerance is still four-plus orders of magnitude tighter than any sum that would indicate a
# genuinely invalid input.
_SIMPLEX_SUM_RTOL = 1.0e-5
_SIMPLEX_SUM_ATOL = 1.0e-8


class UnspecifiedDirichletDimensionError(ValueError):
    """Raised when an information operation needs the dimension of a scalar-alpha law."""


class DictDirichletDistribution(SequenceEncodableProbabilityDistribution):
    """Dirichlet distribution over probability maps keyed by arbitrary values; a scalar alpha denotes
    a symmetric Dirichlet of unspecified dimension."""

    def __init__(self, alpha: dict[Any, float] | float, name: str | None = None) -> None:
        """Create a Dirichlet distribution over keyed probability maps.

        Args:
            alpha: Concentration parameters. Either a dict {value: a_k} of positive reals or a single
                positive scalar (symmetric Dirichlet, dimension inferred from each observation).
            name (Optional[str]): Optional distribution name.

        """
        self.name = name
        self.set_parameters(alpha)

    def __str__(self) -> str:
        return "DictDirichletDistribution(%s, name=%s)" % (str(self.alpha), repr(self.name))

    def get_parameters(self) -> dict | float:
        """Returns the concentration parameters (dict, or scalar if unbounded)."""
        return self.alpha if self.is_unbounded else dict(self.alpha)

    def set_parameters(self, params: dict[Any, float] | float) -> None:
        """Set the concentration parameters.

        Args:
            params: Dict {value: a_k} of positive reals, or a positive scalar for a symmetric
                Dirichlet of unspecified dimension.

        """
        is_scalar = np.isscalar(params) or (isinstance(params, np.ndarray) and params.ndim == 0)
        if is_scalar and not isinstance(params, (bool, np.bool_)):
            try:
                a = float(params)
            except (TypeError, ValueError) as exc:
                raise ValueError("DictDirichletDistribution requires a positive finite concentration alpha.") from exc
            if not np.isfinite(a) or a <= 0.0:
                raise ValueError("DictDirichletDistribution requires a positive finite concentration alpha.")
            self.alpha = a
            self.is_unbounded = True
        else:
            if not isinstance(params, dict):
                raise TypeError("DictDirichletDistribution alpha must be a positive scalar or non-empty dictionary.")
            try:
                vals = np.asarray(list(params.values()), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("DictDirichletDistribution concentration values must be numeric.") from exc
            if vals.size == 0 or not np.all(np.isfinite(vals)) or not np.all(vals > 0.0):
                raise ValueError(
                    "DictDirichletDistribution requires a non-empty dict of positive finite concentration values."
                )
            self.alpha = {key: float(value) for key, value in params.items()}
            self.is_unbounded = False

    def _validated_state(self) -> tuple[dict[Any, float] | float, bool]:
        alpha = self.get_parameters()
        probe = object.__new__(DictDirichletDistribution)
        probe.set_parameters(alpha)
        return probe.alpha, probe.is_unbounded

    def density(self, x: dict[Any, float]) -> float:
        """Density at the probability map x (exp of log_density)."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: dict[Any, float]) -> float:
        """Log-density of the Dirichlet at the probability map x.

        With scalar alpha the dimension is len(x); with dict alpha the observation is scored with the
        concentration entries matching its keys.
        """
        # Boundary handling mirrors the array Dirichlet (dirichlet.py): a zero coordinate makes the
        # density +inf when its alpha < 1 (integrable singularity) and 0 (log -inf) when alpha > 1; an
        # alpha == 1 coordinate contributes nothing there. +inf takes precedence over -inf. Without this,
        # ``log(0) * (alpha - 1)`` silently produced +inf and, with mixed boundaries, +inf + -inf = NaN.
        #
        # A probability map that isn't on the simplex (a negative entry, a non-finite entry, or values
        # that don't sum to one) is validated up front and rejected as -inf -- checked before the
        # alpha == 1 short-circuit below, since a uniform-over-the-simplex density is still only
        # defined ON the simplex, not for an arbitrary input map.
        alpha, is_unbounded = self._validated_state()
        if not isinstance(x, dict):
            return float(-np.inf)
        if is_unbounded:
            a = alpha
            n = len(x)
            if n == 0:
                return float(-np.inf)
            try:
                vals = np.asarray(list(x.values()), dtype=np.float64)
            except (TypeError, ValueError):
                return float(-np.inf)
            if not np.all(np.isfinite(vals)) or np.any(vals < 0.0):
                return float(-np.inf)
            if not np.isclose(float(vals.sum()), 1.0, rtol=_SIMPLEX_SUM_RTOL, atol=_SIMPLEX_SUM_ATOL):
                return float(-np.inf)
            c = gammaln(a) * n - gammaln(a * n)
            if a == 1:
                return float(-c)
            if np.any(vals == 0.0):
                return float(np.inf if a < 1.0 else -np.inf)
            return float(np.sum(np.log(vals)) * (a - 1) - c)
        else:
            if set(x) != set(alpha):
                return float(-np.inf)
            try:
                vals = np.asarray([x[key] for key in alpha], dtype=np.float64)
            except (TypeError, ValueError):
                return float(-np.inf)
            if vals.size == 0 or not np.all(np.isfinite(vals)) or np.any(vals < 0.0):
                return float(-np.inf)
            if not np.isclose(float(vals.sum()), 1.0, rtol=_SIMPLEX_SUM_RTOL, atol=_SIMPLEX_SUM_ATOL):
                return float(-np.inf)

            rv = 0.0
            asum = 0.0
            saw_pos_inf = False
            saw_neg_inf = False
            for k, a in alpha.items():
                v = float(x[k])
                asum += a
                if v == 0.0:
                    if a < 1.0:
                        saw_pos_inf = True
                    elif a > 1.0:
                        saw_neg_inf = True
                    # a == 1 contributes nothing at the boundary
                    continue
                rv += np.log(v) * (a - 1) - gammaln(a)
            if saw_pos_inf:
                return float(np.inf)
            if saw_neg_inf:
                return float(-np.inf)
            return float(rv + gammaln(asum))

    def seq_log_density(self, x: list[dict[Any, float]]) -> np.ndarray:
        """Vectorized log-density at a sequence of probability maps."""
        return np.asarray([self.log_density(u) for u in x], dtype=float)

    def cross_entropy(self, dist: "DictDirichletDistribution") -> float:
        """Cross entropy -E_self[log dist(x)] for a DictDirichlet argument."""
        if isinstance(dist, DictDirichletDistribution):
            self_alpha, self_unbounded = self._validated_state()
            dist_alpha, dist_unbounded = dist._validated_state()
            if self_unbounded and dist_unbounded:
                raise UnspecifiedDirichletDimensionError(
                    "cross_entropy between scalar-alpha dictionary Dirichlet laws requires "
                    "an explicit support dimension."
                )
            if self_unbounded and not dist_unbounded:
                aa = np.asarray(list(dist_alpha.values()), dtype=np.float64)
                a = self_alpha * np.ones(len(aa))
            elif not self_unbounded and dist_unbounded:
                a = np.asarray(list(self_alpha.values()), dtype=np.float64)
                aa = dist_alpha * np.ones(len(a))
            else:
                if set(self_alpha) != set(dist_alpha):
                    raise ValueError("DictDirichletDistribution cross-entropy requires identical dictionary support.")
                keys = list(self_alpha)
                a = np.asarray([self_alpha[k] for k in keys], dtype=np.float64)
                aa = np.asarray([dist_alpha[k] for k in keys], dtype=np.float64)

            return float(
                -((gammaln(np.sum(aa)) - np.sum(gammaln(aa))) + np.dot(digamma(a) - digamma(np.sum(a)), aa - 1))
            )
        else:
            raise NotImplementedError(
                "DictDirichletDistribution.cross_entropy is only implemented for DictDirichlet arguments (got %s)."
                % type(dist).__name__
            )

    def entropy(self) -> float:
        """Returns the differential entropy in nats (dict alpha only)."""
        alpha, is_unbounded = self._validated_state()
        if is_unbounded:
            raise UnspecifiedDirichletDimensionError(
                "entropy for a scalar-alpha dictionary Dirichlet law requires an explicit support dimension."
            )
        a = np.asarray(list(alpha.values()), dtype=np.float64)
        a0 = np.sum(a)
        return float(-((gammaln(a0) - np.sum(gammaln(a))) + np.dot(digamma(a) - digamma(a0), a - 1)))

    def sampler(self, seed: int | None = None) -> "DictDirichletSampler":
        """Returns a DictDirichletSampler for this distribution."""
        return DictDirichletSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ParameterEstimator":
        """DictDirichlet is a parameter prior and is not fit from data by EM."""
        raise NotImplementedError("DictDirichletDistribution is a parameter prior; it has no data estimator.")

    def dist_to_encoder(self) -> "DictDirichletDataEncoder":
        """Returns a DictDirichletDataEncoder for encoding probability maps."""
        return DictDirichletDataEncoder()


class DictDirichletSampler(DistributionSampler):
    """Draws probability maps from a DictDirichletDistribution with dict-valued concentration."""

    def __init__(self, dist: DictDirichletDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> dict | list[dict]:
        """Draw Dirichlet-distributed probability maps over the alpha keys (dict alpha only)."""
        alpha_state, is_unbounded = self.dist._validated_state()
        if is_unbounded:
            raise ValueError(
                "DictDirichletSampler cannot sample from a DictDirichletDistribution with scalar alpha "
                "(unspecified dimension)."
            )
        keys = list(alpha_state)
        alpha = np.asarray([alpha_state[k] for k in keys], dtype=np.float64)
        if size is None:
            return dict(zip(keys, self.rng.dirichlet(alpha)))
        if isinstance(size, (bool, np.bool_)):
            raise TypeError("DictDirichletSampler size must be a non-negative integer.")
        try:
            count = operator.index(size)
        except TypeError as exc:
            raise TypeError("DictDirichletSampler size must be a non-negative integer.") from exc
        if count < 0:
            raise ValueError("DictDirichletSampler size must be non-negative.")
        return [dict(zip(keys, p)) for p in self.rng.dirichlet(alpha, size=count)]


class DictDirichletDataEncoder(DataSequenceEncoder):
    """Pass-through encoder for sequences of probability maps."""

    def __str__(self) -> str:
        return "DictDirichletDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DictDirichletDataEncoder)

    def seq_encode(self, x: Any) -> list:
        """Encode dictionary-Dirichlet observations as a list payload."""
        return list(x)
