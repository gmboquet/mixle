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

from typing import Any

import numpy as np

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.utils.special import digamma, gammaln


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
        return self.alpha

    def set_parameters(self, params: dict[Any, float] | float) -> None:
        """Set the concentration parameters.

        Args:
            params: Dict {value: a_k} of positive reals, or a positive scalar for a symmetric
                Dirichlet of unspecified dimension.

        """
        if isinstance(params, (float, int)) and not isinstance(params, bool):
            self.alpha = float(params)
            self.is_unbounded = True
        else:
            self.alpha = params
            self.is_unbounded = False

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
        if self.is_unbounded:
            a = self.alpha
            n = len(x)
            c = gammaln(a) * n - gammaln(a * n)
            if a == 1:
                return float(-c)
            vals = np.asarray(list(x.values()), dtype=float)
            if np.any(vals < 0.0):
                return float(-np.inf)
            if np.any(vals == 0.0):
                return float(np.inf if a < 1.0 else -np.inf)
            return float(np.sum(np.log(vals)) * (a - 1) - c)
        else:
            rv = 0.0
            asum = 0.0
            saw_pos_inf = False
            saw_neg_inf = False
            for k, v in x.items():
                a = self.alpha[k]
                asum += a
                if v < 0.0:
                    return float(-np.inf)
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
            if self.is_unbounded and not dist.is_unbounded:
                aa = np.asarray(list(dist.alpha.values()))
                a = self.alpha * np.ones(len(aa))
            elif not self.is_unbounded and dist.is_unbounded:
                a = np.asarray(list(self.alpha.values()))
                aa = dist.alpha * np.ones(len(a))
            else:
                keys = list(self.alpha.keys())
                a = np.asarray([self.alpha.get(k) for k in keys])
                aa = np.asarray([dist.alpha.get(k, 0.0) for k in keys])

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
        a = np.asarray(list(self.alpha.values()))
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

    def sample(self, size: int | None = None) -> dict | list[dict]:
        """Draw Dirichlet-distributed probability maps over the alpha keys (dict alpha only)."""
        if self.dist.is_unbounded:
            raise ValueError(
                "DictDirichletSampler cannot sample from a DictDirichletDistribution with scalar alpha "
                "(unspecified dimension)."
            )
        keys = list(self.dist.alpha.keys())
        alpha = np.asarray([self.dist.alpha[k] for k in keys], dtype=float)
        if size is None:
            return dict(zip(keys, self.rng.dirichlet(alpha)))
        else:
            return [dict(zip(keys, p)) for p in self.rng.dirichlet(alpha, size=size)]


class DictDirichletDataEncoder(DataSequenceEncoder):
    """Pass-through encoder for sequences of probability maps."""

    def __str__(self) -> str:
        return "DictDirichletDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DictDirichletDataEncoder)

    def seq_encode(self, x: Any) -> list:
        """Encode dictionary-Dirichlet observations as a list payload."""
        return list(x)
