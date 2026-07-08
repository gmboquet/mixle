"""Vine copula: build a high-dimensional dependence structure from a cascade of bivariate PAIR copulas.

A single copula family imposes ONE kind of dependence on every pair of variables at once -- a Gaussian copula
makes every pair tail-independent, a Clayton makes every pair lower-tail dependent. A vine (Bedford & Cooke;
Aas et al. 2009) breaks that straitjacket: it factors a ``d``-dimensional copula density into ``d(d-1)/2``
BIVARIATE pair copulas arranged in a nested set of trees, each free to be a DIFFERENT family with its OWN
parameter. So one edge can be Gaussian, another Clayton (lower-tail), another Gumbel (upper-tail) -- per-edge
dependence, chosen from the data.

This implements the **canonical vine (C-vine)**: tree 1 couples a root variable to every other; tree 2 couples
a second variable to the rest given the root; and so on. Density evaluation and sampling use each pair copula's
``h``-function (the conditional CDF ``h(a | b) = dC/db``) and its inverse, recursively (Aas et al. 2009,
Algorithms 1-2). Estimation is sequential (stepwise MLE): fit tree 1 on the raw uniform scores, transform to
conditional pseudo-observations via the fitted ``h``-functions, fit tree 2 on those, and so on -- selecting the
best pair-copula family per edge by likelihood.

Because a vine IS a copula (a density on ``(0,1)^d``), :class:`CVineCopulaDistribution` is a drop-in dependence
CORE for :class:`~mixle.stats.combinator.copula.CopulaDistribution` -- pair it with arbitrary marginals exactly
like the Gaussian/Clayton/Frank/Gumbel/Student-t cores.

Reference: Aas, Czado, Frigessi & Bakken, "Pair-copula constructions of multiple dependence"
(Insurance: Mathematics and Economics, 2009).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.stats.multivariate._copula_common import (
    BufferedUScoreAccumulatorFactory,
    UScoreEncoder,
    weighted_kendall_tau,
)

_CLIP = 1.0e-10


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), _CLIP, 1.0 - _CLIP)


# --------------------------------------------------------------------- pair copulas
# Each pair copula exposes logpdf(a, b), h(a, b) = P(A <= a | B = b) = dC/db, and h_inv(w, b) (inverse in the
# first argument). ``fit(a, b, w)`` returns a fitted instance; ``family`` names it. All are bivariate.


class IndependencePairCopula:
    family = "independence"

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.zeros(np.broadcast(a, b).shape)

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _clip01(a)

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _clip01(w)

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> IndependencePairCopula:
        return IndependencePairCopula()


class GaussianPairCopula:
    family = "gaussian"

    def __init__(self, rho: float) -> None:
        self.rho = float(np.clip(rho, -0.999, 0.999))

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        za, zb = norm.ppf(_clip01(a)), norm.ppf(_clip01(b))
        r = self.rho
        return -0.5 * np.log(1.0 - r * r) - (r * r * (za * za + zb * zb) - 2.0 * r * za * zb) / (2.0 * (1.0 - r * r))

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        za, zb = norm.ppf(_clip01(a)), norm.ppf(_clip01(b))
        return _clip01(norm.cdf((za - self.rho * zb) / np.sqrt(1.0 - self.rho**2)))

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        zw, zb = norm.ppf(_clip01(w)), norm.ppf(_clip01(b))
        return _clip01(norm.cdf(zw * np.sqrt(1.0 - self.rho**2) + self.rho * zb))

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> GaussianPairCopula:
        za, zb = norm.ppf(_clip01(a)), norm.ppf(_clip01(b))
        wsum = float(w.sum())
        ma, mb = float((w * za).sum() / wsum), float((w * zb).sum() / wsum)
        ca, cb = za - ma, zb - mb
        cov = float((w * ca * cb).sum() / wsum)
        va, vb = float((w * ca * ca).sum() / wsum), float((w * cb * cb).sum() / wsum)
        rho = cov / np.sqrt(max(va * vb, 1e-12))
        return GaussianPairCopula(rho)


class ClaytonPairCopula:
    family = "clayton"

    def __init__(self, theta: float) -> None:
        self.theta = max(float(theta), 1.0e-6)

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b, th = _clip01(a), _clip01(b), self.theta
        s = a ** (-th) + b ** (-th) - 1.0
        return np.log1p(th) - (1.0 + th) * (np.log(a) + np.log(b)) - (2.0 + 1.0 / th) * np.log(s)

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b, th = _clip01(a), _clip01(b), self.theta
        s = a ** (-th) + b ** (-th) - 1.0
        return _clip01(b ** (-th - 1.0) * s ** (-1.0 - 1.0 / th))

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        w, b, th = _clip01(w), _clip01(b), self.theta
        return _clip01(((w ** (-th / (th + 1.0)) - 1.0) * b ** (-th) + 1.0) ** (-1.0 / th))

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> ClaytonPairCopula:
        tau = float(np.clip(weighted_kendall_tau(a, b, w), 1.0e-4, 0.95))
        return ClaytonPairCopula(2.0 * tau / (1.0 - tau))


_FAMILIES = {
    "independence": IndependencePairCopula,
    "gaussian": GaussianPairCopula,
    "clayton": ClaytonPairCopula,
}
_DEFAULT_CANDIDATES = ("independence", "gaussian", "clayton")


def _fit_best_pair(a: np.ndarray, b: np.ndarray, w: np.ndarray, candidates: tuple[str, ...]) -> Any:
    """Fit each candidate family to the edge and keep the highest weighted log-likelihood -- per-edge selection."""
    best, best_ll = None, -np.inf
    for fam in candidates:
        pc = _FAMILIES[fam].fit(a, b, w)
        ll = float(np.dot(w, pc.logpdf(a, b)))
        if ll > best_ll:
            best, best_ll = pc, ll
    return best


# --------------------------------------------------------------------- the C-vine core


class CVineCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """A canonical-vine (C-vine) copula on ``(0,1)^d``: ``d(d-1)/2`` bivariate pair copulas in a tree cascade.

    ``pairs`` maps ``(tree, position)`` -> a fitted pair copula, ``tree`` in ``1..d-1`` and ``position`` in
    ``1..d-tree`` (tree 1 links the root variable to each other; deeper trees link conditionally). Build one by
    hand, or fit with :class:`CVineCopulaEstimator` (which selects a family per edge).
    """

    def __init__(
        self,
        dim: int,
        pairs: dict[tuple[int, int], Any],
        candidates: tuple[str, ...] = _DEFAULT_CANDIDATES,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if int(dim) < 2:
            raise ValueError("CVineCopulaDistribution needs dim >= 2; got %d" % dim)
        self.dim = int(dim)
        self.candidates = tuple(candidates)  # pair-copula families the estimator selects among, per edge
        # any edge not supplied defaults to independence, so CVineCopulaDistribution(d, {}) is a valid
        # (independence) copula -- the sensible prototype CopulaDistribution/optimize start from before fitting.
        self.pairs = {
            (j, i): pairs.get((j, i), IndependencePairCopula())
            for j in range(1, int(dim))
            for i in range(1, int(dim) - j + 1)
        }
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        fams = ",".join(self.pairs[(j, i)].family for j in range(1, self.dim) for i in range(1, self.dim - j + 1))
        return "CVineCopulaDistribution(dim=%d, [%s])" % (self.dim, fams)

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        u = _clip01(u)
        n, d = u.shape
        loglik = np.zeros(n)
        v = {1: [u[:, k] for k in range(d)]}  # tree-1 pseudo-obs = the raw uniform columns
        for j in range(1, d):
            pivot = v[j][0]
            m = len(v[j])
            for i in range(1, m):
                loglik += self.pairs[(j, i)].logpdf(v[j][i], pivot)
            if j < d - 1:
                v[j + 1] = [self.pairs[(j, i)].h(v[j][i], pivot) for i in range(1, m)]
        return loglik

    def sampler(self, seed: int | None = None) -> CVineCopulaSampler:
        return CVineCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> CVineCopulaEstimator:
        return CVineCopulaEstimator(self.dim, candidates=self.candidates, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder()


class CVineCopulaSampler(DistributionSampler):
    """Inverse C-vine sampling (Aas et al. 2009, Algorithm 2): invert independent uniforms through the h-inverses."""

    def __init__(self, dist: CVineCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None) -> np.ndarray:
        n = 1 if size is None else int(size)
        d = self.dist.dim
        p = self.dist.pairs
        w = self.rng.uniform(_CLIP, 1.0 - _CLIP, size=(n, d))
        # vv[i][j] holds the j-th conditional pseudo-obs used when generating variable i (1-indexed, Aas)
        vv = {(i, j): None for i in range(1, d + 1) for j in range(1, d + 1)}
        x = [None] * (d + 1)
        vv[(1, 1)] = w[:, 0]
        x[1] = w[:, 0]
        for i in range(2, d + 1):
            vv[(i, 1)] = w[:, i - 1]
            for k in range(i - 1, 0, -1):
                vv[(i, 1)] = p[(k, i - k)].h_inv(vv[(i, 1)], vv[(k, k)])
            x[i] = vv[(i, 1)]
            if i == d:
                break
            for jj in range(1, i):
                vv[(i, jj + 1)] = p[(jj, i - jj)].h(vv[(i, jj)], vv[(jj, jj)])
        out = np.column_stack([_clip01(x[i]) for i in range(1, d + 1)])
        return out[0] if size is None else out


class CVineCopulaEstimator(ParameterEstimator):
    """Sequential (stepwise) MLE: fit tree by tree on conditional pseudo-observations, best family per edge."""

    def __init__(
        self,
        dim: int,
        candidates: tuple[str, ...] = _DEFAULT_CANDIDATES,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = dim
        self.candidates = tuple(candidates)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BufferedUScoreAccumulatorFactory:
        return BufferedUScoreAccumulatorFactory(self.dim, keys=self.keys)

    def _independence_vine(self) -> CVineCopulaDistribution:
        pairs = {(j, i): IndependencePairCopula() for j in range(1, self.dim) for i in range(1, self.dim - j + 1)}
        return CVineCopulaDistribution(self.dim, pairs, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> CVineCopulaDistribution:
        u, w = suff_stat
        if len(u) < 2:
            return self._independence_vine()
        u = _clip01(u)
        w = np.asarray(w, dtype=np.float64)
        d = self.dim
        pairs: dict[tuple[int, int], Any] = {}
        v = {1: [u[:, k] for k in range(d)]}
        for j in range(1, d):
            pivot = v[j][0]
            m = len(v[j])
            for i in range(1, m):
                pairs[(j, i)] = _fit_best_pair(v[j][i], pivot, w, self.candidates)
            if j < d - 1:
                v[j + 1] = [pairs[(j, i)].h(v[j][i], pivot) for i in range(1, m)]
        return CVineCopulaDistribution(d, pairs, name=self.name, keys=self.keys)
