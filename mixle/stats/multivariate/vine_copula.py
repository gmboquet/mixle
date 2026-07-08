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
from scipy.special import gammaln
from scipy.stats import norm
from scipy.stats import t as _student_t

from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
)
from mixle.stats.multivariate._copula_common import (
    BufferedUScoreAccumulatorFactory,
    UScoreEncoder,
    maximize_1d,
    weighted_kendall_tau,
)

_CLIP = 1.0e-10


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), _CLIP, 1.0 - _CLIP)


def _bisect_h_inv(pc: Any, w: np.ndarray, b: np.ndarray, iters: int = 60) -> np.ndarray:
    """Invert ``h(a | b) = w`` for ``a`` by bisection -- ``h`` is a conditional CDF, monotone increasing in ``a``.

    A robust, family-agnostic inverse for pair copulas whose closed-form ``h_inv`` is fiddly or numerically
    delicate (Frank, Gumbel, Student-t). Vectorized over the batch.
    """
    w = _clip01(w)
    b = _clip01(b)
    lo = np.full_like(w, _CLIP)
    hi = np.full_like(w, 1.0 - _CLIP)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        under = pc.h(mid, b) < w
        lo = np.where(under, mid, lo)
        hi = np.where(under, hi, mid)
    return 0.5 * (lo + hi)


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


class FrankPairCopula:
    family = "frank"
    _MIN_ABS = 1.0e-4

    def __init__(self, theta: float) -> None:
        self.theta = float(theta)

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b, th = _clip01(a), _clip01(b), self.theta
        if abs(th) < self._MIN_ABS:
            return np.zeros(np.broadcast(a, b).shape)
        h1 = 1.0 - np.exp(-th)
        denom = h1 - (1.0 - np.exp(-th * a)) * (1.0 - np.exp(-th * b))
        return np.log(abs(th)) + np.log(abs(h1)) - th * (a + b) - 2.0 * np.log(np.abs(denom))

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b, th = _clip01(a), _clip01(b), self.theta
        if abs(th) < self._MIN_ABS:
            return a
        ea, eb = np.exp(-th * a), np.exp(-th * b)
        num = eb * (ea - 1.0)
        den = (ea - 1.0) * (eb - 1.0) + (np.exp(-th) - 1.0)
        return _clip01(num / den)

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _bisect_h_inv(self, w, b)

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> FrankPairCopula:
        a, b = _clip01(a), _clip01(b)

        def loglik(theta: float) -> float:
            return float(np.dot(w, FrankPairCopula(theta).logpdf(a, b)))

        theta = maximize_1d(loglik, -40.0, 40.0)
        return FrankPairCopula(0.0 if abs(theta) < FrankPairCopula._MIN_ABS else theta)


class GumbelPairCopula:
    family = "gumbel"

    def __init__(self, theta: float) -> None:
        self.theta = max(float(theta), 1.0)

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b, th = _clip01(a), _clip01(b), self.theta
        if th <= 1.0 + 1e-12:
            return np.zeros(np.broadcast(a, b).shape)
        x, y = -np.log(a), -np.log(b)
        sx = x**th + y**th
        A = sx ** (1.0 / th)
        return (
            -A
            + (th - 1.0) * (np.log(x) + np.log(y))
            + (1.0 / th - 2.0) * np.log(sx)
            + np.log(A + th - 1.0)
            - np.log(a)
            - np.log(b)
        )

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b, th = _clip01(a), _clip01(b), self.theta
        if th <= 1.0 + 1e-12:
            return a
        x, y = -np.log(a), -np.log(b)
        A = (x**th + y**th) ** (1.0 / th)
        # dC/db = C * A^{1-th} * y^{th-1} / b, with C = exp(-A)
        return _clip01(np.exp(-A) * A ** (1.0 - th) * y ** (th - 1.0) / b)

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _bisect_h_inv(self, w, b)

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> GumbelPairCopula:
        tau = float(np.clip(weighted_kendall_tau(a, b, w), 1.0e-4, 0.95))
        return GumbelPairCopula(1.0 / (1.0 - tau))


class StudentTPairCopula:
    family = "student_t"
    _NU_GRID = (3.0, 5.0, 8.0, 15.0, 30.0)

    def __init__(self, rho: float, df: float) -> None:
        self.rho = float(np.clip(rho, -0.999, 0.999))
        self.df = float(df)

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        nu, r = self.df, self.rho
        za, zb = _student_t.ppf(_clip01(a), nu), _student_t.ppf(_clip01(b), nu)
        quad = (za * za - 2.0 * r * za * zb + zb * zb) / (1.0 - r * r)
        # bivariate t-copula density = f_mvt2(za,zb;r,nu) / (f_t(za;nu) f_t(zb;nu))
        log_num = gammaln((nu + 2.0) / 2.0) + gammaln(nu / 2.0) - 2.0 * gammaln((nu + 1.0) / 2.0)
        log_num += -0.5 * np.log(1.0 - r * r) - (nu + 2.0) / 2.0 * np.log1p(quad / nu)
        log_den = -(nu + 1.0) / 2.0 * (np.log1p(za * za / nu) + np.log1p(zb * zb / nu))
        return log_num - log_den

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        nu, r = self.df, self.rho
        za, zb = _student_t.ppf(_clip01(a), nu), _student_t.ppf(_clip01(b), nu)
        arg = (za - r * zb) / np.sqrt((nu + zb * zb) * (1.0 - r * r) / (nu + 1.0))
        return _clip01(_student_t.cdf(arg, nu + 1.0))

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _bisect_h_inv(self, w, b)

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> StudentTPairCopula:
        rho = float(np.sin(np.pi * weighted_kendall_tau(a, b, w) / 2.0))  # elliptical tau-to-rho
        best, best_ll = None, -np.inf
        for nu in StudentTPairCopula._NU_GRID:  # profile df on a small grid
            cand = StudentTPairCopula(rho, nu)
            ll = float(np.dot(w, cand.logpdf(a, b)))
            if ll > best_ll:
                best, best_ll = cand, ll
        return best


_FAMILIES = {
    "independence": IndependencePairCopula,
    "gaussian": GaussianPairCopula,
    "clayton": ClaytonPairCopula,
    "frank": FrankPairCopula,
    "gumbel": GumbelPairCopula,
    "student_t": StudentTPairCopula,
}
_DEFAULT_CANDIDATES = ("independence", "gaussian", "clayton", "frank", "gumbel", "student_t")


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


# --------------------------------------------------------------------- the D-vine core


def _dvine_walk(
    u: np.ndarray, w: np.ndarray | None, pairs: dict[tuple[int, int], Any] | None, candidates: tuple[str, ...]
) -> tuple[np.ndarray, dict[tuple[int, int], Any]]:
    """One pass of the D-vine cascade (Aas et al. 2009, Algorithm 4), shared by scoring and fitting.

    If ``pairs`` is given, score with them; if ``None``, FIT the best family per edge (needs ``w``). Returns
    the per-observation log-density and the pair-copula dict. The h-function ``h(a, b) = F(a | b)`` conditions
    on its second argument; every pair copula here is exchangeable, so ``h(b, a)`` gives ``F(b | a)``.
    """
    d = u.shape[1]
    fit_mode = pairs is None
    p: dict[tuple[int, int], Any] = {} if fit_mode else pairs
    loglik = np.zeros(u.shape[0])
    v: dict[tuple[int, int], np.ndarray] = {(0, i): u[:, i - 1] for i in range(1, d + 1)}

    def edge(j: int, i: int, a: np.ndarray, b: np.ndarray) -> None:
        nonlocal loglik
        pc = _fit_best_pair(a, b, w, candidates) if fit_mode else p[(j, i)]
        if fit_mode:
            p[(j, i)] = pc
        loglik = loglik + pc.logpdf(a, b)

    for i in range(1, d):  # tree 1: consecutive pairs
        edge(1, i, v[(0, i)], v[(0, i + 1)])
    v[(1, 1)] = p[(1, 1)].h(v[(0, 1)], v[(0, 2)])
    if d > 2:
        for k in range(1, d - 2):
            v[(1, 2 * k)] = p[(1, k + 1)].h(v[(0, k + 2)], v[(0, k + 1)])
            v[(1, 2 * k + 1)] = p[(1, k + 1)].h(v[(0, k + 1)], v[(0, k + 2)])
        v[(1, 2 * d - 4)] = p[(1, d - 1)].h(v[(0, d)], v[(0, d - 1)])
    for j in range(2, d):  # deeper trees on the h-transformed pseudo-observations
        for i in range(1, d - j + 1):
            edge(j, i, v[(j - 1, 2 * i - 1)], v[(j - 1, 2 * i)])
        if j == d - 1:
            break
        v[(j, 1)] = p[(j, 1)].h(v[(j - 1, 1)], v[(j - 1, 2)])
        if d > j + 2:
            for i in range(1, d - j - 1):
                v[(j, 2 * i)] = p[(j, i + 1)].h(v[(j - 1, 2 * i + 2)], v[(j - 1, 2 * i + 1)])
                v[(j, 2 * i + 1)] = p[(j, i + 1)].h(v[(j - 1, 2 * i + 1)], v[(j - 1, 2 * i + 2)])
        v[(j, 2 * (d - j) - 2)] = p[(j, d - j)].h(v[(j - 1, 2 * (d - j))], v[(j - 1, 2 * (d - j) - 1)])
    return loglik, p


class DVineCopulaDistribution(SequenceEncodableProbabilityDistribution):
    """A drawable-vine (D-vine) copula on ``(0,1)^d``: the second canonical vine, a PATH of pair copulas.

    Where a C-vine has a star at each tree (one root linked to all), a D-vine has a path: tree 1 couples
    consecutive variables ``(1,2),(2,3),...,(d-1,d)``; deeper trees couple ``(i, i+j)`` given the variables
    between them. Same ``d(d-1)/2`` pair copulas indexed ``(tree, position)``; different (path vs star) tree
    topology. Like the C-vine it is a drop-in dependence core for
    :class:`~mixle.stats.combinator.copula.CopulaDistribution`.
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
            raise ValueError("DVineCopulaDistribution needs dim >= 2; got %d" % dim)
        self.dim = int(dim)
        self.candidates = tuple(candidates)
        self.pairs = {
            (j, i): pairs.get((j, i), IndependencePairCopula())
            for j in range(1, int(dim))
            for i in range(1, int(dim) - j + 1)
        }
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        fams = ",".join(self.pairs[(j, i)].family for j in range(1, self.dim) for i in range(1, self.dim - j + 1))
        return "DVineCopulaDistribution(dim=%d, [%s])" % (self.dim, fams)

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        return _dvine_walk(_clip01(u), None, self.pairs, self.candidates)[0]

    def sampler(self, seed: int | None = None) -> DVineCopulaSampler:
        return DVineCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> DVineCopulaEstimator:
        return DVineCopulaEstimator(self.dim, candidates=self.candidates, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder()


class DVineCopulaSampler(DistributionSampler):
    """Inverse D-vine sampling (Aas et al. 2009, Algorithm 5): invert independent uniforms tree by tree."""

    def __init__(self, dist: DVineCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None) -> np.ndarray:
        n = 1 if size is None else int(size)
        d = self.dist.dim
        p = self.dist.pairs
        w = self.rng.uniform(_CLIP, 1.0 - _CLIP, size=(n, d))
        v: dict[tuple[int, int], np.ndarray] = {}
        x = [None] * (d + 1)
        v[(1, 1)] = w[:, 0]
        x[1] = w[:, 0]
        if d >= 2:
            x[2] = p[(1, 1)].h_inv(w[:, 1], v[(1, 1)])
            v[(2, 1)] = x[2]
            v[(2, 2)] = p[(1, 1)].h(v[(1, 1)], v[(2, 1)])
        for i in range(3, d + 1):
            v[(i, 1)] = w[:, i - 1]
            for k in range(i - 1, 1, -1):
                v[(i, 1)] = p[(k, i - k)].h_inv(v[(i, 1)], v[(i - 1, 2 * k - 2)])
            v[(i, 1)] = p[(1, i - 1)].h_inv(v[(i, 1)], v[(i - 1, 1)])
            x[i] = v[(i, 1)]
            if i == d:
                break
            v[(i, 2)] = p[(1, i - 1)].h(v[(i - 1, 1)], v[(i, 1)])
            v[(i, 3)] = p[(1, i - 1)].h(v[(i, 1)], v[(i - 1, 1)])
            if i > 3:
                for k in range(2, i - 1):
                    v[(i, 2 * k)] = p[(k, i - k)].h(v[(i - 1, 2 * k - 2)], v[(i, 2 * k - 1)])
                    v[(i, 2 * k + 1)] = p[(k, i - k)].h(v[(i, 2 * k - 1)], v[(i - 1, 2 * k - 2)])
            v[(i, 2 * i - 2)] = p[(i - 1, 1)].h(v[(i - 1, 2 * i - 4)], v[(i, 2 * i - 3)])
        out = np.column_stack([_clip01(x[i]) for i in range(1, d + 1)])
        return out[0] if size is None else out


class DVineCopulaEstimator(ParameterEstimator):
    """Sequential (stepwise) MLE for a D-vine: fit tree by tree on conditional pseudo-obs, best family per edge."""

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

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> DVineCopulaDistribution:
        u, w = suff_stat
        if len(u) < 2:
            return DVineCopulaDistribution(self.dim, {}, name=self.name, keys=self.keys)
        _, pairs = _dvine_walk(_clip01(u), np.asarray(w, dtype=np.float64), None, self.candidates)
        return DVineCopulaDistribution(self.dim, pairs, name=self.name, keys=self.keys)
