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

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp
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
    reject_unsupported_pseudo_count,
    u_score_batch,
    validated_buffered_statistic,
    validated_dimension,
    validated_finite_scalar,
    validated_sample_size,
    validated_weights,
    weighted_kendall_tau,
)

_CLIP = 1.0e-10


def _clip01(x: np.ndarray) -> np.ndarray:
    """Stabilize an internally computed probability, never a public input."""
    return np.clip(np.asarray(x, dtype=np.float64), _CLIP, 1.0 - _CLIP)


def _pair_arguments(a: Any, b: Any, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate and broadcast two public pair-copula probability arguments."""
    try:
        aa, bb = np.broadcast_arrays(
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("%s arguments must be broadcast-compatible numeric arrays" % label) from exc
    if np.any(~np.isfinite(aa)) or np.any(~np.isfinite(bb)):
        raise ValueError("%s arguments must be finite" % label)
    if np.any((aa <= 0.0) | (aa >= 1.0)) or np.any((bb <= 0.0) | (bb >= 1.0)):
        raise ValueError("%s arguments must lie strictly inside (0, 1)" % label)
    return aa, bb


def _pair_fit_arguments(a: Any, b: Any, w: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate aligned one-dimensional pair-copula fitting arrays."""
    aa, bb = _pair_arguments(a, b, label="pair-copula fit")
    if aa.ndim != 1:
        raise ValueError("pair-copula fit arguments must be one-dimensional arrays")
    ww = validated_weights(w, len(aa))
    if len(aa) < 2:
        raise ValueError("pair-copula fit requires at least two observation rows")
    if not float(ww.sum()) > 0.0:
        raise ValueError("pair-copula fit requires positive total observation weight")
    return aa, bb, ww


def _clayton_pair_log_s(a: np.ndarray, b: np.ndarray, theta: float) -> np.ndarray:
    powers = np.stack([-theta * np.log(a), -theta * np.log(b)], axis=-1)
    result = np.empty(a.shape, dtype=np.float64)
    direct = np.max(powers, axis=-1) < 500.0
    result[direct] = np.log1p(np.sum(np.expm1(powers[direct]), axis=-1))
    if np.any(~direct):
        log_power_sum = logsumexp(powers[~direct], axis=-1)
        result[~direct] = log_power_sum + np.log1p(-np.exp(-log_power_sum))
    return result


def _gumbel_pair_terms(
    a: np.ndarray,
    b: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = -np.log(a), -np.log(b)
    log_x, log_y = np.log(x), np.log(y)
    delta = np.abs(log_x - log_y)
    correction = np.log1p(np.exp(-theta * delta))
    log_a = np.maximum(log_x, log_y) + correction / theta
    return log_a, np.exp(log_a), delta, correction, log_x, log_y


def _bisect_h_inv(pc: Any, w: np.ndarray, b: np.ndarray, iters: int = 60) -> np.ndarray:
    """Invert ``h(a | b) = w`` for ``a`` by bisection -- ``h`` is a conditional CDF, monotone increasing in ``a``.

    A robust, family-agnostic inverse for pair copulas whose closed-form ``h_inv`` is fiddly or numerically
    delicate (Frank, Gumbel, Student-t). Vectorized over the batch.
    """
    w, b = _pair_arguments(w, b, label="%s inverse conditional CDF" % pc.family)
    lo = np.full_like(w, _CLIP)
    hi = np.full_like(w, 1.0 - _CLIP)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        under = pc.h(mid, b) < w
        lo = np.where(under, mid, lo)
        hi = np.where(under, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------- pair copulas
# Each pair copula exposes logpdf(a, b) (with the library-wide ``log_density`` verb as an alias),
# h(a, b) = P(A <= a | B = b) = dC/db, and h_inv(w, b) (inverse in the first argument). ``fit(a, b, w)``
# returns a fitted instance; ``family`` names it. All are bivariate.


class IndependencePairCopula:
    family = "independence"

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, _ = _pair_arguments(a, b, label="independence pair-copula density")
        return np.zeros(a.shape)

    def log_density(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Alias of :meth:`logpdf` matching the library-wide ``log_density`` verb."""
        return self.logpdf(a, b)

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, _ = _pair_arguments(a, b, label="independence pair-copula conditional CDF")
        return a

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        w, _ = _pair_arguments(w, b, label="independence pair-copula inverse conditional CDF")
        return w

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> IndependencePairCopula:
        _pair_fit_arguments(a, b, w)
        return IndependencePairCopula()


class GaussianPairCopula:
    family = "gaussian"

    def __init__(self, rho: float) -> None:
        self.rho = validated_finite_scalar(rho, label="Gaussian pair-copula rho")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("Gaussian pair-copula rho must lie strictly between -1 and 1")

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _pair_arguments(a, b, label="Gaussian pair-copula density")
        za, zb = norm.ppf(a), norm.ppf(b)
        r = self.rho
        return -0.5 * np.log(1.0 - r * r) - (r * r * (za * za + zb * zb) - 2.0 * r * za * zb) / (2.0 * (1.0 - r * r))

    def log_density(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Alias of :meth:`logpdf` matching the library-wide ``log_density`` verb."""
        return self.logpdf(a, b)

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _pair_arguments(a, b, label="Gaussian pair-copula conditional CDF")
        za, zb = norm.ppf(a), norm.ppf(b)
        return _clip01(norm.cdf((za - self.rho * zb) / np.sqrt(1.0 - self.rho**2)))

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        w, b = _pair_arguments(w, b, label="Gaussian pair-copula inverse conditional CDF")
        zw, zb = norm.ppf(w), norm.ppf(b)
        return _clip01(norm.cdf(zw * np.sqrt(1.0 - self.rho**2) + self.rho * zb))

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> GaussianPairCopula:
        a, b, w = _pair_fit_arguments(a, b, w)
        za, zb = norm.ppf(a), norm.ppf(b)
        wsum = float(w.sum())
        ma, mb = float((w * za).sum() / wsum), float((w * zb).sum() / wsum)
        ca, cb = za - ma, zb - mb
        cov = float((w * ca * cb).sum() / wsum)
        va, vb = float((w * ca * ca).sum() / wsum), float((w * cb * cb).sum() / wsum)
        rho = cov / np.sqrt(max(va * vb, 1e-12))
        rho = float(np.clip(rho, -1.0 + 1.0e-12, 1.0 - 1.0e-12))
        return GaussianPairCopula(rho)


class ClaytonPairCopula:
    family = "clayton"

    def __init__(self, theta: float) -> None:
        self.theta = validated_finite_scalar(theta, label="Clayton pair-copula theta")
        if self.theta < 0.0:
            raise ValueError("Clayton pair-copula theta must be non-negative")

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _pair_arguments(a, b, label="Clayton pair-copula density")
        th = self.theta
        if th == 0.0:
            return np.zeros(a.shape)
        log_s = _clayton_pair_log_s(a, b, th)
        result = np.log1p(th) - (1.0 + th) * (np.log(a) + np.log(b)) - (2.0 + 1.0 / th) * log_s
        if np.any(np.isnan(result)):
            raise FloatingPointError("Clayton pair-copula density was numerically indeterminate")
        return result

    def log_density(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Alias of :meth:`logpdf` matching the library-wide ``log_density`` verb."""
        return self.logpdf(a, b)

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _pair_arguments(a, b, label="Clayton pair-copula conditional CDF")
        th = self.theta
        if th == 0.0:
            return a
        log_s = _clayton_pair_log_s(a, b, th)
        return _clip01(np.exp((-th - 1.0) * np.log(b) + (-1.0 - 1.0 / th) * log_s))

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        w, b = _pair_arguments(w, b, label="Clayton pair-copula inverse conditional CDF")
        th = self.theta
        if th == 0.0:
            return w
        power = th / (th + 1.0)
        log_term = np.log(np.expm1(-power * np.log(w))) - th * np.log(b)
        log_inside = np.logaddexp(0.0, log_term)
        return _clip01(np.exp(-log_inside / th))

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> ClaytonPairCopula:
        a, b, w = _pair_fit_arguments(a, b, w)
        tau = max(weighted_kendall_tau(a, b, w), 0.0)
        if tau >= 1.0 - 1.0e-12:
            raise ValueError("Clayton pair fit is on the comonotonic boundary")
        return ClaytonPairCopula(2.0 * tau / (1.0 - tau))


class FrankPairCopula:
    family = "frank"
    _MIN_ABS = 1.0e-4

    def __init__(self, theta: float) -> None:
        self.theta = validated_finite_scalar(theta, label="Frank pair-copula theta")

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _pair_arguments(a, b, label="Frank pair-copula density")
        th = self.theta
        if abs(th) < self._MIN_ABS:
            return np.zeros(a.shape)
        q = abs(th)
        if th < 0.0:
            b = 1.0 - b
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        terms = np.stack(
            [np.zeros(a.shape), -q * (hi - lo), -q * hi, -q * (1.0 - lo)],
            axis=-1,
        )
        log_bracket, sign = logsumexp(
            terms,
            b=np.asarray([1.0, 1.0, -1.0, -1.0]),
            axis=-1,
            return_sign=True,
        )
        if np.any(sign <= 0.0):
            raise FloatingPointError("Frank pair-copula density denominator was numerically indeterminate")
        return np.log(q) + np.log(-np.expm1(-q)) - q * (hi - lo) - 2.0 * log_bracket

    def log_density(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Alias of :meth:`logpdf` matching the library-wide ``log_density`` verb."""
        return self.logpdf(a, b)

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _pair_arguments(a, b, label="Frank pair-copula conditional CDF")
        th = self.theta
        if abs(th) < self._MIN_ABS:
            return a
        q = abs(th)
        if th < 0.0:
            b = 1.0 - b
        log_num = -q * b + np.log(-np.expm1(-q * a))
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        terms = np.stack(
            [np.zeros(a.shape), -q * (hi - lo), -q * hi, -q * (1.0 - lo)],
            axis=-1,
        )
        log_bracket, sign = logsumexp(
            terms,
            b=np.asarray([1.0, 1.0, -1.0, -1.0]),
            axis=-1,
            return_sign=True,
        )
        if np.any(sign <= 0.0):
            raise FloatingPointError("Frank pair-copula conditional CDF was numerically indeterminate")
        log_den = -q * lo + log_bracket
        return _clip01(np.exp(log_num - log_den))

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        _pair_arguments(w, b, label="Frank pair-copula inverse conditional CDF")
        return _bisect_h_inv(self, w, b)

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> FrankPairCopula:
        a, b, w = _pair_fit_arguments(a, b, w)

        def loglik(theta: float) -> float:
            return float(np.dot(w, FrankPairCopula(theta).logpdf(a, b)))

        theta = maximize_1d(loglik, -40.0, 40.0)
        return FrankPairCopula(0.0 if abs(theta) < FrankPairCopula._MIN_ABS else theta)


class GumbelPairCopula:
    family = "gumbel"

    def __init__(self, theta: float) -> None:
        self.theta = validated_finite_scalar(theta, label="Gumbel pair-copula theta")
        if self.theta < 1.0:
            raise ValueError("Gumbel pair-copula theta must be at least one")

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _pair_arguments(a, b, label="Gumbel pair-copula density")
        th = self.theta
        if th == 1.0:
            return np.zeros(a.shape)
        log_a, A, delta, correction, log_x, log_y = _gumbel_pair_terms(a, b, th)
        power_terms = -th * delta - 2.0 * correction + log_a - log_x - log_y
        result = -A + power_terms + np.log(A + th - 1.0) - np.log(a) - np.log(b)
        if np.any(np.isnan(result)):
            raise FloatingPointError("Gumbel pair-copula density was numerically indeterminate")
        return result

    def log_density(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Alias of :meth:`logpdf` matching the library-wide ``log_density`` verb."""
        return self.logpdf(a, b)

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _pair_arguments(a, b, label="Gumbel pair-copula conditional CDF")
        th = self.theta
        if th == 1.0:
            return a
        log_a, A, _, _, _, log_y = _gumbel_pair_terms(a, b, th)
        # dC/db = C * A^{1-th} * y^{th-1} / b, with C = exp(-A)
        return _clip01(np.exp(-A + (1.0 - th) * log_a + (th - 1.0) * log_y - np.log(b)))

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        _pair_arguments(w, b, label="Gumbel pair-copula inverse conditional CDF")
        return _bisect_h_inv(self, w, b)

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> GumbelPairCopula:
        a, b, w = _pair_fit_arguments(a, b, w)
        tau = max(weighted_kendall_tau(a, b, w), 0.0)
        if tau >= 1.0 - 1.0e-12:
            raise ValueError("Gumbel pair fit is on the comonotonic boundary")
        return GumbelPairCopula(1.0 / (1.0 - tau))


class StudentTPairCopula:
    family = "student_t"
    _NU_GRID = (3.0, 5.0, 8.0, 15.0, 30.0)

    def __init__(self, rho: float, df: float) -> None:
        self.rho = validated_finite_scalar(rho, label="Student-t pair-copula rho")
        self.df = validated_finite_scalar(df, label="Student-t pair-copula df")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("Student-t pair-copula rho must lie strictly between -1 and 1")
        if not self.df > 0.0:
            raise ValueError("Student-t pair-copula df must be positive")

    def logpdf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _pair_arguments(a, b, label="Student-t pair-copula density")
        nu, r = self.df, self.rho
        za, zb = _student_t.ppf(a, nu), _student_t.ppf(b, nu)
        quad = (za * za - 2.0 * r * za * zb + zb * zb) / (1.0 - r * r)
        # bivariate t-copula density = f_mvt2(za,zb;r,nu) / (f_t(za;nu) f_t(zb;nu))
        log_num = gammaln((nu + 2.0) / 2.0) + gammaln(nu / 2.0) - 2.0 * gammaln((nu + 1.0) / 2.0)
        log_num += -0.5 * np.log(1.0 - r * r) - (nu + 2.0) / 2.0 * np.log1p(quad / nu)
        log_den = -(nu + 1.0) / 2.0 * (np.log1p(za * za / nu) + np.log1p(zb * zb / nu))
        return log_num - log_den

    def log_density(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Alias of :meth:`logpdf` matching the library-wide ``log_density`` verb."""
        return self.logpdf(a, b)

    def h(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = _pair_arguments(a, b, label="Student-t pair-copula conditional CDF")
        nu, r = self.df, self.rho
        za, zb = _student_t.ppf(a, nu), _student_t.ppf(b, nu)
        arg = (za - r * zb) / np.sqrt((nu + zb * zb) * (1.0 - r * r) / (nu + 1.0))
        return _clip01(_student_t.cdf(arg, nu + 1.0))

    def h_inv(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        _pair_arguments(w, b, label="Student-t pair-copula inverse conditional CDF")
        return _bisect_h_inv(self, w, b)

    @staticmethod
    def fit(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> StudentTPairCopula:
        a, b, w = _pair_fit_arguments(a, b, w)
        rho = float(np.sin(np.pi * weighted_kendall_tau(a, b, w) / 2.0))  # elliptical tau-to-rho
        rho = float(np.clip(rho, -1.0 + 1.0e-12, 1.0 - 1.0e-12))
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
_FAMILY_PARAMETER_COUNTS = {
    "independence": 0,
    "gaussian": 1,
    "clayton": 1,
    "frank": 1,
    "gumbel": 1,
    "student_t": 2,
}


class VinePairFitError(RuntimeError):
    """Raised when no requested pair-copula family can be fitted on an edge."""


@dataclass(frozen=True)
class PairCandidateEvidence:
    """One candidate's training evidence and complexity-adjusted AIC value."""

    family: str
    parameter_count: int
    weighted_log_likelihood: float | None
    aic: float | None
    fit_error: str | None


@dataclass(frozen=True)
class PairSelectionReceipt:
    """Auditable per-edge model-selection receipt."""

    schema_version: int
    criterion: str
    edge_context: str
    selected_family: str
    total_weight: float
    evidence: tuple[PairCandidateEvidence, ...]


def _validated_candidates(candidates: Any) -> tuple[str, ...]:
    if isinstance(candidates, (str, bytes)):
        raise TypeError("vine candidates must be a nonempty sequence of family names")
    try:
        result = tuple(candidates)
    except TypeError as exc:
        raise TypeError("vine candidates must be a nonempty sequence of family names") from exc
    if not result:
        raise ValueError("vine candidates must contain at least one pair-copula family")
    if any(not isinstance(family, str) for family in result):
        raise TypeError("vine candidate family names must be strings")
    unknown = sorted(set(result) - set(_FAMILIES))
    if unknown:
        raise ValueError("unknown vine candidate family name(s): %s" % ", ".join(unknown))
    if len(set(result)) != len(result):
        raise ValueError("vine candidate family names must be unique")
    return result


def _validated_pair_map(dim: int, pairs: Any, *, vine_name: str) -> dict[tuple[int, int], Any]:
    if not isinstance(pairs, dict):
        raise TypeError("%s pairs must be a mapping from (tree, position) to pair copulas" % vine_name)
    expected = {(tree, position) for tree in range(1, dim) for position in range(1, dim - tree + 1)}
    extra = set(pairs) - expected
    if extra:
        raise ValueError("%s contains invalid pair edge key(s): %s" % (vine_name, sorted(extra)))
    result: dict[tuple[int, int], Any] = {}
    for edge in sorted(expected):
        pair = pairs.get(edge, IndependencePairCopula())
        if (
            not isinstance(getattr(pair, "family", None), str)
            or not pair.family
            or not callable(getattr(pair, "logpdf", None))
            or not callable(getattr(pair, "h", None))
            or not callable(getattr(pair, "h_inv", None))
        ):
            raise TypeError("%s edge %s is not a compatible pair-copula object" % (vine_name, edge))
        result[edge] = pair
    return result


def _fit_best_pair(
    a: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    candidates: tuple[str, ...],
    *,
    edge_context: str = "unspecified vine edge",
) -> Any:
    """Fit candidates and select the minimum-AIC law with an attached evidence receipt."""
    a, b, w = _pair_fit_arguments(a, b, w)
    candidates = _validated_candidates(candidates)
    best, best_aic = None, np.inf
    evidence: list[PairCandidateEvidence] = []
    for fam in candidates:
        parameter_count = _FAMILY_PARAMETER_COUNTS[fam]
        try:
            pc = _FAMILIES[fam].fit(a, b, w)
            ll = float(np.dot(w, pc.logpdf(a, b)))
            if not np.isfinite(ll):
                raise FloatingPointError("candidate likelihood is not finite")
            aic = 2.0 * parameter_count - 2.0 * ll
            evidence.append(PairCandidateEvidence(fam, parameter_count, ll, aic, None))
            if aic < best_aic:
                best, best_aic = pc, aic
        except (FloatingPointError, OverflowError, ValueError) as exc:
            evidence.append(
                PairCandidateEvidence(
                    fam,
                    parameter_count,
                    None,
                    None,
                    "%s: %s" % (type(exc).__name__, exc),
                )
            )
    if best is None:
        failures = "; ".join("%s=%s" % (item.family, item.fit_error) for item in evidence)
        raise VinePairFitError("all pair-copula fits failed for %s: %s" % (edge_context, failures))
    best.selection_receipt = PairSelectionReceipt(
        schema_version=1,
        criterion="aic",
        edge_context=edge_context,
        selected_family=best.family,
        total_weight=float(w.sum()),
        evidence=tuple(evidence),
    )
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
        self.dim = validated_dimension(dim, label="C-vine copula dimension")
        self.candidates = _validated_candidates(candidates)
        # any edge not supplied defaults to independence, so CVineCopulaDistribution(d, {}) is a valid
        # (independence) copula -- the sensible prototype CopulaDistribution/optimize start from before fitting.
        self.pairs = _validated_pair_map(self.dim, pairs, vine_name="C-vine")
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        fams = ",".join(self.pairs[(j, i)].family for j in range(1, self.dim) for i in range(1, self.dim - j + 1))
        return "CVineCopulaDistribution(dim=%d, [%s])" % (self.dim, fams)

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        u = u_score_batch(u, self.dim)
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
        reject_unsupported_pseudo_count(pseudo_count, family="C-vine copula")
        return CVineCopulaEstimator(self.dim, candidates=self.candidates, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder(self.dim)


class CVineCopulaSampler(DistributionSampler):
    """Inverse C-vine sampling (Aas et al. 2009, Algorithm 2): invert independent uniforms through the h-inverses."""

    def __init__(self, dist: CVineCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        n = 1 if size is None else validated_sample_size(size)
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
        self.dim = validated_dimension(dim, label="C-vine copula dimension")
        self.candidates = _validated_candidates(candidates)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BufferedUScoreAccumulatorFactory:
        return BufferedUScoreAccumulatorFactory(self.dim, keys=self.keys)

    def _independence_vine(self) -> CVineCopulaDistribution:
        pairs = {(j, i): IndependencePairCopula() for j in range(1, self.dim) for i in range(1, self.dim - j + 1)}
        return CVineCopulaDistribution(self.dim, pairs, name=self.name, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> CVineCopulaDistribution:
        u, w = validated_buffered_statistic(suff_stat, self.dim, minimum_rows=2, require_positive_weight=True)
        d = self.dim
        pairs: dict[tuple[int, int], Any] = {}
        v = {1: [u[:, k] for k in range(d)]}
        for j in range(1, d):
            pivot = v[j][0]
            m = len(v[j])
            for i in range(1, m):
                pairs[(j, i)] = _fit_best_pair(
                    v[j][i],
                    pivot,
                    w,
                    self.candidates,
                    edge_context="C-vine tree %d position %d" % (j, i),
                )
            if j < d - 1:
                v[j + 1] = [pairs[(j, i)].h(v[j][i], pivot) for i in range(1, m)]
        return CVineCopulaDistribution(
            d,
            pairs,
            candidates=self.candidates,
            name=self.name,
            keys=self.keys,
        )


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
        pc = (
            _fit_best_pair(
                a,
                b,
                w,
                candidates,
                edge_context="D-vine tree %d position %d" % (j, i),
            )
            if fit_mode
            else p[(j, i)]
        )
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
        self.dim = validated_dimension(dim, label="D-vine copula dimension")
        self.candidates = _validated_candidates(candidates)
        self.pairs = _validated_pair_map(self.dim, pairs, vine_name="D-vine")
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        fams = ",".join(self.pairs[(j, i)].family for j in range(1, self.dim) for i in range(1, self.dim - j + 1))
        return "DVineCopulaDistribution(dim=%d, [%s])" % (self.dim, fams)

    def log_density(self, u: np.ndarray) -> float:
        return float(self.seq_log_density(np.atleast_2d(np.asarray(u, dtype=np.float64)))[0])

    def seq_log_density(self, u: np.ndarray) -> np.ndarray:
        u = u_score_batch(u, self.dim)
        return _dvine_walk(u, None, self.pairs, self.candidates)[0]

    def sampler(self, seed: int | None = None) -> DVineCopulaSampler:
        return DVineCopulaSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> DVineCopulaEstimator:
        reject_unsupported_pseudo_count(pseudo_count, family="D-vine copula")
        return DVineCopulaEstimator(self.dim, candidates=self.candidates, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> UScoreEncoder:
        return UScoreEncoder(self.dim)


class DVineCopulaSampler(DistributionSampler):
    """Inverse D-vine sampling (Aas et al. 2009, Algorithm 5): invert independent uniforms tree by tree."""

    def __init__(self, dist: DVineCopulaDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = np.random.RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        n = 1 if size is None else validated_sample_size(size)
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
        self.dim = validated_dimension(dim, label="D-vine copula dimension")
        self.candidates = _validated_candidates(candidates)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> BufferedUScoreAccumulatorFactory:
        return BufferedUScoreAccumulatorFactory(self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray]) -> DVineCopulaDistribution:
        u, w = validated_buffered_statistic(suff_stat, self.dim, minimum_rows=2, require_positive_weight=True)
        _, pairs = _dvine_walk(u, w, None, self.candidates)
        return DVineCopulaDistribution(
            self.dim,
            pairs,
            candidates=self.candidates,
            name=self.name,
            keys=self.keys,
        )
