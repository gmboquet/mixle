"""Fused numba kernels for mixture estimation over heterogeneous data.

Instead of one vectorized numpy pass per distribution per component (the
seq_* path, which makes K * n_fields full passes over the data with
temporaries), each supported distribution contributes a scalar *row kernel*

    log_density(i, k, params, cols) -> float
    accumulate(i, k, w, cols, stats) -> None

where params/stats are tuples of arrays with a leading component axis (so EM
iterations never trigger recompilation: new parameters are passed as data) and
cols is a tuple of flat columnar arrays produced by one generic encoding pass.
Kernels compose structurally - composites add child kernels, sequences sum a
child kernel over an offset range - and numba inlines the composition into one
fused loop: scoring all K components of an arbitrarily nested model touches
each observation once.

This replaces the per-distribution vectorized encoders and seq_* math for
supported models: the encoder is a single generic pass that flattens leaves
into typed columns, and estimation is

    enc = CompiledMixture(model).encode(data)
    model = compiled.fit(enc, estimator)        # fused EM to convergence

The kernel E-step produces the same sufficient-statistic structures as the
legacy accumulators, so the existing estimator M-steps (pseudo-counts
included) are reused unchanged, and results agree with the legacy seq path to
floating-point tolerance.

Supported distributions: Gaussian, LogGaussian, Gamma, Categorical,
IntegerCategorical, Bernoulli, StudentT, Logistic, Weibull, Rayleigh,
Pareto, Uniform, Poisson, Exponential, Geometric, Binomial, NegativeBinomial,
DiagonalGaussian (vector leaf), Optional (missing-data wrapper around any
supported child), Ignored (fixed wrapped dist, scored but never estimated),
Composite (any nesting), Sequence (variable-length, with optional length
model), and a Mixture over same-structure components at the top level.
Unsupported models should continue to use the seq_* path.

Still excluded: latent-variable families (HMMs, LDA/PLSI, tree models, hidden
association) require per-row dynamic programs that do not fit the stateless
scalar row kernel; heterogeneous mixtures break the identical-structure
requirement; full-covariance MVN and von Mises-Fisher need linear algebra /
Bessel-function evaluations outside the scalar kernels' scope for now.

Performance characteristics (Apple silicon, float64): on flat scalar fields
the fused scalar loops run at rough parity with the legacy vectorized numpy
path (numpy's per-field passes are SIMD-optimized; the kernels make a single
threaded pass). On variable-length sequence data - the heterogeneous case
this library targets - fusion is substantially faster: ~8-9x faster scoring and EM
on topic-model-like mixtures of token sequences, because the legacy path
re-walks the flattened token arrays once per component per field while the
kernel path streams each observation exactly once.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.combinator.ignored import IgnoredDistribution
from mixle.stats.combinator.null_dist import NullDistribution
from mixle.stats.combinator.optional import OptionalDistribution
from mixle.stats.combinator.sequence import SequenceDistribution
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution
from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution
from mixle.stats.univariate.continuous.logistic import LogisticDistribution
from mixle.stats.univariate.continuous.pareto import ParetoDistribution
from mixle.stats.univariate.continuous.rayleigh import RayleighDistribution
from mixle.stats.univariate.continuous.student_t import StudentTDistribution
from mixle.stats.univariate.continuous.uniform import UniformDistribution
from mixle.stats.univariate.continuous.weibull import WeibullDistribution
from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution
from mixle.stats.univariate.discrete.binomial import BinomialDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.geometric import GeometricDistribution
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution
from mixle.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution
from mixle.utils.optional_deps import numba

__all__ = ["CompiledMixture", "build_kernel"]

_NEG_INF = float("-inf")


# ----------------------------------------------------------------------------
# Leaf kernels (module level so numba can cache them; inlined into compositions)
# ----------------------------------------------------------------------------


@numba.njit(inline="always", cache=True)
def _gaussian_ld(i, k, params, cols):
    mu, s2, logc = params
    x = cols[0][i]
    d = x - mu[k]
    return logc[k] - 0.5 * d * d / s2[k]


@numba.njit(inline="always", cache=True)
def _gaussian_acc(i, k, w, params, cols, stats):
    x = cols[0][i]
    stats[0][k] += w * x
    stats[1][k] += w * x * x
    stats[2][k] += w


@numba.njit(inline="always", cache=True)
def _categorical_ld(i, k, params, cols):
    logp, log_default = params
    c = cols[0][i]
    if c < 0:
        return log_default[k]
    return logp[k, c]


@numba.njit(inline="always", cache=True)
def _categorical_acc(i, k, w, params, cols, stats):
    c = cols[0][i]
    if c >= 0:
        stats[0][k, c] += w


@numba.njit(inline="always", cache=True)
def _intrange_ld(i, k, params, cols):
    logp, min_val = params
    idx = cols[0][i] - min_val
    if idx < 0 or idx >= logp.shape[1]:
        return _NEG_INF
    return logp[k, idx]


@numba.njit(inline="always", cache=True)
def _intrange_acc(i, k, w, params, cols, stats):
    idx = cols[0][i] - params[1]
    if 0 <= idx < stats[0].shape[1]:
        stats[0][k, idx] += w


@numba.njit(inline="always", cache=True)
def _bernoulli_ld(i, k, params, cols):
    logp, log1mp = params
    if cols[0][i] >= 0.5:
        return logp[k]
    return log1mp[k]


@numba.njit(inline="always", cache=True)
def _poisson_ld(i, k, params, cols):
    # cols[1] holds lgamma(x+1), precomputed once at encode time
    lam, loglam = params
    x = cols[0][i]
    if x < 0:
        return _NEG_INF
    return x * loglam[k] - lam[k] - cols[1][i]


@numba.njit(inline="always", cache=True)
def _count_sum_acc(i, k, w, params, cols, stats):
    stats[0][k] += w
    stats[1][k] += w * cols[0][i]


@numba.njit(inline="always", cache=True)
def _exponential_ld(i, k, params, cols):
    beta, logbeta = params
    x = cols[0][i]
    if x < 0:
        return _NEG_INF
    return -x / beta[k] - logbeta[k]


@numba.njit(inline="always", cache=True)
def _geometric_ld(i, k, params, cols):
    logp, log1mp = params
    x = cols[0][i]
    return (x - 1.0) * log1mp[k] + logp[k]


@numba.njit(inline="always", cache=True)
def _negative_binomial_ld(i, k, params, cols):
    r, logp, log1mp, loggr = params
    x = cols[0][i]
    if x < 0.0:
        return _NEG_INF
    return math.lgamma(x + r[k]) - loggr[k] - cols[1][i] + r[k] * logp[k] + x * log1mp[k]


@numba.njit(inline="always", cache=True)
def _student_t_ld(i, k, params, cols):
    df, loc, scale, logc = params
    z = (cols[0][i] - loc[k]) / scale[k]
    return logc[k] - 0.5 * (df[k] + 1.0) * math.log1p((z * z) / df[k])


@numba.njit(inline="always", cache=True)
def _logistic_ld(i, k, params, cols):
    loc, scale, logscale = params
    z = (cols[0][i] - loc[k]) / scale[k]
    if z >= 0.0:
        return -logscale[k] - z - 2.0 * math.log1p(math.exp(-z))
    return -logscale[k] + z - 2.0 * math.log1p(math.exp(z))


@numba.njit(inline="always", cache=True)
def _weibull_ld(i, k, params, cols):
    shape, scale, logshape, logscale = params
    x = cols[0][i]
    if x < 0.0:
        return _NEG_INF
    if x == 0.0:
        if shape[k] < 1.0:
            return math.inf
        if shape[k] > 1.0:
            return _NEG_INF
        return -logscale[k]
    z = x / scale[k]
    return logshape[k] - logscale[k] + (shape[k] - 1.0) * (cols[1][i] - logscale[k]) - math.pow(z, shape[k])


@numba.njit(inline="always", cache=True)
def _weibull_acc(i, k, w, params, cols, stats):
    x = cols[0][i]
    stats[0][k] += w * x
    stats[1][k] += w * x * x
    stats[2][k] += w


@numba.njit(inline="always", cache=True)
def _rayleigh_ld(i, k, params, cols):
    sigma2, logsigma2 = params
    x = cols[0][i]
    if x <= 0.0:
        return _NEG_INF
    return cols[2][i] - logsigma2[k] - cols[1][i] / (2.0 * sigma2[k])


@numba.njit(inline="always", cache=True)
def _rayleigh_acc(i, k, w, params, cols, stats):
    stats[0][k] += w
    stats[1][k] += w * cols[1][i]


@numba.njit(inline="always", cache=True)
def _pareto_ld(i, k, params, cols):
    xm, alpha, logxm, logalpha = params
    x = cols[0][i]
    if x < xm[k]:
        return _NEG_INF
    return logalpha[k] + alpha[k] * logxm[k] - (alpha[k] + 1.0) * cols[1][i]


@numba.njit(inline="always", cache=True)
def _pareto_acc(i, k, w, params, cols, stats):
    if w > 0.0:
        stats[0][k] += w
        stats[1][k] += w * cols[1][i]
        if cols[0][i] < stats[2][k]:
            stats[2][k] = cols[0][i]


@numba.njit(inline="always", cache=True)
def _uniform_ld(i, k, params, cols):
    low, high, logp = params
    x = cols[0][i]
    if x < low[k] or x > high[k]:
        return _NEG_INF
    return logp[k]


@numba.njit(inline="always", cache=True)
def _uniform_acc(i, k, w, params, cols, stats):
    if w > 0.0:
        x = cols[0][i]
        stats[0][k] += w
        if x < stats[1][k]:
            stats[1][k] = x
        if x > stats[2][k]:
            stats[2][k] = x


@numba.njit(inline="always", cache=True)
def _gamma_ld(i, k, params, cols):
    # cols[1] holds log(x), precomputed once at encode time
    neg_inv_theta, km1, logc = params
    return cols[0][i] * neg_inv_theta[k] + cols[1][i] * km1[k] + logc[k]


@numba.njit(inline="always", cache=True)
def _gamma_acc(i, k, w, params, cols, stats):
    stats[0][k] += w
    stats[1][k] += w * cols[0][i]
    stats[2][k] += w * cols[1][i]


@numba.njit(inline="always", cache=True)
def _log_gaussian_ld(i, k, params, cols):
    # cols[0] holds log(x); a Gaussian on log x with a -log(x) Jacobian term
    mu, s2, logc = params
    lx = cols[0][i]
    d = lx - mu[k]
    return logc[k] - 0.5 * d * d / s2[k] - lx


@numba.njit(inline="always", cache=True)
def _binomial_ld(i, k, params, cols):
    # cols[1] holds the log-binomial coefficient, precomputed at encode time
    # for the compile-time (n, min_val); when a model carries different values
    # (the estimator re-derives both from data) it is recomputed inline.
    logp, log1mp, nv, mv, n0, m0 = params
    n = nv[k]
    xx = cols[0][i] - mv[k]
    if xx < 0.0 or xx > n:
        return _NEG_INF
    if n == n0 and mv[k] == m0:
        base = cols[1][i]
    else:
        base = math.lgamma(n + 1.0) - math.lgamma(xx + 1.0) - math.lgamma(n - xx + 1.0)
    return base + log1mp[k] * (n - xx) + logp[k] * xx


@numba.njit(inline="always", cache=True)
def _binomial_acc(i, k, w, params, cols, stats):
    # cols[2] holds the data-wide (min, max); accumulating w*const keeps the
    # stats additive across chunked/threaded buffers (recovered by w-sum).
    stats[0][k] += w
    stats[1][k] += w * cols[0][i]
    stats[2][k] += w * cols[2][0]
    stats[3][k] += w * cols[2][1]


@numba.njit(inline="always", cache=True)
def _diag_gaussian_ld(i, k, params, cols):
    # vector leaf: cols[0] is the flattened (n*d) observation matrix
    ca, cb, cc = params
    d = ca.shape[1]
    off = i * d
    rv = cc[k]
    for j in range(d):
        x = cols[0][off + j]
        rv += x * x * ca[k, j] + x * cb[k, j]
    return rv


@numba.njit(inline="always", cache=True)
def _diag_gaussian_acc(i, k, w, params, cols, stats):
    d = stats[0].shape[1]
    off = i * d
    for j in range(d):
        x = cols[0][off + j]
        stats[0][k, j] += w * x
        stats[1][k, j] += w * x * x
    stats[2][k] += w


@numba.njit(inline="always", cache=True)
def _ignored_ld(i, k, params, cols):
    # cols[0] holds per-component log-densities precomputed at encode time
    # (the wrapped dists are fixed: scored but never estimated)
    return cols[0][i, k]


@numba.njit(inline="always", cache=True)
def _ignored_acc(i, k, w, params, cols, stats):
    pass


# ----------------------------------------------------------------------------
# Builders: per-node compile of (kernel, acc kernel, params, stats, encoder)
# ----------------------------------------------------------------------------


class _LeafBuilder:
    dtype = np.float64

    def __init__(self, dists):
        self.dists = dists

    def new_sink(self):
        buf = []
        return buf, buf.append

    def freeze(self, buf):
        return (np.asarray(buf, dtype=self.dtype),)

    def stats_to_ss(self, stats, k):
        raise NotImplementedError

    def merge_stats(self, dst, src):
        # Default: every stat field is additive across chunked buffers.
        # dst/src fields are numpy arrays held in a tuple, so accumulate in
        # place (np.add out=) rather than reassigning the tuple slot.
        for d, s in zip(dst, src):
            np.add(d, s, out=d)


class _GaussianB(_LeafBuilder):
    kernel = staticmethod(_gaussian_ld)
    acc_kernel = staticmethod(_gaussian_acc)

    def params(self, dists):
        mu = np.array([d.mu for d in dists], dtype=np.float64)
        s2 = np.array([d.sigma2 for d in dists], dtype=np.float64)
        logc = -0.5 * np.log(2.0 * np.pi * s2)
        return mu, s2, logc

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K), np.zeros(K)

    def stats_to_ss(self, stats, k):
        return stats[0][k], stats[1][k], stats[2][k], stats[2][k]


class _CategoricalB(_LeafBuilder):
    kernel = staticmethod(_categorical_ld)
    acc_kernel = staticmethod(_categorical_acc)

    def __init__(self, dists):
        super().__init__(dists)
        vocab = list(dict.fromkeys(v for d in dists for v in d.pmap.keys()))
        self.vocab = vocab
        self.codes = {v: c for c, v in enumerate(vocab)}

    def params(self, dists):
        K, V = len(dists), len(self.vocab)
        logp = np.empty((K, V), dtype=np.float64)
        log_default = np.empty(K, dtype=np.float64)
        with np.errstate(divide="ignore"):
            for k, d in enumerate(dists):
                for c, v in enumerate(self.vocab):
                    logp[k, c] = np.log(d.pmap.get(v, d.default_value)) - d.log1p_default_value
                log_default[k] = np.log(d.default_value) - d.log1p_default_value
        return logp, log_default

    def new_sink(self):
        buf = []
        codes = self.codes
        return buf, lambda x: buf.append(codes.get(x, -1))

    def freeze(self, buf):
        return (np.asarray(buf, dtype=np.int64),)

    def make_stats(self, K):
        return (np.zeros((K, len(self.vocab))),)

    def stats_to_ss(self, stats, k):
        row = stats[0][k]
        return {v: row[c] for c, v in enumerate(self.vocab) if row[c] > 0}


class _IntRangeB(_LeafBuilder):
    kernel = staticmethod(_intrange_ld)
    acc_kernel = staticmethod(_intrange_acc)
    dtype = np.int64

    def __init__(self, dists):
        super().__init__(dists)
        self.min_val = int(min(d.min_val for d in dists))
        self.max_val = int(max(d.max_val for d in dists))

    def params(self, dists):
        K = len(dists)
        V = self.max_val - self.min_val + 1
        logp = np.full((K, V), _NEG_INF)
        for k, d in enumerate(dists):
            lo = d.min_val - self.min_val
            logp[k, lo : lo + len(d.log_p_vec)] = d.log_p_vec
        return logp, np.int64(self.min_val)

    def make_stats(self, K):
        V = self.max_val - self.min_val + 1
        return (np.zeros((K, V)),)

    def stats_to_ss(self, stats, k):
        return self.min_val, stats[0][k].copy()


class _CountSumB(_LeafBuilder):
    acc_kernel = staticmethod(_count_sum_acc)

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K)

    def stats_to_ss(self, stats, k):
        return stats[0][k], stats[1][k]


class _BernoulliB(_CountSumB):
    kernel = staticmethod(_bernoulli_ld)

    def params(self, dists):
        return (
            np.array([d.log_p for d in dists], dtype=np.float64),
            np.array([d.log_1p for d in dists], dtype=np.float64),
        )

    def freeze(self, buf):
        x = np.asarray(buf)
        if x.size and np.any((x != 0) & (x != 1)):
            raise Exception("BernoulliDistribution requires observations in {False, True} or {0, 1}.")
        return (x.astype(np.float64),)


class _PoissonB(_CountSumB):
    kernel = staticmethod(_poisson_ld)

    def params(self, dists):
        lam = np.array([d.lam for d in dists], dtype=np.float64)
        return lam, np.log(lam)

    def freeze(self, buf):
        from scipy.special import gammaln

        x = np.asarray(buf, dtype=np.float64)
        return x, gammaln(x + 1.0)


class _ExponentialB(_CountSumB):
    kernel = staticmethod(_exponential_ld)

    def params(self, dists):
        beta = np.array([d.beta for d in dists], dtype=np.float64)
        return beta, np.log(beta)


class _GeometricB(_CountSumB):
    kernel = staticmethod(_geometric_ld)

    def params(self, dists):
        p = np.array([d.p for d in dists], dtype=np.float64)
        return np.log(p), np.log1p(-p)


class _NegativeBinomialB(_CountSumB):
    kernel = staticmethod(_negative_binomial_ld)

    def params(self, dists):
        r = np.array([d.r for d in dists], dtype=np.float64)
        return (
            r,
            np.array([d.log_p for d in dists], dtype=np.float64),
            np.array([d.log_1p for d in dists], dtype=np.float64),
            np.array([d.log_gamma_r for d in dists], dtype=np.float64),
        )

    def freeze(self, buf):
        from scipy.special import gammaln

        x = np.asarray(buf, dtype=np.float64)
        if x.size and (np.any(x < 0) or np.any(np.isnan(x)) or np.any(np.floor(x) != x)):
            raise Exception("NegativeBinomialDistribution requires non-negative integer values for x.")
        return x, gammaln(x + 1.0)


class _StudentTB(_LeafBuilder):
    kernel = staticmethod(_student_t_ld)
    acc_kernel = staticmethod(_gaussian_acc)

    def params(self, dists):
        df = np.array([d.df for d in dists], dtype=np.float64)
        loc = np.array([d.loc for d in dists], dtype=np.float64)
        scale = np.array([d.scale for d in dists], dtype=np.float64)
        logc = np.array([d.log_const for d in dists], dtype=np.float64)
        return df, loc, scale, logc

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K), np.zeros(K)

    def stats_to_ss(self, stats, k):
        return stats[0][k], stats[1][k], stats[2][k]


class _LogisticB(_LeafBuilder):
    kernel = staticmethod(_logistic_ld)
    acc_kernel = staticmethod(_gaussian_acc)

    def params(self, dists):
        loc = np.array([d.loc for d in dists], dtype=np.float64)
        scale = np.array([d.scale for d in dists], dtype=np.float64)
        return loc, scale, np.log(scale)

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K), np.zeros(K)

    def stats_to_ss(self, stats, k):
        return stats[0][k], stats[1][k], stats[2][k]


class _WeibullB(_LeafBuilder):
    kernel = staticmethod(_weibull_ld)
    acc_kernel = staticmethod(_weibull_acc)

    def params(self, dists):
        shape = np.array([d.shape for d in dists], dtype=np.float64)
        scale = np.array([d.scale for d in dists], dtype=np.float64)
        return shape, scale, np.log(shape), np.log(scale)

    def freeze(self, buf):
        x = np.asarray(buf, dtype=np.float64)
        if x.size and (np.any(x < 0.0) or np.any(np.isnan(x))):
            raise Exception("WeibullDistribution requires non-negative values for x.")
        with np.errstate(divide="ignore"):
            lx = np.log(x)
        return x, lx

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K), np.zeros(K)

    def stats_to_ss(self, stats, k):
        return stats[0][k], stats[1][k], stats[2][k]


class _RayleighB(_LeafBuilder):
    kernel = staticmethod(_rayleigh_ld)
    acc_kernel = staticmethod(_rayleigh_acc)

    def params(self, dists):
        sigma2 = np.array([d.sigma2 for d in dists], dtype=np.float64)
        return sigma2, np.log(sigma2)

    def freeze(self, buf):
        x = np.asarray(buf, dtype=np.float64)
        if x.size and (np.any(x < 0.0) or np.any(np.isnan(x))):
            raise Exception("RayleighDistribution requires non-negative values for x.")
        with np.errstate(divide="ignore"):
            lx = np.log(x)
        return x, x * x, lx

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K)

    def stats_to_ss(self, stats, k):
        return stats[0][k], stats[1][k]


class _ParetoB(_LeafBuilder):
    kernel = staticmethod(_pareto_ld)
    acc_kernel = staticmethod(_pareto_acc)

    def params(self, dists):
        xm = np.array([d.xm for d in dists], dtype=np.float64)
        alpha = np.array([d.alpha for d in dists], dtype=np.float64)
        return xm, alpha, np.log(xm), np.log(alpha)

    def freeze(self, buf):
        x = np.asarray(buf, dtype=np.float64)
        if x.size and (np.any(x <= 0.0) or np.any(np.isnan(x))):
            raise Exception("ParetoDistribution requires positive values for x.")
        return x, np.log(x)

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K), np.full(K, np.inf)

    def stats_to_ss(self, stats, k):
        return stats[0][k], stats[1][k], stats[2][k]

    def merge_stats(self, dst, src):
        # stats[2] is the support minimum (init +inf); non-additive.
        np.add(dst[0], src[0], out=dst[0])
        np.add(dst[1], src[1], out=dst[1])
        np.minimum(dst[2], src[2], out=dst[2])


class _UniformB(_LeafBuilder):
    kernel = staticmethod(_uniform_ld)
    acc_kernel = staticmethod(_uniform_acc)

    def params(self, dists):
        low = np.array([d.low for d in dists], dtype=np.float64)
        high = np.array([d.high for d in dists], dtype=np.float64)
        return low, high, -np.log(high - low)

    def make_stats(self, K):
        return np.zeros(K), np.full(K, np.inf), np.full(K, -np.inf)

    def stats_to_ss(self, stats, k):
        return stats[0][k], stats[1][k], stats[2][k]

    def merge_stats(self, dst, src):
        # stats[1] is the support minimum (init +inf), stats[2] the maximum
        # (init -inf); both non-additive.
        np.add(dst[0], src[0], out=dst[0])
        np.minimum(dst[1], src[1], out=dst[1])
        np.maximum(dst[2], src[2], out=dst[2])


class _GammaB(_LeafBuilder):
    kernel = staticmethod(_gamma_ld)
    acc_kernel = staticmethod(_gamma_acc)

    def params(self, dists):
        theta = np.array([d.theta for d in dists], dtype=np.float64)
        km1 = np.array([d.k - 1.0 for d in dists], dtype=np.float64)
        logc = np.array([d.log_const for d in dists], dtype=np.float64)
        return -1.0 / theta, km1, logc

    def freeze(self, buf):
        x = np.asarray(buf, dtype=np.float64)
        if x.size and (np.any(x <= 0) or np.any(np.isnan(x))):
            raise Exception("GammaDistribution has support x > 0.")
        return x, np.log(x)

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K), np.zeros(K)

    def stats_to_ss(self, stats, k):
        # legacy GammaAccumulator.value(): (nobs, sum, sum_of_logs)
        return stats[0][k], stats[1][k], stats[2][k]


class _LogGaussianB(_LeafBuilder):
    kernel = staticmethod(_log_gaussian_ld)
    acc_kernel = staticmethod(_gaussian_acc)  # Gaussian moments of the log-x column

    def params(self, dists):
        mu = np.array([d.mu for d in dists], dtype=np.float64)
        s2 = np.array([d.sigma2 for d in dists], dtype=np.float64)
        logc = np.array([d.log_const for d in dists], dtype=np.float64)
        return mu, s2, logc

    def freeze(self, buf):
        lx = np.log(np.asarray(buf, dtype=np.float64))
        if lx.size and (np.any(np.isnan(lx)) or np.any(np.isinf(lx))):
            raise Exception("LogGaussianDistribution requires support x in (0,inf).")
        return (lx,)

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K), np.zeros(K)

    def stats_to_ss(self, stats, k):
        # legacy LogGaussianAccumulator.value(): (log_sum, log_sum2, count, count2)
        return stats[0][k], stats[1][k], stats[2][k], stats[2][k]


class _BinomialB(_LeafBuilder):
    kernel = staticmethod(_binomial_ld)
    acc_kernel = staticmethod(_binomial_acc)

    def __init__(self, dists):
        super().__init__(dists)
        # n is a fixed per-distribution parameter and min_val shifts the
        # support; the estimator re-derives both from the data min/max, so a
        # later model may not match the compile-time values - the kernel then
        # falls back to inline lgamma instead of the precomputed column.
        ns = {int(d.n) for d in dists}
        ms = {int(d.min_val) if d.min_val is not None else 0 for d in dists}
        if len(ns) == 1 and len(ms) == 1:
            self.n0, self.m0 = float(ns.pop()), float(ms.pop())
        else:
            self.n0, self.m0 = -1.0, -1.0  # sentinel: never matches (n >= 0)

    def params(self, dists):
        logp = np.array([d.log_p for d in dists], dtype=np.float64)
        log1mp = np.array([d.log_1p for d in dists], dtype=np.float64)
        nv = np.array([float(d.n) for d in dists], dtype=np.float64)
        mv = np.array([float(d.min_val) if d.min_val is not None else 0.0 for d in dists], dtype=np.float64)
        return logp, log1mp, nv, mv, np.float64(self.n0), np.float64(self.m0)

    def freeze(self, buf):
        from scipy.special import gammaln

        x = np.asarray(buf, dtype=np.float64)
        if x.size and (np.any(x < 0) or np.any(np.isnan(x))):
            raise Exception("BinomialDistribution requires non-negative integer values for x.")
        if self.n0 >= 0:
            xx = x - self.m0
            lbc = gammaln(self.n0 + 1.0) - gammaln(xx + 1.0) - gammaln(self.n0 - xx + 1.0)
        else:
            lbc = np.zeros_like(x)
        mnmx = np.array([x.min(), x.max()], dtype=np.float64) if x.size else np.zeros(2)
        return x, lbc, mnmx

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)

    def stats_to_ss(self, stats, k):
        # legacy BinomialAccumulator.value(): (count, sum, min_val, max_val);
        # min/max are recovered from the w-weighted constants in stats[2:4].
        # NOTE: value()[2] (min_val) can differ from the legacy accumulator here:
        # this fused path recovers the true observed data minimum, whereas the
        # legacy accumulator reports its factory-seeded 0. Benign with the default
        # BinomialEstimator (min_val is not used in the M-step).
        c = stats[0][k]
        if c > 0:
            return c, stats[1][k], int(round(stats[2][k] / c)), int(round(stats[3][k] / c))
        return c, stats[1][k], None, None


class _DiagGaussianB(_LeafBuilder):
    kernel = staticmethod(_diag_gaussian_ld)
    acc_kernel = staticmethod(_diag_gaussian_acc)

    def __init__(self, dists):
        super().__init__(dists)
        self.dim = int(dists[0].dim)
        if any(int(d.dim) != self.dim for d in dists):
            raise ValueError("DiagonalGaussian components must share the same dimension.")

    def params(self, dists):
        ca = np.array([d.ca for d in dists], dtype=np.float64)
        cb = np.array([d.cb for d in dists], dtype=np.float64)
        cc = np.array([d.cc for d in dists], dtype=np.float64)
        return ca, cb, cc

    def new_sink(self):
        buf = []
        ext = buf.extend
        return buf, lambda x: ext(x)

    def freeze(self, buf):
        return (np.asarray(buf, dtype=np.float64),)

    def make_stats(self, K):
        return np.zeros((K, self.dim)), np.zeros((K, self.dim)), np.zeros(K)

    def stats_to_ss(self, stats, k):
        # legacy DiagonalGaussianAccumulator.value(): (sum, sum2, count)
        return stats[0][k].copy(), stats[1][k].copy(), stats[2][k]


def _pair_kernels(f, g):
    @numba.njit(inline="always")
    def ld(i, k, params, cols):
        return f(i, k, params[0], cols[0]) + g(i, k, params[1], cols[1])

    return ld


def _pair_accs(f, g):
    @numba.njit(inline="always")
    def acc(i, k, w, params, cols, stats):
        f(i, k, w, params[0], cols[0], stats[0])
        g(i, k, w, params[1], cols[1], stats[1])

    return acc


class _CompositeB:
    """Children fused by folding into a binary tree of two-term kernels.

    The params/cols/stats for the node mirror the tree as nested 2-tuples;
    _tree/_untree convert between the flat per-slot lists and that shape.
    """

    def __init__(self, dists):
        self.child_builders = [build_kernel([d.dists[j] for d in dists]) for j in range(len(dists[0].dists))]
        kerns = [b.kernel for b in self.child_builders]
        accs = [b.acc_kernel for b in self.child_builders]
        while len(kerns) > 1:
            kerns = [
                _pair_kernels(kerns[i], kerns[i + 1]) if i + 1 < len(kerns) else kerns[i]
                for i in range(0, len(kerns), 2)
            ]
            accs = [_pair_accs(accs[i], accs[i + 1]) if i + 1 < len(accs) else accs[i] for i in range(0, len(accs), 2)]
        self.kernel = kerns[0]
        self.acc_kernel = accs[0]

    @staticmethod
    def _tree(items):
        items = list(items)
        while len(items) > 1:
            items = [(items[i], items[i + 1]) if i + 1 < len(items) else items[i] for i in range(0, len(items), 2)]
        return items[0]

    @staticmethod
    def _untree(node, m):
        # inverse of _tree for m slots
        shape = _CompositeB._tree(list(range(m)))

        def walk(s, v, out):
            if isinstance(s, tuple):
                walk(s[0], v[0], out)
                walk(s[1], v[1], out)
            else:
                out[s] = v

        out = [None] * m
        walk(shape, node, out)
        return out

    def params(self, dists):
        return self._tree(b.params([d.dists[j] for d in dists]) for j, b in enumerate(self.child_builders))

    def new_sink(self):
        subs = [b.new_sink() for b in self.child_builders]
        adders = [a for _, a in subs]

        def add(x):
            for j, a in enumerate(adders):
                a(x[j])

        return [s for s, _ in subs], add

    def freeze(self, bufs):
        return self._tree(b.freeze(bufs[j]) for j, b in enumerate(self.child_builders))

    def make_stats(self, K):
        return self._tree(b.make_stats(K) for b in self.child_builders)

    def stats_to_ss(self, stats, k):
        flat = self._untree(stats, len(self.child_builders))
        return tuple(b.stats_to_ss(flat[j], k) for j, b in enumerate(self.child_builders))

    def merge_stats(self, dst, src):
        dflat = self._untree(dst, len(self.child_builders))
        sflat = self._untree(src, len(self.child_builders))
        for j, b in enumerate(self.child_builders):
            b.merge_stats(dflat[j], sflat[j])


def _sequence_kernel(inner, len_kern, has_len):
    if has_len:

        @numba.njit(inline="always")
        def ld(i, k, params, cols):
            off, inner_cols, len_cols = cols
            rv = 0.0
            for t in range(off[i], off[i + 1]):
                rv += inner(t, k, params[0], inner_cols)
            return rv + len_kern(i, k, params[1], len_cols)

        return ld

    @numba.njit(inline="always")
    def ld(i, k, params, cols):
        off, inner_cols, len_cols = cols
        rv = 0.0
        for t in range(off[i], off[i + 1]):
            rv += inner(t, k, params[0], inner_cols)
        return rv

    return ld


def _sequence_acc(inner, len_acc, has_len):
    if has_len:

        @numba.njit(inline="always")
        def acc(i, k, w, params, cols, stats):
            off, inner_cols, len_cols = cols
            for t in range(off[i], off[i + 1]):
                inner(t, k, w, params[0], inner_cols, stats[0])
            len_acc(i, k, w, params[1], len_cols, stats[1])

        return acc

    @numba.njit(inline="always")
    def acc(i, k, w, params, cols, stats):
        off, inner_cols, len_cols = cols
        for t in range(off[i], off[i + 1]):
            inner(t, k, w, params[0], inner_cols, stats[0])

    return acc


class _SequenceB:
    def __init__(self, dists):
        if any(getattr(d, "len_normalized", False) for d in dists):
            raise ValueError("kernels do not support len_normalized SequenceDistributions.")
        self.inner = build_kernel([d.dist for d in dists])
        self.has_len = not isinstance(dists[0].len_dist, NullDistribution) and dists[0].len_dist is not None
        if self.has_len:
            self.len_b = build_kernel([d.len_dist for d in dists])
        else:
            self.len_b = _PoissonB([PoissonDistribution(1.0)])  # sentinel; never evaluated
        self.kernel = _sequence_kernel(self.inner.kernel, self.len_b.kernel, self.has_len)
        self.acc_kernel = _sequence_acc(self.inner.acc_kernel, self.len_b.acc_kernel, self.has_len)

    def params(self, dists):
        inner_p = self.inner.params([d.dist for d in dists])
        len_p = self.len_b.params([d.len_dist for d in dists]) if self.has_len else self.len_b.params(self.len_b.dists)
        return inner_p, len_p

    def new_sink(self):
        inner_buf, inner_add = self.inner.new_sink()
        len_buf, len_add = self.len_b.new_sink()
        lens = []

        def add(x):
            n = 0
            for e in x:
                inner_add(e)
                n += 1
            lens.append(n)
            len_add(n)

        return (inner_buf, len_buf, lens), add

    def freeze(self, bufs):
        inner_buf, len_buf, lens = bufs
        off = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(np.asarray(lens, dtype=np.int64), out=off[1:])
        return off, self.inner.freeze(inner_buf), self.len_b.freeze(len_buf)

    def make_stats(self, K):
        return self.inner.make_stats(K), self.len_b.make_stats(K)

    def stats_to_ss(self, stats, k):
        len_ss = self.len_b.stats_to_ss(stats[1], k) if self.has_len else None
        return self.inner.stats_to_ss(stats[0], k), len_ss

    def merge_stats(self, dst, src):
        self.inner.merge_stats(dst[0], src[0])
        if self.has_len:
            self.len_b.merge_stats(dst[1], src[1])


def _optional_kernel(inner):
    @numba.njit(inline="always")
    def ld(i, k, params, cols):
        ci = cols[0][i]
        if ci < 0:
            return params[0][k]
        return params[1][k] + inner(ci, k, params[2], cols[1])

    return ld


def _optional_acc(inner):
    @numba.njit(inline="always")
    def acc(i, k, w, params, cols, stats):
        ci = cols[0][i]
        if ci < 0:
            stats[0][k] += w
        else:
            stats[1][k] += w
            inner(ci, k, w, params[2], cols[1], stats[2])

    return acc


class _OptionalB:
    """Missing-data wrapper: a presence-index column routes present rows into
    the (compacted) child columns; missing rows score the missing mass."""

    def __init__(self, dists):
        d0 = dists[0]
        self.nan_missing = d0.missing_value_is_nan
        self.missing_value = d0.missing_value
        for d in dists[1:]:
            same = (
                d.missing_value_is_nan
                if self.nan_missing
                else (not d.missing_value_is_nan and d.missing_value == self.missing_value)
            )
            if not same:
                raise ValueError("Optional components must share the same missing_value.")
        self.inner = build_kernel([d.dist for d in dists])
        self.kernel = _optional_kernel(self.inner.kernel)
        self.acc_kernel = _optional_acc(self.inner.acc_kernel)

    def params(self, dists):
        K = len(dists)
        miss_lp = np.zeros(K)  # log_density: 0.0 missing / child present when no p
        pres_lp = np.zeros(K)
        for k, d in enumerate(dists):
            if d.has_p:
                miss_lp[k] = d.log_p
                pres_lp[k] = d.log_pn
        return miss_lp, pres_lp, self.inner.params([d.dist for d in dists])

    def _is_missing(self, v):
        if self.nan_missing:
            return isinstance(v, (np.floating, float)) and np.isnan(v)
        return v == self.missing_value

    def new_sink(self):
        inner_buf, inner_add = self.inner.new_sink()
        idx = []
        n_present = [0]

        def add(x):
            if self._is_missing(x):
                idx.append(-1)
            else:
                idx.append(n_present[0])
                n_present[0] += 1
                inner_add(x)

        return (idx, inner_buf), add

    def freeze(self, bufs):
        idx, inner_buf = bufs
        return np.asarray(idx, dtype=np.int64), self.inner.freeze(inner_buf)

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K), self.inner.make_stats(K)

    def stats_to_ss(self, stats, k):
        # legacy OptionalEstimatorAccumulator.value(): ([missing_w, present_w], child)
        return [stats[0][k], stats[1][k]], self.inner.stats_to_ss(stats[2], k)

    def merge_stats(self, dst, src):
        np.add(dst[0], src[0], out=dst[0])
        np.add(dst[1], src[1], out=dst[1])
        self.inner.merge_stats(dst[2], src[2])


class _IgnoredB:
    """Fixed wrapped dists: per-component log-densities are precomputed at
    encode time (the wrapped dist scores rows but is never estimated)."""

    def __init__(self, dists):
        self.dists = list(dists)
        self._frozen = [str(d.dist) for d in dists]
        self.encoder = dists[0].dist_to_encoder()
        for d in dists[1:]:
            if d.dist_to_encoder() != self.encoder:
                raise ValueError("Ignored components must share the same data encoder.")
        self._dummy = np.zeros(1)

    kernel = staticmethod(_ignored_ld)
    acc_kernel = staticmethod(_ignored_acc)

    def params(self, dists):
        for d, s in zip(dists, self._frozen):
            if str(d.dist) != s:
                raise ValueError(
                    "IgnoredDistribution wrapped dists changed since compile; rebuild the CompiledMixture."
                )
        return (self._dummy,)

    def new_sink(self):
        buf = []
        return buf, buf.append

    def freeze(self, buf):
        enc = self.encoder.seq_encode(buf)
        mat = np.empty((len(buf), len(self.dists)), dtype=np.float64)
        for k, d in enumerate(self.dists):
            mat[:, k] = d.seq_log_density(enc)
        return (mat,)

    def make_stats(self, K):
        return (np.zeros(1),)

    def stats_to_ss(self, stats, k):
        # legacy IgnoredAccumulator.value(): None (nothing is estimated)
        return None

    def merge_stats(self, dst, src):
        # nothing is estimated; the sentinel slot stays additive (zeros).
        for d, s in zip(dst, src):
            np.add(d, s, out=d)


_BUILDERS = {
    GaussianDistribution: _GaussianB,
    LogGaussianDistribution: _LogGaussianB,
    GammaDistribution: _GammaB,
    BernoulliDistribution: _BernoulliB,
    StudentTDistribution: _StudentTB,
    LogisticDistribution: _LogisticB,
    WeibullDistribution: _WeibullB,
    RayleighDistribution: _RayleighB,
    ParetoDistribution: _ParetoB,
    UniformDistribution: _UniformB,
    CategoricalDistribution: _CategoricalB,
    IntegerCategoricalDistribution: _IntRangeB,
    PoissonDistribution: _PoissonB,
    ExponentialDistribution: _ExponentialB,
    GeometricDistribution: _GeometricB,
    BinomialDistribution: _BinomialB,
    NegativeBinomialDistribution: _NegativeBinomialB,
    DiagonalGaussianDistribution: _DiagGaussianB,
    CompositeDistribution: _CompositeB,
    SequenceDistribution: _SequenceB,
    OptionalDistribution: _OptionalB,
    IgnoredDistribution: _IgnoredB,
}


def build_kernel(dists: Sequence[Any]):
    """Build the fused kernel node for K same-structure distributions."""
    t = type(dists[0])
    if any(type(d) is not t for d in dists):
        raise ValueError(
            "all components must have identical structure (got mixed %s)" % {type(d).__name__ for d in dists}
        )
    if t not in _BUILDERS:
        raise ValueError("no numba kernel for distribution type %s" % t.__name__)
    return _BUILDERS[t](dists)


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------


def _make_score_driver(kern):
    # numba's parallel=True gufunc wrapper rejects nested heterogeneous tuple
    # arguments, so the driver is a serial nopython loop (GIL released) and
    # CompiledMixture parallelizes by running row chunks on a thread pool.
    @numba.njit(nogil=True)
    def drv(s0, s1, K, params, cols, out):
        for i in range(s0, s1):
            for k in range(K):
                out[i, k] = kern(i, k, params, cols)

    return drv


def _make_acc_driver(acc):
    @numba.njit(nogil=True)
    def drv(s0, s1, K, gamma, params, cols, stats):
        for i in range(s0, s1):
            for k in range(K):
                w = gamma[i, k]
                if w > 0.0:
                    acc(i, k, w, params, cols, stats)

    return drv


class CompiledMixture:
    """Fused-kernel estimation engine for a mixture (or single distribution).

    Compile once per model *structure*; parameters are re-read from the current
    model on every call, so EM iterations never recompile.
    """

    def __init__(self, model):
        if isinstance(model, MixtureDistribution):
            self.is_mixture = True
            comps = list(model.components)
        else:
            self.is_mixture = False
            comps = [model]

        self.model = model
        self.builder = build_kernel(comps)
        self.K = len(comps)
        self._score = _make_score_driver(self.builder.kernel)
        self._accumulate = _make_acc_driver(self.builder.acc_kernel)
        self._pool = None

    # -- data ------------------------------------------------------------

    def encode(self, data):
        """Encode raw observations into the compiled mixture's columnar buffers."""
        bufs, add = self.builder.new_sink()
        n = 0
        for x in data:
            add(x)
            n += 1
        return n, self.builder.freeze(bufs)

    # -- scoring -----------------------------------------------------------

    def _components(self, model):
        return list(model.components) if self.is_mixture else [model]

    _MIN_ROWS_PER_THREAD = 50000

    def _executor(self, n_threads):
        if self._pool is None or self._pool._max_workers < n_threads:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=n_threads)
        return self._pool

    def seq_component_log_density(self, enc, model=None, n_threads: int | None = None) -> np.ndarray:
        """Return per-component log-density matrix for encoded observations."""
        model = self.model if model is None else model
        n, cols = enc
        out = np.empty((n, self.K), dtype=np.float64)
        params = self.builder.params(self._components(model))

        if n_threads is None:
            import os

            n_threads = max(1, min(8, os.cpu_count() or 1, n * self.K // self._MIN_ROWS_PER_THREAD))

        if n_threads <= 1:
            self._score(0, n, self.K, params, cols, out)
            return out

        # the nopython driver releases the GIL, so chunked threads scale; use
        # several chunks per thread so asymmetric cores stay load-balanced
        n_chunks = n_threads * 4
        bounds = np.linspace(0, n, n_chunks + 1).astype(np.int64)
        ex = self._executor(n_threads)
        futs = [ex.submit(self._score, bounds[t], bounds[t + 1], self.K, params, cols, out) for t in range(n_chunks)]
        for f in futs:
            f.result()
        return out

    def seq_log_density(self, enc, model=None) -> np.ndarray:
        """Return mixture or single-model log densities for encoded observations."""
        model = self.model if model is None else model
        ll = self.seq_component_log_density(enc, model)
        if not self.is_mixture:
            return ll[:, 0]
        ll = ll + np.asarray(model.log_w).reshape(1, -1)
        mx = ll.max(axis=1, keepdims=True)
        good = np.isfinite(mx[:, 0])
        rv = np.full(ll.shape[0], -np.inf)
        rv[good] = np.log(np.exp(ll[good] - mx[good]).sum(axis=1)) + mx[good, 0]
        return rv

    def posteriors(self, enc, model=None) -> np.ndarray:
        """Return posterior component weights for encoded observations."""
        model = self.model if model is None else model
        ll = self.seq_component_log_density(enc, model)
        if self.is_mixture:
            ll = ll + np.asarray(model.log_w).reshape(1, -1)
        mx = ll.max(axis=1, keepdims=True)
        # Rows whose component log-densities are all -inf (out-of-support
        # observations) have a non-finite max; the softmax would produce NaN
        # responsibilities. Match the legacy seq_posterior convention and assign
        # those rows a uniform 1/K instead (mirrors the seq_log_density guard).
        bad = ~np.isfinite(mx[:, 0])
        if bad.any():
            # zero the shift so the exp/sum below stays finite; the row is
            # overwritten with a uniform 1/K afterward.
            mx[bad] = 0.0
        ll -= mx
        np.exp(ll, out=ll)
        denom = ll.sum(axis=1, keepdims=True)
        if bad.any():
            denom[bad] = 1.0
        ll /= denom
        if bad.any():
            ll[bad] = 1.0 / ll.shape[1]
        return ll

    # -- estimation ----------------------------------------------------------

    def weighted_suff_stats(self, enc, gamma: np.ndarray, model=None, n_threads: int | None = None):
        """Legacy-format sufficient statistics from per-row component weights."""
        model = self.model if model is None else model
        n, cols = enc
        gamma = np.ascontiguousarray(gamma, dtype=np.float64)
        params = self.builder.params(self._components(model))

        if n_threads is None:
            import os

            n_threads = max(1, min(8, os.cpu_count() or 1, n * self.K // self._MIN_ROWS_PER_THREAD))

        if n_threads <= 1:
            stats = self.builder.make_stats(self.K)
            self._accumulate(0, n, self.K, gamma, params, cols, stats)
        else:
            # per-chunk stats buffers (additive), merged after the chunked passes;
            # several chunks per thread keep asymmetric cores load-balanced
            n_chunks = n_threads * 4
            bufs = [self.builder.make_stats(self.K) for _ in range(n_chunks)]
            bounds = np.linspace(0, n, n_chunks + 1).astype(np.int64)
            ex = self._executor(n_threads)
            futs = [
                ex.submit(self._accumulate, bounds[t], bounds[t + 1], self.K, gamma, params, cols, bufs[t])
                for t in range(n_chunks)
            ]
            for f in futs:
                f.result()
            stats = bufs[0]
            for b in bufs[1:]:
                self.builder.merge_stats(stats, b)

        comp_ss = tuple(self.builder.stats_to_ss(stats, k) for k in range(self.K))
        if self.is_mixture:
            return gamma.sum(axis=0), comp_ss
        return comp_ss[0]

    def em_step(self, enc, estimator, model=None, weights: np.ndarray | None = None):
        """One fused EM step; returns the re-estimated model via the legacy M-step."""
        model = self.model if model is None else model
        n, _ = enc
        gamma = self.posteriors(enc, model)
        if weights is not None:
            gamma = gamma * np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        return estimator.estimate(n, self.weighted_suff_stats(enc, gamma))

    def initialize(self, enc, estimator, rng: np.random.RandomState, p: float = 0.1):
        """Random sparse-responsibility initialization (replaces seq_initialize)."""
        n, _ = enc
        keep = (rng.rand(n) <= p).astype(np.float64)
        gamma = rng.dirichlet(np.ones(self.K) / (self.K**2), size=n) * keep[:, None]
        return estimator.estimate(n, self.weighted_suff_stats(enc, gamma))

    def fit(
        self,
        enc,
        estimator,
        max_its: int = 100,
        delta: float = 1.0e-8,
        rng: np.random.RandomState | None = None,
        init_p: float = 0.1,
        model=None,
        out=None,
        max_iter: int | None = None,
    ):
        """EM to convergence with fused kernels. Returns (model, log_likelihood).

        ``max_iter`` is the preferred spelling of ``max_its``; when given it overrides ``max_its``.
        """
        if max_iter is not None:
            max_its = max_iter
        if model is None:
            model = self.initialize(enc, estimator, rng or np.random.RandomState(0), p=init_p)

        old_ll = self.seq_log_density(enc, model).sum()
        for it in range(max_its):
            model = self.em_step(enc, estimator, model=model)
            ll = self.seq_log_density(enc, model).sum()
            if out is not None:
                out.write("Iteration %d: ln[p(Data|Model)]=%e, delta=%e\n" % (it + 1, ll, ll - old_ll))
            if abs(ll - old_ll) < delta:
                old_ll = ll
                break
            old_ll = ll

        return model, old_ll
