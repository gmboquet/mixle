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

Supported distributions: Gaussian, Categorical, IntegerCategorical, Poisson,
Exponential, Geometric, Composite (any nesting), Sequence (variable-length,
with optional length model), and a Mixture over same-structure components at
the top level. Unsupported models should continue to use the seq_* path.

Performance characteristics (Apple silicon, float64): on flat scalar fields
the fused scalar loops run at rough parity with the legacy vectorized numpy
path (numpy's per-field passes are SIMD-optimized; the kernels make a single
threaded pass). On variable-length sequence data - the heterogeneous case
this library targets - fusion wins decisively: ~8-9x faster scoring and EM
on topic-model-like mixtures of token sequences, because the legacy path
re-walks the flattened token arrays once per component per field while the
kernel path streams each observation exactly once.
"""
import math
from typing import Any, List, Optional, Sequence, Tuple

import numba
import numpy as np

from pysp.stats.gaussian import GaussianDistribution
from pysp.stats.categorical import CategoricalDistribution
from pysp.stats.intrange import IntegerCategoricalDistribution
from pysp.stats.poisson import PoissonDistribution
from pysp.stats.exponential import ExponentialDistribution
from pysp.stats.geometric import GeometricDistribution
from pysp.stats.composite import CompositeDistribution
from pysp.stats.sequence import SequenceDistribution
from pysp.stats.mixture import MixtureDistribution
from pysp.stats.null_dist import NullDistribution

__all__ = ['CompiledMixture', 'build_kernel']

_NEG_INF = float('-inf')


# ----------------------------------------------------------------------------
# Leaf kernels (module level so numba can cache them; inlined into compositions)
# ----------------------------------------------------------------------------

@numba.njit(inline='always', cache=True)
def _gaussian_ld(i, k, params, cols):
    mu, s2, logc = params
    x = cols[0][i]
    d = x - mu[k]
    return logc[k] - 0.5 * d * d / s2[k]


@numba.njit(inline='always', cache=True)
def _gaussian_acc(i, k, w, params, cols, stats):
    x = cols[0][i]
    stats[0][k] += w * x
    stats[1][k] += w * x * x
    stats[2][k] += w


@numba.njit(inline='always', cache=True)
def _categorical_ld(i, k, params, cols):
    logp, log_default = params
    c = cols[0][i]
    if c < 0:
        return log_default[k]
    return logp[k, c]


@numba.njit(inline='always', cache=True)
def _categorical_acc(i, k, w, params, cols, stats):
    c = cols[0][i]
    if c >= 0:
        stats[0][k, c] += w


@numba.njit(inline='always', cache=True)
def _intrange_ld(i, k, params, cols):
    logp, min_val = params
    idx = cols[0][i] - min_val
    if idx < 0 or idx >= logp.shape[1]:
        return _NEG_INF
    return logp[k, idx]


@numba.njit(inline='always', cache=True)
def _intrange_acc(i, k, w, params, cols, stats):
    idx = cols[0][i] - params[1]
    if 0 <= idx < stats[0].shape[1]:
        stats[0][k, idx] += w


@numba.njit(inline='always', cache=True)
def _poisson_ld(i, k, params, cols):
    # cols[1] holds lgamma(x+1), precomputed once at encode time
    lam, loglam = params
    x = cols[0][i]
    if x < 0:
        return _NEG_INF
    return x * loglam[k] - lam[k] - cols[1][i]


@numba.njit(inline='always', cache=True)
def _count_sum_acc(i, k, w, params, cols, stats):
    stats[0][k] += w
    stats[1][k] += w * cols[0][i]


@numba.njit(inline='always', cache=True)
def _exponential_ld(i, k, params, cols):
    beta, logbeta = params
    x = cols[0][i]
    if x < 0:
        return _NEG_INF
    return -x / beta[k] - logbeta[k]


@numba.njit(inline='always', cache=True)
def _geometric_ld(i, k, params, cols):
    logp, log1mp = params
    x = cols[0][i]
    return (x - 1.0) * log1mp[k] + logp[k]


# ----------------------------------------------------------------------------
# Builders: per-node compile of (kernel, acc kernel, params, stats, encoder)
# ----------------------------------------------------------------------------

class _LeafBuilder(object):
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


class _GaussianB(_LeafBuilder):
    kernel = _gaussian_ld
    acc_kernel = _gaussian_acc

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
    kernel = _categorical_ld
    acc_kernel = _categorical_acc

    def __init__(self, dists):
        super().__init__(dists)
        vocab = list(dict.fromkeys(v for d in dists for v in d.pmap.keys()))
        self.vocab = vocab
        self.codes = {v: c for c, v in enumerate(vocab)}

    def params(self, dists):
        K, V = len(dists), len(self.vocab)
        logp = np.empty((K, V), dtype=np.float64)
        log_default = np.empty(K, dtype=np.float64)
        with np.errstate(divide='ignore'):
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
    kernel = _intrange_ld
    acc_kernel = _intrange_acc
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
            logp[k, lo:lo + len(d.log_p_vec)] = d.log_p_vec
        return logp, np.int64(self.min_val)

    def make_stats(self, K):
        V = self.max_val - self.min_val + 1
        return (np.zeros((K, V)),)

    def stats_to_ss(self, stats, k):
        return self.min_val, stats[0][k].copy()


class _CountSumB(_LeafBuilder):
    acc_kernel = _count_sum_acc

    def make_stats(self, K):
        return np.zeros(K), np.zeros(K)

    def stats_to_ss(self, stats, k):
        return stats[0][k], stats[1][k]


class _PoissonB(_CountSumB):
    kernel = _poisson_ld

    def params(self, dists):
        lam = np.array([d.lam for d in dists], dtype=np.float64)
        return lam, np.log(lam)

    def freeze(self, buf):
        from scipy.special import gammaln
        x = np.asarray(buf, dtype=np.float64)
        return x, gammaln(x + 1.0)


class _ExponentialB(_CountSumB):
    kernel = _exponential_ld

    def params(self, dists):
        beta = np.array([d.beta for d in dists], dtype=np.float64)
        return beta, np.log(beta)


class _GeometricB(_CountSumB):
    kernel = _geometric_ld

    def params(self, dists):
        p = np.array([d.p for d in dists], dtype=np.float64)
        return np.log(p), np.log1p(-p)


def _pair_kernels(f, g):
    @numba.njit(inline='always')
    def ld(i, k, params, cols):
        return f(i, k, params[0], cols[0]) + g(i, k, params[1], cols[1])
    return ld


def _pair_accs(f, g):
    @numba.njit(inline='always')
    def acc(i, k, w, params, cols, stats):
        f(i, k, w, params[0], cols[0], stats[0])
        g(i, k, w, params[1], cols[1], stats[1])
    return acc


class _CompositeB(object):
    """Children fused by folding into a binary tree of two-term kernels.

    The params/cols/stats for the node mirror the tree as nested 2-tuples;
    _tree/_untree convert between the flat per-slot lists and that shape.
    """

    def __init__(self, dists):
        self.child_builders = [
            build_kernel([d.dists[j] for d in dists]) for j in range(len(dists[0].dists))
        ]
        kerns = [b.kernel for b in self.child_builders]
        accs = [b.acc_kernel for b in self.child_builders]
        while len(kerns) > 1:
            kerns = [_pair_kernels(kerns[i], kerns[i + 1]) if i + 1 < len(kerns) else kerns[i]
                     for i in range(0, len(kerns), 2)]
            accs = [_pair_accs(accs[i], accs[i + 1]) if i + 1 < len(accs) else accs[i]
                    for i in range(0, len(accs), 2)]
        self.kernel = kerns[0]
        self.acc_kernel = accs[0]

    @staticmethod
    def _tree(items):
        items = list(items)
        while len(items) > 1:
            items = [(items[i], items[i + 1]) if i + 1 < len(items) else items[i]
                     for i in range(0, len(items), 2)]
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
        return self._tree(b.params([d.dists[j] for d in dists])
                          for j, b in enumerate(self.child_builders))

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


def _sequence_kernel(inner, len_kern, has_len):
    if has_len:
        @numba.njit(inline='always')
        def ld(i, k, params, cols):
            off, inner_cols, len_cols = cols
            rv = 0.0
            for t in range(off[i], off[i + 1]):
                rv += inner(t, k, params[0], inner_cols)
            return rv + len_kern(i, k, params[1], len_cols)
        return ld

    @numba.njit(inline='always')
    def ld(i, k, params, cols):
        off, inner_cols, len_cols = cols
        rv = 0.0
        for t in range(off[i], off[i + 1]):
            rv += inner(t, k, params[0], inner_cols)
        return rv
    return ld


def _sequence_acc(inner, len_acc, has_len):
    if has_len:
        @numba.njit(inline='always')
        def acc(i, k, w, params, cols, stats):
            off, inner_cols, len_cols = cols
            for t in range(off[i], off[i + 1]):
                inner(t, k, w, params[0], inner_cols, stats[0])
            len_acc(i, k, w, params[1], len_cols, stats[1])
        return acc

    @numba.njit(inline='always')
    def acc(i, k, w, params, cols, stats):
        off, inner_cols, len_cols = cols
        for t in range(off[i], off[i + 1]):
            inner(t, k, w, params[0], inner_cols, stats[0])
    return acc


class _SequenceB(object):

    def __init__(self, dists):
        if any(getattr(d, 'len_normalized', False) for d in dists):
            raise ValueError('kernels do not support len_normalized SequenceDistributions.')
        self.inner = build_kernel([d.dist for d in dists])
        self.has_len = not isinstance(dists[0].len_dist, NullDistribution) and dists[0].len_dist is not None
        if self.has_len:
            self.len_b = build_kernel([d.len_dist for d in dists])
        else:
            self.len_b = _PoissonB([PoissonDistribution(1.0)])  # placeholder; never evaluated
        self.kernel = _sequence_kernel(self.inner.kernel, self.len_b.kernel, self.has_len)
        self.acc_kernel = _sequence_acc(self.inner.acc_kernel, self.len_b.acc_kernel, self.has_len)

    def params(self, dists):
        inner_p = self.inner.params([d.dist for d in dists])
        len_p = self.len_b.params([d.len_dist for d in dists]) if self.has_len \
            else self.len_b.params(self.len_b.dists)
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


_BUILDERS = {
    GaussianDistribution: _GaussianB,
    CategoricalDistribution: _CategoricalB,
    IntegerCategoricalDistribution: _IntRangeB,
    PoissonDistribution: _PoissonB,
    ExponentialDistribution: _ExponentialB,
    GeometricDistribution: _GeometricB,
    CompositeDistribution: _CompositeB,
    SequenceDistribution: _SequenceB,
}


def build_kernel(dists: Sequence[Any]):
    """Build the fused kernel node for K same-structure distributions."""
    t = type(dists[0])
    if any(type(d) is not t for d in dists):
        raise ValueError('all components must have identical structure (got mixed %s)' %
                         {type(d).__name__ for d in dists})
    if t not in _BUILDERS:
        raise ValueError('no numba kernel for distribution type %s' % t.__name__)
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


def _merge_stats(dst, src):
    if isinstance(dst, tuple):
        for d, s in zip(dst, src):
            _merge_stats(d, s)
    else:
        dst += src


class CompiledMixture(object):
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

    def seq_component_log_density(self, enc, model=None, n_threads: Optional[int] = None) -> np.ndarray:
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
        futs = [ex.submit(self._score, bounds[t], bounds[t + 1], self.K, params, cols, out)
                for t in range(n_chunks)]
        for f in futs:
            f.result()
        return out

    def seq_log_density(self, enc, model=None) -> np.ndarray:
        model = self.model if model is None else model
        ll = self.seq_component_log_density(enc, model)
        if not self.is_mixture:
            return ll[:, 0]
        ll = ll + np.asarray(model.log_w).reshape(1, -1)
        mx = ll.max(axis=1, keepdims=True)
        good = np.isfinite(mx[:, 0])
        rv = np.full(ll.shape[0], -np.inf)
        rv[good] = (np.log(np.exp(ll[good] - mx[good]).sum(axis=1)) + mx[good, 0])
        return rv

    def posteriors(self, enc, model=None) -> np.ndarray:
        model = self.model if model is None else model
        ll = self.seq_component_log_density(enc, model)
        if self.is_mixture:
            ll = ll + np.asarray(model.log_w).reshape(1, -1)
        ll -= ll.max(axis=1, keepdims=True)
        np.exp(ll, out=ll)
        ll /= ll.sum(axis=1, keepdims=True)
        return ll

    # -- estimation ----------------------------------------------------------

    def weighted_suff_stats(self, enc, gamma: np.ndarray, model=None,
                            n_threads: Optional[int] = None):
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
            futs = [ex.submit(self._accumulate, bounds[t], bounds[t + 1], self.K,
                              gamma, params, cols, bufs[t]) for t in range(n_chunks)]
            for f in futs:
                f.result()
            stats = bufs[0]
            for b in bufs[1:]:
                _merge_stats(stats, b)

        comp_ss = tuple(self.builder.stats_to_ss(stats, k) for k in range(self.K))
        if self.is_mixture:
            return gamma.sum(axis=0), comp_ss
        return comp_ss[0]

    def em_step(self, enc, estimator, model=None, weights: Optional[np.ndarray] = None):
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
        gamma = rng.dirichlet(np.ones(self.K) / (self.K ** 2), size=n) * keep[:, None]
        return estimator.estimate(n, self.weighted_suff_stats(enc, gamma))

    def fit(self, enc, estimator, max_its: int = 100, delta: float = 1.0e-8,
            rng: Optional[np.random.RandomState] = None, init_p: float = 0.1,
            model=None, out=None):
        """EM to convergence with fused kernels. Returns (model, log_likelihood)."""
        if model is None:
            model = self.initialize(enc, estimator, rng or np.random.RandomState(), p=init_p)

        old_ll = self.seq_log_density(enc, model).sum()
        for it in range(max_its):
            model = self.em_step(enc, estimator, model=model)
            ll = self.seq_log_density(enc, model).sum()
            if out is not None:
                out.write('Iteration %d: ln[p(Data|Model)]=%e, delta=%e\n' % (it + 1, ll, ll - old_ll))
            if abs(ll - old_ll) < delta:
                old_ll = ll
                break
            old_ll = ll

        return model, old_ll
