"""PyTorch estimation engine for mixtures over heterogeneous data.

A tensor counterpart to pysp.stats.kernels: the same generic columnar
encoding (reused from the kernel builders) feeds batched torch ops that score
all n observations and K components in one (n, K) tensor expression per leaf.
Sufficient statistics are tensor reductions (matmul / index_add) emitted in
the legacy accumulator formats, so the existing estimator M-steps - pseudo-
counts included - are reused unchanged and results match the seq_* path to
floating-point tolerance.

Two ways to fit:

- EM (`fit`, `em_step`, `initialize`): exact mirror of the kernel engine, but
  every pass is batched tensor math, so it runs on CUDA/MPS as well as CPU.
- Gradient MLE (`fit_mle`): the new capability. Parameters are reparameterized
  into unconstrained tensors (means, log-variances, category logits, log-rates,
  stick logits for the mixture weights), the exact mixture log-likelihood is
  autograd-differentiable, and any torch optimizer maximizes it directly. This
  needs no closed-form M-step, so it extends to penalized objectives or
  partially frozen parameters without new estimator code.

Device/dtype notes: float64 on CPU/CUDA reproduces the legacy numerics; MPS
only supports float32, so expect ~1e-5 relative agreement there.

Supported distributions match pysp.stats.kernels: Gaussian, LogGaussian,
Gamma, Categorical, IntegerCategorical, Poisson, Exponential, Geometric,
Binomial, DiagonalGaussian (vector leaf), Optional (missing-data wrapper
around any supported child), Ignored (fixed wrapped dist, scored via a
precomputed column but never fitted), Composite (any nesting), Sequence
(variable length, optional length model), and a top-level mixture of
same-structure components.
"""
import math
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from pysp.stats.gaussian import GaussianDistribution
from pysp.stats.log_gaussian import LogGaussianDistribution
from pysp.stats.gamma import GammaDistribution
from pysp.stats.categorical import CategoricalDistribution
from pysp.stats.intrange import IntegerCategoricalDistribution
from pysp.stats.poisson import PoissonDistribution
from pysp.stats.exponential import ExponentialDistribution
from pysp.stats.geometric import GeometricDistribution
from pysp.stats.binomial import BinomialDistribution
from pysp.stats.dmvn import DiagonalGaussianDistribution
from pysp.stats.composite import CompositeDistribution
from pysp.stats.sequence import SequenceDistribution
from pysp.stats.mixture import MixtureDistribution
from pysp.stats.optional import OptionalDistribution
from pysp.stats import kernels as _k

__all__ = ['TorchMixture']


# ----------------------------------------------------------------------------
# per-builder tensor implementations
#
# Each impl provides, for the matching kernels builder type:
#   params(dists)             -> dict of canonical tensors (no grad)
#   raw_params(dists)         -> dict of unconstrained leaf tensors for MLE
#   canonical(raw)            -> canonical tensors from unconstrained ones
#   to_dists(raw)             -> rebuilt pysp distributions (one per component)
#   log_density(params, cols) -> (n, K) tensor
#   stats(params, cols, gamma)-> numpy stats in the kernels builder layout
# ----------------------------------------------------------------------------


class _TBase(object):

    def __init__(self, kb, device, dtype):
        self.kb = kb
        self.device = device
        self.dtype = dtype

    def _t(self, x, dtype=None):
        return torch.as_tensor(np.asarray(x), dtype=dtype or self.dtype, device=self.device)

    def to_tensors(self, cols):
        return tuple(torch.as_tensor(c, device=self.device,
                                     dtype=self.dtype if c.dtype.kind == 'f' else torch.int64)
                     for c in cols)


class _TGaussian(_TBase):

    def params(self, dists):
        mu = self._t([d.mu for d in dists])
        s2 = self._t([d.sigma2 for d in dists])
        return {'mu': mu, 's2': s2}

    def raw_params(self, dists):
        return {'mu': self._t([d.mu for d in dists]).requires_grad_(True),
                'log_s2': self._t([math.log(d.sigma2) for d in dists]).requires_grad_(True)}

    def canonical(self, raw):
        return {'mu': raw['mu'], 's2': raw['log_s2'].exp()}

    def to_dists(self, raw):
        mu = raw['mu'].detach().cpu().numpy()
        s2 = raw['log_s2'].detach().exp().cpu().numpy()
        return [GaussianDistribution(float(m), float(v)) for m, v in zip(mu, s2)]

    def log_density(self, params, cols):
        x = cols[0].unsqueeze(1)                                    # (n, 1)
        mu, s2 = params['mu'].unsqueeze(0), params['s2'].unsqueeze(0)
        return -0.5 * torch.log(2.0 * math.pi * s2) - 0.5 * (x - mu) ** 2 / s2

    def stats(self, params, cols, gamma):
        x = cols[0]
        s1 = (gamma * x.unsqueeze(1)).sum(dim=0)
        s2 = (gamma * (x * x).unsqueeze(1)).sum(dim=0)
        c = gamma.sum(dim=0)
        return (s1.cpu().numpy(), s2.cpu().numpy(), c.cpu().numpy())

    def default_prior(self, strength):
        # NormalGamma(mu0, kappa, a, b) on (mu, 1/sigma2); strength=0 -> flat
        return {'family': 'normalgamma', 'mu0': 0.0, 'kappa': 1.0e-3 * strength,
                'a': 1.0 + 1.0e-3 * strength, 'b': 1.0e-3 * strength}

    def log_prior(self, params, prior):
        tau = 1.0 / params['s2']
        lp = (prior['a'] - 1.0) * torch.log(tau) - prior['b'] * tau
        if prior['kappa'] > 0:
            lp = lp + 0.5 * torch.log(tau) \
                - 0.5 * prior['kappa'] * tau * (params['mu'] - prior['mu0']) ** 2
        return lp.sum()


class _TLogP(_TBase):
    """Shared machinery for table distributions (categorical / integer range)."""

    def _table(self, dists):
        raise NotImplementedError

    def params(self, dists):
        return {'logp': self._t(self._table(dists))}

    def raw_params(self, dists):
        return {'logits': self._t(self._table(dists)).clamp(min=-30.0).requires_grad_(True)}

    def canonical(self, raw):
        return {'logp': torch.log_softmax(raw['logits'], dim=1)}

    def log_density(self, params, cols):
        codes = cols[0]
        valid = codes >= 0
        safe = codes.clamp(min=0)
        out = params['logp'][:, safe].T                              # (n, K)
        return self._invalid(out, valid, params)

    def _invalid(self, out, valid, params):
        return torch.where(valid.unsqueeze(1), out,
                           torch.tensor(float('-inf'), dtype=out.dtype, device=out.device))

    def stats(self, params, cols, gamma):
        codes = cols[0]
        valid = codes >= 0
        V = params['logp'].shape[1]
        counts = torch.zeros((V, gamma.shape[1]), dtype=gamma.dtype, device=gamma.device)
        counts.index_add_(0, codes.clamp(min=0), gamma * valid.unsqueeze(1).to(gamma.dtype))
        return (counts.T.contiguous().cpu().numpy(),)

    def default_prior(self, strength):
        V = len(getattr(self.kb, 'vocab', [])) or (self.kb.max_val - self.kb.min_val + 1)
        return {'family': 'dirichlet', 'alpha': 1.0 + strength / V}

    def log_prior(self, params, prior):
        if prior['alpha'] == 1.0:
            return torch.zeros((), dtype=self.dtype, device=self.device)
        return (prior['alpha'] - 1.0) * params['logp'].clamp(min=-1.0e6).sum()


class _TCategorical(_TLogP):

    def _table(self, dists):
        vocab = self.kb.vocab
        with np.errstate(divide='ignore'):
            return np.array([[math.log(d.pmap.get(v, d.default_value)) - d.log1p_default_value
                              if d.pmap.get(v, d.default_value) > 0 else -np.inf
                              for v in vocab] for d in dists])

    def to_dists(self, raw):
        p = torch.softmax(raw['logits'], dim=1).detach().cpu().numpy()
        return [CategoricalDistribution({v: float(p[k, c]) for c, v in enumerate(self.kb.vocab)})
                for k in range(p.shape[0])]

    def _invalid(self, out, valid, params):
        # unseen value -> default_value mass; matches the kernel path for
        # default_value = 0 (which yields -inf)
        log_def = self._t([math.log(d.default_value) - d.log1p_default_value
                           if d.default_value > 0 else -np.inf for d in self.kb.dists])
        return torch.where(valid.unsqueeze(1), out, log_def.unsqueeze(0))


class _TIntRange(_TLogP):

    def _table(self, dists):
        K, lo = len(dists), self.kb.min_val
        V = self.kb.max_val - lo + 1
        tab = np.full((K, V), -np.inf)
        for k, d in enumerate(dists):
            o = d.min_val - lo
            tab[k, o:o + len(d.log_p_vec)] = d.log_p_vec
        return tab

    def to_dists(self, raw):
        p = torch.softmax(raw['logits'], dim=1).detach().cpu().numpy()
        return [IntegerCategoricalDistribution(self.kb.min_val, list(p[k]))
                for k in range(p.shape[0])]


class _TPoisson(_TBase):

    def params(self, dists):
        lam = self._t([d.lam for d in dists])
        return {'lam': lam}

    def raw_params(self, dists):
        return {'log_lam': self._t([math.log(d.lam) for d in dists]).requires_grad_(True)}

    def canonical(self, raw):
        return {'lam': raw['log_lam'].exp()}

    def to_dists(self, raw):
        return [PoissonDistribution(float(l)) for l in raw['log_lam'].detach().exp().cpu().numpy()]

    def log_density(self, params, cols):
        x, lgx = cols[0].unsqueeze(1), cols[1].unsqueeze(1)
        lam = params['lam'].unsqueeze(0)
        return x * torch.log(lam) - lam - lgx

    def stats(self, params, cols, gamma):
        c = gamma.sum(dim=0)
        s = (gamma * cols[0].unsqueeze(1)).sum(dim=0)
        return (c.cpu().numpy(), s.cpu().numpy())

    def default_prior(self, strength):
        return {'family': 'gamma', 'a': 1.0 + 1.0e-3 * strength, 'b': 1.0e-3 * strength}

    def log_prior(self, params, prior):
        lam = params[self._rate_key]
        rate = lam if self._rate_is_rate else 1.0 / lam
        return ((prior['a'] - 1.0) * torch.log(rate) - prior['b'] * rate).sum()

    _rate_key = 'lam'
    _rate_is_rate = True


class _TExponential(_TPoisson):

    def params(self, dists):
        return {'beta': self._t([d.beta for d in dists])}

    def raw_params(self, dists):
        return {'log_beta': self._t([math.log(d.beta) for d in dists]).requires_grad_(True)}

    def canonical(self, raw):
        return {'beta': raw['log_beta'].exp()}

    def to_dists(self, raw):
        return [ExponentialDistribution(float(b)) for b in raw['log_beta'].detach().exp().cpu().numpy()]

    def log_density(self, params, cols):
        x = cols[0].unsqueeze(1)
        beta = params['beta'].unsqueeze(0)
        return -x / beta - torch.log(beta)

    _rate_key = 'beta'
    _rate_is_rate = False


class _TGeometric(_TPoisson):

    def params(self, dists):
        return {'p': self._t([d.p for d in dists])}

    def raw_params(self, dists):
        return {'logit_p': self._t([math.log(d.p) - math.log1p(-d.p) for d in dists]).requires_grad_(True)}

    def canonical(self, raw):
        return {'p': torch.sigmoid(raw['logit_p'])}

    def to_dists(self, raw):
        return [GeometricDistribution(float(p)) for p in torch.sigmoid(raw['logit_p']).detach().cpu().numpy()]

    def log_density(self, params, cols):
        x = cols[0].unsqueeze(1)
        p = params['p'].unsqueeze(0)
        return (x - 1.0) * torch.log1p(-p) + torch.log(p)

    def default_prior(self, strength):
        return {'family': 'beta', 'a': 1.0 + 1.0e-3 * strength, 'b': 1.0 + 1.0e-3 * strength}

    def log_prior(self, params, prior):
        pp = params['p']
        return ((prior['a'] - 1.0) * torch.log(pp) + (prior['b'] - 1.0) * torch.log1p(-pp)).sum()


class _TGamma(_TBase):

    def params(self, dists):
        return {'k': self._t([d.k for d in dists]),
                'theta': self._t([d.theta for d in dists])}

    def raw_params(self, dists):
        return {'log_k': self._t([math.log(d.k) for d in dists]).requires_grad_(True),
                'log_theta': self._t([math.log(d.theta) for d in dists]).requires_grad_(True)}

    def canonical(self, raw):
        return {'k': raw['log_k'].exp(), 'theta': raw['log_theta'].exp()}

    def to_dists(self, raw):
        kk = raw['log_k'].detach().exp().cpu().numpy()
        th = raw['log_theta'].detach().exp().cpu().numpy()
        return [GammaDistribution(float(a), float(b)) for a, b in zip(kk, th)]

    def log_density(self, params, cols):
        # cols[1] holds log(x), precomputed at encode time by the kernel builder
        x, lx = cols[0].unsqueeze(1), cols[1].unsqueeze(1)
        kk, th = params['k'].unsqueeze(0), params['theta'].unsqueeze(0)
        return (kk - 1.0) * lx - x / th - torch.lgamma(kk) - kk * torch.log(th)

    def stats(self, params, cols, gamma):
        c = gamma.sum(dim=0)
        s = (gamma * cols[0].unsqueeze(1)).sum(dim=0)
        sl = (gamma * cols[1].unsqueeze(1)).sum(dim=0)
        return (c.cpu().numpy(), s.cpu().numpy(), sl.cpu().numpy())

    def default_prior(self, strength):
        # Gamma(a_rate, b_rate) on the rate 1/theta (the conjugate prior given
        # the shape) and Gamma(a_shape, b_shape) on the shape k: the shape has
        # no standard conjugate family, and a gamma prior keeps k > 0 with
        # adjustable mass in the weakly-informative region near k ~ 1 (a_shape
        # slightly above 1 with small b_shape shrinks k gently toward
        # moderate values); strength=0 -> flat on both (exact MLE).
        return {'family': 'gamma-shape-rate',
                'a_rate': 1.0 + 1.0e-3 * strength, 'b_rate': 1.0e-3 * strength,
                'a_shape': 1.0 + 1.0e-3 * strength, 'b_shape': 1.0e-3 * strength}

    def log_prior(self, params, prior):
        rate = 1.0 / params['theta']
        lp = (prior['a_rate'] - 1.0) * torch.log(rate) - prior['b_rate'] * rate
        lp = lp + (prior['a_shape'] - 1.0) * torch.log(params['k']) - prior['b_shape'] * params['k']
        return lp.sum()


class _TLogGaussian(_TGaussian):
    """Gaussian machinery on the pre-encoded log-x column (cols[0] = log x);
    only the density gains the -log x Jacobian term and dists are rebuilt as
    LogGaussian. Stats are the Gaussian moments of log x, which is exactly the
    legacy LogGaussianAccumulator layout; the NormalGamma prior acts on
    (mu, 1/sigma2) of the log-scale parameters."""

    def to_dists(self, raw):
        mu = raw['mu'].detach().cpu().numpy()
        s2 = raw['log_s2'].detach().exp().cpu().numpy()
        return [LogGaussianDistribution(float(m), float(v)) for m, v in zip(mu, s2)]

    def log_density(self, params, cols):
        lx = cols[0].unsqueeze(1)                                    # (n, 1) log x
        mu, s2 = params['mu'].unsqueeze(0), params['s2'].unsqueeze(0)
        return -0.5 * torch.log(2.0 * math.pi * s2) - 0.5 * (lx - mu) ** 2 / s2 - lx


class _TBinomial(_TBase):

    def params(self, dists):
        return {'logp': self._t([d.log_p for d in dists]),
                'log1mp': self._t([d.log_1p for d in dists]),
                'nv': self._t([float(d.n) for d in dists]),
                'mv': self._t([float(d.min_val) if d.min_val is not None else 0.0
                               for d in dists])}

    def raw_params(self, dists):
        # n / min_val are structural (the estimator re-derives them from data
        # extremes, not from a gradient), so only logit_p is optimized; the
        # compile-time values ride along as constants.
        self._nv = self._t([float(d.n) for d in dists])
        self._mv = self._t([float(d.min_val) if d.min_val is not None else 0.0
                            for d in dists])
        self._nm = [(int(d.n), d.min_val) for d in dists]
        p = np.clip(np.array([d.p for d in dists], dtype=np.float64),
                    1.0e-12, 1.0 - 1.0e-12)
        return {'logit_p': self._t(np.log(p) - np.log1p(-p)).requires_grad_(True)}

    def canonical(self, raw):
        lo = raw['logit_p']
        return {'logp': torch.nn.functional.logsigmoid(lo),
                'log1mp': torch.nn.functional.logsigmoid(-lo),
                'nv': self._nv, 'mv': self._mv}

    def to_dists(self, raw):
        p = torch.sigmoid(raw['logit_p']).detach().cpu().numpy()
        return [BinomialDistribution(float(pk), n, min_val=m)
                for pk, (n, m) in zip(p, self._nm)]

    def log_density(self, params, cols):
        # cols[1] holds the log-binomial coefficient precomputed for the
        # compile-time (n, min_val); components whose (n, min_val) moved (the
        # estimator re-derives both from data) recompute it via torch.lgamma.
        x = cols[0].unsqueeze(1)                                     # (n, 1)
        n = params['nv'].unsqueeze(0)                                # (1, K)
        xx = x - params['mv'].unsqueeze(0)                           # (n, K)
        ok = (xx >= 0.0) & (xx <= n)
        xs = torch.where(ok, xx, torch.zeros_like(xx))
        base = torch.lgamma(n + 1.0) - torch.lgamma(xs + 1.0) - torch.lgamma(n - xs + 1.0)
        match = (params['nv'] == self.kb.n0) & (params['mv'] == self.kb.m0)
        base = torch.where(match.unsqueeze(0), cols[1].unsqueeze(1).expand_as(base), base)
        out = base + params['log1mp'].unsqueeze(0) * (n - xx) + params['logp'].unsqueeze(0) * xx
        neg = torch.tensor(float('-inf'), dtype=out.dtype, device=out.device)
        return torch.where(ok, out, neg)

    def stats(self, params, cols, gamma):
        # cols[2] holds the data-wide (min, max); the w-weighted constants keep
        # the stats layout identical to the kernel accumulator (additive form)
        c = gamma.sum(dim=0)
        s = (gamma * cols[0].unsqueeze(1)).sum(dim=0)
        mn = c * cols[2][0]
        mx = c * cols[2][1]
        return (c.cpu().numpy(), s.cpu().numpy(), mn.cpu().numpy(), mx.cpu().numpy())

    def default_prior(self, strength):
        # Beta(a, b) on the success probability p; strength=0 -> Beta(1,1) flat
        return {'family': 'beta', 'a': 1.0 + 1.0e-3 * strength, 'b': 1.0 + 1.0e-3 * strength}

    def log_prior(self, params, prior):
        return ((prior['a'] - 1.0) * params['logp']
                + (prior['b'] - 1.0) * params['log1mp']).sum()


class _TDiagGaussian(_TGaussian):
    """Vectorized over the d dimensions: params are (K, d) tensors and the
    Gaussian NormalGamma prior applies independently per dimension (the
    inherited log_prior is elementwise and sums over both axes)."""

    def params(self, dists):
        return {'mu': self._t(np.array([d.mu for d in dists])),
                's2': self._t(np.array([d.covar for d in dists]))}

    def raw_params(self, dists):
        mu = np.array([d.mu for d in dists], dtype=np.float64)
        s2 = np.array([d.covar for d in dists], dtype=np.float64)
        return {'mu': self._t(mu).requires_grad_(True),
                'log_s2': self._t(np.log(s2)).requires_grad_(True)}

    def to_dists(self, raw):
        mu = raw['mu'].detach().cpu().numpy()
        s2 = raw['log_s2'].detach().exp().cpu().numpy()
        return [DiagonalGaussianDistribution([float(v) for v in m], [float(v) for v in s])
                for m, s in zip(mu, s2)]

    def log_density(self, params, cols):
        # cols[0] is the flattened (n*d) observation matrix from the kernel sink
        d = self.kb.dim
        x = cols[0].view(-1, 1, d)                                   # (n, 1, d)
        mu, s2 = params['mu'].unsqueeze(0), params['s2'].unsqueeze(0)  # (1, K, d)
        return (-0.5 * torch.log(2.0 * math.pi * s2)
                - 0.5 * (x - mu) ** 2 / s2).sum(dim=2)

    def stats(self, params, cols, gamma):
        d = self.kb.dim
        x = cols[0].view(-1, d)
        s1 = gamma.T @ x                                             # (K, d)
        s2 = gamma.T @ (x * x)
        c = gamma.sum(dim=0)
        return (s1.cpu().numpy(), s2.cpu().numpy(), c.cpu().numpy())


class _TOptional(_TBase):
    """Missing-data wrapper: the presence-index column (idx[i] = compacted
    child row, or -1 when missing) routes present rows into the child impl
    exactly as the kernel builder composes it; missing rows score the missing
    mass. The missing probability is optimized as a logit with a Beta prior;
    components built without p (has_p False) score 0 on both branches and
    contribute no p parameter gradient."""

    def __init__(self, kb, device, dtype):
        super().__init__(kb, device, dtype)
        self.child = _impl(kb.inner, device, dtype)
        self._mask = None  # per-component has_p, set by raw_params

    def params(self, dists):
        K = len(dists)
        miss = np.zeros(K)
        pres = np.zeros(K)
        for k, d in enumerate(dists):
            if d.has_p:
                miss[k] = d.log_p
                pres[k] = d.log_pn
        return {'miss_lp': self._t(miss), 'pres_lp': self._t(pres),
                'child': self.child.params([d.dist for d in dists])}

    def raw_params(self, dists):
        self._mask = torch.as_tensor([bool(d.has_p) for d in dists], device=self.device)
        p = np.clip(np.array([d.p for d in dists], dtype=np.float64),
                    1.0e-12, 1.0 - 1.0e-12)
        own = {'logit_p': self._t(np.log(p) - np.log1p(-p)).requires_grad_(True)}
        return [own, self.child.raw_params([d.dist for d in dists])]

    def canonical(self, raw):
        own, child_raw = raw
        lo = own['logit_p']
        zero = torch.zeros_like(lo)
        return {'miss_lp': torch.where(self._mask, torch.nn.functional.logsigmoid(lo), zero),
                'pres_lp': torch.where(self._mask, torch.nn.functional.logsigmoid(-lo), zero),
                'child': self.child.canonical(child_raw)}

    def to_dists(self, raw):
        own, child_raw = raw
        p = torch.sigmoid(own['logit_p']).detach().cpu().numpy()
        mask = self._mask.cpu().numpy()
        return [OptionalDistribution(cd, p=float(p[k]) if mask[k] else None,
                                     missing_value=self.kb.missing_value)
                for k, cd in enumerate(self.child.to_dists(child_raw))]

    def to_tensors(self, cols):
        idx, inner_cols = cols
        return (torch.as_tensor(idx, device=self.device, dtype=torch.int64),
                self.child.to_tensors(inner_cols))

    def log_density(self, params, cols):
        idx, inner_cols = cols
        miss = params['miss_lp'].unsqueeze(0)                        # (1, K)
        child_ld = self.child.log_density(params['child'], inner_cols)  # (m, K)
        if child_ld.shape[0] == 0:                                   # all rows missing
            return miss.expand(idx.shape[0], child_ld.shape[1]).clone()
        pres = params['pres_lp'].unsqueeze(0) + child_ld[idx.clamp(min=0)]
        return torch.where((idx >= 0).unsqueeze(1), pres, miss)

    def stats(self, params, cols, gamma):
        idx, inner_cols = cols
        present = idx >= 0
        pf = present.to(gamma.dtype).unsqueeze(1)
        miss_w = (gamma * (1.0 - pf)).sum(dim=0)
        pres_w = (gamma * pf).sum(dim=0)
        # compacted child rows were emitted in row order, so boolean indexing
        # delivers gamma in exactly the child-column order
        child_stats = self.child.stats(params['child'], inner_cols, gamma[present])
        return (miss_w.cpu().numpy(), pres_w.cpu().numpy(), child_stats)

    def default_prior(self, strength):
        # Beta(a, b) on the missing probability p; strength=0 -> Beta(1,1) flat
        return {'family': 'beta', 'a': 1.0 + 1.0e-3 * strength, 'b': 1.0 + 1.0e-3 * strength,
                'child': self.child.default_prior(strength)}

    def log_prior(self, params, prior):
        # components without p have miss_lp = pres_lp = 0, contributing nothing
        lp = ((prior['a'] - 1.0) * params['miss_lp']
              + (prior['b'] - 1.0) * params['pres_lp']).sum()
        return lp + self.child.log_prior(params['child'], prior['child'])


class _TIgnored(_TBase):
    """Fixed wrapped dists: cols[0] is the (n, K) log-density matrix the kernel
    builder precomputed at encode time. Nothing is estimated - no raw params,
    empty stats - and, mirroring the kernel path, scoring a model whose wrapped
    dists changed since compile raises."""

    def params(self, dists):
        for d, s in zip(dists, self.kb._frozen):
            if str(d.dist) != s:
                raise ValueError('IgnoredDistribution wrapped dists changed since compile; '
                                 'rebuild the TorchMixture.')
        return {}

    def raw_params(self, dists):
        self.params(dists)  # same fixed-dist validation; no parameters to fit
        return {}

    def canonical(self, raw):
        return {}

    def to_dists(self, raw):
        return list(self.kb.dists)

    def log_density(self, params, cols):
        return cols[0]

    def stats(self, params, cols, gamma):
        return (np.zeros(1),)

    def default_prior(self, strength):
        return None

    def log_prior(self, params, prior):
        return torch.zeros((), dtype=self.dtype, device=self.device)


class _TComposite(_TBase):

    def __init__(self, kb, device, dtype):
        super().__init__(kb, device, dtype)
        self.children = [_impl(b, device, dtype) for b in kb.child_builders]

    def _slots(self, dists):
        return [[d.dists[j] for d in dists] for j in range(len(self.children))]

    def params(self, dists):
        return [c.params(s) for c, s in zip(self.children, self._slots(dists))]

    def raw_params(self, dists):
        return [c.raw_params(s) for c, s in zip(self.children, self._slots(dists))]

    def canonical(self, raw):
        return [c.canonical(r) for c, r in zip(self.children, raw)]

    def to_dists(self, raw):
        per_slot = [c.to_dists(r) for c, r in zip(self.children, raw)]
        return [CompositeDistribution(tuple(slot[k] for slot in per_slot))
                for k in range(len(per_slot[0]))]

    def to_tensors(self, cols):
        flat = _k._CompositeB._untree(cols, len(self.children))
        return [c.to_tensors(f) for c, f in zip(self.children, flat)]

    def log_density(self, params, cols):
        out = self.children[0].log_density(params[0], cols[0])
        for c, p, cc in zip(self.children[1:], params[1:], cols[1:]):
            out = out + c.log_density(p, cc)
        return out

    def stats(self, params, cols, gamma):
        return _k._CompositeB._tree(c.stats(p, cc, gamma)
                                    for c, p, cc in zip(self.children, params, cols))

    def default_prior(self, strength):
        return [c.default_prior(strength) for c in self.children]

    def log_prior(self, params, priors):
        return sum(c.log_prior(p, pr) for c, p, pr in zip(self.children, params, priors))


class _TSequence(_TBase):

    def __init__(self, kb, device, dtype):
        super().__init__(kb, device, dtype)
        self.inner = _impl(kb.inner, device, dtype)
        self.len_i = _impl(kb.len_b, device, dtype) if kb.has_len else None

    def params(self, dists):
        inner_p = self.inner.params([d.dist for d in dists])
        len_p = self.len_i.params([d.len_dist for d in dists]) if self.len_i else None
        return inner_p, len_p

    def raw_params(self, dists):
        inner_p = self.inner.raw_params([d.dist for d in dists])
        len_p = self.len_i.raw_params([d.len_dist for d in dists]) if self.len_i else None
        return inner_p, len_p

    def canonical(self, raw):
        return (self.inner.canonical(raw[0]),
                self.len_i.canonical(raw[1]) if self.len_i else None)

    def to_dists(self, raw):
        inner_d = self.inner.to_dists(raw[0])
        if self.len_i:
            len_d = self.len_i.to_dists(raw[1])
            return [SequenceDistribution(d, len_dist=l) for d, l in zip(inner_d, len_d)]
        return [SequenceDistribution(d) for d in inner_d]

    def to_tensors(self, cols):
        off, inner_cols, len_cols = cols
        off_t = torch.as_tensor(off, device=self.device)
        doc_id = torch.repeat_interleave(
            torch.arange(len(off) - 1, device=self.device),
            torch.as_tensor(np.diff(off), device=self.device))
        return (off_t, doc_id, self.inner.to_tensors(inner_cols),
                self.len_i.to_tensors(len_cols) if self.len_i else None)

    def log_density(self, params, cols):
        off, doc_id, inner_cols, len_cols = cols
        elem = self.inner.log_density(params[0], inner_cols)        # (T, K)
        n = off.shape[0] - 1
        out = torch.zeros((n, elem.shape[1]), dtype=elem.dtype, device=elem.device)
        out.index_add_(0, doc_id, elem)
        if self.len_i:
            out = out + self.len_i.log_density(params[1], len_cols)
        return out

    def stats(self, params, cols, gamma):
        off, doc_id, inner_cols, len_cols = cols
        inner_stats = self.inner.stats(params[0], inner_cols, gamma[doc_id])
        len_stats = self.len_i.stats(params[1], len_cols, gamma) if self.len_i \
            else self.len_i_placeholder()
        return inner_stats, len_stats

    def len_i_placeholder(self):
        return _k._PoissonB([PoissonDistribution(1.0)]).make_stats(1)

    def default_prior(self, strength):
        return (self.inner.default_prior(strength),
                self.len_i.default_prior(strength) if self.len_i else None)

    def log_prior(self, params, priors):
        lp = self.inner.log_prior(params[0], priors[0])
        if self.len_i and priors[1] is not None:
            lp = lp + self.len_i.log_prior(params[1], priors[1])
        return lp


_IMPLS = {
    _k._GaussianB: _TGaussian,
    _k._LogGaussianB: _TLogGaussian,
    _k._GammaB: _TGamma,
    _k._CategoricalB: _TCategorical,
    _k._IntRangeB: _TIntRange,
    _k._PoissonB: _TPoisson,
    _k._ExponentialB: _TExponential,
    _k._GeometricB: _TGeometric,
    _k._BinomialB: _TBinomial,
    _k._DiagGaussianB: _TDiagGaussian,
    _k._CompositeB: _TComposite,
    _k._SequenceB: _TSequence,
    _k._OptionalB: _TOptional,
    _k._IgnoredB: _TIgnored,
}


def _impl(kb, device, dtype):
    t = type(kb)
    if t not in _IMPLS:
        raise ValueError('no torch implementation for %s' % t.__name__)
    return _IMPLS[t](kb, device, dtype)


class TorchMixture(object):
    """Batched torch estimation engine for a mixture (or single distribution).

    Encoding is the same one-pass columnar flatten as pysp.stats.kernels; the
    columns are moved to `device` once and reused across EM/MLE iterations.
    """

    def __init__(self, model, device: str = 'cpu', dtype=torch.float64):
        if isinstance(model, MixtureDistribution):
            self.is_mixture = True
            comps = list(model.components)
        else:
            self.is_mixture = False
            comps = [model]

        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype
        self.kb = _k.build_kernel(comps)
        self.impl = _impl(self.kb, self.device, dtype)
        self.K = len(comps)

    # -- data --------------------------------------------------------------

    def encode(self, data):
        bufs, add = self.kb.new_sink()
        n = 0
        for x in data:
            add(x)
            n += 1
        return n, self.impl.to_tensors(self.kb.freeze(bufs))

    def _components(self, model):
        return list(model.components) if self.is_mixture else [model]

    def _log_w(self, model):
        return torch.as_tensor(np.asarray(model.log_w), dtype=self.dtype, device=self.device)

    # -- scoring -------------------------------------------------------------

    def seq_component_log_density(self, enc, model=None) -> np.ndarray:
        model = self.model if model is None else model
        _, cols = enc
        with torch.no_grad():
            params = self.impl.params(self._components(model))
            return self.impl.log_density(params, cols).cpu().numpy()

    def _mix_ll(self, comp_ll, log_w):
        if not self.is_mixture:
            return comp_ll[:, 0]
        return torch.logsumexp(comp_ll + log_w.unsqueeze(0), dim=1)

    def seq_log_density(self, enc, model=None) -> np.ndarray:
        model = self.model if model is None else model
        _, cols = enc
        with torch.no_grad():
            params = self.impl.params(self._components(model))
            comp_ll = self.impl.log_density(params, cols)
            if not self.is_mixture:
                return comp_ll[:, 0].cpu().numpy()
            return self._mix_ll(comp_ll, self._log_w(model)).cpu().numpy()

    def posteriors(self, enc, model=None) -> torch.Tensor:
        model = self.model if model is None else model
        _, cols = enc
        with torch.no_grad():
            params = self.impl.params(self._components(model))
            ll = self.impl.log_density(params, cols)
            if self.is_mixture:
                ll = ll + self._log_w(model).unsqueeze(0)
            return torch.softmax(ll, dim=1)

    # -- EM (legacy M-steps) ---------------------------------------------------

    def weighted_suff_stats(self, enc, gamma, model=None):
        model = self.model if model is None else model
        _, cols = enc
        if not torch.is_tensor(gamma):
            gamma = torch.as_tensor(np.asarray(gamma), dtype=self.dtype, device=self.device)
        with torch.no_grad():
            params = self.impl.params(self._components(model))
            stats = self.impl.stats(params, cols, gamma)
        comp_ss = tuple(self.kb.stats_to_ss(stats, k) for k in range(self.K))
        if self.is_mixture:
            return gamma.sum(dim=0).cpu().numpy(), comp_ss
        return comp_ss[0]

    def em_step(self, enc, estimator, model=None, weights=None):
        model = self.model if model is None else model
        n, _ = enc
        gamma = self.posteriors(enc, model)
        if weights is not None:
            w = torch.as_tensor(np.asarray(weights), dtype=self.dtype, device=self.device)
            gamma = gamma * w.unsqueeze(1)
        return estimator.estimate(n, self.weighted_suff_stats(enc, gamma, model))

    def initialize(self, enc, estimator, rng: np.random.RandomState, p: float = 0.1):
        n, _ = enc
        keep = (rng.rand(n) <= p).astype(np.float64)
        gamma = rng.dirichlet(np.ones(self.K) / (self.K ** 2), size=n) * keep[:, None]
        return estimator.estimate(n, self.weighted_suff_stats(enc, gamma))

    def fit(self, enc, estimator, max_its: int = 100, delta: float = 1.0e-8,
            rng: Optional[np.random.RandomState] = None, init_p: float = 0.1,
            model=None, out=None):
        """EM to convergence with batched tensor passes. Returns (model, ll)."""
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

    # -- gradient MLE / MAP (the new capability) --------------------------------

    def fit_mle(self, enc, model=None, max_its: int = 500, lr: float = 0.05,
                optimizer: str = 'adam', tol: float = 1.0e-7, out=None,
                print_iter: int = 100):
        """Maximize the exact mixture log-likelihood by gradient ascent.

        All parameters are reparameterized into unconstrained tensors (category
        logits, log-variances/rates, mixture-weight logits) and optimized with
        autograd. Needs no closed-form M-step, so penalties or partially frozen
        parameters are one-line extensions. Returns (model, log_likelihood).
        """
        return self._fit_gradient(enc, model, None, None, max_its, lr, optimizer,
                                  tol, out, print_iter, 'MLE')

    def fit_map(self, enc, model=None, priors=None, prior_strength: float = 1.0,
                w_alpha: Optional[float] = None, max_its: int = 500, lr: float = 0.05,
                optimizer: str = 'adam', tol: float = 1.0e-7, out=None,
                print_iter: int = 100):
        """Maximize the posterior log p(data | theta) + log p(theta) by gradient ascent.

        Priors are conjugate families on the *canonical* parameters (the mode is
        parameterization-dependent, and the canonical mode is what users
        expect): NormalGamma on Gaussian (mu, 1/sigma^2) - per dimension for
        DiagonalGaussian and on the log-scale parameters for LogGaussian -
        Dirichlet on categorical/integer-categorical tables and on the mixture
        weights, Gamma on Poisson/Exponential rates, Gamma on the gamma rate
        1/theta and shape k, Beta on geometric/binomial p and on the Optional
        missing probability (Ignored has nothing to fit).

        priors=None builds weakly-informative defaults scaled by prior_strength
        (prior_strength=0 reduces exactly to fit_mle); pass a nested structure
        mirroring the model tree (lists for composite slots, (inner, length)
        pairs for sequences, dicts at the leaves) to customize. w_alpha sets
        the Dirichlet concentration on mixture weights (default 1 +
        prior_strength / K).

        Returns (model, log_posterior).
        """
        if priors is None:
            priors = self.impl.default_prior(prior_strength)
        if w_alpha is None:
            w_alpha = 1.0 + prior_strength / max(self.K, 1)
        return self._fit_gradient(enc, model, priors, float(w_alpha), max_its, lr,
                                  optimizer, tol, out, print_iter, 'MAP')

    def _fit_gradient(self, enc, model, priors, w_alpha, max_its, lr, optimizer,
                      tol, out, print_iter, tag):
        model = self.model if model is None else model
        _, cols = enc

        raw = self.impl.raw_params(self._components(model))
        leaves: List[torch.Tensor] = []

        def collect(r):
            if isinstance(r, dict):
                leaves.extend(r.values())
            elif isinstance(r, (list, tuple)):
                for u in r:
                    if u is not None:
                        collect(u)
        collect(raw)

        w_logits = None
        if self.is_mixture:
            w_logits = torch.log(torch.as_tensor(np.asarray(model.w), dtype=self.dtype,
                                                 device=self.device).clamp(min=1e-12))
            w_logits = w_logits.clone().requires_grad_(True)
            leaves.append(w_logits)

        opt_cls = {'adam': torch.optim.Adam, 'lbfgs': torch.optim.LBFGS}[optimizer]
        opt = opt_cls(leaves, lr=lr)

        def neg_objective():
            params = self.impl.canonical(raw)
            ll = self.impl.log_density(params, cols)
            if self.is_mixture:
                log_w = torch.log_softmax(w_logits, dim=0)
                ll = torch.logsumexp(ll + log_w.unsqueeze(0), dim=1)
            else:
                ll = ll[:, 0]
            obj = ll.sum()
            if priors is not None:
                obj = obj + self.impl.log_prior(params, priors)
                if self.is_mixture and w_alpha is not None and w_alpha != 1.0:
                    obj = obj + (w_alpha - 1.0) * torch.log_softmax(w_logits, dim=0).sum()
            return -obj

        last = float('inf')
        for it in range(max_its):
            if optimizer == 'lbfgs':
                def closure():
                    opt.zero_grad()
                    loss = neg_objective()
                    loss.backward()
                    return loss
                loss = opt.step(closure)
            else:
                opt.zero_grad()
                loss = neg_objective()
                loss.backward()
                opt.step()
            cur = float(loss.detach())
            if out is not None and (it + 1) % print_iter == 0:
                out.write('%s iteration %d: objective=%e\n' % (tag, it + 1, -cur))
            if abs(last - cur) < tol * max(1.0, abs(cur)):
                last = cur
                break
            last = cur

        comps = self.impl.to_dists(raw)
        if self.is_mixture:
            w = torch.softmax(w_logits, dim=0).detach().cpu().numpy()
            fitted = MixtureDistribution(comps, list(w / w.sum()))
        else:
            fitted = comps[0]
        return fitted, -last
