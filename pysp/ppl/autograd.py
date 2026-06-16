"""Analytic-gradient joint log-target for pysp.ppl (Torch autograd).

The Bayesian fitters (MAP / HMC / VI) need the gradient of the joint
``log p(data | theta) + log p(theta)`` in the unconstrained space. Rather than
finite-difference it (slow, O(d) target evals per gradient) or use derivative-free
optimizers, this module builds the *same* target in Torch and differentiates it with
autograd — reusing each distribution's existing ``backend_log_density_from_params``
(no density math is reimplemented).

It is entirely optional: if Torch is missing, or any family in the model has no Torch
scorer (e.g. Categorical), :func:`grad_target` returns ``None`` and the caller falls back
to the numerical path. The target is numerically identical to
``pysp.ppl.inference._build_target`` so results and tests are unchanged — only faster.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pysp.ppl.core import CompositeFamily, RandomVariable, free


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


# --- per-family Torch scorers ------------------------------------------------------
# Each entry is (prep, apply):
#   prep(x_tensor, torch)              -> data_terms (constants; no grad needed)
#   apply(args, data_terms, x, eng)    -> per-point log-density tensor (grad flows via args)
# `args` are the conventional PPL arguments (the same order the user writes), as tensors
# for the inferred slots and python floats for fixed slots. Everything routes through the
# distribution's existing backend_log_density_from_params, so there is no duplicated math.
def _scorers():

    from pysp.stats.leaf.bernoulli import BernoulliDistribution
    from pysp.stats.leaf.beta import BetaDistribution
    from pysp.stats.leaf.binomial import BinomialDistribution
    from pysp.stats.leaf.exponential import ExponentialDistribution
    from pysp.stats.leaf.gamma import GammaDistribution
    from pysp.stats.leaf.gaussian import GaussianDistribution
    from pysp.stats.leaf.geometric import GeometricDistribution
    from pysp.stats.leaf.laplace import LaplaceDistribution
    from pysp.stats.leaf.log_gaussian import LogGaussianDistribution
    from pysp.stats.leaf.logistic import LogisticDistribution
    from pysp.stats.leaf.negative_binomial import NegativeBinomialDistribution
    from pysp.stats.leaf.pareto import ParetoDistribution
    from pysp.stats.leaf.poisson import PoissonDistribution
    from pysp.stats.leaf.rayleigh import RayleighDistribution
    from pysp.stats.leaf.student_t import StudentTDistribution
    from pysp.stats.leaf.weibull import WeibullDistribution

    G = GaussianDistribution.backend_log_density_from_params
    return {
        "Normal": (lambda x, t: (x,), lambda a, dt, x, e: G(x, a[0], a[1] ** 2, e)),
        "LogNormal": (
            lambda x, t: (x,),
            lambda a, dt, x, e: LogGaussianDistribution.backend_log_density_from_params(x, a[0], a[1] ** 2, e),
        ),
        "Exponential": (
            lambda x, t: (x,),
            lambda a, dt, x, e: ExponentialDistribution.backend_log_density_from_params(x, 1.0 / a[0], e),
        ),
        "Bernoulli": (
            lambda x, t: (x,),
            lambda a, dt, x, e: BernoulliDistribution.backend_log_density_from_params(x, a[0], e),
        ),
        "Geometric": (
            lambda x, t: (x,),
            lambda a, dt, x, e: GeometricDistribution.backend_log_density_from_params(x, a[0], e),
        ),
        "StudentT": (
            lambda x, t: (x,),
            lambda a, dt, x, e: StudentTDistribution.backend_log_density_from_params(x, a[0], a[1], a[2], e),
        ),
        "Poisson": (
            lambda x, t: (t.lgamma(x + 1.0),),
            lambda a, dt, x, e: PoissonDistribution.backend_log_density_from_params(x, dt[0], a[0], e),
        ),
        "Gamma": (
            lambda x, t: (t.log(x),),
            lambda a, dt, x, e: GammaDistribution.backend_log_density_from_params(x, dt[0], a[0], 1.0 / a[1], e),
        ),
        "Beta": (
            lambda x, t: (t.log(x), t.log1p(-x)),
            lambda a, dt, x, e: BetaDistribution.backend_log_density_from_params(dt[0], dt[1], a[0], a[1], e),
        ),
        "NegativeBinomial": (
            lambda x, t: (t.lgamma(x + 1.0),),
            lambda a, dt, x, e: NegativeBinomialDistribution.backend_log_density_from_params(x, dt[0], a[0], a[1], e),
        ),
        "Weibull": (
            lambda x, t: (t.log(x),),
            lambda a, dt, x, e: WeibullDistribution.backend_log_density_from_params(x, dt[0], a[0], a[1], e),
        ),
        "Laplace": (
            lambda x, t: (),
            lambda a, dt, x, e: LaplaceDistribution.backend_log_density_from_params(x, a[0], a[1], e),
        ),
        "Logistic": (
            lambda x, t: (),
            lambda a, dt, x, e: LogisticDistribution.backend_log_density_from_params(x, a[0], a[1], e),
        ),
        "Pareto": (
            lambda x, t: (t.log(x),),
            lambda a, dt, x, e: ParetoDistribution.backend_log_density_from_params(x, dt[0], a[0], a[1], e),
        ),
        "Rayleigh": (
            lambda x, t: (x * x, t.log(x)),
            lambda a, dt, x, e: RayleighDistribution.backend_log_density_from_params(x, dt[0], dt[1], a[0], e),
        ),
        "Binomial": (
            lambda x, t: (),
            lambda a, dt, x, e: BinomialDistribution.backend_log_density_from_params(x, a[0], a[1], None, e),
        ),
    }


class GradTarget:
    """A differentiable joint log-target over the unconstrained parameter vector ``u``.

    Mirrors ``inference._build_target`` exactly (data log-likelihood + prior log-densities
    + positivity Jacobian) but is backed by Torch autograd, so ``value_and_grad`` returns
    the analytic gradient. ``slots``/``build``/``unpack``/``dmean``/``dstd`` are shared with
    the numerical builder so the caller constructs results identically.
    """

    def __init__(self, rv, data, slots, build, unpack, dmean, dstd):
        import torch

        self._torch = torch
        self._rv = rv
        self._fam = rv._family
        self.slots = slots
        self.build = build
        self.unpack = unpack
        self.dmean = dmean
        self.dstd = dstd

        from pysp.engines import TorchEngine

        self._eng = TorchEngine(dtype="float64")
        self._scorers = _scorers()
        self._x = torch.tensor(np.asarray(data, dtype=float), dtype=torch.float64)
        prep, _ = self._scorers[self._fam.name]
        self._data_terms = prep(self._x, torch)
        # fixed (non-inferred) args as python floats
        self._fixed = {
            i: float(rv._args[i])
            for i in range(len(rv._args))
            if not (rv._args[i] is free or isinstance(rv._args[i], RandomVariable))
        }

    # -- the target -----------------------------------------------------------
    def _t(self, v):
        # coerce a constant to a float64 tensor (backend ops like gammaln need tensors)
        torch = self._torch
        return v if torch.is_tensor(v) else torch.as_tensor(float(v), dtype=torch.float64)

    def _logtarget_tensor(self, u):
        torch = self._torch
        vals: dict[int, Any] = {}
        logj = u.new_zeros(())
        for k, s in enumerate(self.slots):
            uk = u[k]
            if s.support == "positive":
                vals[s.index] = torch.exp(uk)
                logj = logj + uk
            elif s.support == "unit":
                v = torch.sigmoid(uk)
                vals[s.index] = v
                logj = logj + torch.log(v) + torch.log1p(-v)
            else:
                vals[s.index] = uk
        full = [vals[i] if i in vals else self._t(self._fixed[i]) for i in range(len(self._rv._args))]
        _, apply = self._scorers[self._fam.name]
        ll = apply(full, self._data_terms, self._x, self._eng).sum()
        plp = u.new_zeros(())
        for s in self.slots:
            if s.handle is not None:
                pf = s.handle._family.name
                theta = vals[s.index]
                prep_p, apply_p = self._scorers[pf]
                pargs = [self._t(z) for z in s.handle._args]
                xt = theta.reshape(1)
                plp = plp + apply_p(pargs, prep_p(xt, torch), xt, self._eng).sum()
        return ll + plp + logj

    def _logtarget_batch(self, U):
        """Vectorized joint log-target for a batch ``U`` of shape ``(B, d)`` -> ``(B,)``.

        Identical math to :meth:`_logtarget_tensor` but with a leading batch axis: each
        inferred parameter becomes ``(B, 1)`` and broadcasts against the ``(N,)`` data, so all
        ``B`` points are scored in a single pass (no Python loop). Used by the batched ADVI ELBO.
        """
        torch = self._torch
        B = U.shape[0]
        vals: dict[int, Any] = {}
        logj = U.new_zeros(B)
        for k, s in enumerate(self.slots):
            uk = U[:, k]
            if s.support == "positive":
                vals[s.index] = torch.exp(uk)
                logj = logj + uk
            elif s.support == "unit":
                v = torch.sigmoid(uk)
                vals[s.index] = v
                logj = logj + torch.log(v) + torch.log1p(-v)
            else:
                vals[s.index] = uk
        full = [vals[i].reshape(B, 1) if i in vals else self._t(self._fixed[i]) for i in range(len(self._rv._args))]
        _, apply = self._scorers[self._fam.name]
        ll = apply(full, self._data_terms, self._x, self._eng).sum(dim=1)  # (B, N) -> (B,)
        plp = U.new_zeros(B)
        for s in self.slots:
            if s.handle is not None:
                pf = s.handle._family.name
                prep_p, apply_p = self._scorers[pf]
                pargs = [self._t(z) for z in s.handle._args]
                xt = vals[s.index].reshape(B, 1)
                plp = plp + apply_p(pargs, prep_p(xt, torch), xt, self._eng).sum(dim=1)
        return ll + plp + logj

    def log_target(self, u_np) -> float:
        torch = self._torch
        with torch.no_grad():
            u = torch.tensor(np.asarray(u_np, dtype=float), dtype=torch.float64)
            v = self._logtarget_tensor(u)
        return float(v) if math.isfinite(float(v)) else -1e300

    def value_and_grad(self, u_np) -> tuple[float, np.ndarray]:
        torch = self._torch
        u = torch.tensor(np.asarray(u_np, dtype=float), dtype=torch.float64, requires_grad=True)
        v = self._logtarget_tensor(u)
        (g,) = torch.autograd.grad(v, u)
        return float(v.detach()), g.detach().numpy()

    def grad(self, u_np) -> np.ndarray:
        return self.value_and_grad(u_np)[1]

    # -- ADVI (reparameterized mean-field VB, Adam) ---------------------------
    def advi(
        self, u0, s0, *, samples: int, mc: int, steps: int, lr: float, rng
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimize a diagonal-Gaussian q(u)=N(mean, diag(std^2)) by maximizing a
        reparameterized Monte-Carlo ELBO with Adam. Returns (value_samples, mean_u, std_u)."""
        torch = self._torch
        d = len(self.slots)
        mean = torch.tensor(np.asarray(u0, dtype=float), dtype=torch.float64, requires_grad=True)
        log_std = torch.tensor(np.log(np.asarray(s0, dtype=float)), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([mean, log_std], lr=lr)
        half_entropy_const = 0.5 * d * (1.0 + math.log(2.0 * math.pi))
        gen = torch.Generator().manual_seed(int(rng.randint(1, 2**31)))
        for _ in range(steps):
            opt.zero_grad()
            eps = torch.randn((mc, d), dtype=torch.float64, generator=gen)
            std = torch.exp(log_std)
            U = mean + std * eps  # (mc, d) reparameterized draws
            ll = self._logtarget_batch(U).mean()  # all mc samples scored in one pass
            elbo = ll + log_std.sum() + half_entropy_const
            (-elbo).backward()
            opt.step()
        mean_np = mean.detach().numpy()
        std_np = torch.exp(log_std).detach().numpy()
        Z = rng.standard_normal((samples, d))
        U = mean_np + std_np * Z
        vals = np.empty_like(U)
        for k, s in enumerate(self.slots):
            if s.support == "positive":
                vals[:, k] = np.exp(U[:, k])
            elif s.support == "unit":
                vals[:, k] = 1.0 / (1.0 + np.exp(-U[:, k]))
            else:
                vals[:, k] = U[:, k]
        return vals, mean_np, std_np


def grad_target(rv: RandomVariable, data) -> GradTarget | None:
    """Build a Torch autograd target for a flat model, or ``None`` if unavailable.

    Returns ``None`` (caller falls back to the numerical path) when Torch is missing, the
    model is not a flat ``Sample``, or any family involved (likelihood or a prior) has no
    Torch scorer.
    """
    if not torch_available():
        return None
    if rv._kind != "sample" or isinstance(rv._family, CompositeFamily):
        return None
    from pysp.ppl.inference import _require_flat, _target_parts

    try:
        scorers = _scorers()
    except Exception:
        return None
    if rv._family.name not in scorers:
        return None
    # every prior's family must also have a Torch scorer
    for a in rv._args:
        if isinstance(a, RandomVariable):
            if isinstance(a._family, CompositeFamily) or a._family.name not in scorers:
                return None
    _require_flat(rv)
    fam, slots, build, unpack, (dmean, dstd) = _target_parts(rv, data)
    return GradTarget(rv, data, slots, build, unpack, dmean, dstd)
