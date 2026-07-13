"""Analytic-gradient joint log-target for mixle.ppl (Torch autograd).

The Bayesian fitters (MAP / HMC / VI) need the gradient of the joint
``log p(data | theta) + log p(theta)`` in the unconstrained space. Rather than
finite-difference it (slow, O(d) target evals per gradient) or use derivative-free
optimizers, this module builds the *same* target in Torch and differentiates it with
autograd — reusing each distribution's existing ``backend_log_density_from_params``
(no density math is reimplemented).

It is entirely optional: if Torch is missing, or any family in the model has no Torch
scorer (e.g. Categorical), :func:`grad_target` returns ``None`` and the caller falls back
to the numerical path. The target is numerically identical to
``mixle.ppl.inference._build_target`` so results and tests are unchanged — only faster.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mixle.inference._advi import _advi_optimize  # core-resident ADVI optimizer (ppl -> core)
from mixle.ppl.core import CompositeFamily, RandomVariable, free


def torch_available() -> bool:
    """Return whether Torch can be imported for analytic-gradient PPL routes."""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
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

    from mixle.stats.univariate.continuous.beta import BetaDistribution
    from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
    from mixle.stats.univariate.continuous.gamma import GammaDistribution
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
    from mixle.stats.univariate.continuous.laplace import LaplaceDistribution
    from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution
    from mixle.stats.univariate.continuous.logistic import LogisticDistribution
    from mixle.stats.univariate.continuous.pareto import ParetoDistribution
    from mixle.stats.univariate.continuous.rayleigh import RayleighDistribution
    from mixle.stats.univariate.continuous.student_t import StudentTDistribution
    from mixle.stats.univariate.continuous.weibull import WeibullDistribution
    from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution
    from mixle.stats.univariate.discrete.binomial import BinomialDistribution
    from mixle.stats.univariate.discrete.geometric import GeometricDistribution
    from mixle.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution
    from mixle.stats.univariate.discrete.poisson import PoissonDistribution

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

    ``jacobian=True`` (samplers / VI) includes the support-transform log|J|, making the target
    the joint density in the unconstrained space; ``jacobian=False`` (point estimates, MAP)
    scores the joint in the constrained parameter space, so flat priors reduce to the MLE.
    """

    def __init__(self, rv, data, slots, build, unpack, dmean, dstd, missing="error", jacobian=True):
        import torch

        self._torch = torch
        self._rv = rv
        self._fam = rv._family
        self.slots = slots
        self.build = build
        self.unpack = unpack
        self.dmean = dmean
        self.dstd = dstd
        self._jacobian = bool(jacobian)

        from mixle.engines import TorchEngine

        self._eng = TorchEngine(dtype="float64")
        self._scorers = _scorers()
        x_np = np.asarray(data, dtype=float)
        # missing='marginalize': a NaN observation is integrated out -> its per-point log-density is
        # zeroed in the sum (weight 0). Replace the NaN with a safe sentinel first so the scorer/data_terms
        # stay finite and no NaN poisons the gradient through the masked (zero-weight) branch.
        self._w = None
        if missing == "marginalize":
            nan = np.isnan(x_np)
            if nan.any():
                x_np = np.where(nan, 0.0, x_np)
                self._w = torch.tensor((~nan).astype(float), dtype=torch.float64)
        self._x = torch.tensor(x_np, dtype=torch.float64)
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

    def _loc_scale(self, s, vals):
        """(loc, scale) tensors of a non-centered Normal slot, from constants / hyperparameter slots."""
        pa = s.parent_args or {}
        loc = vals[pa[0]] if 0 in pa else self._t(s.handle._args[0])
        scale = vals[pa[1]] if 1 in pa else self._t(s.handle._args[1])
        return loc, scale

    def _logtarget_tensor(self, u):
        torch = self._torch
        vals: dict[int, Any] = {}
        logj = u.new_zeros(())
        for k, s in enumerate(self.slots):
            uk = u[k]
            if s.reparam == "loc_scale":  # uk is z ~ N(0,1); the parameter is loc + scale*z (non-centered)
                loc, scale = self._loc_scale(s, vals)
                vals[s.index] = loc + scale * uk
            elif s.support == "positive":
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
        lp = apply(full, self._data_terms, self._x, self._eng)
        ll = (lp * self._w).sum() if self._w is not None else lp.sum()  # missing rows have weight 0
        plp = u.new_zeros(())
        for s in self.slots:
            if s.reparam == "loc_scale":  # prior is N(0,1) on z = (value - loc)/scale
                loc, scale = self._loc_scale(s, vals)
                z = (vals[s.index] - loc) / scale
                plp = plp + (-0.5 * z * z - 0.5 * math.log(2.0 * math.pi))
            elif s.handle is not None:
                pf = s.handle._family.name
                theta = vals[s.index]
                prep_p, apply_p = self._scorers[pf]
                # hierarchical prior: substitute a sampled hyperparameter tensor for an RV-valued arg
                pargs = [
                    vals[s.parent_args[j]] if (s.parent_args and j in s.parent_args) else self._t(z)
                    for j, z in enumerate(s.handle._args)
                ]
                xt = theta.reshape(1)
                plp = plp + apply_p(pargs, prep_p(xt, torch), xt, self._eng).sum()
        if not self._jacobian:  # point-estimate objective: constrained-space density (no log|J|)
            return ll + plp
        return ll + plp + logj

    def _logtarget_batch(self, U, x=None, data_terms=None, lik_scale=1.0, w=None):
        """Vectorized joint log-target for a batch ``U`` of shape ``(B, d)`` -> ``(B,)``.

        Identical math to :meth:`_logtarget_tensor` but with a leading batch axis: each
        inferred parameter becomes ``(B, 1)`` and broadcasts against the ``(N,)`` data, so all
        ``B`` points are scored in a single pass (no Python loop). Used by the batched ADVI ELBO.
        ``x``/``data_terms``/``lik_scale`` select a data *minibatch* (the likelihood is summed over
        the subset and rescaled by ``lik_scale = N/B`` for an unbiased full-data estimate)."""
        torch = self._torch
        x = self._x if x is None else x
        data_terms = self._data_terms if data_terms is None else data_terms
        if w is None:
            w = self._w
        B = U.shape[0]
        vals: dict[int, Any] = {}
        logj = U.new_zeros(B)
        for k, s in enumerate(self.slots):
            uk = U[:, k]
            if s.reparam == "loc_scale":
                loc, scale = self._loc_scale(s, vals)
                vals[s.index] = loc + scale * uk
            elif s.support == "positive":
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
        lp = apply(full, data_terms, x, self._eng)  # (B, N)
        lp = lp * w if w is not None else lp  # missing rows have weight 0
        ll = lp.sum(dim=1) * lik_scale  # (B, N) -> (B,)
        plp = U.new_zeros(B)
        for s in self.slots:
            if s.reparam == "loc_scale":
                loc, scale = self._loc_scale(s, vals)
                z = (vals[s.index] - loc) / scale
                plp = plp + (-0.5 * z * z - 0.5 * math.log(2.0 * math.pi))
            elif s.handle is not None:
                pf = s.handle._family.name
                prep_p, apply_p = self._scorers[pf]
                pargs = [
                    vals[s.parent_args[j]].reshape(B, 1) if (s.parent_args and j in s.parent_args) else self._t(z)
                    for j, z in enumerate(s.handle._args)
                ]
                xt = vals[s.index].reshape(B, 1)
                plp = plp + apply_p(pargs, prep_p(xt, torch), xt, self._eng).sum(dim=1)
        if not self._jacobian:  # point-estimate objective: constrained-space density (no log|J|)
            return ll + plp
        return ll + plp + logj

    def log_target(self, u_np) -> float:
        """Evaluate the joint log-target in unconstrained coordinates."""
        torch = self._torch
        with torch.no_grad():
            u = torch.tensor(np.asarray(u_np, dtype=float), dtype=torch.float64)
            v = self._logtarget_tensor(u)
        return float(v) if math.isfinite(float(v)) else -1e300

    def value_and_grad(self, u_np) -> tuple[float, np.ndarray]:
        """Return the joint log-target value and gradient at ``u_np``."""
        torch = self._torch
        u = torch.tensor(np.asarray(u_np, dtype=float), dtype=torch.float64, requires_grad=True)
        v = self._logtarget_tensor(u)
        (g,) = torch.autograd.grad(v, u)
        return float(v.detach()), g.detach().numpy()

    def grad(self, u_np) -> np.ndarray:
        """Return only the gradient of the joint log-target at ``u_np``."""
        return self.value_and_grad(u_np)[1]

    # -- ADVI: mean-field or full-rank Gaussian q, KL or tilted (Renyi-alpha) objective --------
    def advi(
        self,
        u0,
        s0,
        *,
        samples: int,
        mc: int,
        steps: int,
        lr: float,
        rng,
        batch_size: int | None = None,
        family: str = "meanfield",
        alpha: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Fit a Gaussian variational posterior by reparameterized-MC stochastic optimization (Adam).

        ``family``: ``meanfield`` (diagonal q — independent params) or ``fullrank`` (a full
        covariance via a Cholesky factor — captures posterior correlations).
        ``alpha``: the Renyi / tilted objective ``L_alpha = 1/(1-alpha) log E_q[(p/q)^(1-alpha)]``.
        ``alpha=1`` is the usual KL-ELBO; ``alpha=0`` is the importance-weighted (IWAE) bound — both
        mass-covering directions that widen the often-too-narrow KL fit (the importance weights
        ``w=p/q`` are *tilted* by ``1-alpha``). ``batch_size`` subsamples the data per step (SGVB).
        Returns ``(value_samples, mean_u, scale_u, objective)`` where ``objective`` is the final
        variational objective value (ELBO for ``alpha=1``, otherwise the tilted Renyi bound)."""
        n_data = int(self._x.shape[0])
        use_mb = batch_size is not None and 0 < int(batch_size) < n_data

        def log_p_fn(u):  # batched joint log-target, optionally on a fresh data minibatch (SGVB)
            if use_mb:
                idx = self._torch.as_tensor(rng.choice(n_data, size=int(batch_size), replace=False))
                return self._logtarget_batch(
                    u,
                    self._x[idx],
                    tuple(t[idx] for t in self._data_terms),
                    n_data / float(batch_size),
                    w=None if self._w is None else self._w[idx],
                )
            return self._logtarget_batch(u)

        mean_np, scale_np, U, objective = _advi_optimize(
            self._torch,
            log_p_fn,
            u0,
            s0,
            samples=samples,
            mc=mc,
            steps=steps,
            lr=lr,
            rng=rng,
            family=family,
            alpha=alpha,
        )
        from mixle.ppl.inference import _u_to_vals

        # Shared with the samplers' readout: applies each slot's support transform AND the
        # non-centered back-transform ``value = loc + scale * z`` (``reparam == "loc_scale"``),
        # so the returned draws are parameter values, never z-space latents.
        vals = _u_to_vals(self.slots, U)
        return vals, mean_np, scale_np, objective


class MixtureGradTarget(GradTarget):
    """Differentiable joint log-target for a finite **mixture of leaf components** — the analytic
    composite case. Overrides the per-batch likelihood with ``logsumexp_k(log w_k + comp_k(x))``
    so HMC / NUTS / full-rank & tilted VB get analytic gradients over the component parameters and
    (Gamma-represented) mixture weights. Other combinators (HMM, sequence) stay on the numeric path.
    """

    def __init__(self, rv, data, slots, build, dmean, dstd, comp_layouts, weight, comp_family, jacobian=True):
        import torch

        from mixle.engines import TorchEngine

        self._torch = torch
        self._rv = rv
        self.slots = slots
        self.build = build
        self.unpack = None
        self.dmean = dmean
        self.dstd = dstd
        self._jacobian = bool(jacobian)
        self._eng = TorchEngine(dtype="float64")
        self._scorers = _scorers()
        self._x = torch.tensor(np.asarray(data, dtype=float), dtype=torch.float64)
        self._data_terms = self._scorers[comp_family][0](self._x, torch)  # prep shared across same-family comps
        self._comp_layouts = comp_layouts  # per comp: (fam_name, [('slot', idx, support) | ('const', value)])
        self._weight = weight  # ('fixed', w) | ('slots', [idx...], alpha)

    def _logtarget_tensor(self, u):
        return self._logtarget_batch(u.reshape(1, -1))[0]

    def _logtarget_batch(self, U):
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
        comp_ld = []  # each (B, N): per-point log density of component k
        for fam_name, layout in self._comp_layouts:
            prep, apply = self._scorers[fam_name]
            args = [vals[e[1]].reshape(B, 1) if e[0] == "slot" else self._t(e[1]) for e in layout]
            comp_ld.append(apply(args, self._data_terms, self._x, self._eng))
        stacked = torch.stack(comp_ld, dim=0)  # (K, B, N)
        if self._weight[0] == "fixed":
            w_const = torch.as_tensor(np.asarray(self._weight[1], dtype=float), dtype=torch.float64)
            log_w = torch.log(w_const).reshape(-1, 1, 1)  # (K,1,1)
        else:
            g = torch.stack([vals[i] for i in self._weight[1]], dim=1)  # (B, K)
            log_w = torch.log(g / g.sum(dim=1, keepdim=True)).T.reshape(len(self._weight[1]), B, 1)
        ll = torch.logsumexp(log_w + stacked, dim=0).sum(dim=1)  # (B, N) -> (B,)
        plp = U.new_zeros(B)
        for s in self.slots:
            if s.handle is not None:  # component-parameter prior
                prep_p, apply_p = self._scorers[s.handle._family.name]
                pargs = [self._t(z) for z in s.handle._args]
                xt = vals[s.index].reshape(B, 1)
                plp = plp + apply_p(pargs, prep_p(xt, torch), xt, self._eng).sum(dim=1)
        if self._weight[0] == "slots":  # Gamma(alpha_k, 1) prior on each unnormalized weight (Dirichlet rep)
            for j, wi in enumerate(self._weight[1]):
                a = float(self._weight[2][j])
                plp = plp + ((a - 1.0) * torch.log(vals[wi]) - vals[wi] - math.lgamma(a))
        if not self._jacobian:  # point-estimate objective: constrained-space density (no log|J|)
            return ll + plp
        return ll + plp + logj


def _mixture_grad_target(rv, data, scorers, jacobian=True):
    """Build a MixtureGradTarget for a Mix of same-family leaf components, or None if it isn't one
    (heterogeneous components, a composite component, or a missing Torch scorer)."""
    comps, weights_arg = rv._args[0], rv._args[1]
    if not comps:
        return None
    fam0 = comps[0]._family.name
    for c in comps:  # all components: same leaf family with a Torch scorer, scalar args only
        if c._kind != "sample" or isinstance(c._family, CompositeFamily) or c._family.name != fam0:
            return None
        if fam0 not in scorers:
            return None
        for a in c._args:
            if isinstance(a, RandomVariable) and (
                isinstance(a._family, CompositeFamily) or a._family.name not in scorers
            ):
                return None
    from mixle.ppl.inference import _collect_composite

    slots, _rebuild = _collect_composite(rv)
    build = lambda v: __import__("mixle.ppl.core", fromlist=["lower"]).lower(_rebuild(v), target="dist")  # noqa: E731
    # map slots -> (component, arg) and the trailing weight slots, replicating _collect_composite's order
    idx = 0
    comp_layouts = []
    for c in comps:
        layout = []
        for j, a in enumerate(c._args):
            if isinstance(a, RandomVariable) or a is free:
                layout.append(("slot", idx, c._family.support[j]))
                idx += 1
            else:
                layout.append(("const", float(a)))
        comp_layouts.append((c._family.name, layout))
    k = len(comps)
    if weights_arg is None or isinstance(weights_arg, np.ndarray):
        w = np.ones(k) / k if weights_arg is None else np.asarray(weights_arg, dtype=float)
        weight = ("fixed", w)
    else:  # free / Dirichlet -> the trailing K weight slots (Gamma representation)
        if isinstance(weights_arg, RandomVariable) and weights_arg._family.name == "Dirichlet":
            alpha = np.asarray(weights_arg._args[0], dtype=float)
        else:
            alpha = np.ones(k)
        weight = ("slots", list(range(idx, idx + k)), alpha)
        idx += k
    if idx != len(slots):  # structure we didn't model (e.g. nested) -> fall back
        return None
    arr = np.asarray(data, dtype=float)
    dmean, dstd = float(arr.mean()), float(arr.std() or 1.0)
    return MixtureGradTarget(rv, data, slots, build, dmean, dstd, comp_layouts, weight, fam0, jacobian=jacobian)


def grad_target(rv: RandomVariable, data, missing: str = "error", jacobian: bool = True):
    """Build a Torch autograd target for a flat model or a mixture-of-leaves, or ``None``.

    Returns ``None`` (caller falls back to the numerical path) when Torch is missing, or the model
    is neither a flat ``Sample`` nor a supported mixture, or any family has no Torch scorer.
    ``missing='marginalize'`` integrates NaN observations out of the likelihood (flat models only here).
    ``jacobian=False`` builds the constrained-space point-estimate objective (MAP; flat priors == MLE);
    the default keeps the support-transform log|J| the samplers and VI need.
    """
    if not torch_available() or rv._kind != "sample":
        return None
    try:
        scorers = _scorers()
    except Exception:  # noqa: BLE001
        return None
    if isinstance(rv._family, CompositeFamily):
        if rv._family.name == "Mixture":
            if missing == "marginalize":
                return None  # mixture-of-leaves missing handling not wired here; caller raises clearly
            try:
                return _mixture_grad_target(rv, data, scorers, jacobian=jacobian)
            except Exception:  # noqa: BLE001
                return None
        return None
    from mixle.ppl.inference import _is_det_expr, _require_flat, _target_parts

    if rv._family.name not in scorers:
        return None

    def _prior_ok(a) -> bool:  # every prior family (including hierarchical hyperparameters) needs a scorer
        if isinstance(a._family, CompositeFamily) or a._family.name not in scorers:
            return False
        return all(_prior_ok(z) for z in a._args if isinstance(z, RandomVariable))

    for a in rv._args:
        # Deterministic-expression slots (a + b, exp(a), ...) have no single prior family to score; there
        # is no autograd target for them yet, so bail out -- the gradient-free numerical MH path handles them.
        if _is_det_expr(a):
            return None
        if isinstance(a, RandomVariable) and not _prior_ok(a):
            return None
    _require_flat(rv)
    fam, slots, build, unpack, (dmean, dstd) = _target_parts(rv, data)
    return GradTarget(rv, data, slots, build, unpack, dmean, dstd, missing=missing, jacobian=jacobian)
