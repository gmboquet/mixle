"""Bayesian inference for pysp.ppl: parameter MCMC and MAP on a shared joint target.

A model whose parameter slots hold *distributions* (priors) or ``free`` defines a joint
``log p(data | theta) + log p(theta)``. Both MAP (maximize) and MCMC (sample) run on the
exact same target, scored with the existing vectorized ``seq_log_density`` and pysp's
``pysp.utils.mcmc`` kernels — no new inference engine.

Scope (this slice): flat ``Sample`` models (e.g. ``Normal(Normal(0,10), free)``). Latent
composites and the fast conjugate VB E-step are separate slices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from pysp.ppl.core import RandomVariable, free, lower, CompositeFamily

_NEG_INF = -1e300


@dataclass
class _Slot:
    index: int           # position in the family's argument tuple
    prior: Any           # a concrete prior distribution, or None for a flat `free` slot
    positive: bool       # sampled in log-space when True
    name: Optional[str]  # parameter name (prior's name, else "argN")
    handle: Any          # the prior RandomVariable (for .posterior(handle)), or None


class Posterior:
    """Parameter-posterior result attached to a fitted RV's ``.result``.

    Holds value-space draws and the raw ``MCMCResult``. Look up a parameter by its
    RandomVariable handle, its name, or its slot index.
    """

    def __init__(self, slots: List[_Slot], value_samples: np.ndarray, raw: Any):
        self._slots = slots
        self._samples = value_samples            # (n_draws, n_params), value space
        self.raw = raw
        self.acceptance_rate = getattr(raw, "acceptance_rate", None)
        self.predictive = None                   # set by the fitter (posterior predictive)

    def _col(self, param) -> int:
        for k, s in enumerate(self._slots):
            if param is s.handle or param == s.name or param == s.index:
                return k
        raise KeyError(f"no sampled parameter matching {param!r}")

    def samples(self, param=None) -> np.ndarray:
        if param is None:
            return self._samples
        return self._samples[:, self._col(param)]

    def mean(self, param=None):
        return self.samples(param).mean(axis=0)

    def summary(self) -> dict:
        out = {}
        for k, s in enumerate(self._slots):
            col = self._samples[:, k]
            out[s.name] = {
                "mean": float(col.mean()), "std": float(col.std()),
                "q2.5": float(np.percentile(col, 2.5)),
                "q97.5": float(np.percentile(col, 97.5)),
            }
        out["_acceptance_rate"] = self.acceptance_rate
        return out


# --------------------------------------------------------------------------- core
def _require_flat(rv: RandomVariable):
    if rv._kind != "sample" or isinstance(rv._family, CompositeFamily):
        raise NotImplementedError(
            "parameter MCMC/MAP currently supports flat models like "
            "Normal(Normal(0,10), free); composites are a later slice."
        )
    return rv._family


def _slots_of(rv: RandomVariable, fam) -> List[_Slot]:
    slots: List[_Slot] = []
    for i, a in enumerate(rv._args):
        if isinstance(a, RandomVariable):
            slots.append(_Slot(i, lower(a, target="dist"), fam.positive[i],
                               a.name or f"arg{i}", a))
        elif a is free:
            slots.append(_Slot(i, None, fam.positive[i], f"arg{i}", None))
    if not slots:
        raise ValueError("model has no `free`/prior parameters to infer.")
    return slots


def _encoder_for(fam):
    kwargs = fam.seed_at(0.0, 1.0) if fam.seed_at else fam.to_dist(*([1.0] * fam.arity))
    return fam.dist_cls(**kwargs).dist_to_encoder()


def _build_target(rv: RandomVariable, data):
    """Return (log_target(u), slots, fam, enc, (dmean,dstd)) for unconstrained vector u."""
    fam = _require_flat(rv)
    slots = _slots_of(rv, fam)
    arr = np.asarray(data, dtype=float)
    dmean, dstd = float(arr.mean()), float(arr.std() or 1.0)
    enc = _encoder_for(fam).seq_encode(list(data))

    def unpack(u):
        vals, logj = {}, 0.0
        for k, s in enumerate(slots):
            if s.positive:
                v = math.exp(u[k]); logj += float(u[k])   # Jacobian of exp transform
            else:
                v = float(u[k])
            vals[s.index] = v
        return vals, logj

    def build(vals):
        args = [vals.get(i, rv._args[i]) for i in range(len(rv._args))]
        return fam.make_dist(tuple(args), rv._name)

    def log_target(u):
        vals, logj = unpack(u)
        try:
            d = build(vals)
            ll = float(np.sum(d.seq_log_density(enc)))
        except Exception:
            return _NEG_INF
        if not math.isfinite(ll):
            return _NEG_INF
        plp = 0.0
        for s in slots:
            if s.prior is not None:
                plp += float(s.prior.log_density(vals[s.index]))
        return ll + plp + logj

    return log_target, slots, fam, build, unpack, (dmean, dstd)


def _init_u(slots, dmean, dstd) -> np.ndarray:
    u0 = []
    for s in slots:
        if s.positive:
            u0.append(math.log(max(dstd, 1e-2)))
        else:
            u0.append(dmean)
    return np.asarray(u0, dtype=float)


def _init_scale(slots, dstd, n) -> np.ndarray:
    """Per-slot proposal scale ~ posterior width: location std ~ dstd/sqrt(n); a
    log-space (positive) slot ~ 1/sqrt(n). Adaptation then tunes the magnitude."""
    root = math.sqrt(max(n, 1))
    return np.asarray(
        [max((1.0 if s.positive else dstd) / root, 1e-3) for s in slots], dtype=float)


def _finalize(rv, slots, res, build) -> RandomVariable:
    """Convert unconstrained chain samples to value space, build the posterior-mean
    distribution, and attach a Posterior result. Shared by RW-MCMC and HMC."""
    u = np.asarray(res.samples, dtype=float).reshape(len(res.samples), -1)
    vals = np.empty_like(u)
    for k, s in enumerate(slots):
        vals[:, k] = np.exp(u[:, k]) if s.positive else u[:, k]
    mean_vals = {s.index: float(vals[:, k].mean()) for k, s in enumerate(slots)}
    post = Posterior(slots, vals, res)

    def predictive(n, rng):
        idx = rng.randint(len(vals), size=n)
        out = []
        for j in idx:
            d = build({s.index: float(vals[j, k]) for k, s in enumerate(slots)})
            out.append(d.sampler(seed=int(rng.randint(1, 2 ** 31))).sample())
        return np.asarray(out)

    post.predictive = predictive
    return RandomVariable._bound(build(mean_vals), name=rv._name, result=post)


def mcmc_fit(rv: RandomVariable, data, *, draws: int = 2000, burn: int = 1000,
             thin: int = 1, scale: Optional[float] = None, rng=None) -> RandomVariable:
    from pysp.utils.mcmc import metropolis_hastings, AdaptiveRandomWalkProposal

    if rng is None:
        rng = np.random.RandomState()
    log_target, slots, fam, build, unpack, (dmean, dstd) = _build_target(rv, data)
    u0 = _init_u(slots, dmean, dstd)
    init_scale = (scale * np.ones(len(u0))) if scale is not None \
        else _init_scale(slots, dstd, len(data))
    proposal = AdaptiveRandomWalkProposal(init_scale)
    res = metropolis_hastings(log_target, u0, proposal, num_samples=draws,
                              burn_in=burn, thin=thin, rng=rng)
    return _finalize(rv, slots, res, build)


def hmc_fit(rv: RandomVariable, data, *, draws: int = 1000, burn: int = 500,
            step_size: Optional[float] = None, num_steps: int = 15, thin: int = 1,
            rng=None) -> RandomVariable:
    """Hamiltonian Monte Carlo over the parameter posterior.

    Uses pysp's ``hamiltonian_monte_carlo`` with a numerical gradient of the joint
    log-target and a diagonal mass matrix preconditioned to the data-informed posterior
    scale, so trajectories are well-conditioned without manual tuning.
    """
    from pysp.utils.mcmc import hamiltonian_monte_carlo

    if rng is None:
        rng = np.random.RandomState()
    log_target, slots, fam, build, unpack, (dmean, dstd) = _build_target(rv, data)
    u0 = _init_u(slots, dmean, dstd)
    scale = _init_scale(slots, dstd, len(data))     # ~ posterior std per dim
    mass = 1.0 / (scale ** 2)                        # precondition: M ~ inverse posterior cov
    if step_size is None:
        step_size = 2.5 / num_steps                  # tuned: acc~0.98, near-max ESS (preconditioned)

    eps = 1e-5 * np.maximum(np.abs(u0), 1.0)

    def grad(u):
        u = np.asarray(u, dtype=float)
        g = np.empty(len(u))
        for i in range(len(u)):
            up = u.copy(); up[i] += eps[i]
            um = u.copy(); um[i] -= eps[i]
            g[i] = (log_target(up) - log_target(um)) / (2.0 * eps[i])
        return g

    res = hamiltonian_monte_carlo(log_target, grad, u0, num_samples=draws,
                                  step_size=step_size, num_steps=num_steps, mass=mass,
                                  burn_in=burn, thin=thin, rng=rng)
    return _finalize(rv, slots, res, build)


# ---------------------------------------------------- closed-form conjugate Bayes
class ConjugatePosterior:
    """Exact closed-form posterior over a conjugate parameter.

    ``post`` maps parameter name -> {mean, sample(n, rng), name, hyper}. This is the
    ideal case VB approximates: exact, instant, no iteration.
    """

    def __init__(self, post: dict):
        self.post = post
        self.acceptance_rate = None
        self.predictive = None

    def _entry(self, param):
        for nm, e in self.post.items():
            if param == nm or param == e["index"] or param is e["handle"]:
                return e
        raise KeyError(f"no conjugate parameter matching {param!r}")

    def samples(self, param=None, n: int = 4000, rng=None):
        rng = rng or np.random.RandomState()
        if param is None:
            return {nm: e["sample"](n, rng) for nm, e in self.post.items()}
        return self._entry(param)["sample"](n, rng)

    def mean(self, param=None):
        if param is None:
            return {nm: e["mean"] for nm, e in self.post.items()}
        return self._entry(param)["mean"]

    def summary(self) -> dict:
        return {nm: {"mean": e["mean"], "posterior": e["name"], "hyper": e["hyper"]}
                for nm, e in self.post.items()}


def _conj_normal_mean(prior_args, fixed, stats, handle, index):
    m0, s0 = float(prior_args[0]), float(prior_args[1])  # prior mean, sd
    sigma2 = float(fixed[1]) ** 2                          # known variance (slot 1)
    n, sx = stats["n"], stats["sum"]
    prec = 1.0 / s0 ** 2 + n / sigma2
    pm = (m0 / s0 ** 2 + sx / sigma2) / prec
    pv = 1.0 / prec
    return {"index": index, "handle": handle, "name": "Normal",
            "mean": pm, "hyper": {"mean": pm, "sd": math.sqrt(pv)},
            "sample": lambda k, rng: rng.normal(pm, math.sqrt(pv), k)}


def _conj_poisson_gamma(prior_args, fixed, stats, handle, index):
    a, b = float(prior_args[0]), float(prior_args[1])     # Gamma(shape, rate) prior
    n, sx = stats["n"], stats["sum"]
    A, B = a + sx, b + n
    return {"index": index, "handle": handle, "name": "Gamma",
            "mean": A / B, "hyper": {"shape": A, "rate": B},
            "sample": lambda k, rng: rng.gamma(A, 1.0 / B, k)}


def _conj_exponential_gamma(prior_args, fixed, stats, handle, index):
    a, b = float(prior_args[0]), float(prior_args[1])     # Gamma prior on rate
    n, sx = stats["n"], stats["sum"]
    A, B = a + n, b + sx
    return {"index": index, "handle": handle, "name": "Gamma",
            "mean": A / B, "hyper": {"shape": A, "rate": B},
            "sample": lambda k, rng: rng.gamma(A, 1.0 / B, k)}


def _conj_bernoulli_beta(prior_args, fixed, stats, handle, index):
    a, b = float(prior_args[0]), float(prior_args[1])
    n, sx = stats["n"], stats["sum"]
    A, B = a + sx, b + n - sx
    return {"index": index, "handle": handle, "name": "Beta",
            "mean": A / (A + B), "hyper": {"a": A, "b": B},
            "sample": lambda k, rng: rng.beta(A, B, k)}


# (likelihood family, slot index, prior family) -> closed-form posterior builder
_CONJUGATE = {
    ("Normal", 0, "Normal"): _conj_normal_mean,       # unknown mean, known variance
    ("Poisson", 0, "Gamma"): _conj_poisson_gamma,
    ("Exponential", 0, "Gamma"): _conj_exponential_gamma,
    ("Bernoulli", 0, "Beta"): _conj_bernoulli_beta,
}


def conjugate_spec(rv: RandomVariable):
    """Return (builder, prior_slot_index, prior_rv) if exactly one slot is a conjugate
    prior and every other slot is a fixed constant; else None.
    """
    if rv._kind != "sample" or isinstance(rv._family, CompositeFamily):
        return None
    prior_slots = [(i, a) for i, a in enumerate(rv._args) if isinstance(a, RandomVariable)]
    if len(prior_slots) != 1:
        return None
    if any(a is free for a in rv._args):
        return None  # other params must be known for textbook conjugacy
    i, prior_rv = prior_slots[0]
    if prior_rv._kind != "sample" or isinstance(prior_rv._family, CompositeFamily):
        return None
    key = (rv._family.name, i, prior_rv._family.name)
    builder = _CONJUGATE.get(key)
    if builder is None:
        return None
    return builder, i, prior_rv


def conjugate_fit(rv: RandomVariable, data) -> RandomVariable:
    spec = conjugate_spec(rv)
    if spec is None:
        raise NotImplementedError("model is not a registered conjugate pair.")
    builder, idx, prior_rv = spec
    fam = rv._family
    arr = np.asarray(data, dtype=float)
    stats = {"n": float(arr.size), "sum": float(arr.sum()), "sum2": float((arr * arr).sum())}
    fixed = {i: rv._args[i] for i in range(len(rv._args)) if i != idx}
    entry = builder(prior_rv._args, fixed, stats, prior_rv, idx)
    name = prior_rv.name or f"arg{idx}"
    # build the fitted likelihood at the posterior-mean parameter
    full = [entry["mean"] if i == idx else rv._args[i] for i in range(len(rv._args))]
    fitted = fam.make_dist(tuple(full), rv._name)
    cpost = ConjugatePosterior({name: entry})

    def predictive(n, rng):
        pvals = np.atleast_1d(entry["sample"](n, rng))
        out = []
        for v in pvals:
            args = [float(v) if i == idx else rv._args[i] for i in range(len(rv._args))]
            d = fam.make_dist(tuple(args), rv._name)
            out.append(d.sampler(seed=int(rng.randint(1, 2 ** 31))).sample())
        return np.asarray(out)

    cpost.predictive = predictive
    return RandomVariable._bound(fitted, name=rv._name, result=cpost)


# ----------------------------------------------- hierarchical random effects (VB/EM)
class HierarchicalPosterior:
    """Per-group posteriors q(mu_i) = Normal(group_means[i], group_vars[i]) plus the
    fitted hyperparameters of a Normal-Normal random-effects model.
    """

    def __init__(self, group_means, group_vars, hyper):
        self.group_means = np.asarray(group_means)
        self.group_vars = np.asarray(group_vars)
        self.hyper = hyper            # {'m':..., 'tau':..., 'sigma':...}
        self.acceptance_rate = None

    def samples(self, param=None):
        # per-group posterior mean (the random effects)
        return self.group_means

    def summary(self) -> dict:
        return {"hyper": self.hyper, "n_groups": int(self.group_means.size),
                "group_means": self.group_means}


def _group_stats(data):
    groups = [np.asarray(g, dtype=float).reshape(-1) for g in data]
    n_i = np.array([g.size for g in groups], dtype=float)
    sum_i = np.array([g.sum() for g in groups], dtype=float)
    sumsq_i = np.array([float((g * g).sum()) for g in groups], dtype=float)
    return n_i, sum_i, sumsq_i


def _hier_normal_normal(rv, n_i, sum_i, sumsq_i, max_its, tol):
    """mu_i ~ Normal(m, tau^2); y_ij ~ Normal(mu_i, sigma^2). Exact conjugate EM."""
    from pysp.stats.leaf.gaussian import GaussianDistribution
    N = float(n_i.sum())
    gbar = sum_i / np.maximum(n_i, 1.0)
    m, tau2 = float(gbar.mean()), float(gbar.var()) or 1.0
    sigma_arg = rv._args[1]
    sigma_fixed = not (sigma_arg is free or isinstance(sigma_arg, RandomVariable))
    sigma2 = float(sigma_arg) ** 2 if sigma_fixed else max(float((sumsq_i.sum() / N) - (sum_i.sum() / N) ** 2), 1e-3)
    prev = None
    for _ in range(max_its):
        v_i = 1.0 / (1.0 / tau2 + n_i / sigma2)
        mhat = (m / tau2 + sum_i / sigma2) * v_i
        m = float(mhat.mean())
        tau2 = max(float(np.mean(mhat ** 2 + v_i) - m ** 2), 1e-8)
        if not sigma_fixed:
            resid = sumsq_i - 2.0 * mhat * sum_i + n_i * (mhat ** 2 + v_i)
            sigma2 = max(float(resid.sum() / N), 1e-8)
        cur = (m, tau2, sigma2)
        if prev is not None and max(abs(a - b) for a, b in zip(cur, prev)) < tol:
            break
        prev = cur
    pop = GaussianDistribution(mu=m, sigma2=tau2, name=rv._name)
    hyper = {"m": m, "tau": math.sqrt(tau2), "sigma": math.sqrt(sigma2)}
    return pop, mhat, v_i, hyper


def _hier_gamma_poisson(rv, n_i, sum_i, sumsq_i, max_its, tol):
    """lambda_i ~ Gamma(a, b); y_ij ~ Poisson(lambda_i). Conjugate E-step +
    moment-matched population M-step (law of total variance)."""
    from pysp.stats.leaf.gamma import GammaDistribution
    gm = sum_i / np.maximum(n_i, 1.0)
    m = float(gm.mean()); v = float(gm.var()) or m
    b = m / max(v, 1e-6); a = m * b
    prev = None
    for _ in range(max_its):
        A, B = a + sum_i, b + n_i                       # posterior Gamma(A_i, B_i) per group
        Elam, Vlam = A / B, A / (B * B)
        m = float(Elam.mean())
        v = float(np.var(Elam) + Vlam.mean())           # total variance
        b = m / max(v, 1e-8); a = m * b
        cur = (a, b)
        if prev is not None and max(abs(x - y) for x, y in zip(cur, prev)) < tol:
            break
        prev = cur
    pop = GammaDistribution(k=a, theta=1.0 / b, name=rv._name)   # population over rates
    hyper = {"shape": a, "rate": b, "mean": a / b}
    return pop, Elam, Vlam, hyper


def _hier_beta_bernoulli(rv, n_i, sum_i, sumsq_i, max_its, tol):
    """p_i ~ Beta(a, b); y_ij ~ Bernoulli(p_i). Conjugate E-step + moment-matched M-step."""
    from pysp.stats.leaf.beta import BetaDistribution
    gp = sum_i / np.maximum(n_i, 1.0)
    m = float(gp.mean()); v = float(gp.var()) or (m * (1 - m))
    s = max(m * (1 - m) / max(v, 1e-6) - 1, 1e-3); a = m * s; b = (1 - m) * s
    prev = None
    for _ in range(max_its):
        A, B = a + sum_i, b + (n_i - sum_i)             # posterior Beta(A_i, B_i)
        Ep = A / (A + B)
        Vp = A * B / ((A + B) ** 2 * (A + B + 1))
        m = float(Ep.mean())
        v = float(np.var(Ep) + Vp.mean())
        s = max(m * (1 - m) / max(v, 1e-8) - 1, 1e-3); a = m * s; b = (1 - m) * s
        cur = (a, b)
        if prev is not None and max(abs(x - y) for x, y in zip(cur, prev)) < tol:
            break
        prev = cur
    pop = BetaDistribution(a, b, name=rv._name)
    hyper = {"a": a, "b": b, "mean": a / (a + b)}
    return pop, Ep, Vp, hyper


# (likelihood family, prior family) -> hierarchical conjugate EM
_HIERARCHICAL = {
    ("Normal", "Normal"): _hier_normal_normal,
    ("Poisson", "Gamma"): _hier_gamma_poisson,
    ("Bernoulli", "Beta"): _hier_beta_bernoulli,
}


def hierarchical_fit(rv: RandomVariable, data, *, max_its: int = 300,
                     tol: float = 1e-8) -> RandomVariable:
    """Conjugate hierarchical (random-effects) EM, dispatched by conjugate pair.

    Supports Normal-Normal (exact), Gamma-Poisson, and Beta-Bernoulli. ``data`` is a list
    of groups; returns the fitted population distribution plus per-group posteriors.
    """
    fam = rv._family
    prior = rv._args[0]
    if not (isinstance(prior, RandomVariable) and prior._scope == "grouped"):
        raise NotImplementedError("expected a .each() group prior in slot 0.")
    key = (fam.name, prior._family.name)
    impl = _HIERARCHICAL.get(key)
    if impl is None:
        raise NotImplementedError(
            f"hierarchical pair {key} not supported; have {sorted(_HIERARCHICAL)}.")
    n_i, sum_i, sumsq_i = _group_stats(data)
    pop, group_means, group_vars, hyper = impl(rv, n_i, sum_i, sumsq_i, max_its, tol)
    post = HierarchicalPosterior(group_means, group_vars, hyper)
    return RandomVariable._bound(pop, name=rv._name, result=post)


def map_fit(rv: RandomVariable, data, *, rng=None) -> RandomVariable:
    from scipy.optimize import minimize

    log_target, slots, fam, build, unpack, (dmean, dstd) = _build_target(rv, data)
    u0 = _init_u(slots, dmean, dstd)
    res = minimize(lambda u: -log_target(u), u0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    vals, _ = unpack(res.x)
    return RandomVariable._bound(build(vals), name=rv._name)


class _VIResult:
    """Lightweight raw-result holder for a variational fit (mirrors MCMCResult's role)."""

    def __init__(self, elbo, mean, std):
        self.elbo = float(elbo)
        self.variational_mean = mean
        self.variational_std = std
        self.acceptance_rate = None


def vi_fit(rv: RandomVariable, data, *, samples: int = 4000, mc: int = 16,
           max_iter: int = 4000, rng=None) -> RandomVariable:
    """Mean-field variational Bayes (ADVI-style).

    Fits a diagonal-Gaussian variational posterior q(u) = N(mean, diag(std^2)) in the
    unconstrained space to the joint log-target, by maximizing a reparameterized
    Monte-Carlo ELBO (common random numbers -> smooth, deterministic objective optimized
    with scipy). Works for *non-conjugate* priors the closed-form registry can't handle;
    returns a variational Posterior with draws and posterior-predictive.
    """
    from scipy.optimize import minimize

    if rng is None:
        rng = np.random.RandomState()
    log_target, slots, fam, build, unpack, (dmean, dstd) = _build_target(rv, data)
    d = len(slots)
    u0 = _init_u(slots, dmean, dstd)
    s0 = _init_scale(slots, dstd, len(data))
    eps = rng.standard_normal((mc, d))                      # common random numbers

    half_entropy_const = 0.5 * d * (1.0 + math.log(2.0 * math.pi))

    def neg_elbo(phi):
        mean, log_std = phi[:d], phi[d:]
        std = np.exp(log_std)
        U = mean + std * eps                                # (mc, d) reparameterized
        ll = float(np.mean([log_target(U[i]) for i in range(mc)]))
        entropy = float(np.sum(log_std)) + half_entropy_const
        return -(ll + entropy)

    phi0 = np.concatenate([u0, np.log(s0)])
    res = minimize(neg_elbo, phi0, method="Nelder-Mead",
                   options={"maxiter": max_iter, "xatol": 1e-5, "fatol": 1e-5})
    mean, std = res.x[:d], np.exp(res.x[d:])

    Z = rng.standard_normal((samples, d))
    U = mean + std * Z
    vals = np.empty_like(U)
    for k, s in enumerate(slots):
        vals[:, k] = np.exp(U[:, k]) if s.positive else U[:, k]
    mean_vals = {s.index: float(vals[:, k].mean()) for k, s in enumerate(slots)}

    post = Posterior(slots, vals, _VIResult(-res.fun, mean, std))

    def predictive(n, r):
        idx = r.randint(len(vals), size=n)
        out = []
        for j in idx:
            dd = build({s.index: float(vals[j, k]) for k, s in enumerate(slots)})
            out.append(dd.sampler(seed=int(r.randint(1, 2 ** 31))).sample())
        return np.asarray(out)

    post.predictive = predictive
    return RandomVariable._bound(build(mean_vals), name=rv._name, result=post)
