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
from typing import Any

import numpy as np

from pysp.ppl.core import CompositeFamily, RandomVariable, free, lower

_NEG_INF = -1e300


@dataclass
class _Slot:
    index: int           # position in the family's argument tuple
    prior: Any           # a concrete prior distribution, or None for a flat `free` slot
    positive: bool       # sampled in log-space when True (kept for back-compat; see `support`)
    name: str | None  # parameter name (prior's name, else "argN")
    handle: Any          # the prior RandomVariable (for .posterior(handle)), or None
    support: str = "real"  # 'real' | 'positive' (log) | 'unit' (logit) reparameterization


def _to_value(support: str, u: float):
    """Map an unconstrained scalar to the constrained parameter, with the log|Jacobian|."""
    if support == "positive":
        return math.exp(u), float(u)
    if support == "unit":
        v = 1.0 / (1.0 + math.exp(-u))
        return v, math.log(v) + math.log1p(-v)   # d sigmoid/du = v(1-v)
    return float(u), 0.0


def _to_u(support: str, val: float) -> float:
    """Inverse of :func:`_to_value` (constrained value -> unconstrained)."""
    if support == "positive":
        return math.log(max(float(val), 1e-12))
    if support == "unit":
        p = min(max(float(val), 1e-6), 1.0 - 1e-6)
        return math.log(p / (1.0 - p))
    return float(val)


class Posterior:
    """Parameter-posterior result attached to a fitted RV's ``.result``.

    Holds value-space draws and the raw ``MCMCResult``. Look up a parameter by its
    RandomVariable handle, its name, or its slot index.
    """

    def __init__(self, slots: list[_Slot], value_samples: np.ndarray, raw: Any):
        self._slots = slots
        self._samples = value_samples            # (n_draws, n_params), value space
        self.raw = raw
        self.acceptance_rate = getattr(raw, "acceptance_rate", None)
        self.predictive = None                   # set by the fitter (posterior predictive)
        self.rhat = None                         # {param: Gelman-Rubin R-hat} (multi-chain)
        self.ess = None                          # combined effective sample size (multi-chain)
        self.n_chains = 1

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
        if self.rhat is not None:
            out["_rhat"] = self.rhat
            out["_ess"] = self.ess
            out["_n_chains"] = self.n_chains
        return out


# --------------------------------------------------------------------------- core
def _require_flat(rv: RandomVariable):
    if rv._kind != "sample" or isinstance(rv._family, CompositeFamily):
        raise NotImplementedError(
            "parameter MCMC/MAP currently supports flat models like "
            "Normal(Normal(0,10), free); composites are a later slice."
        )
    return rv._family


def _slots_of(rv: RandomVariable, fam) -> list[_Slot]:
    slots: list[_Slot] = []
    for i, a in enumerate(rv._args):
        if isinstance(a, RandomVariable):
            slots.append(_Slot(i, lower(a, target="dist"), fam.positive[i],
                               a.name or f"arg{i}", a, fam.support[i]))
        elif a is free:
            slots.append(_Slot(i, None, fam.positive[i], f"arg{i}", None, fam.support[i]))
    if not slots:
        raise ValueError("model has no `free`/prior parameters to infer.")
    return slots


def _encoder_for(fam):
    # A valid probe distribution to obtain the data encoder. Use support-aware defaults
    # (0 for real, 1 for positive, 0.5 for unit) so bounded families (Bernoulli/Beta/...) are
    # constructed with in-range params; fall back to all-ones if a family needs it.
    if fam.seed_at:
        kwargs = fam.seed_at(0.0, 1.0)
    else:
        defaults = {"real": 0.0, "positive": 1.0, "unit": 0.5}
        try:
            kwargs = fam.to_dist(*[defaults[s] for s in fam.support])
        except Exception:
            kwargs = fam.to_dist(*([1.0] * fam.arity))
    return fam.dist_cls(**kwargs).dist_to_encoder()


def _target_parts(rv: RandomVariable, data):
    """Encoder-free pieces shared by the numerical and autograd targets:
    (fam, slots, build, unpack, (dmean, dstd)). Building the pysp encoder is deferred to
    callers that need it (the autograd path scores the raw data tensor and never does)."""
    fam = _require_flat(rv)
    slots = _slots_of(rv, fam)
    arr = np.asarray(data, dtype=float)
    dmean, dstd = float(arr.mean()), float(arr.std() or 1.0)

    def unpack(u):
        vals, logj = {}, 0.0
        for k, s in enumerate(slots):
            v, lj = _to_value(s.support, u[k])
            vals[s.index] = v
            logj += lj
        return vals, logj

    def build(vals):
        args = [vals.get(i, rv._args[i]) for i in range(len(rv._args))]
        return fam.make_dist(tuple(args), rv._name)

    return fam, slots, build, unpack, (dmean, dstd)


def _build_target(rv: RandomVariable, data):
    """Return (log_target(u), slots, fam, build, unpack, (dmean,dstd)) for unconstrained u."""
    fam, slots, build, unpack, (dmean, dstd) = _target_parts(rv, data)
    enc = _encoder_for(fam).seq_encode(list(data))

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
        if s.support == "positive":
            u0.append(_to_u("positive", max(dstd, 1e-2)))
        elif s.support == "unit":
            u0.append(0.0)                       # logit(0.5)
        else:
            u0.append(dmean)
    return np.asarray(u0, dtype=float)


def _init_scale(slots, dstd, n) -> np.ndarray:
    """Per-slot proposal scale ~ posterior width: a location (real) slot ~ dstd/sqrt(n);
    a transformed (positive/unit) slot ~ 1/sqrt(n). Adaptation then tunes the magnitude."""
    root = math.sqrt(max(n, 1))
    return np.asarray(
        [max((dstd if s.support == "real" else 1.0) / root, 1e-3) for s in slots], dtype=float)


def _finalize(rv, slots, res, build) -> RandomVariable:
    """Convert unconstrained chain samples to value space, build the posterior-mean
    distribution, and attach a Posterior result. Shared by RW-MCMC and HMC."""
    u = np.asarray(res.samples, dtype=float).reshape(len(res.samples), -1)
    vals = np.empty_like(u)
    for k, s in enumerate(slots):
        if s.support == "positive":
            vals[:, k] = np.exp(u[:, k])
        elif s.support == "unit":
            vals[:, k] = 1.0 / (1.0 + np.exp(-u[:, k]))
        else:
            vals[:, k] = u[:, k]
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


def _u_to_vals(slots, u) -> np.ndarray:
    """Map an (n, d) unconstrained sample array to constrained parameter values per slot."""
    u = np.asarray(u, dtype=float).reshape(len(u), -1)
    vals = np.empty_like(u)
    for k, s in enumerate(slots):
        if s.support == "positive":
            vals[:, k] = np.exp(u[:, k])
        elif s.support == "unit":
            vals[:, k] = 1.0 / (1.0 + np.exp(-u[:, k]))
        else:
            vals[:, k] = u[:, k]
    return vals


def _gelman_rubin(chains_u: np.ndarray) -> np.ndarray:
    """Per-dimension Gelman-Rubin R-hat from (n_chains, n_draws, d) unconstrained samples."""
    m, n, _ = chains_u.shape
    if m < 2 or n < 2:
        return np.full(chains_u.shape[-1], np.nan)
    chain_means = chains_u.mean(axis=1)
    W = chains_u.var(axis=1, ddof=1).mean(axis=0)
    B = n * chain_means.var(axis=0, ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(np.maximum(var_hat / np.where(W > 0, W, 1.0), 0.0))


def _mcmc_worker(seed, rv, data, kw):
    """Module-level (picklable) single MCMC chain: rebuilds the target in-process and runs."""
    from pysp.ppl import autograd as _ag
    from pysp.utils.mcmc import AdaptiveRandomWalkProposal, metropolis_hastings

    ag = _ag.grad_target(rv, data)
    if ag is not None:
        log_target, slots, _, dmean, dstd = ag.log_target, ag.slots, ag.build, ag.dmean, ag.dstd
    else:
        log_target, slots, _fam, _build, _unpack, (dmean, dstd) = _build_target(rv, data)
    u0 = _init_u(slots, dmean, dstd)
    scale = kw.get("scale")
    init_scale = (scale * np.ones(len(u0))) if scale is not None \
        else _init_scale(slots, dstd, len(data))
    proposal = AdaptiveRandomWalkProposal(init_scale.copy())
    return metropolis_hastings(log_target, u0, proposal, num_samples=kw["draws"],
                               burn_in=kw["burn"], thin=kw["thin"],
                               rng=np.random.RandomState(seed))


def _hmc_worker(seed, rv, data, kw):
    """Module-level (picklable) single HMC chain: rebuilds the analytic-gradient target and runs."""
    from pysp.ppl import autograd as _ag
    from pysp.utils.mcmc import hamiltonian_monte_carlo

    ag = _ag.grad_target(rv, data)
    if ag is not None:
        slots, dmean, dstd = ag.slots, ag.dmean, ag.dstd
        log_target, grad = ag.log_target, ag.grad
    else:
        log_target, slots, _fam, _build, _unpack, (dmean, dstd) = _build_target(rv, data)
        grad = _finite_diff_grad(log_target, _init_u(slots, dmean, dstd))
    u0 = _init_u(slots, dmean, dstd)
    scale = _init_scale(slots, dstd, len(data))
    mass = 1.0 / (scale ** 2)
    step_size = kw["step_size"] if kw["step_size"] is not None else 2.5 / kw["num_steps"]
    return hamiltonian_monte_carlo(log_target, grad, u0, num_samples=kw["draws"],
                                   step_size=step_size, num_steps=kw["num_steps"], mass=mass,
                                   burn_in=kw["burn"], thin=kw["thin"],
                                   rng=np.random.RandomState(seed))


def _finite_diff_grad(log_target, u_ref):
    eps = 1e-5 * np.maximum(np.abs(np.asarray(u_ref, dtype=float)), 1.0)

    def grad(u):
        u = np.asarray(u, dtype=float)
        g = np.empty(len(u))
        for i in range(len(u)):
            up = u.copy()
            up[i] += eps[i]
            um = u.copy()
            um[i] -= eps[i]
            g[i] = (log_target(up) - log_target(um)) / (2.0 * eps[i])
        return g

    return grad


def _run_chains(run_one, worker, worker_args, chains: int, parallel, rng):
    """Run ``chains`` independent chains.

    ``parallel`` selects the backend: ``False``/``None`` -> sequential; ``True`` or
    ``"process"`` -> a process pool (genuine parallelism — the model pickles cleanly,
    so each worker rebuilds its own Torch target and they run on separate cores);
    ``"thread"`` -> a thread pool (rarely a win: the Torch path is GIL-bound).
    """
    seeds = [int(rng.randint(1, 2 ** 31)) for _ in range(chains)]
    mode = "process" if parallel is True else (parallel or "off")
    if chains > 1 and mode == "process":
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=chains) as ex:
            futs = [ex.submit(worker, s, *worker_args) for s in seeds]
            return [f.result() for f in futs]
    if chains > 1 and mode == "thread":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=chains) as ex:
            return list(ex.map(run_one, seeds))
    return [run_one(s) for s in seeds]


def _finalize_chains(rv, slots, results, build) -> RandomVariable:
    """Combine multiple chains: pool value-space draws, attach R-hat and combined ESS."""
    us = [np.asarray(r.samples, dtype=float).reshape(len(r.samples), -1) for r in results]
    n = min(len(u) for u in us)
    rhat = _gelman_rubin(np.stack([u[:n] for u in us], axis=0))
    vals = _u_to_vals(slots, np.concatenate(us, axis=0))
    mean_vals = {s.index: float(vals[:, k].mean()) for k, s in enumerate(slots)}
    post = Posterior(slots, vals, results[0])
    post.n_chains = len(results)
    post.rhat = {s.name: float(rhat[k]) for k, s in enumerate(slots)}
    try:
        post.ess = float(sum(np.atleast_1d(r.effective_sample_size()).min() for r in results))
    except Exception:
        post.ess = None

    def predictive(n_, rng_):
        idx = rng_.randint(len(vals), size=n_)
        out = []
        for j in idx:
            d = build({s.index: float(vals[j, k]) for k, s in enumerate(slots)})
            out.append(d.sampler(seed=int(rng_.randint(1, 2 ** 31))).sample())
        return np.asarray(out)

    post.predictive = predictive
    return RandomVariable._bound(build(mean_vals), name=rv._name, result=post)


def ensemble_fit(rv: RandomVariable, data, *, draws: int = 1500, burn: int = 500,
                 thin: int = 1, walkers: int | None = None, rng=None) -> RandomVariable:
    """Affine-invariant ensemble MCMC (Goodman & Weare stretch move).

    A population of walkers samples jointly with no per-dimension step tuning; it is invariant
    to affine rescalings, so it mixes well on correlated / poorly-scaled posteriors and gives
    very high ESS/sec on low/medium-dimensional models (no JIT-compile latency). Each ``draws``
    sweep contributes all ``walkers`` states, so the pooled posterior has ``draws*walkers``
    near-independent samples. Uses the fast NumPy scalar log-target (one eval per proposal)."""
    from pysp.utils.mcmc import affine_invariant_ensemble

    if rng is None:
        rng = np.random.RandomState()
    log_target, slots, _fam, build, _unpack, (dmean, dstd) = _build_target(rv, data)
    d = len(slots)
    if walkers is None:
        walkers = max(2 * (d + 1), 8)
    if walkers % 2:
        walkers += 1
    u0 = _init_u(slots, dmean, dstd)
    spread = _init_scale(slots, dstd, len(data)) * math.sqrt(len(data))   # ~ prior/posterior width
    p0 = u0[None, :] + 0.1 * spread[None, :] * rng.standard_normal((walkers, d))
    p0[0] = u0
    res = affine_invariant_ensemble(log_target, p0, num_samples=draws, burn_in=burn,
                                    thin=thin, rng=rng)
    return _finalize(rv, slots, res, build)


def mcmc_fit(rv: RandomVariable, data, *, draws: int = 2000, burn: int = 1000,
             thin: int = 1, scale: float | None = None, rng=None,
             chains: int = 1, parallel: bool = False) -> RandomVariable:
    from pysp.ppl import autograd as _ag
    from pysp.utils.mcmc import AdaptiveRandomWalkProposal, metropolis_hastings

    if rng is None:
        rng = np.random.RandomState()
    ag = _ag.grad_target(rv, data)
    if ag is not None:        # Torch-scored target (fast; also avoids the encoder probe)
        log_target, slots, build, dmean, dstd = ag.log_target, ag.slots, ag.build, ag.dmean, ag.dstd
    else:
        log_target, slots, fam, build, unpack, (dmean, dstd) = _build_target(rv, data)
    u0 = _init_u(slots, dmean, dstd)
    init_scale = (scale * np.ones(len(u0))) if scale is not None \
        else _init_scale(slots, dstd, len(data))

    def run_one(seed):
        proposal = AdaptiveRandomWalkProposal(init_scale.copy())   # per-chain adaptive state
        return metropolis_hastings(log_target, u0, proposal, num_samples=draws,
                                   burn_in=burn, thin=thin, rng=np.random.RandomState(seed))

    if chains == 1:
        return _finalize(rv, slots, run_one(int(rng.randint(1, 2 ** 31))), build)
    kw = {"draws": draws, "burn": burn, "thin": thin, "scale": scale}
    results = _run_chains(run_one, _mcmc_worker, (rv, data, kw), chains, parallel, rng)
    return _finalize_chains(rv, slots, results, build)


def hmc_fit(rv: RandomVariable, data, *, draws: int = 1000, burn: int = 500,
            step_size: float | None = None, num_steps: int = 15, thin: int = 1,
            rng=None, chains: int = 1, parallel: bool = False) -> RandomVariable:
    """Hamiltonian Monte Carlo over the parameter posterior.

    Uses pysp's ``hamiltonian_monte_carlo`` with a numerical gradient of the joint
    log-target and a diagonal mass matrix preconditioned to the data-informed posterior
    scale, so trajectories are well-conditioned without manual tuning.
    """
    from pysp.ppl import autograd as _ag
    from pysp.utils.mcmc import hamiltonian_monte_carlo

    if rng is None:
        rng = np.random.RandomState()

    ag = _ag.grad_target(rv, data)
    if ag is not None:
        # analytic-gradient HMC (one backprop per gradient, vs O(#params) target evals)
        slots, build, dmean, dstd = ag.slots, ag.build, ag.dmean, ag.dstd
        log_target, grad = ag.log_target, ag.grad
    else:
        log_target, slots, fam, build, unpack, (dmean, dstd) = _build_target(rv, data)
        eps = 1e-5 * np.maximum(np.abs(_init_u(slots, dmean, dstd)), 1.0)

        def grad(u):
            u = np.asarray(u, dtype=float)
            g = np.empty(len(u))
            for i in range(len(u)):
                up = u.copy()
                up[i] += eps[i]
                um = u.copy()
                um[i] -= eps[i]
                g[i] = (log_target(up) - log_target(um)) / (2.0 * eps[i])
            return g

    u0 = _init_u(slots, dmean, dstd)
    scale = _init_scale(slots, dstd, len(data))     # ~ posterior std per dim
    mass = 1.0 / (scale ** 2)                        # precondition: M ~ inverse posterior cov
    if step_size is None:
        step_size = 2.5 / num_steps                  # tuned: acc~0.98, near-max ESS (preconditioned)

    def run_one(seed):
        return hamiltonian_monte_carlo(log_target, grad, u0, num_samples=draws,
                                       step_size=step_size, num_steps=num_steps, mass=mass,
                                       burn_in=burn, thin=thin, rng=np.random.RandomState(seed))

    if chains == 1:
        return _finalize(rv, slots, run_one(int(rng.randint(1, 2 ** 31))), build)
    kw = {"draws": draws, "burn": burn, "thin": thin,
          "step_size": step_size, "num_steps": num_steps}
    results = _run_chains(run_one, _hmc_worker, (rv, data, kw), chains, parallel, rng)
    return _finalize_chains(rv, slots, results, build)


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


def _conj_binomial_beta(prior_args, fixed, stats, handle, index):
    # Binomial(n, p) with p ~ Beta(a, b); n known (fixed slot 0). successes = sum_x,
    # failures = n*N - sum_x -> posterior Beta(a + successes, b + failures).
    a, b = float(prior_args[0]), float(prior_args[1])
    n_trials = float(fixed[0])
    N, sx = stats["n"], stats["sum"]
    A, B = a + sx, b + n_trials * N - sx
    return {"index": index, "handle": handle, "name": "Beta",
            "mean": A / (A + B), "hyper": {"a": A, "b": B},
            "sample": lambda k, rng: rng.beta(A, B, k)}


def _conj_geometric_beta(prior_args, fixed, stats, handle, index):
    # Geometric(p) on k>=1 with p ~ Beta(a, b): likelihood ∝ p^N (1-p)^(sum_x - N)
    # -> posterior Beta(a + N, b + sum_x - N).
    a, b = float(prior_args[0]), float(prior_args[1])
    N, sx = stats["n"], stats["sum"]
    A, B = a + N, b + sx - N
    return {"index": index, "handle": handle, "name": "Beta",
            "mean": A / (A + B), "hyper": {"a": A, "b": B},
            "sample": lambda k, rng: rng.beta(A, B, k)}


# (likelihood family, slot index, prior family) -> closed-form posterior builder
_CONJUGATE = {
    ("Normal", 0, "Normal"): _conj_normal_mean,       # unknown mean, known variance
    ("Poisson", 0, "Gamma"): _conj_poisson_gamma,
    ("Exponential", 0, "Gamma"): _conj_exponential_gamma,
    ("Bernoulli", 0, "Beta"): _conj_bernoulli_beta,
    ("Binomial", 1, "Beta"): _conj_binomial_beta,
    ("Geometric", 0, "Beta"): _conj_geometric_beta,
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


# ------------------------------------------------ mixtures of conjugate priors (exact)
def _logbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# (likelihood, slot, prior) -> log marginal likelihood of the data under one prior
# component, up to an additive component-INDEPENDENT constant (enough to reweight a
# mixture-of-conjugate-priors exactly). Same keys as _CONJUGATE.
def _logm_normal_mean(pa, fixed, stats):
    m0, s0 = float(pa[0]), float(pa[1])
    sigma2 = float(fixed[1]) ** 2
    n, sx = stats["n"], stats["sum"]
    prec0 = 1.0 / s0 ** 2
    precP = prec0 + n / sigma2
    bb = m0 * prec0 + sx / sigma2
    return 0.5 * math.log(prec0 / precP) + 0.5 * (bb * bb / precP - m0 * m0 * prec0)


def _logm_poisson_gamma(pa, fixed, stats):
    a, b = float(pa[0]), float(pa[1])
    n, sx = stats["n"], stats["sum"]
    return math.lgamma(a + sx) - math.lgamma(a) + a * math.log(b) - (a + sx) * math.log(b + n)


def _logm_exponential_gamma(pa, fixed, stats):
    a, b = float(pa[0]), float(pa[1])
    n, sx = stats["n"], stats["sum"]
    return math.lgamma(a + n) - math.lgamma(a) + a * math.log(b) - (a + n) * math.log(b + sx)


def _logm_bernoulli_beta(pa, fixed, stats):
    a, b = float(pa[0]), float(pa[1])
    n, sx = stats["n"], stats["sum"]
    return _logbeta(a + sx, b + n - sx) - _logbeta(a, b)


def _logm_binomial_beta(pa, fixed, stats):
    a, b = float(pa[0]), float(pa[1])
    n_tr, N, sx = float(fixed[0]), stats["n"], stats["sum"]
    return _logbeta(a + sx, b + n_tr * N - sx) - _logbeta(a, b)


def _logm_geometric_beta(pa, fixed, stats):
    a, b = float(pa[0]), float(pa[1])
    N, sx = stats["n"], stats["sum"]
    return _logbeta(a + N, b + sx - N) - _logbeta(a, b)


_CONJ_LOGM = {
    ("Normal", 0, "Normal"): _logm_normal_mean,
    ("Poisson", 0, "Gamma"): _logm_poisson_gamma,
    ("Exponential", 0, "Gamma"): _logm_exponential_gamma,
    ("Bernoulli", 0, "Beta"): _logm_bernoulli_beta,
    ("Binomial", 1, "Beta"): _logm_binomial_beta,
    ("Geometric", 0, "Beta"): _logm_geometric_beta,
}


class ConjugateMixturePosterior:
    """Exact posterior for a mixture-of-conjugate-priors model.

    The posterior is again a mixture: the per-component conjugate posteriors with weights
    reweighted by each component's marginal likelihood, ``w'_k ∝ w_k · m_k``. Sampling draws
    a component by ``w'`` then samples that component's conjugate posterior.
    """

    def __init__(self, entries, weights, param_name):
        self.entries = entries            # list of per-component conjugate posterior dicts
        self.weights = np.asarray(weights, dtype=float)   # posterior mixing weights w'
        self.param_name = param_name
        self.acceptance_rate = None
        self.predictive = None

    def mean(self, param=None):
        return float(np.sum(self.weights * np.array([e["mean"] for e in self.entries])))

    def samples(self, param=None, n: int = 4000, rng=None):
        rng = rng or np.random.RandomState()
        comp = rng.choice(len(self.entries), size=n, p=self.weights)
        out = np.empty(n)
        for k, e in enumerate(self.entries):
            m = comp == k
            cnt = int(m.sum())
            if cnt:
                out[m] = np.atleast_1d(e["sample"](cnt, rng))
        return out

    def summary(self) -> dict:
        return {"posterior": "mixture", "weights": self.weights.tolist(),
                "components": [{"mean": e["mean"], "hyper": e["hyper"]} for e in self.entries],
                "mean": self.mean()}


def conjugate_mixture_spec(rv: RandomVariable):
    """Return (builder, logm, slot_index, component_rvs, prior_weights) when exactly one slot
    is a ``Mix`` of conjugate priors (all forming the same registered conjugate pair) and every
    other slot is a fixed constant; else None."""
    if rv._kind != "sample" or isinstance(rv._family, CompositeFamily):
        return None
    if any(a is free for a in rv._args):
        return None
    mix_slots = [(i, a) for i, a in enumerate(rv._args)
                 if isinstance(a, RandomVariable) and isinstance(a._family, CompositeFamily)
                 and a._family.name == "Mixture"]
    other_rv = [a for a in rv._args if isinstance(a, RandomVariable)
                and not (isinstance(a._family, CompositeFamily) and a._family.name == "Mixture")]
    if len(mix_slots) != 1 or other_rv:
        return None
    i, mix = mix_slots[0]
    comps, weights = mix._args
    comps = list(comps)
    if not comps:
        return None
    fam_names = {c._family.name for c in comps
                if c._kind == "sample" and not isinstance(c._family, CompositeFamily)}
    if len(fam_names) != 1:
        return None  # all components must be the same flat conjugate prior family
    key = (rv._family.name, i, next(iter(fam_names)))
    if key not in _CONJUGATE or key not in _CONJ_LOGM:
        return None
    w = np.ones(len(comps)) / len(comps) if weights is None else np.asarray(weights, dtype=float)
    return _CONJUGATE[key], _CONJ_LOGM[key], i, comps, w / w.sum()


def conjugate_mixture_fit(rv: RandomVariable, data) -> RandomVariable:
    spec = conjugate_mixture_spec(rv)
    if spec is None:
        raise NotImplementedError("model is not a mixture of registered conjugate priors.")
    builder, logm, idx, comps, w = spec
    fam = rv._family
    arr = np.asarray(data, dtype=float)
    stats = {"n": float(arr.size), "sum": float(arr.sum()), "sum2": float((arr * arr).sum())}
    fixed = {j: rv._args[j] for j in range(len(rv._args)) if j != idx}

    entries = [builder(c._args, fixed, stats, c, idx) for c in comps]
    logw = np.log(w) + np.array([logm(c._args, fixed, stats) for c in comps])
    logw -= logw.max()
    post_w = np.exp(logw)
    post_w /= post_w.sum()

    post = ConjugateMixturePosterior(entries, post_w, comps[0].name or f"arg{idx}")
    pmean = post.mean()
    full = [pmean if j == idx else rv._args[j] for j in range(len(rv._args))]
    fitted = fam.make_dist(tuple(full), rv._name)

    def predictive(n, rng):
        pvals = np.atleast_1d(post.samples(n=n, rng=rng))
        out = []
        for v in pvals:
            args = [float(v) if j == idx else rv._args[j] for j in range(len(rv._args))]
            out.append(fam.make_dist(tuple(args), rv._name).sampler(seed=int(rng.randint(1, 2 ** 31))).sample())
        return np.asarray(out)

    post.predictive = predictive
    return RandomVariable._bound(fitted, name=rv._name, result=post)


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
    m = float(gm.mean())
    v = float(gm.var()) or m
    b = m / max(v, 1e-6)
    a = m * b
    prev = None
    for _ in range(max_its):
        A, B = a + sum_i, b + n_i                       # posterior Gamma(A_i, B_i) per group
        Elam, Vlam = A / B, A / (B * B)
        m = float(Elam.mean())
        v = float(np.var(Elam) + Vlam.mean())           # total variance
        b = m / max(v, 1e-8)
        a = m * b
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
    m = float(gp.mean())
    v = float(gp.var()) or (m * (1 - m))
    s = max(m * (1 - m) / max(v, 1e-6) - 1, 1e-3)
    a = m * s
    b = (1 - m) * s
    prev = None
    for _ in range(max_its):
        A, B = a + sum_i, b + (n_i - sum_i)             # posterior Beta(A_i, B_i)
        Ep = A / (A + B)
        Vp = A * B / ((A + B) ** 2 * (A + B + 1))
        m = float(Ep.mean())
        v = float(np.var(Ep) + Vp.mean())
        s = max(m * (1 - m) / max(v, 1e-8) - 1, 1e-3)
        a = m * s
        b = (1 - m) * s
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

    from pysp.ppl import autograd as _ag

    g = _ag.grad_target(rv, data)
    if g is not None:
        # analytic-gradient MAP: L-BFGS on the joint posterior (fast, scales with #params)
        u0 = _init_u(g.slots, g.dmean, g.dstd)

        def neg(u):
            v, gr = g.value_and_grad(u)
            return -v, -gr

        res = minimize(neg, u0, jac=True, method="L-BFGS-B", options={"maxiter": 1000})
        vals, _ = g.unpack(res.x)
        return RandomVariable._bound(g.build(vals), name=rv._name)

    # derivative-free fallback (no Torch, or an unsupported family)
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
           max_iter: int = 4000, steps: int = 600, lr: float = 0.05, rng=None) -> RandomVariable:
    """Mean-field variational Bayes (ADVI).

    Fits a diagonal-Gaussian variational posterior q(u) = N(mean, diag(std^2)) in the
    unconstrained space by maximizing a reparameterized Monte-Carlo ELBO. With Torch it
    uses analytic-gradient Adam (fast, scales); otherwise it falls back to a derivative-free
    ELBO optimization. Works for *non-conjugate* priors the closed-form registry can't
    handle; returns a variational Posterior with draws and posterior-predictive.
    """
    from pysp.ppl import autograd as _ag

    if rng is None:
        rng = np.random.RandomState()

    ag = _ag.grad_target(rv, data)
    if ag is not None:
        slots, build = ag.slots, ag.build
        u0 = _init_u(slots, ag.dmean, ag.dstd)
        s0 = _init_scale(slots, ag.dstd, len(data))
        vals, mean, std = ag.advi(u0, s0, samples=samples, mc=mc, steps=steps, lr=lr, rng=rng)
    else:
        from scipy.optimize import minimize

        log_target, slots, fam, build, unpack, (dmean, dstd) = _build_target(rv, data)
        d = len(slots)
        u0 = _init_u(slots, dmean, dstd)
        s0 = _init_scale(slots, dstd, len(data))
        eps = rng.standard_normal((mc, d))                  # common random numbers
        half_entropy_const = 0.5 * d * (1.0 + math.log(2.0 * math.pi))

        def neg_elbo(phi):
            mean, log_std = phi[:d], phi[d:]
            std = np.exp(log_std)
            U = mean + std * eps
            ll = float(np.mean([log_target(U[i]) for i in range(mc)]))
            return -(ll + float(np.sum(log_std)) + half_entropy_const)

        res = minimize(neg_elbo, np.concatenate([u0, np.log(s0)]), method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-5, "fatol": 1e-5})
        mean, std = res.x[:d], np.exp(res.x[d:])
        Z = rng.standard_normal((samples, d))
        U = mean + std * Z
        vals = np.empty_like(U)
        for k, s in enumerate(slots):
            vals[:, k] = np.exp(U[:, k]) if s.positive else U[:, k]

    mean_vals = {s.index: float(vals[:, k].mean()) for k, s in enumerate(slots)}
    post = Posterior(slots, vals, _VIResult(0.0, mean, std))

    def predictive(n, r):
        idx = r.randint(len(vals), size=n)
        out = []
        for j in idx:
            dd = build({s.index: float(vals[j, k]) for k, s in enumerate(slots)})
            out.append(dd.sampler(seed=int(r.randint(1, 2 ** 31))).sample())
        return np.asarray(out)

    post.predictive = predictive
    return RandomVariable._bound(build(mean_vals), name=rv._name, result=post)
