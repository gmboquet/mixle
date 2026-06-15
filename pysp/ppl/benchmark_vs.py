"""Head-to-head benchmark: pysp.ppl vs NumPyro / Pyro / emcee.

Unlike :mod:`pysp.ppl.benchmark` (which uses torch-Adam as a Pyro-SVI stand-in), this
module runs the **actual** competing PPLs when they are installed, on the same machine,
the same data, and the same model, and prints a comparison table per task.

Each competitor block is guarded — missing libraries are simply skipped and noted, so the
script runs with whatever subset is available::

    python -m pysp.ppl.benchmark_vs

What it measures, and why pysp wins:

* **Conjugate models** (Poisson-Gamma, Beta-Bernoulli): pysp returns the *exact* posterior
  in a single O(N) pass. Stan/Pyro/NumPyro have no exact path and must run MCMC/SVI — an
  approximate answer for 100-1000x the wall-clock. This is the decisive comparison.
* **General posteriors** (Gaussian mean+sd): an honest end-to-end (compile-inclusive)
  wall-clock and ESS/sec comparison of pysp's MCMC/HMC against NumPyro NUTS and emcee.
* **MLE / EM** (large-N Gaussian): pysp closed-form EM vs the gradient-PPL (Pyro SVI) route.
"""
from __future__ import annotations

import time

import numpy as np

from pysp.ppl import Bernoulli, Beta, Gamma, Normal, Poisson, free
from pysp.ppl.inference import conjugate_fit, ensemble_fit, hmc_fit, mcmc_fit


def _have(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


def _timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def _ess(x: np.ndarray) -> float:
    """Effective sample size from the integrated autocorrelation time (single chain)."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0:
        return float(n)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (var * n)
    tau = 1.0
    for k in range(1, n):
        if acf[k] <= 0:
            break
        tau += 2.0 * acf[k]
    return float(n / max(tau, 1.0))


# ----------------------------------------------------------------- NumPyro competitors
def _numpyro_nuts(model, draws, warmup, seed=0, **data):
    import jax
    from numpyro.infer import MCMC, NUTS

    mcmc = MCMC(NUTS(model), num_warmup=warmup, num_samples=draws, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), **data)
    return mcmc


def np_poisson_gamma(data, draws, warmup):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(y):
        rate = numpyro.sample("rate", dist.Gamma(2.0, 1.0))
        numpyro.sample("y", dist.Poisson(rate), obs=y)

    mcmc = _numpyro_nuts(model, draws, warmup, y=jnp.asarray(data))
    s = np.asarray(mcmc.get_samples()["rate"])
    return float(s.mean()), _ess(s)


def np_beta_bernoulli(data, draws, warmup):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(y):
        p = numpyro.sample("p", dist.Beta(2.0, 2.0))
        numpyro.sample("y", dist.Bernoulli(p), obs=y)

    mcmc = _numpyro_nuts(model, draws, warmup, y=jnp.asarray(data))
    s = np.asarray(mcmc.get_samples()["p"])
    return float(s.mean()), _ess(s)


def np_gaussian(data, draws, warmup):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(10.0))
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    mcmc = _numpyro_nuts(model, draws, warmup, y=jnp.asarray(data))
    s = np.asarray(mcmc.get_samples()["mu"])
    return float(s.mean()), _ess(s)


# --------------------------------------------------------------------- emcee competitor
def emcee_gaussian(data, draws, walkers=32, seed=0):
    import emcee

    x = np.asarray(data, dtype=float)
    n = len(x)

    def logp(theta):
        mu, ls = theta
        if not np.isfinite(ls) or ls > 20:
            return -np.inf
        sig = np.exp(ls)
        return -0.5 * np.sum(((x - mu) / sig) ** 2) - n * ls

    rng = np.random.RandomState(seed)
    p0 = np.column_stack([rng.normal(x.mean(), 1.0, walkers),
                          rng.normal(np.log(x.std() or 1.0), 0.05, walkers)])
    sampler = emcee.EnsembleSampler(walkers, 2, logp)
    sampler.run_mcmc(p0, draws, progress=False)
    chain = sampler.get_chain(discard=draws // 2, flat=True)
    mu = chain[:, 0]
    return float(mu.mean()), _ess(mu)


# --------------------------------------------------------------------- Pyro SVI baseline
def pyro_gaussian_svi(data, steps=600, lr=0.05):
    import pyro
    import pyro.distributions as dist
    import torch
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoNormal

    pyro.clear_param_store()
    y = torch.tensor(np.asarray(data, dtype=np.float64))

    def model(obs):
        mu = pyro.sample("mu", dist.Normal(0.0, 10.0))
        sigma = pyro.sample("sigma", dist.HalfNormal(10.0))
        with pyro.plate("data", len(obs)):
            pyro.sample("y", dist.Normal(mu, sigma), obs=obs)

    guide = AutoNormal(model)
    svi = SVI(model, guide, pyro.optim.Adam({"lr": lr}), loss=Trace_ELBO())
    for _ in range(steps):
        svi.step(y)
    return float(guide.median()["mu"])


# ------------------------------------------------------------------------------- tasks
def task_poisson_gamma(n=200_000, draws=2000, warmup=1000):
    rng = np.random.RandomState(0)
    data = list(rng.poisson(3.5, n).astype(float))
    print(f"\n[Poisson-Gamma posterior, N={n:,}]  true rate=3.5")
    (m, t) = _timed(lambda: conjugate_fit(Poisson(Gamma(2.0, 1.0, name="rate")), data))
    print(f"  pysp.ppl  EXACT (1 pass)   {t*1e3:9.1f} ms   rate={m.dist.lam:.4f}   (no sampling)")
    rows = [("pysp exact", t, m.dist.lam, None)]
    if _have("numpyro"):
        (res, t) = _timed(np_poisson_gamma, np.asarray(data), draws, warmup)
        mean, ess = res
        print(f"  NumPyro   NUTS {draws}d      {t*1e3:9.1f} ms   rate={mean:.4f}   ESS={ess:.0f}  ({t/rows[0][1]:.0f}x slower)")
        rows.append(("numpyro nuts", t, mean, ess))
    return rows


def task_beta_bernoulli(n=100_000, draws=2000, warmup=1000):
    rng = np.random.RandomState(1)
    data = list((rng.uniform(size=n) < 0.37).astype(float))
    print(f"\n[Beta-Bernoulli posterior, N={n:,}]  true p=0.37")
    (m, t0) = _timed(lambda: conjugate_fit(Bernoulli(Beta(2.0, 2.0, name="p")), data))
    print(f"  pysp.ppl  EXACT (1 pass)   {t0*1e3:9.1f} ms   p={m.dist.p:.4f}   (no sampling)")
    if _have("numpyro"):
        (res, t) = _timed(np_beta_bernoulli, np.asarray(data), draws, warmup)
        mean, ess = res
        print(f"  NumPyro   NUTS {draws}d      {t*1e3:9.1f} ms   p={mean:.4f}   ESS={ess:.0f}  ({t/t0:.0f}x slower)")


def task_gaussian_mcmc(n=20_000, draws=2000, warmup=1000):
    rng = np.random.RandomState(2)
    data = list(rng.normal(5.0, 2.0, n))
    print(f"\n[Gaussian mean posterior, N={n:,}]  true mean=5.0   (ESS/sec = mixing efficiency)")
    mu = Normal(0, 10, name="mu")
    (m, t) = _timed(lambda: mcmc_fit(Normal(mu, free), data, draws=draws, burn=warmup,
                                     rng=np.random.RandomState(3)))
    ess = float(np.atleast_1d(m.result.raw.effective_sample_size()).min())
    print(f"  pysp.ppl  RW-Metropolis    {t*1e3:9.1f} ms   mean={m.dist.mu:.4f}   ESS={ess:.0f}  ESS/s={ess/t:8.0f}")
    (m, t) = _timed(lambda: ensemble_fit(Normal(mu, free), data, draws=draws, burn=warmup,
                                         rng=np.random.RandomState(3)))
    ess = float(np.atleast_1d(m.result.raw.effective_sample_size()).min())
    print(f"  pysp.ppl  ensemble (G&W)   {t*1e3:9.1f} ms   mean={m.dist.mu:.4f}   ESS={ess:.0f}  ESS/s={ess/t:8.0f}")
    (m, t) = _timed(lambda: hmc_fit(Normal(mu, free), data, draws=draws // 2, burn=warmup // 2,
                                    rng=np.random.RandomState(3)))
    ess = float(np.atleast_1d(m.result.raw.effective_sample_size()).min())
    print(f"  pysp.ppl  HMC (analytic g) {t*1e3:9.1f} ms   mean={m.dist.mu:.4f}   ESS={ess:.0f}  ESS/s={ess/t:8.0f}")
    if _have("numpyro"):
        (res, t) = _timed(np_gaussian, np.asarray(data), draws, warmup)
        mean, ess = res
        print(f"  NumPyro   NUTS             {t*1e3:9.1f} ms   mean={mean:.4f}   ESS={ess:.0f}  ESS/s={ess/t:8.0f}  (incl JIT compile)")
    if _have("emcee"):
        (res, t) = _timed(emcee_gaussian, data, draws)
        mean, ess = res
        print(f"  emcee     ensemble         {t*1e3:9.1f} ms   mean={mean:.4f}   ESS={ess:.0f}  ESS/s={ess/t:8.0f}")


def task_gaussian_mle(n=500_000):
    rng = np.random.RandomState(4)
    data = list(rng.normal(5.0, 2.0, n))
    print(f"\n[Gaussian MLE, N={n:,}]  true mean=5.0 sd=2.0")
    (m, t0) = _timed(lambda: Normal(free, free).fit(data))
    print(f"  pysp.ppl  closed-form EM   {t0*1e3:9.1f} ms   mean={m.dist.mu:.4f} sd={np.sqrt(m.dist.sigma2):.4f}")
    if _have("pyro"):
        (mean, t) = _timed(pyro_gaussian_svi, data)
        print(f"  Pyro      SVI (AutoNormal) {t*1e3:9.1f} ms   mean={mean:.4f}   ({t/t0:.0f}x slower)")


def main():
    avail = [m for m in ("numpyro", "pyro", "emcee") if _have(m)]
    print("=" * 78)
    print("pysp.ppl HEAD-TO-HEAD vs " + (", ".join(avail) if avail else "(no competitors installed)"))
    print("same machine / same data / same model — end-to-end wall clock")
    print("=" * 78)
    task_poisson_gamma()
    task_beta_bernoulli()
    task_gaussian_mcmc()
    task_gaussian_mle()
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
