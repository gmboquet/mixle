"""Speed benchmark for pysp.ppl.

Stan / Pyro / NumPyro are not installed here, but **torch is** — and Pyro's SVI is, under
the hood, exactly batched-gradient optimization of a log-likelihood with torch autograd.
So we compare, on the same machine and same data:

  1. Gaussian / GMM MLE: pysp closed-form EM vs a torch-Adam fit (the Pyro-SVI approach).
  2. Conjugate exact posterior vs MCMC for the *same* model — pysp has an exact O(N) path;
     Stan/Pyro have none and must sample. This is the core "prefer over Stan/Pyro" point
     for conjugate/EM models: the answer is exact and ~orders of magnitude faster.
  3. pysp MCMC and HMC throughput (draws/sec, ESS/sec).

Run:  python -m pysp.ppl.benchmark
"""
from __future__ import annotations

import time
import numpy as np

from pysp.ppl import Normal, Poisson, Gamma, Mix, free
from pysp.ppl.inference import conjugate_fit, mcmc_fit, hmc_fit


def _timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


# --------------------------------------------------------------------- torch baselines
def torch_gaussian_mle(data, iters=300, lr=0.1):
    import torch
    x = torch.tensor(np.asarray(data, dtype=np.float64))
    mu = torch.zeros((), requires_grad=True, dtype=torch.float64)
    ls = torch.zeros((), requires_grad=True, dtype=torch.float64)
    opt = torch.optim.Adam([mu, ls], lr=lr)
    c = 0.5 * np.log(2 * np.pi)
    for _ in range(iters):
        opt.zero_grad()
        sig = ls.exp()
        nll = (0.5 * ((x - mu) / sig) ** 2 + ls + c).mean()
        nll.backward()
        opt.step()
    return float(mu.detach()), float(ls.exp().detach())


def torch_gmm_mle(data, init_means, iters=300, lr=0.1):
    import torch
    x = torch.tensor(np.asarray(data, dtype=np.float64))[:, None]            # (N,1)
    means = torch.tensor(np.asarray(init_means, dtype=np.float64), requires_grad=True)
    ls = torch.zeros(len(init_means), dtype=torch.float64, requires_grad=True)
    logits = torch.zeros(len(init_means), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([means, ls, logits], lr=lr)
    c = 0.5 * np.log(2 * np.pi)
    for _ in range(iters):
        opt.zero_grad()
        sig = ls.exp()
        log_comp = -0.5 * ((x - means) / sig) ** 2 - ls - c                  # (N,K)
        logw = torch.log_softmax(logits, dim=0)
        ll = torch.logsumexp(logw + log_comp, dim=1).mean()
        (-ll).backward()
        opt.step()
    return sorted(means.detach().numpy().tolist())


# --------------------------------------------------------------------------- benchmarks
def bench_gaussian(n=500_000):
    rng = np.random.RandomState(0)
    data = list(rng.normal(5.0, 2.0, size=n))
    (m, t_pysp) = _timed(lambda: Normal(free, free).fit(data))
    pysp_mu, pysp_sd = m.dist.mu, np.sqrt(m.dist.sigma2)
    try:
        (res, t_torch) = _timed(torch_gaussian_mle, data)
        torch_mu, torch_sd = res
        torch_line = f"torch-Adam (Pyro-SVI style): {t_torch*1e3:8.1f} ms   mu={torch_mu:.3f} sd={torch_sd:.3f}"
        speedup = f"{t_torch / t_pysp:6.1f}x"
    except Exception as e:
        torch_line = f"torch baseline unavailable ({e})"
        speedup = "n/a"
    print(f"\n[Gaussian MLE, N={n:,}]")
    print(f"  pysp.ppl EM (closed form):   {t_pysp*1e3:8.1f} ms   mu={pysp_mu:.3f} sd={pysp_sd:.3f}")
    print(f"  {torch_line}")
    print(f"  -> pysp.ppl speedup: {speedup}")


def bench_gmm(n=200_000):
    rng = np.random.RandomState(1)
    data = list(np.concatenate([rng.normal(-4, 1, n // 2), rng.normal(4, 1, n // 2)]))
    arr = np.asarray(data)
    # shared k-means++-ish init for a fair convergence comparison
    init = [float(arr.min()), float(arr.max())]
    (m, t_pysp) = _timed(lambda: Mix([Normal(free, free), Normal(free, free)]).fit(
        data, rng=np.random.RandomState(2)))
    pysp_means = sorted(c.mu for c in m.dist.components)
    try:
        (torch_means, t_torch) = _timed(torch_gmm_mle, data, init)
        torch_line = f"torch-Adam GMM (Pyro-SVI style): {t_torch*1e3:8.1f} ms   means={[round(x,2) for x in torch_means]}"
        speedup = f"{t_torch / t_pysp:6.1f}x"
    except Exception as e:
        torch_line = f"torch baseline unavailable ({e})"
        speedup = "n/a"
    print(f"\n[2-component Gaussian mixture, N={n:,}]")
    print(f"  pysp.ppl EM:                  {t_pysp*1e3:8.1f} ms   means={[round(x,2) for x in pysp_means]}")
    print(f"  {torch_line}")
    print(f"  -> pysp.ppl speedup: {speedup}")


def bench_exact_vs_mcmc(n=200_000):
    rng = np.random.RandomState(3)
    data = list(rng.poisson(3.5, size=n).astype(float))
    model = lambda: Poisson(Gamma(2.0, 1.0, name="rate"))
    (m_exact, t_exact) = _timed(lambda: conjugate_fit(model(), data))
    (m_mcmc, t_mcmc) = _timed(lambda: mcmc_fit(model(), data, draws=2000, burn=1000,
                                               rng=np.random.RandomState(4)))
    print(f"\n[Poisson-Gamma posterior, N={n:,}]  (Stan/Pyro have NO exact path -> must MCMC)")
    print(f"  pysp.ppl exact (1 pass):     {t_exact*1e3:8.1f} ms   rate={m_exact.dist.lam:.4f}")
    print(f"  pysp.ppl MCMC (2000 draws):  {t_mcmc*1e3:8.1f} ms   rate={m_mcmc.dist.lam:.4f}")
    print(f"  -> exact is {t_mcmc / t_exact:6.0f}x faster for the same posterior")


def bench_mcmc_throughput(n=20_000):
    rng = np.random.RandomState(5)
    data = list(rng.normal(5.0, 2.0, size=n))
    mu = Normal(0, 10, name="mu")
    for how, kw in (("mcmc", dict(draws=2000, burn=1000)),
                    ("hmc", dict(draws=1000, burn=500))):
        m, t = _timed(lambda: Normal(mu, free).fit(data, how=how, rng=np.random.RandomState(6), **kw))
        ess = float(np.atleast_1d(m.result.raw.effective_sample_size()).min())
        draws = kw["draws"]
        print(f"\n[{how.upper()} throughput, N={n:,}, {draws} draws]")
        print(f"  time={t*1e3:8.1f} ms   acc={m.result.acceptance_rate:.2f}   "
              f"min-ESS={ess:.0f}   ESS/sec={ess / t:8.0f}")


def main():
    print("=" * 72)
    print("pysp.ppl speed benchmark (same machine; torch = the Pyro-SVI substrate)")
    print("=" * 72)
    bench_gaussian()
    bench_gmm()
    bench_exact_vs_mcmc()
    bench_mcmc_throughput()
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
