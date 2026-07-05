"""mixle on the GPU: absolute throughput and memory for a full-covariance GMM.

A capability / throughput study (NOT a cross-machine speedup claim -- the CPU reference
lives in the main M4 benchmark). The same mixle model, fed identical data and init, fit
with the torch CUDA engine as N and dimension grow. Reports fit time, throughput, and
peak GPU memory -- the last showing that dim=128/256 full-covariance mixtures (which OOM'd
before the batched-covariance fix) now fit in a few GB. Emits gpu_results.json.

    python gpu_scaling.py
"""

import json
import time

import numpy as np
import torch

import mixle.stats as st
from mixle.engines import TorchEngine
from mixle.inference import optimize

torch.set_default_dtype(torch.float64)


def make_gmm(n, dim, k, seed=42):
    rng = np.random.RandomState(seed)
    true_means = rng.randn(k, dim) * 3.0
    X = np.empty((n, dim))
    z = rng.randint(0, k, size=n)
    for j in range(k):
        a = rng.randn(dim, dim) * 0.4
        X[z == j] = rng.multivariate_normal(true_means[j], a @ a.T + np.eye(dim), size=int((z == j).sum()))
    means0 = true_means + rng.randn(k, dim) * 0.5  # spread, non-degenerate shared init
    cov0 = np.cov(X.T) + 1e-6 * np.eye(dim)
    return X, means0, cov0, np.full(k, 1.0 / k)


def fit_gpu(X, means0, cov0, w0, its):
    k, dim = means0.shape
    data = list(X)

    def build():
        comps = [st.MultivariateGaussianDistribution(means0[j].copy(), cov0.copy()) for j in range(k)]
        est = st.MixtureEstimator([st.MultivariateGaussianEstimator(dim=dim) for _ in range(k)])
        return st.MixtureDistribution(comps, list(w0)), est

    eng = TorchEngine(device="cuda:0")
    m0, est = build()
    m = optimize(data, est, prev_estimate=m0, max_its=2, delta=None, out=None, engine=eng)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(3):
        m0, est = build()
        t0 = time.perf_counter()
        m = optimize(data, est, prev_estimate=m0, max_its=its, delta=None, out=None, engine=eng)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    enc = m.dist_to_encoder().seq_encode(data)
    ll = float(np.sum(np.asarray(m.seq_log_density(enc)))) / len(data)
    return float(np.median(ts)), ll, peak


def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"=== mixle on GPU ({gpu_name}) -- full-covariance GMM throughput ===")
    out = {"gpu": gpu_name, "panels": {}}

    dim, k, its = 64, 16, 12
    print(f"\n[data scaling] dim={dim} K={k} full-cov, {its} EM iters")
    pts = []
    for n in [50000, 200000, 500000]:
        try:
            t, ll, peak = fit_gpu(*make_gmm(n, dim, k), its)
            thru = n * its / t / 1e6
            print(f"  N={n:>8}: GPU {t * 1e3:8.0f}ms  {thru:6.2f}M row-iters/s  peak {peak:.2f}GB")
            pts.append({"n": n, "dim": dim, "k": k, "gpu": t, "throughput_Mri_s": thru, "gpu_peak_gb": peak, "ll": ll})
        except torch.OutOfMemoryError:
            print(f"  N={n:>8}: OOM")
            torch.cuda.empty_cache()
            pts.append({"n": n, "dim": dim, "k": k, "gpu": None, "oom": True})
    out["panels"]["scale_n"] = {"axis": "n", "fixed": {"dim": dim, "k": k, "its": its}, "points": pts}

    # dim scaling at N=20000 K=8 -- dim=128 full-cov OOM'd (a 21 GB N*K*dim^2 intermediate)
    # before the batched-covariance fix; it now fits in a couple GB.
    n, k, its = 20000, 8, 10
    print(f"\n[dimension scaling] N={n} K={k} full-cov, {its} EM iters")
    pts = []
    for dim in [32, 64, 128]:
        try:
            t, ll, peak = fit_gpu(*make_gmm(n, dim, k), its)
            naive = n * k * dim * dim * 8 / 1e9
            print(
                f"  dim={dim:>4}: GPU {t * 1e3:8.0f}ms  peak {peak:.2f}GB  (pre-fix N*K*dim^2 intermediate = {naive:.0f}GB)"
            )
            pts.append(
                {"dim": dim, "n": n, "k": k, "gpu": t, "gpu_peak_gb": peak, "naive_intermediate_gb": naive, "ll": ll}
            )
        except torch.OutOfMemoryError:
            print(f"  dim={dim:>4}: OOM")
            torch.cuda.empty_cache()
            pts.append({"dim": dim, "n": n, "k": k, "gpu": None, "oom": True})
    out["panels"]["scale_dim"] = {"axis": "dim", "fixed": {"n": n, "k": k, "its": its}, "points": pts}

    with open("gpu_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote gpu_results.json")


if __name__ == "__main__":
    main()
