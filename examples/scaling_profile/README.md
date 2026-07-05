# Scaling studies + profiling — where mixle spends its time, and what to speed up

`scaling_profile.py` measures fit-time scaling (data / model / workers / engines) and cProfiles the hot
paths to find bottlenecks. Run it: `python scaling_profile.py [--gpu] [--heavy] [--only D,K,W,E,P]`.

Findings below are from a full `--gpu --heavy` run on a rented RTX A4000 + 16-core box
(`scale_A4000_16core.log`), cost **$0.014**.

## Scaling behavior (all linear in N — no superlinear surprises)

| family | rows/s | data-scaling slope | notes |
|---|---|---|---|
| GMM | ~0.9 M/s | 0.95 (linear) | fast |
| MVN mixture | ~0.14 M/s | 1.03 (linear) | 6× slower/row than GMM; covariance-bound |
| **HMM** | **~0.01 M/s** | 1.07 (linear) | **~90× slower/row than GMM** — the slowest family |

- **MVN vs dim**: slope 1.31 overall but the tail steepens (dim 64→256 is 14× for 4× dim) — covariance
  ops (O(N·dim²) accumulation + O(dim³) Cholesky) dominate at high dim. dim=256 fit ≈ 2.9 s.
- **GMM vs K**: linear (slope 0.96).

## The hotspots (cProfile self-time)

| family | top hotspot | share |
|---|---|---|
| MVN mixture | `np.einsum` for the covariance second-moment | **~76%** of fit |
| GMM | `mixture.seq_update` (responsibility accumulation) | ~49% |
| HMM | `hidden_markov.seq_update` (forward-backward) + numpy `reduce` | ~40% + ~40% |

## What got fixed (this study)

- **MVN covariance `einsum` → BLAS matmul** (`814f4b7`): the #1 hotspot was `np.einsum("ji,ik->jk", ...)`
  running numpy's naive C loop instead of BLAS. It is exactly `(x.T*w) @ x` — a gemm. Swapped: **20–36×
  faster** on the contraction, **2.7× faster** end-to-end MVN fits. Byte-exact, benefits every MVN user.
- **Self-healing MVN Cholesky** (`845cc1d`): float32 engines (MPS / CUDA) crashed at `cho_factor` on a
  non-PD covariance (precision loss) at higher dim; now symmetrize + minimal jitter only on failure, the
  float64 path byte-unchanged.
- **`torch-cuda` MVN mixture OOM at dim=128** (`825165c`): the fused GPU E-step materialized a per-sample,
  per-component `(N, K, dim, dim)` outer-product tensor (~21 GB) in three places — each really a gemm.
  Fixed all three (per-component `(x*w[:,k]).T@x` second moment; flatten-feature matmul for the forward
  log-density's `<T_n, eta_k>`; `w.T @ arr_flat` in the generic matrix-moment reducer). **Verified on a
  rented RTX 3060 (12 GB):** the exact failing config now fits at **2.674 GB peak** (was a 19.53 GiB
  single-alloc attempt), CUDA-vs-numpy parity Δll=5.8e-11.
- **HMM numba silently off** (`f36a096`): the profiled HMM fit ran numpy Baum-Welch because the
  distribution defaulted `use_numba=False` while the estimator defaulted numba-on, and
  `optimize(prev_estimate=)` encodes via the distribution. Aligned the default → **5.7× faster**
  (670→118 ms), bit-identical. HMM is still the slowest family; deeper wins need custom emission kernels.

## Open bottlenecks (flagged as tasks)

- **`torch-cpu` engine is 11–31× *slower* than numpy** (MVN dim 16→128) — huge per-op dispatch overhead.
  The torch engine only pays off on GPU; on CPU, numpy wins decisively.
- **Single-node `model_parallel` is 0.89× (slower)** at N=200k — parallelism overhead exceeds the compute
  saved until problems are much larger; worth a crossover-size heuristic before auto-parallelizing.
