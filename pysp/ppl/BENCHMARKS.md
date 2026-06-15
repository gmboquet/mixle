# pysp.ppl benchmarks

Two reproducible benchmark scripts:

* `python -m pysp.ppl.benchmark_vs` — **head-to-head against the actual competing PPLs**
  (NumPyro NUTS, Pyro SVI, emcee) when they are installed. Same machine, same data, same model.
* `python -m pysp.ppl.benchmark` — pysp vs a torch-Adam baseline (a faithful stand-in for the
  Pyro-SVI substrate) for environments where the other PPLs aren't installed.

The numbers below are from `benchmark_vs` with `numpyro 0.19`, `pyro 1.9`, `emcee 3.1`,
`jax 0.4.35`, CPU, single machine, `OMP_NUM_THREADS=1`.

## 1. Conjugate / exponential-family models — the decisive win

For any conjugate model, pysp.ppl returns the **exact** posterior in a single O(N) pass.
Stan / Pyro / NumPyro have **no exact path** — they must run MCMC or SVI and get an
*approximate* answer for orders of magnitude more wall-clock.

| Task (N) | pysp.ppl exact | NumPyro NUTS (2000 draws) | speedup | same answer? |
|---|---|---|---|---|
| Poisson-Gamma, N=200k | **3.6 ms** | 3721 ms (ESS 859) | **1027×** | yes — rate 3.5042 vs 3.5043 |
| Beta-Bernoulli, N=100k | **2.5 ms** | 2247 ms (ESS 647) | **906×** | yes — p 0.3720 vs 0.3720 |

This is the headline: for the large, common class of conjugate / exponential-family /
mixture-of-conjugate models, pysp.ppl isn't "competitive with" sampling PPLs — it sidesteps
sampling entirely, returns the *exact* posterior, and wins by ~1000×. `fit()` auto-detects
conjugacy (including **mixtures of conjugate priors**, an exact reweighted-mixture posterior)
and takes this path with no user intervention.

## 2. MLE / EM models

| Task (N) | pysp.ppl | Pyro SVI (AutoNormal) | speedup | same answer? |
|---|---|---|---|---|
| Gaussian MLE, N=500k | **36 ms** (closed-form EM) | 6313 ms | **175×** | yes — mean 4.993 vs 4.992 |

pysp runs the vectorized `seq_update` EM engine and converges in closed-form sufficient-statistic
updates; the gradient-PPL route (Pyro SVI / torch-Adam) iterates Adam to the same answer.

## 3. General (non-conjugate) MCMC — honest accounting

For a generic posterior with no closed form (here: Gaussian mean+sd, N=20k), pysp offers
RW-Metropolis (zero compile overhead) and analytic-gradient HMC (perfect mixing). This is the
class where JIT-compiled NUTS and ensemble samplers are strong, and we report it straight:

| sampler | wall-clock | ESS | ESS/sec | note |
|---|---|---|---|---|
| pysp RW-Metropolis | **416 ms** | 125 | 300 | fastest single-fit wall-clock |
| pysp HMC (analytic grad) | 7383 ms | 1000 | 135 | perfect mixing, torch per-step overhead |
| NumPyro NUTS | 1564 ms | 1473 | 942 | incl. JIT compile; best ESS/sec of the gradient samplers |
| emcee (ensemble) | 3448 ms | 32000 | 9280 | affine-invariant ensemble dominates ESS/sec here |

Honest takeaways for the general case:

* **Single-fit wall-clock**: pysp RW-Metropolis is fastest (416 ms) — no compile step, so for a
  one-shot fit you get an answer before NumPyro has finished tracing.
* **Mixing efficiency (ESS/sec)**: pysp does **not** lead here. NumPyro's JIT-compiled NUTS and
  emcee's ensemble sampler get more effective samples per second on this low-dimensional target.
  pysp HMC achieves the same *perfect* mixing as NUTS (ESS = draws) but pays a per-leapfrog Torch
  dispatch cost; closing that gap (fused exp-family gradients / a compiled trajectory) is the
  open optimization, tracked in the compute-engine design.

## Bottom line vs Stan / Pyro / NumPyro

* **Conjugate / exponential-family / mixture models** (the bulk of applied work): exact and
  **~1000× faster** — a capability the sampling PPLs do not have.
* **EM / MLE models**: **~175× faster**, same answer.
* **General non-conjugate posteriors**: competitive on single-fit wall-clock; NUTS/ensemble lead
  on ESS/sec. We report this rather than hide it.

So pysp.ppl's advantage is decisive precisely where a closed form or sufficient-statistic
structure exists — which it exploits automatically — and it remains a usable general sampler
elsewhere. Reproduce any row with `python -m pysp.ppl.benchmark_vs`.
