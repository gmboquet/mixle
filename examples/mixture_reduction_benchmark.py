"""How much of the gap closes with ZERO gradient steps? Closed-form projection vs iterative refit.

The claim behind ``mixle.inference.project`` is that compressing a structured model onto a smaller one
is, for the right structure, a *closed-form projection* -- no sampling, no EM, no gradient steps. This
benchmark measures that against the honest iterative baseline on Gaussian-mixture reduction (the shape
that shows up as mixture-of-experts heads, Kalman/SSM belief states, GP posteriors):

* **closed-form**  ``reduce_mixture(teacher, M)`` -- Runnalls KL-greedy merging, a handful of
  log-determinants, no data touched.
* **iterative**    ``mixle.ops.project(teacher, M-component GMM)`` -- draw samples from the teacher and
  fit an M-component mixture by EM (the sample-and-refit projection).

Both are scored by the forward KL to the teacher, ``KL(teacher || reduced)``, estimated on a held-out
sample from the teacher (lower = closer). We report KL and wall-clock for each, across target sizes M.

Run:  python examples/mixture_reduction_benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

from mixle.inference.project import reduce_mixture
from mixle.ops import project
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution


def random_gmm(k: int, d: int, seed: int) -> GaussianMixtureDistribution:
    """A well-spread K-component GMM (random means, random diagonal-ish covariances, Dirichlet weights)."""
    rng = np.random.RandomState(seed)
    mus = rng.randn(k, d) * 4.0
    covs = []
    for _ in range(k):
        a = rng.randn(d, d) * 0.5
        covs.append(a @ a.T + np.diag(rng.uniform(0.3, 1.2, d)))  # SPD
    w = rng.dirichlet(np.ones(k) * 2.0)
    return GaussianMixtureDistribution(mus, np.stack(covs), w)


def mean_log_density(dist, xs) -> np.ndarray:
    return np.array([dist.log_density(x) for x in xs])


def kl_to_teacher(teacher, student, eval_x, teacher_logp) -> float:
    """KL(teacher || student) ~= mean_x~teacher [ log p(x) - log q(x) ]."""
    return float(np.mean(teacher_logp - mean_log_density(student, eval_x)))


def main() -> dict:
    d, K = 2, 12
    teacher = random_gmm(K, d, seed=1)
    eval_x = list(teacher.sampler(99).sample(5000))
    teacher_logp = mean_log_density(teacher, eval_x)

    print(f"teacher: {K}-component GMM in {d}-D; KL estimated on 5000 held-out samples\n")
    print(f"{'M':>3}  {'closed-form KL':>15} {'time':>9}   {'EM-refit KL':>12} {'time':>9}   {'speedup':>8}")
    rows = []
    for m in (1, 2, 4, 6, 8):
        t0 = time.time()
        cf = reduce_mixture(teacher, m)
        cf_t = time.time() - t0
        cf_kl = kl_to_teacher(teacher, cf, eval_x, teacher_logp)

        target = random_gmm(m, d, seed=7)  # a fresh M-component GMM to fit by EM
        t0 = time.time()
        em = project(teacher, target.estimator(), n_samples=4000, seed=0, max_its=50)
        em_t = time.time() - t0
        em_kl = kl_to_teacher(teacher, em, eval_x, teacher_logp)

        speedup = em_t / max(cf_t, 1e-9)
        print(f"{m:>3}  {cf_kl:>15.4f} {cf_t * 1e3:>7.1f}ms   {em_kl:>12.4f} {em_t * 1e3:>7.1f}ms   {speedup:>7.0f}x")
        rows.append({"M": m, "cf_kl": cf_kl, "cf_ms": cf_t * 1e3, "em_kl": em_kl, "em_ms": em_t * 1e3})

    mean_speed = np.mean([r["em_ms"] / max(r["cf_ms"], 1e-6) for r in rows])
    # how close is the closed form to the iterative fit, in KL? (ratio > 1 means EM fit tighter)
    ratios = [r["em_kl"] / r["cf_kl"] for r in rows if r["cf_kl"] > 1e-6]
    print(
        f"\nclosed-form is ~{mean_speed:.0f}x faster on average; "
        f"EM-refit KL / closed-form KL = {np.mean(ratios):.2f} on average "
        f"(>1 ⇒ closed-form is actually tighter; <1 ⇒ EM buys accuracy for the compute)."
    )
    return {"rows": rows, "mean_speedup": float(mean_speed)}


if __name__ == "__main__":
    main()
