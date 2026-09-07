"""How much of the gap closes with ZERO gradient steps? Closed-form projection vs iterative refit.

The claim behind ``mixle.inference.project`` is that compressing a structured model onto a smaller one
is, for the right structure, a *closed-form projection* -- no sampling, no EM, no gradient steps. This
benchmark measures that against the honest iterative baseline on Gaussian-mixture reduction (the shape
that shows up as mixture-of-experts heads, Kalman/SSM belief states, GP posteriors):

* **closed-form**  ``reduce_mixture(teacher, M)`` -- Runnalls KL-greedy merging, a handful of
  log-determinants, no data touched.
* **EM (cold)**    ``mixle.ops.project(teacher, M-component GMM)`` -- draw samples from the teacher and
  fit an M-component mixture by EM from the estimator's own random start.
* **EM (warm)**    the same projection started from the closed-form solution (``init=``).

All three are scored by the forward KL to the teacher, ``KL(teacher || reduced)``, estimated on a
held-out sample from the teacher (lower = closer). We report KL and wall-clock for each, across
target sizes M.

Takeaway: the closed form is a strong INITIALIZATION, not a competitor. One cold EM start on this
teacher lands on a degenerate optimum and stays there -- its KL barely moves as M grows, which used
to read as the closed form dominating the refit and inverted the real picture (P10-F03). Started
from the closed-form solution, EM improves on it at every M and keeps improving as M grows. The cold
column is kept because it is what a single random restart actually buys you, and the gap between the
two columns is the point.

Scope: this VALIDATES the primitive on Gaussian mixtures, where ground truth is available. For the
same primitive applied to a real neural teacher (a trained RealNVP flow projected down to a structured
model), see ``project_neural_to_structured.py``.

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
    header = (
        f"{'M':>3}  {'closed-form KL':>15} {'time':>9}   "
        f"{'EM cold KL':>11} {'time':>9}   {'EM warm KL':>11} {'time':>9}"
    )
    print(header)
    rows = []
    for m in (1, 2, 4, 6, 8):
        t0 = time.time()
        cf = reduce_mixture(teacher, m)
        cf_t = time.time() - t0
        cf_kl = kl_to_teacher(teacher, cf, eval_x, teacher_logp)

        target = random_gmm(m, d, seed=7)  # a fresh M-component GMM to fit by EM
        t0 = time.time()
        cold = project(teacher, target.estimator(), n_samples=4000, seed=0, max_its=50, delta=None)
        cold_t = time.time() - t0
        cold_kl = kl_to_teacher(teacher, cold, eval_x, teacher_logp)

        # The same call, started from the closed-form solution: this is how the two primitives are
        # meant to be used together, and it is the column that answers "is the closed form enough?"
        t0 = time.time()
        warm = project(teacher, cf, n_samples=4000, seed=0, max_its=50, init=cf, delta=None)
        warm_t = time.time() - t0
        warm_kl = kl_to_teacher(teacher, warm, eval_x, teacher_logp)

        print(
            f"{m:>3}  {cf_kl:>15.4f} {cf_t * 1e3:>7.1f}ms   "
            f"{cold_kl:>11.4f} {cold_t * 1e3:>7.1f}ms   {warm_kl:>11.4f} {warm_t * 1e3:>7.1f}ms"
        )
        rows.append(
            {
                "M": m,
                "cf_kl": cf_kl,
                "cf_ms": cf_t * 1e3,
                "em_kl": cold_kl,
                "em_ms": cold_t * 1e3,
                "warm_kl": warm_kl,
                "warm_ms": warm_t * 1e3,
            }
        )

    mean_speed = np.mean([r["em_ms"] / max(r["cf_ms"], 1e-6) for r in rows])
    improved = sum(1 for r in rows if r["warm_kl"] <= r["cf_kl"] + 1e-12)
    cold_monotone = all(b["em_kl"] <= a["em_kl"] + 1e-12 for a, b in zip(rows, rows[1:]))
    warm_monotone = all(b["warm_kl"] <= a["warm_kl"] + 1e-12 for a, b in zip(rows, rows[1:]))
    print(
        f"\nclosed-form is ~{mean_speed:.0f}x faster than an EM refit; warm-started EM matched or beat "
        f"the closed form at {improved}/{len(rows)} target sizes."
    )
    print(
        f"KL falls monotonically with M: warm-started EM {warm_monotone}, single cold restart "
        f"{cold_monotone} -- a refit that does not improve with more components is a stuck restart, "
        f"not evidence about the closed form."
    )
    return {"rows": rows, "mean_speedup": float(mean_speed)}


if __name__ == "__main__":
    main()
