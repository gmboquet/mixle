"""Flagship app (G): a physics inverse problem with honest UQ, end to end.

Classification: illustrative -- runs on small synthetic / stand-in data. It shows the
end-to-end workflow shape, not measured results on a real frontier-scale dataset. See
docs/example-execution-manifest.rst for which examples run on real public data.

Recover a physical parameter -- the decay rate ``k`` of ``dy/dt = -k y`` -- from noisy observations of
the trajectory, as a Bayesian inverse problem:

  * the PHYSICS enters as the forward-model log-likelihood (a PPL ``potential``): the posterior over
    ``k`` is prior x physics-evidence, sampled by MCMC;
  * the UQ is HONEST: a credible interval per dataset, plus an ILLUSTRATION of the receipt that
    would matter at scale -- coverage across repeated noise draws, reported with its exact binomial
    interval. Twelve replicates cannot validate a 90% coverage claim (12/12 is consistent with any
    true coverage above ~74%), and the printout says so instead of calling coverage "checked";
  * the CERTIFICATE downgrades: a potential-augmented fit optimizes a modified objective, so the
    block's STATIONARY candidate is CAPPED to UNVERIFIED with the custom potential named in the
    reason, instead of a false closed-form claim. Both numbers are on the certificate --
    ``candidate_guarantee`` is what the block would earn, ``guarantee`` is what it earns without a
    receipt -- and the aggregate reports the capped one.

Everything measured in-process; ~a minute, no GPU, no network.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta

from mixle.ppl import Normal, potential

K_TRUE = 1.4
SIGMA = 0.03
T = np.linspace(0.1, 2.0, 40)


def observe(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.exp(-K_TRUE * T) + SIGMA * rng.randn(len(T))


def infer_k(y_obs: np.ndarray, seed: int, draws: int = 1500):
    """Posterior draws over k: broad prior + the physics forward model as evidence."""
    k = Normal(1.0, 2.0, name="k")

    def physics_ll(kv: float) -> float:
        return -0.5 * float(np.sum((y_obs - np.exp(-kv * T)) ** 2)) / SIGMA**2

    fit = Normal(k, 200.0).fit(  # a vacuous carrier observation; the potential IS the evidence
        [1.0],
        how="mcmc",
        potentials=potential(physics_ll, k),
        draws=draws,
        burn=400,
        rng=np.random.RandomState(seed + 1000),
    )
    return np.asarray(fit._result.samples()).ravel(), fit.certificate


def main() -> None:
    print("=" * 72)
    print(f"PHYSICS INVERSE: recover k from noisy decay observations (true k = {K_TRUE})")
    print("=" * 72)

    y = observe(0)
    draws, cert = infer_k(y, 0)
    lo, hi = np.quantile(draws, [0.05, 0.95])
    print(f"one dataset : posterior mean {draws.mean():.3f}, 90% CI [{lo:.3f}, {hi:.3f}]")
    print(
        f"certificate : {cert.guarantee.name} "
        f"(candidate {cert.blocks[0].candidate_guarantee.name}, capped by the physics potential)"
    )
    print(f"  {cert.blocks[0].reason.split('[')[-1].rstrip(']')}")

    # coverage of the 90% interval over repeated noise draws -- reported WITH its uncertainty.
    # STAT-R6: n=12 replicates illustrate the receipt's shape; they cannot validate a coverage
    # claim, and the exact Clopper-Pearson interval below is what n=12 can actually say.
    n_rep, hits = 12, 0
    for s in range(n_rep):
        d, _ = infer_k(observe(s), s, draws=800)
        lo, hi = np.quantile(d, [0.05, 0.95])
        hits += int(lo <= K_TRUE <= hi)
    ci_lo = float(beta.ppf(0.025, hits, n_rep - hits + 1)) if hits else 0.0
    ci_hi = float(beta.ppf(0.975, hits + 1, n_rep - hits)) if hits < n_rep else 1.0
    print(f"\ncoverage    : the 90% interval bracketed the truth {hits}/{n_rep} times")
    print(f"              exact 95% binomial CI for the coverage probability: [{ci_lo:.1%}, {ci_hi:.1%}]")
    print("twelve replicates ILLUSTRATE the coverage receipt -- consistent with the nominal 90%,")
    print("nowhere near validating it; a real calibration check needs hundreds of replicates.")


if __name__ == "__main__":
    main()
