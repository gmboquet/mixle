"""Copula / vine dependence-structure fitting: joint tails a Gaussian-copula assumption would miss.

mixle's 0.7.0 line added copula/vine dependence-structure fitting -- "fit a copula or vine when continuous
columns are correlated, heavy joint tails included." This demonstrates the direct, test-proven path:
:class:`~mixle.stats.combinator.copula.CopulaDistribution` (mixle.stats.combinator.copula) glues arbitrary
marginals to a pluggable dependence core via Sklar's theorem, and
:class:`~mixle.stats.multivariate.rvine_copula.RVineCopulaDistribution` is a regular vine that automatically
discovers its tree structure AND a per-edge pair-copula family (Dissmann's algorithm) from data.

The scenario: three heterogeneous columns (Gamma, Gaussian, Gamma) generated with genuine LOWER-TAIL
dependence via a Clayton copula -- the kind of joint co-crash a symmetric Gaussian copula cannot represent
(it is tail-independent by construction, no matter how strong its correlation). We fit the vine directly and
print what it actually found (the per-edge families), then quantify the payoff instead of just asserting it:
draw from the fitted dependence core, measure the empirical joint lower-tail rate, and compare it against the
independence baseline AND a Gaussian-copula-cored fit on the SAME data. The vine's fitted rate lands close to
the true generator's; the Gaussian copula understates it, because ellipticity cannot see the tail co-movement.

(There is also an automatic path -- bare ``optimize(data, out=None)`` runs a BIC-driven internal search over
copula cores, see ``mixle.inference.copula_structure`` -- but it does not surface the fitted structure as
directly, so this example uses the explicit constructor path.)

Run: ``python examples/copula_vine_example.py``
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma as spgamma
from scipy.stats import norm as spnorm

import mixle.stats as st
from mixle.inference import optimize
from mixle.stats.combinator.copula import CopulaDistribution
from mixle.stats.multivariate.clayton_copula import ClaytonCopulaDistribution
from mixle.stats.multivariate.gaussian_copula import GaussianCopulaDistribution
from mixle.stats.multivariate.rvine_copula import RVineCopulaDistribution

DIM = 3
THETA_TRUE = 4.0  # Clayton lower-tail parameter; Kendall's tau = theta / (theta + 2) ~= 0.67 (strong)
TAIL_Q = 0.1  # "joint lower tail": all columns below their own 10th percentile, simultaneously


def generate(seed: int, n: int) -> list[tuple[float, float, float]]:
    """Heterogeneous (Gamma, Gaussian, Gamma) columns coupled by genuine Clayton lower-tail dependence.

    Sample uniform scores from a Clayton copula (real joint lower-tail co-movement), then push each column
    through its own inverse CDF -- the probability-integral transform in reverse -- so the data looks like a
    realistic heterogeneous dataset, not literally uniform.
    """
    u = ClaytonCopulaDistribution(DIM, theta=THETA_TRUE).sampler(seed).sample(n)
    x0 = spgamma.ppf(u[:, 0], a=2.0, scale=2.0)
    x1 = spnorm.ppf(u[:, 1], loc=5.0, scale=2.0)
    x2 = spgamma.ppf(u[:, 2], a=1.5, scale=3.0)
    return list(zip(x0.tolist(), x1.tolist(), x2.tolist()))


def prototype(copula) -> CopulaDistribution:
    """A CopulaDistribution prototype: heterogeneous marginals + a pluggable dependence core to fit."""
    return CopulaDistribution(
        [st.GammaDistribution(1.0, 1.0), st.GaussianDistribution(0.0, 1.0), st.GammaDistribution(1.0, 1.0)],
        copula,
    )


def joint_lower_tail_rate(u: np.ndarray, q: float = TAIL_Q) -> float:
    """Empirical P(every column < q) on uniform margins -- q doubles as the per-column threshold."""
    return float(np.mean((u < q).all(axis=1)))


def main() -> None:
    train = generate(seed=0, n=5000)
    print("# mixle copula / vine dependence-structure fitting\n")
    print(f"{DIM} heterogeneous columns (Gamma, Gaussian, Gamma), Clayton-coupled (theta={THETA_TRUE}, lower tail)")
    print(f"  e.g. row 0: {tuple(round(v, 2) for v in train[0])}\n")

    # --- fit the R-vine-cored copula directly: the explicit, test-proven path ---------------------------
    print("fitting CopulaDistribution([Gamma, Gaussian, Gamma], RVineCopulaDistribution(3, [])) ...")
    vine_proto = prototype(RVineCopulaDistribution(DIM, []))
    vine_fit = optimize(train, vine_proto.estimator(), prev_estimate=vine_proto, max_its=3, out=None)
    edges = [(e.a, e.b, e.copula.family) for tree in vine_fit.copula.trees for e in tree]
    print(
        f"   fitted marginals: Gamma(k={vine_fit.marginals[0].k:.2f}), "
        f"Gaussian(mu={vine_fit.marginals[1].mu:.2f}), Gamma(k={vine_fit.marginals[2].k:.2f})"
    )
    print(f"   discovered edges (var_a, var_b, family): {edges}\n")

    # --- contrast: same data, a Gaussian-copula core (elliptical, tail-independent by construction) ------
    print("fitting the same data with a GaussianCopulaDistribution core (elliptical, tail-independent) ...")
    gauss_proto = prototype(GaussianCopulaDistribution(np.eye(DIM)))
    gauss_fit = optimize(train, gauss_proto.estimator(), prev_estimate=gauss_proto, max_its=3, out=None)
    print(f"   fitted correlation matrix:\n{np.round(gauss_fit.copula.corr, 3)}\n")

    # --- the payoff: quantify captured joint lower-tail rate, do not just assert it -----------------------
    n_check = 100_000
    u_true = ClaytonCopulaDistribution(DIM, theta=THETA_TRUE).sampler(1).sample(n_check)
    u_vine = vine_fit.copula.sampler(2).sample(n_check)
    u_gauss = gauss_fit.copula.sampler(3).sample(n_check)

    rate_indep = TAIL_Q**DIM
    rate_true = joint_lower_tail_rate(u_true)
    rate_vine = joint_lower_tail_rate(u_vine)
    rate_gauss = joint_lower_tail_rate(u_gauss)

    print(f"joint lower-tail rate P(all {DIM} columns < {TAIL_Q:.0%} quantile), n={n_check}:")
    print(f"   independence baseline ({TAIL_Q}^{DIM})    : {rate_indep:.5f}")
    print(f"   true Clayton generator             : {rate_true:.5f}  ({rate_true / rate_indep:6.1f}x independence)")
    print(f"   fitted R-vine (per-edge search)    : {rate_vine:.5f}  ({rate_vine / rate_indep:6.1f}x independence)")
    print(f"   fitted Gaussian copula (elliptical): {rate_gauss:.5f}  ({rate_gauss / rate_indep:6.1f}x independence)")
    print(
        f"\n=> the vine recovers {rate_vine / rate_true:.0%} of the true joint-crash rate; the Gaussian copula, "
        f"tail-independent by construction, recovers only {rate_gauss / rate_true:.0%} of it -- same data, "
        "same IFM fit procedure, only the dependence-core family differs."
    )
    print(
        "\n(bare optimize(data, out=None) reaches a copula automatically via a BIC-driven search over cores -- "
        "see mixle.inference.copula_structure -- this example uses the explicit constructor path so the fitted "
        "structure above is directly visible.)"
    )


if __name__ == "__main__":
    main()
