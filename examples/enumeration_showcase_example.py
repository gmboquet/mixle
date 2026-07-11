"""Structural enumeration — a mixle capability no other PPL offers.

Classification: illustrative -- the discrete model is constructed in-file. Enumeration over a
composed model *is* the subject here, so there is no external dataset a stand-in would replace;
it demonstrates a capability, not a benchmark result on real data.

Stan / Pyro / NumPyro / PyMC can compute a *marginal* over discrete latents (variable elimination,
logsumexp). None gives you a probability-*ranked* view of a composed model's support: iterate it in
descending probability, ask the exact rank of a value, or unrank the k-th most probable value at an
arbitrary deep k -- all through the *same* recursion that fits the model.

This runs end to end (no downloads). It shows:
  1. top-k of a composed heterogeneous *discrete record* (category x count x count),
  2. exact rank(value) <-> seek(index) round-trip at the 10,000th most probable record,
  3. the 95% nucleus (smallest set covering 95% of the mass) without materializing the support,
  4. the honest story for a *non-decomposable* model (a mixture): a certified estimate with a
     standard error and an exact/approximate flag -- never a silent approximation.
"""

import numpy as np

from mixle.stats import (
    CompositeDistribution as Composite,
)
from mixle.stats import (
    GeometricDistribution,
    IntegerCategoricalDistribution,
    MixtureDistribution,
    PoissonDistribution,
)


def main():
    # A heterogeneous *record*: (category in {0,1,2}, a Poisson count, a Geometric count). Its support is
    # the full product -- effectively unbounded -- yet rank<->value is an exact count-DP at any depth.
    rec = Composite(
        (
            IntegerCategoricalDistribution(0, [0.5, 0.3, 0.2]),
            PoissonDistribution(4.0),
            GeometricDistribution(0.3),
        )
    )
    e = rec.enumerator()

    print("== composed discrete record:  (category, Poisson count, Geometric count) ==\n")

    print("1) the 5 most probable records (descending probability):")
    for value, lp in e.top_k(5):
        print(f"     {value}   p={np.exp(lp):.5f}")

    print("\n2) deep exact unranking + rank round-trip:")
    r = e.seek(10_000)  # the ~10,000th most probable record, by structural count-DP
    print(f"     seek(10_000) -> value {r.value}   log_prob {r.log_prob:.3f}")
    print(f"     certified rank bracket: [{r.rank_lower}, {r.rank_upper}]  (exact={r.exact})")
    back = e.rank(r.value)
    print(f"     rank({r.value}) -> {back.rank}  (round-trips to 10,000: {back.rank == 10_000})")

    print("\n3) the 95% nucleus (smallest set covering 95% of the mass), without enumerating the rest:")
    nuc = e.nucleus_size(0.95)
    print(f"     ~{nuc.size_lower}-{nuc.size_upper} records cover {nuc.covered_mass:.3f} of the probability")

    # Non-decomposable: a mixture's exact marginal rank is provably hard, so mixle returns a CERTIFIED
    # estimate (rank + standard error + exact flag), never a silent approximation.
    print("\n== non-decomposable model: Mixture(Poisson(2), Poisson(20)) ==\n")
    m = MixtureDistribution([PoissonDistribution(2.0), PoissonDistribution(20.0)], [0.5, 0.5])
    rr = m.enumerator().rank(5)
    print(f"4) rank(5) -> {rr.rank}  (exact={rr.exact}, stderr={rr.stderr:.3g}, method={rr.method!r})")
    print(f"     cumulative probability above value 5: {rr.cumulative_probability:.4f}")

    print(
        "\nNo Stan/Pyro/NumPyro/PyMC API answers 'what is the 10,000th most probable record' for a composed "
        "heterogeneous model. In mixle it is the same enumerator() recursion that fits the model."
    )


if __name__ == "__main__":
    main()
