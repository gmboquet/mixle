"""Automatic dependency-structure learning: model the cross-field dependence a composite throws away.

mixle's tagline is "automatic inference for composable models of heterogeneous data." A CompositeDistribution
composes heterogeneous fields, but models them as *independent* -- and when the fields actually depend on one
another, that is badly wrong. This shows the gap and closes it: on a heterogeneous record where a category
shifts a real's mean and that real drives a count, ``learn_structure`` discovers the dependency graph across
mixed families and fits a joint model that beats the independent composite by hundreds of nats on held-out data
-- while still scoring, sampling, and composing like any mixle distribution.

Run: ``python structure_learning_example.py``
"""

from __future__ import annotations

import numpy as np

import mixle.stats as st
from mixle.inference import fit, learn_mixture_structure, learn_structure


def generate(seed: int, n: int = 800) -> list[tuple]:
    """Records (category, real, count) with real dependencies: category -> real mean, real -> count rate."""
    r = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        c = r.choice(["A", "B", "C"])
        mean = {"A": -4.0, "B": 0.0, "C": 4.0}[c]
        x = float(mean + r.randn())
        k = int(r.poisson(np.exp(x / 4.0 + 1.0)))
        rows.append((str(c), x, k))
    return rows


def total_ll(model, data) -> float:
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


def main() -> None:
    train, test = generate(1), generate(2)

    print("heterogeneous records: (category, real, count)")
    print(f"   e.g. {train[0]}\n")

    print("the usual composite -- fields modeled as INDEPENDENT:")
    composite = fit(
        train,
        st.CompositeEstimator((st.CategoricalEstimator(), st.GaussianEstimator(), st.PoissonEstimator())),
        max_its=40,
        out=None,
    )
    ll_comp = total_ll(composite, test)
    print(f"   held-out log-likelihood: {ll_comp:.1f}\n")

    print("learn_structure -- discover and model the cross-field dependence:")
    model = learn_structure(train)
    ll_struct = total_ll(model, test)
    names = {0: "category", 1: "real", 2: "count"}
    edges = ", ".join(f"{names[p]} -> {names[c]}" for p, c in model.edges()) or "none"
    print(f"   discovered dependencies: {edges}")
    print(f"   held-out log-likelihood: {ll_struct:.1f}")
    print(f"   => {ll_struct - ll_comp:+.1f} nats vs the independent composite (same data, same families)\n")

    # the deep form: two latent regimes, each with its OWN dependency structure
    two_regime_train = _two_regime(3)
    two_regime_test = _two_regime(4)
    print("two hidden regimes, each with a different (category -> real) relationship:")
    tree1 = learn_structure(two_regime_train)
    mix_ind = fit(
        two_regime_train,
        st.MixtureEstimator(
            [st.CompositeEstimator((st.CategoricalEstimator(), st.GaussianEstimator())) for _ in range(2)]
        ),
        max_its=80,
        out=None,
    )
    mot = learn_mixture_structure(two_regime_train, 2, restarts=4)
    print(f"   single dependency tree        : {total_ll(tree1, two_regime_test):.1f}")
    print(f"   mixture of INDEPENDENT models : {sum(mix_ind.log_density(d) for d in two_regime_test):.1f}")
    print(f"   MIXTURE OF DEPENDENCY TREES   : {total_ll(mot, two_regime_test):.1f}")
    print(f"   per-cluster structure found   : {[c.edges() for c in mot.components]}")
    print("   => only a mixture of trees discovers BOTH the clustering AND each cluster's dependence.")


def _two_regime(seed: int, n: int = 1600) -> list[tuple]:
    """Two latent clusters that differ in level and each carry a category -> real dependence."""
    r = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        z = r.randint(0, 2)
        c = "hi" if r.rand() < 0.5 else "lo"
        base = 5.0 if z == 0 else -5.0
        out.append((str(c), float(base + (3.0 if c == "hi" else -3.0) + r.randn())))
    return out


if __name__ == "__main__":
    main()
