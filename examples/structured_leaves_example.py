"""Heterogeneous structured leaves: a set-valued field inside a mixture.

The example uses *structured* leaves as first-class fields of a record: here a **set** field
(`BernoulliSetDistribution`: a variable-size set drawn from a vocabulary, each element included independently)
beside a Normal and a Poisson field, with all three fitted by one EM loop.

The model is ``Mixture(Composite(Gaussian, BernoulliSet, Poisson))`` over records ``(real, tag_set,
count)``. Two latent clusters differ in *every* field -- mean, tag-inclusion propensities, and rate. A
correct fit recovers all of them in each component, because the estimator/encoder trees mirror the model
tree (the structural-mirror invariant): the set field's inclusion probabilities are updated by the same
EM that updates the Normal's mean.

Run: ``python examples/structured_leaves_example.py``
"""

from __future__ import annotations

import numpy as np

from mixle.inference import optimize
from mixle.stats import BernoulliSetDistribution as BSet
from mixle.stats import CompositeDistribution as Comp
from mixle.stats import GaussianDistribution as G
from mixle.stats import MixtureDistribution as Mix
from mixle.stats import PoissonDistribution as P

VOCAB = ["x", "y", "z", "w"]
CLUSTERS = [
    {"mu": -3.0, "rate": 2.0, "incl": {"x": 0.85, "y": 0.80, "z": 0.10, "w": 0.10}},
    {"mu": 3.0, "rate": 9.0, "incl": {"x": 0.10, "y": 0.10, "z": 0.85, "w": 0.80}},
]


def make_data(n_per=700, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for cl in CLUSTERS:
        for _ in range(n_per):
            tag_set = [t for t in VOCAB if rng.rand() < cl["incl"][t]]
            rows.append((float(rng.normal(cl["mu"], 1.0)), tag_set, int(rng.poisson(cl["rate"]))))
    return rows


def main():
    data = make_data()
    uniform = {t: 0.5 for t in VOCAB}
    proto = Mix(
        [
            Comp((G(-2, 1), BSet(dict(uniform)), P(3.0))),
            Comp((G(2, 1), BSet(dict(uniform)), P(6.0))),
        ],
        [0.5, 0.5],
    )
    m = optimize(data, proto.estimator(), prev_estimate=proto, max_its=100, out=None)

    print("# Heterogeneous record (real, tag-SET, count) in a 2-component mixture\n")
    print("true clusters:")
    for cl in CLUSTERS:
        print(f"  mu={cl['mu']:+.1f}  rate={cl['rate']:.0f}  set.incl={cl['incl']}")
    print("\nfitted components (every field recovered by one EM):")
    ok = True
    for i, c in enumerate(sorted(m.components, key=lambda c: c.dists[0].mu)):
        incl = {t: round(float(c.dists[1].pmap.get(t, 0.0)), 2) for t in VOCAB}
        mu, rate = c.dists[0].mu, c.dists[2].lam
        print(f"  comp{i}: mean={mu:+.2f}  rate={rate:.2f}  set.incl={incl}")
        truth = CLUSTERS[i]
        ok = ok and abs(mu - truth["mu"]) < 0.3 and abs(rate - truth["rate"]) < 1.0
        ok = ok and all(abs(incl[t] - truth["incl"][t]) < 0.1 for t in VOCAB)
    print(f"\n-> all three fields (real, SET, count) recovered in every component: {ok}")
    print(
        "\nThe set field is a first-class structured leaf: its per-element inclusion probabilities are "
        "estimated by the same EM that fits the Normal and the Poisson, because the estimator tree mirrors "
        "the model tree. That structured-leaf depth -- not the mere act of composing -- is the moat."
    )


if __name__ == "__main__":
    main()
