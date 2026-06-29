"""automatic.get_prototype: the "I just have data, show me the model" front door.

get_estimator(data) returns the estimator; get_prototype(data) returns an initialized-but-unfitted
*distribution* whose tree mirrors the detected families, which the user can inspect / tweak and then fit.
"""

import unittest

import numpy as np

from mixle.inference import optimize
from mixle.utils.automatic import get_prototype


class GetPrototypeTest(unittest.TestCase):
    def test_returns_inspectable_composite_mirroring_fields(self):
        rng = np.random.RandomState(0)
        data = [(float(rng.normal(2, 1)), int(rng.poisson(4)), rng.choice(["a", "b"])) for _ in range(800)]
        proto = get_prototype(data, seed=0)
        self.assertEqual(type(proto).__name__, "CompositeDistribution")
        families = [type(d).__name__ for d in proto.dists]
        self.assertEqual(families, ["GaussianDistribution", "PoissonDistribution", "CategoricalDistribution"])

    def test_prototype_flows_straight_into_a_fit(self):
        rng = np.random.RandomState(1)
        data = [(float(rng.normal(2.0, 1.0)),) for _ in range(1000)]
        proto = get_prototype(data, seed=0)
        m = optimize(data, proto, max_its=30, out=None)
        self.assertAlmostEqual(float(m.dists[0].mu), 2.0, delta=0.15)

    def test_reproducible_with_seed(self):
        rng = np.random.RandomState(2)
        data = list(rng.normal(0, 1, 500))
        a = get_prototype(data, seed=7)
        b = get_prototype(data, seed=7)
        self.assertAlmostEqual(float(a.mu), float(b.mu))


if __name__ == "__main__":
    unittest.main()
