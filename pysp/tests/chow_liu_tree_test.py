import unittest

import numpy as np

from pysp.stats import (
    BernoulliEstimator,
    CategoricalDistribution,
    ChowLiuTreeDistribution,
    ChowLiuTreeEstimator,
    GaussianDistribution,
    GaussianEstimator,
)
from pysp.utils.enumeration import freeze


class ChowLiuTreeTestCase(unittest.TestCase):

    @staticmethod
    def _fit(data, estimators, root=0):
        est = ChowLiuTreeEstimator(estimators, root=root)
        acc = est.accumulator_factory().make()
        for row in data:
            acc.update(row, 1.0, None)
        return est.estimate(len(data), acc.value())

    def test_recovers_noisy_binary_chain(self):
        rng = np.random.RandomState(13)
        data = []
        for _ in range(2000):
            a = int(rng.randint(0, 2))
            b = a ^ int(rng.rand() < 0.08)
            c = b ^ int(rng.rand() < 0.08)
            data.append((a, b, c))

        model = self._fit(data, [BernoulliEstimator()] * 3)
        edges = {frozenset((child, parent)) for child, parent in enumerate(model.parents)
                 if parent is not None}

        self.assertEqual(model.parents[0], None)
        self.assertIn(frozenset((0, 1)), edges)
        self.assertIn(frozenset((1, 2)), edges)
        self.assertEqual(len(edges), 2)
        self.assertTrue(np.isfinite(model.log_density((1, 1, 0))))

        enc = model.dist_to_encoder().seq_encode(data[:20])
        scalar = np.asarray([model.log_density(row) for row in data[:20]])
        np.testing.assert_allclose(model.seq_log_density(enc), scalar)

    def test_can_use_generic_child_estimators(self):
        rng = np.random.RandomState(19)
        data = []
        for _ in range(200):
            label = 'left' if rng.rand() < 0.55 else 'right'
            mean = -2.0 if label == 'left' else 3.0
            data.append((label, float(rng.normal(mean, 0.1))))

        model = self._fit(data, [
            CategoricalDistribution({'left': 0.5, 'right': 0.5}),
            GaussianDistribution(0.0, 1.0),
        ])

        self.assertEqual(model.parents, [None, 0])
        self.assertIsInstance(model.conditional_dists[1][freeze('left')], GaussianDistribution)
        self.assertIsInstance(model.conditional_dists[1][freeze('right')], GaussianDistribution)
        self.assertLess(model.conditional_dists[1][freeze('left')].mu, -1.8)
        self.assertGreater(model.conditional_dists[1][freeze('right')].mu, 2.8)
        self.assertTrue(np.isfinite(model.log_density(('left', -2.0))))
        sample = model.sampler(seed=1).sample()
        self.assertEqual(len(sample), 2)

    def test_enumerator_scores_finite_discrete_tree(self):
        dist = ChowLiuTreeDistribution(
            parents=[None, 0],
            marginal_dists=[
                BernoulliEstimator().estimate(None, (10.0, 6.0)),
                BernoulliEstimator().estimate(None, (10.0, 5.0)),
            ],
            conditional_dists=[
                {},
                {
                    freeze(False): BernoulliEstimator().estimate(None, (4.0, 1.0)),
                    freeze(True): BernoulliEstimator().estimate(None, (6.0, 5.0)),
                },
            ],
            default_dists=[None, BernoulliEstimator().estimate(None, (10.0, 5.0))],
            feature_order=[0, 1])

        items = list(dist.enumerator())
        self.assertEqual(len(items), 4)
        self.assertEqual(len({freeze(v) for v, _ in items}), 4)
        for value, lp in items:
            self.assertAlmostEqual(lp, dist.log_density(value), places=12)


if __name__ == '__main__':
    unittest.main()
