import unittest

import numpy as np

from pysp.engines import NumpyEngine, TorchEngine, torch
from pysp.stats import (
    BernoulliEstimator,
    CategoricalDistribution,
    ChowLiuTreeDistribution,
    ChowLiuTreeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    backend_seq_log_density,
    capabilities_for,
    declaration_for,
)
from pysp.utils.enumeration import freeze


def _assert_suff_close(test_case, actual, expected):
    if isinstance(actual, dict):
        test_case.assertEqual(set(actual.keys()), set(expected.keys()))
        for key in actual:
            _assert_suff_close(test_case, actual[key], expected[key])
        return
    if isinstance(actual, (tuple, list)):
        test_case.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            _assert_suff_close(test_case, a, e)
        return
    if actual is None or expected is None:
        test_case.assertEqual(actual, expected)
        return
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                                   rtol=1.0e-12, atol=1.0e-12)
        return
    if isinstance(actual, (str, bytes, bool)):
        test_case.assertEqual(actual, expected)
        return
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                               rtol=1.0e-12, atol=1.0e-12)


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

    def test_backend_scoring_and_metadata_delegate_to_children(self):
        dist = ChowLiuTreeDistribution(
            parents=[None, 0],
            marginal_dists=[
                CategoricalDistribution({'left': 0.55, 'right': 0.45}),
                GaussianDistribution(0.0, 1.0),
            ],
            conditional_dists=[
                {},
                {
                    freeze('left'): GaussianDistribution(-2.0, 0.25),
                    freeze('right'): GaussianDistribution(3.0, 0.5),
                },
            ],
            default_dists=[None, GaussianDistribution(0.0, 4.0)],
            feature_order=[0, 1])
        data = [('left', -2.1), ('right', 3.2), ('left', -1.9), ('right', 2.7)]
        enc = dist.dist_to_encoder().seq_encode(data)
        expected = dist.seq_log_density(enc)

        np.testing.assert_allclose(
            backend_seq_log_density(dist, enc, NumpyEngine()),
            expected,
            rtol=1.0e-12,
            atol=1.0e-12)
        if torch is not None:
            torch_scores = backend_seq_log_density(dist, enc, TorchEngine())
            np.testing.assert_allclose(
                TorchEngine().to_numpy(torch_scores),
                expected,
                rtol=1.0e-12,
                atol=1.0e-12)

        capabilities = capabilities_for(dist)
        self.assertEqual(capabilities.engine_ready, ('numpy', 'torch'))
        self.assertEqual(capabilities.kernel_status, 'generic_composite')

        declaration = declaration_for(dist)
        self.assertEqual(declaration.name, 'chow_liu_tree')
        self.assertEqual(declaration.statistic_names,
                         ('total_weight', 'num_features', 'marginal_counts', 'marginal_values',
                          'joint_counts', 'marginals', 'conditionals'))
        self.assertIn('marginal_0', declaration.child_roles)
        self.assertIn("conditional_1_given_0='left'", declaration.child_roles)
        self.assertFalse(declaration.differentiable)

    def test_accumulator_scale_matches_reweighted_update(self):
        data = [
            (False, False, False),
            (False, True, True),
            (True, True, True),
            (True, True, False),
            (True, False, False),
        ]
        weights = np.linspace(0.5, 1.5, len(data))
        c = 0.37
        estimator = ChowLiuTreeEstimator([BernoulliEstimator()] * 3)
        enc = estimator.accumulator_factory().make().acc_to_encoder().seq_encode(data)

        acc = estimator.accumulator_factory().make()
        acc.seq_update(enc, weights, None)
        self.assertIs(acc.scale(c), acc)

        expected = estimator.accumulator_factory().make()
        expected.seq_update(enc, weights * c, None)
        _assert_suff_close(self, acc.value(), expected.value())

        scaled_model = estimator.estimate(float(weights.sum() * c), acc.value())
        expected_model = estimator.estimate(float(weights.sum() * c), expected.value())
        self.assertEqual(scaled_model.parents, expected_model.parents)
        np.testing.assert_allclose(
            scaled_model.seq_log_density(scaled_model.dist_to_encoder().seq_encode(data)),
            expected_model.seq_log_density(expected_model.dist_to_encoder().seq_encode(data)),
            rtol=1.0e-10,
            atol=1.0e-10)


if __name__ == '__main__':
    unittest.main()
