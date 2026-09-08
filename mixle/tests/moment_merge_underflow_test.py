"""The central-moment merges survive the near-zero responsibilities EM hands a dying component.

The Pébay third-moment update divided by ``count**2``. A Dirichlet-process mixture switching a
component off gives it responsibilities around 1e-170; ``count`` is a valid positive weight there,
but ``count * count`` underflows to 0.0 and the merge raised ``ZeroDivisionError`` from inside
``fit`` (the model-based-embeddings notebook, 0.8.2 corpus). The two-division form is exact where
the old one was and finite everywhere ``count`` is.
"""

from __future__ import annotations

import unittest

import numpy as np

from mixle.stats import ExponentiallyModifiedGaussianEstimator, SkewNormalEstimator


class MomentMergeUnderflowTest(unittest.TestCase):
    ESTIMATORS = (ExponentiallyModifiedGaussianEstimator, SkewNormalEstimator)

    def _moments(self, estimator_cls, scale: float):
        rng = np.random.RandomState(0)
        x1, x2 = rng.normal(2.0, 1.0, 40), rng.normal(5.0, 1.5, 40)
        est = estimator_cls()
        acc = est.accumulator_factory().make()
        enc = est.dist_to_encoder() if hasattr(est, "dist_to_encoder") else None
        for x in (x1, x2):
            data = enc.seq_encode(x) if enc is not None else x
            acc.seq_update(data, np.full(len(x), scale), None)
        return acc.count, acc.mean, acc.m2 / acc.count, acc.m3 / acc.count

    def test_tiny_weights_merge_without_a_zero_division(self):
        for estimator_cls in self.ESTIMATORS:
            with self.subTest(estimator=estimator_cls.__name__):
                count, mean, var, third = self._moments(estimator_cls, 1e-170)
                self.assertGreater(count, 0.0)
                self.assertTrue(np.isfinite([mean, var, third]).all())

    def test_the_merged_moments_are_weight_scale_invariant(self):
        for estimator_cls in self.ESTIMATORS:
            with self.subTest(estimator=estimator_cls.__name__):
                _, mean1, var1, third1 = self._moments(estimator_cls, 1.0)
                _, mean2, var2, third2 = self._moments(estimator_cls, 1e-170)
                np.testing.assert_allclose([mean2, var2, third2], [mean1, var1, third1], rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
