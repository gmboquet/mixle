"""Tests for the SemiMix semi-supervised mixture on the PPL surface (WS-F)."""

import importlib.util
import unittest

import numpy as np

from pysp.ppl import Normal, SemiMix, free
from pysp.stats.latent.ss_mixture import SemiSupervisedMixtureDistribution

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _labeled_data(seed=0):
    rng = np.random.RandomState(seed)
    left = rng.normal(-4.0, 1.0, 150)
    right = rng.normal(4.0, 1.0, 150)
    data = []
    # A few hard labels per component anchor the assignment; the rest are unlabeled.
    for i, x in enumerate(left):
        data.append((float(x), [(0, 1.0)] if i < 10 else None))
    for i, x in enumerate(right):
        data.append((float(x), [(1, 1.0)] if i < 10 else None))
    return data


class PplSemiMixTest(unittest.TestCase):
    def test_semimix_builds_ss_distribution_and_fits(self):
        data = _labeled_data()
        model = SemiMix([Normal(free, free), Normal(free, free)]).fit(data)
        result = model.dist
        self.assertIsInstance(result, SemiSupervisedMixtureDistribution)
        # The two component means should land near the true -4 / +4, anchored by the labels.
        means = sorted(float(c.mu) for c in result.components)
        self.assertLess(means[0], -2.0)
        self.assertGreater(means[1], 2.0)

    def test_semimix_scores_value_prior_pairs(self):
        data = _labeled_data()
        model = SemiMix([Normal(free, free), Normal(free, free)]).fit(data)
        result = model.dist
        # Unlabeled and labeled observations both score finitely.
        self.assertTrue(np.isfinite(result.log_density((3.5, None))))
        self.assertTrue(np.isfinite(result.log_density((3.5, [(1, 1.0)]))))


if __name__ == "__main__":
    unittest.main()
