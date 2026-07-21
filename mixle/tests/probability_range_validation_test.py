"""Constructor-time probability-range validation (0.8.0 audit follow-up).

PR #433 (worklist S-3) rejected out-of-range probabilities at construction for
``CategoricalDistribution`` and ``MixtureDistribution``'s component/weight length match, on the
principle that an invalid "probability" must fail at the constructor rather than silently
propagate into ``log_density()`` as ``nan`` or an out-of-[0,1] density. The fix never reached
these structurally identical siblings, so each accepted invalid input and produced a silently
wrong answer instead of a clear error. Grouped here because they are one bug class, not four.
"""

import unittest

import numpy as np

from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution
from mixle.stats.univariate.discrete.integer_uniform_spike import IntegerUniformSpikeDistribution


class MixtureWeightValidationTestCase(unittest.TestCase):
    def test_negative_weight_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)], [-0.5, 1.5])

    def test_valid_weights_still_construct(self):
        m = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
        self.assertTrue(np.isfinite(m.log_density(0.5)))


class IntegerCategoricalValidationTestCase(unittest.TestCase):
    def test_negative_or_over_one_probability_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            IntegerCategoricalDistribution(min_val=0, p_vec=[-0.5, 1.5])

    def test_valid_probabilities_still_construct(self):
        d = IntegerCategoricalDistribution(min_val=0, p_vec=[0.3, 0.7])
        self.assertTrue(np.isfinite(d.log_density(0)))


class IntegerUniformSpikeValidationTestCase(unittest.TestCase):
    def test_out_of_range_p_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            IntegerUniformSpikeDistribution(k=0, num_vals=3, p=1.5, min_val=0)
        with self.assertRaises(ValueError):
            IntegerUniformSpikeDistribution(k=0, num_vals=3, p=-0.1, min_val=0)

    def test_valid_p_still_constructs(self):
        d = IntegerUniformSpikeDistribution(k=0, num_vals=3, p=0.6, min_val=0)
        self.assertTrue(np.isfinite(d.log_density(0)))


if __name__ == "__main__":
    unittest.main()
