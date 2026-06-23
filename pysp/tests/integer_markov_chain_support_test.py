"""IntegerMarkovChainEstimator must honor its declared num_values (support) when re-estimating.

Regression test: estimating from a data shard that does not contain every value used to *shrink*
num_values to the observed max, which then broke the index arithmetic (ValueError: invalid entry in
coordinates array) for any later shard / held-out sequence containing a higher value -- a frequent failure
when streaming or with rare symbols.
"""

import io
import unittest

import numpy as np

from pysp.inference.estimation import optimize
from pysp.stats import IntegerMarkovChainEstimator


class IntegerMarkovChainSupportTest(unittest.TestCase):
    def test_declared_num_values_is_preserved_on_sparse_data(self):
        rng = np.random.RandomState(0)
        sparse = [list(rng.randint(0, 8, 9)) for _ in range(40)]  # only tokens 0..7 appear
        m = optimize(sparse, IntegerMarkovChainEstimator(33, pseudo_count=0.5), max_its=1, rng=rng, out=io.StringIO())
        self.assertEqual(m.num_values, 33)
        self.assertEqual(np.asarray(m.cond_dist).shape, (33, 33))

    def test_scores_value_unseen_during_fit(self):
        rng = np.random.RandomState(1)
        m = optimize(
            [list(rng.randint(0, 8, 9)) for _ in range(40)],
            IntegerMarkovChainEstimator(33, pseudo_count=0.5),
            max_its=1,
            rng=rng,
            out=io.StringIO(),
        )
        ll = m.seq_log_density(m.dist_to_encoder().seq_encode([[30, 5, 30]]))  # token 30 never seen
        self.assertTrue(np.isfinite(ll[0]))


if __name__ == "__main__":
    unittest.main()
