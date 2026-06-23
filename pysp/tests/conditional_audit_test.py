"""Regression: ConditionalDistributionDataEncoder.seq_encode must keep the three
parallel tuples (cond_vals / eobs_vals / idx_vals) aligned even when a conditioning
value is absent from dmap and there is no default.

Before the fix, the ``null_default_encoder`` branch of ``seq_encode`` appended to
``eobs_vals`` only when the key was present, while ``idx_vals`` was appended
unconditionally and ``cond_vals`` covered every group. That desync made
``eobs_vals`` shorter than the others, so ``seq_log_density`` / ``seq_update``
indexed past the end (IndexError) or silently mis-scored groups whenever the
absent key was not the last group iterated.
"""

import unittest

import numpy as np

from pysp.stats import PoissonDistribution
from pysp.stats.combinator.conditional import ConditionalDistribution


class ConditionalAbsentKeyAlignmentTest(unittest.TestCase):
    def setUp(self):
        # Two real conditional components, NO default distribution -> has_default False
        # (the documented -inf-for-absent-key case), and no given distribution.
        self.d = ConditionalDistribution({0: PoissonDistribution(2.0), 2: PoissonDistribution(9.0)})
        self.assertFalse(self.d.has_default)

    def _naive(self, data):
        return np.array([self.d.log_density(p) for p in data])

    def test_seq_encode_tuples_stay_aligned(self):
        # Absent conditioning value (1) is ordered FIRST so a desync mis-indexes.
        data = [(1, 5), (0, 3), (2, 8), (0, 1)]
        enc = self.d.dist_to_encoder()
        x = enc.seq_encode(data)
        _sz, cond_vals, eobs_vals, idx_vals, _given = x
        self.assertEqual(len(eobs_vals), len(cond_vals))
        self.assertEqual(len(idx_vals), len(cond_vals))

    def test_seq_log_density_matches_naive_with_absent_key(self):
        data = [(1, 5), (0, 3), (2, 8), (0, 1)]
        enc = self.d.dist_to_encoder()
        ld = self.d.seq_log_density(enc.seq_encode(data))
        ref = self._naive(data)
        # absent key -> -inf, present keys -> Poisson log-density
        np.testing.assert_array_equal(np.isneginf(ld), np.isneginf(ref))
        finite = np.isfinite(ref)
        np.testing.assert_allclose(ld[finite], ref[finite])

    def test_seq_update_does_not_crash_or_misalign(self):
        data = [(1, 5), (0, 3), (2, 8), (0, 1)]
        est = self.d.estimator()
        acc = est.accumulator_factory().make()
        enc = self.d.dist_to_encoder()
        weights = np.ones(len(data))
        # Must not raise (IndexError from desynced eobs_vals) and must consume the
        # aligned tuples. Pass the distribution as the prior estimate so the present
        # keys resolve through estimate.dmap.
        acc.seq_update(enc.seq_encode(data), weights, self.d)


if __name__ == "__main__":
    unittest.main()
