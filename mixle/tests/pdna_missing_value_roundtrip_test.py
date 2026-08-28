"""Regression coverage for T1-02: a model auto-fit (via optimize()) from data whose missing
entries are pandas' pd.NA singleton could not score or re-encode its own training data afterward,
even though the NaN-spelled equivalent round-tripped fine. campaign4_series_sentinel_test.py
already pins that a plain vs. nullable-dtype Series pick the SAME missing_value (dtype
consistency); it does not cover this: that raw pd.NA values, once a model exists, must still be
recognized as missing by log_density/seq_encode -- the round-trip-scoring half of the same defect.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from mixle.inference import optimize

_TRAIN = [37.0 + (i % 7) * 0.3 for i in range(300)]


class PdNaRoundtripScoringTest(unittest.TestCase):
    def test_model_fit_from_a_bare_list_of_pdna_scores_its_own_missing_rows(self):
        # This is the finding's exact reproduction shape: a nullable Series converted to a plain
        # Python list (as a caller assembling records from several pandas columns would do),
        # which has already lost the Series' dtype by the time it reaches optimize().
        s_na = pd.Series(_TRAIN + [None] * 5, dtype="Float64")
        col = list(s_na)
        self.assertTrue(col[-1] is pd.NA)  # genuine pd.NA objects, not None

        model = optimize(col, out=None)

        # log_density on the sentinel itself must not crash trying to float() an NAType.
        density = model.log_density(pd.NA)
        self.assertTrue(np.isfinite(density))

        # Re-encoding the SAME training data (still carrying raw pd.NA) must not crash either.
        encoded = model.dist_to_encoder().seq_encode(col)
        scores = model.seq_log_density(encoded)
        self.assertTrue(np.all(np.isfinite(scores)))
        # The 5 missing rows all cost the same sentinel log-density.
        np.testing.assert_allclose(scores[-5:], density)

    def test_matches_the_nan_spelled_equivalent(self):
        s_na = pd.Series(_TRAIN + [None] * 5, dtype="Float64")
        s_nan = pd.Series(_TRAIN + [float("nan")] * 5, dtype="float64")
        col_na, col_nan = list(s_na), list(s_nan)

        m_na = optimize(col_na, out=None)
        m_nan = optimize(col_nan, out=None)

        self.assertTrue(np.isnan(m_na.missing_value))
        self.assertEqual(m_na.p, m_nan.p)

        scores_na = m_na.seq_log_density(m_na.dist_to_encoder().seq_encode(col_na))
        scores_nan = m_nan.seq_log_density(m_nan.dist_to_encoder().seq_encode(col_nan))
        np.testing.assert_allclose(scores_na, scores_nan)

    def test_a_model_fit_directly_from_the_series_also_scores_raw_pdna(self):
        # Covers the OTHER shape the finding's guidance called out: even the Series-direct path
        # (whose dtype-consistency half campaign4_series_sentinel_test.py already pins) could not
        # score a raw pd.NA value before this fix, despite already carrying missing_value=nan.
        s_na = pd.Series(_TRAIN + [None] * 5, dtype="Float64")
        model = optimize(s_na, out=None)
        self.assertTrue(np.isfinite(model.log_density(pd.NA)))

    def test_int64_nullable_column_round_trips_too(self):
        s_na = pd.Series(list(range(300)) + [None] * 5, dtype="Int64")
        col = list(s_na)
        model = optimize(col, out=None)
        self.assertTrue(np.isnan(model.missing_value))
        self.assertTrue(np.isfinite(model.log_density(pd.NA)))
        scores = model.seq_log_density(model.dist_to_encoder().seq_encode(col))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_an_unrelated_explicit_sentinel_does_not_start_matching_pdna(self):
        # Guard against over-widening: pd.NA should only be treated as equivalent to a NaN
        # missing_value (mirroring how any two NaN floats already match), not to an arbitrary
        # numeric sentinel a caller picked on purpose.
        from mixle.inference import estimate
        from mixle.stats import GaussianEstimator, OptionalEstimator

        est = OptionalEstimator(GaussianEstimator(), missing_value=-999.0, est_prob=True)
        model = estimate(_TRAIN[:50] + [-999.0] * 3, est)
        self.assertEqual(model.missing_value, -999.0)
        with self.assertRaises(TypeError):
            model.log_density(pd.NA)


if __name__ == "__main__":
    unittest.main()
