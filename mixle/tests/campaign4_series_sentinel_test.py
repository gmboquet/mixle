"""Regression coverage for T2-02's Series half: campaign four found that ``optimize()``'s
auto-inference gave a bare pandas Series a different missing-value sentinel depending on whether
its dtype was plain (``NaN``) or nullable-extension (``pd.NA``), so a model fitted from one dtype
family could not score records built from the other. The DataFrame case was already fixed; this
pins the same fix applied to the three Series call sites: ``normalize_input`` (profiling, the
auto-inference entry), ``_data_records_for_encoding`` (the encode side optimize() actually uses),
and ``_tabular_records`` (the Model/propose entry).
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from mixle.inference import optimize
from mixle.lifecycle import Model
from mixle.stats import GaussianEstimator

_VALUES = [37.0, 41.2, 39.5, 44.1, 38.8, 40.0] * 25


class SeriesSentinelConsistencyTest(unittest.TestCase):
    def test_plain_and_nullable_series_pick_the_same_missing_value(self):
        plain = optimize(pd.Series(_VALUES + [np.nan]), out=None)
        nullable = optimize(pd.Series(_VALUES + [pd.NA], dtype="Float64"), out=None)
        self.assertEqual(type(plain), type(nullable))
        self.assertTrue(np.isnan(plain.missing_value))
        self.assertTrue(np.isnan(nullable.missing_value))

    def test_plain_and_nullable_series_score_a_held_out_value_identically(self):
        plain = optimize(pd.Series(_VALUES + [np.nan]), out=None)
        nullable = optimize(pd.Series(_VALUES + [pd.NA], dtype="Float64"), out=None)
        self.assertEqual(plain.log_density(37.0), nullable.log_density(37.0))
        self.assertTrue(np.isfinite(plain.log_density(37.0)))

    def test_model_fit_accepts_a_bare_series(self):
        model = Model(GaussianEstimator()).fit(pd.Series(_VALUES))
        self.assertEqual(type(model.fitted).__name__, "GaussianDistribution")

    def test_model_fit_on_a_nullable_series_matches_a_plain_one(self):
        # GaussianEstimator() is explicit, not auto-inference, so it correctly refuses raw missing
        # values on either dtype (only the Optional-wrapping auto-inference path handles them) --
        # this test is about column_records' dtype consistency, not missingness, so it uses data
        # with none.
        plain = Model(GaussianEstimator()).fit(pd.Series(_VALUES))
        nullable = Model(GaussianEstimator()).fit(pd.Series(_VALUES, dtype="Float64"))
        self.assertAlmostEqual(plain.fitted.mu, nullable.fitted.mu, places=9)
        for value in (np.nan, pd.NA):
            with self.assertRaises(ValueError):
                dtype = "Float64" if value is pd.NA else None
                Model(GaussianEstimator()).fit(pd.Series(_VALUES[:-5] + [value] * 5, dtype=dtype))

    def test_mapping_of_columns_normalizes_a_nullable_column(self):
        # Exercises the mapping branch's column_records call; the CategoricalDistribution result
        # for mapping-of-columns input is a separate, pre-existing behavior of optimize() unrelated
        # to pd.NA (confirmed: an all-float mapping with no missing values gives the same family),
        # so this only pins that the call does not raise and both dtypes give the SAME family.
        plain = optimize({"x": _VALUES + [None]}, out=None)
        nullable = optimize({"x": pd.array(_VALUES + [pd.NA], dtype="Float64")}, out=None)
        self.assertEqual(type(plain), type(nullable))


if __name__ == "__main__":
    unittest.main()
