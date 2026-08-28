"""Regression coverage for T4-03: optimize()/fit()'s default auto-inference (estimator=None)
silently reclassified a native +inf/-inf value as a fitted 'missingness' indicator
(OptionalDistribution(..., missing_value=+/-inf)) with zero warning -- contradicting both
optimize()'s own docstring (which lists only None/NaN/pd.NA/pd.NaT as the sentinels auto-inference
treats as gaps) and docs/stability-and-missing-data.rst's stated contract that default fitting
routes should reject non-finite observations instead of silently changing the data. The fitted
missingness rate itself was already correctly priced (log(p), not the earlier zero-cost-sentinel
bug); the defect is that the reclassification was undisclosed for the estimator=None route while
the SAME family, passed explicitly (bypassing auto-inference), correctly rejects a non-finite
observation outside its support.
"""

from __future__ import annotations

import math
import unittest
import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from mixle.inference import optimize
from mixle.stats import GaussianEstimator
from mixle.stats.combinator.optional import OptionalDistribution
from mixle.utils.automatic import get_estimator


def _finite_normal_plus_one_inf(seed: int, sign: float = 1.0) -> list[float]:
    rng = np.random.RandomState(seed)
    values = list(rng.normal(0.0, 1.0, size=300))
    values.append(sign * math.inf)
    return values


class AutoInferenceInfinityDisclosureTest(unittest.TestCase):
    def test_optimize_discloses_pos_inf_reclassification(self):
        # The finding's exact reproduction shape: a DataFrame column of otherwise-ordinary floats
        # with one stray +inf, fit via optimize()'s default auto-inference.
        df = pd.DataFrame({"x": _finite_normal_plus_one_inf(seed=0)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(df, max_its=5)

        # The fitted behavior itself is unchanged by this fix: still the missingness wrapper, with
        # the rate priced at exactly 1/301 (this is NOT the earlier zero-cost-sentinel bug).
        self.assertIsInstance(model, OptionalDistribution)
        self.assertEqual(model.missing_value, math.inf)
        self.assertAlmostEqual(model.p, 1.0 / 301.0)

        matches = [w for w in caught if issubclass(w.category, UserWarning) and "+inf/-inf" in str(w.message)]
        self.assertEqual(len(matches), 1, "expected exactly one disclosure warning, got: %r" % caught)
        self.assertIn("estimator=None", str(matches[0].message))

    def test_optimize_discloses_neg_inf_reclassification_too(self):
        df = pd.DataFrame({"x": _finite_normal_plus_one_inf(seed=1, sign=-1.0)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(df, max_its=5)

        self.assertIsInstance(model, OptionalDistribution)
        self.assertEqual(model.missing_value, -math.inf)

        matches = [w for w in caught if issubclass(w.category, UserWarning) and "+inf/-inf" in str(w.message)]
        self.assertEqual(len(matches), 1)

    def test_get_estimator_warns_directly_naming_the_field(self):
        # get_estimator() is the shared entry both optimize() and fit() reach for estimator=None;
        # pin the disclosure there too, on a bare list (no DataFrame column wrapping).
        data = _finite_normal_plus_one_inf(seed=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_estimator(data)

        matches = [w for w in caught if issubclass(w.category, UserWarning) and "+inf/-inf" in str(w.message)]
        self.assertEqual(len(matches), 1)
        self.assertIn("$", str(matches[0].message))  # format_path's root marker names the field

    def test_ordinary_finite_data_is_unaffected(self):
        # Regression guard (task step 3): the disclosure must be scoped exactly to the degenerate
        # (non-finite-carrying) case, not fire for ordinary well-scaled data on the same code path.
        finite = list(np.random.RandomState(3).normal(0.0, 1.0, size=300))
        df = pd.DataFrame({"x": finite})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(df, max_its=5)

        self.assertEqual(caught, [])
        self.assertNotIsInstance(model, OptionalDistribution)

    def test_explicit_estimator_bypasses_auto_inference_and_still_rejects_inf(self):
        # Contrast fixed by the finding's own evidence, not new behavior from this fix: passing the
        # family explicitly follows the stricter family-level default-route contract.
        with self.assertRaises(ValueError):
            optimize([1.0, 2.0, 3.0, math.inf, 2.5, 1.8], GaussianEstimator(), out=None)


if __name__ == "__main__":
    unittest.main()
