"""Exchangeability preconditions (M2): the taxonomy, wired into create()/synthesize() provenance."""

import unittest

import numpy as np

from mixle.data.exchangeability import exchangeability_check
from mixle.inference import create, synthesize


def _rng():
    return np.random.RandomState(0)


class TaxonomyTest(unittest.TestCase):
    def test_iid_is_exchangeable(self):
        data = [float(x) for x in _rng().normal(5, 2, 200)]
        rep = exchangeability_check(data)
        self.assertEqual(rep.label, "exchangeable")
        self.assertTrue(rep.exchangeable)

    def test_trend_is_labeled_trend(self):
        rng = _rng()
        data = [float(0.05 * i + rng.randn()) for i in range(200)]
        self.assertEqual(exchangeability_check(data).label, "trend")

    def test_step_change_is_labeled_shift_not_trend(self):
        rng = _rng()
        data = [float(rng.randn()) for _ in range(100)] + [float(5 + rng.randn()) for _ in range(100)]
        # a step also rank-correlates with position; the within-half probe disambiguates
        self.assertEqual(exchangeability_check(data).label, "shift")

    def test_records_are_checked_per_numeric_field(self):
        rng = _rng()
        recs = [("a" if i % 2 else "b", float(0.05 * i + rng.randn())) for i in range(200)]
        rep = exchangeability_check(recs)
        self.assertEqual(rep.label, "trend")
        self.assertTrue(any(f["field"] == "field[1]" for f in rep.fields))

    def test_small_n_is_inconclusive_not_exchangeable(self):
        # MXR-080-0053: too little data to have testing power was never TESTED, so it must not be
        # reported as the same "exchangeable" verdict a genuine pass gets.
        rep = exchangeability_check([1.0, 2.0, 3.0])
        self.assertEqual(rep.label, "inconclusive")
        self.assertFalse(rep.exchangeable)
        self.assertIn("no power", rep.fields[0]["note"])

    def test_non_numeric_is_inconclusive_not_exchangeable(self):
        # MXR-080-0053: no numeric surface to probe is also untested, not a vacuous "exchangeable".
        rep = exchangeability_check(["a"] * 30)
        self.assertEqual(rep.label, "inconclusive")
        self.assertFalse(rep.exchangeable)
        self.assertIn("no numeric fields", rep.fields[0]["note"])


class InconclusiveAndInvalidTest(unittest.TestCase):
    """MXR-080-0053: untested (too little data, no numeric surface) and invalid (non-finite) inputs
    must be distinguishable from a genuine "tested and found exchangeable" verdict."""

    def test_dict_shaped_numeric_records_are_actually_probed(self):
        # previously: dict rows were never recognized as having a numeric surface, and silently fell
        # back to the vacuous "no numeric fields to test" note even though "x" is plainly numeric.
        recs = [{"x": float(x)} for x in _rng().normal(5, 2, 200)]
        rep = exchangeability_check(recs)
        self.assertEqual(rep.label, "exchangeable")
        self.assertEqual(rep.fields[0]["field"], "x")
        self.assertIn("trend_p", rep.fields[0])  # a real probe ran, not a vacuous note

    def test_dict_shaped_numeric_records_detect_a_real_trend(self):
        # proves dict support is a genuine probe (catches a real violation), not just a relabeling.
        recs = [{"x": float(0.05 * i)} for i in range(200)]
        rep = exchangeability_check(recs)
        self.assertEqual(rep.label, "trend")

    def test_dict_records_with_no_numeric_field_are_inconclusive(self):
        recs = [{"label": "a"} for _ in range(30)]
        rep = exchangeability_check(recs)
        self.assertEqual(rep.label, "inconclusive")
        self.assertFalse(rep.exchangeable)

    def test_all_nan_dataset_is_inconclusive_not_exchangeable(self):
        rep = exchangeability_check([float("nan")] * 30)
        self.assertEqual(rep.label, "inconclusive")
        self.assertFalse(rep.exchangeable)
        self.assertEqual(rep.fields[0]["verdict"], "invalid")

    def test_some_non_finite_values_flag_the_field_invalid(self):
        data = [float(x) for x in _rng().normal(5, 2, 30)]
        data[5] = float("inf")
        rep = exchangeability_check(data)
        self.assertEqual(rep.label, "inconclusive")
        self.assertEqual(rep.fields[0]["verdict"], "invalid")
        self.assertIn("1 of 30", rep.fields[0]["note"])

    def test_a_clean_field_is_still_tested_when_a_sibling_field_is_invalid(self):
        # one all-NaN field alongside a genuinely trending field: the clean field must still be
        # tested and drive the aggregate verdict, not be swallowed by its invalid sibling.
        rng = _rng()
        recs = [(float("nan"), float(0.05 * i + rng.randn())) for i in range(200)]
        rep = exchangeability_check(recs)
        self.assertEqual(rep.label, "trend")
        verdicts = {f["field"]: f["verdict"] for f in rep.fields}
        self.assertEqual(verdicts["field[0]"], "invalid")
        self.assertEqual(verdicts["field[1]"], "trend")


class WiringTest(unittest.TestCase):
    def test_create_records_the_verdict_in_provenance(self):
        rng = _rng()
        trend = [float(0.05 * i + rng.randn()) for i in range(200)]
        art = create(trend, seed=0)
        self.assertEqual(art.provenance["exchangeability"]["label"], "trend")  # the warning travels

    def test_synthesize_from_real_rows_records_the_verdict(self):
        data = [float(x) for x in _rng().normal(5, 2, 100)]
        ds = synthesize(data, n=10, seed=0)
        self.assertEqual(ds.provenance["exchangeability"]["label"], "exchangeable")

    def test_synthesize_from_a_callable_has_no_verdict(self):
        ds = synthesize(lambda rng: float(rng.randn()), n=5, seed=0)
        self.assertIsNone(ds.provenance["exchangeability"])  # nothing real to test; honest None


if __name__ == "__main__":
    unittest.main()
