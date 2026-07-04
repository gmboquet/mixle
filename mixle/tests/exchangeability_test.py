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

    def test_small_n_passes_with_a_no_power_note(self):
        rep = exchangeability_check([1.0, 2.0, 3.0])
        self.assertTrue(rep.exchangeable)
        self.assertIn("no power", rep.fields[0]["note"])

    def test_non_numeric_passes_vacuously(self):
        rep = exchangeability_check(["a"] * 30)
        self.assertTrue(rep.exchangeable)


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
