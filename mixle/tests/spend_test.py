"""Spend ledger (mixle.system.spend), CARD SPEND-a: a summable cost total shared across every spending subsystem."""

import unittest

from mixle.system import Spend


class SpendTest(unittest.TestCase):
    def test_add_sums_every_field(self):
        a = Spend(frontier_calls=1, oracle_calls=2, wall_ms=10.0, dollars=0.5)
        b = Spend(frontier_calls=3, oracle_calls=0, wall_ms=5.0, dollars=1.5)
        total = a + b
        self.assertEqual(total, Spend(frontier_calls=4, oracle_calls=2, wall_ms=15.0, dollars=2.0))

    def test_zero_value_is_the_additive_identity(self):
        a = Spend(frontier_calls=2, oracle_calls=1, wall_ms=3.0, dollars=0.1)
        self.assertEqual(a + Spend(), a)

    def test_total_units_counts_frontier_and_oracle_calls_only(self):
        s = Spend(frontier_calls=2, oracle_calls=3, wall_ms=999.0, dollars=999.0)
        self.assertEqual(s.total_units(), 5.0)

    def test_to_dict_round_trips_construction(self):
        s = Spend(frontier_calls=1, oracle_calls=2, wall_ms=3.0, dollars=4.0)
        self.assertEqual(Spend(**s.to_dict()), s)


class SpendLedgerInvariantTest(unittest.TestCase):
    """MXR-080-1685: Spend had no construction invariants, so work could create budget."""

    def test_negative_counts_cannot_create_budget(self):
        self.assertEqual(Spend(frontier_calls=5).total_units(), 5.0)
        for kwargs in ({"frontier_calls": -5}, {"oracle_calls": -1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                Spend(**kwargs)

    def test_call_counts_must_be_exact_non_boolean_integers(self):
        for bad in (True, 1.5, "2", None):
            with self.subTest(value=repr(bad)), self.assertRaises(ValueError):
                Spend(frontier_calls=bad)

    def test_measured_costs_must_be_finite_and_nonnegative(self):
        for field in ("wall_ms", "dollars"):
            for bad in (float("nan"), float("inf"), -1.0):
                with self.subTest(field=repr(field), value=repr(bad)), self.assertRaises(ValueError):
                    Spend(**{field: bad})

    def test_invariants_are_rechecked_after_addition_and_round_trip(self):
        total = Spend(frontier_calls=2, wall_ms=1.0) + Spend(oracle_calls=3, dollars=0.5)
        self.assertEqual(total.total_units(), 5.0)
        self.assertEqual(Spend(**total.to_dict()), total)
        with self.assertRaises(ValueError):
            Spend(**{**total.to_dict(), "wall_ms": float("nan")})


if __name__ == "__main__":
    unittest.main()
