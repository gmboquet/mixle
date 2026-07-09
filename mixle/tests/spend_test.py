"""Spend ledger (mixle.spend), CARD SPEND-a: a summable cost total shared across every spending subsystem."""

import unittest

from mixle.spend import Spend


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


if __name__ == "__main__":
    unittest.main()
