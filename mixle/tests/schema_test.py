"""Typed structured schema (mixle.stats.combinator.schema): named, validated heterogeneous records."""

import math
import unittest

import numpy as np

import mixle.stats as st
from mixle.stats.combinator.schema import Schema


def _schema():
    return Schema.from_fields(
        [
            ("user", str, st.CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})),
            ("count", int, st.PoissonDistribution(3.0)),
            ("score", "number", st.GaussianDistribution(0.0, 1.0)),
        ]
    )


class SchemaTest(unittest.TestCase):
    def test_named_log_density_matches_positional_composite(self):
        s = _schema()
        record = {"user": "a", "count": 5, "score": 0.7}
        self.assertAlmostEqual(s.log_density(record), s.composite.log_density(("a", 5, 0.7)))

    def test_validate_rejects_type_violation(self):
        s = _schema()
        with self.assertRaises(TypeError):
            s.log_density({"user": "a", "count": "five", "score": 0.7})  # count must be int
        with self.assertRaises(TypeError):
            s.log_density({"user": 3, "count": 5, "score": 0.7})  # user must be str

    def test_validate_rejects_missing_or_extra_fields(self):
        s = _schema()
        with self.assertRaises(ValueError):
            s.to_tuple({"user": "a", "count": 5})  # missing 'score'
        with self.assertRaises(ValueError):
            s.to_tuple({"user": "a", "count": 5, "score": 0.7, "extra": 1})

    def test_numpy_scalars_pass_validation(self):
        s = _schema()
        rec = {"user": np.str_("b"), "count": np.int64(2), "score": np.float64(1.5)}
        self.assertTrue(math.isfinite(s.log_density(rec)))  # numpy scalar types accepted

    def test_sample_returns_typed_record_dicts(self):
        s = _schema()
        one = s.sample(seed=0)
        self.assertEqual(set(one), {"user", "count", "score"})
        self.assertIsInstance(one["user"], (str, np.str_))
        many = s.sample(seed=1, size=20)
        self.assertEqual(len(many), 20)
        self.assertTrue(all(set(r) == {"user", "count", "score"} for r in many))

    def test_tuple_round_trip(self):
        s = _schema()
        rec = {"user": "c", "count": 1, "score": -0.5}
        self.assertEqual(s.from_tuple(s.to_tuple(rec)), rec)

    def test_marginal_sub_schema(self):
        s = _schema()
        m = s.marginal(["score", "user"])
        self.assertEqual(m.names, ["score", "user"])
        # scoring the marginal works on the reduced record
        self.assertTrue(math.isfinite(m.log_density({"score": 0.1, "user": "a"})))

    def test_errors_for_bad_construction(self):
        with self.assertRaises(ValueError):
            Schema([])
        with self.assertRaises(ValueError):
            Schema.from_fields(
                [("x", int, st.PoissonDistribution(1.0)), ("x", int, st.PoissonDistribution(2.0))]
            )  # duplicate name


if __name__ == "__main__":
    unittest.main()
