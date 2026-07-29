"""Fused E-steps reject unsafe observation-weight layouts before JIT entry."""

import unittest

import numpy as np

from mixle.stats import (
    CompositeDistribution,
    GaussianDistribution,
    MixtureDistribution,
)
from mixle.stats.compute.fused_codegen import fused_accumulate
from mixle.stats.compute.fused_nested import fused_nested_accumulate


def _flat_case():
    model = GaussianDistribution(0.0, 1.0)
    return fused_accumulate, model, model.dist_to_encoder().seq_encode([0.0, 1.0])


def _nested_case():
    mixture = MixtureDistribution(
        [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
        [0.5, 0.5],
    )
    model = CompositeDistribution((GaussianDistribution(0.0, 1.0), mixture))
    rows = [(0.0, -0.5), (1.0, 0.5)]
    return fused_nested_accumulate, model, model.dist_to_encoder().seq_encode(rows)


class FusedWeightValidationTest(unittest.TestCase):
    def test_weight_count_must_match_encoded_rows(self):
        for make_case in (_flat_case, _nested_case):
            accumulate, model, encoded = make_case()
            with self.subTest(path=repr(make_case.__name__), mismatch="short"):
                with self.assertRaisesRegex(ValueError, "2 encoded rows but 1 weights"):
                    accumulate(model, encoded, np.ones(1))
            with self.subTest(path=repr(make_case.__name__), mismatch="long"):
                with self.assertRaisesRegex(ValueError, "2 encoded rows but 3 weights"):
                    accumulate(model, encoded, np.ones(3))

    def test_weights_must_be_one_dimensional_and_finite(self):
        for make_case in (_flat_case, _nested_case):
            accumulate, model, encoded = make_case()
            with self.subTest(path=repr(make_case.__name__), invalid="matrix"):
                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    accumulate(model, encoded, np.ones((2, 1)))
            for invalid in (np.array([1.0, np.nan]), np.array([1.0, np.inf])):
                with self.subTest(path=repr(make_case.__name__), invalid=repr(invalid[1])):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        accumulate(model, encoded, invalid)


if __name__ == "__main__":
    unittest.main()
