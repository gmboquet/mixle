"""Fast contract tests for round-trip and abstention diagnostics."""

import unittest

import numpy as np

from mixle.reason.cycle_consistency import cycle_inconsistency, posterior_mean_estimate, selective_error


class _Sampler:
    def __init__(self, transform):
        self.transform = transform

    def sample_given_batch(self, values):
        return self.transform(np.asarray(values, dtype=float))


class CycleConsistencyContractTest(unittest.TestCase):
    def test_constant_wrong_round_trip_is_not_reported_as_perfect(self):
        sampler = _Sampler(lambda values: np.zeros_like(values))
        score = cycle_inconsistency(
            sampler,
            np.array([10.0]),
            n_draws=5,
            forward=lambda target: target,
            scale=2.0,
        )
        self.assertEqual(score, 25.0)

    def test_exact_round_trip_has_zero_scaled_error(self):
        sampler = _Sampler(lambda values: values / 2.0)
        score = cycle_inconsistency(
            sampler,
            np.array([4.0]),
            n_draws=5,
            forward=lambda target: 2.0 * target,
            scale=0.5,
        )
        self.assertEqual(score, 0.0)

    def test_round_trip_requires_positive_scale_and_valid_draws(self):
        sampler = _Sampler(lambda values: values[:-1])
        with self.assertRaisesRegex(ValueError, "rows"):
            cycle_inconsistency(
                sampler,
                np.array([1.0]),
                n_draws=5,
                forward=lambda target: target,
                scale=1.0,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            cycle_inconsistency(
                _Sampler(lambda values: values),
                np.array([1.0]),
                n_draws=5,
                forward=lambda target: target,
                scale=0.0,
            )

    def test_posterior_mean_validates_sample_count(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            posterior_mean_estimate(_Sampler(lambda values: values), np.array([1.0]), n_draws=0)

    def test_selective_error_requires_aligned_finite_examples(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            selective_error([1.0, 2.0], [0.1], 0.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            selective_error([1.0, np.nan], [0.1, 0.2], 0.5)

    def test_selective_error_uses_the_same_ranked_examples(self):
        self.assertEqual(selective_error([10.0, 1.0, 5.0], [0.3, 0.1, 0.2], 2 / 3), 3.0)


if __name__ == "__main__":
    unittest.main()
