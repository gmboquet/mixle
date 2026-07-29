"""Fail-closed contracts for component-batched sampling and scatter."""

import unittest

import numpy as np

from mixle.stats.compute._sampling import scatter_component_draws


class _Sampler:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def sample(self, *, size):
        self.calls.append(size)
        return self.result(size) if callable(self.result) else self.result


class ComponentScatterContractTest(unittest.TestCase):
    def test_invalid_assignments_are_rejected_before_any_sampler_runs(self):
        sampler = _Sampler(lambda size: list(range(size)))
        invalid = (
            ([0, 2], 2, "between"),
            ([0, -1], 2, "between"),
            ([0.0, 0.0], 2, "exact integer"),
            ([True, False], 2, "booleans"),
            ([0], 2, "exactly size"),
            ([[0, 0]], 2, "one-dimensional"),
        )
        for state, size, message in invalid:
            with self.subTest(state=repr(state)), self.assertRaisesRegex(ValueError, message):
                scatter_component_draws(state, [sampler], size)
        self.assertEqual(sampler.calls, [])

    def test_size_must_be_a_positive_exact_integer(self):
        sampler = _Sampler([1])
        for size in (0, -1, 1.5, True):
            with self.subTest(size=repr(size)), self.assertRaisesRegex(ValueError, "positive integer"):
                scatter_component_draws([0], [sampler], size)
        self.assertEqual(sampler.calls, [])

    def test_sampler_batch_length_and_array_leading_shape_are_verified(self):
        with self.assertRaisesRegex(ValueError, "returned 1 draws, expected 2"):
            scatter_component_draws([0, 0], [_Sampler([1])], 2)
        with self.assertRaisesRegex(ValueError, r"leading shape \(2, \.\.\.\)"):
            scatter_component_draws([0, 0], [_Sampler(np.asarray([1]))], 2)

    def test_every_requested_draw_is_scattered_in_original_order(self):
        left = _Sampler(lambda size: np.arange(size, dtype=np.int64) + 10)
        right = _Sampler(lambda size: np.arange(size, dtype=np.float64) + 20.5)
        draws = scatter_component_draws([1, 0, 1, 0], [left, right], 4)
        np.testing.assert_allclose(draws, [20.5, 10.0, 21.5, 11.0])
        self.assertNotIn(None, draws)

    def test_heterogeneous_trailing_shapes_use_checked_list_scatter(self):
        vector = _Sampler(np.asarray([[1.0, 2.0]]))
        scalar = _Sampler(np.asarray([3.0]))
        draws = scatter_component_draws([0, 1], [vector, scalar], 2)
        np.testing.assert_array_equal(draws[0], [1.0, 2.0])
        self.assertEqual(draws[1], 3.0)


if __name__ == "__main__":
    unittest.main()
