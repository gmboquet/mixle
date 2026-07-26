"""Explicit non-contiguous component layouts for resident DTensor statistics."""

import unittest
from dataclasses import dataclass

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import GaussianEstimator, MixtureEstimator
from mixle.stats.compute.stacked import (
    ComponentShardLayout,
    StackedMixtureResidentStats,
    estimate_component_shard_value,
    tie_component_shard_values,
)


@dataclass(frozen=True)
class _Chunk:
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]


class _ChunkedTensor:
    def __init__(self, local, chunks):
        self._local = np.asarray(local)
        self._chunks = tuple(_Chunk((start,), (size,)) for start, size in chunks)

    def to_local(self):
        return self._local

    def __create_chunk_list__(self):
        return self._chunks


class DTensorShardLayoutTest(unittest.TestCase):
    def setUp(self):
        self.estimator = MixtureEstimator([GaussianEstimator() for _ in range(7)])
        self.chunks = ((2, 1), (5, 2))
        self.counts = _ChunkedTensor([1.0, 2.0, 3.0], self.chunks)
        self.stats = (
            _ChunkedTensor([2.0, 10.0, 18.0], self.chunks),
            _ChunkedTensor([5.0, 52.0, 111.0], self.chunks),
            _ChunkedTensor([1.0, 2.0, 3.0], self.chunks),
            _ChunkedTensor([1.0, 2.0, 3.0], self.chunks),
        )

    def test_layout_exposes_indices_and_minimal_ranges(self):
        layout = ComponentShardLayout((2, 5, 6))
        self.assertFalse(layout.contiguous)
        self.assertEqual(layout.ranges, ((2, 3), (5, 7)))
        contiguous = ComponentShardLayout((2, 3, 4))
        self.assertTrue(contiguous.contiguous)
        self.assertEqual(contiguous.legacy_selection(), 2)

    def test_local_value_and_m_step_preserve_noncontiguous_indices(self):
        resident = StackedMixtureResidentStats(self.counts, self.stats, NUMPY_ENGINE, object)
        selection, value = resident.local_value()
        self.assertEqual(selection, (2, 5, 6))
        self.assertEqual(len(value[1]), 3)

        result = estimate_component_shard_value(self.estimator, selection, value, total_count=6.0)
        self.assertEqual(result.component_indices, (2, 5, 6))
        self.assertEqual(result.component_ranges, ((2, 3), (5, 7)))
        np.testing.assert_allclose(result.weights, np.array([1.0, 2.0, 3.0]) / 6.0)

        tied = tie_component_shard_values(self.estimator, ((selection, value),))
        self.assertEqual(tied[0][0], selection)

    def test_every_statistic_tensor_must_match_count_layout(self):
        mismatched = (
            _ChunkedTensor([2.0, 10.0, 18.0], ((2, 1), (4, 2))),
            self.stats[1],
        )
        resident = StackedMixtureResidentStats(self.counts, mismatched, NUMPY_ENGINE, object)
        with self.assertRaisesRegex(ValueError, "maps global components"):
            resident.local_value()

    def test_selection_cardinality_is_validated(self):
        with self.assertRaisesRegex(ValueError, "2 indices.*3 components"):
            estimate_component_shard_value(
                self.estimator,
                (2, 5),
                (np.ones(3), ((1.0, 1.0),) * 3),
                total_count=3.0,
            )


if __name__ == "__main__":
    unittest.main()
