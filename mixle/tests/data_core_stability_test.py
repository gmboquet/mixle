"""Two real bugs found in mixle/data/ beyond schema.py's Boolean/conform_record fixes (separate PR).

1. graph_source._coerce_pair_tuple lacked the fallback _coerce_pair_list has, so a plain adjacency
   matrix given as NESTED TUPLES (e.g. ((0, 1), (1, 0))) raised, while the identical input as nested
   LISTS was already handled correctly -- the exact same input shape, different outcome by type.
2. MaterializedSource silently accepted a one-shot iterator/generator despite its Sequence type hint;
   since materialize()/records() re-derive from self._data on every call (correct and cheap for a
   real Sequence, unlike LazySource's caching), a one-shot iterable would silently materialize
   empty/partial on the SECOND call, far from the constructor that actually caused it.
"""

import unittest

import mixle.stats  # noqa: F401  -- fully initialize the package to avoid a circular import
from mixle.data.core import MaterializedSource, as_source
from mixle.data.sources.graph_source import _extract_observation


class GraphPairTupleVsListConsistencyTest(unittest.TestCase):
    def test_a_plain_adjacency_matrix_as_nested_tuples_now_works(self):
        obs = _extract_observation(((0, 1), (1, 0)))
        self.assertEqual(obs.adjacency.tolist(), [[0.0, 1.0], [1.0, 0.0]])

    def test_nested_tuples_and_nested_lists_produce_the_same_result(self):
        obs_tuple = _extract_observation(((0, 1), (1, 0)))
        obs_list = _extract_observation([[0, 1], [1, 0]])
        self.assertEqual(obs_tuple.adjacency.tolist(), obs_list.adjacency.tolist())

    def test_a_genuine_adjacency_assignments_tuple_pair_still_works(self):
        # regression guard: the primary (adjacency, assignments) 2-tuple interpretation must be
        # unaffected -- only the fallback for a misinterpreted plain-matrix input is new.
        adj = [[0, 1], [1, 0]]
        obs = _extract_observation((adj, [0, 1]))
        self.assertEqual(obs.adjacency.tolist(), [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(list(obs.block_assignments), [0, 1])


class MaterializedSourceRequiresReiterableTest(unittest.TestCase):
    def test_a_one_shot_generator_is_rejected_at_construction(self):
        with self.assertRaises(TypeError):
            MaterializedSource(x for x in [1, 2, 3])

    def test_as_source_rejects_a_one_shot_generator_too(self):
        with self.assertRaises(TypeError):
            as_source(x for x in [1, 2, 3])

    def test_a_list_still_works(self):
        src = MaterializedSource([1, 2, 3])
        self.assertEqual(src.materialize(), [1, 2, 3])
        self.assertEqual(src.materialize(), [1, 2, 3])  # idempotent, re-derivable

    def test_a_numpy_array_still_works(self):
        # numpy.ndarray does NOT register as collections.abc.Sequence but IS safely re-iterable --
        # the fix must check __len__, not isinstance(..., Sequence).
        import numpy as np

        src = MaterializedSource(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(list(src.materialize()), [1.0, 2.0, 3.0])
        self.assertEqual(list(src.materialize()), [1.0, 2.0, 3.0])

    def test_a_tuple_still_works(self):
        src = MaterializedSource((1, 2, 3))
        self.assertEqual(src.materialize(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
