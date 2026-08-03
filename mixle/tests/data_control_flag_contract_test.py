"""Data-layer semantic flags are exact Booleans, not truthiness (MXR-080-1886).

``bool("false")`` is ``True``, and a flag read from a config file, an environment variable or a CLI
argument arrives as a string. Both controls here change what the data MEANS rather than merely how it
is presented:

* ``shuffle`` decides dataset order, so ``shuffle="false"`` shuffled anyway -- making a run that asked
  for a deterministic order non-deterministic, and silently changing what the model saw.
* ``directed`` decides which entries of an adjacency matrix are read and what symmetry is required,
  and it participates in ``GraphDataEncoder._signature()`` -- so ``directed="false"`` both executed a
  directed encoder and wrote that wrong semantics into the encoder identity that
  ``save_encoded``/``load_encoded`` key their compatibility check on.

``seed`` is here for the same reason: ``RandomState(True)`` seeds with 1 rather than raising, so the
comment claiming the constructor validated the seed was wrong about the one input that matters.
"""

import unittest

import numpy as np

from mixle.data.sources.graph_source import GraphDataEncoder
from mixle.data.stream_token_source import stream_token_source

NOT_A_BOOL = ("false", "true", "", 0, 1, None, [], 2.0)


class ShuffleFlagTest(unittest.TestCase):
    def _ids(self) -> np.ndarray:
        return np.arange(64)

    def test_a_non_boolean_shuffle_is_refused(self):
        for value in NOT_A_BOOL:
            with self.subTest(shuffle=repr(value)):
                with self.assertRaisesRegex(TypeError, "must be an actual Boolean"):
                    stream_token_source(self._ids(), 4, 2, shuffle=value)

    def test_both_real_booleans_still_work(self):
        for value in (True, False):
            with self.subTest(shuffle=value):
                batches = list(stream_token_source(self._ids(), 4, 2, epochs=1, shuffle=value, seed=0))
                self.assertTrue(batches)

    def test_shuffle_false_really_is_source_order(self):
        ordered = list(stream_token_source(self._ids(), 4, 2, epochs=1, shuffle=False, seed=0))
        again = list(stream_token_source(self._ids(), 4, 2, epochs=1, shuffle=False, seed=7))
        # No shuffling means the seed cannot matter; under truthiness this was not guaranteed.
        for (left, _), (right, _) in zip(ordered, again):
            np.testing.assert_array_equal(left, right)


class SeedFlagTest(unittest.TestCase):
    def test_a_boolean_seed_is_refused(self):
        # RandomState(True) silently seeds with 1.
        with self.assertRaises((TypeError, ValueError)):
            stream_token_source(np.arange(64), 4, 2, seed=True)

    def test_an_ordinary_seed_still_works(self):
        self.assertTrue(list(stream_token_source(np.arange(64), 4, 2, seed=3)))


class DirectedFlagTest(unittest.TestCase):
    def test_a_non_boolean_directed_is_refused(self):
        for value in NOT_A_BOOL:
            with self.subTest(directed=repr(value)):
                with self.assertRaisesRegex(TypeError, "must be an actual Boolean"):
                    GraphDataEncoder(directed=value)

    def test_both_real_booleans_still_work(self):
        self.assertIs(GraphDataEncoder(directed=True).directed, True)
        self.assertIs(GraphDataEncoder(directed=False).directed, False)

    def test_the_encoder_identity_reflects_the_declared_semantics(self):
        # `directed` participates in the signature save_encoded/load_encoded compatibility is keyed
        # on, which is why accepting a truthy string mattered beyond this constructor.
        self.assertNotEqual(str(GraphDataEncoder(directed=True)), str(GraphDataEncoder(directed=False)))


if __name__ == "__main__":
    unittest.main()
