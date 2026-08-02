"""Scoring is a hypothetical; acquisition happens once, after selection (MXR-080-1884).

``_score_candidate`` drew a realized outcome from the shared RNG for EVERY candidate it scored, and
then discarded all but the winner's. Scoring acquires nothing -- it asks what each candidate *would*
show -- so every rejected candidate was consuming the stream and shifting it for the one actually
picked. The belief the run continued from therefore depended on how many unrelated items happened to
be in the pool, which is not a property a seeded run may have.

Two smaller defects in the same surface: an outcome probability was validated through ``float()``
while the original value was retained and later multiplied, and a belief whose ``copy()`` returns
``self`` defeated the clone-before-update defence entirely.
"""

import tempfile
import unittest

import numpy as np
from numpy.random import RandomState

from mixle.inference.belief import CategoricalBelief
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.eig_retrieve import (
    EvidenceOutcome,
    EvidenceOutcomes,
    _clone,
    eig_retrieve,
)


class CountingRandomState(RandomState):
    """Counts realization draws so the test can assert WHEN the stream is consumed."""

    draws = 0

    def choice(self, *args, **kwargs):
        type(self).draws += 1
        return super().choice(*args, **kwargs)


def _uncertain(scale: float) -> EvidenceOutcomes:
    return EvidenceOutcomes([EvidenceOutcome(0.5, [scale, 0.0]), EvidenceOutcome(0.5, [0.0, scale])])


def _evidence_fn(item: SubstrateItem) -> EvidenceOutcomes:
    return _uncertain(0.05 if item.payload.get("tag") == "decoy" else 3.0)


def _run(n_decoys: int, k: int = 1):
    CountingRandomState.draws = 0
    substrate = Substrate(tempfile.mkdtemp())
    for index in range(n_decoys):
        substrate.put(SubstrateItem(kind="text", text=f"decoy{index}", payload={"tag": "decoy"}))
    substrate.put(SubstrateItem(kind="text", text="informative", payload={"tag": "info"}))
    retrieval = eig_retrieve(substrate, CategoricalBelief([0.5, 0.5]), _evidence_fn, k=k, seed=CountingRandomState(1))
    return retrieval, CountingRandomState.draws


class AcquisitionTimingTest(unittest.TestCase):
    def test_the_stream_is_consumed_once_per_selected_candidate_not_per_scored_one(self):
        for n_decoys in (0, 1, 3, 7):
            with self.subTest(decoys=n_decoys):
                retrieval, draws = _run(n_decoys)
                self.assertEqual(len(retrieval.items), 1)
                self.assertEqual(draws, 1)

    def test_the_selected_item_does_not_depend_on_pool_size(self):
        first, _ = _run(0)
        for n_decoys in (1, 3, 7):
            with self.subTest(decoys=n_decoys):
                other, _ = _run(n_decoys)
                self.assertEqual([i.text for i in other.items], [i.text for i in first.items])
                np.testing.assert_allclose(other.scores, first.scores)


class OutcomeProbabilityTest(unittest.TestCase):
    def test_a_probability_that_is_not_a_real_number_is_refused(self):
        for bad in ("0.5", True, None, [0.5]):
            with self.subTest(probability=repr(bad)):
                with self.assertRaises(TypeError):
                    EvidenceOutcome(bad, "evidence")

    def test_a_non_finite_or_negative_probability_is_refused(self):
        for bad in (float("nan"), float("inf"), -0.1):
            with self.subTest(probability=repr(bad)):
                with self.assertRaises(ValueError):
                    EvidenceOutcome(bad, "evidence")

    def test_the_checked_value_is_the_stored_value(self):
        # float() validated a conversion that was then thrown away; the raw value did the arithmetic.
        self.assertIsInstance(EvidenceOutcome(np.float32(0.5), "e").probability, float)
        self.assertEqual(EvidenceOutcome(np.float32(0.5), "e").probability, 0.5)


class CloneIndependenceTest(unittest.TestCase):
    def test_a_copy_that_returns_self_still_yields_an_independent_clone(self):
        class SelfCopy:
            def __init__(self):
                self.values = [1]

            def copy(self):
                return self

        original = SelfCopy()
        self.assertIsNot(_clone(original), original)

    def test_a_belief_that_cannot_be_cloned_at_all_is_refused(self):
        class Immovable:
            def copy(self):
                return self

            def __deepcopy__(self, memo):
                return self

        with self.assertRaisesRegex(ValueError, "returned the SAME object"):
            _clone(Immovable())

    def test_an_ordinary_copy_is_still_used(self):
        class RealCopy:
            def __init__(self, value=1):
                self.value = value

            def copy(self):
                return RealCopy(self.value)

        original = RealCopy()
        clone = _clone(original)
        self.assertIsNot(clone, original)
        self.assertEqual(clone.value, 1)


if __name__ == "__main__":
    unittest.main()
