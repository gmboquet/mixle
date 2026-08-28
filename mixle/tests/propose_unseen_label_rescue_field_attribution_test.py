"""Pins T2-01's third occurrence: propose()'s unseen-label-rescue disclosure named the WRONG field
whenever a sequence/composite-decomposed field preceded the real excluded field in column order.

``_unseen_label_rescue`` names the excluded field by indexing its ``field_paths`` argument with a
TOP-LEVEL ``fitted.dists`` index (one entry per original column). ``propose()`` built that argument
from ``rec.fields``, which is LEAF-level -- a sequence field contributes multiple leaf paths (e.g.
``$[0]['element']``, ``$[0]['length']``) for the ONE top-level column it is. So the moment a
sequence/composite field precedes the real culprit column, every later top-level index reads the
wrong leaf's name out of that flat list: a 2-column (sequence, identifier) frame blamed
``$[0]['length']`` -- the sequence's own length sub-model, which scored perfectly finitely -- for
what was actually the identifier column, ``$[1]``, going unseen on every held-out row.
"""

from __future__ import annotations

import unittest

import numpy as np

import mixle


def _sequence_then_identifier_rows(n=300, seed=0):
    rng = np.random.RandomState(seed)
    sequences = [list(rng.randint(0, 10, size=int(rng.poisson(4)) + 1)) for _ in range(n)]
    user_id = [f"user_{i}" for i in range(n)]
    return [(sequences[i], user_id[i]) for i in range(n)]


def _rescue_note(model):
    notes = [f.get("partial_verification") for f in model.frontier if "partial_verification" in f]
    assert notes, f"expected a rescued/partially-verified candidate; frontier was {model.frontier}"
    return notes[0]


class ProposeUnseenLabelRescueFieldAttributionTest(unittest.TestCase):
    def test_sequence_field_ahead_of_the_real_culprit_is_attributed_correctly(self):
        rows = _sequence_then_identifier_rows()
        m = mixle.propose(rows, seed=0, max_candidates=3)
        note = _rescue_note(m)

        # The identifier column ($[1]) is the real culprit -- every held-out row's id is unseen at
        # train time. The sequence column's length sub-model ($[0]['length']) is well-behaved: a
        # Poisson-length field's held-out lengths overlap its training lengths almost completely.
        self.assertIn("$[1]", note, f"expected the identifier field ($[1]) named in: {note!r}")
        self.assertNotIn("$[0]", note, f"the well-behaved sequence field ($[0]) must not be blamed: {note!r}")

    def test_swapping_column_order_still_attributes_to_the_same_real_field(self):
        # Identifier column first this time -- it becomes $[0], the sequence field becomes $[1].
        # A correct fix must track the ACTUAL culprit column, not always default to a fixed index.
        rows = _sequence_then_identifier_rows()
        swapped = [(user_id, sequence) for sequence, user_id in rows]
        m = mixle.propose(swapped, seed=0, max_candidates=3)
        note = _rescue_note(m)

        self.assertIn("$[0]", note, f"expected the identifier field ($[0]) named in: {note!r}")
        self.assertNotIn("$[1]", note, f"the well-behaved sequence field ($[1]) must not be blamed: {note!r}")

    def test_both_orderings_exclude_the_same_amount_of_score(self):
        # The real culprit's contribution is excluded either way (heldout_mean_log_density agrees
        # across both column orders) -- pre-fix, only the NAME in the disclosure was wrong, not the
        # actual scoring; this pins that the fix does not change the (already-correct) scoring math.
        rows = _sequence_then_identifier_rows()
        swapped = [(user_id, sequence) for sequence, user_id in rows]
        m1 = mixle.propose(rows, seed=0, max_candidates=3)
        m2 = mixle.propose(swapped, seed=0, max_candidates=3)

        score1 = next(f["heldout_mean_log_density"] for f in m1.frontier if "partial_verification" in f)
        score2 = next(f["heldout_mean_log_density"] for f in m2.frontier if "partial_verification" in f)
        self.assertAlmostEqual(score1, score2, places=6)

    def test_clean_data_with_a_sequence_field_is_unaffected(self):
        # No unseen labels anywhere: the rescue path (and this fix's grouping logic) must never
        # trigger, and the sequence field's presence alone must not spuriously affect verification.
        rng = np.random.RandomState(0)
        n = 300
        sequences = [list(rng.randint(0, 5, size=int(rng.poisson(4)) + 1)) for _ in range(n)]
        amount = rng.normal(10, 2, size=n)
        rows = [(sequences[i], float(amount[i])) for i in range(n)]

        m = mixle.propose(rows, seed=0, max_candidates=3)
        self.assertFalse(any("partial_verification" in f for f in m.frontier))


if __name__ == "__main__":
    unittest.main()
