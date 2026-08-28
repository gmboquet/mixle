"""Pins T2-01: one identifier-like/high-cardinality field must not void verification for the rest.

``propose()``'s scoring loop used to treat ANY non-finite held-out score anywhere in a candidate's
per-row score array as fatal for the WHOLE candidate. A moderately messy categorical field --
mostly common values, plus a handful of rare/near-singleton ones -- legitimately meets a held-out
label its training split never saw, and CategoricalDistribution's documented ``default_value=0.0``
scores that at ``-inf`` (mixle/stats/univariate/discrete/categorical.py). That single field's -inf
then poisoned the joint row score for every OTHER, perfectly well-behaved field too, so every
candidate "failed" and the whole frontier fell back to the unverified heuristic recommendation --
on one of the most common real tabular column shapes (a user_id/SKU-style field with a long tail).
"""

import unittest

import numpy as np

import mixle


def _records(n=300, seed=0):
    rng = np.random.RandomState(seed)
    amount = rng.normal(50, 10, size=n)
    plan = rng.choice(["basic", "pro", "enterprise"], size=n)
    active = rng.choice([True, False], size=n)
    # a moderately messy categorical column: four common labels (70 rows each) plus twenty
    # near-singleton labels (one row each) -- exactly the shape the finding measured as enough to
    # reproduce the collapse, not a literal one-row-per-label identifier column.
    common = np.repeat(["north", "south", "east", "west"], 70)
    singles = np.array([f"rare_{i}" for i in range(20)])
    messy_col = np.concatenate([common, singles])
    rng.shuffle(messy_col)
    return [(float(amount[i]), str(plan[i]), bool(active[i]), str(messy_col[i])) for i in range(n)]


class ProposeIdentifierFieldVerificationTest(unittest.TestCase):
    def test_messy_categorical_field_does_not_void_whole_frontier(self):
        m = mixle.propose(_records(), fit=True)

        verified = [f for f in m.frontier if "heldout_mean_log_density" in f]
        self.assertGreater(
            len(verified),
            0,
            f"expected at least one verified candidate; frontier was {m.frontier}",
        )
        self.assertFalse(
            any("no candidate could be verified" in n for n in m.notes),
            f"propose() fell back to the unverified heuristic; notes were {m.notes}",
        )
        self.assertEqual(m.evidence.get("certificate", {}).get("status"), "succeeded")

        # The exclusion is disclosed, not silent: some verified candidate names the field and the
        # documented cause rather than pretending every field scored cleanly.
        self.assertTrue(
            any("held-out verification score" in n and "unseen in training" in n for n in m.notes),
            f"expected the excluded-field disclosure in notes; got {m.notes}",
        )

    def test_clean_data_is_unaffected(self):
        # No messy field at all: the rescue path must never trigger, and scoring must be identical
        # to a plain, fully-finite verification (the guard is scoped to the non-finite branch only).
        rng = np.random.RandomState(0)
        n = 300
        amount = rng.normal(50, 10, size=n)
        plan = rng.choice(["basic", "pro", "enterprise"], size=n)
        rows = [(float(amount[i]), str(plan[i])) for i in range(n)]

        m = mixle.propose(rows, fit=True)

        self.assertFalse(any("partial_verification" in f for f in m.frontier))
        self.assertFalse(any("unseen in training" in n for n in m.notes))
        self.assertEqual(m.evidence.get("certificate", {}).get("status"), "succeeded")


if __name__ == "__main__":
    unittest.main()
