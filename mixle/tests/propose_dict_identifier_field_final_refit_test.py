"""T3-01: ``_refresh_frozen_identifier_leaves`` (mixle/lifecycle.py) closed the final-refit crash from
``propose_identifier_field_final_refit_test.py`` only for TUPLE-shaped rows, because its guard was
``if not isinstance(estimator, CompositeEstimator) ...: return estimator``. Dict-shaped rows (the
ordinary meaning of ``propose()``'s own documented "a list of records" input, and what
``df.to_dict('records')`` produces) build a ``RecordEstimator`` (mixle/stats/combinator/record.py),
not a ``CompositeEstimator`` -- so that guard made the refresh a silent no-op for dict rows, and the
exact same "fused EM did not produce a finite objective from its non-finite initial model" crash
reproduced identically whenever the winning candidate carried a frozen (Ignored) identifier-like
field. This file is the dict-shaped mirror of that test file; it must have failed before the fix
extended ``_refresh_frozen_identifier_leaves`` to also rebuild ``RecordEstimator``'s keyed children.
"""

import unittest

import numpy as np

import mixle


def _dict_records_with_unique_identifier(n=300, seed=0):
    rng = np.random.RandomState(seed)
    amount = rng.normal(50, 10, size=n)
    plan = rng.choice(["basic", "pro", "enterprise"], size=n)
    active = rng.choice([True, False], size=n)
    ident = np.array([f"id_{i}" for i in range(n)])
    rng.shuffle(ident)
    return [
        {"amount": float(amount[i]), "plan": str(plan[i]), "active": bool(active[i]), "user_id": str(ident[i])}
        for i in range(n)
    ]


class ProposeDictIdentifierFieldFinalRefitTest(unittest.TestCase):
    def test_fully_unique_identifier_column_does_not_crash_final_refit(self):
        # Note: this deliberately does not assert evidence["certificate"]["status"] == "succeeded".
        # certify() (mixle/inference/planning.py's verify_estimation_conditions) indexes rows
        # positionally (row[0], row[1], ...) and raises KeyError on dict rows regardless of any
        # identifier field -- a separate, pre-existing gap unrelated to T3-01/_refresh_frozen_
        # identifier_leaves (reproduces identically on dict data with no identifier field at all,
        # both before and after this fix). What T3-01 fixes, and what matters here, is that the
        # final refit itself no longer raises.
        m = mixle.propose(_dict_records_with_unique_identifier(), fit=True)
        self.assertIsNotNone(m.fitted)
        self.assertEqual(type(m.spec).__name__, "RecordEstimator")
        self.assertTrue(
            any("held-out verification score" in n and "unseen in training" in n for n in m.notes),
            f"expected T2-01's excluded-field disclosure in notes; got {m.notes}",
        )

    def test_refit_scores_every_full_data_row_finitely(self):
        records = _dict_records_with_unique_identifier()
        m = mixle.propose(records, fit=True)
        scores = [m.fitted.log_density(r) for r in records]
        self.assertTrue(
            all(np.isfinite(s) for s in scores),
            f"expected every row of the data the refit trained on to score finitely; got {scores}",
        )

    def test_genuinely_novel_identifier_still_scores_negative_infinity(self):
        # The fix widens the frozen leaf's support to the full FIT data, not to "anything at all" --
        # a value that never appeared anywhere the model was fit from must still score -inf, exactly
        # the documented finite-support convention every other automatically fitted categorical field
        # gets (mixle/utils/automatic/factories.py's _get_identifier_estimator).
        records = _dict_records_with_unique_identifier()
        m = mixle.propose(records, fit=True)
        novel_row = {"amount": 50.0, "plan": "basic", "active": True, "user_id": "id_never_seen_anywhere"}
        self.assertFalse(np.isfinite(m.fitted.log_density(novel_row)))

    def test_clean_dict_data_without_a_frozen_field_is_unaffected(self):
        # No Ignored/identifier field at all: the refresh must never trigger, and the refit must
        # succeed exactly as it always did (this shape already worked before T3-01's fix; it must
        # keep working, unchanged, after it).
        rng = np.random.RandomState(0)
        n = 300
        amount = rng.normal(50, 10, size=n)
        plan = rng.choice(["basic", "pro", "enterprise"], size=n)
        rows = [{"amount": float(amount[i]), "plan": str(plan[i])} for i in range(n)]

        m = mixle.propose(rows, fit=True)

        self.assertIsNotNone(m.fitted)
        self.assertFalse(any("unseen in training" in n for n in m.notes))


if __name__ == "__main__":
    unittest.main()
