"""``mixle.propose(data, fit=True)``'s final full-data refit step used to crash whenever the winning
candidate carried a frozen (``Ignored``) identifier-like/high-cardinality field: that field's frozen
``CategoricalDistribution`` support only ever covered the TRAINING split (STAT-RR18-01's train/held-out
separation happens before any candidate is even built -- see ``propose()``'s docstring), and a
fully-unique-per-row identifier column guarantees every held-out row's value is absent from it.
``IgnoredEstimator``'s accumulator never re-estimates its child (mixle/stats/combinator/ignored.py), so
the refit's initial model -- and every later EM iteration, since nothing about this field ever changes
-- scored those rows at ``-inf`` forever, and mixle/inference/estimation.py's ``_em_loop`` raised
"EM did not produce a finite objective from its non-finite initial model" on data the frontier had just
finished verifying.

Independent of T2-01 (see ``propose_identifier_field_verification_test.py``), which only fixed the
verification-SCORING aggregation inside the candidate loop -- this crash reproduces identically via a
direct ``optimize()`` call, bypassing ``propose()`` entirely, on code with T2-01 already applied.
``_refresh_frozen_identifier_leaves`` (mixle/lifecycle.py) rebuilds each frozen leaf's support from the
FULL dataset right before the final refit, closing the gap without reopening the STAT-RR18-01 leak (a
frozen leaf's support never participates in candidate ranking -- T2-01's own rescue already excludes it
from every candidate's held-out score) or changing the leaf's finite-support semantics (a value that
never appears anywhere in the fit data still scores ``-inf``, same as every other automatically fitted
categorical field).
"""

import unittest

import numpy as np

import mixle


def _records_with_unique_identifier(n=300, seed=0):
    rng = np.random.RandomState(seed)
    amount = rng.normal(50, 10, size=n)
    plan = rng.choice(["basic", "pro", "enterprise"], size=n)
    active = rng.choice([True, False], size=n)
    ident = np.array([f"id_{i}" for i in range(n)])
    rng.shuffle(ident)
    return [(float(amount[i]), str(plan[i]), bool(active[i]), str(ident[i])) for i in range(n)]


class ProposeIdentifierFieldFinalRefitTest(unittest.TestCase):
    def test_fully_unique_identifier_column_does_not_crash_final_refit(self):
        m = mixle.propose(_records_with_unique_identifier(), fit=True)
        self.assertIsNotNone(m.fitted)
        self.assertEqual(m.evidence.get("certificate", {}).get("status"), "succeeded")
        # T2-01's exclusion still fires for verification scoring (unaffected by this fix, which acts
        # only on the later full-data refit) -- both repairs are visible together on the same data.
        self.assertTrue(
            any("held-out verification score" in n and "unseen in training" in n for n in m.notes),
            f"expected T2-01's excluded-field disclosure in notes; got {m.notes}",
        )

    def test_refit_scores_every_full_data_row_finitely(self):
        records = _records_with_unique_identifier()
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
        records = _records_with_unique_identifier()
        m = mixle.propose(records, fit=True)
        novel_row = (50.0, "basic", True, "id_never_seen_anywhere")
        self.assertFalse(np.isfinite(m.fitted.log_density(novel_row)))

    def test_an_identifier_field_alongside_a_multi_leaf_sequence_field_does_not_crash(self):
        # An adversarial review of the fix that added `top_level_field_paths` (T2-01, third
        # occurrence) found this call site was left passing the OLD leaf-level `field_paths` --
        # correct only when every top-level field decomposes into exactly one leaf. A sequence
        # field decomposes into TWO leaves (its element distribution and its length distribution),
        # so `len(field_paths) != len(model.spec's top-level children)` the moment one is present
        # alongside an identifier field, and the length check inside
        # `_refresh_frozen_identifier_leaves` always failed, silently no-opping and reopening the
        # exact crash this whole file exists to close.
        rng = np.random.RandomState(0)
        n = 300
        amount = rng.normal(50, 10, size=n)
        plan = rng.choice(["basic", "pro", "enterprise"], size=n)
        ident = np.array([f"id_{i}" for i in range(n)])
        rng.shuffle(ident)
        # A short, fixed vocabulary shared by every row avoids an unrelated, separate limitation
        # (a fixed-support element estimator refusing an out-of-range value at refit time) --
        # this test targets the leaf-count mismatch alone, not that different defect class.
        vocab = ["a", "b", "c"]
        history = [[str(rng.choice(vocab)) for _ in range(int(rng.poisson(3)) + 1)] for _ in range(n)]
        rows = [(float(amount[i]), str(plan[i]), str(ident[i]), history[i]) for i in range(n)]

        m = mixle.propose(rows, fit=True)

        self.assertIsNotNone(m.fitted)
        self.assertEqual(m.evidence.get("certificate", {}).get("status"), "succeeded")
        scores = [m.fitted.log_density(r) for r in rows]
        self.assertTrue(
            all(np.isfinite(s) for s in scores),
            f"expected every row of the data the refit trained on to score finitely; got {scores}",
        )

    def test_clean_data_without_a_frozen_field_is_unaffected(self):
        # No Ignored/identifier field at all: the refresh must never trigger, and the refit must
        # succeed exactly as it always did.
        rng = np.random.RandomState(0)
        n = 300
        amount = rng.normal(50, 10, size=n)
        plan = rng.choice(["basic", "pro", "enterprise"], size=n)
        rows = [(float(amount[i]), str(plan[i])) for i in range(n)]

        m = mixle.propose(rows, fit=True)

        self.assertIsNotNone(m.fitted)
        self.assertEqual(m.evidence.get("certificate", {}).get("status"), "succeeded")
        self.assertFalse(any("unseen in training" in n for n in m.notes))


if __name__ == "__main__":
    unittest.main()
