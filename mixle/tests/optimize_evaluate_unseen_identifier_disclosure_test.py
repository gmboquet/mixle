"""Pins T2-01: optimize()'s docstring must disclose identifier-freezing, and Model.evaluate() must
warn (not stay silent) when a -inf held-out score comes from it.

optimize()'s estimator=None auto-inference (mixle.utils.automatic.get_estimator) freezes an
identifier-like/high-cardinality field into IgnoredDistribution(CategoricalDistribution(...,
default_value=0.0)), which scores an unseen label at -inf -- this policy is intentional and already
correctly disclosed in DatumNode.get_estimator's internal docstring, but optimize()'s own public
docstring never mentioned it, unlike propose()'s Model.explain()/notes, which discloses the
identical situation. Separately, Model.evaluate() explicitly raises ValueError on NaN/+inf output
but had no isneginf check, so this exact -inf passed straight into mean_log_density/total_log_density
with zero signal.
"""

import unittest
import warnings

import numpy as np

from mixle.inference import optimize
from mixle.lifecycle import Model, propose


def _records(n, seed, fresh_ids=False, offset=0):
    rng = np.random.RandomState(seed)
    return [
        {
            "amount": float(rng.normal(10.0, 2.0)),
            "plan": str(rng.choice(["basic", "pro", "enterprise"])),
            "user_id": f"{'fresh_' if fresh_ids else ''}user_{offset + i}",
        }
        for i in range(n)
    ]


class OptimizeDocstringDisclosesIdentifierFreezingTest(unittest.TestCase):
    def test_docstring_mentions_identifier_freezing_and_unseen_label_score(self):
        doc = optimize.__doc__
        self.assertIn("Identifier-like", doc)
        self.assertIn("IgnoredDistribution", doc)
        self.assertIn("-inf", doc)


class OptimizeAutoInferenceFreezesIdentifierColumnTest(unittest.TestCase):
    def test_unseen_identifier_scores_neg_inf(self):
        # structure="off" isolates the identifier-freezing behavior from the unrelated
        # dependency-structure-search path (auto structure search on this schema hits a
        # separate, already-tracked bug in the Bayesian-network scorer).
        rows = _records(60, seed=0)
        model = optimize(rows, structure="off", out=None)
        seen = rows[0]
        unseen = dict(seen, user_id="totally_unseen_user_xyz")
        ld_seen = model.log_density(seen)
        ld_unseen = model.log_density(unseen)
        self.assertTrue(np.isfinite(ld_seen))
        self.assertEqual(ld_unseen, float("-inf"))


class ModelEvaluateWarnsOnNegInfTest(unittest.TestCase):
    def test_propose_fit_then_evaluate_on_fresh_ids_warns(self):
        m = propose(_records(60, seed=0), fit=True, seed=0)
        held_out = _records(20, seed=1, fresh_ids=True, offset=1000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = m.evaluate(held_out)
        self.assertEqual(result["total_log_density"], float("-inf"))
        matches = [w for w in caught if issubclass(w.category, UserWarning) and "evaluate()" in str(w.message)]
        self.assertEqual(len(matches), 1, f"expected exactly one evaluate() UserWarning; got {caught}")
        self.assertIn("-inf", str(matches[0].message))
        self.assertIn("unseen", str(matches[0].message))

    def test_ordinary_finite_data_evaluates_without_warning(self):
        # Regression guard: the new -inf check must not fire (or otherwise change output) for
        # ordinary, non-degenerate held-out data with no unseen identifier labels.
        m = propose(_records(60, seed=0), fit=True, seed=0)
        held_out = _records(20, seed=1)  # same user_id pool as training, no fresh ids
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = m.evaluate(held_out)
        self.assertTrue(np.isfinite(result["total_log_density"]))
        self.assertEqual(caught, [])

    def test_neg_inf_warns_but_nan_and_pos_inf_still_raise(self):
        class _Scorer:
            def __init__(self, scores):
                self._scores = scores

            def dist_to_encoder(self):
                class _Encoder:
                    def seq_encode(self, rows):
                        return rows

                return _Encoder()

            def seq_log_density(self, encoded):
                return self._scores

        for scores in ([1.0, float("nan"), 2.0], [1.0, float("inf"), 2.0]):
            model = Model()
            model.fitted = _Scorer(scores)
            with self.assertRaises(ValueError):
                model.evaluate([1, 2, 3])

        model = Model()
        model.fitted = _Scorer([1.0, float("-inf"), 2.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.evaluate([1, 2, 3])
        self.assertEqual(result["total_log_density"], float("-inf"))
        self.assertEqual(len(caught), 1)


if __name__ == "__main__":
    unittest.main()
