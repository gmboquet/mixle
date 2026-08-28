"""An adversarial re-review of T2-01's fix found a real misdiagnosis in
``mixle.lifecycle._unseen_label_rescue``: it excused a field from held-out verification as an
"unseen categorical label" whenever the field's -inf score, after unwrapping OptionalDistribution/
IgnoredDistribution wrappers, bottomed out at a plain CategoricalDistribution with
``default_value=0.0`` -- without ever checking that the -inf actually came from the leaf's
unseen-label path. ``OptionalDistribution.seq_log_density`` independently scores a genuinely
MISSING held-out value at ``self.log_p``, which is also ``-inf`` whenever the wrapper's fitted
``p == 0.0`` (no missing rows in the training split, an entirely ordinary outcome for auto-fit
missingness models). A field fit that way and then handed a real missing held-out value was
silently excused under the wrong diagnosis, masking a genuine missing-value generalization
failure rather than the documented, benign unseen-label case this rescue exists for.
"""

from __future__ import annotations

import unittest

import numpy as np

from mixle.lifecycle import _unseen_label_rescue
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.combinator.optional import OptionalDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


def _composite_and_encoding(cat_field, other_rows, held_out):
    fitted = CompositeDistribution([cat_field, GaussianDistribution(0.0, 1.0)])
    enc = fitted.dist_to_encoder().seq_encode(held_out)
    scores = np.asarray(fitted.seq_log_density(enc), dtype=np.float64)
    return fitted, enc, scores


class UnseenLabelRescueMissingnessAttributionTest(unittest.TestCase):
    def test_a_missing_value_under_a_zero_probability_wrapper_is_not_excused_as_unseen_label(self):
        # p=0.0: this field's Optional wrapper never saw a missing value in its training split --
        # an entirely ordinary outcome, not evidence of anything wrong -- but its log_p is -inf, so
        # a genuinely missing held-out value scores -inf from the WRAPPER, not from the categorical
        # leaf's default_value=0.0 unseen-label path.
        cat_field = OptionalDistribution(
            CategoricalDistribution({"a": 0.5, "b": 0.5}, default_value=0.0), p=0.0, missing_value=None
        )
        held_out = [(None, 0.3), ("a", 0.1), ("b", -0.2)]
        fitted, enc, scores = _composite_and_encoding(cat_field, None, held_out)

        self.assertFalse(np.isfinite(scores[0]))  # row 0 (the missing value) is -inf, as expected
        rescued = _unseen_label_rescue(fitted, enc, scores, ["field0", "field1"])
        self.assertIsNone(rescued, "a missing-value -inf under p=0.0 must not be excused as an unseen label")

    def test_a_genuinely_unseen_label_under_a_nonzero_probability_wrapper_is_still_rescued(self):
        # p=0.5: the wrapper's own log_p/log_pn are both finite, so any -inf here is unambiguously
        # the categorical leaf's own unseen-label behavior -- the case this rescue is FOR.
        cat_field = OptionalDistribution(
            CategoricalDistribution({"a": 0.5, "b": 0.5}, default_value=0.0), p=0.5, missing_value=None
        )
        held_out = [("unseen_label", 0.3), ("a", 0.1), ("b", -0.2), (None, 0.05)]
        fitted, enc, scores = _composite_and_encoding(cat_field, None, held_out)

        self.assertFalse(np.isfinite(scores[0]))  # row 0 (the unseen label) is -inf
        self.assertTrue(np.isfinite(scores[3]))  # row 3 (missing, but p=0.5 so this is finite)
        rescued = _unseen_label_rescue(fitted, enc, scores, ["field0", "field1"])
        self.assertIsNotNone(rescued)
        reduced, note = rescued
        self.assertTrue(np.isfinite(reduced).all())
        self.assertIn("unseen in training", note)

    def test_a_bare_categorical_field_with_no_optional_wrapper_is_unaffected(self):
        # No OptionalDistribution in the chain at all -- the guard this fix adds must not change
        # behavior for the case T2-01's own regression test exercises (a field with no missingness).
        cat_field = CategoricalDistribution({"a": 0.5, "b": 0.5}, default_value=0.0)
        held_out = [("unseen_label", 0.3), ("a", 0.1), ("b", -0.2)]
        fitted, enc, scores = _composite_and_encoding(cat_field, None, held_out)

        self.assertFalse(np.isfinite(scores[0]))
        rescued = _unseen_label_rescue(fitted, enc, scores, ["field0", "field1"])
        self.assertIsNotNone(rescued)
        reduced, note = rescued
        self.assertTrue(np.isfinite(reduced).all())
        self.assertIn("unseen in training", note)

    def test_p_equal_one_wrapper_also_refuses_the_rescue(self):
        # The symmetric edge: p=1.0 makes log_pn (the non-missing branch's own additive term) -inf,
        # so a present-but-scored-through-the-leaf row's -inf is equally ambiguous.
        cat_field = OptionalDistribution(
            CategoricalDistribution({"a": 0.5, "b": 0.5}, default_value=0.0), p=1.0, missing_value=None
        )
        held_out = [("a", 0.1), ("b", -0.2), (None, 0.05)]
        fitted, enc, scores = _composite_and_encoding(cat_field, None, held_out)

        self.assertFalse(np.isfinite(scores).all())
        rescued = _unseen_label_rescue(fitted, enc, scores, ["field0", "field1"])
        self.assertIsNone(rescued)


if __name__ == "__main__":
    unittest.main()
