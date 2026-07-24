"""Constructor-time probability-range validation (0.8.0 audit follow-up).

PR #433 (worklist S-3) rejected out-of-range probabilities at construction for
``CategoricalDistribution`` and ``MixtureDistribution``'s component/weight length match, on the
principle that an invalid "probability" must fail at the constructor rather than silently
propagate into ``log_density()`` as ``nan`` or an out-of-[0,1] density. The fix never reached
these structurally identical siblings, so each accepted invalid input and produced a silently
wrong answer instead of a clear error. Grouped here because they are one bug class, not four.

A later, independent review found the same bug class still open in ``CategoricalDistribution``
itself -- not a sibling PR #433 missed, but a gap in that fix's own coverage: ``pmap`` values were
checked for non-negative but never for finite, so a NaN probability constructed silently and
propagated into ``density()``/``log_density()`` as NaN. ``CategoricalValidationTestCase`` covers
that, plus the pre-existing negative-probability rejection, which had no direct regression test of
its own despite being live since PR #433.

The same review also flagged that ``pmap`` is never checked to sum to 1, so a map with the wrong
total mass constructs silently and ``density()`` -- which normalizes on the assumption
``sum(pmap.values()) == 1`` -- returns values that don't sum to 1 over the support. That half was
investigated and deliberately NOT turned into a constructor-time rejection (or a silent
renormalization): mixle/tests/quantized_index_test.py and fused_em_mixtures_test.py/
hmixture_engine_test.py construct a partial pmap (e.g. ``{2: 0.5}``) purely as a cheap object to
call ``.estimator()``/``.quantized_index()`` on, never inspecting the values, and
mixle/tests/sparse_mixture_test.py constructs ``pmap={"a": 0.005, "b": 0.005}`` with
``default_value=0.99`` specifically so ``default_value`` dominates every in-pmap probability --
a sum-to-1 rejection breaks the first group, and a renormalization would rescale "a"/"b" up and
defeat the second group's whole premise. ``test_probabilities_not_summing_to_one_construct_by_design``
pins this as an intentional (if imperfect) constraint, not an oversight, so a future change doesn't
silently re-break those call sites.

A second-wave finding on the same review flagged ``CategoricalDistribution``'s ``default_value``
(the density assigned to any label outside ``pmap``) as accepting values outside a valid
probability law -- negative, NaN, or greater than 1. Before the fix, ``self.default_value`` was
silently clamped into ``[0, 1]``, but ``log_default_value``/``log1p_default_value`` were computed
from the raw, unclamped ``default_value`` argument: an out-of-range input desynced the two, so
``density()`` (built from ``self.default_value``) and ``log_density()`` (built from
``log1p_default_value``) disagreed with each other, and a NaN ``default_value`` made
``log_density()`` return ``nan`` for every label *inside* ``pmap`` too, not just the default
branch. This is the same class/constructor as the ``pmap`` finding above (``CategoricalDistribution.
__init__``), just a different parameter -- not a separate bug in a sibling class. No call site
anywhere in the codebase passes ``default_value`` outside ``[0, 1]`` (the widest is 0.99 in
``sparse_mixture_test.py``), so the rejection has no legitimate counterexample to preserve, unlike
the ``pmap`` sum-to-1 case above. ``IntegerCategoricalDistribution`` has no ``default_value``
concept at all (no out-of-vocabulary fallback), so it is not affected by this finding.
"""

import math
import unittest

import numpy as np

from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution
from mixle.stats.univariate.discrete.integer_uniform_spike import IntegerUniformSpikeDistribution


class MixtureWeightValidationTestCase(unittest.TestCase):
    def test_negative_weight_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)], [-0.5, 1.5])

    def test_valid_weights_still_construct(self):
        m = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
        self.assertTrue(np.isfinite(m.log_density(0.5)))


class CategoricalValidationTestCase(unittest.TestCase):
    def test_nan_probability_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            CategoricalDistribution({"a": float("nan"), "b": 0.5})

    def test_negative_probability_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            CategoricalDistribution({"a": -0.5, "b": 1.5})

    def test_probabilities_not_summing_to_one_construct_by_design(self):
        # See this module's docstring: a hard rejection here would break existing call sites
        # (quantized_index_test.py, fused_em_mixtures_test.py, hmixture_engine_test.py,
        # sparse_mixture_test.py) that rely on constructing an intentionally non-normalized pmap.
        # This pins that as deliberate so a future change doesn't silently re-break them.
        over = CategoricalDistribution({"a": 0.3, "b": 0.3, "c": 0.3, "d": 0.3})  # sums to 1.2
        under = CategoricalDistribution({"a": 0.2, "b": 0.2})  # sums to 0.4
        self.assertAlmostEqual(sum(over.pmap.values()), 1.2)
        self.assertAlmostEqual(sum(under.pmap.values()), 0.4)

    def test_valid_probabilities_still_construct(self):
        d = CategoricalDistribution({"a": 0.3, "b": 0.7})
        self.assertTrue(np.isfinite(d.log_density("a")))

    def test_probabilities_summing_to_one_within_float_tolerance_still_construct(self):
        # 1/3 + 1/3 + 1/3 == 0.9999999999999999 in float64, not exactly 1.0.
        d = CategoricalDistribution({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
        self.assertTrue(np.isfinite(d.log_density("a")))

    def test_nan_default_value_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            CategoricalDistribution({"a": 0.6, "b": 0.4}, default_value=float("nan"))

    def test_negative_default_value_rejected_at_construction(self):
        # assertRaisesRegex, not just assertRaises: a negative default_value already reached
        # math.log(default_value) unguarded and raised ValueError("math domain error") -- a
        # ValueError, but an opaque one that never names default_value as the culprit. The
        # message must come from the actual validation, not that incidental crash.
        with self.assertRaisesRegex(ValueError, "default_value"):
            CategoricalDistribution({"a": 0.6, "b": 0.4}, default_value=-0.5)

    def test_default_value_over_one_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            CategoricalDistribution({"a": 0.6, "b": 0.4}, default_value=2.0)

    def test_infinite_default_value_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            CategoricalDistribution({"a": 0.6, "b": 0.4}, default_value=float("inf"))

    def test_default_value_boundary_values_still_construct(self):
        zero = CategoricalDistribution({"a": 0.6, "b": 0.4}, default_value=0.0)
        one = CategoricalDistribution({"a": 0.6, "b": 0.4}, default_value=1.0)
        self.assertTrue(np.isfinite(zero.log_density("a")))
        self.assertTrue(np.isfinite(one.log_density("a")))

    def test_default_value_consistent_between_density_and_log_density(self):
        # Regression for the desync bug described in this module's docstring: density() and
        # log_density() must agree for every default_value actually accepted, including the
        # default_value-dominant configuration sparse_mixture_test.py relies on.
        d = CategoricalDistribution({"a": 0.005, "b": 0.005}, default_value=0.99)
        outside = d.density("not_in_pmap")
        self.assertAlmostEqual(math.log(outside), d.log_density("not_in_pmap"), places=12)


class IntegerCategoricalValidationTestCase(unittest.TestCase):
    def test_negative_or_over_one_probability_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            IntegerCategoricalDistribution(min_val=0, p_vec=[-0.5, 1.5])

    def test_valid_probabilities_still_construct(self):
        d = IntegerCategoricalDistribution(min_val=0, p_vec=[0.3, 0.7])
        self.assertTrue(np.isfinite(d.log_density(0)))


class IntegerUniformSpikeValidationTestCase(unittest.TestCase):
    def test_out_of_range_p_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            IntegerUniformSpikeDistribution(k=0, num_vals=3, p=1.5, min_val=0)
        with self.assertRaises(ValueError):
            IntegerUniformSpikeDistribution(k=0, num_vals=3, p=-0.1, min_val=0)

    def test_valid_p_still_constructs(self):
        d = IntegerUniformSpikeDistribution(k=0, num_vals=3, p=0.6, min_val=0)
        self.assertTrue(np.isfinite(d.log_density(0)))


if __name__ == "__main__":
    unittest.main()
