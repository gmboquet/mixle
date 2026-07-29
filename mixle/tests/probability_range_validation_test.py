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

A third-wave finding reopened ``MixtureDistribution.__init__``'s own weight check
(``MixtureWeightValidationTestCase`` above already covered negative weights, via PR #433, and the
length-mismatch guard). Two gaps: (1) the negative-weight check had the exact same blind spot as
``pmap``/``p_vec`` -- ``np.any(self.w < 0.0)`` lets a NaN entry through silently (``nan < 0.0`` is
always ``False``), reaching ``log_density()``/sampling as ``nan``, now fixed the same way. (2)
unlike ``pmap``/``p_vec`` (deliberately "stored as given", per the sum-to-1 discussion above),
``MixtureDistribution``'s own docstring commits to ``w`` being interpreted as simplex weights that
"should sum to one" -- and nothing enforced that. A weight vector summing to, say, 0.3 constructed
silently and produced a mixture whose density integrates to 0.3, not 1: a silently invalid
probability model, not a deliberately-permissive one. A codebase-wide AST scan of every
``MixtureDistribution(...)`` call site with a literal weight argument (547 call sites) found none
relying on a non-1 sum, so -- unlike the ``pmap`` case -- this is enforced as a hard constructor
rejection. The tolerance (``rtol=1e-5, atol=1e-8``) matches the established simplex-sum-to-one
convention already used for the same kind of check elsewhere (``SymmetricDirichletDistribution``,
``DictDirichletDistribution``), not ``dirichlet.py``'s tighter ``1e-10``/``1e-12``: real fitted
mixture weights are not guaranteed float64 precision end to end (the gradient-fit path's torch
softmax renormalization was measured landing ~6e-8 from 1.0 under ``precision="float32"``, outside
a naive ``1e-10``/``1e-12`` bound but comfortably inside this one).

A fourth-wave finding: the pmap-need-not-sum-to-1 permissiveness described two paragraphs above
has a downstream inconsistency in ``CategoricalSampler``, which turns ``pmap`` into parallel
``levels``/``probs`` arrays and used to pass ``probs`` straight into ``RandomState.choice(...,
p=probs)`` -- a numpy API that requires ``p`` to sum to exactly 1 and raises ``ValueError:
probabilities do not sum to 1`` otherwise. So a pmap that legitimately doesn't sum to 1
(deliberately supported, per above -- e.g. ``sparse_mixture_test.py``'s
``default_value``-dominant construction) constructed without error but crashed the first time
anyone actually sampled from it. ``density()``/``log_density()`` already tolerate a
non-1-summing ``pmap`` by dividing by ``(1 + default_value)``; ``CategoricalSampler`` now
matches by normalizing ``probs`` by their own sum before sampling, so it draws from the relative
proportions of ``pmap``'s registered labels regardless of whether they happen to sum to 1.
``CategoricalSamplerValidationTestCase`` covers this, plus the sampler's own new rejection of an
all-zero-weight (or empty) ``pmap`` at construction time, since there is then no relative
proportion left to sample from and ``RandomState.choice`` cannot sample from an all-zero-weight
distribution either.

A fifth-wave finding, in a different module entirely: ``IntegerChowLiuTreeDistribution.__init__``
(``mixle/stats/trees/integer_chow_liu_tree.py``) accepted its per-feature ``conditional_log_densities``
tables with no check that they exponentiate to an actual probability distribution. A reproduced model
built from a table with raw "probabilities" ``[1.04, 1.04]`` (summing to 2.08, not 1.0) constructed
without error; ``log_density()``/``seq_log_density()`` kept returning finite (just meaningless) scores,
and the sampler crashed with numpy's "probabilities do not sum to 1", deep inside
``np.random.choice``, on the first call. Unlike ``CategoricalDistribution.pmap`` /
``IntegerCategoricalDistribution.p_vec``, this class's own factorization contract (a tree of
``P(root) * prod P(child | parent)`` terms) requires every table to be a genuine distribution for
``log_density()`` to mean anything, so -- like ``MixtureDistribution.w`` -- this is a hard rejection: a
1-d root table must sum to 1.0 once exponentiated, and a 2-d conditional table (indexed
``table[parent_val, child_val]``) must sum to 1.0 along each row. One legitimate exception: a
conditional row is also accepted summing to ~0.0 (every entry ``-inf``), because
``IntegerChowLiuTreeEstimator.estimate()`` itself produces exactly that for a feature whose realized
value range is narrower than the tree's shared ``num_states`` -- confirmed directly against the
estimator (not just reasoned about) in
``IntegerChowLiuTreeValidationTestCase.test_unreachable_parent_state_row_still_constructs`` below; a
plain "every row sums to 1" rule would have turned that legitimate estimator output into a
constructor-time rejection.
"""

import math
import unittest

import numpy as np

from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.trees.integer_chow_liu_tree import IntegerChowLiuTreeDistribution, IntegerChowLiuTreeEstimator
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution
from mixle.stats.univariate.discrete.integer_uniform_spike import IntegerUniformSpikeDistribution


class MixtureWeightValidationTestCase(unittest.TestCase):
    def test_negative_weight_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)], [-0.5, 1.5])

    def test_nan_weight_rejected_at_construction(self):
        # `nan < 0.0` is always False, so a NaN entry alone (no negative entry alongside it) used to
        # pass the old check straight through and reach log_density()/sampling as nan (matches the
        # same gap just fixed in CategoricalDistribution.pmap / IntegerCategoricalDistribution.p_vec).
        with self.assertRaises(ValueError):
            MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)], [float("nan"), 0.5])

    def test_infinite_weight_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)], [float("inf"), 0.5])

    def test_weights_not_summing_to_one_rejected_at_construction(self):
        # Unlike CategoricalDistribution.pmap / IntegerCategoricalDistribution.p_vec (see this
        # module's docstring), MixtureDistribution's own docstring commits to `w` summing to one, and
        # no existing call site relies on it not doing so -- so this is a hard rejection, not a
        # documented-permissive gap. Regression pin for the actual pre-fix failure mode: weights
        # summing to 0.3 used to construct without error and the density silently integrated to 0.3
        # instead of 1.0 (numerically confirmed via seq_log_density over a fine grid during triage);
        # assertRaisesRegex (not just assertRaises) pins that the message names the real cause.
        with self.assertRaisesRegex(ValueError, "sum to one"):
            MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 1.0)], [0.1, 0.2])

    def test_weights_summing_to_over_one_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 1.0)], [0.7, 0.7])

    def test_valid_weights_still_construct(self):
        m = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])
        self.assertTrue(np.isfinite(m.log_density(0.5)))

    def test_weights_summing_to_one_within_float_tolerance_still_construct(self):
        # 1/3 + 1/3 + 1/3 == 0.9999999999999999 in float64, not exactly 1.0.
        m = MixtureDistribution(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0), GaussianDistribution(2.0, 1.0)],
            [1 / 3, 1 / 3, 1 / 3],
        )
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
        # Constructing is legal; claiming to be a proper law is not. The derived flag reports the
        # truth so consumers that need a normalized law can check instead of assuming.
        self.assertFalse(over.is_normalized_probability)
        self.assertFalse(under.is_normalized_probability)

    def test_empty_support_constructs_by_design(self):
        # Support-enumeration code (truncation_bound_test.py) builds the empty-support categorical
        # to assert a zero-item support is vacuously exhausted; rejecting it here broke that.
        empty = CategoricalDistribution({})
        self.assertEqual(empty.pmap, {})
        self.assertFalse(empty.is_normalized_probability)

    def test_normalized_pmap_is_reported_as_a_probability_law(self):
        self.assertTrue(CategoricalDistribution({"a": 0.25, "b": 0.75}).is_normalized_probability)
        # An explicit scoring-only declaration overrides the numbers: it is a likelihood factor.
        self.assertFalse(CategoricalDistribution({"a": 0.25, "b": 0.75}, scoring_only=True).is_normalized_probability)

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


class CategoricalSamplerValidationTestCase(unittest.TestCase):
    def test_sampler_from_non_normalized_pmap_does_not_crash(self):
        # Regression for the fourth-wave finding in this module's docstring: a pmap that
        # legitimately doesn't sum to 1 (test_probabilities_not_summing_to_one_construct_by_design
        # above pins that this constructs fine) used to crash at RandomState.choice(p=...) time
        # instead, since CategoricalSampler passed pmap's raw values through unnormalized.
        d = CategoricalDistribution({"a": 0.1, "x": 0.1}, default_value=0.3)  # pmap sums to 0.2
        samples = d.sampler(seed=0).sample(2000)
        self.assertEqual(len(samples), 2000)
        # Sampling can only ever draw pmap's registered levels, never a synthesized "default" label
        # (the sampler has no way to generatively produce an out-of-vocabulary label).
        self.assertTrue(set(samples) <= {"a", "x"})

    def test_sampler_from_non_normalized_pmap_preserves_relative_proportions(self):
        # "b" is 3x as likely as "a" (relative proportion 0.3:0.1) even though pmap sums to 0.4,
        # not 1 -- normalizing must preserve relative proportions, not just avoid crashing.
        d = CategoricalDistribution({"a": 0.1, "b": 0.3})
        samples = d.sampler(seed=0).sample(20000)
        ratio = samples.count("b") / samples.count("a")
        self.assertAlmostEqual(ratio, 3.0, delta=0.3)  # loose tolerance: large-sample frequency check

    def test_sampler_from_default_value_dominant_pmap_does_not_crash(self):
        # The exact construction sparse_mixture_test.py relies on for a different reason (making
        # default_value dominate every in-pmap probability): pmap sums to 0.01, far from 1.
        d = CategoricalDistribution({"a": 0.005, "b": 0.005}, default_value=0.99)
        samples = d.sampler(seed=0).sample(1000)
        self.assertTrue(set(samples) <= {"a", "b"})

    def test_sampler_rejects_explicitly_scoring_only_models(self):
        # The one refusal that survives: an *explicit* scoring_only=True declaration marks the
        # object a likelihood factor rather than a generative law, so there is nothing to draw
        # from. A merely non-1-summing pmap is not such a declaration (the tests above).
        for distribution in (
            CategoricalDistribution({"a": 0.1, "x": 0.1}, default_value=0.3, scoring_only=True),
            CategoricalDistribution({"a": 0.1, "b": 0.3}, scoring_only=True),
            CategoricalDistribution({"a": 0.25, "b": 0.75}, scoring_only=True),
        ):
            with self.subTest(distribution=repr(distribution)), self.assertRaisesRegex(ValueError, "normalized"):
                distribution.sampler(seed=0)

    def test_sampler_from_normalized_pmap_still_matches_relative_proportions(self):
        # Sanity check that normalizing by pmap's own sum leaves the common already-normalized
        # case (pmap sums to 1) numerically unaffected.
        d = CategoricalDistribution({"a": 0.25, "b": 0.75})
        samples = d.sampler(seed=0).sample(20000)
        freq_b = samples.count("b") / len(samples)
        self.assertAlmostEqual(freq_b, 0.75, delta=0.02)

    def test_sampler_rejects_all_zero_weight_pmap(self):
        # np.random.RandomState.choice cannot sample from an all-zero-weight distribution either;
        # this must fail clearly at sampler construction, not as a confusing 0/0 or numpy error.
        d = CategoricalDistribution({"a": 0.0, "b": 0.0})
        with self.assertRaisesRegex(ValueError, "positive probability"):
            d.sampler(seed=0)

    def test_sampler_rejects_empty_pmap(self):
        d = CategoricalDistribution({})
        with self.assertRaisesRegex(ValueError, "positive probability"):
            d.sampler(seed=0)


class IntegerCategoricalValidationTestCase(unittest.TestCase):
    def test_negative_or_over_one_probability_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            IntegerCategoricalDistribution(min_val=0, p_vec=[-0.5, 1.5])

    def test_nan_probability_rejected_at_construction(self):
        # `nan < 0.0` is always False, so a NaN entry alone (no negative entry alongside it) used to
        # pass the old check straight through and reach log_density() as nan.
        with self.assertRaises(ValueError):
            IntegerCategoricalDistribution(min_val=0, p_vec=[float("nan"), 0.5])

    def test_valid_probabilities_still_construct(self):
        d = IntegerCategoricalDistribution(min_val=0, p_vec=[0.3, 0.7])
        self.assertTrue(np.isfinite(d.log_density(0)))

    def test_unnormalized_weights_over_one_still_construct(self):
        # matches CategoricalDistribution.pmap: entries need not sum to 1, and an individual entry may
        # exceed 1.0 (an unnormalized weight vector), so only finiteness and non-negativity are rejected.
        d = IntegerCategoricalDistribution(min_val=0, p_vec=[1.5, 2.0])
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


class IntegerChowLiuTreeValidationTestCase(unittest.TestCase):
    def test_unnormalized_root_table_rejected_at_construction(self):
        # Regression pin for the actual reported failure mode: a root (single-feature, no edges)
        # table built from raw "probabilities" [1.04, 1.04] -- summing to 2.08, not 1.0 -- used to
        # construct without error. log_density() kept returning finite (just meaningless) scores;
        # sampling was where it actually failed, deep inside np.random.choice, with "probabilities
        # do not sum to 1", far from the actual mistake.
        bad_root = np.log(np.array([1.04, 1.04]))
        self.assertAlmostEqual(float(np.exp(bad_root).sum()), 2.08, places=8)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            IntegerChowLiuTreeDistribution([None], [bad_root])

    def test_unnormalized_conditional_row_rejected_at_construction(self):
        root = np.log([0.6, 0.4])
        bad_edge = np.log(np.array([[0.5, 0.5], [0.9, 0.9]]))  # second row sums to 1.8, not 1.0
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            IntegerChowLiuTreeDistribution([None, 0], [root, bad_edge])

    def test_nan_entry_rejected_at_construction(self):
        # `nan < 0.0` (and every other ordered comparison) is always False, so a plain non-negativity
        # check alone -- the pmap/p_vec style check -- would let a NaN entry straight through; this
        # class instead requires every entry finite (or -inf) up front, matching the same gap fixed
        # for CategoricalDistribution.pmap / IntegerCategoricalDistribution.p_vec / MixtureDistribution.w
        # elsewhere in this module.
        with self.assertRaises(ValueError):
            IntegerChowLiuTreeDistribution([None], [np.array([0.0, float("nan")])])

    def test_infinite_entry_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            IntegerChowLiuTreeDistribution([None], [np.array([0.0, float("inf")])])

    def test_wrong_dimension_root_table_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            IntegerChowLiuTreeDistribution([None], [np.log([[0.5, 0.5], [0.5, 0.5]])])

    def test_wrong_dimension_conditional_table_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            IntegerChowLiuTreeDistribution([None, 0], [np.log([0.6, 0.4]), np.log([0.5, 0.5])])

    def test_valid_tables_still_construct(self):
        dist = IntegerChowLiuTreeDistribution([None, 0], [np.log([0.6, 0.4]), np.log([[0.7, 0.3], [0.2, 0.8]])])
        self.assertTrue(np.isfinite(dist.log_density([0, 0])))
        samples = dist.sampler(seed=0).sample(50)
        self.assertEqual(len(samples), 50)

    def test_tables_summing_to_one_within_float_tolerance_still_construct(self):
        # 1/3 + 1/3 + 1/3 == 0.9999999999999999 in float64, not exactly 1.0.
        root = np.log(np.array([1 / 3, 1 / 3, 1 / 3]))
        dist = IntegerChowLiuTreeDistribution([None], [root])
        self.assertTrue(np.isfinite(dist.log_density([0])))

    def test_explicit_zero_probability_state_still_constructs(self):
        # A -inf entry (an explicit, legitimate zero-probability state) must not be confused with a
        # NaN or +inf entry: this table still sums to 1.0 once exponentiated, so it must construct.
        root = np.log(np.array([0.5, 0.5, 0.0]))
        dist = IntegerChowLiuTreeDistribution([None], [root])
        self.assertEqual(dist.log_density([2]), float("-inf"))

    def test_unreachable_parent_state_row_still_constructs(self):
        # See this module's docstring: IntegerChowLiuTreeEstimator.estimate() legitimately produces a
        # conditional table with a row that sums to 0.0 (every entry -inf), not 1.0, whenever a parent
        # feature's realized value range is narrower than the tree's shared num_states. Confirmed
        # directly against the estimator (not synthesized by hand), since this is exactly the
        # scenario a naive "every row must sum to 1" check would wrongly reject: feature 0 and 1 here
        # only ever take values 0/1, but feature 2 rarely reaches up to 4, so num_states is forced to
        # 5 for every feature even though 0/1 are the only states feature 0 (the learned parent of the
        # other two) ever actually takes.
        rng = np.random.RandomState(0)
        data = []
        for _ in range(500):
            f0 = int(rng.randint(0, 2))
            f1 = f0 if rng.rand() < 0.9 else 1 - f0
            f2 = int(rng.randint(0, 5)) if rng.rand() < 0.02 else f0
            data.append([f0, f1, f2])

        estimator = IntegerChowLiuTreeEstimator()
        acc = estimator.accumulator_factory().make()
        for row in data:
            acc.update(row, 1.0, None)
        model = estimator.estimate(len(data), acc.value())  # must not raise

        row_sums = [
            np.exp(table).sum(axis=1) for table in model.conditional_log_densities if np.asarray(table).ndim == 2
        ]
        self.assertTrue(any(np.any(np.isclose(rs, 0.0, atol=1e-8)) for rs in row_sums))

        samples = model.sampler(seed=1).sample(200)
        self.assertEqual(len(samples), 200)


if __name__ == "__main__":
    unittest.main()
