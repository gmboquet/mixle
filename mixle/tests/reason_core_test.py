"""Regression tests for mixle.reason.core audit fixes (0.8.0 exhaustive code review).

MXR-080-0272 (marginal entropy/attribution bookkeeping), MXR-080-0273 (predict() UQ validation),
MXR-080-0274 (Latent/mechanistic/block_selector dimension and covariance validation), and
MXR-080-0275 (fold-order-dependent linear attribution) each get their own TestCase below, named
after the finding ID they cover.
"""

import itertools
import unittest

import numpy as np

from mixle.inference.belief import GaussianBelief
from mixle.reason.core import Latent, block_selector, reason
from mixle.reason.core import LinearGaussianEvidence as Evidence


class Mxr0272MarginalEntropyTest(unittest.TestCase):
    """``ReasonedAnswer.marginal()`` used to keep the FULL-STATE prior entropy and full-state
    per-source contributions after replacing the posterior with a lower-dimensional marginal, so
    querying a coordinate evidence never touched reported nonzero "information gain" -- entropy
    mixed in from a different-dimensional space, not a genuine reduction in uncertainty about that
    coordinate. The audit's own example (an isotropic 2-D unit-variance prior, evidence that reads
    only coordinate 0, querying the untouched coordinate 1) reported 1.4189 nats; the correct answer
    is exactly 0.0, since coordinate 1's marginal posterior is identical to its marginal prior.
    """

    def test_marginal_of_untouched_coordinate_reports_zero_information_gain(self):
        prior = Latent.vector(2, var=1.0)
        ev = [Evidence(H=[[1.0, 0.0]], y=[3.0], R=[[0.01]], name="probe0")]
        ans = reason(prior, ev)
        untouched = ans.marginal([1])

        # Document the magnitude of the bug this fixes: the OLD (buggy) computation -- full-state
        # prior entropy minus the marginal posterior entropy -- is exactly the audit's 1.4189 nats.
        buggy_leaked_gain = prior.entropy() - untouched.belief.entropy()
        self.assertAlmostEqual(buggy_leaked_gain, 1.4189385332046727, places=9)

        # The correct answer: coordinate 1 was never observed, so its information gain is exactly 0.
        self.assertAlmostEqual(untouched.information_gain(), 0.0, places=9)

    def test_marginal_attribution_of_untouched_coordinate_is_zero(self):
        prior = Latent.vector(2, var=1.0)
        ev = [Evidence(H=[[1.0, 0.0]], y=[3.0], R=[[0.01]], name="probe0")]
        ans = reason(prior, ev)
        attr = ans.marginal([1]).attribution()
        self.assertAlmostEqual(attr["probe0"], 0.0, places=9)

    def test_marginal_of_touched_coordinate_keeps_the_genuine_gain(self):
        # Negative control: the fix must not zero out attribution/gain across the board -- the
        # TOUCHED coordinate still reports (essentially) the full information gain, since this
        # isotropic prior's coordinates are independent and evidence only informs coordinate 0.
        prior = Latent.vector(2, var=1.0)
        ev = [Evidence(H=[[1.0, 0.0]], y=[3.0], R=[[0.01]], name="probe0")]
        ans = reason(prior, ev)
        touched = ans.marginal([0])
        self.assertGreater(touched.information_gain(), 1.0)
        self.assertAlmostEqual(touched.information_gain(), ans.information_gain(), places=9)
        self.assertAlmostEqual(touched.attribution()["probe0"], ans.attribution()["probe0"], places=9)

    def test_marginal_information_gain_matches_direct_marginal_computation(self):
        # A CORRELATED (non-isotropic) prior, so the marginal's entropy genuinely differs from a
        # naive per-coordinate decomposition -- cross-check against GaussianBelief directly rather
        # than against reason()'s own bookkeeping.
        mean = [0.0, 0.0, 0.0]
        cov = [[4.0, 1.5, 0.0], [1.5, 3.0, 0.5], [0.0, 0.5, 2.0]]
        prior = GaussianBelief(mean, cov)
        ev = [Evidence(H=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], y=[1.0, -1.0], R=np.eye(2) * 0.2, name="obs")]
        ans = reason(prior, ev)

        for idx in ([0], [1], [2], [0, 2], [1, 2]):
            expected = prior.marginal(idx).entropy() - ans.belief.marginal(idx).entropy()
            got = ans.marginal(idx).information_gain()
            self.assertAlmostEqual(got, expected, places=9, msg=f"idx={idx}")

    def test_nested_marginal_calls_compose_like_a_single_marginal(self):
        mean = [0.0, 0.0, 0.0, 0.0]
        cov = np.array(
            [
                [3.0, 0.5, 0.2, 0.0],
                [0.5, 2.0, 0.0, 0.1],
                [0.2, 0.0, 1.5, 0.3],
                [0.0, 0.1, 0.3, 1.0],
            ]
        )
        prior = GaussianBelief(mean, cov)
        ev = [Evidence(H=np.eye(4)[[0, 3]], y=[1.0, -0.5], R=np.eye(2) * 0.1, name="obs")]
        ans = reason(prior, ev)

        # marginal([2, 0]) then marginal([1]) (local index 1 of [2,0] is global coordinate 0) must
        # match marginal([0]) computed directly in one step.
        nested = ans.marginal([2, 0]).marginal([1])
        direct = ans.marginal([0])
        self.assertAlmostEqual(nested.information_gain(), direct.information_gain(), places=9)
        np.testing.assert_allclose(nested.mean, direct.mean, atol=1e-10)
        np.testing.assert_allclose(nested.cov(), direct.cov(), atol=1e-10)
        self.assertAlmostEqual(nested.attribution()["obs"], direct.attribution()["obs"], places=9)

    def test_attribution_still_sums_to_information_gain_after_marginal(self):
        # The telescoping-sum invariant (sum of per-source sequential gains == total gain) held for
        # the full-state answer before this fix and must still hold within a marginal query.
        prior = Latent.vector(3, var=5.0)
        ev = [
            Evidence(H=[[1.0, 0.0, 0.0]], y=[1.0], R=[[0.5]], name="a"),
            Evidence(H=[[0.3, 1.0, 0.0]], y=[0.5], R=[[0.7]], name="b"),
        ]
        ans = reason(prior, ev)
        sub = ans.marginal([0, 1])
        self.assertAlmostEqual(sum(sub.attribution().values()), sub.information_gain(), places=9)


class Mxr0273PredictValidationTest(unittest.TestCase):
    """``ReasonedAnswer.predict()`` used to silently reshape an ``H`` whose declared width was wrong
    whenever its total element count happened to divide evenly by the latent dimension, and accepted
    a non-square, asymmetric, non-finite, or indefinite ``R`` -- including a bare negative scalar,
    which became a negative aleatoric VARIANCE in the returned decomposition. Every case here must now
    raise a ``ValueError`` instead of returning a plausible-looking but invalid decomposition.
    """

    def setUp(self):
        prior = Latent.vector(3, var=1.0)
        ev = [Evidence(H=np.eye(3), y=[1.0, 2.0, 3.0], R=np.eye(3) * 0.1, name="full")]
        self.ans = reason(prior, ev)

    def test_wrong_width_operator_that_divides_evenly_is_rejected_not_reshaped(self):
        # shape (1, 6): declared width is wrong for this 3-dim belief, but 6 elements divide evenly
        # by 3 -- the old code silently reshaped this to (2, 3) instead of raising.
        with self.assertRaisesRegex(ValueError, "must have 3 columns"):
            self.ans.predict(H=np.ones((1, 6)), R=0.0)

    def test_negative_noise_scalar_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.ans.predict(H=[[1.0, 0.0, 0.0]], R=-5.0)

    def test_indefinite_noise_matrix_is_rejected(self):
        # eigenvalues 3 and -1 (ac=1 < b^2=4): not a covariance matrix for any interpretation.
        with self.assertRaisesRegex(ValueError, "positive semi-definite"):
            self.ans.predict(H=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], R=[[1.0, 2.0], [2.0, 1.0]])

    def test_asymmetric_noise_that_is_indefinite_once_symmetrized_is_rejected(self):
        # R itself is asymmetric ([[1,10],[0,1]]); symmetrizing (the established GaussianBelief /
        # _safe_cholesky convention) gives [[1,5],[5,1]] with eigenvalues 6 and -4 -- genuinely not PSD.
        with self.assertRaisesRegex(ValueError, "positive semi-definite"):
            self.ans.predict(H=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], R=[[1.0, 10.0], [0.0, 1.0]])

    def test_non_finite_noise_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.ans.predict(H=[[1.0, 0.0, 0.0]], R=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.ans.predict(H=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], R=[[1.0, 0.0], [0.0, float("inf")]])

    def test_non_square_noise_matrix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.ans.predict(H=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_wrong_length_vector_noise_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "length 2"):
            self.ans.predict(H=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], R=[1.0, 2.0, 3.0])

    def test_negative_entry_in_vector_noise_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.ans.predict(H=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], R=[1.0, -0.2])

    def test_valid_scalar_and_matrix_noise_still_predict_correctly(self):
        # Negative control: legitimate inputs are unaffected, and match the closed-form values from
        # reason_frontdoor_test.py::PredictionUQTest.
        prior = Latent.vector(1, var=9.0)
        ans = reason(prior, [Evidence([[1.0]], [4.0], [[1.0]], "obs")])
        post_var = float(ans.sd()[0]) ** 2
        dec = ans.predict(H=[[1.0]], R=0.25)
        self.assertEqual(dec.kind, "variance")
        self.assertAlmostEqual(float(np.reshape(dec.epistemic, -1)[0]), post_var, places=10)
        self.assertAlmostEqual(float(np.reshape(dec.aleatoric, -1)[0]), 0.25, places=10)

        # A valid SYMMETRIC matrix R is accepted and its diagonal used as the aleatoric variances.
        dec2 = self.ans.predict(H=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], R=[[0.3, 0.1], [0.1, 0.3]])
        np.testing.assert_allclose(dec2.aleatoric, [0.3, 0.3])
        np.testing.assert_allclose(dec2.total, dec2.epistemic + dec2.aleatoric)


class Mxr0274FactoryValidationTest(unittest.TestCase):
    """``Latent.vector``/``Latent.mechanistic`` silently truncated a fractional dimension or step
    count with ``int()``, ``mechanistic()`` never validated its ``x0_cov``/``process_cov`` arguments
    at all, and ``block_selector()`` accepted a negative step (silently selecting a DIFFERENT,
    in-range block via Python-style wraparound with no signal) or an out-of-range one (falling
    through to a confusing low-level numpy broadcast error instead of a clear validation message).
    """

    def test_fractional_vector_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dim must be an integer"):
            Latent.vector(2.9, var=1.0)

    def test_nonpositive_vector_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            Latent.vector(0, var=1.0)

    def test_fractional_mechanistic_steps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "steps must be an integer"):
            Latent.mechanistic(np.array([[1.0]]), steps=3.7)

    def test_mechanistic_rejects_negative_variance_x0_cov(self):
        with self.assertRaisesRegex(ValueError, "x0_cov must be positive semi-definite"):
            Latent.mechanistic(np.array([[1.0]]), steps=3, x0_cov=[[-1.0]])

    def test_mechanistic_rejects_indefinite_process_cov(self):
        # eigenvalues 3 and -1: not a covariance matrix at all.
        A = np.eye(2)
        with self.assertRaisesRegex(ValueError, "process_cov must be positive semi-definite"):
            Latent.mechanistic(A, steps=3, process_cov=[[1.0, 2.0], [2.0, 1.0]])

    def test_mechanistic_rejects_non_finite_covariance(self):
        with self.assertRaisesRegex(ValueError, "x0_cov must be finite"):
            Latent.mechanistic(np.array([[1.0]]), steps=3, x0_cov=[[float("nan")]])

    def test_mechanistic_rejects_wrong_shape_covariance(self):
        with self.assertRaisesRegex(ValueError, r"x0_cov must have shape \(1, 1\)"):
            Latent.mechanistic(np.array([[1.0]]), steps=3, x0_cov=[[1.0, 0.0], [0.0, 1.0]])

    def test_mechanistic_rejects_wrong_shape_x0_mean(self):
        with self.assertRaisesRegex(ValueError, r"x0_mean must have shape \(2,\)"):
            Latent.mechanistic(np.eye(2), steps=3, x0_mean=[1.0, 2.0, 3.0])

    def test_mechanistic_still_builds_a_valid_prior_for_good_input(self):
        # Negative control, reusing mechanistic_latent_test.py's own valid parameters.
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        P0 = np.eye(2) * 2.0
        Q = np.eye(2) * 0.1
        prior = Latent.mechanistic(A, steps=5, x0_mean=[1.0, -1.0], x0_cov=P0, process_cov=Q)
        self.assertEqual(np.size(prior.mean()), 10)
        evals = np.linalg.eigvalsh(prior.cov())
        self.assertGreaterEqual(evals.min(), -1e-9)

    def test_block_selector_rejects_negative_step_instead_of_silently_wrapping(self):
        # Pre-fix, step=-2 against n_blocks=4 silently selected block 2 (Python-style wraparound)
        # with no error at all -- confirmed by direct reproduction against the pre-fix code.
        for step in (-1, -2, -3, -4, -5):
            with self.assertRaisesRegex(ValueError, r"step must be in \[0, 4\)"):
                block_selector(step, n_blocks=4, block_dim=3)

    def test_block_selector_rejects_out_of_range_step(self):
        for step in (4, 5, 100):
            with self.assertRaisesRegex(ValueError, r"step must be in \[0, 4\)"):
                block_selector(step, n_blocks=4, block_dim=3)

    def test_block_selector_rejects_wrong_width_within(self):
        with self.assertRaisesRegex(ValueError, "within must have 3 columns"):
            block_selector(0, n_blocks=4, block_dim=3, within=[[1.0, 1.0]])

    def test_block_selector_rejects_fractional_or_nonpositive_dims(self):
        with self.assertRaisesRegex(ValueError, "block_dim must be an integer"):
            block_selector(0, n_blocks=4, block_dim=3.5)
        with self.assertRaisesRegex(ValueError, "n_blocks must be a positive integer"):
            block_selector(0, n_blocks=0, block_dim=3)

    def test_block_selector_still_selects_the_correct_block_for_valid_input(self):
        # Negative control, matching mechanistic_latent_test.py's own assertions.
        H = block_selector(2, n_blocks=4, block_dim=3)
        self.assertEqual(H.shape, (3, 12))
        self.assertTrue(np.allclose(H[:, 6:9], np.eye(3)))
        self.assertEqual(H[:, :6].sum(), 0.0)
        self.assertEqual(H[:, 9:].sum(), 0.0)


class Mxr0275AttributionOrderTest(unittest.TestCase):
    """The default (sequential) ``attribution()`` credits whichever of two REDUNDANT linear-Gaussian
    sources is folded first with substantially more of the shared information gain, even though the
    final posterior is identical either way -- module documentation presented the linear path as
    order-independent without disclosing that this specific credit split is not. ``method="shapley"``
    provides an order-invariant alternative; the sequential default is unchanged (and must stay that
    way, since reason_nonlinear_test.py's order-dependence regression test depends on it).
    """

    def _redundant_sources(self):
        prior = Latent.vector(1, var=100.0)
        eA = Evidence(H=[[1.0]], y=[5.0], R=[[1.0]], name="A")
        eB = Evidence(H=[[1.0]], y=[5.2], R=[[1.0]], name="B")
        return prior, eA, eB

    def test_sequential_attribution_is_fold_order_dependent_for_redundant_sources(self):
        prior, eA, eB = self._redundant_sources()
        ab = reason(prior, [eA, eB])
        ba = reason(prior, [eB, eA])

        # The audit's own numbers: whichever source is folded FIRST gets the larger share.
        self.assertAlmostEqual(ab.attribution()["A"], 2.30756025842063, places=9)
        self.assertAlmostEqual(ab.attribution()["B"], 0.34409219560890825, places=9)
        self.assertAlmostEqual(ba.attribution()["B"], 2.30756025842063, places=9)
        self.assertAlmostEqual(ba.attribution()["A"], 0.34409219560890825, places=9)

        # The posteriors themselves agree exactly -- only the credit split disagrees.
        np.testing.assert_allclose(ab.mean, ba.mean, atol=1e-12)
        np.testing.assert_allclose(ab.cov(), ba.cov(), atol=1e-12)

    def test_shapley_attribution_is_order_invariant_for_the_same_redundant_sources(self):
        prior, eA, eB = self._redundant_sources()
        ab = reason(prior, [eA, eB])
        ba = reason(prior, [eB, eA])

        shap_ab = ab.attribution(method="shapley")
        shap_ba = ba.attribution(method="shapley")
        # Same dict regardless of which order reason() was originally called with.
        self.assertAlmostEqual(shap_ab["A"], shap_ba["A"], places=9)
        self.assertAlmostEqual(shap_ab["B"], shap_ba["B"], places=9)

        # For N=2, the exact Shapley value is the average of "credited when first" and "credited
        # when second" -- hand-derived from the sequential numbers above.
        expected = 0.5 * (2.30756025842063 + 0.34409219560890825)
        self.assertAlmostEqual(shap_ab["A"], expected, places=9)
        self.assertAlmostEqual(shap_ab["B"], expected, places=9)

    def test_shapley_efficiency_sums_to_total_information_gain(self):
        prior, eA, eB = self._redundant_sources()
        ans = reason(prior, [eA, eB])
        shap = ans.attribution(method="shapley")
        self.assertAlmostEqual(sum(shap.values()), ans.information_gain(), places=9)

    def test_shapley_matches_sequential_for_non_redundant_orthogonal_sources(self):
        # When sources constrain DISJOINT coordinates, there is no redundancy to allocate: sequential
        # and Shapley must agree exactly, regardless of fold order.
        prior = Latent.vector(2, var=8.0)
        ev = [
            Evidence([[1.0, 0.0]], [1.0], [[0.5]], "gravity"),
            Evidence([[0.0, 1.0]], [2.0], [[2.0]], "magnetic"),
        ]
        ans = reason(prior, ev)
        seq = ans.attribution()
        shap = ans.attribution(method="shapley")
        self.assertAlmostEqual(seq["gravity"], shap["gravity"], places=9)
        self.assertAlmostEqual(seq["magnetic"], shap["magnetic"], places=9)

    def test_shapley_exact_matches_brute_force_permutation_average(self):
        prior = Latent.vector(1, var=50.0)
        sources = [
            Evidence(H=[[1.0]], y=[3.0], R=[[1.0]], name="a"),
            Evidence(H=[[1.0]], y=[3.3], R=[[1.5]], name="b"),
            Evidence(H=[[1.0]], y=[2.8], R=[[0.8]], name="c"),
        ]
        ans = reason(prior, sources)
        got = ans.attribution(method="shapley")

        names = ["a", "b", "c"]
        contrib = dict.fromkeys(names, 0.0)
        permutations = list(itertools.permutations(range(3)))
        for perm in permutations:
            belief = prior
            before = belief.entropy()
            for i in perm:
                e = sources[i]
                belief = belief.update(e.H, e.y, e.R)
                after = belief.entropy()
                contrib[names[i]] += before - after
                before = after
        expected = {k: v / len(permutations) for k, v in contrib.items()}
        for name in names:
            self.assertAlmostEqual(got[name], expected[name], places=9, msg=name)

    def test_shapley_after_marginal_query_stays_order_invariant_and_efficient(self):
        prior = Latent.vector(2, var=10.0)
        sA = Evidence(H=[[1.0, 0.0]], y=[3.0], R=[[1.0]], name="A")
        sB = Evidence(H=[[0.6, 0.4]], y=[2.0], R=[[1.0]], name="B")  # touches both coords

        sub = reason(prior, [sA, sB]).marginal([0])
        sub_reversed = reason(prior, [sB, sA]).marginal([0])

        shap = sub.attribution(method="shapley")
        shap_reversed = sub_reversed.attribution(method="shapley")
        self.assertAlmostEqual(shap["A"], shap_reversed["A"], places=9)
        self.assertAlmostEqual(shap["B"], shap_reversed["B"], places=9)
        self.assertAlmostEqual(sum(shap.values()), sub.information_gain(), places=9)

    def test_shapley_sampling_path_for_many_sources_is_deterministic_given_a_seed(self):
        # Above _SHAPLEY_EXACT_MAX_SOURCES, attribution() falls back to sampled permutations --
        # still efficient (sums to the total gain, an exact identity for ANY set of permutations,
        # sampled or not) and reproducible given a fixed rng.
        prior = Latent.vector(1, var=1000.0)
        many = [Evidence(H=[[1.0]], y=[float(i)], R=[[1.0]], name=f"s{i}") for i in range(9)]
        ans = reason(prior, many)

        first = ans.attribution(method="shapley", n_permutations=30, rng=0)
        second = ans.attribution(method="shapley", n_permutations=30, rng=0)
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(first.values()), ans.information_gain(), places=6)

    def test_unknown_attribution_method_raises(self):
        prior, eA, eB = self._redundant_sources()
        ans = reason(prior, [eA, eB])
        with self.assertRaisesRegex(ValueError, "must be 'sequential' or 'shapley'"):
            ans.attribution(method="bogus")

    def test_default_attribution_method_is_sequential(self):
        # Explicit regression guard: reason_nonlinear_test.py's order-dependence test relies on the
        # DEFAULT staying sequential (order-dependent for redundant/nonlinear evidence), not shapley.
        prior, eA, eB = self._redundant_sources()
        ans = reason(prior, [eA, eB])
        self.assertEqual(ans.attribution(), ans.attribution(method="sequential"))


if __name__ == "__main__":
    unittest.main()
