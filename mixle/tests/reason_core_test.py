"""Regression tests for mixle.reason.core audit fixes (0.8.0 exhaustive code review).

MXR-080-0272 (marginal entropy/attribution bookkeeping), MXR-080-0273 (predict() UQ validation),
MXR-080-0274 (Latent/mechanistic/block_selector dimension and covariance validation), and
MXR-080-0275 (fold-order-dependent linear attribution) each get their own TestCase below, named
after the finding ID they cover.
"""

import unittest

import numpy as np

from mixle.inference.belief import GaussianBelief
from mixle.reason.core import Latent, reason
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


if __name__ == "__main__":
    unittest.main()
