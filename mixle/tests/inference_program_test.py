"""Multi-hop inference programs (roadmap M5, part (a)/(b)).

Fixture: two separately-fit :class:`~mixle.reason.cross_modal.CrossModalJoint` objects sharing a
categorical "B" modality -- ``joint_AB`` (A=Gaussian, B=Categorical) and ``joint_BC`` (B=Categorical,
C=Gaussian) -- standing in for the card's own "image field -> shared latent -> predicted field ->
text field" chain. Evidence ``A=0.0`` is engineered to be EXACTLY equidistant between ``joint_AB``'s
two regimes, so the true posterior over B is an exact 50/50 tie -- and ``joint_BC``'s two regimes route
B="lo"/"hi" to two far-apart Gaussian modes for C. This makes "marginalizing the middle hop matters"
concrete and checkable in closed form: the correct P(C|A=0) is a genuinely bimodal mixture with mean
~0, while collapsing B's posterior to a single point (arg-max of an exact tie) routes ALL mass to one
mode and reports a mean ~10 away from the true one.
"""

import unittest

import numpy as np

from mixle.reason.cross_modal import CrossModalJoint
from mixle.reason.inference_program import InferenceHop, run_inference_program
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


def _joint_ab() -> CrossModalJoint:
    return CrossModalJoint.from_components(
        names=("A", "B"),
        component_fields=[
            (GaussianDistribution(mu=-3.0, sigma2=1.0), CategoricalDistribution(pmap={"lo": 0.95, "hi": 0.05})),
            (GaussianDistribution(mu=3.0, sigma2=1.0), CategoricalDistribution(pmap={"lo": 0.05, "hi": 0.95})),
        ],
        weights=[0.5, 0.5],
    )


def _joint_bc() -> CrossModalJoint:
    return CrossModalJoint.from_components(
        names=("B", "C"),
        component_fields=[
            (CategoricalDistribution(pmap={"lo": 0.99, "hi": 0.01}), GaussianDistribution(mu=-10.0, sigma2=0.25)),
            (CategoricalDistribution(pmap={"lo": 0.01, "hi": 0.99}), GaussianDistribution(mu=10.0, sigma2=0.25)),
        ],
        weights=[0.5, 0.5],
    )


def _analytic_c_given_a(joint_ab: CrossModalJoint, joint_bc: CrossModalJoint, a: float) -> MixtureDistribution:
    """Closed-form (enumerated, no simulation) P(C | A=a): sum over B's exact posterior support."""
    b_post = joint_ab.infer({"A": a}, ["B"])
    components, weights = [], []
    for b in ("lo", "hi"):
        p_b = b_post.density((b,))
        c_post = joint_bc.infer({"B": b}, ["C"])
        for comp, cw in zip(c_post.components, c_post.w):
            components.append(comp)
            weights.append(p_b * float(cw))
    return MixtureDistribution(components, w=np.asarray(weights, dtype=np.float64))


class InferenceProgramTwoHopVsNaiveTest(unittest.TestCase):
    """Acceptance criterion (a): the 2-hop sampled program matches the analytic answer that a naive
    single-point ("moment") shortcut gets wrong."""

    def setUp(self):
        self.joint_ab = _joint_ab()
        self.joint_bc = _joint_bc()
        self.hops = [
            InferenceHop(joint=self.joint_ab, target=("B",)),
            InferenceHop(joint=self.joint_bc, target=("C",), carry={"B": "B"}),
        ]
        self.analytic = _analytic_c_given_a(self.joint_ab, self.joint_bc, a=0.0)

    def test_b_posterior_is_an_exact_tie_by_construction(self):
        b_post = self.joint_ab.infer({"A": 0.0}, ["B"])
        self.assertAlmostEqual(b_post.density(("lo",)), 0.5, places=10)
        self.assertAlmostEqual(b_post.density(("hi",)), 0.5, places=10)

    def test_analytic_marginal_is_bimodal_with_mean_near_zero(self):
        # sanity check on the hand-built ground truth itself before trusting it as the oracle
        mean = sum(w * c.dists[0].mu for c, w in zip(self.analytic.components, self.analytic.w))
        self.assertAlmostEqual(mean, 0.0, places=6)
        self.assertEqual(len(self.analytic.components), 4)  # 2 B-values x 2 joint_bc regimes each

    def test_sampled_propagation_matches_the_analytic_answer(self):
        result = run_inference_program({"A": 0.0}, self.hops, propagation="sampled", n_samples=4000, seed=0)
        self.assertAlmostEqual(result.mean("C"), 0.0, delta=0.75)
        # density spot-checks against the closed-form mixture, not just the mean
        for x in (-10.0, 0.0, 10.0):
            got = result.density((x,))
            expected = self.analytic.density((x,))
            self.assertAlmostEqual(got, expected, delta=max(0.15, 0.25 * expected))

    def test_moment_propagation_gets_the_answer_measurably_wrong(self):
        result = run_inference_program({"A": 0.0}, self.hops, propagation="moment", seed=0)
        # collapsing B's exact 50/50 posterior to one point routes ALL mass to one Gaussian mode
        # (near -10 or +10), nowhere near the true bimodal mean of ~0.
        self.assertGreater(abs(result.mean("C") - 0.0), 8.0)

    def test_receipt_records_the_propagation_choice(self):
        sampled = run_inference_program({"A": 0.0}, self.hops, propagation="sampled", n_samples=100, seed=0)
        moment = run_inference_program({"A": 0.0}, self.hops, propagation="moment", seed=0)
        self.assertEqual(sampled.receipt.propagation, "sampled")
        self.assertEqual(sampled.receipt.n_particles, 100)
        self.assertEqual(moment.receipt.propagation, "moment")
        self.assertEqual(moment.receipt.n_hops, 2)
        self.assertEqual(sampled.receipt.hop_targets, [("B",), ("C",)])

    def test_invalid_propagation_rejected(self):
        with self.assertRaises(ValueError):
            run_inference_program({"A": 0.0}, self.hops, propagation="bogus")

    def test_first_hop_cannot_carry(self):
        bad_hops = [
            InferenceHop(joint=self.joint_ab, target=("B",), carry={"X": "A"}),
            InferenceHop(joint=self.joint_bc, target=("C",), carry={"B": "B"}),
        ]
        with self.assertRaises(ValueError):
            run_inference_program({"A": 0.0}, bad_hops)

    def test_empty_program_rejected(self):
        with self.assertRaises(ValueError):
            run_inference_program({"A": 0.0}, [])


class InferenceProgramSingleHopReduceTest(unittest.TestCase):
    """A 1-hop program is exactly ``CrossModalJoint.infer`` -- no propagation machinery engaged."""

    def test_single_hop_matches_direct_infer(self):
        joint = _joint_ab()
        hops = [InferenceHop(joint=joint, target=("B",))]
        result = run_inference_program({"A": -3.0}, hops, propagation="sampled", n_samples=50, seed=1)
        direct = joint.infer({"A": -3.0}, ["B"])
        for b in ("lo", "hi"):
            self.assertAlmostEqual(result.density((b,)), direct.density((b,)), places=6)


if __name__ == "__main__":
    unittest.main()
