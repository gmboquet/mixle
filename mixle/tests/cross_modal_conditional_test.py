"""Cross-modal reasoning = conditional inference in a shared-latent joint (workstream L2).

Generalizes ``mixle/reason/cycle_consistency.py``'s round-trip closure diagnostic (previously proven
only for a single neural conditional-density transport, workstream F5) to a typed-grammar
:class:`~mixle.reason.cross_modal.CrossModalJoint`: a :class:`~mixle.stats.latent.mixture.MixtureDistribution`
over :class:`~mixle.stats.combinator.composite.CompositeDistribution` fields, where the shared mixture
component index IS the latent regime tying heterogeneous modalities together. Because the joint is a
typed grammar object (not an opaque transport), its true marginals are available in closed form, so this
generalization compares the round-trip receipt directly against the true marginal rather than only
against itself (see ``joint_cycle_consistency_receipt``'s docstring for the full comparison).
"""

import math
import unittest

import numpy as np

from mixle.reason.cross_modal import CrossModalJoint
from mixle.reason.cycle_consistency import joint_cycle_consistency_receipt
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


def _gaussian_pdf(x: float, mu: float, sigma2: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma2)) / math.sqrt(2.0 * math.pi * sigma2)


def _two_modality_joint() -> CrossModalJoint:
    """A 2-component, 2-modality (image=Gaussian, text=Categorical) joint with hand-derivable posteriors.

    regime 0 (w=0.6): image ~ N(-3, 1), text ~ Categorical({"cat": 0.9, "dog": 0.1})
    regime 1 (w=0.4): image ~ N(3, 1),  text ~ Categorical({"cat": 0.1, "dog": 0.9})
    """
    return CrossModalJoint.from_components(
        names=("image", "text"),
        component_fields=[
            (
                GaussianDistribution(mu=-3.0, sigma2=1.0),
                CategoricalDistribution(pmap={"cat": 0.9, "dog": 0.1}),
            ),
            (
                GaussianDistribution(mu=3.0, sigma2=1.0),
                CategoricalDistribution(pmap={"cat": 0.1, "dog": 0.9}),
            ),
        ],
        weights=[0.6, 0.4],
    )


class CrossModalAnalyticPosteriorTest(unittest.TestCase):
    """Acceptance criterion 1: cross-modal posteriors on a synthetic joint match hand-derived analytic answers."""

    def setUp(self):
        self.joint = _two_modality_joint()

    def test_condition_on_text_infer_image_matches_analytic_posterior(self):
        posterior = self.joint.infer({"text": "cat"}, ["image"])

        w0 = 0.6 * 0.9
        w1 = 0.4 * 0.1
        expected_w = np.asarray([w0, w1]) / (w0 + w1)

        np.testing.assert_allclose(posterior.w, expected_w, atol=1e-10)
        # component means are untouched by conditioning on the OTHER modality
        self.assertAlmostEqual(posterior.components[0].dists[0].mu, -3.0)
        self.assertAlmostEqual(posterior.components[1].dists[0].mu, 3.0)

        # spot-check the actual posterior density against the hand-derived mixture density
        for x in (-3.0, 0.0, 3.0, 5.0):
            expected_density = expected_w[0] * _gaussian_pdf(x, -3.0, 1.0) + expected_w[1] * _gaussian_pdf(x, 3.0, 1.0)
            self.assertAlmostEqual(posterior.density((x,)), expected_density, places=8)

    def test_condition_on_image_infer_text_matches_analytic_posterior(self):
        x = -3.0
        posterior = self.joint.infer({"image": x}, ["text"])

        p0 = 0.6 * _gaussian_pdf(x, -3.0, 1.0)
        p1 = 0.4 * _gaussian_pdf(x, 3.0, 1.0)
        expected_w = np.asarray([p0, p1]) / (p0 + p1)

        expected_p_cat = expected_w[0] * 0.9 + expected_w[1] * 0.1
        expected_p_dog = expected_w[0] * 0.1 + expected_w[1] * 0.9

        self.assertAlmostEqual(posterior.density(("cat",)), expected_p_cat, places=8)
        self.assertAlmostEqual(posterior.density(("dog",)), expected_p_dog, places=8)
        self.assertAlmostEqual(posterior.density(("cat",)) + posterior.density(("dog",)), 1.0, places=8)

    def test_empty_observation_recovers_the_prior_marginal(self):
        posterior = self.joint.infer({}, ["image"])
        np.testing.assert_allclose(posterior.w, [0.6, 0.4])
        for x in (-3.0, 0.0, 3.0):
            expected_density = 0.6 * _gaussian_pdf(x, -3.0, 1.0) + 0.4 * _gaussian_pdf(x, 3.0, 1.0)
            self.assertAlmostEqual(posterior.density((x,)), expected_density, places=8)


class CrossModalThreeModalitySubsetTest(unittest.TestCase):
    """Acceptance criterion 3: condition on ANY subset, infer ANY other subset, for >= 3 modalities."""

    def setUp(self):
        # regime 0 (w=0.5): image ~ N(-4, 0.5), text -> mostly "cat", audio ~ N(-1, 0.2)
        # regime 1 (w=0.5): image ~ N(4, 0.5),  text -> mostly "dog", audio ~ N(1, 0.2)
        self.joint = CrossModalJoint.from_components(
            names=("image", "text", "audio"),
            component_fields=[
                (
                    GaussianDistribution(mu=-4.0, sigma2=0.5),
                    CategoricalDistribution(pmap={"cat": 0.95, "dog": 0.05}),
                    GaussianDistribution(mu=-1.0, sigma2=0.2),
                ),
                (
                    GaussianDistribution(mu=4.0, sigma2=0.5),
                    CategoricalDistribution(pmap={"cat": 0.05, "dog": 0.95}),
                    GaussianDistribution(mu=1.0, sigma2=0.2),
                ),
            ],
            weights=[0.5, 0.5],
        )

    def _regime_posterior(self, observed):
        """Hand-computed regime posterior for a given ``{modality: value}`` observation, used as the
        ground truth every ``infer`` result below is checked against."""
        log_w = np.log([0.5, 0.5])
        dists = {
            "image": [-4.0, 4.0],  # means, sigma2=0.5 for both regimes
            "audio": [-1.0, 1.0],  # means, sigma2=0.2 for both regimes
        }
        for name, value in observed.items():
            if name == "text":
                p = [0.95 if value == "cat" else 0.05, 0.05 if value == "cat" else 0.95]
                log_w = log_w + np.log(p)
            elif name in dists:
                sigma2 = 0.5 if name == "image" else 0.2
                mus = dists[name]
                log_w = log_w + np.log([_gaussian_pdf(value, mu, sigma2) for mu in mus])
        w = np.exp(log_w - np.max(log_w))
        return w / w.sum()

    def test_condition_on_one_infer_other_two_jointly(self):
        post = self.joint.infer({"audio": -1.0}, ["image", "text"])
        expected_w = self._regime_posterior({"audio": -1.0})
        np.testing.assert_allclose(post.w, expected_w, atol=1e-10)
        self.assertAlmostEqual(post.components[0].dists[0].mu, -4.0)
        self.assertEqual(post.components[0].dists[1].pmap, {"cat": 0.95, "dog": 0.05})

    def test_condition_on_two_infer_the_remaining_one(self):
        post = self.joint.infer({"image": -4.0, "text": "cat"}, ["audio"])
        expected_w = self._regime_posterior({"image": -4.0, "text": "cat"})
        np.testing.assert_allclose(post.w, expected_w, atol=1e-6)
        for x in (-1.0, 0.0, 1.0):
            expected_density = expected_w[0] * _gaussian_pdf(x, -1.0, 0.2) + expected_w[1] * _gaussian_pdf(x, 1.0, 0.2)
            self.assertAlmostEqual(post.density((x,)), expected_density, places=6)

    def test_condition_on_none_infer_all_three_jointly(self):
        post = self.joint.infer({})
        np.testing.assert_allclose(post.w, [0.5, 0.5])
        self.assertEqual(post.components[0].count, 3)

    def test_target_order_matches_requested_order_not_composite_order(self):
        forward = self.joint.infer({"audio": -1.0}, ["text", "image"])
        backward = self.joint.infer({"audio": -1.0}, ["image", "text"])
        # same underlying posterior, tuple fields swapped
        self.assertAlmostEqual(forward.density(("cat", -4.0)), backward.density((-4.0, "cat")), places=8)

    def test_target_must_not_overlap_observed(self):
        with self.assertRaises(ValueError):
            self.joint.infer({"image": -4.0}, ["image"])

    def test_unknown_modality_name_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.joint.infer({"smell": 1.0})


class CycleConsistencyReceiptFlagsBrokenProjectionTest(unittest.TestCase):
    """Acceptance criterion 2: the generalized cycle-consistency receipt flags a broken projection."""

    def setUp(self):
        # Regimes are well-separated (10 std apart in image, near-deterministic text) so both the
        # well-behaved round trip and the broken one are essentially noise-free -- the receipt gap
        # between them should be large and not a Monte-Carlo coin flip.
        # Regime weights are deliberately skewed (0.9/0.1): a broken backward projection that swaps
        # regime<->text assignments then also swaps which image mode each text label routes back to,
        # which (combined with the skew) produces a large, unambiguous round-trip weight inversion
        # rather than a small perturbation.
        self.well_behaved = CrossModalJoint.from_components(
            names=("image", "text"),
            component_fields=[
                (
                    GaussianDistribution(mu=-5.0, sigma2=0.25),
                    CategoricalDistribution(pmap={"cat": 0.99, "dog": 0.01}),
                ),
                (
                    GaussianDistribution(mu=5.0, sigma2=0.25),
                    CategoricalDistribution(pmap={"cat": 0.01, "dog": 0.99}),
                ),
            ],
            weights=[0.9, 0.1],
        )
        # Same image/weight structure, but the regime -> text mapping is SWAPPED relative to
        # ``well_behaved``: regime 0 (image near -5) is now the "mostly dog" regime and vice versa.
        # Using this as the BACKWARD leg of the round trip means inferring image-from-text routes to
        # the wrong Gaussian mode almost every time -- a deliberately broken A<-B projection.
        self.broken = CrossModalJoint.from_components(
            names=("image", "text"),
            component_fields=[
                (
                    GaussianDistribution(mu=-5.0, sigma2=0.25),
                    CategoricalDistribution(pmap={"cat": 0.01, "dog": 0.99}),
                ),
                (
                    GaussianDistribution(mu=5.0, sigma2=0.25),
                    CategoricalDistribution(pmap={"cat": 0.99, "dog": 0.01}),
                ),
            ],
            weights=[0.9, 0.1],
        )

    def test_well_behaved_receipt_is_near_zero(self):
        receipt = joint_cycle_consistency_receipt(
            self.well_behaved, "image", "text", n_round_trip=200, n_kl_samples=400, seed=0
        )
        self.assertLess(receipt, 0.1)

    def test_broken_backward_projection_is_clearly_elevated(self):
        well_receipt = joint_cycle_consistency_receipt(
            self.well_behaved, "image", "text", n_round_trip=200, n_kl_samples=400, seed=0
        )
        broken_receipt = joint_cycle_consistency_receipt(
            self.well_behaved,
            "image",
            "text",
            backward_joint=self.broken,
            n_round_trip=200,
            n_kl_samples=400,
            seed=0,
        )
        self.assertGreater(broken_receipt, max(well_receipt * 10.0, 1.0))


if __name__ == "__main__":
    unittest.main()
