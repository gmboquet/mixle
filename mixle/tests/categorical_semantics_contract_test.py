"""A categorical reports what its numbers are, and composition decides admissibility for itself.

MXR-080-1841: an empty pmap and one summing to 1.6 both reported ``EXACT``. The label was bent
because ``_owned_generative_components`` refuses ``LIKELIHOOD_FACTOR``, and mixtures, HMMs and
segment models here are all built from categoricals of exactly those shapes. These tests pin both
halves at once, because fixing either alone regresses the other.

MXR-080-1220: the same mislabelling is what let a no-evidence fit pose as a law.
"""

import unittest

from mixle.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    GaussianDistribution,
    HiddenMarkovModelDistribution,
    MixtureDistribution,
)
from mixle.stats.compute.pdist import DensitySemantics


def _law(**pmap) -> CategoricalDistribution:
    return CategoricalDistribution(pmap or {"a": 0.5, "b": 0.5})


class DerivedSemanticsTest(unittest.TestCase):
    """``density_semantics()`` is a function of the parameters, not of what is convenient."""

    def test_a_normalized_pmap_is_exact(self):
        self.assertIs(_law(a=0.5, b=0.5).density_semantics(), DensitySemantics.EXACT)

    def test_a_pmap_that_does_not_sum_to_one_is_a_factor(self):
        heavy = CategoricalDistribution({"a": 0.8, "b": 0.8})  # total mass 1.6
        self.assertIs(heavy.density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)
        self.assertFalse(heavy.is_normalized_probability)

    def test_the_empty_pmap_is_a_factor(self):
        self.assertIs(CategoricalDistribution({}).density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)

    def test_open_world_smoothing_is_a_factor(self):
        # A positive default_value assigns mass to every unseen label, so there is no finite total.
        smoothed = CategoricalDistribution({"a": 0.5, "b": 0.5}, default_value=0.1)
        self.assertIs(smoothed.density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)

    def test_the_label_and_the_flag_never_disagree(self):
        for pmap, default in (({"a": 1.0}, 0.0), ({}, 0.0), ({"a": 0.8, "b": 0.8}, 0.0), ({"a": 1.0}, 0.25)):
            with self.subTest(pmap=pmap, default_value=default):
                dist = CategoricalDistribution(pmap, default_value=default)
                exact = dist.density_semantics() is DensitySemantics.EXACT
                self.assertEqual(exact, dist.is_normalized_probability)


class ComposabilityTest(unittest.TestCase):
    """Honest labels must not cost composability -- that trade is what produced the wrong label."""

    def test_an_unnormalized_component_still_builds_a_mixture(self):
        # A constant scale on one component is absorbed into its own mixing weight, so
        # responsibilities are unchanged and the E-step does not care.
        mixture = MixtureDistribution([CategoricalDistribution({"a": 0.95}), _law()], [0.5, 0.5])
        self.assertIs(mixture.density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)

    def test_a_zero_responsibility_component_still_builds_a_mixture(self):
        # The empty pmap is what EM produces for a component that won no weight this iteration.
        MixtureDistribution([CategoricalDistribution({}), _law()], [0.5, 0.5])

    def test_an_unnormalized_emission_still_builds_an_hmm(self):
        HiddenMarkovModelDistribution(
            [CategoricalDistribution({"a": 0.95}), _law()], [0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]]
        )

    def test_a_scoring_only_component_is_still_refused(self):
        # An author's declaration that the object is not generative is not something the composition
        # layer may reason its way around -- this is the case the gate exists for.
        factor = CategoricalDistribution({"a": 0.5, "b": 0.5}, scoring_only=True)
        self.assertFalse(factor.composable_as_component())
        with self.assertRaisesRegex(TypeError, "likelihood factors found at indices"):
            MixtureDistribution([factor, _law()], [0.5, 0.5])

    def test_a_non_categorical_factor_without_the_declaration_is_refused(self):
        # Admissibility is opt-in: absent the declaration a factor is still refused, so this cannot
        # become a blanket hole for every LIKELIHOOD_FACTOR.
        class _Undeclared(GaussianDistribution):
            def density_semantics(self):
                return DensitySemantics.LIKELIHOOD_FACTOR

        with self.assertRaisesRegex(TypeError, "likelihood factors found at indices"):
            MixtureDistribution([_Undeclared(0.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])


class NoEvidenceFitTest(unittest.TestCase):
    """A fit with no evidence is a defined result that says what it is (MXR-080-1220)."""

    def _empty_fit(self) -> CategoricalDistribution:
        return CategoricalEstimator().estimate(None, {})

    def test_it_does_not_claim_to_be_a_law(self):
        fitted = self._empty_fit()
        self.assertIs(fitted.density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)
        self.assertFalse(fitted.is_normalized_probability)

    def test_it_scores_everything_as_impossible(self):
        self.assertEqual(self._empty_fit().log_density("anything"), float("-inf"))

    def test_sampling_it_names_the_cause_rather_than_inventing_a_draw(self):
        with self.assertRaises(ValueError) as caught:
            self._empty_fit().sampler(seed=0)
        message = str(caught.exception)
        self.assertIn("pmap is empty", message)
        self.assertIn("no evidence", message)

    def test_it_remains_usable_as_the_component_em_produces(self):
        # Refusing to construct it would abort a whole fit over a component that recovers on the
        # next E-step, which is why the result exists at all.
        MixtureDistribution([self._empty_fit(), _law()], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
