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

    def _mass(self, dist, keys=("a", "b")) -> float:
        import numpy as np

        return sum(float(np.exp(dist.log_density(key))) for key in keys)

    def test_a_sub_probability_mixture_refuses_to_sample(self):
        """A scaled component's constant cancels in the E-step, NOT in the composite (MXR-080-1857).

        A mass-0.75 component under weight 0.5 leaves the mixture at total mass 0.875. The mixture
        reported LIKELIHOOD_FACTOR correctly, but its sampler drew as though the weights were draw
        probabilities, so the scorer and the sampler described different objects.
        """
        mixture = MixtureDistribution([CategoricalDistribution({"a": 0.5, "b": 0.25}), _law()], [0.5, 0.5])
        self.assertAlmostEqual(self._mass(mixture), 0.875)
        with self.assertRaisesRegex(ValueError, "likelihood factor, not a normalized law"):
            mixture.sampler(seed=0)

    def test_an_open_world_component_refuses_to_sample_but_still_scores(self):
        # An open-world smoothed component has infinite total mass, so there is no finite scale to
        # absorb and no coherent mixture to sample. Scoring it remains legitimate and is tested
        # elsewhere (sparse_mixture_test builds exactly this to check a bound), so the object must
        # stay constructible -- refusing at construction would be the guard overreaching.
        mixture = MixtureDistribution(
            [CategoricalDistribution({"a": 0.5, "b": 0.5}, default_value=0.25), _law()], [0.5, 0.5]
        )
        self.assertTrue(float(mixture.log_density("a")) < 0.0)
        with self.assertRaisesRegex(ValueError, "cannot be sampled"):
            mixture.sampler(seed=0)

    def test_a_normalized_mixture_still_samples(self):
        mixture = MixtureDistribution([_law(), CategoricalDistribution({"a": 0.9, "b": 0.1})], [0.5, 0.5])
        self.assertAlmostEqual(self._mass(mixture), 1.0)
        self.assertEqual(len(mixture.sampler(seed=0).sample(3)), 3)

    def test_a_gaussian_mixture_is_unaffected(self):
        mixture = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(2.0, 1.0)], [0.5, 0.5])
        self.assertEqual(len(mixture.sampler(seed=0).sample(3)), 3)

    def test_a_non_categorical_factor_without_the_declaration_is_refused(self):
        # Admissibility is opt-in: absent the declaration a factor is still refused, so this cannot
        # become a blanket hole for every LIKELIHOOD_FACTOR.
        class _Undeclared(GaussianDistribution):
            def density_semantics(self):
                return DensitySemantics.LIKELIHOOD_FACTOR

        with self.assertRaisesRegex(TypeError, "likelihood factors found at indices"):
            MixtureDistribution([_Undeclared(0.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5])


class NoEvidenceFitTest(unittest.TestCase):
    """A successful estimator call returns a generative law, or does not succeed (MXR-080-1220).

    The earlier resolution returned a supportless categorical for a zero-evidence fit and described
    it honestly as a likelihood factor. That was still a SUCCESSFUL call handing back an object whose
    sampler raised, and it was worse than it looked: an all ``-inf`` component can never win
    responsibility again, so the dead-component case it existed to serve could not actually recover.
    """

    def _zero_responsibility_statistic(self):
        """What EM hands the estimator when a component earns no weight on any row."""
        import numpy as np

        seen = CategoricalDistribution({"a": 0.5, "b": 0.5})
        accumulator = CategoricalEstimator().accumulator_factory().make()
        accumulator.seq_update(seen.dist_to_encoder().seq_encode(["a", "b", "a"]), np.zeros(3), seen)
        return accumulator.value()

    def test_a_seen_label_survives_earning_no_weight(self):
        # count_map holds positive counts only, so the support had to be recorded separately.
        self.assertEqual(self._zero_responsibility_statistic(), {"a": 0.0, "b": 0.0})

    def test_a_dead_component_estimates_to_a_uniform_law(self):
        fitted = CategoricalEstimator().estimate(None, self._zero_responsibility_statistic())
        self.assertIs(fitted.density_semantics(), DensitySemantics.EXACT)
        self.assertTrue(fitted.is_normalized_probability)
        self.assertEqual(fitted.pmap, {"a": 0.5, "b": 0.5})

    def test_that_result_is_generative(self):
        fitted = CategoricalEstimator().estimate(None, self._zero_responsibility_statistic())
        self.assertEqual(len(fitted.sampler(seed=0).sample(3)), 3)
        MixtureDistribution([fitted, _law()], [0.5, 0.5]).sampler(seed=0)

    def test_it_can_win_responsibility_again(self):
        # The point of the dead-component case: a uniform scores finitely, so the next E-step can
        # give it weight. The supportless object it replaced scored -inf on every row forever.
        fitted = CategoricalEstimator().estimate(None, self._zero_responsibility_statistic())
        self.assertTrue(float(fitted.log_density("a")) > float("-inf"))

    def test_an_estimator_given_nothing_at_all_returns_a_declared_factor(self):
        """The one case that remains non-generative, and why refusing it is not available.

        With nothing accumulated there is no label set to place a distribution over. Raising was
        tried and reverted: a learned segment fitted on all-empty sequences, a gated mixture whose
        evidence is impossible for a component, and hidden association all reach this legitimately,
        and refusing broke all three. The result says what it is instead of posing as a law, and a
        mixture built over it is a factor too and refuses to sample (MXR-080-1857).
        """
        fitted = CategoricalEstimator().estimate(None, {})
        self.assertIs(fitted.density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)
        self.assertFalse(fitted.is_normalized_probability)
        self.assertEqual(fitted.log_density("anything"), float("-inf"))
        with self.assertRaisesRegex(ValueError, "cannot be sampled|no evidence"):
            MixtureDistribution([fitted, _law()], [0.5, 0.5]).sampler(seed=0)

    def test_a_label_the_component_cannot_explain_is_not_its_support(self):
        # Impossible evidence must not become a component's support: resetting a dead component to a
        # uniform over labels it never modelled would invent a law from evidence it cannot explain.
        import numpy as np

        modelled = CategoricalDistribution({"a": 1.0})
        accumulator = CategoricalEstimator().accumulator_factory().make()
        encoder = CategoricalDistribution({"a": 0.5, "outside": 0.5}).dist_to_encoder()
        accumulator.seq_update(encoder.seq_encode(["outside"]), np.zeros(1), modelled)
        self.assertEqual(accumulator.value(), {})

    def test_a_declared_support_still_gives_a_no_data_uniform(self):
        fitted = CategoricalEstimator(suff_stat={"a": 1.0, "b": 1.0}).estimate(None, {})
        self.assertEqual(fitted.pmap, {"a": 0.5, "b": 0.5})
        self.assertTrue(fitted.is_normalized_probability)

    def test_an_ordinary_fit_is_unchanged(self):
        # count_map still carries positive counts only, so nothing about a normal fit moves.
        import numpy as np

        seen = CategoricalDistribution({"a": 0.5, "b": 0.5})
        accumulator = CategoricalEstimator().accumulator_factory().make()
        accumulator.seq_update(seen.dist_to_encoder().seq_encode(["a", "b", "a"]), np.ones(3), seen)
        self.assertEqual(accumulator.value(), {"a": 2.0, "b": 1.0})
        self.assertEqual(CategoricalEstimator().estimate(None, accumulator.value()).pmap, {"a": 2 / 3, "b": 1 / 3})


if __name__ == "__main__":
    unittest.main()
