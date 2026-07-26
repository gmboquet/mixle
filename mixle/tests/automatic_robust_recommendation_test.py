"""MarginalFieldProfile.robust_recommendation(): combine the in-sample (BIC) model choice with the
held-out validation check into one principled final answer, instead of leaving their disagreement as
a text note nobody acts on. Held-out generalization is stronger evidence than in-sample penalized
likelihood, but a validation preference that is itself ambiguous (a small, noisy gap) should not
overturn BIC -- only a decisive disagreement should.
"""

import unittest

import numpy as np

from mixle.utils.automatic import analyze_structure
from mixle.utils.automatic.profiling import AMBIGUOUS_SCORE_GAP_BITS, MarginalFieldProfile, _gaussian_bits


def _profile(recommendation, validation_recommendation=None, validation_score_gap_bits=None):
    return MarginalFieldProfile(
        path=(),
        role="field",
        count=100,
        missing_count=0,
        missing_fraction=0.0,
        observed_count=100,
        kind="numeric",
        recommendation=recommendation,
        validation_recommendation=validation_recommendation,
        validation_score_gap_bits=validation_score_gap_bits,
    )


class RobustRecommendationUnitTest(unittest.TestCase):
    def test_gaussian_differential_code_length_is_not_clipped(self):
        variance = 1.0e-6
        expected = 0.5 * np.log2(2.0 * np.pi * np.e * variance)
        self.assertAlmostEqual(_gaussian_bits(variance), expected)
        self.assertLess(_gaussian_bits(variance), 0.0)

    def test_no_validation_evidence_keeps_the_marginal_recommendation(self):
        p = _profile("gaussian")
        self.assertEqual(p.robust_recommendation(), "gaussian")

    def test_validation_agrees_no_change(self):
        p = _profile("gaussian", validation_recommendation="gaussian", validation_score_gap_bits=1.0)
        self.assertEqual(p.robust_recommendation(), "gaussian")

    def test_decisive_validation_disagreement_overrides(self):
        gap = AMBIGUOUS_SCORE_GAP_BITS + 0.01
        p = _profile("lognormal", validation_recommendation="gamma", validation_score_gap_bits=gap)
        self.assertEqual(p.robust_recommendation(), "gamma")

    def test_ambiguous_validation_disagreement_does_not_override(self):
        gap = AMBIGUOUS_SCORE_GAP_BITS - 0.001
        p = _profile("lognormal", validation_recommendation="gamma", validation_score_gap_bits=gap)
        self.assertEqual(p.robust_recommendation(), "lognormal")

    def test_exactly_at_the_ambiguous_boundary_overrides(self):
        # the gate is a strict '<' check against the shared AMBIGUOUS_SCORE_GAP_BITS constant (the
        # same "is this gap ambiguous" check used elsewhere in this module, e.g. the "top validation
        # models are close" note), so the exact boundary value itself counts as decisive: only a gap
        # STRICTLY BELOW the threshold is "too close to call".
        p = _profile("lognormal", validation_recommendation="gamma", validation_score_gap_bits=AMBIGUOUS_SCORE_GAP_BITS)
        self.assertEqual(p.robust_recommendation(), "gamma")

    def test_missing_gap_with_disagreement_does_not_override(self):
        p = _profile("lognormal", validation_recommendation="gamma", validation_score_gap_bits=None)
        self.assertEqual(p.robust_recommendation(), "lognormal")


class RobustRecommendationIntegrationTest(unittest.TestCase):
    def test_a_known_decisive_disagreement_is_surfaced_and_overrides(self):
        # a few large outliers pull the marginal (in-sample) BIC toward integer_categorical, but a
        # held-out split clearly favors Poisson -- a real, reproducible disagreement case.
        data = [0] * 30 + [1] * 10 + [10] * 2
        profile = analyze_structure(
            data, pairwise=False, validation_seed=17, validation_fraction=0.25, validation_min_count=10
        )
        field = profile.fields[0]
        self.assertEqual(field.recommendation, "integer_categorical")
        self.assertEqual(field.validation_recommendation, "poisson")
        self.assertEqual(field.robust_recommendation(), "poisson")
        self.assertEqual(type(profile.recommend()).__name__, "PoissonEstimator")
        self.assertIs(profile.selection.estimator, profile.recommend())
        self.assertEqual(profile.selection.decisions[0].selected_recommendation, "poisson")
        self.assertTrue(profile.selection.decisions[0].validation_overrode)
        self.assertIn("gap is decisive -- robust_recommendation() overrides to poisson", field.validation_notes)

    def test_robust_recommendation_is_serialized_in_summary(self):
        data = list(np.random.RandomState(3).normal(0.0, 1.0, 300))
        profile = analyze_structure(data, pairwise=False)
        summary = profile.fields[0].summary()
        self.assertIn("robust_recommendation", summary)
        self.assertEqual(summary["robust_recommendation"], profile.fields[0].robust_recommendation())

    def test_no_disagreement_case_robust_recommendation_matches_marginal(self):
        # plain, well-behaved Gaussian data: no reason for validation to disagree, so robust_recommendation
        # should just equal the marginal recommendation.
        data = list(np.random.RandomState(4).normal(5.0, 2.0, 500))
        profile = analyze_structure(data, pairwise=False)
        field = profile.fields[0]
        self.assertEqual(field.robust_recommendation(), field.recommendation)

    def test_existing_disagreement_note_wording_is_unchanged(self):
        # regression guard: the pre-existing exact note text must survive untouched (other consumers
        # may match on it) -- the new decisive-override note is a SEPARATE, additional entry.
        data = [0] * 30 + [1] * 10 + [10] * 2
        profile = analyze_structure(
            data, pairwise=False, validation_seed=17, validation_fraction=0.25, validation_min_count=10
        )
        field = profile.fields[0]
        self.assertIn(
            "validation prefers poisson over marginal recommendation integer_categorical", field.validation_notes
        )

    def test_registered_family_participates_in_validation_and_deployment(self):
        data = list(np.random.RandomState(9).laplace(2.0, 1.5, 1000))
        profile = analyze_structure(data, pairwise=False)
        field = profile.fields[0]
        self.assertEqual(field.recommendation, "laplace")
        self.assertIn("laplace", field.validation_scores_bits)
        self.assertEqual(field.robust_recommendation(), "laplace")
        self.assertEqual(type(profile.recommend()).__name__, "LaplaceEstimator")


if __name__ == "__main__":
    unittest.main()
