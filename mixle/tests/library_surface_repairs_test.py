"""What the shipped example scripts exercise: surrogates, screens, and degenerate inputs.

Passes 09 and 10 of the ten adversarial reviews of the 0.8.1 candidate ran every example script in
the corpus. Four library defects surfaced only there: a Bayesian-optimization surrogate that crashed
with a raw torch ``_LinAlgError`` on 11 of 12 seeds, a graph family that silently clipped the
probabilities its own positions implied, a dependency screen that dropped a numeric field for the
FAMILY it was recommended rather than for its values, and a family of degenerate-input crashes whose
messages named the wrong precondition.
"""

from __future__ import annotations

import math
import unittest
import warnings

import numpy as np

import mixle.stats as S
from mixle.stats import RandomDotProductGraphDistribution


def _quiet(callable_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return callable_()


class BayesOptSurrogateTest(unittest.TestCase):
    """P09-F03: ``minimize`` worked on the example's seed and crashed on eleven others."""

    @staticmethod
    def _quadratic(point):
        return float((point[0] - 1.0) ** 2 + (point[1] + 2.0) ** 2)

    def test_every_seed_reaches_the_optimum_of_a_noise_free_quadratic(self):
        from mixle.doe import minimize

        for seed in range(6):
            with self.subTest(seed=seed):
                result = _quiet(
                    lambda seed=seed: minimize(
                        self._quadratic, [(-5.0, 5.0), (-5.0, 5.0)], n_init=5, n_iter=15, seed=seed
                    )
                )
                self.assertLess(float(np.hypot(result.best_x[0] - 1.0, result.best_x[1] + 2.0)), 0.5)

    def test_a_singular_kernel_matrix_is_repaired_and_disclosed(self):
        import torch

        from mixle.models import GaussianProcessRegressor

        gp = GaussianProcessRegressor(noise=1.0e-8, jitter=1.0e-14)
        self.assertEqual(gp.numerical_repairs(), ())
        rank_one = torch.ones((6, 6), dtype=torch.float64)
        eye = torch.eye(6, dtype=torch.float64)
        chol = gp._chol(rank_one, eye)
        self.assertEqual(tuple(chol.shape), (6, 6))
        repairs = gp.numerical_repairs()
        self.assertEqual(len(repairs), 1)
        self.assertIn("covariance-ridged", repairs[0])

    def test_a_matrix_no_jitter_can_rescue_raises_a_named_error(self):
        import torch

        from mixle.models import GaussianProcessRegressor

        gp = GaussianProcessRegressor()
        indefinite = torch.zeros((5, 5), dtype=torch.float64)
        indefinite[0, 0] = -1.0
        with self.assertRaises(ValueError) as caught:
            gp._chol(indefinite, torch.eye(5, dtype=torch.float64))
        self.assertIn("Gaussian-process covariance", str(caught.exception))

    def test_a_clean_fit_reports_no_repair(self):
        from mixle.models import GaussianProcessRegressor

        gp = GaussianProcessRegressor()
        x = np.linspace(0.0, 3.0, 12).reshape(-1, 1)
        _quiet(lambda: gp.fit(x, np.sin(x).ravel(), max_its=25))
        self.assertEqual(gp.numerical_repairs(), ())


class RandomDotProductClipTest(unittest.TestCase):
    """P09-F12: a clamp that changes which model is being sampled must be disclosed."""

    def test_positions_whose_products_exceed_one_report_the_clip(self):
        positions = np.random.RandomState(0).rand(8, 2)
        distribution = RandomDotProductGraphDistribution(positions)
        repairs = distribution.numerical_repairs()
        self.assertEqual(len(repairs), 1)
        self.assertIn("edge-probability-clipped", repairs[0])
        self.assertIn("rank-2", repairs[0])
        self.assertLessEqual(float(np.max(distribution.edge_marginals())), 1.0)

    def test_positions_inside_the_unit_ball_report_nothing(self):
        positions = np.random.RandomState(0).rand(8, 2) / np.sqrt(2.0)
        self.assertEqual(RandomDotProductGraphDistribution(positions).numerical_repairs(), ())

    def test_the_count_is_of_off_diagonal_entries_only(self):
        # The diagonal is zeroed regardless, so a self-product above one is not a clip anyone can see.
        positions = np.array([[0.9, 0.9], [0.05, 0.05]])
        self.assertGreater(float(positions[0] @ positions[0]), 1.0)
        self.assertEqual(RandomDotProductGraphDistribution(positions).numerical_repairs(), ())


class PairwiseScreenTest(unittest.TestCase):
    """P10-F01: a numeric column was screened for its recommended FAMILY, not for its values."""

    @staticmethod
    def _planted(seed=0, n=240):
        rng = np.random.RandomState(seed)
        category = rng.choice(["paid", "free"], size=n, p=[0.4, 0.6])
        amount = np.where(category == "paid", rng.normal(2.0, 0.5, n), rng.normal(0.5, 0.3, n))
        flag = np.where(category == "paid", rng.random(n) < 0.9, rng.random(n) < 0.05)
        return [(float(a), str(c), bool(f)) for a, c, f in zip(amount, category, flag)]

    def test_a_bimodal_numeric_field_is_screened_against_its_cause(self):
        from mixle.utils.automatic import analyze_structure

        profile = _quiet(lambda: analyze_structure(self._planted()))
        pairs = {(hint.left, hint.right) for hint in profile.pairwise_hints}
        self.assertEqual(profile.pairwise_pairs_checked, 3)
        self.assertIn(((0,), (1,)), pairs)
        self.assertIn(((1,), (2,)), pairs)

    def test_a_low_cardinality_numeric_field_is_still_encoded_exactly(self):
        from mixle.utils.automatic.profiling import _encode_for_pairwise, _profile_series

        # Three distinct levels: the discrete encoder represents them exactly, which is strictly more
        # informative than binning them, so the new binning route must not take this field away.
        values = [float([0.5, 2.5, 9.5][v % 3]) for v in range(200)]
        profile = _profile_series((0,), "field", values)
        self.assertNotIn(profile.recommendation, ("gaussian", "poisson"))
        encoded = _encode_for_pairwise(profile, values, max_cardinality=20, num_bins=8)
        self.assertIsNotNone(encoded)
        self.assertEqual(encoded[1], "empirical_discrete")

    def test_a_continuous_field_recommended_something_other_than_gaussian_is_binned(self):
        from mixle.utils.automatic.profiling import _encode_for_pairwise, _profile_series

        rng = np.random.RandomState(0)
        values = [float(v) for v in np.concatenate([rng.normal(-4.0, 0.4, 150), rng.normal(4.0, 0.4, 150)])]
        profile = _profile_series((0,), "field", values)
        self.assertNotIn(profile.recommendation, ("gaussian", "poisson"))
        encoded = _encode_for_pairwise(profile, values, max_cardinality=20, num_bins=8)
        self.assertIsNotNone(encoded)
        self.assertEqual(encoded[1], "quantile_bins")


class DegenerateInputTest(unittest.TestCase):
    """P10-F13: a refusal that names the wrong precondition is worse than no refusal."""

    def test_a_sampler_fit_with_fewer_observations_than_parameters_is_refused(self):
        from mixle.ppl import Mix, Normal, free

        model = Mix([Normal(free, free), Normal(free, free)], name="two_components")
        with self.assertRaises(ValueError) as caught:
            model.fit([-6.0, 6.0], how="mcmc", draws=200, burn=100, rng=np.random.RandomState(0))
        message = str(caught.exception)
        self.assertIn("two_components", message)
        self.assertIn("2 observation(s)", message)
        self.assertIn("no prior", message)

    def test_a_parameter_with_a_proper_prior_is_never_refused_for_sample_size(self):
        """A proper prior gives a proper posterior at any n, including one observation.

        The first version of this guard counted every slot against every ROW and refused ordinary
        Bayesian updates that ``map`` fitted on the same data and that 0.8.1 fitted on every route
        (R02-F03) -- the fail-closed-guard defect class this release exists to remove.
        """
        from mixle.ppl import Gamma, Normal

        rows = list(np.random.RandomState(0).normal(5.0, 2.0, 400))
        mu, sd = Normal(0.0, 10.0, name="mu"), Gamma(2.0, 2.0, name="sd")
        for route in ("mcmc", "laplace", "map"):
            with self.subTest(route=route):
                fitted = _quiet(
                    lambda route=route: Normal(mu, sd).fit(
                        rows[:1],
                        how=route,
                        rng=np.random.RandomState(0),
                        **({} if route == "map" else {"draws": 60, "burn": 20} if route == "mcmc" else {}),
                    )
                )
                self.assertIsNotNone(fitted)

    def test_a_row_of_a_vector_model_is_counted_as_the_scalars_it_carries(self):
        """Three draws from a 5-dimensional model are fifteen numbers, not three."""
        from mixle.ppl import DiagGaussian, free

        rows = np.random.RandomState(0).normal(size=(3, 5)).tolist()
        fitted = _quiet(
            lambda: DiagGaussian(5, mean=free(5, name="a"), var=np.full(5, 0.25)).fit(
                rows, how="laplace", rng=np.random.RandomState(0)
            )
        )
        self.assertIsNotNone(fitted)
        single = _quiet(
            lambda: DiagGaussian(2, mean=free(2, name="b"), var=np.full(2, 0.25)).fit(
                [[0.0, 1.0]], how="mcmc", draws=20, burn=5, rng=np.random.RandomState(0)
            )
        )
        self.assertIsNotNone(single)

    def test_the_refusal_names_the_parameters_that_have_no_prior(self):
        from mixle.ppl import Mix, Normal, free

        with self.assertRaises(ValueError) as caught:
            Mix([Normal(free, free), Normal(free, free)], name="two_components").fit(
                [-6.0, 6.0], how="mcmc", draws=200, burn=100, rng=np.random.RandomState(0)
            )
        message = str(caught.exception)
        self.assertIn("no prior", message)
        self.assertIn("2 observation(s)", message)
        self.assertNotIn("sampler on it wanders", message)  # laplace takes this path too

    def test_an_identified_sampler_fit_is_untouched(self):
        from mixle.ppl import Normal, free

        rows = list(np.random.RandomState(0).normal(3.0, 1.0, size=40))
        fitted = _quiet(
            lambda: Normal(free, free).fit(rows, how="mcmc", draws=200, burn=100, rng=np.random.RandomState(0))
        )
        self.assertTrue(math.isfinite(fitted.summary()["arg0"]["mean"]))

    def test_a_parameter_that_overflows_its_mapping_names_the_family(self):
        from mixle.ppl.core import _FAMILIES

        family = _FAMILIES["Normal"]
        with self.assertRaises(ValueError) as caught:
            family.make_dist((0.0, 1.0e200), None)
        self.assertIn("Normal", str(caught.exception))
        self.assertIn("overflow", str(caught.exception))

    def test_a_projection_below_the_target_component_count_is_refused(self):
        from mixle.ops import project

        source = S.MixtureDistribution(
            [S.GaussianDistribution(-6.0, 1.0), S.GaussianDistribution(6.0, 1.0)], [0.5, 0.5]
        )
        with self.assertRaises(ValueError) as caught:
            project(source, S.MixtureEstimator([S.GaussianEstimator()] * 8), n_samples=1, seed=0)
        self.assertIn("8-component", str(caught.exception))
        self.assertIn("1 draw(s)", str(caught.exception))
        self.assertIsNotNone(
            _quiet(lambda: project(source, S.MixtureEstimator([S.GaussianEstimator()] * 2), n_samples=400, seed=0))
        )

    def test_an_empty_evaluation_split_is_named_as_one(self):
        from mixle.stats import seq_encode
        from mixle.utils.evaluation import empirical_kl_divergence

        law = S.GaussianDistribution(0.0, 1.0)
        with self.assertRaises(ValueError) as caught:
            empirical_kl_divergence(law, S.GaussianDistribution(0.5, 1.0), seq_encode([], model=law))
        self.assertIn("scored 0 observations", str(caught.exception))

    def test_a_joint_mixture_scores_an_empty_encoded_batch_as_no_rows(self):
        from mixle.stats import seq_encode
        from mixle.stats.latent.joint_mixture import JointMixtureDistribution

        model = JointMixtureDistribution(
            [S.GaussianDistribution(-2.0, 1.0), S.GaussianDistribution(2.0, 1.0)],
            [S.GaussianDistribution(-1.0, 1.0), S.GaussianDistribution(1.0, 1.0)],
            joint_weights=[[0.4, 0.1], [0.1, 0.4]],
        )
        scores = model.seq_log_density(seq_encode([], model=model)[0][1])
        self.assertEqual(tuple(np.shape(scores)), (0,))

    def test_a_combinator_over_an_empty_categorical_is_still_a_usable_component(self):
        """What a component that won no responsibility re-estimates to must stay composable."""
        from mixle.stats.combinator.composite import CompositeDistribution
        from mixle.stats.combinator.sequence import SequenceDistribution
        from mixle.stats.compute.pdist import DensitySemantics

        vacuous = S.CategoricalDistribution({})
        self.assertIs(vacuous.density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)
        self.assertTrue(vacuous.composable_as_component())
        composite = CompositeDistribution([vacuous, S.GaussianDistribution(0.0, 1.0)])
        self.assertIs(composite.density_semantics(), DensitySemantics.LIKELIHOOD_FACTOR)
        self.assertTrue(composite.composable_as_component())
        sequence = SequenceDistribution(S.CategoricalDistribution({"a": 1.0}), len_dist=vacuous)
        self.assertTrue(sequence.composable_as_component())
        self.assertEqual(
            len(
                S.MixtureDistribution(
                    [
                        composite,
                        CompositeDistribution(
                            [S.CategoricalDistribution({"a": 1.0}), S.GaussianDistribution(0.0, 1.0)]
                        ),
                    ],
                    [0.5, 0.5],
                ).components
            ),
            2,
        )

    def test_a_sequence_law_with_no_length_model_is_still_refused_with_its_own_advice(self):
        from mixle.stats.combinator.sequence import SequenceDistribution

        scoring_only = SequenceDistribution(S.CategoricalDistribution({"a": 1.0}))
        self.assertFalse(scoring_only.composable_as_component())
        with self.assertRaises(TypeError) as caught:
            S.MixtureDistribution([scoring_only, scoring_only], [0.5, 0.5])
        self.assertIn("len_dist", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
