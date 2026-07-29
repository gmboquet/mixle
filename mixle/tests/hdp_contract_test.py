import unittest

import numpy as np
from numpy.random import RandomState

from mixle.stats.bayes.hierarchical_dirichlet_process_mixture import (
    HDPGroup,
    HierarchicalDirichletProcessMixtureAccumulator,
    HierarchicalDirichletProcessMixtureDistribution,
    HierarchicalDirichletProcessMixtureEstimator,
)
from mixle.stats.combinator.null_dist import NullDistribution
from mixle.stats.univariate.continuous.gaussian import (
    GaussianAccumulator,
    GaussianDistribution,
    GaussianEstimator,
)
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution


class _FixedScoreGaussian(GaussianDistribution):
    def __init__(self, score):
        super().__init__(0.0, 1.0)
        self.score = float(score)

    def log_density(self, x):
        return self.score

    def seq_log_density(self, x):
        return np.full(len(x), self.score)


def _model(scores=(0.0, -1.0), beta=(0.5, 0.5)):
    return HierarchicalDirichletProcessMixtureDistribution(
        [_FixedScoreGaussian(score) for score in scores],
        beta,
        alpha=2.0,
        gamma=1.5,
    )


class HierarchicalDirichletProcessStateContractTestCase(unittest.TestCase):
    def test_constructor_rejects_invalid_nested_geometry(self):
        gaussian = GaussianDistribution(0.0, 1.0)
        cases = (
            ([], [], 1.0, 1.0),
            ([gaussian], [0.5, 0.5], 1.0, 1.0),
            ([gaussian], [np.nan], 1.0, 1.0),
            ([gaussian], [0.5], 1.0, 1.0),
            ([gaussian], [1.0], 0.0, 1.0),
            ([gaussian], [1.0], 1.0, np.inf),
        )
        for args in cases:
            with self.subTest(args=repr(args)), self.assertRaises((TypeError, ValueError)):
                HierarchicalDirichletProcessMixtureDistribution(*args)
        with self.assertRaises(ValueError):
            HierarchicalDirichletProcessMixtureDistribution(
                [GaussianDistribution(0.0, 1.0), PoissonDistribution(1.0)],
                [0.5, 0.5],
                1.0,
                1.0,
            )

    def test_fitted_group_state_requires_ids_and_row_stochastic_support(self):
        components = [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)]
        cases = (
            {"group_weights": [[0.5, 0.5]]},
            {"group_weights": [[0.5]], "group_ids": ["a"]},
            {"group_weights": [[0.5, np.nan]], "group_ids": ["a"]},
            {"group_weights": [[0.2, 0.2]], "group_ids": ["a"]},
            {"group_weights": [[0.5, 0.5]], "group_ids": ["a", "b"]},
            {
                "group_weights": [[0.5, 0.5], [0.4, 0.6]],
                "group_ids": ["a", "a"],
            },
        )
        for kwargs in cases:
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises((TypeError, ValueError)):
                HierarchicalDirichletProcessMixtureDistribution(
                    components,
                    [0.5, 0.5],
                    1.0,
                    1.0,
                    **kwargs,
                )
        with self.assertRaises(ValueError):
            HierarchicalDirichletProcessMixtureDistribution(
                components,
                [1.0, 0.0],
                1.0,
                1.0,
                group_weights=[[0.5, 0.5]],
                group_ids=["a"],
            )

    def test_set_parameters_rejects_partial_updates_atomically(self):
        model = _model()
        before = model.log_density([0.0])
        with self.assertRaises(ValueError):
            model.set_parameters(([0.5, 0.5], 2.0, 1.5, [_FixedScoreGaussian(4.0)]))
        self.assertEqual(model.log_density([0.0]), before)
        model.set_parameters(
            (
                [0.25, 0.75],
                3.0,
                2.0,
                [_FixedScoreGaussian(2.0), _FixedScoreGaussian(3.0)],
            )
        )
        np.testing.assert_allclose(model.beta, [0.25, 0.75])
        self.assertEqual(model.alpha, 3.0)
        model.components.append(_FixedScoreGaussian(0.0))
        with self.assertRaisesRegex(RuntimeError, "structure changed"):
            model.log_density([0.0])

    def test_estimator_rejects_invalid_concentrations_and_empty_atoms(self):
        for args in (
            ([], 1.0, 1.0),
            ([GaussianEstimator()], 0.0, 1.0),
            ([GaussianEstimator()], 1.0, np.nan),
        ):
            with self.subTest(args=repr(args)), self.assertRaises((TypeError, ValueError)):
                HierarchicalDirichletProcessMixtureEstimator(
                    args[0],
                    gamma=args[1],
                    alpha=args[2],
                )

    def test_string_round_trip_preserves_group_identity(self):
        model = HierarchicalDirichletProcessMixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 2.0)],
            [0.4, 0.6],
            1.3,
            2.1,
            group_weights=[[0.2, 0.8]],
            group_ids=["training-group"],
            name="hdp",
        )
        restored = eval(
            str(model),
            {
                "HierarchicalDirichletProcessMixtureDistribution": (HierarchicalDirichletProcessMixtureDistribution),
                "GaussianDistribution": GaussianDistribution,
                "NullDistribution": NullDistribution,
            },
        )
        np.testing.assert_allclose(restored.beta, model.beta)
        np.testing.assert_allclose(restored.group_weights, model.group_weights)
        self.assertEqual(restored.group_ids, model.group_ids)
        self.assertEqual(restored.name, model.name)


class HierarchicalDirichletProcessEvidenceContractTestCase(unittest.TestCase):
    def test_global_and_local_zero_weights_remain_impossible(self):
        global_model = _model(scores=(0.0, np.inf), beta=(1.0, 0.0))
        self.assertEqual(global_model.log_density([0.0]), 0.0)
        np.testing.assert_array_equal(global_model.group_posteriors([[0.0]]), [[1.0, 0.0]])

        local_group = HDPGroup("local", [0.0])
        local_model = HierarchicalDirichletProcessMixtureDistribution(
            [_FixedScoreGaussian(0.0), _FixedScoreGaussian(np.inf)],
            [0.5, 0.5],
            1.0,
            1.0,
            group_weights=[[1.0, 0.0]],
            group_ids=["local"],
        )
        encoded = local_model.dist_to_encoder().seq_encode([local_group])
        self.assertEqual(local_model.seq_local_elbo(encoded)[0], 0.0)
        np.testing.assert_array_equal(local_model.group_posteriors([local_group]), [[1.0, 0.0]])

    def test_sampler_never_revives_globally_zero_atom(self):
        model = HierarchicalDirichletProcessMixtureDistribution(
            [CategoricalDistribution({"left": 1.0}), CategoricalDistribution({"right": 1.0})],
            [1.0, 0.0],
            2.0,
            1.0,
        )
        draws = model.sampler(seed=3).sample_group(200)
        self.assertEqual(set(draws), {"left"})

    def test_impossible_observation_has_zero_posterior_and_statistics(self):
        model = _model(scores=(-np.inf, -np.inf))
        np.testing.assert_array_equal(model.group_posteriors([[0.0]]), [[0.0, 0.0]])
        self.assertTrue(np.isneginf(model.log_density([0.0])))
        accumulator = HierarchicalDirichletProcessMixtureAccumulator([GaussianAccumulator(), GaussianAccumulator()])
        encoded = model.dist_to_encoder().seq_encode([HDPGroup("g", [0.0])])
        accumulator.seq_update(encoded, np.asarray([2.0]), model)
        counts = next(iter(accumulator.group_counts.values()))
        np.testing.assert_array_equal(counts, np.zeros(2))


class HierarchicalDirichletProcessIdentityContractTestCase(unittest.TestCase):
    def setUp(self):
        self.model = HierarchicalDirichletProcessMixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
            2.0,
            1.0,
        )

    def make_accumulator(self):
        return HierarchicalDirichletProcessMixtureAccumulator([GaussianAccumulator(), GaussianAccumulator()])

    def test_duplicate_content_requires_explicit_group_ids(self):
        encoder = self.model.dist_to_encoder()
        with self.assertRaisesRegex(ValueError, "HDPGroup"):
            encoder.seq_encode([[0.0], [0.0]])
        encoded = encoder.seq_encode([HDPGroup("first", [0.0]), HDPGroup("second", [0.0])])
        self.assertEqual(encoder.row_count(encoded), 2)

    def test_merge_is_order_independent_and_preserves_group_ids(self):
        groups = [HDPGroup("a", [-1.0]), HDPGroup("b", [1.0])]
        encoded = [self.model.dist_to_encoder().seq_encode([group]) for group in groups]
        left = self.make_accumulator()
        right = self.make_accumulator()
        left.seq_update(encoded[0], np.ones(1), self.model)
        right.seq_update(encoded[1], np.ones(1), self.model)

        lr = self.make_accumulator().combine(left.value()).combine(right.value())
        rl = self.make_accumulator().combine(right.value()).combine(left.value())
        self.assertEqual(set(lr.group_counts), set(rl.group_counts))
        for group_id in lr.group_counts:
            np.testing.assert_allclose(lr.group_counts[group_id], rl.group_counts[group_id])
        self.assertEqual(lr.model_version, rl.model_version)

        estimator = HierarchicalDirichletProcessMixtureEstimator(
            [GaussianEstimator(), GaussianEstimator()],
            gamma=self.model.gamma,
            alpha=self.model.alpha,
        )
        fitted_lr = estimator.estimate(None, lr.value())
        fitted_rl = estimator.estimate(None, rl.value())
        np.testing.assert_allclose(fitted_lr.beta, fitted_rl.beta)
        for group in groups:
            np.testing.assert_allclose(
                fitted_lr.group_weight(group),
                fitted_rl.group_weight(group),
            )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            lr.combine(left.value())

    def test_batch_weight_geometry_is_validated_before_mutation(self):
        accumulator = self.make_accumulator()
        encoded = self.model.dist_to_encoder().seq_encode([HDPGroup("a", [0.0]), HDPGroup("b", [1.0])])
        before = accumulator.value()
        for weights in (
            np.ones(1),
            np.asarray([1.0, -1.0]),
            np.asarray([1.0, np.nan]),
        ):
            with self.subTest(weights=repr(weights)), self.assertRaises(ValueError):
                accumulator.seq_update(encoded, weights, self.model)
            self.assertEqual(accumulator.value()[0], before[0])

    def test_merge_rejects_different_model_versions(self):
        group_a = HDPGroup("a", [0.0])
        group_b = HDPGroup("b", [1.0])
        first = self.make_accumulator()
        first.seq_update(
            self.model.dist_to_encoder().seq_encode([group_a]),
            np.ones(1),
            self.model,
        )
        other_model = HierarchicalDirichletProcessMixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.8, 0.2],
            2.0,
            1.0,
        )
        second = self.make_accumulator()
        second.seq_update(
            other_model.dist_to_encoder().seq_encode([group_b]),
            np.ones(1),
            other_model,
        )
        with self.assertRaisesRegex(ValueError, "model versions"):
            first.combine(second.value())


class HierarchicalDirichletProcessEstimatorContractTestCase(unittest.TestCase):
    def test_estimator_rejects_malformed_serialized_state(self):
        estimator = HierarchicalDirichletProcessMixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        accumulator = estimator.accumulator_factory().make()
        encoded = accumulator.acc_to_encoder().seq_encode([HDPGroup("g", [0.0])])
        accumulator.seq_initialize(encoded, np.ones(1), RandomState(2))
        valid = accumulator.value()
        malformed = (
            ({"g": [1.0]}, None, None, None, valid[4], valid[5]),
            ({"g": [1.0, -1.0]}, None, None, None, valid[4], valid[5]),
            ({"g": [1.0, np.nan]}, None, None, None, valid[4], valid[5]),
            (valid[0], [0.5, 0.5], None, None, valid[4], valid[5]),
            (valid[0], [0.5, 0.5], 0.0, "version", valid[4], valid[5]),
            (valid[0], None, None, None, valid[4][:-1], valid[5]),
        )
        for value in malformed:
            with self.subTest(value=repr(value)), self.assertRaises((TypeError, ValueError)):
                estimator.estimate(None, value)

    def test_zero_previous_beta_is_handled_without_flooring_or_nan(self):
        estimator = HierarchicalDirichletProcessMixtureEstimator(
            [GaussianEstimator(), GaussianEstimator()],
            gamma=1.0,
            alpha=2.0,
        )
        accumulator = estimator.accumulator_factory().make()
        encoded = accumulator.acc_to_encoder().seq_encode([HDPGroup("g", [0.0, 0.5])])
        accumulator.seq_initialize(encoded, np.ones(1), RandomState(3))
        value = list(accumulator.value())
        value[1] = np.asarray([1.0, 0.0])
        value[2] = 2.0
        value[3] = "validated-model-version"
        fitted = estimator.estimate(None, tuple(value))
        self.assertTrue(np.all(np.isfinite(fitted.beta)))
        self.assertTrue(np.all(fitted.beta > 0.0))
        self.assertAlmostEqual(float(fitted.beta.sum()), 1.0)
        self.assertTrue(fitted.fit_metadata["converged"])
        self.assertEqual(fitted.fit_metadata["repairs"], ())


if __name__ == "__main__":
    unittest.main()
