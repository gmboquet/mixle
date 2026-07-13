"""Tests for the cross-layer API-consistency fixes (review ledger Part IV).

Covers: ``seed=`` on the fit verbs with int-``rng`` coercion (S-1); ``log_density`` /
``sample(size=)`` aliases on ``ppl.RandomVariable`` (S-2); mixture length and categorical
non-negativity constructor validation (S-3); empty-data ``ValueError`` in ``optimize`` and the
``raise Exception`` -> ``raise ValueError`` narrowing (S-4); scalar ``pseudo_count`` broadcast in
tuple-arity estimators (S-6); GP ``fit`` returning the model (S-7); the ``ppl.GaussianObs``
co-export (S-8); the HMM ``components=`` alias (S-9); pair-copula ``log_density`` (S-10); the
double-RNG ``TypeError`` in ``mixle.stats.sample`` (S-13); and the ``register_encoded_data_backend``
re-export (S-14).
"""

import importlib.util
import unittest

import numpy as np
from numpy.random import RandomState

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _gaussian_data(n: int = 60, seed: int = 7) -> list:
    return list(RandomState(seed).randn(n))


class OptimizeSeedAliasTestCase(unittest.TestCase):
    """S-1: ``optimize``/``fit``/``best_of`` accept ``seed=`` and coerce an integer ``rng``."""

    def test_seed_is_accepted_and_deterministic(self):
        from mixle.inference import optimize
        from mixle.stats import GaussianEstimator

        data = _gaussian_data()
        m1 = optimize(data, GaussianEstimator(), seed=3, max_its=3)
        m2 = optimize(data, GaussianEstimator(), seed=3, max_its=3)
        self.assertEqual(str(m1), str(m2))

    def test_int_rng_is_coerced_like_seed(self):
        from mixle.inference import optimize
        from mixle.stats import GaussianEstimator

        data = _gaussian_data()
        m_seed = optimize(data, GaussianEstimator(), seed=3, max_its=3)
        m_rng_int = optimize(data, GaussianEstimator(), rng=3, max_its=3)
        m_rng = optimize(data, GaussianEstimator(), rng=RandomState(3), max_its=3)
        self.assertEqual(str(m_seed), str(m_rng_int))
        self.assertEqual(str(m_seed), str(m_rng))

    def test_seed_and_rng_together_raise(self):
        from mixle.inference import optimize
        from mixle.stats import GaussianEstimator

        with self.assertRaises(TypeError):
            optimize(_gaussian_data(), GaussianEstimator(), seed=3, rng=RandomState(0))

    def test_fit_threads_seed(self):
        from mixle.inference import fit, optimize
        from mixle.stats import GaussianEstimator

        data = _gaussian_data()
        m_fit = fit(data, GaussianEstimator(), seed=5, max_its=3)
        m_opt = optimize(data, GaussianEstimator(), rng=RandomState(5), max_its=3, reuse_estep_ll=False)
        self.assertEqual(str(m_fit), str(m_opt))
        with self.assertRaises(TypeError):
            fit(data, GaussianEstimator(), seed=5, rng=RandomState(0))

    def test_best_of_accepts_seed(self):
        from mixle.inference.estimation import best_of
        from mixle.stats import GaussianEstimator

        data = _gaussian_data()
        ll1, m1 = best_of(data, None, GaussianEstimator(), trials=2, max_its=3, init_p=0.5, delta=1e-6, seed=11)
        ll2, m2 = best_of(data, None, GaussianEstimator(), trials=2, max_its=3, init_p=0.5, delta=1e-6, rng=11)
        self.assertEqual(str(m1), str(m2))
        self.assertAlmostEqual(ll1, ll2)
        with self.assertRaises(TypeError):
            best_of(
                data,
                None,
                GaussianEstimator(),
                trials=1,
                max_its=2,
                init_p=0.5,
                delta=1e-6,
                rng=RandomState(0),
                seed=1,
            )


class RandomVariableDialectAliasTestCase(unittest.TestCase):
    """S-2: ``log_density``/``sample(size=)`` on ``ppl.RandomVariable`` match the stats dialect."""

    def test_log_density_delegates_to_log_prob(self):
        from mixle.ppl import Normal

        rv = Normal(0.0, 1.0)
        self.assertEqual(rv.log_density(0.3), rv.log_prob(0.3))
        xs = [0.1, -0.4, 2.0]
        self.assertTrue(np.allclose(rv.log_density(xs), rv.log_prob(xs)))

    def test_sample_size_alias(self):
        from mixle.ppl import Normal

        rv = Normal(0.0, 1.0)
        by_n = rv.sample(n=5, seed=1)
        by_size = rv.sample(size=5, seed=1)
        self.assertEqual(len(by_n), len(by_size))
        self.assertIs(type(by_n), type(by_size))
        self.assertTrue(np.allclose(np.asarray(by_n), np.asarray(by_size)))

    def test_sample_double_supply_raises(self):
        from mixle.ppl import Normal

        with self.assertRaises(TypeError):
            Normal(0.0, 1.0).sample(n=5, size=5)


class CompositeCtorValidationTestCase(unittest.TestCase):
    """S-3: composite constructors reject silently-wrong parameterizations."""

    def test_mixture_component_weight_length_mismatch_raises(self):
        from mixle.stats.latent.mixture import MixtureDistribution
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        with self.assertRaises(ValueError):
            MixtureDistribution([GaussianDistribution(0.0, 1.0)], [0.5, 0.5])

    def test_mixture_matched_lengths_still_construct(self):
        from mixle.stats.latent.mixture import MixtureDistribution
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        m = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.4, 0.6])
        self.assertTrue(np.isfinite(m.log_density(0.5)))

    def test_categorical_negative_probability_raises(self):
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        with self.assertRaises(ValueError):
            CategoricalDistribution({"a": -0.5, "b": 1.5})

    def test_categorical_nonnegative_still_constructs(self):
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        c = CategoricalDistribution({"a": 0.6, "b": 0.4})
        self.assertAlmostEqual(c.density("a"), 0.6)


class EmptyDataAndValueErrorNarrowingTestCase(unittest.TestCase):
    """S-4: empty-data misuse raises ``ValueError``; former bare-``Exception`` sites narrowed."""

    def test_optimize_empty_data_raises_value_error(self):
        from mixle.inference import optimize
        from mixle.stats import GaussianEstimator

        with self.assertRaises(ValueError):
            optimize([], GaussianEstimator())

    def test_optimize_none_data_raises_value_error_with_message(self):
        from mixle.inference import optimize
        from mixle.stats import GaussianEstimator

        with self.assertRaises(ValueError) as ctx:
            optimize(None, GaussianEstimator())
        self.assertEqual(str(ctx.exception), "Optimization called with empty data or enc_data.")

    def test_fit_none_data_raises_value_error(self):
        from mixle.inference import fit
        from mixle.stats import GaussianEstimator

        with self.assertRaises(ValueError) as ctx:
            fit(None, GaussianEstimator())
        self.assertEqual(str(ctx.exception), "fit called with empty data or enc_data.")

    def test_former_exception_sites_now_raise_value_error(self):
        # representative narrowed sites, message-preserving (S-4)
        from mixle.stats.compute.sequence import seq_encode
        from mixle.stats.latent.dirac_length import DiracLengthMixtureDistribution
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        with self.assertRaises(ValueError) as ctx:
            seq_encode([1.0, 2.0])  # no encoder/estimator/dist supplied
        self.assertEqual(str(ctx.exception), "At least one arg: encoder, estimator, or dist must be passed.")

        with self.assertRaises(ValueError) as ctx:
            GaussianDistribution(0.0, 1.0).dist_to_encoder().seq_encode([1.0, float("nan")])
        self.assertEqual(str(ctx.exception), "GaussianDistribution requires support x in (-inf,inf).")

        with self.assertRaises(ValueError) as ctx:
            DiracLengthMixtureDistribution(len_dist=GaussianDistribution(0.0, 1.0), p=1.5, v=0)
        self.assertEqual(str(ctx.exception), "p must be between (0,1].")


class PseudoCountScalarBroadcastTestCase(unittest.TestCase):
    """S-6: tuple-arity estimator ctors accept a scalar ``pseudo_count`` like the facades."""

    def test_gaussian_estimator_scalar_equals_tuple(self):
        from mixle.stats import GaussianEstimator

        e_scalar = GaussianEstimator(pseudo_count=1.0, suff_stat=(0.0, 1.0))
        e_tuple = GaussianEstimator(pseudo_count=(1.0, 1.0), suff_stat=(0.0, 1.0))
        self.assertEqual(e_scalar.pseudo_count, e_tuple.pseudo_count)

    def test_gaussian_estimator_scalar_fit_matches_tuple_fit(self):
        from mixle.inference import optimize
        from mixle.stats import GaussianEstimator

        data = _gaussian_data()
        m_scalar = optimize(data, GaussianEstimator(pseudo_count=1.0, suff_stat=(0.0, 1.0)), max_its=3)
        m_tuple = optimize(data, GaussianEstimator(pseudo_count=(1.0, 1.0), suff_stat=(0.0, 1.0)), max_its=3)
        self.assertEqual(str(m_scalar), str(m_tuple))

    def test_all_tuple_arity_estimators_broadcast(self):
        from mixle.stats.latent.hidden_markov import HiddenMarkovEstimator
        from mixle.stats.latent.joint_mixture import JointMixtureEstimator
        from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianEstimator
        from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianEstimator
        from mixle.stats.univariate.continuous.gamma import GammaEstimator
        from mixle.stats.univariate.continuous.gaussian import GaussianEstimator
        from mixle.stats.univariate.continuous.log_gaussian import LogGaussianEstimator

        for cls in (GaussianEstimator, LogGaussianEstimator, DiagonalGaussianEstimator):
            self.assertEqual(cls(pseudo_count=2.0).pseudo_count, (2.0, 2.0), cls.__name__)
        self.assertEqual(MultivariateGaussianEstimator(pseudo_count=2.0).pseudo_count, (2.0, 2.0))
        self.assertEqual(GammaEstimator(pseudo_count=2.0).pseudo_count, (2.0, 2.0))
        self.assertEqual(
            HiddenMarkovEstimator(estimators=[GaussianEstimator(), GaussianEstimator()], pseudo_count=2.0).pseudo_count,
            (2.0, 2.0),
        )
        self.assertEqual(
            JointMixtureEstimator(
                estimators1=[GaussianEstimator()], estimators2=[GaussianEstimator()], pseudo_count=2.0
            ).pseudo_count,
            (2.0, 2.0, 2.0),
        )

    def test_tuple_and_none_pass_through_unchanged(self):
        from mixle.stats import GaussianEstimator
        from mixle.stats.latent.lda import LDAEstimator

        self.assertEqual(GaussianEstimator(pseudo_count=(1.0, None)).pseudo_count, (1.0, None))
        self.assertIsNone(LDAEstimator(estimators=[GaussianEstimator()]).pseudo_count)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class GaussianProcessFitReturnTestCase(unittest.TestCase):
    """S-7: GP ``fit`` returns the model; the tuple lives behind ``return_result=True``."""

    def test_fit_returns_self(self):
        from mixle.models import GaussianProcessRegressor

        x = np.linspace(-1.0, 1.0, 12)[:, None]
        y = np.sin(x[:, 0])
        gp = GaussianProcessRegressor(lengthscale=0.4, amplitude=0.6, noise=0.4)
        self.assertIs(gp.fit(x, y, max_its=10, out=None), gp)

    def test_return_result_path_unchanged(self):
        from mixle.models import GaussianProcessRegressor

        x = np.linspace(-1.0, 1.0, 12)[:, None]
        y = np.sin(x[:, 0])
        gp = GaussianProcessRegressor(lengthscale=0.4, amplitude=0.6, noise=0.4)
        result = gp.fit(x, y, max_its=10, out=None, return_result=True)
        self.assertTrue(np.isfinite(result.value))
        self.assertGreater(result.iterations, 0)


class PplGaussianObsExportTestCase(unittest.TestCase):
    """S-8: the linear-Gaussian observation helper is co-exported as ``GaussianObs``."""

    def test_gaussian_obs_is_gaussian(self):
        from mixle.ppl import Gaussian, GaussianObs

        self.assertIs(GaussianObs, Gaussian)

    def test_normal_is_still_the_distribution(self):
        from mixle.ppl import Normal

        rv = Normal(0.0, 1.0)
        self.assertTrue(np.isfinite(rv.log_prob(0.0)))


class HmmComponentsAliasTestCase(unittest.TestCase):
    """S-9: ``HiddenMarkovModelDistribution`` accepts ``components=`` for ``topics=``."""

    def _emissions(self):
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution

        return [CategoricalDistribution({"a": 1.0}), CategoricalDistribution({"a": 0.5, "b": 0.5})]

    def test_components_alias_builds_same_model(self):
        from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

        comps = self._emissions()
        trans = [[0.5, 0.5], [0.2, 0.8]]
        by_topics = HiddenMarkovModelDistribution(comps, [0.5, 0.5], trans)
        by_components = HiddenMarkovModelDistribution(components=comps, w=[0.5, 0.5], transitions=trans)
        self.assertTrue(np.allclose(by_topics.w, by_components.w))
        self.assertTrue(np.allclose(by_topics.transitions, by_components.transitions))
        self.assertIs(by_components.topics, comps)
        seq = ["a", "b", "a"]
        self.assertAlmostEqual(by_topics.log_density(seq), by_components.log_density(seq))

    def test_double_supply_raises(self):
        from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

        comps = self._emissions()
        trans = [[0.5, 0.5], [0.2, 0.8]]
        with self.assertRaises(TypeError):
            HiddenMarkovModelDistribution(comps, [0.5, 0.5], trans, components=comps)
        with self.assertRaises(TypeError):
            HiddenMarkovModelDistribution(w=[0.5, 0.5], transitions=trans)  # neither supplied


class PairCopulaLogDensityTestCase(unittest.TestCase):
    """S-10: the six pair copulas answer the library-wide ``log_density`` verb."""

    def test_log_density_equals_logpdf_for_all_six(self):
        from mixle.stats.multivariate.vine_copula import (
            ClaytonPairCopula,
            FrankPairCopula,
            GaussianPairCopula,
            GumbelPairCopula,
            IndependencePairCopula,
            StudentTPairCopula,
        )

        a = np.array([0.2, 0.5, 0.9])
        b = np.array([0.3, 0.7, 0.4])
        copulas = [
            IndependencePairCopula(),
            GaussianPairCopula(rho=0.4),
            ClaytonPairCopula(theta=1.5),
            FrankPairCopula(theta=2.0),
            GumbelPairCopula(theta=1.7),
            StudentTPairCopula(rho=0.3, df=5.0),
        ]
        for pc in copulas:
            self.assertTrue(np.allclose(pc.log_density(a, b), pc.logpdf(a, b)), type(pc).__name__)


class StatsSampleDoubleRngTestCase(unittest.TestCase):
    """S-13: ``mixle.stats.sample`` rejects two randomness sources."""

    def test_double_rng_raises_type_error(self):
        import mixle.stats as stats
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        g = GaussianDistribution(0.0, 1.0)
        with self.assertRaises(TypeError):
            stats.sample(g, 3, seed=1, rng=RandomState(1))

    def test_single_source_still_works(self):
        import mixle.stats as stats
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        g = GaussianDistribution(0.0, 1.0)
        self.assertEqual(len(stats.sample(g, 3, seed=1)), 3)
        self.assertEqual(len(stats.sample(g, 3, rng=RandomState(1))), 3)


class ParallelBackendReexportTestCase(unittest.TestCase):
    """S-14: ``register_encoded_data_backend`` is importable as the README advertises."""

    def test_reexport(self):
        from mixle.utils.parallel import register_encoded_data_backend
        from mixle.utils.parallel.planner import register_encoded_data_backend as planner_fn

        self.assertIs(register_encoded_data_backend, planner_fn)


if __name__ == "__main__":
    unittest.main()
