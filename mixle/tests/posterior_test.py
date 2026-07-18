"""The Posterior algebra: the posterior() factory over latent / parameter / predictive variables."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixle.inference import ParameterPosterior, PredictivePosterior, posterior
from mixle.inference.mcmc.parameter_bridge import sample_parameter_posterior
from mixle.stats import sample
from mixle.stats.compute.posterior import LatentPosterior, Posterior
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution
from mixle.utils.optional_deps import HAS_PANDAS
from mixle.utils.optional_deps import pandas as pd

_SKIP_NO_PANDAS = unittest.skipUnless(HAS_PANDAS, "pandas not installed; pip install mixle[pandas]")


class PredictivePosteriorTest(unittest.TestCase):
    def test_plug_in_predictive_matches_sampler(self):
        g = GaussianDistribution(2.0, 1.0)
        post = posterior(g, over="predictive")
        self.assertIsInstance(post, PredictivePosterior)
        self.assertIsInstance(post, Posterior)
        draws = np.asarray(post.samples(5000, rng=np.random.RandomState(0)), dtype=float)
        self.assertAlmostEqual(draws.mean(), 2.0, delta=0.1)  # recovers the model mean
        self.assertTrue(np.isscalar(post.sample(rng=np.random.RandomState(1))) or np.ndim(post.sample(0)) == 0)


class ParameterPosteriorConjugateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = (rng.random_sample(400) < 0.7).astype(int).tolist()  # Bernoulli(0.7)
        self.post = posterior(BernoulliDistribution(0.5), self.data, over="params", prior={"a": 1.0, "b": 1.0})

    def test_factory_returns_parameter_posterior(self):
        self.assertIsInstance(self.post, ParameterPosterior)
        self.assertEqual(self.post.kind, "conjugate")

    def test_mean_recovers_parameter(self):
        m = self.post.mean()
        # Beta posterior mean of p concentrates near the data frequency (~0.7)
        p = float(np.ravel(list(m.values())[0])[0]) if isinstance(m, dict) else float(np.ravel(m)[0])
        self.assertAlmostEqual(p, 0.7, delta=0.06)

    def test_sample_and_interval(self):
        one = self.post.sample(rng=np.random.RandomState(1))
        self.assertIsInstance(one, dict)  # a parameter set
        ci = self.post.interval(0.9)
        self.assertIsInstance(ci, dict)
        for lo_hi in ci.values():
            self.assertEqual(np.asarray(lo_hi).shape[0], 2)  # [lo, hi]

    def test_auto_picks_conjugate(self):
        self.assertEqual(posterior(BernoulliDistribution(0.5), self.data, over="params").kind, "conjugate")

    @_SKIP_NO_PANDAS
    def test_to_dataframe_matches_analytic_beta_posterior(self):
        # Beta(1, 1) prior + 400 Bernoulli(0.7) draws (289 successes, seed 0) -> Beta(290, 112) exactly;
        # to_dataframe(n, rng) must equal an independent rng.beta(290, 112, size=n) draw column-for-column.
        s = int(sum(self.data))
        a, b = 1.0 + s, 1.0 + len(self.data) - s
        self.assertEqual((a, b), (290.0, 112.0))  # pins the fixture so a future data change is caught here
        df = self.post.to_dataframe(25, rng=np.random.RandomState(2))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["p"])
        self.assertEqual(df.shape, (25, 1))
        expected = np.random.RandomState(2).beta(a, b, size=25)
        np.testing.assert_array_equal(df["p"].to_numpy(), expected)
        self.assertTrue(bool((df["p"] > 0.0).all() and (df["p"] < 1.0).all()))  # Beta(290, 112) support

    @_SKIP_NO_PANDAS
    def test_to_parquet_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "beta_posterior.parquet"
            self.post.to_parquet(path, 15, rng=np.random.RandomState(3))
            roundtrip = pd.read_parquet(path)
            expected = self.post.to_dataframe(15, rng=np.random.RandomState(3))
            pd.testing.assert_frame_equal(roundtrip, expected)


class ParameterPosteriorMCMCTest(unittest.TestCase):
    def test_mcmc_path_when_forced(self):
        rng = np.random.RandomState(0)
        data = (rng.normal(3.0, 1.0, size=200)).tolist()
        post = posterior(
            GaussianDistribution(0.0, 1.0), data, over="params", method="mcmc", steps=300, burn_in=100, seed=0
        )
        self.assertIsInstance(post, ParameterPosterior)
        self.assertEqual(post.kind, "mcmc")
        draws = post.samples(10, rng=np.random.RandomState(1))
        self.assertEqual(len(draws), 10)
        mean = np.asarray(post.mean(), dtype=float)
        self.assertAlmostEqual(float(mean[0]), 3.0, delta=0.5)  # posterior mean of mu near 3

    @_SKIP_NO_PANDAS
    def test_to_dataframe_two_param_family_gets_positional_columns(self):
        # GaussianDistribution's parameter_bridge maps theta -> the tuple (mu, sigma2) -- no names
        # travel through MCMCResult, so to_dataframe() must fall back to generic param_0/param_1,
        # column-for-column identical to a direct call to .samples() with the same rng.
        rng = np.random.RandomState(0)
        data = (rng.normal(3.0, 1.0, size=200)).tolist()
        post = posterior(
            GaussianDistribution(0.0, 1.0), data, over="params", method="mcmc", steps=300, burn_in=100, seed=0
        )
        df = post.to_dataframe(20, rng=np.random.RandomState(3))
        self.assertEqual(list(df.columns), ["param_0", "param_1"])
        self.assertEqual(df.shape, (20, 2))
        expected = np.asarray(post.samples(20, rng=np.random.RandomState(3)), dtype=float)
        np.testing.assert_array_equal(df.to_numpy(), expected)

    @_SKIP_NO_PANDAS
    def test_to_dataframe_scalar_family_gets_single_column(self):
        # PoissonDistribution's bridge maps theta -> a bare float (not even a 1-tuple); to_dataframe()
        # must still produce exactly one column, matching the raw samples exactly.
        rng = np.random.RandomState(0)
        data = rng.poisson(4.0, size=200).tolist()
        post = posterior(PoissonDistribution(1.0), data, over="params", method="mcmc", steps=300, burn_in=100, seed=0)
        df = post.to_dataframe(12, rng=np.random.RandomState(4))
        self.assertEqual(list(df.columns), ["param_0"])
        self.assertEqual(df.shape, (12, 1))
        expected = np.asarray(post.samples(12, rng=np.random.RandomState(4)), dtype=float)
        np.testing.assert_array_equal(df["param_0"].to_numpy(), expected)
        self.assertTrue(bool((df["param_0"] > 0.0).all()))  # Poisson rate posterior support

    @_SKIP_NO_PANDAS
    def test_to_dataframe_rejects_rebuilt_distribution_samples(self):
        # return_distributions=True maps each MCMC draw to a rebuilt distribution object -- there is no
        # natural tabular form for that, so to_dataframe() must refuse rather than fabricate columns.
        rng = np.random.RandomState(0)
        data = (rng.normal(3.0, 1.0, size=50)).tolist()
        result = sample_parameter_posterior(
            GaussianDistribution(0.0, 1.0), data, steps=50, burn_in=10, seed=0, return_distributions=True
        )
        post = ParameterPosterior.from_mcmc(result)
        with self.assertRaises(TypeError):
            post.to_dataframe(5, rng=np.random.RandomState(0))


class LatentPosteriorFactoryTest(unittest.TestCase):
    def test_over_latent_dispatches_to_model(self):
        m = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5])
        x = [-3.0, -2.8, 3.1, 2.9]
        post = posterior(m, x, over="latent")
        self.assertIsInstance(post, LatentPosterior)
        self.assertIsInstance(post, Posterior)
        labels = post.mode()
        self.assertEqual(list(labels), [0, 0, 1, 1])  # clear assignment to the two components

    def test_unsupported_over_raises(self):
        with self.assertRaises(ValueError):
            posterior(GaussianDistribution(0.0, 1.0), over="nonsense")
        with self.assertRaises(TypeError):
            posterior(GaussianDistribution(0.0, 1.0), [1.0], over="latent")  # Gaussian has no latent_posterior


class FacadeStillWorksTest(unittest.TestCase):
    def test_sample_facade_unchanged(self):
        g = GaussianDistribution(0.0, 1.0)
        draws = sample(g, 100, seed=0)
        self.assertEqual(len(np.asarray(draws)), 100)


if __name__ == "__main__":
    unittest.main()
