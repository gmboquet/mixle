"""Universal correctness tests for the exponential-family canonical map.

The reconstruction test validates *every* family's spec at once: for random x,
``log h(x) + <eta, T(x)> - A(eta)`` must equal the family's own ``log_density``.
A wrong eta/T/A/h in any family fails here -- this retro-tests the existing leaf
specs as well as the newly added inverse_gamma / von_mises / geometric and the
Composite / Sequence / conditional (GLM) closures.
"""

import unittest

import numpy as np

from pysp.ppl import Bernoulli, Field, Normal, Poisson, free
from pysp.stats.combinator.composite import CompositeDistribution
from pysp.stats.combinator.sequence import SequenceDistribution
from pysp.stats.directional.von_mises import VonMisesDistribution
from pysp.stats.exp_family import (
    ConditionalExponentialFamilyForm,
    ExponentialFamilyForm,
    is_exponential_family,
    to_exponential_family,
)
from pysp.stats.multivariate.categorical_multinomial import MultinomialDistribution
from pysp.stats.univariate.continuous.beta import BetaDistribution
from pysp.stats.univariate.continuous.exponential import ExponentialDistribution
from pysp.stats.univariate.continuous.gamma import GammaDistribution
from pysp.stats.univariate.continuous.gaussian import GaussianDistribution
from pysp.stats.univariate.continuous.half_normal import HalfNormalDistribution
from pysp.stats.univariate.continuous.inverse_gamma import InverseGammaDistribution
from pysp.stats.univariate.continuous.inverse_gaussian import InverseGaussianDistribution
from pysp.stats.univariate.continuous.log_gaussian import LogGaussianDistribution
from pysp.stats.univariate.continuous.rayleigh import RayleighDistribution
from pysp.stats.univariate.discrete.bernoulli import BernoulliDistribution
from pysp.stats.univariate.discrete.binomial import BinomialDistribution
from pysp.stats.univariate.discrete.geometric import GeometricDistribution
from pysp.stats.univariate.discrete.logseries import LogSeriesDistribution
from pysp.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution
from pysp.stats.univariate.discrete.poisson import PoissonDistribution


def _leaf_cases():
    rng = np.random.RandomState(0)
    return [
        (GaussianDistribution(1.3, 2.1), rng.normal(0.0, 2.0, 32)),
        (BernoulliDistribution(0.37), rng.randint(0, 2, 32).astype(float)),
        (BetaDistribution(2.0, 3.0), rng.uniform(0.05, 0.95, 32)),
        (BinomialDistribution(0.4, 10), rng.randint(0, 11, 32).astype(float)),
        (ExponentialDistribution(1.5), rng.exponential(0.7, 32)),
        (GammaDistribution(2.0, 1.3), rng.gamma(2.0, 0.8, 32)),
        (HalfNormalDistribution(1.7), np.abs(rng.normal(0.0, 1.7, 32))),
        (InverseGaussianDistribution(1.0, 2.0), rng.wald(1.0, 2.0, 32)),
        (LogGaussianDistribution(0.0, 1.0), np.exp(rng.normal(0.0, 1.0, 32))),
        (PoissonDistribution(3.2), rng.poisson(3.2, 32).astype(float)),
        (RayleighDistribution(1.2), rng.rayleigh(1.2, 32)),
        (InverseGammaDistribution(2.5, 1.7), 1.0 / rng.gamma(2.5, 1.0 / 1.7, 32)),
        (VonMisesDistribution(0.7, 2.3), rng.vonmises(0.7, 2.3, 32)),
        (GeometricDistribution(0.35), rng.geometric(0.35, 32).astype(float)),
        (NegativeBinomialDistribution(4.0, 0.6), rng.negative_binomial(4.0, 0.6, 32).astype(float)),
        (LogSeriesDistribution(0.6), rng.logseries(0.6, 32).astype(float)),
    ]


def _reference_log_density(dist, x):
    enc = dist.dist_to_encoder().seq_encode(list(x))
    return np.asarray(dist.seq_log_density(enc), dtype=np.float64)


class LeafExponentialFamilyTest(unittest.TestCase):
    def test_reconstruction(self):
        for dist, x in _leaf_cases():
            with self.subTest(dist=type(dist).__name__):
                form = to_exponential_family(dist)
                self.assertIsNotNone(form)
                self.assertIsInstance(form, ExponentialFamilyForm)
                self.assertTrue(is_exponential_family(dist))

                eta = form.natural_parameters()
                t = np.asarray(form.sufficient_statistics(x), dtype=np.float64)
                a = float(form.log_partition())
                h = np.asarray(form.log_base_measure(x), dtype=np.float64)

                self.assertEqual(t.shape[1], eta.shape[0])
                self.assertEqual(form.dim, eta.shape[0])

                recon = h + t @ eta - a
                ref = _reference_log_density(dist, x)
                np.testing.assert_allclose(recon, ref, atol=1e-9)

                # the form's own log_density agrees with the family log_density
                np.testing.assert_allclose(np.asarray(form.log_density(x)), ref, atol=1e-9)

    def test_mean_parameters_match_empirical(self):
        # Families with a closed-form dual map use the exact grad-A path; verify it
        # against a large empirical mean of T over independent samples.
        for dist in (
            InverseGammaDistribution(3.0, 2.0),
            VonMisesDistribution(0.4, 2.5),
            GeometricDistribution(0.3),
            LogSeriesDistribution(0.5),
        ):
            with self.subTest(dist=type(dist).__name__):
                form = to_exponential_family(dist)
                self.assertIsNotNone(form.from_natural(form.natural_parameters()))
                mp = form.mean_parameters()
                samples = dist.sampler(7).sample(200000)
                emp = np.asarray(form.sufficient_statistics(samples), dtype=np.float64).mean(axis=0)
                np.testing.assert_allclose(mp, emp, rtol=0.05, atol=0.05)

    def test_from_natural_round_trip(self):
        for dist in (
            InverseGammaDistribution(2.5, 1.7),
            VonMisesDistribution(0.7, 2.3),
            GeometricDistribution(0.35),
            LogSeriesDistribution(0.55),
        ):
            with self.subTest(dist=type(dist).__name__):
                form = to_exponential_family(dist)
                recovered = form.from_natural(form.natural_parameters())
                self.assertIsNotNone(recovered)
                np.testing.assert_allclose(
                    to_exponential_family(recovered).natural_parameters(),
                    form.natural_parameters(),
                    atol=1e-9,
                )


class CompositeExponentialFamilyTest(unittest.TestCase):
    def test_product_reconstruction(self):
        comp = CompositeDistribution((GaussianDistribution(1.0, 2.0), PoissonDistribution(3.0)))
        form = to_exponential_family(comp)
        self.assertIsNotNone(form)
        rng = np.random.RandomState(1)
        xs = [(float(rng.normal()), int(rng.poisson(3.0))) for _ in range(20)]

        eta = form.natural_parameters()
        t = np.asarray(form.sufficient_statistics(xs), dtype=np.float64)
        a = float(form.log_partition())
        h = np.asarray(form.log_base_measure(xs), dtype=np.float64)
        self.assertEqual(t.shape[1], eta.shape[0])

        recon = h + t @ eta - a
        ref = np.array([comp.log_density(x) for x in xs])
        np.testing.assert_allclose(recon, ref, atol=1e-9)
        np.testing.assert_allclose(np.asarray(form.log_density(xs)), ref, atol=1e-9)

    def test_non_exp_family_child_returns_none(self):
        from pysp.stats.univariate.continuous.laplace import LaplaceDistribution

        comp = CompositeDistribution((GaussianDistribution(0.0, 1.0), LaplaceDistribution(0.0, 1.0)))
        self.assertIsNone(to_exponential_family(comp))


class SequenceExponentialFamilyTest(unittest.TestCase):
    def test_iid_reconstruction(self):
        seq = SequenceDistribution(ExponentialDistribution(1.5), len_dist=None)
        form = to_exponential_family(seq)
        self.assertIsNotNone(form)
        rng = np.random.RandomState(2)
        seqs = [list(rng.exponential(0.7, rng.randint(1, 5))) for _ in range(15)]

        eta = form.natural_parameters()
        t = np.asarray(form.sufficient_statistics(seqs), dtype=np.float64)
        a = float(form.log_partition())
        ns = np.array([len(s) for s in seqs], dtype=np.float64)

        # h = 0 for Exponential; joint = <eta, sum_t T> - n * A
        recon = t @ eta - ns * a
        ref = np.array([seq.log_density(s) for s in seqs])
        np.testing.assert_allclose(recon, ref, atol=1e-9)
        np.testing.assert_allclose(np.asarray(form.log_density(seqs)), ref, atol=1e-9)

    def test_length_modeled_returns_none(self):
        seq = SequenceDistribution(ExponentialDistribution(1.5), len_dist=PoissonDistribution(2.0))
        self.assertIsNone(to_exponential_family(seq))


class MultinomialExponentialFamilyTest(unittest.TestCase):
    def test_multinomial_reconstruction(self):
        # A multinomial over a Poisson element: log p(x) = sum_j c_j logPoisson(v_j).
        mn = MultinomialDistribution(PoissonDistribution(3.0))  # len_dist=Null, len_normalized=False
        form = to_exponential_family(mn)
        self.assertIsNotNone(form)
        xs = [[(0, 2.0), (3, 1.0)], [(2, 4.0)], [(1, 1.0), (5, 2.0), (0, 1.0)]]

        eta = form.natural_parameters()
        t = np.asarray(form.sufficient_statistics(xs), dtype=np.float64)
        a = float(form.log_partition())
        h = np.asarray(form.log_base_measure(xs), dtype=np.float64)
        ns = np.array([sum(c for _, c in obs) for obs in xs], dtype=np.float64)
        self.assertEqual(t.shape[1], eta.shape[0])
        self.assertEqual(form.dim, 1)

        # Count-weighted sufficient statistic: T = sum_j c_j * x_j (Poisson T(x) = x).
        np.testing.assert_allclose(t[:, 0], [3.0, 8.0, 11.0], atol=1e-9)
        # Joint reconstruction scales A by the total count n: <eta, T> - n*A + h.
        recon = h + t @ eta - ns * a
        ref = np.array([mn.log_density(x) for x in xs])
        np.testing.assert_allclose(recon, ref, atol=1e-9)
        np.testing.assert_allclose(np.asarray(form.log_density(xs)), ref, atol=1e-9)

    def test_length_modeled_returns_none(self):
        mn = MultinomialDistribution(PoissonDistribution(3.0), len_dist=PoissonDistribution(2.0))
        self.assertIsNone(to_exponential_family(mn))

    def test_length_normalized_returns_none(self):
        mn = MultinomialDistribution(PoissonDistribution(3.0), len_normalized=True)
        self.assertIsNone(to_exponential_family(mn))

    def test_non_exp_family_child_returns_none(self):
        from pysp.stats.univariate.continuous.laplace import LaplaceDistribution

        mn = MultinomialDistribution(LaplaceDistribution(0.0, 1.0))
        self.assertIsNone(to_exponential_family(mn))


class ConditionalExponentialFamilyTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.N = 3000
        self.x = rng.normal(0, 1, self.N)
        self.z = rng.normal(0, 1, self.N)
        self.rng = rng
        self.given = {"x": [0.0, 1.0, 2.0], "z": [0.5, -0.5, 0.0]}

    def test_normal_conditional(self):
        y = 2.0 * self.x - 1.5 * self.z + 0.7 + self.rng.normal(0, 0.5, self.N)
        m = Normal(free * Field("x") + free * Field("z") + free, free).fit(
            list(y), given={"x": list(self.x), "z": list(self.z)}
        )
        form = m.result.to_exponential_family()
        self.assertIsInstance(form, ConditionalExponentialFamilyForm)
        np.testing.assert_allclose(form.mean(self.given), m.result.predict(self.given), atol=1e-9)
        mu = form.mean(self.given)
        yy = np.array([0.3, 1.9, 4.5])
        ref = np.array([GaussianDistribution(float(mu[i]), m.result.sigma**2).log_density(yy[i]) for i in range(3)])
        np.testing.assert_allclose(form.log_density(yy, self.given), ref, atol=1e-9)

    def test_bernoulli_conditional(self):
        p = 1.0 / (1.0 + np.exp(-(2.0 * self.x - 1.0 * self.z + 0.5)))
        y = (self.rng.random(self.N) < p).astype(float)
        m = Bernoulli(free * Field("x") + free * Field("z") + free).fit(
            list(y), given={"x": list(self.x), "z": list(self.z)}
        )
        form = m.result.to_exponential_family()
        np.testing.assert_allclose(form.mean(self.given), m.result.predict(self.given), atol=1e-9)
        mu = form.mean(self.given)
        yy = np.array([1, 0, 1])
        ref = np.array([BernoulliDistribution(float(mu[i])).log_density(int(yy[i])) for i in range(3)])
        np.testing.assert_allclose(form.log_density(yy, self.given), ref, atol=1e-9)

    def test_poisson_conditional(self):
        lam = np.exp(0.5 * self.x - 0.3 * self.z + 0.2)
        y = self.rng.poisson(lam).astype(float)
        m = Poisson(free * Field("x") + free * Field("z") + free).fit(
            list(y), given={"x": list(self.x), "z": list(self.z)}
        )
        form = m.result.to_exponential_family()
        np.testing.assert_allclose(form.mean(self.given), m.result.predict(self.given), atol=1e-9)
        mu = form.mean(self.given)
        yy = np.array([0, 2, 5])
        ref = np.array([PoissonDistribution(float(mu[i])).log_density(int(yy[i])) for i in range(3)])
        np.testing.assert_allclose(form.log_density(yy, self.given), ref, atol=1e-9)


class NonExpFamilyTest(unittest.TestCase):
    def test_returns_none_for_non_exp_family(self):
        from pysp.stats.univariate.continuous.laplace import LaplaceDistribution

        d = LaplaceDistribution(0.0, 1.0)
        self.assertIsNone(to_exponential_family(d))
        self.assertFalse(is_exponential_family(d))


if __name__ == "__main__":
    unittest.main()
