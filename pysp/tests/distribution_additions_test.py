import unittest

import numpy as np
import scipy.stats

from pysp.stats import (
    BernoulliDistribution, BernoulliEstimator,
    BetaDistribution, BetaEstimator,
    LaplaceDistribution, LaplaceEstimator,
    MixtureDistribution, MixtureEstimator,
    NegativeBinomialDistribution, NegativeBinomialEstimator,
    ParetoDistribution, ParetoEstimator,
    RayleighDistribution, RayleighEstimator,
    StudentTDistribution, StudentTEstimator,
    UniformDistribution, UniformEstimator,
    seq_encode, seq_estimate, seq_log_density_sum,
)


class StandardDistributionAdditionsTestCase(unittest.TestCase):

    def test_string_round_trip(self):
        dists = [
            BernoulliDistribution(0.3, name='b', keys='k'),
            NegativeBinomialDistribution(3.0, 0.45, name='nb', keys='k'),
            BetaDistribution(2.0, 5.0, name='beta', keys='k'),
            StudentTDistribution(5.0, loc=1.0, scale=2.0, name='t', keys='k'),
            LaplaceDistribution(1.0, 2.0, name='laplace', keys='k'),
            UniformDistribution(-1.0, 3.0, name='uniform', keys='k'),
            ParetoDistribution(2.0, 3.0, name='pareto', keys='k'),
            RayleighDistribution(2.0, name='rayleigh', keys='k'),
        ]
        for dist in dists:
            self.assertEqual(str(eval(str(dist))), str(dist))

    def test_seq_log_density_matches_scalar(self):
        dists = [
            BernoulliDistribution(0.3),
            NegativeBinomialDistribution(3.0, 0.45),
            BetaDistribution(2.0, 5.0),
            StudentTDistribution(5.0, loc=1.0, scale=2.0),
            LaplaceDistribution(1.0, 2.0),
            UniformDistribution(-1.0, 3.0),
            ParetoDistribution(2.0, 3.0),
            RayleighDistribution(2.0),
        ]
        for dist in dists:
            data = dist.sampler(3).sample(50)
            enc = dist.dist_to_encoder().seq_encode(data)
            seq_ll = dist.seq_log_density(enc)
            scalar_ll = np.asarray([dist.log_density(x) for x in data])
            self.assertTrue(np.allclose(seq_ll, scalar_ll, rtol=1.0e-12, atol=1.0e-12), str(dist))

    def test_scipy_density_matches(self):
        beta = BetaDistribution(2.5, 4.0)
        for x in [0.1, 0.35, 0.9]:
            self.assertAlmostEqual(beta.log_density(x), scipy.stats.beta.logpdf(x, 2.5, 4.0), places=10)

        nb = NegativeBinomialDistribution(3.0, 0.45)
        for x in [0, 1, 4, 12]:
            self.assertAlmostEqual(nb.log_density(x), scipy.stats.nbinom.logpmf(x, 3.0, 0.45), places=10)

        t = StudentTDistribution(5.0, loc=1.0, scale=2.0)
        self.assertAlmostEqual(t.log_density(1.7), scipy.stats.t.logpdf(1.7, 5.0, loc=1.0, scale=2.0), places=10)

        lap = LaplaceDistribution(1.0, 2.0)
        self.assertAlmostEqual(lap.log_density(-0.25), scipy.stats.laplace.logpdf(-0.25, loc=1.0, scale=2.0), places=10)

        unif = UniformDistribution(-1.0, 3.0)
        self.assertAlmostEqual(unif.log_density(0.5), scipy.stats.uniform.logpdf(0.5, loc=-1.0, scale=4.0), places=10)

        pareto = ParetoDistribution(2.0, 3.0)
        self.assertAlmostEqual(pareto.log_density(5.0), scipy.stats.pareto.logpdf(5.0, 3.0, scale=2.0), places=10)

        ray = RayleighDistribution(2.0)
        self.assertAlmostEqual(ray.log_density(1.5), scipy.stats.rayleigh.logpdf(1.5, scale=2.0), places=10)

    def test_closed_form_estimators(self):
        acc = BernoulliEstimator().accumulator_factory().make()
        data = [True, False, True, True, False]
        for x in data:
            acc.update(x, 1.0, None)
        self.assertAlmostEqual(BernoulliEstimator().estimate(None, acc.value()).p, 3.0 / 5.0)

        acc = NegativeBinomialEstimator(r=3.0).accumulator_factory().make()
        data = [0, 1, 2, 5]
        for x in data:
            acc.update(x, 1.0, None)
        expected = 3.0 * len(data) / (3.0 * len(data) + sum(data))
        self.assertAlmostEqual(NegativeBinomialEstimator(r=3.0).estimate(None, acc.value()).p, expected)

        data = np.asarray([0.0, 1.0, 100.0])
        acc = LaplaceEstimator().accumulator_factory().make()
        enc = LaplaceDistribution(0.0, 1.0).dist_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), None)
        lap = LaplaceEstimator().estimate(None, acc.value())
        self.assertAlmostEqual(lap.mu, 1.0)
        self.assertAlmostEqual(lap.b, np.mean(np.abs(data - 1.0)))

        data = np.asarray([-3.0, 2.0, 5.0])
        acc = UniformEstimator().accumulator_factory().make()
        acc.seq_update(UniformDistribution(-1.0, 1.0).dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        unif = UniformEstimator().estimate(None, acc.value())
        self.assertAlmostEqual(unif.low, -3.0)
        self.assertAlmostEqual(unif.high, 5.0)

        data = np.asarray([2.0, 4.0, 8.0])
        acc = ParetoEstimator().accumulator_factory().make()
        enc = ParetoDistribution(1.0, 1.0).dist_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), None)
        pareto = ParetoEstimator().estimate(None, acc.value())
        self.assertAlmostEqual(pareto.xm, 2.0)
        self.assertAlmostEqual(pareto.alpha, len(data) / np.sum(np.log(data / 2.0)))

        data = np.asarray([1.0, 2.0, 3.0])
        acc = RayleighEstimator().accumulator_factory().make()
        enc = RayleighDistribution(1.0).dist_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), None)
        ray = RayleighEstimator().estimate(None, acc.value())
        self.assertAlmostEqual(ray.sigma, np.sqrt(np.sum(data * data) / (2.0 * len(data))))

        data = StudentTDistribution(6.0, loc=2.0, scale=1.5).sampler(5).sample(200)
        est = StudentTEstimator(df=6.0)
        enc = seq_encode(data, estimator=est)
        fitted = seq_estimate(enc, est, StudentTDistribution(6.0))
        self.assertGreater(fitted.scale, 0.0)
        self.assertTrue(np.isfinite(fitted.loc))

    def test_beta_estimator_is_finite_and_improves_likelihood(self):
        dist = BetaDistribution(2.0, 5.0)
        data = dist.sampler(7).sample(300)
        enc = seq_encode(data, model=dist)
        start = BetaDistribution(1.0, 1.0)
        fitted = seq_estimate(enc, BetaEstimator(), start)
        self.assertGreater(fitted.a, 0.0)
        self.assertGreater(fitted.b, 0.0)
        self.assertGreater(seq_log_density_sum(enc, dist)[1], seq_log_density_sum(enc, start)[1])
        self.assertGreater(seq_log_density_sum(enc, fitted)[1], seq_log_density_sum(enc, start)[1])

    def test_enumerators_descend(self):
        bern_items = BernoulliDistribution(0.7).enumerator().top_k(2)
        self.assertEqual([x for x, _ in bern_items], [True, False])
        self.assertGreaterEqual(bern_items[0][1], bern_items[1][1])

        nb_items = NegativeBinomialDistribution(3.0, 0.45).enumerator().top_k(30)
        lps = [lp for _, lp in nb_items]
        self.assertTrue(np.all(np.diff(lps) <= 1.0e-12))

    def test_fused_kernel_log_density_matches_seq(self):
        try:
            from pysp.stats.kernels import CompiledMixture
        except Exception as exc:
            self.skipTest('compiled kernels unavailable: %s' % exc)

        dists = [
            StudentTDistribution(5.0, loc=1.0, scale=2.0),
            UniformDistribution(-1.0, 3.0),
            ParetoDistribution(2.0, 3.0),
            RayleighDistribution(2.0),
        ]
        for dist in dists:
            data = dist.sampler(4).sample(60)
            cm = CompiledMixture(dist)
            enc_k = cm.encode(data)
            enc_s = seq_encode(data, model=dist)
            self.assertTrue(np.allclose(cm.seq_log_density(enc_k), dist.seq_log_density(enc_s[0][1]),
                                        rtol=1.0e-10, atol=1.0e-12),
                            str(dist))


class StandardDistributionTorchTestCase(unittest.TestCase):

    def test_torch_em_matches_seq_for_bernoulli_mixture(self):
        try:
            from pysp.stats.torch_engine import TorchMixture
        except Exception as exc:
            self.skipTest('torch engine unavailable: %s' % exc)

        model = MixtureDistribution([BernoulliDistribution(0.2), BernoulliDistribution(0.8)], [0.4, 0.6])
        data = model.sampler(2).sample(200)
        est = MixtureEstimator([BernoulliEstimator(), BernoulliEstimator()])

        tm = TorchMixture(model)
        enc_t = tm.encode(data)
        enc_s = seq_encode(data, model=model)
        seq_model = seq_estimate(enc_s, est, model)
        torch_model = tm.em_step(enc_t, est, model=model)

        self.assertTrue(np.allclose(
            sorted(c.p for c in seq_model.components),
            sorted(c.p for c in torch_model.components),
            atol=1.0e-12,
        ))

    def test_torch_em_matches_seq_for_negative_binomial_mixture(self):
        try:
            from pysp.stats.torch_engine import TorchMixture
        except Exception as exc:
            self.skipTest('torch engine unavailable: %s' % exc)

        model = MixtureDistribution([
            NegativeBinomialDistribution(3.0, 0.35),
            NegativeBinomialDistribution(3.0, 0.75),
        ], [0.5, 0.5])
        data = model.sampler(3).sample(200)
        est = MixtureEstimator([NegativeBinomialEstimator(3.0), NegativeBinomialEstimator(3.0)])

        tm = TorchMixture(model)
        enc_t = tm.encode(data)
        enc_s = seq_encode(data, model=model)
        seq_model = seq_estimate(enc_s, est, model)
        torch_model = tm.em_step(enc_t, est, model=model)

        self.assertTrue(np.allclose(
            sorted(c.p for c in seq_model.components),
            sorted(c.p for c in torch_model.components),
            atol=1.0e-12,
        ))

    def test_torch_log_density_matches_seq_for_new_continuous_leaves(self):
        try:
            from pysp.stats.torch_engine import TorchMixture
        except Exception as exc:
            self.skipTest('torch engine unavailable: %s' % exc)

        dists = [
            StudentTDistribution(5.0, loc=1.0, scale=2.0),
            ParetoDistribution(2.0, 3.0),
            RayleighDistribution(2.0),
        ]
        for dist in dists:
            data = dist.sampler(8).sample(80)
            tm = TorchMixture(dist)
            enc_t = tm.encode(data)
            enc_s = seq_encode(data, model=dist)
            self.assertTrue(np.allclose(tm.seq_log_density(enc_t), dist.seq_log_density(enc_s[0][1]),
                                        rtol=1.0e-10, atol=1.0e-12),
                            str(dist))


if __name__ == '__main__':
    unittest.main()
