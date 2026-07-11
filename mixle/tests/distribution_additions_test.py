import unittest

import numpy as np
import scipy.special
import scipy.stats

from mixle.inference import seq_estimate
from mixle.stats import (
    AffineTransform,
    BernoulliDistribution,
    BernoulliEstimator,
    BetaDistribution,
    BetaEstimator,
    CategoricalDistribution,
    ExpTransform,
    GaussianDistribution,
    GaussianEstimator,
    GumbelDistribution,
    GumbelEstimator,
    HalfNormalDistribution,
    HalfNormalEstimator,
    InverseGaussianDistribution,
    InverseGaussianEstimator,
    LaplaceDistribution,
    LaplaceEstimator,
    LogisticDistribution,
    LogisticEstimator,
    MixtureDistribution,
    MixtureEstimator,
    NegativeBinomialDistribution,
    NegativeBinomialEstimator,
    ParetoDistribution,
    ParetoEstimator,
    PointMassDistribution,
    PointMassEstimator,
    RayleighDistribution,
    RayleighEstimator,
    StudentTDistribution,
    StudentTEstimator,
    TransformDistribution,
    TransformEstimator,
    UniformDistribution,
    UniformEstimator,
    WeibullDistribution,
    WeibullEstimator,
    seq_encode,
    seq_log_density_sum,
)


class StandardDistributionAdditionsTestCase(unittest.TestCase):
    def test_string_round_trip(self):
        dists = [
            BernoulliDistribution(0.3, name="b", keys="k"),
            PointMassDistribution("fixed", name="pm", keys="k"),
            NegativeBinomialDistribution(3.0, 0.45, name="nb", keys="k"),
            BetaDistribution(2.0, 5.0, name="beta", keys="k"),
            StudentTDistribution(5.0, loc=1.0, scale=2.0, name="t", keys="k"),
            LaplaceDistribution(1.0, 2.0, name="laplace", keys="k"),
            LogisticDistribution(loc=1.0, scale=2.0, name="logistic", keys="k"),
            TransformDistribution(
                GaussianDistribution(0.0, 1.0), transform=AffineTransform(loc=2.0, scale=3.0), name="affine", keys="k"
            ),
            UniformDistribution(-1.0, 3.0, name="uniform", keys="k"),
            WeibullDistribution(1.5, 2.0, name="weibull", keys="k"),
            ParetoDistribution(2.0, 3.0, name="pareto", keys="k"),
            RayleighDistribution(2.0, name="rayleigh", keys="k"),
            InverseGaussianDistribution(2.0, 3.0, name="invgauss", keys="k"),
            HalfNormalDistribution(1.5, name="halfnorm", keys="k"),
            GumbelDistribution(loc=2.0, scale=1.5, name="gumbel", keys="k"),
        ]
        for dist in dists:
            self.assertEqual(str(eval(str(dist))), str(dist))

    def test_seq_log_density_matches_scalar(self):
        dists = [
            BernoulliDistribution(0.3),
            PointMassDistribution("fixed"),
            NegativeBinomialDistribution(3.0, 0.45),
            BetaDistribution(2.0, 5.0),
            StudentTDistribution(5.0, loc=1.0, scale=2.0),
            LaplaceDistribution(1.0, 2.0),
            LogisticDistribution(loc=1.0, scale=2.0),
            TransformDistribution(GaussianDistribution(0.0, 1.0), transform=AffineTransform(loc=2.0, scale=3.0)),
            UniformDistribution(-1.0, 3.0),
            WeibullDistribution(1.5, 2.0),
            ParetoDistribution(2.0, 3.0),
            RayleighDistribution(2.0),
            InverseGaussianDistribution(2.0, 3.0),
            HalfNormalDistribution(1.5),
            GumbelDistribution(loc=2.0, scale=1.5),
        ]
        for dist in dists:
            data = dist.sampler(3).sample(50)
            enc = dist.dist_to_encoder().seq_encode(data)
            seq_ll = dist.seq_log_density(enc)
            scalar_ll = np.asarray([dist.log_density(x) for x in data])
            self.assertTrue(np.allclose(seq_ll, scalar_ll, rtol=1.0e-12, atol=1.0e-12), str(dist))

        pm = PointMassDistribution("fixed")
        pm_enc = pm.dist_to_encoder().seq_encode(["fixed", "other", "fixed"])
        np.testing.assert_array_equal(pm.seq_log_density(pm_enc), np.asarray([0.0, -np.inf, 0.0]))

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

        logistic = LogisticDistribution(loc=1.0, scale=2.0)
        self.assertAlmostEqual(
            logistic.log_density(-0.25), scipy.stats.logistic.logpdf(-0.25, loc=1.0, scale=2.0), places=10
        )

        aff = TransformDistribution(GaussianDistribution(0.5, 4.0), transform=AffineTransform(loc=1.0, scale=3.0))
        self.assertTrue(aff.density_correction)
        self.assertAlmostEqual(aff.log_density(2.5), scipy.stats.norm.logpdf(2.5, loc=2.5, scale=6.0), places=10)

        exp = TransformDistribution(GaussianDistribution(0.25, 1.5), transform=ExpTransform())
        self.assertAlmostEqual(
            exp.log_density(1.25), scipy.stats.lognorm.logpdf(1.25, s=np.sqrt(1.5), scale=np.exp(0.25)), places=10
        )
        self.assertEqual(exp.log_density(-1.0), -np.inf)
        exp_enc = exp.dist_to_encoder().seq_encode([-1.0, 1.25])
        np.testing.assert_array_equal(np.isneginf(exp.seq_log_density(exp_enc)), np.asarray([True, False]))

        tcat = TransformDistribution(
            CategoricalDistribution({0: 0.6, 1: 0.4}), transform=AffineTransform(loc=10.0, scale=2.0)
        )
        self.assertFalse(tcat.density_correction)
        self.assertAlmostEqual(tcat.log_density(12.0), np.log(0.4), places=10)
        self.assertEqual(tcat.log_density(11.0), -np.inf)

        class LabelTransform:
            def forward(self, x):
                return "v%s" % x

            def inverse(self, y):
                return int(y[1:])

            def invalid_inverse_value(self):
                return 0

        labels = TransformDistribution(
            CategoricalDistribution({0: 0.6, 1: 0.4}), transform=LabelTransform(), density_correction=False
        )
        labels_enc = labels.dist_to_encoder().seq_encode(["v0", "v1"])
        np.testing.assert_allclose(labels.seq_log_density(labels_enc), np.log([0.6, 0.4]))

        unif = UniformDistribution(-1.0, 3.0)
        self.assertAlmostEqual(unif.log_density(0.5), scipy.stats.uniform.logpdf(0.5, loc=-1.0, scale=4.0), places=10)

        weib = WeibullDistribution(1.5, 2.0)
        self.assertAlmostEqual(weib.log_density(1.25), scipy.stats.weibull_min.logpdf(1.25, 1.5, scale=2.0), places=10)

        pareto = ParetoDistribution(2.0, 3.0)
        self.assertAlmostEqual(pareto.log_density(5.0), scipy.stats.pareto.logpdf(5.0, 3.0, scale=2.0), places=10)

        ray = RayleighDistribution(2.0)
        self.assertAlmostEqual(ray.log_density(1.5), scipy.stats.rayleigh.logpdf(1.5, scale=2.0), places=10)

        # scipy parameterizes invgauss(mu_shape, scale=lam) with mean = mu_shape * lam, matching
        # InverseGaussianDistribution(mu, lam) when mu_shape = mu / lam.
        ig = InverseGaussianDistribution(2.0, 3.0)
        for x in [0.5, 1.5, 4.0]:
            self.assertAlmostEqual(ig.log_density(x), scipy.stats.invgauss.logpdf(x, 2.0 / 3.0, scale=3.0), places=10)

        hn = HalfNormalDistribution(1.5)
        for x in [0.0, 0.75, 3.0]:
            self.assertAlmostEqual(hn.log_density(x), scipy.stats.halfnorm.logpdf(x, scale=1.5), places=10)
        self.assertEqual(hn.log_density(-0.5), -np.inf)

        gum = GumbelDistribution(loc=2.0, scale=1.5)
        for x in [-1.0, 2.0, 5.5]:
            self.assertAlmostEqual(gum.log_density(x), scipy.stats.gumbel_r.logpdf(x, loc=2.0, scale=1.5), places=10)

    def test_closed_form_estimators(self):
        acc = BernoulliEstimator().accumulator_factory().make()
        data = [True, False, True, True, False]
        for x in data:
            acc.update(x, 1.0, None)
        self.assertAlmostEqual(BernoulliEstimator().estimate(None, acc.value()).p, 3.0 / 5.0)

        acc = NegativeBinomialEstimator(r=3.0, estimate_r=False).accumulator_factory().make()
        data = [0, 1, 2, 5]
        for x in data:
            acc.update(x, 1.0, None)
        expected = 3.0 * len(data) / (3.0 * len(data) + sum(data))
        self.assertAlmostEqual(
            NegativeBinomialEstimator(r=3.0, estimate_r=False).estimate(None, acc.value()).p, expected
        )

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

        loc0, scale0, n = 1.25, 2.5, 100.0
        var0 = (np.pi * np.pi / 3.0) * scale0 * scale0
        fitted = LogisticEstimator().estimate(None, (n * loc0, n * (var0 + loc0 * loc0), n))
        self.assertAlmostEqual(fitted.loc, loc0)
        self.assertAlmostEqual(fitted.scale, scale0)

        shape0, scale0 = 1.7, 2.3
        mean0 = scale0 * np.exp(scipy.special.gammaln(1.0 + 1.0 / shape0))
        second0 = scale0 * scale0 * np.exp(scipy.special.gammaln(1.0 + 2.0 / shape0))
        fitted = WeibullEstimator().estimate(None, (n * mean0, n * second0, n))
        self.assertAlmostEqual(fitted.shape, shape0, places=6)
        self.assertAlmostEqual(fitted.scale, scale0, places=6)

        # Inverse Gaussian MLE is closed form: mu = mean(x), 1/lam = mean(1/x) - 1/mu.
        ig_data = np.asarray([0.5, 1.0, 2.0, 4.0])
        acc = InverseGaussianEstimator().accumulator_factory().make()
        acc.seq_update(InverseGaussianDistribution(1.0, 1.0).dist_to_encoder().seq_encode(ig_data), np.ones(4), None)
        fitted = InverseGaussianEstimator().estimate(None, acc.value())
        mu_hat = float(np.mean(ig_data))
        self.assertAlmostEqual(fitted.mu, mu_hat, places=10)
        self.assertAlmostEqual(fitted.lam, 1.0 / (np.mean(1.0 / ig_data) - 1.0 / mu_hat), places=8)

        # Half-normal MLE is closed form: sigma = sqrt(mean(x**2)).
        hn_data = np.asarray([0.5, 1.0, 2.0, 3.0])
        acc = HalfNormalEstimator().accumulator_factory().make()
        acc.seq_update(HalfNormalDistribution(1.0).dist_to_encoder().seq_encode(hn_data), np.ones(4), None)
        fitted = HalfNormalEstimator().estimate(None, acc.value())
        self.assertAlmostEqual(fitted.sigma, np.sqrt(np.mean(hn_data**2)), places=10)

        # Gumbel moment estimator inverts mean = loc + scale*gamma and var = (pi^2/6) scale^2.
        euler_gamma = 0.5772156649015328606
        loc0, scale0, n = 1.5, 2.0, 100.0
        mean0 = loc0 + scale0 * euler_gamma
        var0 = (np.pi**2 / 6.0) * scale0**2
        fitted = GumbelEstimator().estimate(None, (n * mean0, n * (var0 + mean0**2), n))
        self.assertAlmostEqual(fitted.scale, scale0, places=6)
        self.assertAlmostEqual(fitted.loc, loc0, places=6)

        pm = PointMassEstimator("fixed").estimate(None, None)
        self.assertIsInstance(pm, PointMassDistribution)
        self.assertEqual(pm.log_density("fixed"), 0.0)
        self.assertEqual(pm.log_density("other"), -np.inf)
        pm_acc = PointMassEstimator("fixed").accumulator_factory().make()
        self.assertIsNone(pm_acc.value())

        base = np.asarray([-2.0, -1.0, 0.0, 1.0, 4.0])
        data = 2.0 + 3.0 * base
        transform = AffineTransform(loc=2.0, scale=3.0)
        dist = TransformDistribution(GaussianDistribution(0.0, 1.0), transform=transform)
        est = TransformEstimator(GaussianEstimator(), transform=transform)
        enc = seq_encode(data, estimator=est)
        fitted = seq_estimate(enc, est, dist)
        self.assertIsInstance(fitted, TransformDistribution)
        self.assertAlmostEqual(fitted.dist.mu, base.mean())
        self.assertAlmostEqual(fitted.dist.sigma2, np.mean((base - base.mean()) ** 2))

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
            from mixle.stats.compute.fused_kernels import CompiledMixture
        except Exception as exc:  # noqa: BLE001
            self.skipTest("compiled kernels unavailable: %s" % exc)

        dists = [
            StudentTDistribution(5.0, loc=1.0, scale=2.0),
            LogisticDistribution(loc=1.0, scale=2.0),
            UniformDistribution(-1.0, 3.0),
            WeibullDistribution(1.5, 2.0),
            ParetoDistribution(2.0, 3.0),
            RayleighDistribution(2.0),
        ]
        for dist in dists:
            data = dist.sampler(4).sample(60)
            cm = CompiledMixture(dist)
            enc_k = cm.encode(data)
            enc_s = seq_encode(data, model=dist)
            self.assertTrue(
                np.allclose(cm.seq_log_density(enc_k), dist.seq_log_density(enc_s[0][1]), rtol=1.0e-10, atol=1.0e-12),
                str(dist),
            )


class StandardDistributionTorchTestCase(unittest.TestCase):
    def test_torch_em_matches_seq_for_bernoulli_mixture(self):
        try:
            from mixle.stats.compute.torch_mixture import TorchMixture
        except Exception as exc:  # noqa: BLE001
            self.skipTest("torch engine unavailable: %s" % exc)

        model = MixtureDistribution([BernoulliDistribution(0.2), BernoulliDistribution(0.8)], [0.4, 0.6])
        data = model.sampler(2).sample(200)
        est = MixtureEstimator([BernoulliEstimator(), BernoulliEstimator()])

        tm = TorchMixture(model)
        enc_t = tm.encode(data)
        enc_s = seq_encode(data, model=model)
        seq_model = seq_estimate(enc_s, est, model)
        torch_model = tm.em_step(enc_t, est, model=model)

        self.assertTrue(
            np.allclose(
                sorted(c.p for c in seq_model.components),
                sorted(c.p for c in torch_model.components),
                atol=1.0e-12,
            )
        )

    def test_torch_em_matches_seq_for_negative_binomial_mixture(self):
        try:
            from mixle.stats.compute.torch_mixture import TorchMixture
        except Exception as exc:  # noqa: BLE001
            self.skipTest("torch engine unavailable: %s" % exc)

        model = MixtureDistribution(
            [
                NegativeBinomialDistribution(3.0, 0.35),
                NegativeBinomialDistribution(3.0, 0.75),
            ],
            [0.5, 0.5],
        )
        data = model.sampler(3).sample(200)
        est = MixtureEstimator([NegativeBinomialEstimator(3.0), NegativeBinomialEstimator(3.0)])

        tm = TorchMixture(model)
        enc_t = tm.encode(data)
        enc_s = seq_encode(data, model=model)
        seq_model = seq_estimate(enc_s, est, model)
        torch_model = tm.em_step(enc_t, est, model=model)

        self.assertTrue(
            np.allclose(
                sorted(c.p for c in seq_model.components),
                sorted(c.p for c in torch_model.components),
                atol=1.0e-12,
            )
        )

    def test_torch_log_density_matches_seq_for_new_continuous_leaves(self):
        try:
            from mixle.stats.compute.torch_mixture import TorchMixture
        except Exception as exc:  # noqa: BLE001
            self.skipTest("torch engine unavailable: %s" % exc)

        dists = [
            StudentTDistribution(5.0, loc=1.0, scale=2.0),
            LogisticDistribution(loc=1.0, scale=2.0),
            WeibullDistribution(1.5, 2.0),
            ParetoDistribution(2.0, 3.0),
            RayleighDistribution(2.0),
        ]
        for dist in dists:
            data = dist.sampler(8).sample(80)
            tm = TorchMixture(dist)
            enc_t = tm.encode(data)
            enc_s = seq_encode(data, model=dist)
            self.assertTrue(
                np.allclose(tm.seq_log_density(enc_t), dist.seq_log_density(enc_s[0][1]), rtol=1.0e-10, atol=1.0e-12),
                str(dist),
            )


if __name__ == "__main__":
    unittest.main()
