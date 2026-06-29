"""Tests for the Bayesian (conjugate-prior / variational) machinery in mixle.stats.

Covers:
  - conjugate posterior updates checked against textbook closed forms,
  - prior density evaluations checked against scipy.stats,
  - consistency between log_density, expected_log_density and their seq_* forms,
  - convergence of the mixle.inference.estimation.fit driver: the penalized objective
    (data term + prior term) must increase monotonically for MAP-EM mixtures
    and the ELBO must increase monotonically for DPM variational inference.

Conventions exercised here:
  * conjugate priors live in mixle.stats.bayes.normal_gamma / normal_wishart / multivariate_normal_gamma /
    catdirichlet / gamma / beta / dirichlet;
  * the estimator API is the 2-arg ``estimate(nobs, suff_stat)``, so calls become
    ``estimate(None, suff_stat)``;
  * the posterior fit driver is ``mixle.inference.estimation.fit``; it prints ``OBJ=``
    lines that the convergence checks parse;
  * ``mixture_prior`` is in mixle.stats.latent.mixture and returns the tuple
    ``(weight_prior, component_priors)``;
  * sequence encoders are obtained via ``dist.dist_to_encoder().seq_encode(...)``;
  * the Markov chain / HMM use dict-based parameterizations and integer states
    (see mixle.stats.sequences.markov_chain / hidden_markov);
  * mixle.stats leaf distributions return -inf from scalar log_density for
    out-of-support values but raise ValueError at seq_encode time (a deliberate
    input-validation contract).
"""

import io
import itertools
import unittest

import numpy as np
import scipy.stats

from mixle.inference import estimate, initialize, seq_estimate
from mixle.inference.estimation import fit as fit_driver
from mixle.stats import (
    seq_encode,
)
from mixle.stats.bayes.dict_dirichlet import DictDirichletDistribution
from mixle.stats.bayes.dirichlet import DirichletDistribution
from mixle.stats.bayes.dirichlet_process_mixture import DirichletProcessMixtureEstimator
from mixle.stats.bayes.hierarchical_dirichlet_process_mixture import (
    HierarchicalDirichletProcessMixtureDistribution,
    HierarchicalDirichletProcessMixtureEstimator,
)
from mixle.stats.bayes.multivariate_normal_gamma import MultivariateNormalGammaDistribution
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution
from mixle.stats.bayes.normal_wishart import NormalWishartDistribution
from mixle.stats.combinator.conditional import ConditionalDistribution
from mixle.stats.combinator.optional import OptionalDistribution
from mixle.stats.latent.hidden_markov import (
    HiddenMarkovModelDistribution,
    HiddenMarkovModelEstimator,
)
from mixle.stats.latent.mixture import (
    MixtureDistribution,
    MixtureEstimator,
    mixture_prior,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)
from mixle.stats.sequences.markov_chain import MarkovChainDistribution, MarkovChainEstimator
from mixle.stats.sets.bernoulli_set import BernoulliSetDistribution
from mixle.stats.univariate.continuous.beta import BetaDistribution
from mixle.stats.univariate.continuous.exponential import ExponentialDistribution, ExponentialEstimator
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution, GaussianEstimator
from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution
from mixle.stats.univariate.discrete.binomial import BinomialDistribution, BinomialEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator
from mixle.stats.univariate.discrete.geometric import GeometricDistribution
from mixle.stats.univariate.discrete.integer_categorical import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
)
from mixle.stats.univariate.discrete.point_mass import PointMassDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution, PoissonEstimator
from mixle.utils.evaluation import k_fold_split_index
from mixle.utils.special import stirling2


def _encode(dist, data):
    """Encode ``data`` with the stats encoder for ``dist``."""
    return dist.dist_to_encoder().seq_encode(data)


def fit(data, est):
    """Accumulate ``data`` and return ``(acc, posterior_estimate)``.

    Uses the 2-arg ``estimate(None, suff_stat)`` stats estimator API.
    """
    acc = est.accumulator_factory().make()
    for x in data:
        acc.update(x, 1.0, None)
    return acc, est.estimate(None, acc.value())


class ConjugateUpdateTestCase(unittest.TestCase):
    """Posterior hyperparameters must match the textbook closed forms."""

    def test_gaussian_normal_gamma_posterior(self):
        mu0, lam0, a0, b0 = 1.0, 2.0, 3.0, 4.0
        est = GaussianEstimator(prior=NormalGammaDistribution(mu0, lam0, a0, b0))
        data = np.array([0.5, 1.5, 2.0, -1.0, 0.0])
        _, d = fit(data, est)

        n = len(data)
        xbar = data.mean()
        mu_n = (lam0 * mu0 + data.sum()) / (lam0 + n)
        lam_n = lam0 + n
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * np.sum((data - xbar) ** 2) + 0.5 * lam0 * n * (xbar - mu0) ** 2 / (lam0 + n)

        pm, pl, pa, pb = d.prior.get_parameters()
        self.assertAlmostEqual(pm, mu_n, places=10)
        self.assertAlmostEqual(pl, lam_n, places=10)
        self.assertAlmostEqual(pa, a_n, places=10)
        self.assertAlmostEqual(pb, b_n, places=8)
        # MAP of the joint normal-gamma: sigma2 = b_n / (a_n - 1/2)
        self.assertAlmostEqual(d.sigma2, pb / (pa - 0.5), places=10)

    def test_poisson_gamma_posterior(self):
        k0, theta0 = 2.0, 0.5
        est = PoissonEstimator(prior=GammaDistribution(k0, theta0))
        data = np.array([1, 0, 3, 2, 4])
        _, d = fit(data, est)

        k_n = k0 + data.sum()
        theta_n = theta0 / (len(data) * theta0 + 1.0)
        pk, pt = d.prior.get_parameters()
        self.assertAlmostEqual(pk, k_n, places=10)
        self.assertAlmostEqual(pt, theta_n, places=10)
        self.assertAlmostEqual(d.lam, (k_n - 1.0) * theta_n, places=10)

    def test_binomial_beta_posterior(self):
        a0, b0, n = 2.0, 3.0, 10
        est = BinomialEstimator(max_val=n, prior=BetaDistribution(a0, b0))
        data = np.array([3, 7, 5, 2])
        _, d = fit(data, est)

        a_n = a0 + data.sum()
        b_n = b0 + len(data) * n - data.sum()
        pa, pb = d.prior.get_parameters()
        self.assertAlmostEqual(pa, a_n, places=10)
        self.assertAlmostEqual(pb, b_n, places=10)
        self.assertAlmostEqual(d.p, (a_n - 1.0) / (a_n + b_n - 2.0), places=10)

    def test_exponential_gamma_posterior(self):
        # The stats exponential is scale-parameterized (d.beta = 1/rate); a
        # Gamma(k, theta) prior is over the rate with scale theta, so the
        # posterior is shape k + n and rate 1/theta + sum.
        k0, theta0 = 2.0, 0.25
        est = ExponentialEstimator(prior=GammaDistribution(k0, theta0))
        data = np.array([0.5, 1.5, 1.0])
        _, d = fit(data, est)

        pk, pt = d.prior.get_parameters()
        self.assertAlmostEqual(pk, k0 + len(data), places=10)
        self.assertAlmostEqual(1.0 / pt, 1.0 / theta0 + data.sum(), places=8)
        # MAP rate (pk - 1)/(1/pt); stats stores its reciprocal in d.beta.
        self.assertAlmostEqual(1.0 / d.beta, (pk - 1.0) / (1.0 / pt), places=8)

    def test_categorical_dirichlet_posterior(self):
        est = CategoricalEstimator(prior=DictDirichletDistribution({"a": 2.0, "b": 1.5, "c": 1.0}))
        data = ["a", "a", "b", "c", "a", "b"]
        _, d = fit(data, est)

        cpp = d.prior.get_parameters()
        self.assertAlmostEqual(cpp["a"], 2.0 + 3, places=10)
        self.assertAlmostEqual(cpp["b"], 1.5 + 2, places=10)
        self.assertAlmostEqual(cpp["c"], 1.0 + 1, places=10)
        # probabilities are a valid distribution
        self.assertAlmostEqual(sum(np.exp(d.log_density(k)) for k in "abc"), 1.0, places=10)

    def test_categorical_zero_probability_is_explicit_neg_inf(self):
        d = CategoricalDistribution({"a": 1.0}, default_value=0.0)
        self.assertEqual(d.log_density("b"), -np.inf)
        enc = _encode(d, ["a", "b"])
        np.testing.assert_allclose(d.seq_log_density(enc), np.asarray([0.0, -np.inf]))


class MixtureConjugatePriorTestCase(unittest.TestCase):
    """Mixture priors should compose weight and component conjugate priors."""

    def test_mixture_prior_helper_sets_weight_and_component_priors(self):
        weight_prior = DirichletDistribution(np.asarray([2.0, 3.0]))
        component_priors = [
            NormalGammaDistribution(0.0, 1.0, 2.0, 3.0),
            NormalGammaDistribution(5.0, 2.0, 4.0, 6.0),
        ]

        prior = mixture_prior(weight_prior, component_priors)
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()], prior=prior)

        self.assertIs(est.prior, weight_prior)
        self.assertIs(est.estimators[0].get_prior(), component_priors[0])
        self.assertIs(est.estimators[1].get_prior(), component_priors[1])

        # The stats mixture prior is the tuple (weight_prior, component_priors).
        rt = est.get_prior()
        self.assertIs(rt[0], weight_prior)
        self.assertIs(rt[1][0], component_priors[0])
        self.assertIs(rt[1][1], component_priors[1])

        dist = MixtureDistribution(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.4, 0.6],
            prior={"weights": weight_prior, "components": component_priors},
        )

        self.assertIs(dist.prior, weight_prior)
        self.assertIs(dist.components[0].get_prior(), component_priors[0])
        self.assertIs(dist.components[1].get_prior(), component_priors[1])

    def test_mixture_estimate_carries_weight_and_component_posteriors(self):
        weight_prior = DirichletDistribution(np.asarray([2.0, 3.0]))
        component_priors = [
            NormalGammaDistribution(0.0, 1.0, 2.0, 3.0),
            NormalGammaDistribution(5.0, 2.0, 4.0, 6.0),
        ]
        est = MixtureEstimator(
            [GaussianEstimator(), GaussianEstimator()],
            prior=mixture_prior(weight_prior, component_priors),
        )

        counts = np.asarray([3.0, 5.0])
        # Stats Gaussian sufficient statistics are (sum_x, sum_xx, nobs, nobs).
        comp_stats = (
            (3.0, 5.0, 3.0, 3.0),
            (20.0, 102.0, 5.0, 5.0),
        )
        model = est.estimate(None, (counts, comp_stats))

        expected_weight_alpha = weight_prior.get_parameters() + counts
        np.testing.assert_allclose(model.get_prior()[0].get_parameters(), expected_weight_alpha)
        expected_w_mode = expected_weight_alpha - 1.0
        expected_w_mode /= expected_w_mode.sum()
        np.testing.assert_allclose(model.w, expected_w_mode)

        expected_components = [
            GaussianEstimator(prior=component_priors[i]).estimate(None, comp_stats[i]) for i in range(2)
        ]
        for component, expected in zip(model.components, expected_components):
            np.testing.assert_allclose(component.get_prior().get_parameters(), expected.get_prior().get_parameters())


class PriorDensityTestCase(unittest.TestCase):
    def test_beta_log_density_matches_scipy(self):
        d = BetaDistribution(2.5, 3.5)
        for x in [0.1, 0.5, 0.9]:
            self.assertAlmostEqual(d.log_density(x), scipy.stats.beta.logpdf(x, 2.5, 3.5), places=10)
            self.assertAlmostEqual(d.density(x), scipy.stats.beta.pdf(x, 2.5, 3.5), places=10)

    # NOTE: entropy()/cross_entropy() helpers on leaf likelihoods and on
    # Beta/Dirichlet priors are not part of mixle.stats -- only the conjugate
    # prior families that need entropy for the ELBO (NormalGamma, NormalWishart,
    # MultivariateNormalGamma) expose them. Tests for that legacy method surface
    # (Beta/Dirichlet entropy, discrete entropy/cross-entropy signs, count-moment
    # helpers) have no mixle.stats behavior to assert, so they are not included.

    def test_normal_gamma_log_density(self):
        d = NormalGammaDistribution(1.0, 2.0, 3.0, 4.0)
        mu, tau = 0.5, 1.5
        ref = scipy.stats.gamma.logpdf(tau, 3.0, scale=1.0 / 4.0) + scipy.stats.norm.logpdf(
            mu, loc=1.0, scale=np.sqrt(1.0 / (2.0 * tau))
        )
        self.assertAlmostEqual(d.log_density((mu, tau)), ref, places=10)

    def test_normal_gamma_cross_entropy_self_is_entropy(self):
        d = NormalGammaDistribution(1.0, 2.0, 3.0, 4.0)
        self.assertAlmostEqual(d.cross_entropy(d), d.entropy(), places=8)

    def test_normal_gamma_sampler_runs(self):
        d = NormalGammaDistribution(0.0, 1.0, 3.0, 2.0)
        samples = d.sampler(seed=1).sample(size=5)
        self.assertEqual(len(samples), 5)
        for mu, tau in samples:
            self.assertTrue(np.isfinite(mu) and tau > 0)

    def test_multivariate_normal_gamma_cross_entropy_contract(self):
        d = MultivariateNormalGammaDistribution(
            np.array([0.0, 1.0]), np.array([2.0, 3.0]), np.array([4.0, 5.0]), np.array([6.0, 7.0])
        )
        self.assertAlmostEqual(d.cross_entropy(d), d.entropy(), places=10)
        with self.assertRaises(NotImplementedError):
            d.cross_entropy(NormalGammaDistribution(0.0, 1.0, 2.0, 3.0))

    def test_support_boundaries_match_scalar_and_raise_on_seq_encode(self):
        # Scalar log_density returns -inf for out-of-support values...
        with np.errstate(divide="ignore", invalid="ignore"):
            self.assertEqual(BetaDistribution(2.0, 3.0).log_density(0.0), -np.inf)
            self.assertEqual(BetaDistribution(2.0, 3.0).log_density(1.0), -np.inf)
            self.assertEqual(ExponentialDistribution(2.0).log_density(-1.0), -np.inf)
            self.assertEqual(GammaDistribution(2.0, 3.0).log_density(-1.0), -np.inf)
            self.assertEqual(GeometricDistribution(0.3).log_density(0), -np.inf)
            self.assertEqual(PoissonDistribution(3.0).log_density(-1), -np.inf)

        # ...but mixle.stats validates the support at seq_encode time.
        for dist, data in [
            (ExponentialDistribution(2.0), [-1.0, 0.5]),
            (GammaDistribution(2.0, 3.0), [-1.0, 0.0, 2.0]),
            (GeometricDistribution(0.3), [-1, 0, 1, 2]),
            (PoissonDistribution(3.0), [-1, 0, 3]),
        ]:
            with self.subTest(dist=type(dist).__name__):
                with self.assertRaises(ValueError):
                    _encode(dist, data)

    def test_seq_expected_log_density_falls_back_without_conjugate_prior(self):
        from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution

        cases = [
            (BernoulliDistribution(0.3), [True, False]),
            (IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5]), [0, 1, 2]),
            (
                OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.25, missing_value=None),
                [None, -1.0, 0.5],
            ),
        ]
        for dist, data in cases:
            with self.subTest(dist=type(dist).__name__):
                enc = _encode(dist, data)
                np.testing.assert_allclose(dist.seq_expected_log_density(enc), dist.seq_log_density(enc))

    def test_base_expected_log_density_falls_back_to_plugin_density(self):
        cases = [
            (PointMassDistribution("x"), ["x", "y"]),
            (BernoulliSetDistribution({"a": 0.25, "b": 0.75}), [{"a"}, {"b"}, set()]),
        ]
        for dist, data in cases:
            with self.subTest(dist=type(dist).__name__):
                enc = _encode(dist, data)
                self.assertEqual(dist.expected_log_density(data[0]), dist.log_density(data[0]))
                np.testing.assert_allclose(dist.seq_expected_log_density(enc), dist.seq_log_density(enc))

    def test_conditional_distribution_without_default_matches_scalar_and_vectorized(self):
        dist = ConditionalDistribution({"seen": GaussianDistribution(0.0, 1.0)}, default_dist=None)
        data = [("seen", 0.0), ("missing", 1.0)]
        enc = _encode(dist, data)

        self.assertEqual(dist.log_density(data[1]), -np.inf)
        np.testing.assert_allclose(dist.seq_log_density(enc), np.asarray([dist.log_density(x) for x in data]))


class DensityConsistencyTestCase(unittest.TestCase):
    def battery(self):
        return [
            GaussianDistribution(1.0, 2.0),
            LogGaussianDistribution(0.5, 1.5),
            PoissonDistribution(4.0),
            ExponentialDistribution(2.0),
            GeometricDistribution(0.3),
            BinomialDistribution(0.4, 8),
            IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5]),
        ]

    def test_seq_log_density_matches_scalar(self):
        for dist in self.battery():
            data = dist.sampler(seed=3).sample(size=40)
            enc = _encode(dist, data)
            seq_ll = dist.seq_log_density(enc)
            scalar_ll = np.asarray([dist.log_density(x) for x in data])
            self.assertTrue(
                np.allclose(seq_ll, scalar_ll, rtol=1e-10, atol=1e-12), "seq/scalar mismatch for %s" % str(dist)
            )

    def test_seq_expected_log_density_matches_scalar(self):
        for dist in self.battery():
            data = dist.sampler(seed=4).sample(size=40)
            enc = _encode(dist, data)
            seq_ell = dist.seq_expected_log_density(enc)
            scalar_ell = np.asarray([dist.expected_log_density(x) for x in data])
            self.assertTrue(
                np.allclose(seq_ell, scalar_ell, rtol=1e-10, atol=1e-12),
                "seq/scalar expected mismatch for %s" % str(dist),
            )

    def test_expected_log_density_approaches_log_density_for_sharp_prior(self):
        # with a very concentrated prior, E_q[log p(x|theta)] -> log p(x|theta_0)
        mu0, s2 = 1.0, 2.0
        n_pseudo = 1.0e8
        prior = NormalGammaDistribution(mu0, n_pseudo, n_pseudo / 2.0, (n_pseudo / 2.0) * s2)
        d = GaussianDistribution(mu0, s2, prior=prior)
        for x in [-1.0, 1.0, 3.0]:
            self.assertAlmostEqual(d.expected_log_density(x), d.log_density(x), places=5)


def _ng():
    """A reusable component prior for the variational mixture tests."""
    return NormalGammaDistribution(0.0, 1.0e-3, 1.0, 1.0)


class OptimizeConvergenceTestCase(unittest.TestCase):
    @staticmethod
    def run_optimize(est, data, max_its, seed=2, delta=1.0e-9):
        objs: list[float] = []

        class _Trace:  # capture the per-iteration objective via the structured em_record hook
            def write(self, _s):
                pass

            def flush(self):
                pass

            def em_record(self, i, ll, dll, vll, obj_label):
                objs.append(ll)

        model = fit_driver(
            data, est, max_its=max_its, delta=delta, rng=np.random.RandomState(seed), out=_Trace(), print_iter=1
        )
        return model, np.asarray(objs)

    def test_map_em_mixture_objective_monotone(self):
        truth = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.4, 0.6])
        data = truth.sampler(seed=1).sample(800)

        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        model, objs = self.run_optimize(est, data, max_its=40)

        self.assertGreater(len(objs), 3)
        self.assertTrue(np.all(np.diff(objs) >= -1.0e-6), "penalized objective decreased: %s" % str(np.diff(objs)))

        mus = sorted(c.mu for c in model.components)
        self.assertAlmostEqual(mus[0], -3.0, delta=0.3)
        self.assertAlmostEqual(mus[1], 3.0, delta=0.3)

    def test_dpm_elbo_monotone(self):
        truth = MixtureDistribution(
            [GaussianDistribution(-4.0, 0.5), GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 2.0)],
            [0.3, 0.4, 0.3],
        )
        data = truth.sampler(seed=1).sample(800)

        K = 10
        est = DirichletProcessMixtureEstimator(
            [GaussianEstimator(prior=_ng()) for _ in range(K)], prior=GammaDistribution(2, 1)
        )
        # delta=None runs the full schedule so the variational atoms separate.
        model, objs = self.run_optimize(est, data, max_its=25, seed=3, delta=None)

        self.assertGreater(len(objs), 5)
        self.assertTrue(np.all(np.diff(objs) >= -1.0e-5), "ELBO decreased: %s" % str(np.diff(objs)))
        # weights are a valid distribution
        self.assertAlmostEqual(model.w.sum(), 1.0, places=8)
        self.assertTrue(np.all(model.w >= 0))

    def test_dpm_single_component_truncation_is_valid(self):
        data = GaussianDistribution(0.0, 1.0).sampler(seed=1).sample(80)
        est = DirichletProcessMixtureEstimator([GaussianEstimator(prior=_ng())], prior=GammaDistribution(2, 1))
        model, objs = self.run_optimize(est, data, max_its=5, seed=3)

        self.assertGreater(len(objs), 0)
        self.assertEqual(len(model.components), 1)
        self.assertAlmostEqual(model.w[0], 1.0, places=12)
        self.assertTrue(np.isfinite(model.a))
        self.assertTrue(np.all(np.isfinite(model.g)))
        enc = seq_encode(data, model, num_chunks=1)
        local_elbo = model.seq_local_elbo(enc[0][1])
        comp_expected = model.components[0].seq_expected_log_density(enc[0][1])
        np.testing.assert_allclose(local_elbo, comp_expected, rtol=1.0e-12, atol=1.0e-12)

    def test_dpm_scalar_and_vectorized_updates_agree(self):
        truth = MixtureDistribution([GaussianDistribution(-3.0, 0.7), GaussianDistribution(2.0, 1.2)], [0.45, 0.55])
        data = truth.sampler(seed=11).sample(250)
        est = DirichletProcessMixtureEstimator(
            [GaussianEstimator(prior=_ng()) for _ in range(5)], prior=GammaDistribution(2, 1)
        )
        prev = initialize(data, est, rng=np.random.RandomState(12), p=0.8)

        scalar = estimate(data, est, prev)
        vector = seq_estimate(seq_encode(data, prev, num_chunks=4), est, prev)

        np.testing.assert_allclose(scalar.w, vector.w, rtol=1.0e-12, atol=1.0e-12)
        self.assertAlmostEqual(scalar.a, vector.a, places=10)
        np.testing.assert_allclose(
            sorted(c.mu for c in scalar.components), sorted(c.mu for c in vector.components), rtol=1.0e-12, atol=1.0e-12
        )

    def test_optimize_runs_with_plain_estimator(self):
        # the fit driver falls back gracefully for non-mixture estimators
        data = GaussianDistribution(2.0, 1.0).sampler(seed=5).sample(200)
        buf = io.StringIO()
        model = fit_driver(data, GaussianEstimator(), max_its=5, rng=np.random.RandomState(1), out=buf)
        self.assertAlmostEqual(model.mu, 2.0, delta=0.3)

    def test_optimize_returns_accepted_step_before_delta_termination(self):
        data = GaussianDistribution(0.0, 1.0).sampler(seed=1).sample(200)
        prev = GaussianDistribution(10.0, 1.0)
        model = fit_driver(data, GaussianEstimator(), max_its=1, delta=1.0e99, prev_estimate=prev, out=io.StringIO())

        self.assertLess(abs(model.mu), 0.5)

    def test_optimize_restores_numpy_error_state(self):
        data = GaussianDistribution(0.0, 1.0).sampler(seed=2).sample(30)
        old_err = np.seterr(divide="raise")
        try:
            fit_driver(data, GaussianEstimator(), max_its=1, rng=np.random.RandomState(2), out=io.StringIO())
            self.assertEqual(np.geterr()["divide"], "raise")
        finally:
            np.seterr(**old_err)

    def test_local_drivers_support_custom_estimator_api(self):
        # The stats drivers accept a minimal hand-rolled estimator that
        # implements the snake_case accumulator/encoder contract.
        class CustomAccumulator:
            def __init__(self):
                self.n = 0.0
                self.s = 0.0

            def update(self, x, weight, estimate):
                self.n += weight
                self.s += weight * x

            def initialize(self, x, weight, rng):
                self.update(x, weight, None)

            def seq_initialize(self, x, weights, rng):
                self.seq_update(x, weights, None)

            def seq_update(self, x, weights, estimate):
                x = np.asarray(x, dtype=float)
                weights = np.asarray(weights, dtype=float)
                self.n += float(weights.sum())
                self.s += float(np.dot(x, weights))

            def combine(self, suff_stat):
                self.n += suff_stat[0]
                self.s += suff_stat[1]
                return self

            def value(self):
                return self.n, self.s

            def from_value(self, suff_stat):
                self.n, self.s = suff_stat
                return self

            def acc_to_encoder(self):
                return GaussianDistribution(0.0, 1.0).dist_to_encoder()

            def key_merge(self, stats_dict):
                return None

            def key_replace(self, stats_dict):
                return None

            def keys(self):
                return []

        class CustomFactory:
            def make(self):
                return CustomAccumulator()

        class CustomEstimator:
            def accumulator_factory(self):
                return CustomFactory()

            def estimate(self, nobs, suff_stat):
                self.last_nobs = nobs
                n, s = suff_stat
                return GaussianDistribution(s / n, 1.0)

            def keys(self):
                return []

        data = [1.0, 2.0, 4.0]
        est = CustomEstimator()
        expected = np.mean(data)

        local_model = estimate(data, est)
        self.assertAlmostEqual(local_model.mu, expected)
        self.assertAlmostEqual(est.last_nobs, len(data))

        init_model = initialize(data, est, rng=np.random.RandomState(1), p=1.0)
        self.assertAlmostEqual(init_model.mu, expected)
        self.assertAlmostEqual(est.last_nobs, len(data))

        enc = seq_encode(data, local_model, num_chunks=2)
        seq_model = seq_estimate(enc, est, local_model)
        self.assertAlmostEqual(seq_model.mu, expected)


class NormalWishartTestCase(unittest.TestCase):
    def test_one_dim_equals_normal_gamma(self):
        # NW(mu, kappa, W=1/(2b), nu=2a) in 1-d must equal NG(mu, kappa, a, b)
        ng = NormalGammaDistribution(0.5, 2.0, 3.0, 4.0)
        nw = NormalWishartDistribution([0.5], 2.0, [[1.0 / (2 * 4.0)]], 2 * 3.0)
        for mu, tau in [(0.7, 1.3), (-1.0, 0.2), (0.5, 3.0)]:
            self.assertAlmostEqual(ng.log_density((mu, tau)), nw.log_density(([mu], [[tau]])), places=10)
        self.assertAlmostEqual(ng.entropy(), nw.entropy(), places=8)

    def test_cross_entropy_self_is_entropy(self):
        nw = NormalWishartDistribution(np.array([1.0, -1.0]), 2.0, np.eye(2) * 0.4, 5.0)
        self.assertAlmostEqual(nw.cross_entropy(nw), nw.entropy(), places=10)

    def test_sampler_runs(self):
        nw = NormalWishartDistribution(np.zeros(2), 1.0, np.eye(2), 4.0)
        for mu, lam in nw.sampler(seed=1).sample(size=3):
            self.assertTrue(np.all(np.isfinite(mu)))
            self.assertTrue(np.all(np.linalg.eigvalsh(lam) > 0))


class MultivariateGaussianTestCase(unittest.TestCase):
    def make_dist(self):
        return MultivariateGaussianDistribution([1.0, -2.0], [[2.0, 0.6], [0.6, 1.0]])

    def test_log_density_matches_scipy(self):
        d = self.make_dist()
        for x in [np.array([0.0, 0.0]), np.array([2.0, -3.0])]:
            ref = scipy.stats.multivariate_normal.logpdf(x, d.mu, d.covar)
            self.assertAlmostEqual(d.log_density(x), ref, places=10)

    def test_seq_matches_scalar(self):
        d = self.make_dist()
        data = d.sampler(seed=1).sample(size=30)
        enc = _encode(d, data)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in data]))
        self.assertTrue(np.allclose(d.seq_expected_log_density(enc), [d.expected_log_density(u) for u in data]))

    def test_normal_wishart_posterior_closed_form(self):
        m0 = np.array([0.5, -0.5])
        kappa0, nu0 = 2.0, 4.0
        w0 = np.eye(2) * 0.5
        est = MultivariateGaussianEstimator(2, prior=NormalWishartDistribution(m0, kappa0, w0, nu0))

        data = self.make_dist().sampler(seed=2).sample(size=25)
        _, d = fit(data, est)

        xs = np.asarray(data)
        n = len(xs)
        xbar = xs.mean(axis=0)
        scatter = np.einsum("ni,nj->ij", xs - xbar, xs - xbar)

        m_n, kappa_n, w_n, nu_n = d.prior.get_parameters()
        self.assertAlmostEqual(kappa_n, kappa0 + n, places=10)
        self.assertAlmostEqual(nu_n, nu0 + n, places=10)
        self.assertTrue(np.allclose(m_n, (kappa0 * m0 + n * xbar) / (kappa0 + n)))

        dmu = xbar - m0
        w_n_inv_ref = np.linalg.inv(w0) + scatter + (kappa0 * n / (kappa0 + n)) * np.outer(dmu, dmu)
        self.assertTrue(np.allclose(np.linalg.inv(w_n), w_n_inv_ref, rtol=1e-8))

        # MAP covariance: W_n^-1 / (nu_n - d)
        self.assertTrue(np.allclose(d.covar, w_n_inv_ref / (nu_n - 2), rtol=1e-8))

    def test_recovery(self):
        truth = self.make_dist()
        data = truth.sampler(seed=3).sample(size=4000)
        _, m = fit(data, MultivariateGaussianEstimator(2))
        self.assertTrue(np.allclose(m.mu, truth.mu, atol=0.1))
        self.assertTrue(np.allclose(m.covar, truth.covar, atol=0.15))


class MarkovChainTestCase(unittest.TestCase):
    PI = np.array([0.6, 0.3, 0.1])
    A_MAT = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])

    def make_dist(self):
        pi, a_mat = self.PI, self.A_MAT
        return MarkovChainDistribution(
            {i: pi[i] for i in range(3)},
            {i: {j: a_mat[i, j] for j in range(3)} for i in range(3)},
            len_dist=CategoricalDistribution({12: 1.0}),
        )

    def test_seq_matches_scalar(self):
        d = self.make_dist()
        seqs = d.sampler(seed=1).sample(size=25)
        enc = _encode(d, seqs)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in seqs]))
        self.assertTrue(np.allclose(d.seq_expected_log_density(enc), [d.expected_log_density(u) for u in seqs]))

    # NOTE: the Dirichlet posterior-count test is not included here -- the stats
    # MarkovChain conjugate path's set_prior cannot be enabled with a pure
    # mixle.stats Dirichlet import, and the closed-form posterior counts are
    # already covered by
    # mixle/tests/stats_bayes_markov_test.py::test_markov_chain_posterior_closed_form.

    def test_recovery(self):
        seqs = self.make_dist().sampler(seed=2).sample(size=800)
        est = MarkovChainEstimator(len_estimator=CategoricalEstimator())
        _, m = fit(seqs, est)
        init_prob_vec = np.array([m.init_prob_map[i] for i in range(3)])
        transition_mat = np.array([[m.transition_map[i][j] for j in range(3)] for i in range(3)])
        self.assertTrue(np.allclose(init_prob_vec, self.PI, atol=0.06))
        self.assertTrue(np.allclose(transition_mat, self.A_MAT, atol=0.05))


class HiddenMarkovModelTestCase(unittest.TestCase):
    def make_dist(self, length=10):
        topics = [GaussianDistribution(-5.0, 1.0), GaussianDistribution(5.0, 1.0)]
        return HiddenMarkovModelDistribution(
            topics=topics,
            w=[0.7, 0.3],
            transitions=[[0.9, 0.1], [0.2, 0.8]],
            len_dist=CategoricalDistribution({length: 1.0}),
        )

    def test_seq_matches_scalar(self):
        d = self.make_dist()
        seqs = d.sampler(seed=1).sample(size=15)
        enc = _encode(d, seqs)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in seqs]))

    def test_forward_matches_brute_force(self):
        # forward log-likelihood vs explicit sum over all state paths
        d = self.make_dist(length=4)
        x = d.sampler(seed=2).sample_seq()

        tot = -np.inf
        for path in itertools.product(range(2), repeat=len(x)):
            lp = np.log(d.w[path[0]]) + d.topics[path[0]].log_density(x[0])
            for t in range(1, len(x)):
                lp += np.log(d.transitions[path[t - 1], path[t]]) + d.topics[path[t]].log_density(x[t])
            tot = np.logaddexp(tot, lp)
        tot += d.len_dist.log_density(len(x))
        self.assertAlmostEqual(d.log_density(x), tot, places=8)

    def test_viterbi_recovers_separated_states(self):
        d = self.make_dist()
        x = [-5.1, -4.9, 5.2, 4.8, -5.0]
        self.assertEqual(list(d.viterbi(x)), [0, 0, 1, 1, 0])

    def test_optimize_monotone_and_recovery(self):
        truth = self.make_dist()
        seqs = truth.sampler(seed=3).sample(size=200)

        est = HiddenMarkovModelEstimator(
            [GaussianEstimator(), GaussianEstimator()], len_estimator=CategoricalEstimator()
        )
        m, objs = OptimizeConvergenceTestCase.run_optimize(est, seqs, max_its=30, seed=4, delta=1.0e-9)
        self.assertGreater(len(objs), 3)
        self.assertTrue(np.all(np.diff(objs) >= -1.0e-6), "HMM penalized objective decreased: %s" % str(np.diff(objs)))

        order = np.argsort([t.mu for t in m.topics])
        mus = [m.topics[i].mu for i in order]
        self.assertAlmostEqual(mus[0], -5.0, delta=0.3)
        self.assertAlmostEqual(mus[1], 5.0, delta=0.3)

        trans = m.transitions[np.ix_(order, order)]
        self.assertTrue(np.allclose(trans, truth.transitions, atol=0.08))


class NestedHMMTestCase(unittest.TestCase):
    """HMMs as components of mixtures and DPMs."""

    @staticmethod
    def make_truth():
        len_d = CategoricalDistribution({12: 1.0})
        h1 = HiddenMarkovModelDistribution(
            topics=[GaussianDistribution(-6.0, 1.0), GaussianDistribution(-1.0, 1.0)],
            w=[0.5, 0.5],
            transitions=[[0.95, 0.05], [0.05, 0.95]],
            len_dist=len_d,
        )
        h2 = HiddenMarkovModelDistribution(
            topics=[GaussianDistribution(2.0, 1.0), GaussianDistribution(7.0, 1.0)],
            w=[0.5, 0.5],
            transitions=[[0.3, 0.7], [0.7, 0.3]],
            len_dist=len_d,
        )
        return MixtureDistribution([h1, h2], [0.6, 0.4])

    @staticmethod
    def hmm_est():
        return HiddenMarkovModelEstimator(
            [GaussianEstimator(), GaussianEstimator()], len_estimator=CategoricalEstimator()
        )

    def test_mixture_of_hmms_monotone(self):
        truth = self.make_truth()
        data = truth.sampler(seed=1).sample(150)

        est = MixtureEstimator([self.hmm_est(), self.hmm_est()])
        model, objs = OptimizeConvergenceTestCase.run_optimize(est, data, max_its=15)

        self.assertTrue(np.all(np.diff(objs) >= -1.0e-6), "mixture-of-HMMs objective decreased")
        self.assertAlmostEqual(model.w.sum(), 1.0, places=10)

    def test_dpm_of_hmms_monotone(self):
        truth = self.make_truth()
        data = truth.sampler(seed=1).sample(150)

        est = DirichletProcessMixtureEstimator([self.hmm_est() for _ in range(4)], prior=GammaDistribution(2, 1))
        model, objs = OptimizeConvergenceTestCase.run_optimize(est, data, max_its=15, seed=3, delta=None)

        self.assertTrue(np.all(np.diff(objs) >= -1.0e-5), "DPM-of-HMMs ELBO decreased")
        self.assertAlmostEqual(model.w.sum(), 1.0, places=8)
        self.assertTrue(np.all(model.w >= 0))


class HierarchicalDPMTestCase(unittest.TestCase):
    @staticmethod
    def make_truth():
        atoms = [GaussianDistribution(m, 0.5) for m in [-6.0, -2.0, 2.0, 6.0]]
        return HierarchicalDirichletProcessMixtureDistribution(
            atoms, beta=[0.4, 0.3, 0.2, 0.1], alpha=3.0, gamma=2.0, len_dist=CategoricalDistribution({40: 1.0})
        )

    @classmethod
    def fit_model(cls, groups, k=8, max_its=60, seed=2):
        est = HierarchicalDirichletProcessMixtureEstimator(
            [GaussianEstimator(prior=_ng()) for _ in range(k)],
            gamma=2.0,
            alpha=3.0,
            len_estimator=CategoricalEstimator(),
        )
        # delta=None runs the full schedule (the HDP beta step is approximate, so
        # the fit acceptance gate keeps the best accepted model).
        return OptimizeConvergenceTestCase.run_optimize(est, groups, max_its=max_its, seed=seed, delta=None)

    def test_objective_increases_and_recovers_atoms(self):
        truth = self.make_truth()
        groups = truth.sampler(seed=1).sample(size=60)
        model, objs = self.fit_model(groups)

        self.assertGreater(objs[-1], objs[0])
        # every true atom location is covered by some effective fitted atom
        true_mus = [-6.0, -2.0, 2.0, 6.0]
        eff_mus = [model.components[i].mu for i in np.flatnonzero(model.beta > 0.03)]
        for tm in true_mus:
            self.assertTrue(
                min(abs(tm - em) for em in eff_mus) < 0.6, "true atom %.1f not recovered (got %s)" % (tm, str(eff_mus))
            )

    def test_group_weights_track_usage(self):
        truth = self.make_truth()
        groups = truth.sampler(seed=1).sample(size=40)
        model, _ = self.fit_model(groups, max_its=40)

        self.assertEqual(model.group_weights.shape, (40, 8))
        self.assertTrue(np.allclose(model.group_weights.sum(axis=1), 1.0))
        gp = model.group_posteriors(groups)
        cors = [np.corrcoef(model.group_weights[j], gp[j])[0, 1] for j in range(40)]
        self.assertGreater(np.median(cors), 0.9)

    def test_new_group_scoring_is_finite(self):
        truth = self.make_truth()
        groups = truth.sampler(seed=1).sample(size=30)
        model, _ = self.fit_model(groups, max_its=20)

        held_out = truth.sampler(seed=9).sample(size=5)
        enc = _encode(model, held_out)
        ll = model.seq_log_density(enc)
        self.assertTrue(np.all(np.isfinite(ll)))
        # scalar/seq consistency for new-group scoring
        for g, l in zip(held_out, ll):
            self.assertAlmostEqual(model.log_density(g), l, places=8)


class UtilityTestCase(unittest.TestCase):
    def test_k_fold_split_index(self):
        idx = k_fold_split_index(100, 5, np.random.RandomState(1))
        self.assertEqual(len(idx), 100)
        for i in range(5):
            self.assertEqual((idx == i).sum(), 20)

    def test_stirling2(self):
        # The Poisson.moment() helper is not part of mixle.stats; the underlying
        # Stirling-number-of-the-second-kind utility still is.
        self.assertEqual(stirling2(6, 3), 90)


class IntegerCategoricalConventionTestCase(unittest.TestCase):
    """mixle.stats IntegerCategorical uses the (min_val, p_vec) order and the
    matching keyword names."""

    def test_argument_conventions_agree(self):
        pv = [0.5, 0.3, 0.2]
        d_pos = IntegerCategoricalDistribution(2, pv)
        d_kw = IntegerCategoricalDistribution(p_vec=pv, min_val=2)
        self.assertEqual(d_pos.min_val, d_kw.min_val)
        for x in (2, 3, 4):
            self.assertAlmostEqual(d_pos.log_density(x), d_kw.log_density(x), places=12)

    def test_estimator_keyword_aliases(self):
        e1 = IntegerCategoricalEstimator(min_val=0, max_val=3)
        e2 = IntegerCategoricalEstimator(min_val=0, max_val=3)
        self.assertEqual((e1.min_val, e1.max_val), (e2.min_val, e2.max_val))


if __name__ == "__main__":
    unittest.main()
