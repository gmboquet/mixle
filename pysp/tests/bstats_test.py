"""Tests for the Bayesian (conjugate-prior / variational) machinery in pysp.bstats.

Covers:
  - conjugate posterior updates checked against textbook closed forms,
  - prior density evaluations checked against scipy.stats,
  - consistency between log_density, expected_log_density and their seq_* forms,
  - convergence of bestimation.optimize: the penalized objective
    (data term + prior term) must increase monotonically for MAP-EM mixtures
    and the ELBO must increase monotonically for DPM variational inference.
"""
import io
import unittest

import numpy as np
import scipy.stats

from pysp.bstats import (
    BetaDistribution, BinomialDistribution, BinomialEstimator,
    CategoricalDistribution, CategoricalEstimator, DirichletDistribution,
    ExponentialDistribution, ExponentialEstimator, GaussianDistribution,
    GaussianEstimator, GeometricDistribution, LogGaussianDistribution,
    LogGaussianEstimator, MixtureDistribution, MixtureEstimator,
    PoissonDistribution, PoissonEstimator, IntegerCategoricalDistribution,
)
from pysp.bstats.bestimation import optimize, k_fold_split_index
from pysp.bstats.dpm import DirichletProcessMixtureEstimator
from pysp.bstats.normgamma import NormalGammaDistribution
from pysp.bstats.gamma import GammaDistribution
from pysp.utils.special import stirling2


def fit(data, est):
    acc = est.accumulator_factory().make()
    for x in data:
        acc.update(x, 1.0, None)
    return acc, est.estimate(acc.value())


class ConjugateUpdateTestCase(unittest.TestCase):
    """Posterior hyperparameters must match the textbook closed forms."""

    def test_gaussian_normal_gamma_posterior(self):
        mu0, lam0, a0, b0 = 1.0, 2.0, 3.0, 4.0
        est = GaussianEstimator(prior=NormalGammaDistribution(mu0, lam0, a0, b0))
        data = np.array([0.5, 1.5, 2.0, -1.0, 0.0])
        _, d = fit(data, est)

        n = len(data)
        xbar = data.mean()
        mu_n = (lam0*mu0 + data.sum())/(lam0 + n)
        lam_n = lam0 + n
        a_n = a0 + n/2.0
        b_n = b0 + 0.5*np.sum((data - xbar)**2) + 0.5*lam0*n*(xbar - mu0)**2/(lam0 + n)

        pm, pl, pa, pb = d.prior.get_parameters()
        self.assertAlmostEqual(pm, mu_n, places=10)
        self.assertAlmostEqual(pl, lam_n, places=10)
        self.assertAlmostEqual(pa, a_n, places=10)
        self.assertAlmostEqual(pb, b_n, places=8)
        # MAP of the joint normal-gamma: sigma2 = b_n / (a_n - 1/2)
        self.assertAlmostEqual(d.sigma2, pb/(pa - 0.5), places=10)

    def test_poisson_gamma_posterior(self):
        k0, theta0 = 2.0, 0.5
        est = PoissonEstimator(prior=GammaDistribution(k0, theta0))
        data = np.array([1, 0, 3, 2, 4])
        _, d = fit(data, est)

        k_n = k0 + data.sum()
        theta_n = theta0/(len(data)*theta0 + 1.0)
        pk, pt = d.prior.get_parameters()
        self.assertAlmostEqual(pk, k_n, places=10)
        self.assertAlmostEqual(pt, theta_n, places=10)
        self.assertAlmostEqual(d.lam, (k_n - 1.0)*theta_n, places=10)

    def test_binomial_beta_posterior(self):
        a0, b0, n = 2.0, 3.0, 10
        est = BinomialEstimator(n, prior=BetaDistribution(a0, b0))
        data = np.array([3, 7, 5, 2])
        _, d = fit(data, est)

        a_n = a0 + data.sum()
        b_n = b0 + len(data)*n - data.sum()
        pa, pb = d.prior.get_parameters()
        self.assertAlmostEqual(pa, a_n, places=10)
        self.assertAlmostEqual(pb, b_n, places=10)
        self.assertAlmostEqual(d.p, (a_n - 1.0)/(a_n + b_n - 2.0), places=10)

    def test_exponential_gamma_posterior(self):
        # bstats exponential is rate-parameterized; Gamma(k, theta) prior on
        # the rate with scale theta; posterior shape k + n, rate 1/theta + sum
        k0, theta0 = 2.0, 0.25
        est = ExponentialEstimator(prior=GammaDistribution(k0, theta0))
        data = np.array([0.5, 1.5, 1.0])
        _, d = fit(data, est)

        pk, pt = d.prior.get_parameters()
        self.assertAlmostEqual(pk, k0 + len(data), places=10)
        self.assertAlmostEqual(1.0/pt, 1.0/theta0 + data.sum(), places=8)
        self.assertAlmostEqual(d.lam, (pk - 1.0)/(1.0/pt), places=8)

    def test_categorical_dirichlet_posterior(self):
        from pysp.bstats.catdirichlet import DictDirichletDistribution
        est = CategoricalEstimator(prior=DictDirichletDistribution({'a': 2.0, 'b': 1.5, 'c': 1.0}))
        data = ['a', 'a', 'b', 'c', 'a', 'b']
        _, d = fit(data, est)

        cpp = d.prior.get_parameters()
        self.assertAlmostEqual(cpp['a'], 2.0 + 3, places=10)
        self.assertAlmostEqual(cpp['b'], 1.5 + 2, places=10)
        self.assertAlmostEqual(cpp['c'], 1.0 + 1, places=10)
        # probabilities are a valid distribution
        self.assertAlmostEqual(sum(np.exp(d.log_density(k)) for k in 'abc'), 1.0, places=10)


class PriorDensityTestCase(unittest.TestCase):

    def test_beta_log_density_matches_scipy(self):
        d = BetaDistribution(2.5, 3.5)
        for x in [0.1, 0.5, 0.9]:
            self.assertAlmostEqual(d.log_density(x), scipy.stats.beta.logpdf(x, 2.5, 3.5), places=10)
            self.assertAlmostEqual(d.density(x), scipy.stats.beta.pdf(x, 2.5, 3.5), places=10)

    def test_beta_entropy_matches_scipy(self):
        d = BetaDistribution(2.5, 3.5)
        self.assertAlmostEqual(d.entropy(), scipy.stats.beta.entropy(2.5, 3.5), places=10)
        # cross entropy with itself equals entropy
        self.assertAlmostEqual(d.cross_entropy(d), d.entropy(), places=10)

    def test_normal_gamma_log_density(self):
        d = NormalGammaDistribution(1.0, 2.0, 3.0, 4.0)
        mu, tau = 0.5, 1.5
        ref = (scipy.stats.gamma.logpdf(tau, 3.0, scale=1.0/4.0)
               + scipy.stats.norm.logpdf(mu, loc=1.0, scale=np.sqrt(1.0/(2.0*tau))))
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

    def test_dirichlet_entropy_matches_scipy(self):
        alpha = np.array([1.5, 2.5, 3.0])
        d = DirichletDistribution(alpha)
        self.assertAlmostEqual(d.entropy(), scipy.stats.dirichlet.entropy(alpha), places=10)


class DensityConsistencyTestCase(unittest.TestCase):

    def battery(self):
        return [
            (GaussianDistribution(1.0, 2.0), None),
            (LogGaussianDistribution(0.5, 1.5), None),
            (PoissonDistribution(4.0), None),
            (ExponentialDistribution(2.0), None),
            (GeometricDistribution(0.3), None),
            (BinomialDistribution(8, 0.4), None),
            (IntegerCategoricalDistribution([0.2, 0.3, 0.5]), None),
        ]

    def test_seq_log_density_matches_scalar(self):
        for dist, _ in self.battery():
            data = dist.sampler(seed=3).sample(size=40)
            enc = dist.seq_encode(data)
            seq_ll = dist.seq_log_density(enc)
            scalar_ll = np.asarray([dist.log_density(x) for x in data])
            self.assertTrue(np.allclose(seq_ll, scalar_ll, rtol=1e-10, atol=1e-12),
                            'seq/scalar mismatch for %s' % str(dist))

    def test_seq_expected_log_density_matches_scalar(self):
        for dist, _ in self.battery():
            data = dist.sampler(seed=4).sample(size=40)
            enc = dist.seq_encode(data)
            seq_ell = dist.seq_expected_log_density(enc)
            scalar_ell = np.asarray([dist.expected_log_density(x) for x in data])
            self.assertTrue(np.allclose(seq_ell, scalar_ell, rtol=1e-10, atol=1e-12),
                            'seq/scalar expected mismatch for %s' % str(dist))

    def test_expected_log_density_approaches_log_density_for_sharp_prior(self):
        # with a very concentrated prior, E_q[log p(x|theta)] -> log p(x|theta_0)
        mu0, s2 = 1.0, 2.0
        n_pseudo = 1.0e8
        prior = NormalGammaDistribution(mu0, n_pseudo, n_pseudo/2.0, (n_pseudo/2.0)*s2)
        d = GaussianDistribution(mu0, s2, prior=prior)
        for x in [-1.0, 1.0, 3.0]:
            self.assertAlmostEqual(d.expected_log_density(x), d.log_density(x), places=5)


class OptimizeConvergenceTestCase(unittest.TestCase):

    @staticmethod
    def run_optimize(est, data, max_its, seed=2):
        buf = io.StringIO()
        model = optimize(data, est, max_its=max_its, delta=1.0e-9,
                         rng=np.random.RandomState(seed), out=buf, print_iter=1)
        lines = buf.getvalue().splitlines()
        objs = [float(l.split('OBJ=')[1].split(',')[0]) for l in lines if 'OBJ=' in l]
        return model, np.asarray(objs)

    def test_map_em_mixture_objective_monotone(self):
        truth = MixtureDistribution(
            [GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.4, 0.6])
        data = truth.sampler(seed=1).sample(800)

        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        model, objs = self.run_optimize(est, data, max_its=40)

        self.assertGreater(len(objs), 3)
        self.assertTrue(np.all(np.diff(objs) >= -1.0e-6),
                        'penalized objective decreased: %s' % str(np.diff(objs)))

        mus = sorted(c.mu for c in model.components)
        self.assertAlmostEqual(mus[0], -3.0, delta=0.3)
        self.assertAlmostEqual(mus[1], 3.0, delta=0.3)

    def test_dpm_elbo_monotone(self):
        truth = MixtureDistribution(
            [GaussianDistribution(-4.0, 0.5), GaussianDistribution(0.0, 1.0),
             GaussianDistribution(5.0, 2.0)], [0.3, 0.4, 0.3])
        data = truth.sampler(seed=1).sample(800)

        K = 10
        est = DirichletProcessMixtureEstimator([GaussianEstimator() for _ in range(K)])
        model, objs = self.run_optimize(est, data, max_its=25, seed=3)

        self.assertGreater(len(objs), 5)
        self.assertTrue(np.all(np.diff(objs) >= -1.0e-5),
                        'ELBO decreased: %s' % str(np.diff(objs)))
        # weights are a valid distribution, sorted decreasing by usage
        self.assertAlmostEqual(model.w.sum(), 1.0, places=8)
        self.assertTrue(np.all(model.w >= 0))

    def test_optimize_runs_with_plain_estimator(self):
        # model_log_density falls back gracefully for non-mixture estimators
        data = GaussianDistribution(2.0, 1.0).sampler(seed=5).sample(200)
        buf = io.StringIO()
        model = optimize(data, GaussianEstimator(), max_its=5, rng=np.random.RandomState(1), out=buf)
        self.assertAlmostEqual(model.mu, 2.0, delta=0.3)


class NormalWishartTestCase(unittest.TestCase):

    def test_one_dim_equals_normal_gamma(self):
        # NW(mu, kappa, W=1/(2b), nu=2a) in 1-d must equal NG(mu, kappa, a, b)
        from pysp.bstats.normwishart import NormalWishartDistribution
        ng = NormalGammaDistribution(0.5, 2.0, 3.0, 4.0)
        nw = NormalWishartDistribution([0.5], 2.0, [[1.0/(2*4.0)]], 2*3.0)
        for mu, tau in [(0.7, 1.3), (-1.0, 0.2), (0.5, 3.0)]:
            self.assertAlmostEqual(ng.log_density((mu, tau)), nw.log_density(([mu], [[tau]])), places=10)
        self.assertAlmostEqual(ng.entropy(), nw.entropy(), places=8)

    def test_cross_entropy_self_is_entropy(self):
        from pysp.bstats.normwishart import NormalWishartDistribution
        nw = NormalWishartDistribution(np.array([1.0, -1.0]), 2.0, np.eye(2)*0.4, 5.0)
        self.assertAlmostEqual(nw.cross_entropy(nw), nw.entropy(), places=10)

    def test_sampler_runs(self):
        from pysp.bstats.normwishart import NormalWishartDistribution
        nw = NormalWishartDistribution(np.zeros(2), 1.0, np.eye(2), 4.0)
        for mu, lam in nw.sampler(seed=1).sample(size=3):
            self.assertTrue(np.all(np.isfinite(mu)))
            self.assertTrue(np.all(np.linalg.eigvalsh(lam) > 0))


class MultivariateGaussianTestCase(unittest.TestCase):

    def make_dist(self):
        from pysp.bstats import MultivariateGaussianDistribution
        return MultivariateGaussianDistribution([1.0, -2.0], [[2.0, 0.6], [0.6, 1.0]])

    def test_log_density_matches_scipy(self):
        d = self.make_dist()
        for x in [np.array([0.0, 0.0]), np.array([2.0, -3.0])]:
            ref = scipy.stats.multivariate_normal.logpdf(x, d.mu, d.covar)
            self.assertAlmostEqual(d.log_density(x), ref, places=10)

    def test_seq_matches_scalar(self):
        d = self.make_dist()
        data = d.sampler(seed=1).sample(size=30)
        enc = d.seq_encode(data)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in data]))
        self.assertTrue(np.allclose(d.seq_expected_log_density(enc),
                                    [d.expected_log_density(u) for u in data]))

    def test_normal_wishart_posterior_closed_form(self):
        from pysp.bstats import MultivariateGaussianEstimator
        from pysp.bstats.normwishart import NormalWishartDistribution

        m0 = np.array([0.5, -0.5])
        kappa0, nu0 = 2.0, 4.0
        w0 = np.eye(2)*0.5
        est = MultivariateGaussianEstimator(2, prior=NormalWishartDistribution(m0, kappa0, w0, nu0))

        data = self.make_dist().sampler(seed=2).sample(size=25)
        _, d = fit(data, est)

        xs = np.asarray(data)
        n = len(xs)
        xbar = xs.mean(axis=0)
        scatter = np.einsum('ni,nj->ij', xs - xbar, xs - xbar)

        m_n, kappa_n, w_n, nu_n = d.prior.get_parameters()
        self.assertAlmostEqual(kappa_n, kappa0 + n, places=10)
        self.assertAlmostEqual(nu_n, nu0 + n, places=10)
        self.assertTrue(np.allclose(m_n, (kappa0*m0 + n*xbar)/(kappa0 + n)))

        dmu = xbar - m0
        w_n_inv_ref = np.linalg.inv(w0) + scatter + (kappa0*n/(kappa0 + n))*np.outer(dmu, dmu)
        self.assertTrue(np.allclose(np.linalg.inv(w_n), w_n_inv_ref, rtol=1e-8))

        # MAP covariance: W_n^-1 / (nu_n - d)
        self.assertTrue(np.allclose(d.covar, w_n_inv_ref/(nu_n - 2), rtol=1e-8))

    def test_recovery(self):
        from pysp.bstats import MultivariateGaussianEstimator
        truth = self.make_dist()
        data = truth.sampler(seed=3).sample(size=4000)
        _, m = fit(data, MultivariateGaussianEstimator(2))
        self.assertTrue(np.allclose(m.mu, truth.mu, atol=0.1))
        self.assertTrue(np.allclose(m.covar, truth.covar, atol=0.15))


class MarkovChainTestCase(unittest.TestCase):

    def make_dist(self):
        from pysp.bstats import MarkovChainDistribution
        pi = np.array([0.6, 0.3, 0.1])
        a_mat = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
        return MarkovChainDistribution(pi, a_mat, len_dist=CategoricalDistribution({12: 1.0}))

    def test_seq_matches_scalar(self):
        d = self.make_dist()
        seqs = d.sampler(seed=1).sample(size=25)
        enc = d.seq_encode(seqs)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in seqs]))
        self.assertTrue(np.allclose(d.seq_expected_log_density(enc),
                                    [d.expected_log_density(u) for u in seqs]))

    def test_dirichlet_posterior_counts(self):
        from pysp.bstats import MarkovChainEstimator
        est = MarkovChainEstimator(2, prior=(DirichletDistribution(np.array([2.0, 3.0])),
                                             [DirichletDistribution(np.array([1.5, 1.5])),
                                              DirichletDistribution(np.array([1.0, 2.0]))]))
        seqs = [[0, 0, 1], [1, 1, 0], [0, 1, 1]]
        _, d = fit(seqs, est)

        prior = d.get_prior()
        init_posterior, row_posteriors = prior.dists[0], prior.dists[1].dists
        # init counts: state0 twice, state1 once
        self.assertTrue(np.allclose(init_posterior.get_parameters(), [2.0 + 2, 3.0 + 1]))
        # transitions from 0: 0->0 once, 0->1 twice ; from 1: 1->1 twice, 1->0 once
        self.assertTrue(np.allclose(row_posteriors[0].get_parameters(), [1.5 + 1, 1.5 + 2]))
        self.assertTrue(np.allclose(row_posteriors[1].get_parameters(), [1.0 + 1, 2.0 + 2]))

    def test_recovery(self):
        from pysp.bstats import MarkovChainEstimator
        truth = self.make_dist()
        seqs = truth.sampler(seed=2).sample(size=800)
        est = MarkovChainEstimator(3, len_estimator=CategoricalEstimator())
        _, m = fit(seqs, est)
        self.assertTrue(np.allclose(m.init_prob_vec, truth.init_prob_vec, atol=0.06))
        self.assertTrue(np.allclose(m.transition_mat, truth.transition_mat, atol=0.05))


class HiddenMarkovModelTestCase(unittest.TestCase):

    def make_dist(self):
        from pysp.bstats import HiddenMarkovModelDistribution
        topics = [GaussianDistribution(-5.0, 1.0), GaussianDistribution(5.0, 1.0)]
        return HiddenMarkovModelDistribution(topics, [0.7, 0.3], [[0.9, 0.1], [0.2, 0.8]],
                                             len_dist=CategoricalDistribution({10: 1.0}))

    def test_seq_matches_scalar(self):
        d = self.make_dist()
        seqs = d.sampler(seed=1).sample(size=15)
        enc = d.seq_encode(seqs)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(u) for u in seqs]))

    def test_forward_matches_brute_force(self):
        # forward log-likelihood vs explicit sum over all state paths
        d = self.make_dist()
        x = d.sampler(seed=2).sample_seq(4)
        import itertools
        tot = -np.inf
        for path in itertools.product(range(2), repeat=len(x)):
            lp = np.log(d.w[path[0]]) + d.topics[path[0]].log_density(x[0])
            for t in range(1, len(x)):
                lp += np.log(d.transitions[path[t-1], path[t]]) + d.topics[path[t]].log_density(x[t])
            tot = np.logaddexp(tot, lp)
        tot += d.len_dist.log_density(len(x))
        self.assertAlmostEqual(d.log_density(x), tot, places=8)

    def test_viterbi_recovers_separated_states(self):
        d = self.make_dist()
        x = [-5.1, -4.9, 5.2, 4.8, -5.0]
        self.assertEqual(d.viterbi(x), [0, 0, 1, 1, 0])

    def test_optimize_monotone_and_recovery(self):
        from pysp.bstats import HiddenMarkovModelEstimator
        truth = self.make_dist()
        seqs = truth.sampler(seed=3).sample(size=200)

        est = HiddenMarkovModelEstimator([GaussianEstimator(), GaussianEstimator()],
                                         len_estimator=CategoricalEstimator())
        buf = io.StringIO()
        m = optimize(seqs, est, max_its=30, delta=1.0e-9, rng=np.random.RandomState(4),
                     out=buf, print_iter=1)

        objs = [float(l.split('OBJ=')[1].split(',')[0]) for l in buf.getvalue().splitlines() if 'OBJ=' in l]
        self.assertGreater(len(objs), 3)
        self.assertTrue(np.all(np.diff(objs) >= -1.0e-6),
                        'HMM penalized objective decreased: %s' % str(np.diff(objs)))

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
        from pysp.bstats import HiddenMarkovModelDistribution
        len_d = CategoricalDistribution({12: 1.0})
        h1 = HiddenMarkovModelDistribution(
            [GaussianDistribution(-6.0, 1.0), GaussianDistribution(-1.0, 1.0)],
            [0.5, 0.5], [[0.95, 0.05], [0.05, 0.95]], len_dist=len_d)
        h2 = HiddenMarkovModelDistribution(
            [GaussianDistribution(2.0, 1.0), GaussianDistribution(7.0, 1.0)],
            [0.5, 0.5], [[0.3, 0.7], [0.7, 0.3]], len_dist=len_d)
        return MixtureDistribution([h1, h2], [0.6, 0.4])

    @staticmethod
    def hmm_est():
        from pysp.bstats import HiddenMarkovModelEstimator
        return HiddenMarkovModelEstimator([GaussianEstimator(), GaussianEstimator()],
                                          len_estimator=CategoricalEstimator())

    def test_hmm_prior_is_composable(self):
        from pysp.bstats.composite import CompositeDistribution
        truth = self.make_truth()
        hmm = truth.components[0]
        p = hmm.get_prior()
        self.assertIsInstance(p, CompositeDistribution)
        # cross-entropy of the prior with itself equals its entropy
        self.assertAlmostEqual(p.cross_entropy(p), p.entropy(), places=8)
        # set_prior round-trips through the composable form
        hmm.set_prior(p)
        self.assertIsInstance(hmm.get_prior(), CompositeDistribution)

    def test_mixture_of_hmms_monotone(self):
        truth = self.make_truth()
        data = truth.sampler(seed=1).sample(150)

        est = MixtureEstimator([self.hmm_est(), self.hmm_est()])
        model, objs = OptimizeConvergenceTestCase.run_optimize(est, data, max_its=15)

        self.assertTrue(np.all(np.diff(objs) >= -1.0e-6),
                        'mixture-of-HMMs objective decreased')
        self.assertAlmostEqual(model.w.sum(), 1.0, places=10)

    def test_dpm_of_hmms_monotone(self):
        from pysp.bstats.dpm import DirichletProcessMixtureEstimator
        truth = self.make_truth()
        data = truth.sampler(seed=1).sample(150)

        est = DirichletProcessMixtureEstimator([self.hmm_est() for _ in range(4)])
        model, objs = OptimizeConvergenceTestCase.run_optimize(est, data, max_its=15, seed=3)

        self.assertTrue(np.all(np.diff(objs) >= -1.0e-5), 'DPM-of-HMMs ELBO decreased')
        self.assertAlmostEqual(model.w.sum(), 1.0, places=8)
        self.assertTrue(np.all(model.w >= 0))


class HierarchicalDPMTestCase(unittest.TestCase):

    @staticmethod
    def make_truth():
        from pysp.bstats import HierarchicalDirichletProcessMixtureDistribution
        atoms = [GaussianDistribution(m, 0.5) for m in [-6.0, -2.0, 2.0, 6.0]]
        return HierarchicalDirichletProcessMixtureDistribution(
            atoms, beta=[0.4, 0.3, 0.2, 0.1], alpha=3.0, gamma=2.0,
            len_dist=CategoricalDistribution({40: 1.0}))

    @classmethod
    def fit_model(cls, groups, k=8, max_its=60, seed=2):
        from pysp.bstats import HierarchicalDirichletProcessMixtureEstimator
        est = HierarchicalDirichletProcessMixtureEstimator(
            [GaussianEstimator() for _ in range(k)], gamma=2.0, alpha=3.0,
            len_estimator=CategoricalEstimator())
        return OptimizeConvergenceTestCase.run_optimize(est, groups, max_its=max_its, seed=seed)

    def test_objective_increases_and_recovers_atoms(self):
        truth = self.make_truth()
        groups = truth.sampler(seed=1).sample(size=60)
        model, objs = self.fit_model(groups)

        self.assertGreater(objs[-1], objs[0])
        # every true atom location is covered by some effective fitted atom
        true_mus = [-6.0, -2.0, 2.0, 6.0]
        eff_mus = [model.components[i].mu for i in np.flatnonzero(model.beta > 0.03)]
        for tm in true_mus:
            self.assertTrue(min(abs(tm - em) for em in eff_mus) < 0.6,
                            'true atom %.1f not recovered (got %s)' % (tm, str(eff_mus)))

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
        enc = model.seq_encode(held_out)
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

    def test_poisson_moments_via_stirling(self):
        d = PoissonDistribution(2.0)
        # E[X^2] = lam + lam^2 ; E[X^3] = lam + 3 lam^2 + lam^3
        self.assertAlmostEqual(d.moment(2), 2.0 + 4.0, places=10)
        self.assertAlmostEqual(d.moment(3), 2.0 + 3*4.0 + 8.0, places=10)
        self.assertEqual(stirling2(6, 3), 90)


if __name__ == '__main__':
    unittest.main()
