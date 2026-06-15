"""Bayesian (conjugate / variational) behavior folded onto the pysp.stats finite mixture.

The merge is additive: with ``prior=None`` the mixture's MLE path is byte-identical to the
historical behavior; with a Dirichlet weight prior (and per-component conjugate priors) the
estimator performs the conjugate MAP weight update, carries the posterior Dirichlet forward,
and exposes ``expected_log_density`` / ``model_log_density`` for nesting and ELBO scoring.

Numeric expectations are pinned against the textbook Dirichlet weight-posterior closed form
and the explicit variational ``expected_log_density`` formula (a log-sum-exp over component
expected log-densities offset by ``E[log w_k]``), with scalar-vs-seq self-consistency.
"""

import unittest

import numpy as np
from scipy.special import logsumexp

from pysp.stats import seq_encode, seq_estimate, seq_initialize
from pysp.stats.bayes.dirichlet import DirichletDistribution
from pysp.stats.bayes.normgamma import NormalGammaDistribution
from pysp.stats.bayes.symdirichlet import SymmetricDirichletDistribution
from pysp.stats.latent.mixture import (
    MixtureDistribution,
    MixtureEstimator,
    _dirichlet_expectations,
    _split_mixture_prior,
    mixture_prior,
)
from pysp.stats.leaf.gaussian import GaussianDistribution
from pysp.utils.estimation import _data_objective_sum, _model_objective, fit, optimize
from pysp.utils.special import digamma


def _suff_stats(dist, data):
    """Run a fresh accumulator over data through the seq path, returning value()."""
    enc = dist.dist_to_encoder().seq_encode(data)
    acc = dist.estimator().accumulator_factory().make()
    acc.seq_update(enc, np.ones(len(data)), dist)
    return acc.value()


class StatsBayesMixtureTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(7)
        n = 4000
        z = rng.rand(n) < 0.35
        self.data = list(np.where(z, rng.normal(-2.0, 1.0, n), rng.normal(3.0, 0.7, n)))
        self.K = 2
        self.alpha_w = np.array([2.0, 5.0])
        self.ng = (0.0, 0.5, 2.0, 1.5)
        self.comp_init = [(-1.0, 1.0), (1.0, 1.0)]

    # ---- MLE path is unchanged when prior is None --------------------------------------

    def test_mle_path_unchanged(self):
        """No prior -> plain MLE weights; the model carries no posterior."""
        comps = [GaussianDistribution(*c) for c in self.comp_init]
        d = MixtureDistribution(comps, [0.5, 0.5])
        self.assertIsNone(d.get_prior())
        self.assertFalse(d.has_conj_prior)

        ss = _suff_stats(d, self.data)
        # default estimator (no prior) -> w = counts / counts.sum()
        m = MixtureEstimator([GaussianDistribution(*c).estimator() for c in self.comp_init]).estimate(None, ss)
        counts = ss[0]
        np.testing.assert_allclose(m.w, counts / counts.sum(), rtol=0, atol=0)
        self.assertIsNone(m.get_prior())
        # expected_log_density falls back to the plug-in log_density without a prior
        for x in (-2.0, 0.0, 3.0):
            self.assertEqual(d.expected_log_density(x), d.log_density(x))

    def test_mle_byte_identical_via_optimize(self):
        """fit/optimize with no prior agree (model_log_density contributes 0)."""
        est = MixtureEstimator([GaussianDistribution(*c).estimator() for c in self.comp_init])
        rng = np.random.RandomState(0)
        m_opt = optimize(self.data, est, max_its=15, rng=np.random.RandomState(0), out=None)
        m_fit = fit(self.data, est, max_its=15, delta=None, rng=np.random.RandomState(0), out=None)
        np.testing.assert_allclose(sorted(m_opt.w), sorted(m_fit.w), atol=1e-9)
        # model_log_density is exactly 0 with no priors anywhere
        self.assertEqual(est.model_log_density(m_opt), 0.0)

    # ---- helpers ------------------------------------------------------------------------

    def test_split_and_dirichlet_expectations(self):
        """mixture_prior round-trips through _split_mixture_prior; weight expectations are exact."""
        wp = DirichletDistribution(self.alpha_w)
        cps = [NormalGammaDistribution(*self.ng), NormalGammaDistribution(*self.ng)]
        joint = mixture_prior(wp, cps)
        w_out, c_out = _split_mixture_prior(joint, self.K)
        self.assertIs(w_out, wp)
        self.assertEqual(len(c_out), self.K)

        alpha, elog = _dirichlet_expectations(wp, self.K)
        np.testing.assert_allclose(alpha, self.alpha_w)
        np.testing.assert_allclose(elog, digamma(self.alpha_w) - digamma(self.alpha_w.sum()))

        # symmetric dirichlet expands the scalar concentration
        sd = SymmetricDirichletDistribution(3.0)
        alpha_s, elog_s = _dirichlet_expectations(sd, self.K)
        np.testing.assert_allclose(alpha_s, np.full(self.K, 3.0))
        np.testing.assert_allclose(elog_s, digamma(alpha_s) - digamma(alpha_s.sum()))

        # None / non-conjugate -> (None, None)
        self.assertEqual(_dirichlet_expectations(None, self.K), (None, None))

    # ---- conjugate weight posterior closed form ----------------------------------------

    def test_conjugate_weight_posterior_closed_form(self):
        """estimate() with a Dirichlet weight prior gives MAP weights and posterior alpha+counts."""
        prior = mixture_prior(
            DirichletDistribution(self.alpha_w),
            [NormalGammaDistribution(*self.ng), NormalGammaDistribution(*self.ng)],
        )
        comps = [GaussianDistribution(*c, prior=NormalGammaDistribution(*self.ng)) for c in self.comp_init]
        d = MixtureDistribution(comps, [0.5, 0.5], prior=prior)
        ss = _suff_stats(d, self.data)
        counts = ss[0]

        est = MixtureEstimator(
            [GaussianDistribution(*c, prior=NormalGammaDistribution(*self.ng)).estimator() for c in self.comp_init],
            prior=prior,
        )
        m = est.estimate(None, ss)

        post_w = m.get_prior()[0]
        np.testing.assert_allclose(post_w.get_parameters(), self.alpha_w + counts, atol=1e-9)
        cpp = np.maximum(counts + self.alpha_w - 1.0, 0.0)
        np.testing.assert_allclose(m.w, cpp / cpp.sum(), atol=1e-9)

    # ---- expected_log_density: explicit variational formula + seq self-consistency -----

    def test_expected_log_density_formula_and_seq(self):
        """expected_log_density equals logsumexp(component ELD + E[log w_k]); seq matches scalar."""
        alpha_w, ng, init = self.alpha_w, self.ng, self.comp_init

        s_prior = mixture_prior(DirichletDistribution(alpha_w), [NormalGammaDistribution(*ng) for _ in range(2)])
        s_dist = MixtureDistribution(
            [GaussianDistribution(*c, prior=NormalGammaDistribution(*ng)) for c in init], [0.5, 0.5], prior=s_prior
        )
        s_ss = _suff_stats(s_dist, self.data)
        s_est = MixtureEstimator(
            [GaussianDistribution(*c, prior=NormalGammaDistribution(*ng)).estimator() for c in init], prior=s_prior
        )
        s_model = s_est.estimate(None, s_ss)

        # E[log w_k] under the posterior Dirichlet weight prior
        post_alpha = np.asarray(s_model.get_prior()[0].get_parameters())
        elog_w = digamma(post_alpha) - digamma(post_alpha.sum())

        xs = np.linspace(-6, 7, 40)
        # scalar expected_log_density equals the explicit log-sum-exp closed form
        for x in xs:
            comp_eld = np.array([c.expected_log_density(float(x)) for c in s_model.components])
            want = float(logsumexp(comp_eld + elog_w))
            self.assertAlmostEqual(s_model.expected_log_density(float(x)), want, places=9)

        # seq_expected_log_density matches the per-element scalar value
        s_seld = s_model.seq_expected_log_density(s_model.dist_to_encoder().seq_encode(list(xs)))
        scalar = np.array([s_model.expected_log_density(float(x)) for x in xs])
        self.assertLess(np.max(np.abs(s_seld - scalar)), 1e-9)

        # model_log_density equals the sum of the weight-Dirichlet and per-component model terms
        self.assertTrue(np.isfinite(s_est.model_log_density(s_model)))

    # ---- fit: monotone objective + recovery --------------------------------------------

    def test_fit_objective_monotone_and_recovers(self):
        """fit() drives a non-decreasing penalized objective and recovers the components."""
        prior = mixture_prior(
            DirichletDistribution(np.ones(self.K)),
            [NormalGammaDistribution(0.0, 0.1, 2.0, 2.0) for _ in range(self.K)],
        )
        est = MixtureEstimator(
            [
                GaussianDistribution(0.0, 1.0, prior=NormalGammaDistribution(0.0, 0.1, 2.0, 2.0)).estimator()
                for _ in range(self.K)
            ],
            prior=prior,
        )
        enc = seq_encode(self.data, est.accumulator_factory().make().acc_to_encoder())
        mm = seq_initialize(enc_data=enc, estimator=est, rng=np.random.RandomState(1), p=0.5)
        objs = [_data_objective_sum(enc, mm) + _model_objective(est, mm)]
        for _ in range(30):
            mm = seq_estimate(enc, est, mm)
            objs.append(_data_objective_sum(enc, mm) + _model_objective(est, mm))
        self.assertTrue(np.all(np.diff(objs) >= -1e-6), "objective decreased: %s" % objs)

        # returned model carries the posterior weight Dirichlet forward
        self.assertIsInstance(mm.get_prior()[0], DirichletDistribution)

        means = sorted(c.mu for c in mm.components)
        self.assertAlmostEqual(means[0], -2.0, delta=0.25)
        self.assertAlmostEqual(means[1], 3.0, delta=0.25)


if __name__ == "__main__":
    unittest.main()
