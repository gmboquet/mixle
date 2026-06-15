"""Bayesian (conjugate / variational) behavior folded onto pysp.stats BernoulliSet.

Follows the proven bstats -> stats merge template: the frequentist Bernoulli-set leaf gains a
per-element Beta conjugate posterior estimate, ``expected_log_density``, and a posterior-returning
estimate while its MLE path (``prior=None``) stays byte-identical. Numeric expectations mirror the
historical ``pysp.bstats.setdist`` assertions.
"""

import unittest

import numpy as np

import pysp.bstats.setdist as bsetdist
from pysp.bstats.beta import BetaDistribution as BBetaDistribution
from pysp.stats.beta import BetaDistribution
from pysp.stats.setdist import (
    BernoulliSetDataEncoder,
    BernoulliSetDistribution,
    BernoulliSetEstimator,
)
from pysp.utils.special import digamma


class StatsBayesSetDistTestCase(unittest.TestCase):
    def setUp(self):
        # Inclusion counts (obs_cnt) and total weighted set count (tot_cnt).
        self.obs = {"a": 7.0, "b": 3.0, "c": 4.0, "d": 1.0}
        self.tot = 10.0
        self.suff_stat = (dict(self.obs), self.tot)
        # Asymmetric prior keeps every posterior mode strictly interior to (0, 1).
        self.a0, self.b0 = 2.0, 3.0

    # ------------------------------------------------------------------ MLE
    def test_mle_path_unchanged(self):
        """prior=None -> plain relative-frequency point estimate; no posterior attached."""
        m = BernoulliSetEstimator().estimate(None, self.suff_stat)
        for k, v in self.obs.items():
            self.assertAlmostEqual(m.pmap[k], v / self.tot, places=14)
        self.assertIsNone(m.get_prior())
        self.assertIsNone(m.get_posteriors())
        self.assertFalse(m.has_conj_prior)

    def test_distribution_mle_default_has_no_prior(self):
        """Constructing a distribution without a prior leaves it a plain point model."""
        d = BernoulliSetDistribution({"a": 0.6, "b": 0.2})
        self.assertIsNone(d.get_prior())
        self.assertFalse(d.has_conj_prior)
        self.assertAlmostEqual(d.expected_log_density(["a"]), d.log_density(["a"]), places=14)

    def test_estimator_propagates_prior(self):
        """A fitted conjugate model's estimator() carries the same prior forward."""
        prior = BetaDistribution(self.a0, self.b0)
        m = BernoulliSetEstimator(prior=prior).estimate(None, self.suff_stat)
        est2 = m.estimator()
        self.assertTrue(est2.has_conj_prior)
        self.assertIs(est2.prior, prior)

    # ------------------------------------------------- conjugate posteriors
    def test_conjugate_posterior_closed_form(self):
        """Per-element posterior is Beta(a + v, b + tot - v)."""
        m = BernoulliSetEstimator(prior=BetaDistribution(self.a0, self.b0)).estimate(None, self.suff_stat)
        post = m.get_posteriors()
        for k, v in self.obs.items():
            pa, pb = post[k]
            self.assertAlmostEqual(pa, self.a0 + v, places=12)
            self.assertAlmostEqual(pb, self.b0 + (self.tot - v), places=12)

    def test_posterior_mode_parity_with_bstats(self):
        """Posterior-mode inclusion probabilities match bstats (decoding its negative encoding)."""
        m = BernoulliSetEstimator(prior=BetaDistribution(self.a0, self.b0)).estimate(None, self.suff_stat)
        bp = bsetdist.bernoulli_beta_posterior_mode(self.obs, self.tot, (self.a0, self.b0))
        bp_plain = {k: (1.0 + v if v < 0 else v) for k, v in bp.items()}
        maxdiff = max(abs(bp_plain[k] - m.pmap[k]) for k in self.obs)
        self.assertLess(maxdiff, 1.0e-9)

    # ----------------------------------------------- expected_log_density
    def _ref_expected_log_density(self, post, x):
        rv = 0.0
        for _k, (a, b) in post.items():
            rv += digamma(b) - digamma(a + b)
        for u in x:
            a, b = post[u]
            rv += (digamma(a) - digamma(a + b)) - (digamma(b) - digamma(a + b))
        return rv

    def test_expected_log_density_formula(self):
        """expected_log_density equals the VB E[log p] closed form (scalar + seq)."""
        m = BernoulliSetEstimator(prior=BetaDistribution(self.a0, self.b0)).estimate(None, self.suff_stat)
        post = m.get_posteriors()
        xs_list = [["a", "c"], ["b"], [], ["a", "b", "c", "d"]]
        for x in xs_list:
            self.assertAlmostEqual(m.expected_log_density(x), self._ref_expected_log_density(post, x), places=12)
        enc = BernoulliSetDataEncoder().seq_encode(xs_list)
        seq = m.seq_expected_log_density(enc)
        ref = np.array([self._ref_expected_log_density(post, x) for x in xs_list])
        self.assertTrue(np.allclose(seq, ref, atol=1.0e-12))
        # scalar vs seq self-consistency
        scalar = np.array([m.expected_log_density(x) for x in xs_list])
        self.assertTrue(np.allclose(seq, scalar, atol=1.0e-12))

    def test_expected_log_density_falls_back_without_prior(self):
        """Without a prior, expected_log_density reduces to the plug-in log_density."""
        m = BernoulliSetEstimator().estimate(None, self.suff_stat)
        for x in (["a"], ["a", "c"], []):
            self.assertAlmostEqual(m.expected_log_density(x), m.log_density(x), places=12)
        enc = BernoulliSetDataEncoder().seq_encode([["a"], ["a", "c"], []])
        self.assertTrue(np.allclose(m.seq_expected_log_density(enc), m.seq_log_density(enc), atol=1.0e-12))

    # ------------------------------------------------- model_log_density
    def test_model_log_density_parity_with_bstats_beta(self):
        """model_log_density equals the summed Beta log-prior (parity vs bstats Beta)."""
        prior = BetaDistribution(self.a0, self.b0)
        m = BernoulliSetEstimator(prior=prior).estimate(None, self.suff_stat)
        sm = BernoulliSetEstimator(prior=prior).model_log_density(m)
        ref = sum(BBetaDistribution(self.a0, self.b0).log_density(p) for p in m.pmap.values())
        self.assertAlmostEqual(sm, ref, places=12)

    def test_model_log_density_zero_without_prior(self):
        """No prior -> zero global ELBO term."""
        m = BernoulliSetEstimator().estimate(None, self.suff_stat)
        self.assertEqual(BernoulliSetEstimator().model_log_density(m), 0.0)


if __name__ == "__main__":
    unittest.main()
