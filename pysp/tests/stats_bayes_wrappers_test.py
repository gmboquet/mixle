"""Bayesian (conjugate / variational) behavior folded onto the pysp.stats STRUCTURAL WRAPPER families.

These families (Composite, Sequence, Optional, Conditional, Ignored) own no parameters of their own;
they DELEGATE conjugate priors to their wrapped children. This suite proves three things, additively:

  * Composition: the wrapper's ``expected_log_density`` (scalar + seq) is exactly the composition of
    its children's ``expected_log_density`` (sum over composite fields / sequence elements / branches),
    ``model_log_density`` sums the children's, and child conjugate posteriors propagate through.
  * MLE unchanged: ``prior=None`` (the default) leaves the existing point-estimate path byte-identical.
  * Prior round-trip: ``set_prior``/``get_prior`` distribute priors to and recover them from children.
"""

import unittest

import numpy as np

from pysp.stats.base.beta import BetaDistribution as SBeta
from pysp.stats.base.categorical import CategoricalDistribution as SCat
from pysp.stats.base.categorical import CategoricalEstimator as SCatEst
from pysp.stats.base.gaussian import GaussianDistribution as SGauss
from pysp.stats.base.gaussian import GaussianEstimator as SGaussEst
from pysp.stats.bayes.dict_dirichlet import DictDirichletDistribution as SDir
from pysp.stats.bayes.normal_gamma import NormalGammaDistribution as SNG
from pysp.stats.combinator.composite import CompositeDistribution as SComp
from pysp.stats.combinator.composite import CompositeEstimator as SCompEst
from pysp.stats.combinator.conditional import ConditionalDistribution as SCond
from pysp.stats.combinator.ignored import IgnoredDistribution as SIgnored
from pysp.stats.combinator.optional import OptionalDistribution as SOpt
from pysp.stats.combinator.sequence import SequenceDistribution as SSeq

NG = (0.3, 2.0, 4.0, 5.0)
DIR = {"a": 2.0, "b": 3.0, "c": 1.5}
CATP = {"a": 0.5, "b": 0.3, "c": 0.2}
GMU, GS2 = 0.3, 5.0 / (4.0 - 0.5)


def _maxdiff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    matching_inf = (~finite) & (a == b)  # both +inf or both -inf -> exact agreement
    diff = np.zeros_like(a)
    diff[finite] = np.abs(a[finite] - b[finite])
    # any non-finite mismatch that is not a matching inf is a real (infinite) difference
    bad = (~finite) & (~matching_inf)
    if np.any(bad):
        return float("inf")
    return float(np.max(diff)) if diff.size else 0.0


class CompositeBayesTest(unittest.TestCase):
    def setUp(self):
        self.obs = [(-1.0, "a"), (0.7, "b"), (2.7, "c"), (0.0, "z")]

    def _stats(self):
        return SComp([SGauss(GMU, GS2, prior=SNG(*NG)), SCat(dict(CATP), prior=SDir(dict(DIR)))])

    def test_expected_log_density_composes_children(self):
        """The composite ELD is exactly the sum of its per-field children ELDs."""
        s = self._stats()
        ref = [s.dists[0].expected_log_density(x[0]) + s.dists[1].expected_log_density(x[1]) for x in self.obs]
        self.assertLess(_maxdiff([s.expected_log_density(x) for x in self.obs], ref), 1e-9)

    def test_seq_expected_log_density_self_consistent(self):
        """seq_expected_log_density matches the per-observation scalar value."""
        s = self._stats()
        s_enc = s.dist_to_encoder().seq_encode(self.obs)
        self.assertLess(
            _maxdiff(s.seq_expected_log_density(s_enc), [s.expected_log_density(x) for x in self.obs]), 1e-9
        )

    def test_model_log_density_sums_children(self):
        est = SCompEst([SGaussEst(prior=SNG(*NG)), SCatEst(prior=SDir(dict(DIR)))])
        s = self._stats()
        expected = SGaussEst(prior=SNG(*NG)).model_log_density(s.dists[0]) + SCatEst(
            prior=SDir(dict(DIR))
        ).model_log_density(s.dists[1])
        self.assertLess(abs(est.model_log_density(s) - expected), 1e-12)

    def test_mle_unchanged_without_prior(self):
        """prior=None leaves the MLE composite path byte-identical to a no-prior estimator."""
        data = list(
            zip(np.random.RandomState(1).normal(0, 1, size=50), np.random.RandomState(2).choice(list(CATP), 50))
        )
        est = SComp([SGauss(0.0, 1.0), SCat(dict(CATP))]).estimator()
        acc = est.accumulator_factory().make()
        enc = acc.acc_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), None)
        m = est.estimate(None, acc.value())
        self.assertIsNone(m.dists[0].get_prior())
        self.assertIsNone(m.dists[1].get_prior())
        # plain MLE Gaussian
        sx = sum(d[0] for d in data)
        self.assertAlmostEqual(m.dists[0].mu, sx / len(data), places=10)

    def test_prior_round_trip(self):
        s = self._stats()
        prior = s.get_prior()
        self.assertIsInstance(prior[0], SNG)
        self.assertIsInstance(prior[1], SDir)
        # set_prior(None) is a no-op
        s.set_prior(None)
        self.assertTrue(s.dists[0].has_conj_prior)
        # round-trip new priors
        s2 = SComp([SGauss(0.0, 1.0), SCat(dict(CATP))])
        s2.set_prior([SNG(*NG), SDir(dict(DIR))])
        self.assertTrue(s2.dists[0].has_conj_prior)
        self.assertTrue(s2.dists[1].has_conj_prior)
        with self.assertRaises(ValueError):
            s2.set_prior([SNG(*NG)])

    def test_child_posterior_propagates(self):
        data = list(np.random.RandomState(7).normal(2.5, 1.5, size=200))
        sx, sxx, n = float(np.sum(data)), float(np.sum(np.square(data))), float(len(data))
        standalone = SGaussEst(prior=SNG(*NG)).estimate(None, (sx, sxx, n, n))
        comp_est = self._stats().estimator()
        post = comp_est.estimate(None, ((sx, sxx, n, n), {"a": 50.0, "b": 30.0, "c": 20.0}))
        self.assertAlmostEqual(post.dists[0].mu, standalone.mu, places=12)
        self.assertAlmostEqual(post.dists[0].sigma2, standalone.sigma2, places=12)
        self.assertIsInstance(post.dists[0].get_prior(), SNG)
        self.assertIsInstance(post.dists[1].get_prior(), SDir)


class SequenceBayesTest(unittest.TestCase):
    def setUp(self):
        self.obs = [[-1.0, 0.5, 2.0], [], [0.3]]

    def test_expected_log_density_composes_elements(self):
        """The sequence ELD is the sum of the wrapped child's ELD over the elements."""
        s = SSeq(SGauss(GMU, GS2, prior=SNG(*NG)))
        ref = [sum(s.dist.expected_log_density(v) for v in seq) for seq in self.obs]
        self.assertLess(_maxdiff([s.expected_log_density(x) for x in self.obs], ref), 1e-9)

    def test_seq_expected_log_density_self_consistent(self):
        """seq_expected_log_density matches the per-sequence scalar value."""
        s = SSeq(SGauss(GMU, GS2, prior=SNG(*NG)))
        s_enc = s.dist_to_encoder().seq_encode(self.obs)
        self.assertLess(
            _maxdiff(s.seq_expected_log_density(s_enc), [s.expected_log_density(x) for x in self.obs]), 1e-9
        )

    def test_set_prior_updates_entry_estimator(self):
        """Regression: SequenceEstimator.set_prior threads to the entry/len estimators (not a fixed dist)."""
        est = SSeq(SGauss(0.0, 1.0)).estimator()
        est.set_prior((SNG(*NG), None))
        self.assertIsInstance(est.estimator.get_prior(), SNG)
        self.assertTrue(est.estimator.has_conj_prior)
        # None is a no-op
        est.set_prior(None)
        self.assertIsInstance(est.estimator.get_prior(), SNG)

    def test_prior_round_trip(self):
        s = SSeq(SGauss(GMU, GS2, prior=SNG(*NG)))
        entry_prior, len_prior = s.get_prior()
        self.assertIsInstance(entry_prior, SNG)
        s2 = SSeq(SGauss(0.0, 1.0))
        s2.set_prior((SNG(*NG), None))
        self.assertTrue(s2.dist.has_conj_prior)

    def test_mle_unchanged_without_prior(self):
        est = SSeq(SGauss(0.0, 1.0)).estimator()
        acc = est.accumulator_factory().make()
        enc = acc.acc_to_encoder().seq_encode(self.obs)
        acc.seq_update(enc, np.ones(len(self.obs)), None)
        m = est.estimate(None, acc.value())
        flat = [v for seq in self.obs for v in seq]
        self.assertAlmostEqual(m.dist.mu, sum(flat) / len(flat), places=10)
        self.assertIsNone(m.dist.get_prior())


class OptionalBayesTest(unittest.TestCase):
    def setUp(self):
        self.obs = [None, -1.0, 0.5, None, 2.7]

    def test_expected_log_density_composes_branches(self):
        """For present values the Optional ELD is E[log(1-p)] + child ELD; missing -> E[log p]."""
        from pysp.utils.special import digamma

        a, b = 2.0, 5.0
        s = SOpt(SGauss(GMU, GS2, prior=SNG(*NG)), p=0.3, prior=(SBeta(a, b), SNG(*NG)))
        e_log_p = digamma(a) - digamma(a + b)  # E[log p_missing]
        e_log_1mp = digamma(b) - digamma(a + b)  # E[log (1 - p_missing)]
        ref = []
        for x in self.obs:
            if x is None:
                ref.append(e_log_p)
            else:
                ref.append(e_log_1mp + s.dist.expected_log_density(x))
        self.assertLess(_maxdiff([s.expected_log_density(x) for x in self.obs], ref), 1e-9)

    def test_seq_expected_log_density_self_consistent(self):
        """seq_expected_log_density matches the per-observation scalar value."""
        s = SOpt(SGauss(GMU, GS2, prior=SNG(*NG)), p=0.3, prior=(SBeta(2.0, 5.0), SNG(*NG)))
        s_enc = s.dist_to_encoder().seq_encode(self.obs)
        self.assertLess(
            _maxdiff(s.seq_expected_log_density(s_enc), [s.expected_log_density(x) for x in self.obs]), 1e-9
        )

    def test_conjugate_posterior_closed_form(self):
        """Beta posterior on p + base posterior match the textbook closed form."""
        psum, nsum = 12.0, 88.0
        sx, sxx, n = 30.0, 200.0, nsum
        s_est = SOpt(SGauss(GMU, GS2, prior=SNG(*NG)), p=0.3, prior=(SBeta(2.0, 5.0), SNG(*NG))).estimator()
        s_post = s_est.estimate(None, ([psum, nsum], (sx, sxx, n, n)))
        # textbook Beta posterior-mode closed form for p
        a, b = 2.0, 5.0
        self.assertAlmostEqual(s_post.p, (psum + a - 1.0) / (psum + nsum + a + b - 2.0), places=12)
        pa, pb = s_post.get_prior()[0].get_parameters()
        self.assertAlmostEqual(pa, a + psum, places=12)
        self.assertAlmostEqual(pb, b + nsum, places=12)

    def test_mle_unchanged_without_prior(self):
        est = SOpt(SGauss(0.0, 1.0), p=0.3).estimator()
        acc = est.accumulator_factory().make()
        enc = acc.acc_to_encoder().seq_encode(self.obs)
        acc.seq_update(enc, np.ones(len(self.obs)), None)
        m = est.estimate(None, acc.value())
        # empirical missing fraction (2 of 5 are None)
        self.assertAlmostEqual(m.p, 2.0 / 5.0, places=10)
        self.assertFalse(m.has_conj_prior)

    def test_prior_round_trip(self):
        s = SOpt(SGauss(GMU, GS2, prior=SNG(*NG)), p=0.3, prior=(SBeta(2.0, 5.0), SNG(*NG)))
        p_prior, dist_prior = s.get_prior()
        self.assertIsInstance(p_prior, SBeta)
        self.assertIsInstance(dist_prior, SNG)
        self.assertTrue(s.has_conj_prior)
        s2 = SOpt(SGauss(0.0, 1.0), p=0.3)
        self.assertFalse(s2.has_conj_prior)


class ConditionalBayesTest(unittest.TestCase):
    def setUp(self):
        self.obs = [(0, -1.0), (1, 0.5), (0, 2.0), (1, -0.3)]

    def _dist(self):
        return SCond(
            {0: SGauss(0.0, 1.0), 1: SGauss(1.0, 2.0)},
            prior=({0: SNG(0.0, 1.0, 2.0, 3.0), 1: SNG(1.0, 2.0, 3.0, 4.0)}, None, None),
        )

    def test_expected_log_density_composes_branches(self):
        s = self._dist()
        ref = [s.dmap[c].expected_log_density(y) for c, y in self.obs]
        self.assertLess(_maxdiff([s.expected_log_density(x) for x in self.obs], ref), 1e-9)

    def test_seq_expected_log_density_self_consistent(self):
        s = self._dist()
        enc = s.dist_to_encoder().seq_encode(self.obs)
        self.assertLess(_maxdiff(s.seq_expected_log_density(enc), [s.expected_log_density(x) for x in self.obs]), 1e-9)

    def test_prior_round_trip(self):
        s = self._dist()
        branch, default_prior, given_prior = s.get_prior()
        self.assertIsInstance(branch[0], SNG)
        self.assertIsInstance(branch[1], SNG)
        self.assertTrue(s.dmap[0].has_conj_prior)
        # None no-op
        s.set_prior(None)
        self.assertTrue(s.dmap[1].has_conj_prior)

    def test_model_log_density_sums_branches(self):
        s = self._dist()
        est = s.estimator()
        # priors propagate from dist children into branch estimators
        total = sum(est.estimator_map[k].model_log_density(s.dmap[k]) for k in s.dmap)
        self.assertLess(abs(est.model_log_density(s) - total), 1e-12)

    def test_mle_unchanged_without_prior(self):
        d0 = SCond({0: SGauss(0.0, 1.0), 1: SGauss(1.0, 2.0)})
        est = d0.estimator()
        acc = est.accumulator_factory().make()
        enc = acc.acc_to_encoder().seq_encode(self.obs)
        # EM contract: pass the current estimate through seq_update
        acc.seq_update(enc, np.ones(len(self.obs)), d0)
        m = est.estimate(None, acc.value())
        self.assertIsNone(m.dmap[0].get_prior())


class IgnoredBayesTest(unittest.TestCase):
    def test_delegates_prior_to_wrapped(self):
        inner = SGauss(GMU, GS2, prior=SNG(*NG))
        ig = SIgnored(inner)
        self.assertIsInstance(ig.get_prior(), SNG)
        # expected_log_density delegates to the wrapped distribution
        self.assertAlmostEqual(ig.expected_log_density(0.7), inner.expected_log_density(0.7), places=12)
        # set_prior delegates
        ig.set_prior(SNG(0.0, 1.0, 2.0, 3.0))
        self.assertEqual(inner.get_prior().get_parameters()[1], 1.0)

    def test_no_prior_delegates_log_density(self):
        inner = SGauss(0.0, 1.0)
        ig = SIgnored(inner)
        self.assertIsNone(ig.get_prior())
        self.assertAlmostEqual(ig.expected_log_density(1.1), inner.log_density(1.1), places=12)


if __name__ == "__main__":
    unittest.main()
