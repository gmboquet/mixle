"""Bayesian (conjugate / variational) behavior folded onto the pysp.stats Dirichlet-prior group.

Covers the categorical (DictDirichlet prior) and integer-categorical (Dirichlet / SymmetricDirichlet
prior) leaves, plus the ported DictDirichlet / SymmetricDirichlet prior families. Mirrors the proven
Gaussian template: a frequentist leaf gains conjugate posterior estimation, ``expected_log_density``,
and a posterior-returning ``fit`` while its MLE path stays unchanged. Numeric expectations are pinned
against textbook conjugate closed forms.
"""

import unittest

import numpy as np

from pysp.stats.bayes.catdirichlet import DictDirichletDistribution
from pysp.stats.bayes.dirichlet import DirichletDistribution
from pysp.stats.bayes.symdirichlet import SymmetricDirichletDistribution
from pysp.stats.leaf.categorical import CategoricalDistribution, CategoricalEstimator
from pysp.stats.leaf.integer_categorical import (
    IntegerCategoricalDataEncoder,
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
)
from pysp.utils.special import digamma, gammaln


class StatsBayesCategoricalTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(7)
        self.data = rng.choice(["a", "b", "c"], p=[0.5, 0.3, 0.2], size=300).tolist()
        self.count_map = {}
        for v in self.data:
            self.count_map[v] = self.count_map.get(v, 0.0) + 1.0
        self.keys = sorted(self.count_map.keys())

    def test_mle_path_unchanged(self):
        """No prior -> plain relative-frequency MLE; estimator carries no posterior."""
        est = CategoricalEstimator()
        self.assertFalse(est.has_conj_prior)
        m = est.estimate(None, dict(self.count_map))
        total = sum(self.count_map.values())
        for k in self.keys:
            self.assertAlmostEqual(m.pmap[k], self.count_map[k] / total, places=12)
        self.assertIsNone(m.get_prior())
        self.assertFalse(m.has_conj_prior)

    def test_conjugate_posterior_dict_prior(self):
        """estimate() with a DictDirichlet (dict alpha) returns the MAP probs + posterior Dirichlet."""
        alpha = {"a": 2.0, "b": 3.5, "c": 1.2}
        est = CategoricalEstimator(prior=DictDirichletDistribution(dict(alpha)))
        self.assertTrue(est.has_conj_prior)
        m = est.estimate(None, dict(self.count_map))
        # MAP: (count + alpha - 1) clamped, normalized
        num = {k: max((alpha[k] - 1) + self.count_map[k], 0.0) for k in self.keys}
        z = sum(num.values())
        for k in self.keys:
            self.assertAlmostEqual(m.pmap[k], num[k] / z, places=12)
        # posterior alpha = count + alpha
        post = m.get_prior().get_parameters()
        for k in self.keys:
            self.assertAlmostEqual(post[k], self.count_map[k] + alpha[k], places=12)

    def test_conjugate_posterior_scalar_prior(self):
        """A scalar (symmetric) DictDirichlet alpha gives the symmetric MAP estimate."""
        a = 1.5
        est = CategoricalEstimator(prior=DictDirichletDistribution(a))
        m = est.estimate(None, dict(self.count_map))
        num = {k: max((a - 1) + self.count_map[k], 0.0) for k in self.keys}
        z = sum(num.values())
        for k in self.keys:
            self.assertAlmostEqual(m.pmap[k], num[k] / z, places=12)

    def test_expected_log_density_formula(self):
        """expected_log_density equals the VB E[log p_k] = digamma(a_k) - digamma(sum a) closed form."""
        alpha = {"a": 2.0, "b": 3.5, "c": 1.2}
        pmap = {"a": 0.5, "b": 0.3, "c": 0.2}
        d = CategoricalDistribution(pmap, prior=DictDirichletDistribution(dict(alpha)))
        asum = digamma(sum(alpha.values()))
        for k in self.keys:
            self.assertAlmostEqual(d.expected_log_density(k), digamma(alpha[k]) - asum, places=12)
        # seq parity
        enc = d.dist_to_encoder().seq_encode(self.data)
        seq = d.seq_expected_log_density(enc)
        scalar = np.asarray([d.expected_log_density(x) for x in self.data])
        self.assertTrue(np.allclose(seq, scalar, atol=1e-12))
        # no prior -> plug-in
        d0 = CategoricalDistribution(pmap)
        self.assertAlmostEqual(d0.expected_log_density("a"), d0.log_density("a"), places=12)
        self.assertTrue(np.allclose(d0.seq_expected_log_density(enc), d0.seq_log_density(enc), atol=1e-12))

    def test_model_log_density(self):
        """model_log_density scores the model pmap under the DictDirichlet prior; 0 with no prior."""
        alpha = {"a": 2.0, "b": 3.5, "c": 1.2}
        prior = DictDirichletDistribution(dict(alpha))
        est = CategoricalEstimator(prior=prior)
        m = est.estimate(None, dict(self.count_map))
        self.assertAlmostEqual(est.model_log_density(m), float(prior.log_density(m.pmap)), places=12)
        self.assertEqual(CategoricalEstimator().model_log_density(m), 0.0)

    def test_estimator_propagates_prior(self):
        """.estimator() carries the prior forward from the distribution."""
        prior = DictDirichletDistribution(1.7)
        d = CategoricalDistribution({"a": 0.5, "b": 0.5}, prior=prior)
        self.assertIs(d.estimator().get_prior(), prior)
        self.assertTrue(d.has_conj_prior)


class StatsBayesIntegerCategoricalTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(11)
        self.idata = rng.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1], size=300)
        self.cvec = np.bincount(self.idata, minlength=4).astype(float)
        self.suff = (0, self.cvec.copy())

    def test_mle_path_unchanged(self):
        """No prior -> plain relative-frequency MLE; estimator carries no posterior."""
        est = IntegerCategoricalEstimator()
        self.assertFalse(est.has_conj_prior)
        m = est.estimate(None, (0, self.cvec.copy()))
        self.assertTrue(np.allclose(m.p_vec, self.cvec / self.cvec.sum(), atol=1e-12))

    def test_conjugate_posterior_vector_prior(self):
        """estimate() with a vector Dirichlet returns the MAP probs + posterior Dirichlet."""
        alpha = np.array([2.0, 3.0, 1.5, 1.2])
        est = IntegerCategoricalEstimator(prior=DirichletDistribution(alpha.copy()))
        self.assertTrue(est.has_conj_prior)
        m = est.estimate(None, (0, self.cvec.copy()))
        num = np.maximum(self.cvec + (alpha - 1), 0.0)
        self.assertTrue(np.allclose(m.p_vec, num / num.sum(), atol=1e-12))
        self.assertTrue(np.allclose(m.get_prior().get_parameters(), self.cvec + alpha, atol=1e-12))

    def test_conjugate_posterior_symmetric_prior(self):
        """A SymmetricDirichlet prior broadcasts the scalar concentration over the support."""
        a = 1.8
        est = IntegerCategoricalEstimator(prior=SymmetricDirichletDistribution(a))
        m = est.estimate(None, (0, self.cvec.copy()))
        num = np.maximum(self.cvec + (a - 1), 0.0)
        self.assertTrue(np.allclose(m.p_vec, num / num.sum(), atol=1e-12))
        self.assertTrue(np.allclose(m.get_prior().get_parameters(), self.cvec + a, atol=1e-12))

    def test_expected_log_density_formula(self):
        """expected_log_density equals digamma(a_k) - digamma(sum a) over the support."""
        alpha = np.array([2.0, 3.0, 1.5, 1.2])
        p = self.cvec / self.cvec.sum()
        d = IntegerCategoricalDistribution(0, p, prior=DirichletDistribution(alpha.copy()))
        expected = digamma(alpha) - digamma(alpha.sum())
        for i in range(4):
            self.assertAlmostEqual(d.expected_log_density(i), expected[i], places=12)
        # out-of-support
        self.assertEqual(d.expected_log_density(-1), -np.inf)
        # seq parity
        enc = IntegerCategoricalDataEncoder().seq_encode(self.idata)
        seq = d.seq_expected_log_density(enc)
        scalar = np.asarray([d.expected_log_density(int(x)) for x in self.idata])
        self.assertTrue(np.allclose(seq, scalar, atol=1e-12))
        # no prior -> plug-in
        d0 = IntegerCategoricalDistribution(0, p)
        self.assertTrue(np.allclose(d0.seq_expected_log_density(enc), d0.seq_log_density(enc), atol=1e-12))

    def test_model_log_density(self):
        """model_log_density scores the model p_vec under the Dirichlet prior; 0 with no prior."""
        alpha = np.array([2.0, 3.0, 1.5, 1.2])
        prior = DirichletDistribution(alpha.copy())
        est = IntegerCategoricalEstimator(prior=prior)
        m = est.estimate(None, (0, self.cvec.copy()))
        self.assertAlmostEqual(est.model_log_density(m), float(prior.log_density(m.p_vec)), places=12)
        self.assertEqual(IntegerCategoricalEstimator().model_log_density(m), 0.0)

    def test_estimator_propagates_prior(self):
        """.estimator() carries the prior forward from the distribution."""
        prior = SymmetricDirichletDistribution(2.0)
        d = IntegerCategoricalDistribution(0, [0.25, 0.25, 0.25, 0.25], prior=prior)
        self.assertIs(d.estimator().get_prior(), prior)
        self.assertTrue(d.has_conj_prior)


class StatsDirichletPriorFamilyTestCase(unittest.TestCase):
    def test_dict_dirichlet_log_density_and_entropy(self):
        """DictDirichlet log_density / entropy match the closed forms."""
        alpha = {"a": 2.0, "b": 3.5, "c": 1.2}
        d = DictDirichletDistribution(dict(alpha))
        x = {"a": 0.2, "b": 0.5, "c": 0.3}
        rv = gammaln(sum(alpha.values()))
        for k in alpha:
            rv += np.log(x[k]) * (alpha[k] - 1) - gammaln(alpha[k])
        self.assertAlmostEqual(d.log_density(x), rv, places=12)
        a = np.asarray(list(alpha.values()))
        a0 = a.sum()
        ent = -((gammaln(a0) - np.sum(gammaln(a))) + np.dot(digamma(a) - digamma(a0), a - 1))
        self.assertAlmostEqual(d.entropy(), ent, places=12)

    def test_dict_dirichlet_scalar(self):
        """Scalar DictDirichlet alpha matches the symmetric closed form."""
        d = DictDirichletDistribution(1.7)
        x = {"a": 0.2, "b": 0.5, "c": 0.3}
        n = len(x)
        c = gammaln(1.7) * n - gammaln(1.7 * n)
        expected = np.sum(np.log(list(x.values()))) * (1.7 - 1) - c
        self.assertAlmostEqual(d.log_density(x), expected, places=12)

    def test_symmetric_dirichlet_log_density(self):
        """SymmetricDirichlet log_density matches the closed form."""
        d = SymmetricDirichletDistribution(2.3)
        x = np.array([0.2, 0.5, 0.3])
        nc = len(x) * gammaln(2.3) - gammaln(len(x) * 2.3)
        expected = np.sum(np.log(x) * (2.3 - 1)) - nc
        self.assertAlmostEqual(d.log_density(x), expected, places=12)
        # seq parity
        xs = np.array([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]])
        seq = d.seq_log_density(d.dist_to_encoder().seq_encode(xs))
        self.assertTrue(np.allclose(seq, [d.log_density(r) for r in xs], atol=1e-12))


if __name__ == "__main__":
    unittest.main()
