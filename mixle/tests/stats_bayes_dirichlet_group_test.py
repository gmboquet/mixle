"""Bayesian (conjugate / variational) behavior folded onto the mixle.stats Dirichlet-prior group.

Covers the categorical (DictDirichlet prior) and integer-categorical (Dirichlet / SymmetricDirichlet
prior) leaves, plus the ported DictDirichlet / SymmetricDirichlet prior families. Mirrors the proven
Gaussian template: a frequentist leaf gains conjugate posterior estimation, ``expected_log_density``,
and a posterior-returning ``fit`` while its MLE path stays unchanged. Numeric expectations are pinned
against textbook conjugate closed forms.
"""

import unittest

import numpy as np

from mixle.stats.bayes.dict_dirichlet import DictDirichletDistribution
from mixle.stats.bayes.dirichlet import DirichletAccumulator, DirichletDistribution
from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator
from mixle.stats.univariate.discrete.integer_categorical import (
    IntegerCategoricalDataEncoder,
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
)
from mixle.utils.special import digamma, gammaln


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


class DirichletFamilySimplexValidationTestCase(unittest.TestCase):
    """Every Dirichlet-family distribution over the simplex must (1) reject a non-positive/non-finite
    concentration at construction and (2) score an observation that is not on the simplex (a negative
    entry, or a vector that doesn't sum to one) as -inf/0.0, on both the scalar and vectorized paths.
    """

    def test_dirichlet_rejects_invalid_alpha(self):
        for bad_alpha in ([0.0, 1.0], [-1.0, 1.0], [float("nan"), 1.0], [float("inf"), 1.0]):
            with self.assertRaises(ValueError):
                DirichletDistribution(bad_alpha)

    def test_dirichlet_seq_log_density_rejects_off_simplex(self):
        # log_density (the scalar path) already validated the simplex; seq_log_density did not -- it
        # computed straight from the encoder's clipped log representation, silently returning a finite
        # (and, for a negative entry, wildly wrong) value instead of -inf.
        d = DirichletDistribution([2.0, 3.0])
        rows = [[2.0, 3.0], [-0.5, 1.5], [0.5, 0.6], [0.4, 0.6]]  # last row is the only valid one
        enc = d.dist_to_encoder().seq_encode(rows)
        seq = d.seq_log_density(enc)
        scalar = np.array([d.log_density(np.array(r)) for r in rows])
        np.testing.assert_array_equal(seq[:3], [-np.inf, -np.inf, -np.inf])
        np.testing.assert_allclose(seq, scalar, atol=1e-12)
        self.assertTrue(np.isfinite(seq[3]))

    def test_symmetric_dirichlet_rejects_invalid_alpha(self):
        for bad_alpha in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SymmetricDirichletDistribution(bad_alpha)
            d = SymmetricDirichletDistribution(2.0)
            with self.assertRaises(ValueError):
                d.set_parameters(bad_alpha)

    def test_symmetric_dirichlet_rejects_off_simplex(self):
        d = SymmetricDirichletDistribution(2.0)
        for bad_x in ([2.0, 3.0], [-0.5, 1.5], [0.5, 0.6], [float("nan"), 0.5]):
            self.assertEqual(d.log_density(np.array(bad_x)), -np.inf, msg=bad_x)
        rows = [[2.0, 3.0], [-0.5, 1.5], [0.3, 0.7]]  # last row is the only valid one
        seq = d.seq_log_density(d.dist_to_encoder().seq_encode(rows))
        np.testing.assert_array_equal(seq[:2], [-np.inf, -np.inf])
        self.assertTrue(np.isfinite(seq[2]))
        self.assertAlmostEqual(seq[2], d.log_density(np.array(rows[2])), places=12)

    def test_symmetric_dirichlet_alpha_one_still_rejects_off_simplex(self):
        # alpha == 1 short-circuits to the (constant) normalizer -nc; that short-circuit must not
        # bypass the simplex check, or ANY input would score as the uniform density.
        d = SymmetricDirichletDistribution(1.0)
        self.assertEqual(d.log_density(np.array([5.0, 10.0])), -np.inf)
        self.assertTrue(np.isfinite(d.log_density(np.array([0.3, 0.7]))))

    def test_dict_dirichlet_rejects_invalid_alpha(self):
        for bad_scalar in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                DictDirichletDistribution(bad_scalar)
        for bad_dict in ({"a": 0.0, "b": 1.0}, {"a": -1.0, "b": 1.0}, {"a": float("nan"), "b": 1.0}):
            with self.assertRaises(ValueError):
                DictDirichletDistribution(bad_dict)

    def test_dict_dirichlet_rejects_off_simplex(self):
        d = DictDirichletDistribution({"a": 2.0, "b": 3.0})
        self.assertEqual(d.log_density({"a": 2.0, "b": 3.0}), -np.inf)  # positive but doesn't sum to 1
        self.assertEqual(d.log_density({"a": 0.5, "b": 0.6}), -np.inf)  # sums to 1.1
        self.assertTrue(np.isfinite(d.log_density({"a": 0.4, "b": 0.6})))
        # negative entries were already rejected pre-fix; keep covering it for regression parity.
        self.assertEqual(d.log_density({"a": -0.5, "b": 1.5}), -np.inf)

    def test_dict_dirichlet_alpha_one_still_rejects_off_simplex(self):
        # The scalar (is_unbounded) alpha == 1 short-circuit had the same bypass as SymmetricDirichlet.
        d = DictDirichletDistribution(1.0)
        self.assertEqual(d.log_density({"a": 5.0, "b": 10.0}), -np.inf)
        self.assertTrue(np.isfinite(d.log_density({"a": 0.3, "b": 0.7})))

    def test_dict_dirichlet_accepts_float32_precision_simplex_row(self):
        # A row that sums to 1 only up to float32 precision (~3e-8 off here) is a legitimate simplex
        # point, not a genuinely invalid input -- e.g. a gradient-fit categorical pmap routed through a
        # lower-precision torch dtype before reaching this prior. A naive float64-tuned 1e-10/1e-12
        # bound would reject it.
        vals = np.array([0.3, 0.3, 0.4], dtype=np.float32)
        pmap = {"a": float(vals[0]), "b": float(vals[1]), "c": float(vals[2])}
        self.assertGreater(abs(sum(pmap.values()) - 1.0), 1e-10)  # confirms this is genuinely off float64-exact
        d = DictDirichletDistribution({"a": 2.0, "b": 2.0, "c": 2.0})
        self.assertTrue(np.isfinite(d.log_density(pmap)))
        du = DictDirichletDistribution(2.0)
        self.assertTrue(np.isfinite(du.log_density(pmap)))

    def test_symmetric_dirichlet_accepts_float32_precision_simplex_row(self):
        # Same float32-precision case as dict_dirichlet above, for its sibling prior -- this is the
        # exact shape of row IntegerMarkovChainDistribution's float32 cond_dist produces.
        row = np.array([0.3, 0.3, 0.4], dtype=np.float32).astype(float)
        self.assertGreater(abs(float(row.sum()) - 1.0), 1e-10)
        d = SymmetricDirichletDistribution(2.0)
        self.assertTrue(np.isfinite(d.log_density(row)))
        seq = d.seq_log_density(d.dist_to_encoder().seq_encode([row]))
        self.assertTrue(np.isfinite(seq[0]))


class DirichletAccumulatorUpdateTestCase(unittest.TestCase):
    """Boundary observations have infinite log statistics and cannot be fitted."""

    def test_list_and_ndarray_boundary_inputs_are_rejected(self):
        x_list = [0.0, 0.5, 0.5]
        for value in (x_list, np.asarray(x_list)):
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                DirichletAccumulator(dim=3).update(value, 1.0, None)

    def test_boundary_rejection_is_atomic(self):
        acc = DirichletAccumulator(dim=3)
        before = acc.value()
        with self.assertRaises(ValueError):
            acc.update([0.0, 0.5, 0.5], 1.0, None)
        after = acc.value()
        self.assertEqual(after[0], before[0])
        for actual, expected in zip(after[1:], before[1:]):
            np.testing.assert_array_equal(actual, expected)

    def test_boundary_rejection_with_nondefault_weight(self):
        acc = DirichletAccumulator(dim=2)
        with self.assertRaises(ValueError):
            acc.update([0.0, 1.0], 2.5, None)
        self.assertEqual(acc.counts, 0.0)


if __name__ == "__main__":
    unittest.main()
