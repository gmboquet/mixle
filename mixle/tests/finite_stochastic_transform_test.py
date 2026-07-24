"""Tests for the finite stochastic-transform combinator (noisy-channel deconvolution).

Covers the output marginal density/enumeration/sampling and -- the point of the combinator -- that
estimation recovers the latent source through the channel WITHOUT inverting it: a single E-step over
the aggregated output counts (R @ n_y) feeds the source's own estimator, so a free categorical source
and a structured (Binomial) source are both recovered, the latter staying parametric.
"""

import unittest

import numpy as np

from mixle.enumeration.algorithms import freeze
from mixle.stats.combinator.finite_stochastic_transform import FiniteStochasticTransformDistribution as FST
from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution
from mixle.stats.univariate.discrete.binomial import BinomialDistribution
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution as IC

TOL = 1e-12


def _fit(estimator, model, data, iters):
    for _ in range(iters):
        acc = estimator.accumulator_factory().make()
        enc = model.dist_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data)), model)
        model = estimator.estimate(len(data), acc.value())
    return model


class FiniteStochasticTransformTestCase(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.6, 0.2, 0.1], [0.0, 0.1, 0.3, 0.6]])
        self.px = np.array([0.5, 0.3, 0.2])
        self.dist = FST(IC(0, list(self.px)), self.K)

    def test_output_marginal_density(self):
        py = self.px @ self.K
        for y in range(4):
            self.assertAlmostEqual(self.dist.log_density(y), float(np.log(py[y])), delta=TOL)
        self.assertAlmostEqual(sum(np.exp(self.dist.log_density(y)) for y in range(4)), 1.0, delta=TOL)
        # out-of-range outputs have zero mass
        self.assertEqual(self.dist.log_density(4), -np.inf)
        self.assertEqual(self.dist.log_density(-1), -np.inf)

    def test_seq_log_density(self):
        xs = [0, 1, 2, 3, 3, 0, 7]  # 7 is out of range -> -inf
        enc = self.dist.dist_to_encoder().seq_encode(xs)
        got = self.dist.seq_log_density(enc)
        np.testing.assert_allclose(got, [self.dist.log_density(x) for x in xs], atol=TOL)

    def test_enumerator_matches_sorted_marginal(self):
        py = self.px @ self.K
        items = list(self.dist.enumerator())
        order = list(np.argsort(-py, kind="stable"))
        self.assertEqual([y for y, _ in items], order)
        for y, lp in items:
            self.assertAlmostEqual(lp, self.dist.log_density(y), delta=TOL)
        self.assertEqual(len({freeze(y) for y, _ in items}), len(items))
        lps = [lp for _, lp in items]
        self.assertTrue(all(lps[i] >= lps[i + 1] - TOL for i in range(len(lps) - 1)))

    def test_sampler_matches_marginal(self):
        py = self.px @ self.K
        s = self.dist.sampler(0).sample(40000)
        emp = np.bincount(s, minlength=4) / 40000.0
        np.testing.assert_allclose(emp, py, atol=0.01)

    def test_estimation_recovers_free_categorical_source(self):
        rng = np.random.RandomState(1)
        true_px = np.array([0.55, 0.30, 0.15])
        true = FST(IC(0, list(true_px)), self.K)
        data = true.sampler(2).sample(40000)
        est = FST(IC(0, [1 / 3, 1 / 3, 1 / 3]), self.K).estimator()
        fitted = _fit(est, FST(IC(0, [1 / 3, 1 / 3, 1 / 3]), self.K), data, iters=100)
        np.testing.assert_allclose(fitted.dist.p_vec, true_px, atol=0.02)

    def test_aggregated_estep_equals_per_observation(self):
        # The aggregated E-step (R @ n_y over the n distinct outputs) must produce the SAME expected
        # source counts as distributing each observation's posterior individually -- the short-circuit
        # is exact, not an approximation.
        data = [0, 0, 1, 2, 3, 3, 3, 1, 0, 2]
        model = self.dist
        # aggregated path (the accumulator)
        est = model.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(model.dist_to_encoder().seq_encode(data), np.ones(len(data)), model)
        agg_counts = np.asarray(acc.value()[1], dtype=float)  # IntegerCategorical suff stat counts
        # explicit per-observation posterior sum
        logpx = model._log_px
        manual = np.zeros(model.num_source)
        for y in data:
            joint = logpx + model.log_kernel[:, y]
            r = np.exp(joint - np.logaddexp.reduce(joint))
            manual += r
        np.testing.assert_allclose(agg_counts, manual, atol=1e-9)

    def test_estimation_recovers_structured_binomial_source(self):
        # The source's own estimator is reused, so a parametric source is recovered AS that family --
        # something an unconstrained channel inversion cannot guarantee.
        m = 6
        K = np.eye(m) * 0.6 + np.full((m, m), 0.4 / m)
        K = K / K.sum(axis=1, keepdims=True)
        true = FST(BinomialDistribution(p=0.6, n=5), K)
        data = true.sampler(3).sample(30000)
        est = FST(BinomialDistribution(p=0.5, n=5), K).estimator()
        fitted = _fit(est, FST(BinomialDistribution(p=0.5, n=5), K), data, iters=60)
        self.assertIsInstance(fitted.dist, BinomialDistribution)
        self.assertAlmostEqual(fitted.dist.p, 0.6, delta=0.02)

    def test_kernel_validation(self):
        with self.assertRaises(ValueError):
            FST(IC(0, [0.5, 0.5]), np.array([[0.5, 0.5], [0.0, 0.0]]))  # zero row
        with self.assertRaises(ValueError):
            FST(IC(0, [1.0]), np.array([0.5, 0.5]))  # not 2-D

    def test_kernel_row_count_must_match_source_state_count(self):
        # A Bernoulli source has 2 states (0 and 1); a 1-row kernel is structurally incompatible
        # even though `_row_stochastic` itself has nothing to object to (each of its 1 row is a
        # valid distribution over outputs). Without the shape check, this used to construct
        # successfully with an output marginal that silently summed to 0.5 instead of 1.0, and
        # crashed on sampling whenever the source's latent state 1 was drawn (out of bounds for a
        # 1-row kernel).
        with self.assertRaises(ValueError):
            FST(BernoulliDistribution(0.5), np.array([[0.9, 0.1]]))
        with self.assertRaises(ValueError):
            # Too many rows is just as structurally wrong as too few.
            FST(BernoulliDistribution(0.5), np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]))
        # Negative control: a 2-row kernel matches the Bernoulli source's 2 states and must still
        # construct normally, with the output marginal correctly summing to 1 and sampling never
        # crashing regardless of which latent source state is drawn.
        dist = FST(BernoulliDistribution(0.5), np.array([[0.9, 0.1], [0.2, 0.8]]))
        total = sum(np.exp(dist.log_density(y)) for y in range(dist.num_output))
        self.assertAlmostEqual(total, 1.0, delta=TOL)
        s = dist.sampler(0)
        for _ in range(200):
            s.sample()

    def test_seq_encode_rejects_fractional_and_nan_observations(self):
        # The batch/vectorized encoding path must reject the same malformed observations the
        # scalar log_density path already rejects -- 0.5 is not a valid integer output state --
        # instead of silently truncating a fractional or NaN entry to a valid-looking integer
        # before the validity mask is ever computed.
        enc = self.dist.dist_to_encoder().seq_encode([0.5, 1.5, float("nan"), 2])
        y, valid = enc
        np.testing.assert_array_equal(valid, [False, False, False, True])
        got = self.dist.seq_log_density(enc)
        self.assertEqual(got[0], -np.inf)
        self.assertEqual(got[1], -np.inf)
        self.assertEqual(got[2], -np.inf)
        # Negative control: a genuinely valid integer entry in the same batch is unaffected and
        # matches the scalar path exactly (the fix must not reject valid observations too).
        self.assertAlmostEqual(got[3], self.dist.log_density(2), delta=TOL)
        # And the scalar path already rejects the same fractional value (this is the behavior the
        # batch path must now match).
        self.assertEqual(self.dist.log_density(0.5), -np.inf)


if __name__ == "__main__":
    unittest.main()
