"""Tests for the finite stochastic-transform combinator (noisy-channel deconvolution).

Covers the output marginal density/enumeration/sampling and -- the point of the combinator -- that
estimation recovers the latent source through the channel WITHOUT inverting it: a single E-step over
the aggregated output counts (R @ n_y) feeds the source's own estimator, so a free categorical source
and a structured (Binomial) source are both recovered, the latter staying parametric.
"""

import unittest

import numpy as np

from pysp.enumeration.algorithms import freeze
from pysp.stats.combinator.finite_stochastic_transform import FiniteStochasticTransformDistribution as FST
from pysp.stats.leaf.binomial import BinomialDistribution
from pysp.stats.leaf.integer_categorical import IntegerCategoricalDistribution as IC

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


if __name__ == "__main__":
    unittest.main()
