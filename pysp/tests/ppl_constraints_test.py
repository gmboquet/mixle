"""pysp.ppl: constraints and inequalities over random variables and their relations.

Covers the shared Constraint abstraction in both modes:
  * generative — RV-vs-RV / linear-expression inequalities, &/|/~ combinators, joint
    conditioning via constrain(...), and single-RV truncation via .given(...);
  * inference — fit(..., constraints=...) restricting the feasible parameter region for
    map / mcmc / ensemble.
"""

import unittest

import numpy as np

from pysp.ppl import Beta, Constraint, Normal, constrain, free


class GenerativeRelationTestCase(unittest.TestCase):
    def test_single_rv_truncation_backcompat(self):
        x = Normal(0, 1)
        q = x.given(x > 0)
        s = np.asarray(q.sample(60000, seed=1))
        self.assertGreaterEqual(s.min(), 0.0)
        self.assertAlmostEqual(s.mean(), np.sqrt(2 / np.pi), delta=0.02)  # half-normal

    def test_rv_vs_rv_inequality(self):
        a, b = Normal(0, 1), Normal(1, 1)  # P(a<b)=Phi(1/sqrt2)=0.7602
        pair = constrain(a < b)
        S = pair.sample(40000, seed=2)
        self.assertEqual(S.shape[1], 2)
        self.assertTrue(np.all(S[:, 0] < S[:, 1]))
        self.assertAlmostEqual(pair.prob(), 0.7602, delta=0.01)

    def test_three_way_ordering_with_and(self):
        a, b, c = Normal(0, 1, name="a"), Normal(0, 1, name="b"), Normal(0, 1, name="c")
        tri = constrain(a < b, b < c)  # P(ordered) = 1/6
        T = tri.sample(30000, seed=3)
        self.assertTrue(np.all((T[:, 0] < T[:, 1]) & (T[:, 1] < T[:, 2])))
        self.assertAlmostEqual(tri.prob(), 1.0 / 6.0, delta=0.01)
        self.assertEqual(tri.columns, ["a", "b", "c"])

    def test_linear_expression(self):
        a, b = Normal(2, 1, name="a"), Normal(0, 1, name="b")
        lin = constrain(a - b > 0.5)  # 2a + ... linear relation
        L = lin.sample(20000, seed=4)
        self.assertTrue(np.all(L[:, 0] - L[:, 1] > 0.5))

    def test_or_and_not_combinators(self):
        x = Normal(0, 1, name="x")
        c = (x < -1) | (x > 1)  # tails
        self.assertIsInstance(c, Constraint)
        s = np.asarray(constrain(c).sample(40000, seed=5))[:, 0]
        self.assertTrue(np.all((s < -1) | (s > 1)))
        c2 = ~(x > 0)
        s2 = np.asarray(constrain(c2).sample(20000, seed=6))[:, 0]
        self.assertTrue(np.all(s2 <= 0))

    def test_chained_comparison_is_rejected(self):
        a, b, c = Normal(0, 1), Normal(0, 1), Normal(0, 1)
        with self.assertRaises(TypeError):
            _ = a < b < c  # must use (a<b) & (b<c)

    def test_given_with_other_rv_rejected(self):
        a, b = Normal(0, 1), Normal(0, 1)
        with self.assertRaises(ValueError):
            a.given(a < b)  # multi-RV -> use constrain(...)

    def test_joint_log_prob_normalized(self):
        a, b = Normal(0, 1, name="a"), Normal(0, 1, name="b")
        j = constrain(a < b)
        # density at a feasible point = base joint - log P(region); infeasible -> -inf
        lp = j.log_prob(np.array([[-0.5, 0.5], [0.5, -0.5]]))
        self.assertTrue(np.isfinite(lp[0]))
        self.assertEqual(lp[1], -np.inf)


class ConstrainedInferenceTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.beta(2.0, 5.0, 3000))  # true a=2 < b=5

    def _model(self):
        a = Normal(2, 5, name="alpha")
        b = Normal(5, 5, name="beta")
        return Beta(a, b), a, b

    def test_mcmc_respects_ordering(self):
        m, a, b = self._model()
        fit = m.fit(self.data, how="mcmc", constraints=a < b, draws=1200, burn=400, rng=np.random.RandomState(1))
        self.assertLess(fit.params["a"], fit.params["b"])
        self.assertAlmostEqual(fit.params["a"], 2.0, delta=0.6)
        self.assertAlmostEqual(fit.params["b"], 5.0, delta=1.2)

    def test_ensemble_respects_ordering(self):
        m, a, b = self._model()
        fit = m.fit(self.data, how="ensemble", constraints=a < b, draws=800, burn=300, rng=np.random.RandomState(2))
        self.assertLess(fit.params["a"], fit.params["b"])

    def test_map_respects_ordering(self):
        m, a, b = self._model()
        fit = m.fit(self.data, how="map", constraints=a < b)
        self.assertLess(fit.params["a"], fit.params["b"])

    def test_auto_routes_to_map_under_constraints(self):
        m, a, b = self._model()
        fit = m.fit(self.data, constraints=a < b)  # auto must not pick a conjugate path
        self.assertLess(fit.params["a"], fit.params["b"])

    def test_reversed_constraint_forces_boundary(self):
        # demand a > b (against the data) -> feasible region pins them together near the boundary
        m, a, b = self._model()
        fit = m.fit(self.data, how="ensemble", constraints=a > b, draws=800, burn=300, rng=np.random.RandomState(3))
        self.assertGreaterEqual(fit.params["a"], fit.params["b"])

    def test_conjugate_with_constraints_rejected(self):
        a = Normal(2, 5, name="alpha")
        b = Normal(5, 5, name="beta")
        with self.assertRaises(ValueError):
            Beta(a, b).fit(self.data, how="conjugate", constraints=a < b)

    def test_constraint_on_non_prior_rejected(self):
        z = Normal(0, 1, name="z")
        with self.assertRaises(ValueError):
            Normal(free, free).fit(self.data, how="map", constraints=z > 0)


if __name__ == "__main__":
    unittest.main()
