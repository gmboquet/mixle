"""The improvement operators + scoped registry (mixle.evolve.operators)."""

import unittest

import numpy as np

from mixle.evolve import (
    AutoSelect,
    OnlineUpdate,
    Recalibrate,
    Refit,
    nll_objective,
    register_operator,
    registered_operators,
    unregister_operator,
)
from mixle.evolve.objective import pointwise_log_density
from mixle.stats import GaussianDistribution


class OperatorTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(3.0, 2.0, 400))
        self.champion = GaussianDistribution(0.0, 1.0)  # deliberately wrong
        self.nll = nll_objective()
        self.ctx = {"parent_hash": "parent0"}

    def test_each_operator_returns_a_valid_fitted_model(self):
        for op in [Refit(), OnlineUpdate(mode="streaming"), OnlineUpdate(mode="incremental"), AutoSelect()]:
            with self.subTest(op=op.name):
                self.assertTrue(op.applicable(self.champion, self.data, ctx=self.ctx))
                cand = op.propose(self.champion, self.data, ctx=self.ctx)
                self.assertEqual(cand.parent_hash, "parent0")
                # the proposed model must score finite per-observation log densities.
                ld = pointwise_log_density(cand.model, self.data)
                self.assertEqual(ld.shape[0], len(self.data))
                self.assertTrue(np.all(np.isfinite(ld)))
                # and it must be a real improvement over the bad champion.
                self.assertLess(self.nll.scalar(cand.model, self.data), self.nll.scalar(self.champion, self.data))

    def test_recalibrate_returns_valid_model_and_scores_off_split(self):
        op = Recalibrate(seed=0)
        self.assertTrue(op.applicable(GaussianDistribution(3.0, 2.0), self.data, ctx=self.ctx))
        cand = op.propose(GaussianDistribution(3.0, 2.0), self.data, ctx=self.ctx)
        # exact change-of-variables: log density on a *different* split is finite and split-safe.
        rng = np.random.RandomState(99)
        other = list(rng.normal(3.0, 2.0, 50))
        ld = pointwise_log_density(cand.model, other)
        self.assertEqual(ld.shape[0], 50)
        self.assertTrue(np.all(np.isfinite(ld)))
        self.assertIn("temperature", cand.meta)

    def test_recalibrate_density_normalizes(self):
        # the recalibrated density must integrate to ~1 (exactness check on the Jacobian).
        from mixle.evolve.operators import _RecalibratedModel

        base = GaussianDistribution(0.0, 1.0)
        recal = _RecalibratedModel(base, temperature=1.5, center=0.0)
        grid = np.linspace(-12, 12, 4001)
        dens = np.exp([recal.log_density(float(x)) for x in grid])
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        mass = trapezoid(dens, grid)
        self.assertAlmostEqual(mass, 1.0, places=3)

    def test_posterior_carry_applicability_is_honest(self):
        # conjugate Gaussian -> applicable; the streaming mode is always applicable.
        self.assertTrue(OnlineUpdate(mode="posterior_carry").applicable(self.champion, self.data, ctx={}))
        self.assertTrue(OnlineUpdate(mode="streaming").applicable(self.champion, self.data, ctx={}))

    def test_scoped_registry_register_and_unregister(self):
        op = Refit(name="custom_refit")
        register_operator(op)
        self.assertIn("custom_refit", registered_operators())
        unregister_operator("custom_refit")
        self.assertNotIn("custom_refit", registered_operators())


if __name__ == "__main__":
    unittest.main()
