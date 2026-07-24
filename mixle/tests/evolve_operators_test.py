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


class RecalibratedEncodedScoringTest(unittest.TestCase):
    """A _RecalibratedModel must expose ONE density per observation: its public encoded/batch route
    (``dist_to_encoder().seq_encode(...)`` -> ``seq_log_density(...)``, the contract every other
    distribution in this codebase honors) must agree with its scalar ``log_density`` path and its
    split-safe ``seq_log_density_raw`` path for the same observation. Pre-fix, ``dist_to_encoder()``
    delegated straight to the base model's own encoder, so the encoded path scored the UNTRANSFORMED
    raw observation and only added the Jacobian -- a different (wrong) density than the scalar path."""

    def test_encoded_path_agrees_with_scalar_and_raw_batch_paths(self):
        # Pins the exact repro: standard Gaussian, temperature=2, center=0, observation=2. Pre-fix,
        # the encoded path answered -3.6120857... (Jacobian only, no value transform) while the
        # scalar/raw-batch paths correctly answered -2.1120857....
        from mixle.evolve.operators import _RecalibratedModel

        base = GaussianDistribution(0.0, 1.0)
        recal = _RecalibratedModel(base, temperature=2.0, center=0.0)

        scalar = recal.log_density(2.0)
        raw_batch = recal.seq_log_density_raw([2.0])
        enc = recal.dist_to_encoder().seq_encode([2.0])
        encoded = recal.seq_log_density(enc)

        self.assertAlmostEqual(scalar, -2.112085713764618, places=6)
        self.assertAlmostEqual(float(raw_batch[0]), scalar, places=9)
        self.assertAlmostEqual(float(encoded[0]), scalar, places=9)

    def test_encoded_path_agrees_over_a_grid_of_observations_and_temperatures(self):
        from mixle.evolve.operators import _RecalibratedModel

        base = GaussianDistribution(1.5, 2.0)
        rng = np.random.RandomState(0)
        xs = rng.normal(1.5, 3.0, 25)
        for temperature, center in [(0.6, 1.5), (1.0, 0.0), (3.3, -2.0)]:
            with self.subTest(temperature=temperature, center=center):
                recal = _RecalibratedModel(base, temperature=temperature, center=center)
                scalars = np.array([recal.log_density(float(x)) for x in xs])
                enc = recal.dist_to_encoder().seq_encode(list(xs))
                encoded = np.asarray(recal.seq_log_density(enc), dtype=float)
                np.testing.assert_allclose(encoded, scalars, rtol=1e-10, atol=1e-10)

    def test_recalibrate_operator_output_honors_the_same_invariant(self):
        # end-to-end: a model produced by the real Recalibrate operator (not a hand-built
        # _RecalibratedModel) must still agree between its scalar and public-encoded scoring paths.
        rng = np.random.RandomState(0)
        data = list(rng.normal(3.0, 2.0, 200))
        cand = Recalibrate(seed=0).propose(GaussianDistribution(3.0, 2.0), data, ctx={"parent_hash": None})
        recal = cand.model

        probe = list(np.random.RandomState(7).normal(3.0, 2.0, 30))
        scalars = np.array([recal.log_density(float(x)) for x in probe])
        enc = recal.dist_to_encoder().seq_encode(probe)
        encoded = np.asarray(recal.seq_log_density(enc), dtype=float)
        np.testing.assert_allclose(encoded, scalars, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
