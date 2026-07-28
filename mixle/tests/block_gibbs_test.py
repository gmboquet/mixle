"""Block Gibbs with per-block inference dispatch: conjugate + Metropolis in one model (Phase 7)."""

import unittest

import numpy as np

from mixle.inference.block_gibbs import BlockGibbs, ConjugateBlock, MetropolisBlock


class BlockGibbsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.y = rng.normal(2.0, 1.5, 200)
        self.n = len(self.y)

    def _model(self):
        y, n = self.y, self.n

        def draw_mu(state, r):  # conjugate Gaussian full conditional (prior N(0, 100))
            s2 = state["sigma"] ** 2
            pv = 1.0 / (n / s2 + 1 / 100.0)
            return r.normal(pv * (y.sum() / s2), np.sqrt(pv))

        def logp_sigma(sigma, state):  # non-conjugate (half-Cauchy prior) -> Metropolis
            if sigma <= 0:
                return -np.inf
            return -0.5 * np.sum((y - state["mu"]) ** 2) / sigma**2 - n * np.log(sigma) - np.log(1 + sigma**2)

        return ConjugateBlock("mu", draw_mu), MetropolisBlock("sigma", logp_sigma, scale=0.2)

    def test_recovers_the_posterior(self):
        mu_b, sig_b = self._model()
        ch = BlockGibbs([mu_b, sig_b], init={"mu": 0.0, "sigma": 1.0}).run(4000, burn=1000, seed=1)
        self.assertAlmostEqual(ch["mu"].mean(), self.y.mean(), delta=0.05)
        self.assertAlmostEqual(ch["sigma"].mean(), self.y.std(), delta=0.1)

    def test_dispatches_distinct_update_kinds(self):
        mu_b, sig_b = self._model()
        self.assertEqual(mu_b.kind, "conjugate")
        self.assertEqual(sig_b.kind, "metropolis")
        BlockGibbs([mu_b, sig_b], init={"mu": 0.0, "sigma": 1.0}).run(2000, burn=500, seed=1)
        self.assertGreater(sig_b.acceptance_rate, 0.1)  # the Metropolis block actually moves
        self.assertLess(sig_b.acceptance_rate, 0.9)

    def test_conjugate_block_matches_analytic_spread(self):
        mu_b, sig_b = self._model()
        ch = BlockGibbs([mu_b, sig_b], init={"mu": 0.0, "sigma": 1.0}).run(4000, burn=1000, seed=1)
        self.assertAlmostEqual(ch["mu"].std(), 1.5 / np.sqrt(self.n), delta=0.03)  # Gaussian conditional spread

    def test_mixed_matches_all_metropolis(self):
        mu_b, sig_b = self._model()
        mixed = BlockGibbs([mu_b, sig_b], init={"mu": 0.0, "sigma": 1.0}).run(4000, burn=1000, seed=1)
        y, n = self.y, self.n

        def logp_mu(mu, state):
            return -0.5 * np.sum((y - mu) ** 2) / state["sigma"] ** 2 - 0.5 * mu**2 / 100.0

        _, sig_b2 = self._model()
        allmh = BlockGibbs([MetropolisBlock("mu", logp_mu, 0.2), sig_b2], init={"mu": 0.0, "sigma": 1.0}).run(
            4000, burn=1000, seed=2
        )
        self.assertAlmostEqual(mixed["mu"].mean(), allmh["mu"].mean(), delta=0.05)
        self.assertAlmostEqual(mixed["sigma"].mean(), allmh["sigma"].mean(), delta=0.05)

    def test_metropolis_block_stops_adapting_after_burn_in(self):
        # The class's own docstring says the scale is "adapted... toward a ~0.4 acceptance rate"
        # during burn-in; before the fix, the `self._tot % 50 == 0` adaptation check fired
        # unconditionally for the entire run, so the scale kept changing during the post-burn-in
        # "stationary" phase the retained samples are drawn from -- contrary to the very validity
        # requirement adaptive MCMC needs (a transition kernel that keeps changing after burn-in
        # breaks the chain's stationarity guarantee for the retained samples).
        rng = np.random.RandomState(0)
        state = {"x": 0.0}
        block = MetropolisBlock("x", lambda x, s: -0.5 * x * x, scale=0.2)
        for _ in range(50):
            state["x"] = block.update(state, rng, in_burn_in=True)
        scale_after_burn_in = block.scale
        for _ in range(200):
            state["x"] = block.update(state, rng, in_burn_in=False)
        self.assertEqual(block.scale, scale_after_burn_in)


class RetainedStateAliasingTest(unittest.TestCase):
    """MXR-080-1629: retained samples must be snapshots, not aliases of a reused buffer."""

    def _in_place_block(self):
        counter = {"n": 0}

        def draw(state, rng):
            buf = state["z"]
            counter["n"] += 1
            buf[0] = float(counter["n"])  # idiomatic in-place update, returns the same buffer
            return buf

        return ConjugateBlock("z", draw)

    def test_in_place_updates_do_not_rewrite_chain_history(self):
        chain = BlockGibbs([self._in_place_block()], init={"z": np.array([0.0])}).run(3, burn=0, seed=0)
        np.testing.assert_allclose(np.asarray(chain["z"]).ravel(), [1.0, 2.0, 3.0])

    def test_callers_initial_state_is_not_mutated(self):
        caller_buffer = np.array([0.0])
        BlockGibbs([self._in_place_block()], init={"z": caller_buffer}).run(3, burn=0, seed=0)
        np.testing.assert_allclose(caller_buffer, [0.0])

    def test_scalar_blocks_are_unaffected(self):
        block = ConjugateBlock("v", lambda state, rng: state["v"] + 1.0)
        chain = BlockGibbs([block], init={"v": 0.0}).run(3, burn=0, seed=0)
        np.testing.assert_allclose(np.asarray(chain["v"]).ravel(), [1.0, 2.0, 3.0])


class RunReproducibilityTest(unittest.TestCase):
    """MXR-080-1630: the same sampler, seed, and init must reproduce the same chain."""

    def _sampler(self):
        block = MetropolisBlock("t", lambda v, s: -0.5 * float(np.asarray(v) ** 2), scale=1.0)
        return block, BlockGibbs([block], init={"t": 0.0})

    def test_rerunning_with_the_same_seed_reproduces_the_chain(self):
        block, sampler = self._sampler()
        first = sampler.run(50, burn=200, seed=1)
        second = sampler.run(50, burn=200, seed=1)
        np.testing.assert_allclose(np.asarray(first["t"]), np.asarray(second["t"]))

    def test_adaptation_state_is_reset_between_runs(self):
        block, sampler = self._sampler()
        sampler.run(50, burn=200, seed=1)
        scale_after_first, tot_after_first = block.scale, block._tot
        sampler.run(50, burn=200, seed=1)
        self.assertEqual(block.scale, scale_after_first)
        self.assertEqual(block._tot, tot_after_first)  # not the cumulative total of both runs

    def test_resume_continues_from_previous_adaptation(self):
        block, sampler = self._sampler()
        sampler.run(50, burn=200, seed=1)
        tot_after_first = block._tot
        sampler.run(50, burn=200, seed=1, resume=True)
        self.assertGreater(block._tot, tot_after_first)


class InvalidControlTest(unittest.TestCase):
    """MXR-080-1631: unusable controls must be rejected, not returned as a frozen/empty chain."""

    def test_non_positive_and_non_finite_scales_are_rejected(self):
        for bad in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                MetropolisBlock("t", lambda v, s: 0.0, scale=bad)

    def test_invalid_sample_counts_are_rejected(self):
        sampler = BlockGibbs([ConjugateBlock("z", lambda s, r: 1.0)], init={"z": 0.0})
        for bad in (0, -2):
            with self.assertRaises(ValueError):
                sampler.run(bad, burn=0, seed=0)
        for bad in (2.5, True):
            with self.assertRaises(TypeError):
                sampler.run(bad, burn=0, seed=0)

    def test_negative_burn_in_is_rejected(self):
        sampler = BlockGibbs([ConjugateBlock("z", lambda s, r: 1.0)], init={"z": 0.0})
        with self.assertRaises(ValueError):
            sampler.run(3, burn=-2, seed=0)

    def test_duplicate_block_names_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            BlockGibbs([ConjugateBlock("z", lambda s, r: 1.0), ConjugateBlock("z", lambda s, r: 2.0)], init={"z": 0.0})
        self.assertIn("unique", str(ctx.exception))

    def test_block_without_an_initial_value_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            BlockGibbs([ConjugateBlock("z", lambda s, r: 1.0)], init={"other": 0.0})
        self.assertIn("no starting value", str(ctx.exception))

    def test_valid_controls_still_run(self):
        sampler = BlockGibbs([ConjugateBlock("z", lambda s, r: 1.0)], init={"z": 0.0})
        self.assertEqual(np.asarray(sampler.run(3, burn=1, seed=0)["z"]).shape, (3,))


if __name__ == "__main__":
    unittest.main()
