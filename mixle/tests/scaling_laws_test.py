"""F5 acceptance tests: scaling-law fits + compute allocation (mixle/ppl/scaling_laws.py).

Three claims, each with its own test class:

1. :class:`ReproducesKnownExponentsTest` -- ``fit_scaling_law`` (mixle's own ``mixle.ppl`` MCMC
   machinery, not ``scipy.curve_fit``) recovers exponents close to the REAL, PUBLISHED Chinchilla
   exponents (Hoffmann et al. 2022, arXiv:2203.15556, Table 2 "Approach 3"), fit on SYNTHETIC data
   generated from that published functional form + noise (see ``generate_synthetic_chinchilla_data``'s
   docstring for why synthetic-from-known-exponents, not a real per-run table: no network access
   in this environment).
2. :class:`AllocatorBeatsHeuristicTest` -- the DOE-based allocator (``mixle.doe.bayesopt``) reaches
   lower predicted loss at matched compute than the fixed "20 tokens per parameter" heuristic,
   evaluated under the fitted scaling law's own predictions.
3. :class:`UncertaintyReceiptsTest` -- the posterior over the exponents is genuinely informative:
   it narrows with more data, and a 90% credible interval brackets the true exponent at roughly
   the expected rate across repeated synthetic re-fits with different noise seeds.
"""

from __future__ import annotations

import unittest

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.ppl.scaling_laws import (
    CHINCHILLA_ALPHA,
    CHINCHILLA_BETA,
    CHINCHILLA_E,
    ScalingLawAllocationController,
    allocate_compute,
    allocate_compute_learned,
    allocate_fixed_heuristic,
    fit_scaling_law,
    generate_synthetic_chinchilla_data,
)


class ReproducesKnownExponentsTest(unittest.TestCase):
    def test_fitted_exponents_close_to_published_chinchilla_values(self):
        # SYNTHETIC data (see module docstring): generated from the published Hoffmann et al. 2022
        # exponents alpha=0.34, beta=0.28 (E=1.69) with realistic i.i.d. Gaussian noise.
        records = generate_synthetic_chinchilla_data(n_points=60, seed=0, noise_sd=0.015)
        fit = fit_scaling_law(records, seed=0)

        alpha_hat, beta_hat, e_hat = fit.mean("alpha"), fit.mean("beta"), fit.mean("E")
        print(  # real measured numbers, printed for the record (not just asserted)
            f"fitted alpha={alpha_hat:.4f} (true {CHINCHILLA_ALPHA}), "
            f"beta={beta_hat:.4f} (true {CHINCHILLA_BETA}), E={e_hat:.4f} (true {CHINCHILLA_E})"
        )
        self.assertLess(abs(alpha_hat - CHINCHILLA_ALPHA), 0.05)
        self.assertLess(abs(beta_hat - CHINCHILLA_BETA), 0.05)
        self.assertLess(abs(e_hat - CHINCHILLA_E), 0.3)

        # the published exponents should also fall inside the fitted 90% credible intervals
        alpha_lo, alpha_hi = fit.hdi("alpha", 0.9)
        beta_lo, beta_hi = fit.hdi("beta", 0.9)
        self.assertTrue(alpha_lo - 0.02 <= CHINCHILLA_ALPHA <= alpha_hi + 0.02)
        self.assertTrue(beta_lo - 0.02 <= CHINCHILLA_BETA <= beta_hi + 0.02)


@unittest.skipUnless(_HAS_TORCH, "allocate_compute fits a GaussianProcessRegressor (torch)")
class AllocatorBeatsHeuristicTest(unittest.TestCase):
    def test_doe_allocator_beats_fixed_20to1_heuristic_at_matched_compute(self):
        records = generate_synthetic_chinchilla_data(n_points=60, seed=1, noise_sd=0.015)
        fit = fit_scaling_law(records, seed=1)

        budgets = [1.0e20, 6.0e21, 1.0e23]
        wins = 0
        for i, budget in enumerate(budgets):
            n_doe, d_doe = allocate_compute(fit, budget, seed=i)
            n_heur, d_heur = allocate_fixed_heuristic(budget)

            # both allocations must actually spend (approximately) the same compute
            self.assertAlmostEqual(6.0 * n_doe * d_doe / budget, 1.0, delta=1.0e-6)
            self.assertAlmostEqual(6.0 * n_heur * d_heur / budget, 1.0, delta=1.0e-6)

            loss_doe = fit.predict_mean(n_doe, d_doe)
            loss_heur = fit.predict_mean(n_heur, d_heur)
            print(
                f"budget={budget:.1e}: DOE N={n_doe:.3e} D={d_doe:.3e} loss={loss_doe:.5f} | "
                f"heuristic(20:1) N={n_heur:.3e} D={d_heur:.3e} loss={loss_heur:.5f}"
            )
            self.assertLessEqual(loss_doe, loss_heur + 1.0e-6)
            wins += int(loss_doe < loss_heur - 1.0e-6)

        self.assertGreaterEqual(wins, 2)  # the DOE allocator wins (strictly) on most budgets


class UncertaintyReceiptsTest(unittest.TestCase):
    def test_credible_interval_narrows_with_more_data(self):
        few = fit_scaling_law(generate_synthetic_chinchilla_data(n_points=15, seed=2, noise_sd=0.015), seed=2)
        many = fit_scaling_law(generate_synthetic_chinchilla_data(n_points=150, seed=2, noise_sd=0.015), seed=2)

        width_few = few.hdi("alpha", 0.9)[1] - few.hdi("alpha", 0.9)[0]
        width_many = many.hdi("alpha", 0.9)[1] - many.hdi("alpha", 0.9)[0]
        print(f"alpha 90% HDI width: n=15 -> {width_few:.4f}, n=150 -> {width_many:.4f}")
        self.assertLess(width_many, width_few)

    def test_90pct_credible_interval_has_roughly_90pct_coverage(self):
        n_reps = 10
        hits_alpha = 0
        hits_beta = 0
        for seed in range(n_reps):
            records = generate_synthetic_chinchilla_data(n_points=50, seed=100 + seed, noise_sd=0.015)
            fit = fit_scaling_law(records, seed=100 + seed)
            lo_a, hi_a = fit.hdi("alpha", 0.9)
            lo_b, hi_b = fit.hdi("beta", 0.9)
            hits_alpha += int(lo_a <= CHINCHILLA_ALPHA <= hi_a)
            hits_beta += int(lo_b <= CHINCHILLA_BETA <= hi_b)
        print(f"90% HDI coverage over {n_reps} re-fits: alpha {hits_alpha}/{n_reps}, beta {hits_beta}/{n_reps}")
        # a 90% interval should bracket the truth most of the time; with only n_reps=10 replicates
        # this is a loose (not exact-90%) check -- same tolerant-coverage pattern as
        # examples/flagship_physics_inverse.py's own coverage receipt.
        self.assertGreaterEqual(hits_alpha, 6)
        self.assertGreaterEqual(hits_beta, 6)


@unittest.skipUnless(_HAS_TORCH, "allocate_compute_learned fits a GaussianProcessRegressor (torch)")
class LearnedAllocationControllerTest(unittest.TestCase):
    """Smoke test for the optional D5-pattern (mixle.inference.conditional_jit_controller)
    allocation controller -- not the primary acceptance path (that is AllocatorBeatsHeuristicTest
    against allocate_compute), but confirms the controller wiring actually runs, produces a
    constraint-respecting (N, D) split, and its logged ledger grows as it warm-starts across
    several compute budgets."""

    def test_controller_proposes_feasible_allocations_and_warm_starts(self):
        records = generate_synthetic_chinchilla_data(n_points=60, seed=3, noise_sd=0.015)
        fit = fit_scaling_law(records, seed=3)

        controller = ScalingLawAllocationController(seed=0)
        budgets = [1.0e20, 1.0e21, 1.0e22, 1.0e23, 6.0e21]
        for i, budget in enumerate(budgets):
            n_val, d_val, controller = allocate_compute_learned(fit, budget, controller=controller)
            self.assertGreater(n_val, 0.0)
            self.assertGreater(d_val, 0.0)
            self.assertAlmostEqual(6.0 * n_val * d_val / budget, 1.0, delta=1.0e-6)
            self.assertEqual(len(controller.design), i + 1)  # each call logs exactly one row


if __name__ == "__main__":
    unittest.main()
