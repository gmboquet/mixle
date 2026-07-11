"""Tests for the D3 block-coordinate-ascent EM scheduler (mixle.inference.block_em).

Acceptance criteria under test (see the ConditionalJIT track's D3 item):

1. Monotone-objective test -- observed-data log likelihood never
   decreases round-to-round under ``schedule="auto"``, the same coordinate-ascent guarantee
   vanilla EM has, in the same style as D2's own monotone-objective test.
2. Work receipt on the D2 fixture -- ``schedule="auto"`` reaches the SAME target objective using fewer
   component log-density evaluations. This is explicitly not treated as a wall-clock guarantee.
3. Degenerates to vanilla EM -- when every eligible block's gain-per-cost score is
   indistinguishable, ``schedule="auto"``'s behavior is numerically indistinguishable from
   vanilla full-tree EM's.
"""

import unittest

import numpy as np

from mixle.inference.block_em import run_block_em
from mixle.inference.em import PosteriorTransformEM, observed_log_likelihood, run_em
from mixle.inference.estimation import optimize
from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator, seq_encode


def _make_problem(seed=42, nobs=400):
    """A mixture with 2 slow-converging real components plus 6 far-away decoy components.

    Identical to the D2 fixture (``mixle.tests.freeze_rollup_test._make_problem``) -- reused
    directly per the roadmap item's instruction to mirror D2's own test fixture, rather than
    importing across test modules (test modules are not meant to be a shared library surface).
    """
    truth = MixtureDistribution([GaussianDistribution(-5.0, 0.6), GaussianDistribution(5.0, 0.6)], [0.5, 0.5])
    data = truth.sampler(seed=seed).sample(size=nobs)
    start_components = [
        GaussianDistribution(-0.3, 3.0),
        GaussianDistribution(0.3, 3.0),
        GaussianDistribution(-14.0, 3.0),
        GaussianDistribution(14.0, 3.0),
        GaussianDistribution(-40.0, 3.0),
        GaussianDistribution(40.0, 3.0),
        GaussianDistribution(-70.0, 3.0),
        GaussianDistribution(70.0, 3.0),
    ]
    start = MixtureDistribution(start_components, [0.4, 0.4] + [0.025] * 6)
    estimator = MixtureEstimator([GaussianEstimator() for _ in range(8)])
    enc = seq_encode(data, model=start)
    return start, estimator, enc


def _make_symmetric_problem(seed=3, nobs=200, num_components=4):
    """A mixture of ``num_components`` BYTE-IDENTICAL components at equal weight.

    Every component has the exact same distribution and the exact same weight, so D1's
    gain-per-cost report (residual, cost) is identical across every component every round (the
    scheduler scores with a fixed shared seed -- see ``block_em._SCORE_SEED`` -- specifically so
    this holds deterministically, not by RNG luck). This is the literal "no useful discrimination
    to make" scenario the degeneration acceptance criterion asks for.
    """
    truth = GaussianDistribution(0.0, 1.5)
    data = truth.sampler(seed=seed).sample(size=nobs)
    start = MixtureDistribution(
        [GaussianDistribution(0.1, 2.0) for _ in range(num_components)],
        [1.0 / num_components] * num_components,
    )
    estimator = MixtureEstimator([GaussianEstimator() for _ in range(num_components)])
    enc = seq_encode(data, model=start)
    return start, estimator, enc


class BlockEMMonotonicityTestCase(unittest.TestCase):
    def test_free_energy_is_monotone_round_to_round(self):
        """Observed-data log likelihood never decreases round-to-round under schedule='auto'.

        Same style/assertion as D2's ``FreezeRollupMonotonicityTestCase`` -- the D-track's
        correctness backbone is that a learned/greedy scheduling decision changes SPEED, never
        whether the objective goes up; ``run_block_em`` enforces this with an accept/reject gate on every
        round's real objective, so this test checks that gate is wired correctly end to end.
        """
        start, estimator, enc = _make_problem(seed=7, nobs=300)
        _, history = run_block_em(enc, estimator, start, max_its=150, delta=1.0e-10, budget_fraction=0.5)

        self.assertGreater(len(history), 1)
        objectives = [h.objective for h in history]
        for i in range(1, len(objectives)):
            self.assertGreaterEqual(
                objectives[i],
                objectives[i - 1] - 1.0e-9,
                "objective decreased from round %d to %d: %r -> %r" % (i - 1, i, objectives[i - 1], objectives[i]),
            )
        # At least one round should actually have scheduled a genuine subset (n_active <
        # n_components - n_frozen) -- otherwise the monotonicity check is vacuous (every round
        # behaved like full-tree EM and the scheduler was never really exercised).
        self.assertTrue(
            any(h.n_active < h.n_components - h.n_frozen and not h.degenerate_round for h in history),
            "scheduler never actually chose a proper subset of blocks during this run",
        )
        self.assertTrue(all(h.wall_time_seconds > 0.0 for h in history))
        self.assertTrue(all(np.isfinite(h.measured_q_gain) for h in history))


class BlockEMSpeedupTestCase(unittest.TestCase):
    def test_evaluation_count_receipt_matches_active_fraction(self):
        """The scheduler reaches a shared target with fewer component-density evaluations.

        This pins removed model work only. Python scheduling, cache hashing, and block ranking are
        real costs, so the result must not be quoted as a wall-clock speedup.
        """
        start, estimator, enc = _make_problem()
        num_components = start.num_components

        _, block_history = run_block_em(
            enc,
            estimator,
            start,
            max_its=400,
            delta=None,  # run the full budget -- target crossing is measured from the trace itself
            budget_fraction=0.5,
            weight_tol=0.05,
            q_gain_tol=1.0e-5,
            weight_delta_tol=1.0e-11,
            freeze_patience=10,
        )

        # Vanilla's own per-round trace, over the same number of rounds, so both traces are
        # measured the same way. PosteriorTransformEM's own E-step (K components) plus run_em's
        # explicit objective(candidate) convergence check (another K components) is the same
        # 2-evaluations-per-component-per-round structure D2's own speedup test accounts for.
        strategy = PosteriorTransformEM()
        objective = observed_log_likelihood(enc)
        vanilla_model = start
        vanilla_trace = [objective(vanilla_model)]
        for _ in range(len(block_history)):
            vanilla_model = strategy.step(enc, estimator, vanilla_model, objective=objective).model
            vanilla_trace.append(objective(vanilla_model))
        vanilla_cum_evals = [2 * num_components * (i + 1) for i in range(len(vanilla_trace))]

        block_trace = [h.objective for h in block_history]
        block_cum_evals = list(np.cumsum([h.n_log_density_evals for h in block_history]))

        # Pick a target objective both traces actually reach (a small margin below whichever trajectory's
        # own final value is smaller), rather than demanding the two runs land on the EXACT same
        # final fixed point -- block-coordinate EM on a real (non-synthetic-for-this-test) mixture
        # can legitimately settle into a very slightly different local optimum than vanilla EM
        # depending on update order (both are still valid coordinate ascent -- see the
        # monotone-objective test); "reaches the same target in fewer evaluations" is the honest,
        # order-independent form of the roadmap item's acceptance criterion.
        target = min(vanilla_trace[-1], block_trace[-1]) - 1.0e-3

        def _evals_to_target(trace, cum_evals):
            for value, evals in zip(trace, cum_evals):
                if value >= target:
                    return evals
            self.fail("trace never reached the target objective")

        vanilla_evals = _evals_to_target(vanilla_trace, vanilla_cum_evals)
        block_evals = _evals_to_target(block_trace, block_cum_evals)

        self.assertLess(
            block_evals, vanilla_evals, "block-EM needed at least as many log-density evals as vanilla EM would"
        )
        ratio = vanilla_evals / block_evals
        self.assertGreater(ratio, 1.1)
        print(
            "\nblock-EM work receipt: target objective=%.4f reached in %d vs %d log-density evals, ratio=%.3fx"
            % (target, block_evals, vanilla_evals, ratio)
        )


class BlockEMDegenerationTestCase(unittest.TestCase):
    def test_degenerates_to_vanilla_em_when_blocks_are_indistinguishable(self):
        """When every eligible block has the same gain-per-cost, schedule='auto' must do exactly
        what vanilla full-tree EM does -- a literal test of the "degenerates to vanilla EM"
        acceptance criterion, not just an assumption.
        """
        start, estimator, enc = _make_symmetric_problem()
        max_its = 40

        model, history = run_block_em(
            enc,
            estimator,
            start,
            max_its=max_its,
            delta=None,  # run every round, so this exactly matches run_em's fixed max_its loop
            budget_fraction=0.5,  # a real budget cap -- the degeneration must come from the TIE, not from a trivially permissive budget
        )
        reference = run_em(enc, estimator, start, strategy=PosteriorTransformEM(), max_its=max_its, delta=None)

        objective = observed_log_likelihood(enc)
        self.assertAlmostEqual(objective(model), objective(reference), places=6)
        np.testing.assert_allclose(model.w, reference.w, atol=1.0e-8)
        np.testing.assert_allclose(
            sorted(c.mu for c in model.components), sorted(c.mu for c in reference.components), atol=1.0e-6
        )

        # Every round in this run should have been recognized as having no useful discrimination
        # to make (all-identical components => all-identical gain-per-cost scores).
        self.assertTrue(len(history) > 0)
        self.assertTrue(all(h.degenerate_round for h in history), "expected every round to be a degenerate round")
        self.assertTrue(all(h.n_active == h.n_components - h.n_frozen for h in history))

    def test_full_tree_every_round_escape_hatch_matches_vanilla_em(self):
        """The explicit ``full_tree_every_round=True`` escape hatch is a second, independent way
        to hit the same degeneration property (spec: "or a full_tree_every_round=True-style
        escape hatch"), even on a fixture where components genuinely differ (so a real budget
        WOULD otherwise produce a proper subset -- see BlockEMSpeedupTestCase).
        """
        start, estimator, enc = _make_problem(seed=99, nobs=200)
        max_its = 25

        model, history = run_block_em(
            enc, estimator, start, max_its=max_its, delta=None, budget_fraction=0.1, full_tree_every_round=True
        )
        reference = run_em(enc, estimator, start, strategy=PosteriorTransformEM(), max_its=max_its, delta=None)

        objective = observed_log_likelihood(enc)
        self.assertAlmostEqual(objective(model), objective(reference), places=6)
        np.testing.assert_allclose(model.w, reference.w, atol=1.0e-8)
        self.assertTrue(all(h.degenerate_round for h in history))


class BlockEMOptimizeDispatchTestCase(unittest.TestCase):
    def test_optimize_schedule_auto_is_a_real_dispatchable_api(self):
        """``optimize(..., schedule="auto")`` is the real, top-level, user-facing entry point the
        roadmap item asks for -- not just an internal ``run_block_em`` function nobody calls.
        """
        start, estimator, enc = _make_problem(seed=13, nobs=200)

        fitted = optimize(
            None,
            estimator,
            enc_data=enc,
            prev_estimate=start,
            max_its=30,
            delta=1.0e-9,
            schedule="auto",
            out=None,
        )
        self.assertIsInstance(fitted, MixtureDistribution)
        self.assertEqual(fitted.num_components, start.num_components)
        # A legitimate fit: total responsibility-weighted mass should be non-trivial and finite.
        objective = observed_log_likelihood(enc)
        self.assertTrue(np.isfinite(objective(fitted)))

    def test_optimize_schedule_auto_falls_back_for_non_mixture_models(self):
        """``schedule='auto'`` on a model block-EM does not know how to schedule (anything that
        isn't a plain local MLE MixtureDistribution/MixtureEstimator fit) must silently behave
        like ``schedule='full'`` -- never error, never silently do nothing.
        """
        data = GaussianDistribution(0.0, 1.0).sampler(seed=1).sample(size=200)
        estimator = GaussianEstimator()

        fitted_auto = optimize(data, estimator, max_its=10, delta=1.0e-9, schedule="auto", out=None, rng=None)
        fitted_full = optimize(data, estimator, max_its=10, delta=1.0e-9, schedule="full", out=None, rng=None)
        self.assertAlmostEqual(fitted_auto.mu, fitted_full.mu, places=8)
        self.assertAlmostEqual(fitted_auto.sigma2, fitted_full.sigma2, places=8)


if __name__ == "__main__":
    unittest.main()
