"""End-to-end tests for the first executable typed local update path."""

import json

import numpy as np
import pytest

from mixle.experimental.typed_runtime import run_typed_mixture_em
from mixle.inference.em import PosteriorTransformEM, observed_log_likelihood
from mixle.stats import (
    DirichletDistribution,
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    seq_encode,
)

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _problem(seed=19, nobs=240):
    truth = MixtureDistribution(
        [GaussianDistribution(-4.0, 0.7), GaussianDistribution(4.0, 0.7)],
        [0.55, 0.45],
    )
    data = truth.sampler(seed=seed).sample(size=nobs)
    start = MixtureDistribution(
        [
            GaussianDistribution(-1.0, 3.0),
            GaussianDistribution(1.0, 3.0),
            GaussianDistribution(-18.0, 2.0),
            GaussianDistribution(18.0, 2.0),
        ],
        [0.45, 0.45, 0.05, 0.05],
    )
    estimator = MixtureEstimator([GaussianEstimator() for _ in start.components])
    return start, estimator, seq_encode(data, model=start)


class TypedLocalExecutionTest:
    def test_real_partial_updates_are_monotone_receipted_and_cheaper_than_full_tree_work(self):
        start, estimator, encoded = _problem()
        rounds = 12
        run = run_typed_mixture_em(encoded, estimator, start, max_its=rounds, delta=None)

        assert len(run.rounds) == rounds
        assert all(np.isfinite(run.objective_trace))
        assert all(right >= left - 1.0e-9 for left, right in zip(run.objective_trace, run.objective_trace[1:]))
        assert all(len(receipt.active_components) < start.num_components for receipt in run.rounds)
        assert all(receipt.coordinator_nodes == (run.graph.root_node,) for receipt in run.rounds)
        assert all(run.graph.root_node not in receipt.schedule.selected_nodes for receipt in run.rounds)
        assert all(receipt.gain_attribution == "joint_with_coordinator" for receipt in run.rounds)
        assert run.total_model_evaluations < 2 * start.num_components * rounds
        assert any(receipt.invalidation is not None for receipt in run.rounds)
        json.dumps(run.as_dict(), allow_nan=False)

    def test_reaches_same_target_with_fewer_component_evaluations_than_full_tree_em(self):
        start, estimator, encoded = _problem(nobs=80)
        rounds = 60
        typed = run_typed_mixture_em(encoded, estimator, start, max_its=rounds, delta=None)

        objective = observed_log_likelihood(encoded)
        strategy = PosteriorTransformEM()
        full_model = start
        full_trace = []
        for _ in range(rounds):
            full_model = strategy.step(encoded, estimator, full_model, objective=objective).model
            full_trace.append(objective(full_model))

        target = min(typed.objective_trace[-1], full_trace[-1]) - 1.0e-3
        typed_evaluations = next(
            sum(receipt.work.model_evaluations for receipt in typed.rounds[: index + 1])
            for index, value in enumerate(typed.objective_trace)
            if value >= target
        )
        full_evaluations = next(
            2 * start.num_components * (index + 1) for index, value in enumerate(full_trace) if value >= target
        )

        assert typed_evaluations < full_evaluations
        assert full_evaluations / typed_evaluations > 1.05

    def test_scheduler_receipt_maps_typed_nodes_to_real_component_indices(self):
        start, estimator, encoded = _problem(nobs=80)
        run = run_typed_mixture_em(encoded, estimator, start, max_its=3, delta=None)
        component_ids = {id(component) for component in start.components}

        for receipt in run.rounds:
            selected_models = {id(run.graph.node(node_id).model) for node_id in receipt.schedule.selected_nodes}
            active_models = {id(start.components[index]) for index in receipt.active_components}
            assert selected_models == active_models
            assert selected_models <= component_ids

    def test_shared_component_fails_before_execution_until_joint_proposals_exist(self):
        shared = GaussianDistribution(0.0, 1.0)
        model = MixtureDistribution([shared, shared], [0.5, 0.5])
        estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        encoded = seq_encode([0.0, 1.0], model=model)

        with pytest.raises(NotImplementedError, match="shared component"):
            run_typed_mixture_em(encoded, estimator, model)

    def test_conjugate_weight_prior_fails_before_execution_until_map_adapter_exists(self):
        start, _, encoded = _problem(nobs=40)
        estimator = MixtureEstimator(
            [GaussianEstimator() for _ in start.components],
            prior=DirichletDistribution(np.full(start.num_components, 2.0)),
        )

        with pytest.raises(NotImplementedError, match="observed-data MLE"):
            run_typed_mixture_em(encoded, estimator, start)

    def test_bad_candidate_is_rejected_without_objective_regression(self, monkeypatch):
        start, estimator, encoded = _problem(nobs=80)

        from mixle.experimental.typed_runtime import local

        original = local._m_step

        def bad_step(enc_data, est, model, gamma, inactive):
            candidate = original(enc_data, est, model, gamma, inactive)
            return MixtureDistribution(
                [
                    GaussianDistribution(1.0e6, 1.0) if i not in inactive else component
                    for i, component in enumerate(candidate.components)
                ],
                candidate.w,
            )

        monkeypatch.setattr(local, "_m_step", bad_step)
        run = run_typed_mixture_em(encoded, estimator, start, max_its=1, delta=None)
        receipt = run.rounds[0]
        assert not receipt.accepted
        assert receipt.committed_objective == pytest.approx(receipt.objective_before)
        assert receipt.invalidation is None
        assert run.model is start


def _realistic_scale_held_out_problems():
    """Five DIFFERENT fit problems (different seed and dataset size each), 3-9x the toy-scale
    (nobs=80) fixture used by ``test_reaches_same_target_with_fewer_component_evaluations_...``
    above -- mirrors ``conditional_jit_controller_test.py``'s ``_held_out_problems()`` convention
    of reporting real, per-problem numbers across several fits instead of one cherry-picked ratio.
    """
    return [(7, 300), (13, 500), (19, 700), (42, 300), (99, 500)]


class RealisticScaleEfficiencyBenchmarkTestCase:
    """Reports typed local EM's efficiency against full-tree EM at a more realistic scale than the
    existing toy-scale test above, across several held-out problems.

    Honesty note: at this scale the per-problem ratio is modest and not always above 1 -- typed
    occasionally needs marginally MORE component evaluations than full-tree EM to reach the same
    shared target objective (see the mean printed below, and compare against the toy-scale test's
    reliable >1.05x). This is measured on the SAME small (2 real + 2 decoy) fixture family as the
    toy-scale test, verified quality-safe: full-tree and typed EM reach the same or a very close
    objective on every held-out problem here. Contrast this with
    ``LocalOptimumDivergenceRiskTestCase``, which uses a DIFFERENT, many-decoy fixture family where
    typed and full-tree EM can settle at genuinely different objectives -- a regime where an
    evaluation-count ratio would be comparing apples to oranges, not a regime this test's fixture
    exercises. Given the realistic-scale ratio's modesty, this test asserts what is actually
    reliable here -- quality parity and a sane, non-degenerate ratio on every held-out problem --
    and reports (rather than hard-asserts a floor on) the aggregate picture, mirroring
    ``conditional_jit_controller_test.py``'s ``LearnedVsGreedyAcceptanceTestCase`` precedent of
    printing real per-problem numbers instead of promising a universal speedup.
    """

    def test_efficiency_and_quality_parity_across_held_out_problems(self):
        rounds = 100
        ratios = []
        for seed, nobs in _realistic_scale_held_out_problems():
            start, estimator, encoded = _problem(seed=seed, nobs=nobs)
            typed = run_typed_mixture_em(encoded, estimator, start, max_its=rounds, delta=None)

            objective = observed_log_likelihood(encoded)
            strategy = PosteriorTransformEM()
            full_model = start
            full_trace = []
            for _ in range(rounds):
                full_model = strategy.step(encoded, estimator, full_model, objective=objective).model
                full_trace.append(objective(full_model))

            assert all(np.isfinite(typed.objective_trace))
            assert all(right >= left - 1.0e-9 for left, right in zip(typed.objective_trace, typed.objective_trace[1:]))

            diff = abs(typed.objective_trace[-1] - full_trace[-1])
            assert diff < 10.0, (
                "seed=%d nobs=%d: typed and full-tree EM diverged by %.4f nats -- outside the range "
                "this fixture family is verified quality-safe for" % (seed, nobs, diff)
            )

            target = min(typed.objective_trace[-1], full_trace[-1]) - 1.0e-3
            typed_evals = next(
                (
                    sum(receipt.work.model_evaluations for receipt in typed.rounds[: index + 1])
                    for index, value in enumerate(typed.objective_trace)
                    if value >= target
                ),
                None,
            )
            full_evals = next(
                (2 * start.num_components * (index + 1) for index, value in enumerate(full_trace) if value >= target),
                None,
            )
            assert typed_evals is not None, "seed=%d nobs=%d: typed never reached the shared target" % (seed, nobs)
            assert full_evals is not None, "seed=%d nobs=%d: full-tree EM never reached the shared target" % (
                seed,
                nobs,
            )

            ratio = full_evals / typed_evals
            ratios.append(ratio)
            print(
                "\n[typed local EM, realistic scale] seed=%d nobs=%d: diff=%.4f typed=%d evals full=%d evals "
                "ratio=%.3fx" % (seed, nobs, diff, typed_evals, full_evals, ratio)
            )
            # A sanity floor, not a performance promise -- see class docstring on why the ratio is
            # not reliably above 1 at this scale.
            assert ratio > 0.5, "seed=%d nobs=%d: typed needed far more evaluations than full-tree EM" % (seed, nobs)

        print(
            "\n[typed local EM, realistic scale] mean ratio across %d held-out problems: %.3fx"
            % (len(ratios), np.mean(ratios))
        )


def _many_decoy_problem(seed=42, nobs=400):
    """Mirrors ``freeze_rollup_test.py``'s ``_make_problem`` exactly (same numbers) so the two
    modules' behavior on the identical fixture is directly comparable: 2 slow-converging real
    components plus 6 far-away decoys whose optimal weight is small but nonzero.
    """
    truth = MixtureDistribution([GaussianDistribution(-5.0, 0.6), GaussianDistribution(5.0, 0.6)], [0.5, 0.5])
    data = truth.sampler(seed=seed).sample(size=nobs)
    start = MixtureDistribution(
        [
            GaussianDistribution(-0.3, 3.0),
            GaussianDistribution(0.3, 3.0),
            GaussianDistribution(-14.0, 3.0),
            GaussianDistribution(14.0, 3.0),
            GaussianDistribution(-40.0, 3.0),
            GaussianDistribution(40.0, 3.0),
            GaussianDistribution(-70.0, 3.0),
            GaussianDistribution(70.0, 3.0),
        ],
        [0.4, 0.4] + [0.025] * 6,
    )
    estimator = MixtureEstimator([GaussianEstimator() for _ in range(8)])
    return start, estimator, seq_encode(data, model=start)


class LocalOptimumDivergenceRiskTestCase:
    """Characterizes a real, previously-undocumented limitation of budgeted local EM.

    Unlike ``mixle.inference.freeze_rollup`` (which only ever skips a component once its own
    convergence is demonstrated, and so reliably reaches the SAME fixed point as full-tree EM --
    see ``freeze_rollup_test.py``'s ``FreezeRollupSpeedupTestCase`` on this identical fixture),
    ``run_typed_mixture_em``'s default budget rations updates among components from round zero,
    before any gain evidence exists to justify the ranking. On a non-convex mixture likelihood
    this asynchronous coordinate-ascent process can settle at a different, meaningfully worse
    fixed point than full-tree EM -- confirmed here, and confirmed (in exploratory testing, not
    asserted below) to persist in some form at every ``budget_fraction`` short of 1.0, i.e. short
    of updating every component every round. This is not a bug fixed by tuning the default
    budget; it is a documented tradeoff (see local.py's ``run_typed_mixture_em`` docstring and
    this package's README, "Local-optimum risk on multimodal objectives"). This test exists so a
    future change that makes the gap dramatically worse -- or silently closes it, which would
    mean this note and its citation should be removed -- gets noticed rather than passing quietly.
    """

    def test_typed_local_em_can_settle_at_a_different_fixed_point_than_full_tree_em(self):
        start, estimator, encoded = _many_decoy_problem()
        rounds = 500
        typed = run_typed_mixture_em(encoded, estimator, start, max_its=rounds, delta=None)

        objective = observed_log_likelihood(encoded)
        strategy = PosteriorTransformEM()
        full_model = start
        full_trace = []
        for _ in range(rounds):
            full_model = strategy.step(encoded, estimator, full_model, objective=objective).model
            full_trace.append(objective(full_model))

        # Both trajectories must still be individually well-behaved: monotone and finite. The
        # limitation under test is WHICH fixed point each lands on, not whether either is broken.
        assert all(np.isfinite(typed.objective_trace))
        assert all(right >= left - 1.0e-9 for left, right in zip(typed.objective_trace, typed.objective_trace[1:]))
        assert all(np.isfinite(full_trace))
        assert all(right >= left - 1.0e-9 for left, right in zip(full_trace, full_trace[1:]))

        # Both traces have actually settled (small last-20-round movement), so the comparison
        # below is between two fixed points, not an artifact of one trajectory still moving.
        assert max(typed.objective_trace[-20:]) - min(typed.objective_trace[-20:]) < 1.0e-2
        assert max(full_trace[-20:]) - min(full_trace[-20:]) < 1.0e-2

        diff = abs(typed.objective_trace[-1] - full_trace[-1])
        # Non-vacuous: this fixture is known (see class docstring) to actually diverge, not just
        # converge slowly -- a diff near zero here would mean the limitation this test documents
        # has been fixed, which is good news, but means this test and its README/docstring
        # citations are stale and should be revisited rather than left describing a risk that no
        # longer reproduces.
        assert diff > 3.0, (
            "expected fixture-specific divergence was not reproduced -- if a real fix landed, "
            "update this test and the limitation notes it documents instead of loosening this bound"
        )
        # Regression ceiling: catches the gap becoming dramatically worse, not the gap existing.
        assert diff < 40.0, "typed local EM diverged from full-tree EM by far more than previously observed"

        # However far apart the two objectives land, typed's own final model must still be a
        # valid mixture -- this is a different-optimum risk, not a license to produce garbage.
        assert np.isfinite(typed.model.w).all()
        assert typed.model.w.sum() == pytest.approx(1.0, abs=1.0e-6)
        assert (typed.model.w >= 0.0).all()
