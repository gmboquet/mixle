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
