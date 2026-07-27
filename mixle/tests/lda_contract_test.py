"""Probability, evidence, and convergence contracts for latent Dirichlet allocation."""

import numpy as np
import pytest

from mixle.stats import CategoricalDistribution, CategoricalEstimator
from mixle.stats.latent.lda import (
    LDAConvergenceError,
    LDADistribution,
    LDAEstimator,
    seq_posterior_with_diagnostics,
    update_alpha,
)
from mixle.utils.vector import ImpossibleEvidenceError


def _model(**kwargs):
    return LDADistribution(
        [
            CategoricalDistribution({"a": 0.8, "b": 0.2}),
            CategoricalDistribution({"a": 0.1, "b": 0.9}),
        ],
        [0.7, 0.4],
        **kwargs,
    )


@pytest.mark.parametrize(
    "topics,alpha,kwargs",
    [
        ([], [], {}),
        ([CategoricalDistribution({"a": 1.0})], [1.0, 2.0], {}),
        ([CategoricalDistribution({"a": 1.0})], [0.0], {}),
        ([CategoricalDistribution({"a": 1.0})], [np.inf], {}),
        ([CategoricalDistribution({"a": 1.0})], [1.0], {"gamma_threshold": 0.0}),
        ([CategoricalDistribution({"a": 1.0})], [1.0], {"max_gamma_iter": 1.5}),
    ],
)
def test_distribution_rejects_invalid_geometry_and_controls(topics, alpha, kwargs):
    with pytest.raises((TypeError, ValueError)):
        LDADistribution(topics, alpha, **kwargs)


@pytest.mark.parametrize(
    "document",
    [
        [("a", True)],
        [("a", 0.0)],
        [("a", -1.0)],
        [("a", np.inf)],
        [("a", np.nan)],
        [("a", 1.0, "extra")],
    ],
)
def test_encoder_rejects_invalid_grouped_counts(document):
    with pytest.raises((TypeError, ValueError)):
        _model().dist_to_encoder().seq_encode([document])


def test_encoded_ids_counts_and_warm_starts_are_validated():
    model = _model()
    enc = model.dist_to_encoder().seq_encode([[("a", 1.0)]])
    with pytest.raises(ValueError, match="document IDs"):
        model.seq_log_density((1, np.array([1]), enc[2], None, enc[4]))
    with pytest.raises(ValueError, match="counts"):
        model.seq_log_density((1, enc[1], np.array([0.0]), None, enc[4]))
    with pytest.raises(ValueError, match="warm-start"):
        model.seq_log_density((1, enc[1], enc[2], np.ones((1, 3)), enc[4]))
    assert model.dist_to_encoder().row_count(enc) == 1


def test_impossible_evidence_scores_minus_infinity_and_update_is_transactional():
    model = _model()
    enc = model.dist_to_encoder().seq_encode([[("missing", 1.0)]])
    assert np.isneginf(model.seq_log_density(enc)[0])
    responsibilities, _, _, diagnostics = seq_posterior_with_diagnostics(model, enc)
    np.testing.assert_array_equal(responsibilities, np.zeros((1, 2)))
    assert diagnostics.impossible_documents == (0,)

    accumulator = LDAEstimator([CategoricalEstimator(), CategoricalEstimator()]).accumulator_factory().make()
    before = accumulator.value()
    with pytest.raises(ImpossibleEvidenceError):
        accumulator.seq_update(enc, np.array([1.0]), model)
    after = accumulator.value()
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2] == before[2]
    np.testing.assert_array_equal(after[3], before[3])
    assert after[4] == before[4]


def test_zero_weight_impossible_evidence_adds_no_training_mass():
    model = _model()
    enc = model.dist_to_encoder().seq_encode([[("missing", 1.0)]])
    accumulator = model.estimator().accumulator_factory().make()
    accumulator.seq_update(enc, np.array([0.0]), model)
    _, sum_logs, document_count, topic_counts, topic_stats, _ = accumulator.value()
    np.testing.assert_array_equal(sum_logs, np.zeros(2))
    assert document_count == 0.0
    np.testing.assert_array_equal(topic_counts, np.zeros(2))
    assert topic_stats == [{}, {}]


def test_document_coordinate_ascent_reports_budget_exhaustion():
    model = _model(gamma_threshold=1.0e-30, max_gamma_iter=1)
    enc = model.dist_to_encoder().seq_encode([[("a", 3.0), ("b", 2.0)]])
    with pytest.raises(LDAConvergenceError) as caught:
        model.seq_log_density(enc)
    diagnostics = caught.value.diagnostics
    assert not diagnostics.converged
    assert diagnostics.iterations == diagnostics.max_iterations == 1
    assert diagnostics.termination_reason == "iteration_budget_exhausted"


def test_alpha_update_is_bounded_monotone_and_receipted():
    alpha, iterations, diagnostics = update_alpha(
        np.array([1.0, 1.0]),
        np.array([-1.0, -1.0]),
        1.0e-10,
        max_iter=100,
        return_diagnostics=True,
    )
    assert diagnostics.converged
    assert iterations == diagnostics.iterations <= 100
    assert np.all(alpha > 0.0)
    assert np.all(np.diff(diagnostics.objective_trace) >= -1.0e-12)

    with pytest.raises(LDAConvergenceError) as caught:
        update_alpha(
            np.array([1.0, 1.0]),
            np.array([-2.0, -0.5]),
            1.0e-30,
            max_iter=1,
        )
    assert caught.value.diagnostics.termination_reason == "iteration_budget_exhausted"


@pytest.mark.parametrize(
    "weights",
    [np.array([True]), np.array([-1.0]), np.array([np.nan]), np.array([1.0, 2.0])],
)
def test_accumulator_rejects_invalid_document_weights(weights):
    model = _model()
    enc = model.dist_to_encoder().seq_encode([[("a", 1.0)]])
    accumulator = model.estimator().accumulator_factory().make()
    with pytest.raises((TypeError, ValueError)):
        accumulator.seq_update(enc, weights, model)
