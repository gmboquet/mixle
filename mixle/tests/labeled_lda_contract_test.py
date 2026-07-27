"""Joint-law, label-schema, evidence, and optimizer contracts for labeled LDA."""

import numpy as np
import pytest

from mixle.engines import NUMPY_ENGINE
from mixle.stats import CategoricalDistribution, CategoricalEstimator, PoissonDistribution
from mixle.stats.compute.backend import backend_seq_log_density
from mixle.stats.latent.labeled_lda import (
    LabeledLDADistribution,
    LabeledLDAEstimator,
    update_alpha_coupled,
)
from mixle.stats.latent.lda import LDAConvergenceError
from mixle.utils.vector import ImpossibleEvidenceError


def _model(*, structural=False, **kwargs):
    set_dist = CategoricalDistribution({(0,): 0.25, (1,): 0.75}) if structural else None
    len_dist = PoissonDistribution(3.0) if structural else None
    return LabeledLDADistribution(
        [
            CategoricalDistribution({"a": 0.8, "b": 0.2}),
            CategoricalDistribution({"a": 0.1, "b": 0.9}),
        ],
        [[0.7, 0.4], [0.5, 1.2]],
        set_dist=set_dist,
        len_dist=len_dist,
        **kwargs,
    )


@pytest.mark.parametrize(
    "alphas,kwargs",
    [
        ([], {}),
        ([[1.0]], {}),
        ([[1.0, 0.0]], {}),
        ([[1.0, np.inf]], {}),
        ([[1.0, 1.0]], {"gamma_threshold": 0.0}),
        ([[1.0, 1.0]], {"max_gamma_iter": 1.5}),
    ],
)
def test_distribution_rejects_invalid_alpha_geometry_and_controls(alphas, kwargs):
    topics = [CategoricalDistribution({"a": 1.0}), CategoricalDistribution({"a": 1.0})]
    with pytest.raises((TypeError, ValueError)):
        LabeledLDADistribution(topics, alphas, **kwargs)


@pytest.mark.parametrize("labels", [[], [-1], [2], [0.5], [True], ["0"]])
def test_encoder_rejects_invalid_label_sets(labels):
    with pytest.raises((TypeError, ValueError)):
        _model().dist_to_encoder().seq_encode([([("a", 1.0)], labels)])


def test_encoder_canonicalizes_duplicate_labels_and_preserves_document_rows():
    encoder = _model().dist_to_encoder()
    duplicate = encoder.seq_encode([([("a", 1.0)], [1, 0, 1])])
    canonical = encoder.seq_encode([([("a", 1.0)], [0, 1])])
    np.testing.assert_array_equal(duplicate[5], np.array([0, 1]))
    np.testing.assert_array_equal(duplicate[6], np.array([2]))
    assert encoder.row_count(duplicate) == 1
    np.testing.assert_array_equal(duplicate[5:8][0], canonical[5:8][0])
    assert _model().seq_log_density(duplicate)[0] == pytest.approx(_model().seq_log_density(canonical)[0])


def test_joint_score_includes_label_set_and_document_length_laws():
    conditional = _model()
    joint = _model(structural=True)
    observation = ([("a", 2.0), ("b", 1.0)], [1])
    conditional_score = conditional.log_density(observation)
    expected = conditional_score + joint.set_dist.log_density((1,)) + joint.len_dist.log_density(3.0)
    assert joint.log_density(observation) == pytest.approx(expected)

    encoded = joint.dist_to_encoder().seq_encode([observation])
    backend = np.asarray(NUMPY_ENGINE.to_numpy(backend_seq_log_density(joint, encoded, NUMPY_ENGINE)))
    np.testing.assert_allclose(backend, joint.seq_log_density(encoded), rtol=1.0e-12, atol=1.0e-12)


def test_estimator_preserves_fixed_structural_laws():
    model = _model(structural=True)
    estimator = model.estimator()
    assert estimator.set_dist is model.set_dist
    assert estimator.len_dist is model.len_dist

    encoded = model.dist_to_encoder().seq_encode(
        [([("a", 2.0)], [0]), ([("b", 3.0)], [1]), ([("a", 1.0), ("b", 1.0)], [0])]
    )
    accumulator = estimator.accumulator_factory().make()
    accumulator.seq_update(encoded, np.ones(3), model)
    fitted = estimator.estimate(3.0, accumulator.value())
    assert fitted.set_dist is model.set_dist
    assert fitted.len_dist is model.len_dist


def test_impossible_evidence_scores_minus_infinity_and_update_is_transactional():
    model = _model()
    encoded = model.dist_to_encoder().seq_encode([([("missing", 1.0)], [0])])
    assert np.isneginf(model.seq_log_density(encoded)[0])
    estimator = LabeledLDAEstimator([CategoricalEstimator(), CategoricalEstimator()], num_alphas=2)
    accumulator = estimator.accumulator_factory().make()
    before = accumulator.value()
    with pytest.raises(ImpossibleEvidenceError):
        accumulator.seq_update(encoded, np.array([1.0]), model)
    after = accumulator.value()
    assert after[1].stats == before[1].stats
    assert after[2] == before[2]
    np.testing.assert_array_equal(after[3], before[3])
    assert after[4] == before[4]

    joint = _model(structural=True)
    impossible_set = joint.dist_to_encoder().seq_encode([([("a", 1.0)], [0, 1])])
    assert np.isneginf(joint.seq_log_density(impossible_set)[0])
    structural_accumulator = joint.estimator().accumulator_factory().make()
    with pytest.raises(ImpossibleEvidenceError):
        structural_accumulator.seq_update(impossible_set, np.array([1.0]), joint)


def test_coupled_alpha_optimizer_is_bounded_and_reports_failure():
    with pytest.raises(LDAConvergenceError) as caught:
        update_alpha_coupled(
            np.ones((2, 2)),
            [(0,), (0, 1), (1,)],
            np.array([10.0, 10.0, 10.0]),
            np.array([[-0.4, -1.4], [-0.8, -0.8], [-1.4, -0.4]]),
            1.0e-30,
            max_its=1,
        )
    diagnostics = caught.value.diagnostics
    assert not diagnostics.converged
    assert diagnostics.iterations == diagnostics.max_iterations == 1
    assert diagnostics.termination_reason == "iteration_budget_exhausted"
