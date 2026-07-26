"""Classification and ROC contracts that prevent optimistic evaluation."""

import numpy as np
import pytest

from mixle.utils.metrics import auc, classify, roc_auc, roc_curve


class _Encoder:
    def seq_encode(self, rows):
        return rows


class _Classifier:
    def __init__(self, scores=None):
        self.scores = scores or {"a": [2.0, 0.0], "b": [0.0, 2.0]}

    def dist_to_encoder(self):
        return _Encoder()

    def seq_log_density(self, rows):
        return np.asarray(self.scores[rows[0][0]], dtype=float)


def test_classify_uses_one_declared_column_per_unique_label():
    ranks, true_probabilities, observed, probabilities = classify(
        [("a", "first"), ("b", "second")],
        _Classifier(),
        labels=["b", "a"],
    )
    np.testing.assert_array_equal(ranks, [0, 0])
    np.testing.assert_array_equal(observed, ["a", "b"])
    assert set(probabilities) == {"a", "b"}
    assert np.all(true_probabilities > 0.8)
    np.testing.assert_allclose(
        probabilities["a"] + probabilities["b"],
        np.ones(2),
    )


@pytest.mark.parametrize(
    ("data", "labels", "message"),
    [
        ([], ["a"], "at least one"),
        ([("a", 1)], ["a", "a"], "unique"),
        ([("missing", 1)], ["a", "b"], "outside"),
        ([("a", 1, 2)], ["a"], "exactly"),
    ],
)
def test_classify_rejects_invalid_label_probability_contracts(data, labels, message):
    with pytest.raises(ValueError, match=message):
        classify(data, _Classifier(), labels=labels)


def test_classify_rejects_short_or_nonfinite_score_vectors():
    with pytest.raises(ValueError, match="one score per row"):
        classify(
            [("a", 1), ("b", 2)],
            _Classifier(scores={"a": [0.0], "b": [0.0]}),
            labels=["a", "b"],
        )
    with pytest.raises(ValueError, match="finite"):
        classify(
            [("a", 1), ("b", 2)],
            _Classifier(scores={"a": [0.0, np.inf], "b": [0.0, 0.0]}),
            labels=["a", "b"],
        )


def test_equal_positive_and_negative_scores_have_half_auc():
    detection, false_alarm = roc_curve([0.5], [0.5])
    np.testing.assert_allclose(detection, [0.0, 1.0])
    np.testing.assert_allclose(false_alarm, [0.0, 1.0])
    assert roc_auc([0.5], [0.5]) == pytest.approx(0.5)


def test_roc_groups_larger_tie_blocks_at_one_threshold():
    detection, false_alarm = roc_curve([0.8, 0.5], [0.8, 0.2])
    np.testing.assert_allclose(detection, [0.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(false_alarm, [0.0, 0.5, 0.5, 1.0])


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        ([], [0.0]),
        ([0.0], []),
        ([np.nan], [0.0]),
        ([[0.5]], [0.0]),
    ],
)
def test_roc_rejects_missing_classes_nonfinite_and_nonvector_scores(positive, negative):
    with pytest.raises(ValueError):
        roc_curve(positive, negative)


def test_auc_rejects_empty_or_nonfinite_curves():
    with pytest.raises(ValueError, match="two curve points"):
        auc([0.0], [0.0])
    with pytest.raises(ValueError, match="finite"):
        auc([0.0, 1.0], [0.0, np.nan])
