"""Leakage and malformed inputs must fail before coverage-bearing results are issued."""

import numpy as np
import pytest

from mixle.inference.conformal import (
    conformal_label_sets,
    cv_plus,
    jackknife_plus,
    mondrian_conformal,
    split_conformal,
    weighted_conformal,
)
from mixle.inference.cross_validation import (
    group_kfold,
    purged_kfold,
    spatial_block_kfold,
    stratified_kfold,
    time_series_split,
)


def _mean_predict(x_train, y_train, x_eval):
    return np.full(len(x_eval), np.mean(y_train))


@pytest.mark.parametrize(
    "builder",
    [
        lambda: time_series_split(20, 3, gap=-1),
        lambda: time_series_split(20, 3, max_train_size=0),
        lambda: purged_kfold(20, 4, embargo=-1),
        lambda: purged_kfold(20, 4, embargo=20),
        lambda: stratified_kfold(np.array([0, 0, 1, 1]), 4),
        lambda: group_kfold(np.array([0, 0, 1, 1]), 1),
        lambda: spatial_block_kfold(np.zeros((5, 2)), 2),
    ],
)
def test_invalid_fold_geometry_is_rejected(builder):
    with pytest.raises(ValueError):
        builder()


def test_every_returned_temporal_fold_is_nonempty_and_disjoint():
    for folds in (time_series_split(30, 4, gap=2), purged_kfold(30, 3, embargo=2)):
        for train, test in folds:
            assert len(train) and len(test)
            assert not np.intersect1d(train, test).size


@pytest.mark.parametrize(
    "call",
    [
        lambda: split_conformal(np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0])),
        lambda: split_conformal(np.array([0.0]), np.array([0.0]), np.array([np.nan])),
        lambda: weighted_conformal(
            np.array([0.0]), np.array([0.0]), np.array([np.inf]), np.array([1.0])
        ),
        lambda: mondrian_conformal(
            np.array([0.0]), np.array([0.0]), np.array([np.nan]), np.array([0.0]), np.array([0])
        ),
        lambda: jackknife_plus(np.ones((1, 1)), np.ones(1), _mean_predict, np.ones((1, 1))),
        lambda: cv_plus(np.ones((3, 1)), np.ones(2), _mean_predict, np.ones((1, 1))),
        lambda: cv_plus(np.ones((3, 1)), np.ones(3), _mean_predict, np.ones((1, 1)), n_folds=4),
    ],
)
def test_malformed_conformal_inputs_are_rejected(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_fit_predict_output_must_have_exact_finite_shape():
    def too_short(x_train, y_train, x_eval):
        return np.zeros(max(0, len(x_eval) - 1))

    def nonfinite(x_train, y_train, x_eval):
        return np.full(len(x_eval), np.nan)

    x = np.arange(6.0)[:, None]
    for predictor in (too_short, nonfinite):
        with pytest.raises(ValueError):
            cv_plus(x, x[:, 0], predictor, x[:2], n_folds=3)


@pytest.mark.parametrize(
    ("calibration", "test_prob", "qhat"),
    [
        (np.array([1.2]), np.array([[0.5, 0.5]]), None),
        (np.array([0.8]), np.array([[0.6, 0.6]]), None),
        (np.array([0.8]), np.array([[0.5, 0.5]]), np.nan),
        (np.array([0.8]), np.array([[0.5, 0.5]]), -0.1),
        (np.array([0.8]), np.array([[0.5, 0.5]]), 1.1),
    ],
)
def test_classification_conformal_requires_probabilities_and_valid_quantiles(
    calibration, test_prob, qhat
):
    with pytest.raises(ValueError):
        conformal_label_sets(calibration, test_prob, qhat=qhat)
