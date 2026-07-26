"""Focused evidence contracts for held-out evaluation utilities."""

import numpy as np
import pytest

import mixle.utils.evaluation as evaluation


def test_empirical_kl_is_the_mean_log_density_ratio(monkeypatch):
    scores = np.array(
        [
            [0.0, np.log(2.0), np.log(2.0)],
            [0.0, 0.0, 0.0],
        ]
    )
    monkeypatch.setattr(evaluation, "seq_log_density", lambda data, estimate: [scores])
    estimate, bad1, bad2 = evaluation.empirical_kl_divergence(
        object(),
        object(),
        [(3, object())],
    )
    assert estimate == pytest.approx(2.0 * np.log(2.0) / 3.0)
    assert (bad1, bad2) == (0, 0)


def test_empirical_kl_counts_every_nonfinite_score_and_requires_joint_evidence(monkeypatch):
    scores = np.array(
        [
            [0.0, np.inf, np.nan, -np.inf],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    monkeypatch.setattr(evaluation, "seq_log_density", lambda data, estimate: [scores])
    estimate, bad1, bad2 = evaluation.empirical_kl_divergence(
        object(),
        object(),
        [(4, object())],
    )
    assert estimate == 0.0
    assert (bad1, bad2) == (3, 0)

    monkeypatch.setattr(
        evaluation,
        "seq_log_density",
        lambda data, estimate: [np.array([[-np.inf], [np.nan]])],
    )
    with pytest.raises(ValueError, match="no observations"):
        evaluation.empirical_kl_divergence(object(), object(), [(1, object())])


class _PointMassAtZero:
    def log_density(self, value):
        return 0.0 if value == 0 else float("-inf")

    def cdf(self, value):
        return 1.0 if value >= 0 else 0.0


def test_chi_square_impossible_observation_forces_rejection():
    statistic, dof, pvalue = evaluation.chi_square_test(
        [1],
        _PointMassAtZero(),
        lo=0,
        hi=1,
    )
    assert statistic == float("inf")
    assert dof == 0
    assert pvalue == 0.0


@pytest.mark.parametrize(
    ("data", "lo", "hi", "message"),
    [
        ([0.5], 0, 1, "observations"),
        ([True], 0, 1, "observations"),
        ([0], 1, 0, "lo"),
        ([0], 0.5, 1, "lo"),
    ],
)
def test_chi_square_rejects_invalid_discrete_experiments(data, lo, hi, message):
    with pytest.raises(ValueError, match=message):
        evaluation.chi_square_test(data, _PointMassAtZero(), lo=lo, hi=hi)


class _InconsistentDistribution:
    def log_density(self, value):
        return np.log(0.75)

    def cdf(self, value):
        return 0.5 if value >= 0 else 0.0


def test_chi_square_rejects_probabilities_inconsistent_with_the_cdf():
    with pytest.raises(ValueError, match="sum to one"):
        evaluation.chi_square_test([0], _InconsistentDistribution(), lo=0, hi=0)


def test_k_fold_split_conserves_every_row_and_balances_folds():
    assignments = evaluation.k_fold_split_index(11, 3, np.random.RandomState(4))
    assert assignments.shape == (11,)
    counts = np.bincount(assignments, minlength=3)
    assert counts.sum() == 11
    assert counts.min() > 0
    assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize(
    ("size", "folds"),
    [
        (0, 2),
        (4, 1),
        (4, 5),
        (4.5, 2),
        (4, True),
    ],
)
def test_k_fold_rejects_infeasible_or_inexact_sizes(size, folds):
    with pytest.raises(ValueError):
        evaluation.k_fold_split_index(size, folds, np.random.RandomState(0))


def test_proportional_split_uses_every_index_exactly_once():
    partitions = evaluation.partition_data_index(
        11,
        [0.0, 0.33, 0.67],
        np.random.RandomState(8),
    )
    assert [len(partition) for partition in partitions] == [0, 4, 7]
    flattened = np.concatenate(partitions)
    np.testing.assert_array_equal(np.sort(flattened), np.arange(11))

    data_parts = evaluation.partition_data(
        list("abcdefghijk"),
        [0.5, 0.5],
        np.random.RandomState(2),
    )
    assert sorted(item for part in data_parts for item in part) == list("abcdefghijk")


@pytest.mark.parametrize(
    "proportions",
    [
        [],
        [0.4, 0.4],
        [0.5, -0.5, 1.0],
        [0.5, np.nan, 0.5],
        [[0.5, 0.5]],
    ],
)
def test_proportional_split_rejects_invalid_or_nonexhaustive_proportions(proportions):
    with pytest.raises(ValueError):
        evaluation.partition_data_index(10, proportions, np.random.RandomState(0))
