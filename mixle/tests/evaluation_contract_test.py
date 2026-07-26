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
