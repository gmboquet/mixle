"""Validity, numerical, and input contracts for anytime e-process evidence."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.experimental.e_process import (
    EProcess,
    MeanShiftDetector,
    PredictableLikelihoodRatio,
    TestMartingaleValidity,
    normal_mixture_eprocess,
    normal_mixture_log_e,
)


def _validity() -> TestMartingaleValidity:
    return TestMartingaleValidity(
        null_hypothesis="conditional observations follow p_t",
        alternative_hypothesis="conditional observations follow q_t",
        assumptions=("p_t and q_t depend only on past observations",),
        normalization_evidence="q_t is a normalized conditional density for every history",
    )


def test_generic_process_requires_a_typed_test_martingale() -> None:
    with pytest.raises(TypeError):
        EProcess(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TestMartingaleValidity("null", "alternative", (), "normalized")


def test_nonfinite_conditional_log_ratios_fail_closed() -> None:
    process = EProcess(
        PredictableLikelihoodRatio(
            log_alternative=lambda _history, _x: np.inf,
            log_null=lambda _history, _x: 0.0,
            validity=_validity(),
        )
    )
    with pytest.raises(ValueError):
        process.update(1.0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan, np.inf])
def test_invalid_alpha_cannot_produce_an_anytime_decision(alpha: float) -> None:
    detector_args = {"mu0": 0.0, "sigma": 1.0, "tau": 1.0}
    with pytest.raises(ValueError):
        MeanShiftDetector(**detector_args, alpha=alpha)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu0": np.nan, "sigma": 1.0, "tau": 1.0},
        {"mu0": 0.0, "sigma": 0.0, "tau": 1.0},
        {"mu0": 0.0, "sigma": 1.0, "tau": 0.0},
        {"mu0": 0.0, "sigma": np.inf, "tau": 1.0},
    ],
)
def test_invalid_normal_mixture_parameters_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        normal_mixture_eprocess([0.0, 1.0], **kwargs)


@pytest.mark.parametrize("t", [-1, 0.5, np.nan])
def test_normal_mixture_time_must_be_a_nonnegative_integer(t: float) -> None:
    with pytest.raises(ValueError):
        normal_mixture_log_e(0.0, t, sigma=1.0, tau=1.0)  # type: ignore[arg-type]


def test_drift_report_discloses_overflow_without_using_it_for_decisions() -> None:
    report = MeanShiftDetector(mu0=0.0, sigma=1.0, tau=1.0, alpha=0.05).scan([1.0e6])
    assert report.detected
    assert report.numerical_status == "overflow"
    assert np.isinf(report.peak_e_value)
    assert np.isfinite(report.peak_log_e_value)
    assert report.assumptions
