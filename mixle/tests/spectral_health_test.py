"""Descriptive, abstaining receipts for weight-matrix spectra."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mixle.experimental.spectral_health import (
    effective_rank,
    model_spectral_report,
    singular_values,
    spectral_health,
    stable_rank,
)


def _rand_orth(n, m, rng):
    q, _ = np.linalg.qr(rng.standard_normal((n, m)))
    return q[:, : min(n, m)]


def _with_spectrum(sv, *, n=128, m=64, seed=0):
    rng = np.random.default_rng(seed)
    return _rand_orth(n, min(n, m), rng) @ np.diag(sv) @ _rand_orth(m, min(n, m), rng).T


def _descriptive_regimes():
    idx = np.arange(1, 65, dtype=float)
    rng = np.random.default_rng(0)
    random_matrix = rng.standard_normal((128, 64))
    mild_tail = _with_spectrum(idx**-0.35, seed=1)
    heavy_tail = _with_spectrum(idx**-1.3, seed=2)
    return random_matrix, mild_tail, heavy_tail


def test_tail_exponent_describes_constructed_spectra_without_diagnosis():
    random_matrix, mild_tail, heavy_tail = _descriptive_regimes()
    receipts = [spectral_health(weight, min_tail_points=40) for weight in (random_matrix, mild_tail, heavy_tail)]
    assert receipts[1].alpha > receipts[2].alpha
    assert receipts[0].alpha is not None
    assert all(receipt.verdict is None for receipt in receipts)
    assert all("No calibrated diagnostic model" in receipt.diagnostic_reason for receipt in receipts)


def test_good_tail_fit_has_sample_size_ks_and_bootstrap_interval():
    structured = _with_spectrum(np.arange(1, 65, dtype=float) ** -0.35, seed=3)
    receipt = spectral_health(structured, min_tail_points=40, bootstrap_samples=100)
    assert receipt.tail_fit_status == "descriptive-fit"
    assert receipt.tail_fit_accepted
    assert receipt.tail_points >= 40
    assert receipt.ks_d is not None and receipt.ks_d <= 0.1
    assert receipt.alpha_ci_low < receipt.alpha < receipt.alpha_ci_high


def test_small_matrix_abstains_without_nonfinite_receipt_values():
    receipt = spectral_health(np.eye(3), min_tail_points=4)
    assert receipt.tail_fit_status == "insufficient-tail"
    assert not receipt.tail_fit_accepted
    assert receipt.alpha is None
    assert receipt.ks_d is None
    assert receipt.verdict is None
    json.dumps(receipt.as_dict(), allow_nan=False)


def test_stable_and_effective_rank_remain_descriptive():
    random_matrix, mild_tail, heavy_tail = _descriptive_regimes()
    stable = [stable_rank(singular_values(weight)) for weight in (random_matrix, mild_tail, heavy_tail)]
    effective = [effective_rank(singular_values(weight)) for weight in (random_matrix, mild_tail, heavy_tail)]
    assert stable[0] > stable[1] > stable[2]
    assert effective[0] > effective[2]


def test_receipt_is_data_free_deterministic_and_json_safe():
    structured = _with_spectrum(np.arange(1, 65, dtype=float) ** -0.35, seed=1)
    first = spectral_health(structured, min_tail_points=40, bootstrap_samples=100, seed=7)
    second = spectral_health(structured, min_tail_points=40, bootstrap_samples=100, seed=7)
    assert first.as_dict() == second.as_dict()
    encoded = json.dumps(first.as_dict(), allow_nan=False)
    assert "diagnostic_reason" in encoded


def test_model_report_preserves_names_and_abstains_per_layer():
    idx = np.arange(1, 65, dtype=float)
    report = model_spectral_report(
        {
            "layer.0": _with_spectrum(idx**-0.35, seed=1),
            "layer.1": _with_spectrum(idx**-1.3, seed=2),
        }
    )
    assert set(report) == {"layer.0", "layer.1"}
    assert all(receipt.verdict is None for receipt in report.values())


@pytest.mark.parametrize(
    "weight,exception,match",
    [
        (1.0, ValueError, "two dimensions"),
        (np.array([1.0, 2.0]), ValueError, "two dimensions"),
        (np.empty((0, 3)), ValueError, "empty"),
        (np.array([[1.0 + 1.0j]]), TypeError, "complex"),
        (np.array([[np.nan]]), ValueError, "finite"),
        (np.array([[np.inf]]), ValueError, "finite"),
    ],
)
def test_invalid_weight_inputs_fail_closed(weight, exception, match):
    with pytest.raises(exception, match=match):
        spectral_health(weight)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"min_tail_points": 0}, "min_tail_points"),
        ({"bootstrap_samples": 0}, "bootstrap_samples"),
        ({"max_ks": 0.0}, "max_ks"),
        ({"max_ks": float("nan")}, "max_ks"),
        ({"confidence": 1.0}, "confidence"),
        ({"seed": True}, "seed"),
    ],
)
def test_invalid_fit_controls_fail_closed(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        spectral_health(np.eye(8), **kwargs)
