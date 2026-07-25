"""Fail-closed domain tests for experimental certified density bounds."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mixle.experimental.certified_bounds import DensityBoundReceipt, certified_density_bounds


def _gaussian(mu: float = 0.0, sigma2: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(mu=mu, sigma2=sigma2)


def _mixture(weights: object, components: list[object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(components=components or [_gaussian(), _gaussian(2.0)], w=weights)


def test_bound_receipt_records_the_validated_proof_domain() -> None:
    receipt = certified_density_bounds(_mixture([2.0, 1.0]), -3.0, 4.0)

    assert isinstance(receipt, DensityBoundReceipt)
    assert tuple(receipt) == (receipt.lower, receipt.upper)
    assert receipt.interval == (-3.0, 4.0)
    assert receipt.component_means == (0.0, 2.0)
    assert receipt.component_variances == (1.0, 1.0)
    np.testing.assert_allclose(receipt.normalized_weights, (2.0 / 3.0, 1.0 / 3.0))
    assert receipt.rule == "gaussian-weighted-interval-sum/v1"


@pytest.mark.parametrize("sigma2", [0.0, -1.0, np.inf, np.nan])
def test_certification_rejects_invalid_gaussian_variance(sigma2: float) -> None:
    with pytest.raises(ValueError):
        certified_density_bounds(_gaussian(sigma2=sigma2), -1.0, 1.0)


@pytest.mark.parametrize(
    "weights",
    [
        [1.0],
        [1.0, 2.0, 3.0],
        [-1.0, 2.0],
        [0.0, 0.0],
        [np.nan, 1.0],
        [np.inf, 1.0],
    ],
)
def test_certification_rejects_invalid_mixture_weights(weights: list[float]) -> None:
    with pytest.raises(ValueError):
        certified_density_bounds(_mixture(weights), -1.0, 1.0)


@pytest.mark.parametrize("lo, hi", [(2.0, 1.0), (np.nan, 1.0), (0.0, np.inf)])
def test_certification_rejects_invalid_intervals(lo: float, hi: float) -> None:
    with pytest.raises(ValueError):
        certified_density_bounds(_gaussian(), lo, hi)
