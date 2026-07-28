"""Focused release-contract probes for forward uncertainty propagation."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.doe.propagate import propagate, unscented_transform


@pytest.mark.parametrize("method", ["montecarlo", "unscented"])
def test_propagation_rejects_materially_asymmetric_covariance(method):
    covariance = np.array([[2.0, 1.0], [0.0, 2.0]])
    with pytest.raises(ValueError, match="symmetric"):
        propagate(
            lambda x: x.sum(axis=1),
            np.zeros(2),
            covariance,
            method=method,
            n=8,
        )


def test_monte_carlo_uses_the_validated_roundoff_symmetric_covariance():
    asymmetric_roundoff = np.array([[2.0, 0.3 + 1e-12], [0.3, 1.0]])
    symmetric = 0.5 * (asymmetric_roundoff + asymmetric_roundoff.T)
    from_roundoff = propagate(
        lambda x: x,
        np.zeros(2),
        asymmetric_roundoff,
        method="montecarlo",
        n=8,
        seed=4,
    )
    from_symmetric = propagate(
        lambda x: x,
        np.zeros(2),
        symmetric,
        method="montecarlo",
        n=8,
        seed=4,
    )
    np.testing.assert_array_equal(from_roundoff["samples"], from_symmetric["samples"])


@pytest.mark.parametrize("count", [True, np.bool_(False), 2.0, 2.5])
def test_propagation_count_is_an_exact_non_boolean_integer(count):
    with pytest.raises(ValueError, match="n must be a positive"):
        propagate(lambda x: x[:, 0], np.zeros(1), np.eye(1), n=count)


@pytest.mark.parametrize("mean", [np.array(0.0), np.array([[0.0]]), np.array([])])
def test_propagation_mean_is_nonempty_and_exactly_one_dimensional(mean):
    with pytest.raises(ValueError, match="one-dimensional"):
        propagate(lambda x: x[:, 0], mean, np.eye(1), n=4)
    with pytest.raises(ValueError, match="one-dimensional"):
        unscented_transform(lambda x: x[:, 0], mean, np.eye(1))


@pytest.mark.parametrize("beta", [-10.0, -1e-12, True])
def test_unscented_beta_cannot_create_negative_uncertainty(beta):
    with pytest.raises(ValueError, match="beta"):
        unscented_transform(lambda x: x[:, 0] ** 2, np.zeros(1), np.eye(1), beta=beta)


def test_unscented_valid_nonlinear_covariance_remains_psd():
    mean, covariance = unscented_transform(
        lambda x: np.column_stack((x[:, 0], x[:, 0] ** 2)),
        np.zeros(1),
        np.eye(1),
        alpha=0.5,
        beta=2.0,
    )
    assert np.isfinite(mean).all()
    assert np.isfinite(covariance).all()
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= 0.0
