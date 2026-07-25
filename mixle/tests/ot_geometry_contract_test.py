"""Mass-splitting, symmetry, and validation contracts for Gaussian component transport."""

import numpy as np
import pytest

from mixle.experimental.ot_geometry import (
    bures_distance_sq,
    bures_wasserstein_params,
    mixture_barycenter_with_receipt,
)
from mixle.stats import GaussianDistribution, MixtureDistribution, MultivariateGaussianDistribution

pytestmark = pytest.mark.experimental


def _mixture(means, weights):
    return MixtureDistribution([GaussianDistribution(mean, 1.0) for mean in means], weights)


def test_transport_splits_component_mass_when_marginals_require_it():
    first = _mixture([0.0, 10.0], [0.5, 0.5])
    second = _mixture([0.0, 10.0], [0.25, 0.75])
    result, receipt = mixture_barycenter_with_receipt([first, second])
    assert receipt.positive_transport_atoms == 3
    assert receipt.output_components == 3
    assert receipt.max_marginal_error < 1e-12
    assert receipt.mass_error < 1e-12
    assert np.isclose(sum(result.w), 1.0)
    assert np.allclose([component.mu for component in result.components], [0.0, 5.0, 10.0])
    assert np.allclose(result.w, [0.25, 0.25, 0.5])


def test_input_and_component_order_do_not_privilege_a_first_mixture():
    first = _mixture([-3.0, 2.0], [0.4, 0.6])
    second = _mixture([-1.0, 5.0, 8.0], [0.2, 0.5, 0.3])
    first_reversed = _mixture([2.0, -3.0], [0.6, 0.4])
    second_reversed = _mixture([8.0, 5.0, -1.0], [0.3, 0.5, 0.2])

    forward, forward_receipt = mixture_barycenter_with_receipt([first, second], weights=[0.3, 0.7])
    reversed_, reversed_receipt = mixture_barycenter_with_receipt(
        [second_reversed, first_reversed],
        weights=[0.7, 0.3],
    )
    assert forward.to_json() == reversed_.to_json()
    assert forward_receipt.objective == reversed_receipt.objective


@pytest.mark.parametrize(
    "cov1,cov2,match",
    [
        ([[1.0, 2.0], [0.0, 1.0]], np.eye(2), "symmetric"),
        ([[1.0, 0.0], [0.0, -1.0]], np.eye(2), "positive definite"),
        (np.eye(2), np.eye(3), "shape"),
        ([[float("nan")]], [[1.0]], "finite"),
    ],
)
def test_invalid_covariances_fail_instead_of_discarding_complex_parts(cov1, cov2, match):
    with pytest.raises(ValueError, match=match):
        bures_distance_sq(cov1, cov2)


def test_mean_and_covariance_dimensions_must_agree():
    with pytest.raises(ValueError, match="shape"):
        bures_wasserstein_params([0.0, 1.0], np.eye(1), [0.0, 1.0], np.eye(2))


def test_invalid_component_and_barycentric_weights_fail_closed():
    class FakeMixture:
        components = [GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)]
        w = [1.0, -0.5]

    valid = _mixture([0.0], [1.0])
    with pytest.raises(ValueError, match="non-negative"):
        mixture_barycenter_with_receipt([FakeMixture(), valid])
    with pytest.raises(ValueError, match="strictly positive"):
        mixture_barycenter_with_receipt([valid, valid], weights=[1.0, 0.0])


def test_joint_support_guard_prevents_unbounded_cartesian_expansion():
    mixtures = [_mixture([0.0, 1.0, 2.0], [1 / 3, 1 / 3, 1 / 3]) for _ in range(4)]
    with pytest.raises(ValueError, match="exceeding"):
        mixture_barycenter_with_receipt(mixtures, max_joint_atoms=10)


def test_multivariate_components_preserve_full_mean_and_covariance():
    first = MixtureDistribution([MultivariateGaussianDistribution(np.zeros(2), np.eye(2))], [1.0])
    second = MixtureDistribution([MultivariateGaussianDistribution(np.full(2, 2.0), 4.0 * np.eye(2))], [1.0])
    result, receipt = mixture_barycenter_with_receipt([first, second])
    component = result.components[0]
    assert isinstance(component, MultivariateGaussianDistribution)
    assert np.allclose(component.mu, [1.0, 1.0])
    assert np.allclose(component.covar, 2.25 * np.eye(2))
    assert receipt.output_components == 1
