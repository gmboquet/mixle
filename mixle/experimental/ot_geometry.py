"""Validated Bures geometry and mass-splitting component transport for Gaussian mixtures.

The Gaussian metric and barycenter use symmetric eigendecompositions, explicit covariance validation, and
a converged fixed-point iteration. Gaussian-mixture barycenters are solved as a symmetric multi-marginal
linear program over tuples of component atoms. Unlike one-to-one matching against a privileged first
mixture, this formulation permits mass splitting, supports unequal component counts, and is invariant to
input/component ordering after canonicalization.

The mixture construction is a Wasserstein barycenter in the discrete space of Gaussian components under
the squared Bures-Wasserstein ground metric. It is not a claim that component-restricted transport equals
unrestricted transport between the corresponding continuous mixture densities.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.optimize import linprog

from mixle.stats import GaussianDistribution, MixtureDistribution, MultivariateGaussianDistribution

__all__ = [
    "MixtureBarycenterReceipt",
    "bures_distance_sq",
    "bures_wasserstein_params",
    "bures_wasserstein",
    "gaussian_barycenter_params",
    "gaussian_barycenter",
    "mixture_barycenter_with_receipt",
    "mixture_barycenter",
]


@dataclass(frozen=True)
class MixtureBarycenterReceipt:
    """Optimization and marginal-conservation evidence for a component barycenter."""

    solver: str
    solver_status: int
    solver_message: str
    objective: float
    joint_atom_count: int
    positive_transport_atoms: int
    output_components: int
    max_marginal_error: float
    mass_error: float
    input_component_counts: tuple[int, ...]


def _exact_positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive exact integer.")
    return int(value)


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _validated_mean(mean: Any, name: str) -> np.ndarray:
    result = np.asarray(mean, dtype=float)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector.")
    return result


def _validated_covariance(cov: Any, name: str, *, dimension: int | None = None) -> np.ndarray:
    result = np.asarray(cov, dtype=float)
    if result.ndim == 0:
        result = result.reshape(1, 1)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a non-empty square matrix.")
    if dimension is not None and result.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    scale = max(1.0, float(np.linalg.norm(result, ord=2)))
    symmetry_tolerance = 1e-10 * scale
    if not np.allclose(result, result.T, rtol=0.0, atol=symmetry_tolerance):
        raise ValueError(f"{name} must be symmetric.")
    result = 0.5 * (result + result.T)
    eigenvalues = np.linalg.eigvalsh(result)
    if float(eigenvalues.min()) <= 0:
        raise ValueError(f"{name} must be positive definite.")
    return result


def _sqrtm_psd(matrix: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    """Real symmetric PSD square root without complex-part truncation."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite square matrix.")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    tolerance = 1e-10 * scale
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite.")
    root = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T
    return 0.5 * (root + root.T)


def _inverse_sqrtm_pd(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if float(eigenvalues.min()) <= 0:
        raise ValueError("barycenter iterate lost positive definiteness.")
    result = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _validated_weights(weights: Any, n: int, name: str, *, strictly_positive: bool) -> np.ndarray:
    if n <= 0:
        raise ValueError(f"{name} cannot describe an empty collection.")
    result = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    if result.shape != (n,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of shape {(n,)}.")
    if strictly_positive and np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    if not strictly_positive and np.any(result < 0):
        raise ValueError(f"{name} must be non-negative.")
    total = float(result.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive total mass.")
    return result / total


def bures_distance_sq(cov1: Any, cov2: Any) -> float:
    """Squared Bures distance between two positive-definite covariance matrices."""
    c1 = _validated_covariance(cov1, "cov1")
    c2 = _validated_covariance(cov2, "cov2", dimension=c1.shape[0])
    s1 = _sqrtm_psd(c1, name="cov1")
    inner = _sqrtm_psd(s1 @ c2 @ s1, name="Bures inner covariance")
    value = float(np.trace(c1 + c2 - 2.0 * inner))
    tolerance = 1e-9 * max(1.0, float(np.trace(c1 + c2)))
    if value < -tolerance or not math.isfinite(value):
        raise RuntimeError("Bures covariance distance produced an invalid negative/non-finite value.")
    return max(value, 0.0)


def bures_wasserstein_params(mean1: Any, cov1: Any, mean2: Any, cov2: Any) -> float:
    """Exact ``W2`` between two non-degenerate Gaussian distributions."""
    m1 = _validated_mean(mean1, "mean1")
    m2 = _validated_mean(mean2, "mean2")
    if m2.shape != m1.shape:
        raise ValueError("mean1 and mean2 must have the same dimension.")
    c1 = _validated_covariance(cov1, "cov1", dimension=m1.size)
    c2 = _validated_covariance(cov2, "cov2", dimension=m1.size)
    squared = float(np.sum((m1 - m2) ** 2)) + bures_distance_sq(c1, c2)
    return math.sqrt(max(squared, 0.0))


def _gaussian_params(gaussian: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract and validate ``(mean, covariance)`` from a supported Mixle Gaussian."""
    if hasattr(gaussian, "mu") and hasattr(gaussian, "sigma2"):
        mean = _validated_mean(gaussian.mu, "Gaussian mean")
        covariance = _validated_covariance(gaussian.sigma2, "Gaussian covariance", dimension=mean.size)
        return mean, covariance
    if hasattr(gaussian, "mu") and hasattr(gaussian, "covar"):
        mean = _validated_mean(gaussian.mu, "Gaussian mean")
        covariance = _validated_covariance(gaussian.covar, "Gaussian covariance", dimension=mean.size)
        return mean, covariance
    if hasattr(gaussian, "mean") and (hasattr(gaussian, "covar") or hasattr(gaussian, "cov")):
        mean = _validated_mean(gaussian.mean, "Gaussian mean")
        covariance_value = getattr(gaussian, "covar", None)
        covariance_value = covariance_value if covariance_value is not None else gaussian.cov
        covariance = _validated_covariance(covariance_value, "Gaussian covariance", dimension=mean.size)
        return mean, covariance
    raise TypeError(f"cannot read Gaussian parameters from {type(gaussian).__name__}")


def _make_gaussian(mean: np.ndarray, covariance: np.ndarray) -> Any:
    if mean.size == 1:
        return GaussianDistribution(float(mean[0]), float(covariance[0, 0]))
    return MultivariateGaussianDistribution(mean.copy(), covariance.copy())


def bures_wasserstein(g1: Any, g2: Any) -> float:
    """Exact ``W2`` between two supported Mixle Gaussian distributions."""
    m1, c1 = _gaussian_params(g1)
    m2, c2 = _gaussian_params(g2)
    return bures_wasserstein_params(m1, c1, m2, c2)


def gaussian_barycenter_params(
    means: list[Any],
    covs: list[Any],
    weights: Any = None,
    *,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Converged Bures barycenter parameters using the standard covariance fixed-point map."""
    if not isinstance(means, (list, tuple)) or not isinstance(covs, (list, tuple)) or not means:
        raise ValueError("means and covs must be non-empty sequences.")
    if len(means) != len(covs):
        raise ValueError("means and covs must have the same length.")
    validated_means = [_validated_mean(mean, f"means[{index}]") for index, mean in enumerate(means)]
    dimension = validated_means[0].size
    if any(mean.shape != (dimension,) for mean in validated_means):
        raise ValueError("all Gaussian means must have the same dimension.")
    validated_covariances = [
        _validated_covariance(cov, f"covs[{index}]", dimension=dimension) for index, cov in enumerate(covs)
    ]
    normalized_weights = _validated_weights(weights, len(validated_means), "weights", strictly_positive=True)
    max_iter = _exact_positive_int(max_iter, "max_iter")
    tol = _finite_positive(tol, "tol")

    mean = np.sum(
        np.stack([weight * value for weight, value in zip(normalized_weights, validated_means)]),
        axis=0,
    )
    covariance = np.sum(
        np.stack([weight * value for weight, value in zip(normalized_weights, validated_covariances)]),
        axis=0,
    )
    converged = False
    for _ in range(max_iter):
        covariance_half = _sqrtm_psd(covariance, name="barycenter iterate")
        covariance_inv_half = _inverse_sqrtm_pd(covariance)
        transport_sum = np.sum(
            np.stack(
                [
                    weight
                    * _sqrtm_psd(
                        covariance_half @ component_covariance @ covariance_half,
                        name="barycenter transported covariance",
                    )
                    for weight, component_covariance in zip(normalized_weights, validated_covariances)
                ]
            ),
            axis=0,
        )
        nxt = covariance_inv_half @ transport_sum @ transport_sum @ covariance_inv_half
        nxt = 0.5 * (nxt + nxt.T)
        relative_error = float(np.linalg.norm(nxt - covariance, ord="fro")) / max(
            float(np.linalg.norm(covariance, ord="fro")),
            1e-15,
        )
        covariance = nxt
        if relative_error <= tol:
            converged = True
            break
    if not converged:
        raise RuntimeError(f"Gaussian barycenter covariance did not converge in {max_iter} iterations.")
    _validated_covariance(covariance, "computed barycenter covariance", dimension=dimension)
    return mean, covariance


def gaussian_barycenter(gaussians: list[Any], weights: Any = None) -> Any:
    """Bures barycenter of supported Mixle Gaussians."""
    if not isinstance(gaussians, (list, tuple)) or not gaussians:
        raise ValueError("gaussians must be a non-empty sequence.")
    parameters = [_gaussian_params(gaussian) for gaussian in gaussians]
    mean, covariance = gaussian_barycenter_params(
        [parameters_i[0] for parameters_i in parameters],
        [parameters_i[1] for parameters_i in parameters],
        weights,
    )
    return _make_gaussian(mean, covariance)


def _components_and_weights(mixture: Any) -> tuple[list[Any], np.ndarray]:
    if not hasattr(mixture, "components"):
        raise TypeError("every input must expose Gaussian mixture components.")
    components = list(mixture.components)
    if not components:
        raise ValueError("mixtures must contain at least one component.")
    raw_weights = getattr(mixture, "w", getattr(mixture, "weights", None))
    weights = _validated_weights(raw_weights, len(components), "component weights", strictly_positive=False)

    retained = [(component, float(weight)) for component, weight in zip(components, weights) if weight > 0]
    if not retained:
        raise ValueError("mixtures must contain positive component mass.")
    for component, _ in retained:
        _gaussian_params(component)
    retained.sort(key=lambda item: _component_key(item[0], item[1]))
    return [item[0] for item in retained], np.asarray([item[1] for item in retained])


def _component_key(component: Any, weight: float) -> tuple[Any, ...]:
    mean, covariance = _gaussian_params(component)
    return (*mean.tolist(), *covariance.reshape(-1).tolist(), weight)


def _mixture_key(record: tuple[list[Any], np.ndarray], barycentric_weight: float) -> tuple[Any, ...]:
    components, weights = record
    flattened: list[float] = [barycentric_weight]
    for component, weight in zip(components, weights):
        flattened.extend(_component_key(component, float(weight)))
    return tuple(flattened)


def _tuple_barycenter(
    components: tuple[Any, ...],
    barycentric_weights: np.ndarray,
) -> tuple[Any, float]:
    parameters = [_gaussian_params(component) for component in components]
    mean, covariance = gaussian_barycenter_params(
        [parameter[0] for parameter in parameters],
        [parameter[1] for parameter in parameters],
        barycentric_weights,
    )
    objective = sum(
        float(weight) * bures_wasserstein_params(mean, covariance, parameter[0], parameter[1]) ** 2
        for weight, parameter in zip(barycentric_weights, parameters)
    )
    return _make_gaussian(mean, covariance), objective


def mixture_barycenter_with_receipt(
    mixtures: list[Any],
    weights: Any = None,
    *,
    max_joint_atoms: int = 100_000,
    mass_tolerance: float = 1e-9,
) -> tuple[MixtureDistribution, MixtureBarycenterReceipt]:
    """Solve the symmetric mass-splitting Gaussian-component Wasserstein barycenter."""
    if not isinstance(mixtures, (list, tuple)) or not mixtures:
        raise ValueError("mixtures must be a non-empty sequence.")
    max_joint_atoms = _exact_positive_int(max_joint_atoms, "max_joint_atoms")
    mass_tolerance = _finite_positive(mass_tolerance, "mass_tolerance")
    barycentric_weights = _validated_weights(weights, len(mixtures), "mixture weights", strictly_positive=True)

    records = [_components_and_weights(mixture) for mixture in mixtures]
    dimensions = {_gaussian_params(component)[0].size for components, _ in records for component in components}
    if len(dimensions) != 1:
        raise ValueError("all Gaussian mixture components must have the same dimension.")

    ordered = sorted(
        zip(records, barycentric_weights),
        key=lambda item: _mixture_key(item[0], float(item[1])),
    )
    records = [item[0] for item in ordered]
    barycentric_weights = np.asarray([item[1] for item in ordered])
    component_counts = tuple(len(components) for components, _ in records)
    joint_atom_count = math.prod(component_counts)
    if joint_atom_count > max_joint_atoms:
        raise ValueError(
            f"multi-marginal support has {joint_atom_count} atoms, exceeding max_joint_atoms={max_joint_atoms}."
        )

    joint_indices = list(itertools.product(*(range(count) for count in component_counts)))
    joint_components: list[Any] = []
    costs = np.empty(joint_atom_count)
    for atom, indices in enumerate(joint_indices):
        component_tuple = tuple(
            records[mixture_index][0][component_index] for mixture_index, component_index in enumerate(indices)
        )
        joint_component, costs[atom] = _tuple_barycenter(component_tuple, barycentric_weights)
        joint_components.append(joint_component)

    n_constraints = sum(component_counts)
    constraint_matrix = np.zeros((n_constraints, joint_atom_count))
    marginal_mass = np.empty(n_constraints)
    row = 0
    for mixture_index, ((_, component_weights), component_count) in enumerate(zip(records, component_counts)):
        for component_index in range(component_count):
            constraint_matrix[row] = [
                1.0 if indices[mixture_index] == component_index else 0.0 for indices in joint_indices
            ]
            marginal_mass[row] = component_weights[component_index]
            row += 1

    solution = linprog(
        costs,
        A_eq=constraint_matrix,
        b_eq=marginal_mass,
        bounds=(0.0, None),
        method="highs",
        # The conservation check below demands mass_tolerance (default 1e-9), but HiGHS's DEFAULT
        # primal feasibility tolerance is 1e-7 -- the guard was stricter than the solve was asked
        # to be, and whether it passed depended on the LP instance (a solution at 1.27e-8 marginal
        # error, well inside the solver's own contract, was refused on one platform). Ask the
        # solver for a tolerance a decade tighter than the guard, floored at HiGHS's 1e-10 minimum.
        options={
            "primal_feasibility_tolerance": max(1.0e-10, mass_tolerance / 10.0),
            "dual_feasibility_tolerance": max(1.0e-10, mass_tolerance / 10.0),
        },
    )
    if not solution.success or solution.x is None or not np.all(np.isfinite(solution.x)):
        raise RuntimeError(f"multi-marginal barycenter LP failed: {solution.message}")
    transport = np.clip(solution.x, 0.0, None)
    max_marginal_error = float(np.max(np.abs(constraint_matrix @ transport - marginal_mass)))
    mass_error = abs(float(transport.sum()) - 1.0)
    if max_marginal_error > mass_tolerance or mass_error > mass_tolerance:
        raise RuntimeError(
            "multi-marginal barycenter LP violated marginal conservation: "
            f"max_marginal_error={max_marginal_error}, mass_error={mass_error}."
        )

    positive = np.flatnonzero(transport > mass_tolerance)
    if positive.size == 0:
        raise RuntimeError("multi-marginal barycenter LP returned no positive transport mass.")
    output = [(joint_components[index], float(transport[index])) for index in positive]
    output.sort(key=lambda item: _component_key(item[0], item[1]))
    output_components = [item[0] for item in output]
    output_weights = np.asarray([item[1] for item in output])
    output_weights /= output_weights.sum()
    distribution = MixtureDistribution(output_components, output_weights.tolist())
    receipt = MixtureBarycenterReceipt(
        solver="scipy.optimize.linprog(highs)",
        solver_status=int(solution.status),
        solver_message=str(solution.message),
        objective=float(solution.fun),
        joint_atom_count=joint_atom_count,
        positive_transport_atoms=int(positive.size),
        output_components=len(output_components),
        max_marginal_error=max_marginal_error,
        mass_error=mass_error,
        input_component_counts=component_counts,
    )
    return distribution, receipt


def mixture_barycenter(
    mixtures: list[Any],
    weights: Any = None,
    *,
    max_joint_atoms: int = 100_000,
    mass_tolerance: float = 1e-9,
) -> MixtureDistribution:
    """Return the mass-splitting component-Wasserstein barycenter.

    Use :func:`mixture_barycenter_with_receipt` when solver and conservation evidence is required.
    """
    distribution, _ = mixture_barycenter_with_receipt(
        mixtures,
        weights,
        max_joint_atoms=max_joint_atoms,
        mass_tolerance=mass_tolerance,
    )
    return distribution
