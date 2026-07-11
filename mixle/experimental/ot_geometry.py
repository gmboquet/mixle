"""P6 (experimental) -- optimal-transport geometry of model space.

Models interpolate badly in parameter space (rebasin symmetry) but well in *distribution* space.
For Gaussians the 2-Wasserstein distance and its barycenters are **closed form** (the Bures
metric), so merging Gaussian mixtures -- for ensemble compression, federated fusion, or L1
crossover -- can be done in transport space instead of by naive parameter averaging.

This module provides:

* :func:`bures_wasserstein` -- the exact ``W2`` between two Gaussians (scalar or full covariance);
* :func:`gaussian_barycenter` -- the Bures barycenter of weighted Gaussians (fixed-point);
* :func:`mixture_barycenter` -- a Wasserstein barycenter of Gaussian mixtures via component OT
  (Hungarian alignment on the pairwise Bures cost), returning a mixle ``MixtureDistribution``.

Honest scope (P6 kill criterion): whether the transport barycenter beats plain ensembling for a
given merge is an empirical question the test *measures* rather than assumes. The provable content
here is the metric (axioms + the closed-form values) and the barycenter's defining optimality.

Exploratory ``mixle.experimental`` code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import sqrtm

from mixle.stats import GaussianDistribution, MixtureDistribution


def _as_cov(cov: Any) -> np.ndarray:
    c = np.asarray(cov, dtype=float)
    return c.reshape(1, 1) if c.ndim == 0 else c


def _sqrtm_psd(m: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root (real), robust to tiny imaginary parts from ``sqrtm``."""
    if m.shape == (1, 1):
        return np.sqrt(np.maximum(m, 0.0))
    r = sqrtm(m)
    return np.real(r)


def bures_distance_sq(cov1: Any, cov2: Any) -> float:
    """Squared Bures distance between two covariance matrices (the covariance part of ``W2^2``).

    ``B^2 = tr(C1 + C2 - 2 (C1^{1/2} C2 C1^{1/2})^{1/2})``.
    """
    c1, c2 = _as_cov(cov1), _as_cov(cov2)
    s1 = _sqrtm_psd(c1)
    inner = _sqrtm_psd(s1 @ c2 @ s1)
    val = np.trace(c1 + c2 - 2.0 * inner)
    return float(max(val, 0.0))


def bures_wasserstein_params(mean1: Any, cov1: Any, mean2: Any, cov2: Any) -> float:
    """Exact ``W2`` between ``N(mean1, cov1)`` and ``N(mean2, cov2)`` from raw parameters."""
    m1 = np.atleast_1d(np.asarray(mean1, dtype=float))
    m2 = np.atleast_1d(np.asarray(mean2, dtype=float))
    mean_term = float(np.sum((m1 - m2) ** 2))
    return float(np.sqrt(mean_term + bures_distance_sq(cov1, cov2)))


def _gaussian_params(g: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(mean, cov)`` from a mixle Gaussian (1-D ``GaussianDistribution`` or MVN)."""
    if hasattr(g, "mu") and hasattr(g, "sigma2"):
        return np.atleast_1d(np.asarray(g.mu, dtype=float)), _as_cov(g.sigma2)
    if hasattr(g, "mean") and (hasattr(g, "covar") or hasattr(g, "cov")):
        cov = getattr(g, "covar", None)
        cov = cov if cov is not None else g.cov
        return np.atleast_1d(np.asarray(g.mean, dtype=float)), _as_cov(cov)
    raise TypeError(f"cannot read Gaussian parameters from {type(g).__name__}")


def bures_wasserstein(g1: Any, g2: Any) -> float:
    """Exact ``W2`` between two mixle Gaussian distributions."""
    m1, c1 = _gaussian_params(g1)
    m2, c2 = _gaussian_params(g2)
    return bures_wasserstein_params(m1, c1, m2, c2)


def gaussian_barycenter_params(
    means: list[Any], covs: list[Any], weights: Any = None, *, max_iter: int = 100, tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Bures barycenter ``(mean, cov)`` of weighted Gaussians via the fixed-point iteration.

    The barycenter mean is the weighted mean of means; its covariance solves
    ``S = sum_i w_i (S^{1/2} C_i S^{1/2})^{1/2}``, reached by iterating that map.
    """
    ms = [np.atleast_1d(np.asarray(m, dtype=float)) for m in means]
    cs = [_as_cov(c) for c in covs]
    n = len(ms)
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()

    mean = sum(wi * mi for wi, mi in zip(w, ms))
    s = sum(wi * ci for wi, ci in zip(w, cs))  # initialize at the covariance average
    for _ in range(max_iter):
        s_half = _sqrtm_psd(s)
        nxt = sum(wi * _sqrtm_psd(s_half @ ci @ s_half) for wi, ci in zip(w, cs))
        if np.max(np.abs(nxt - s)) < tol:
            s = nxt
            break
        s = nxt
    return mean, s


def gaussian_barycenter(gaussians: list[Any], weights: Any = None) -> GaussianDistribution:
    """Bures barycenter of mixle 1-D Gaussians, returned as a ``GaussianDistribution``."""
    params = [_gaussian_params(g) for g in gaussians]
    means = [p[0] for p in params]
    covs = [p[1] for p in params]
    mean, cov = gaussian_barycenter_params(means, covs, weights)
    return GaussianDistribution(float(mean[0]), float(cov[0, 0]))


def _components_and_weights(m: Any) -> tuple[list[Any], np.ndarray]:
    comps = list(m.components)
    w = np.asarray(getattr(m, "w", getattr(m, "weights", None)), dtype=float)
    return comps, w / w.sum()


def mixture_barycenter(mixtures: list[Any], weights: Any = None) -> MixtureDistribution:
    """Wasserstein barycenter of Gaussian mixtures via component OT (Hungarian alignment).

    All mixtures must have the same number of components. Components are aligned to the first
    mixture by minimum total Bures ``W2`` (an exact assignment), then each aligned group is
    Bures-barycentered and its mixing weights averaged.
    """
    from scipy.optimize import linear_sum_assignment

    per = [_components_and_weights(m) for m in mixtures]
    k = len(per)
    m0_comps, m0_w = per[0]
    r = len(m0_comps)
    for comps, _ in per:
        if len(comps) != r:
            raise ValueError("mixture_barycenter requires all mixtures to have equal component counts")

    mix_w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, dtype=float)
    mix_w = mix_w / mix_w.sum()

    # Align every mixture's components to mixture 0 via a Bures-cost assignment.
    aligned_comps: list[list[Any]] = [m0_comps]
    aligned_w: list[np.ndarray] = [m0_w]
    for comps, w in per[1:]:
        cost = np.array([[bures_wasserstein(a, b) for b in comps] for a in m0_comps])
        _, col = linear_sum_assignment(cost)
        aligned_comps.append([comps[j] for j in col])
        aligned_w.append(np.asarray([w[j] for j in col]))

    out_comps: list[GaussianDistribution] = []
    out_w = np.zeros(r)
    for slot in range(r):
        group = [aligned_comps[m][slot] for m in range(k)]
        out_comps.append(gaussian_barycenter(group, mix_w))
        out_w[slot] = float(np.sum([mix_w[m] * aligned_w[m][slot] for m in range(k)]))
    out_w = out_w / out_w.sum()
    return MixtureDistribution(out_comps, out_w.tolist())
