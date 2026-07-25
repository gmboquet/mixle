"""P3 (experimental) -- conjugate-computation variational inference (CVI), the theory under D-track.

Khan & Lin's conjugate-computation VI updates a model in **natural-parameter space**: it takes a
natural-gradient step of the ELBO, which for a conjugate exponential-family part reduces to the
exact conjugate (Bayes / EM) update, and for a non-conjugate part becomes an ordinary gradient
step -- one update rule for the whole model. This is the principle the D-track's closed-form /
gradient split should be *derived* from rather than scheduled heuristically.

The self-contained, provable core (this module) is the first half of the card's experiment: **a
single natural-gradient CVI step with unit step size reproduces the exact conjugate update** for
exponential-family leaves. We show it for three conjugate pairs -- Normal-Normal, Beta-Bernoulli,
Gamma-Poisson -- in their natural parameterization, plus that streaming the data in any chunking
gives the identical posterior (natural parameters are additive), and that a damped step
(``rho < 1``) converges to the same fixed point over iterations. The mixed conjugate+GradLeaf-flow
model and the iterations-to-F comparison are the torch follow-up.

Exploratory ``mixle.experimental`` code (P3 card).
"""

from __future__ import annotations

from typing import Any

import numpy as np

_FAMILIES = frozenset({"normal_normal", "beta_bernoulli", "gamma_poisson"})


def _validated_family(family: str) -> str:
    if family not in _FAMILIES:
        raise ValueError(f"unknown conjugate family {family!r}")
    return family


def _validated_obs_var(obs_var: Any) -> float:
    if (
        isinstance(obs_var, (bool, np.bool_))
        or not np.isscalar(obs_var)
        or not np.isfinite(obs_var)
        or float(obs_var) <= 0.0
    ):
        raise ValueError("obs_var must be a finite positive scalar.")
    return float(obs_var)


def _validated_prior(family: str, prior: Any) -> tuple[float, float]:
    family = _validated_family(family)
    try:
        values = tuple(prior)
    except TypeError as exc:
        raise ValueError("prior must contain exactly two finite scalar parameters.") from exc
    if len(values) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) for value in values
    ):
        raise ValueError("prior must contain exactly two finite scalar parameters.")
    a, b = float(values[0]), float(values[1])
    if family == "normal_normal":
        if b <= 0.0:
            raise ValueError("Normal prior precision must be positive.")
    elif a <= 0.0 or b <= 0.0:
        raise ValueError("Beta/Gamma prior parameters must be positive.")
    return a, b


def _validated_data(family: str, data: Any) -> np.ndarray:
    family = _validated_family(family)
    try:
        values = np.asarray(list(data), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("data must be a one-dimensional iterable of finite observations.") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("data must be a one-dimensional iterable of finite observations.")
    if family == "beta_bernoulli" and not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("Bernoulli observations must be exactly 0 or 1.")
    if family == "gamma_poisson" and not np.all((values >= 0.0) & (values == np.floor(values))):
        raise ValueError("Poisson observations must be non-negative integers.")
    return values


def _validated_rho(rho: Any) -> float:
    if isinstance(rho, (bool, np.bool_)) or not np.isscalar(rho) or not np.isfinite(rho) or not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be finite and satisfy 0 < rho <= 1.")
    return float(rho)


def _sufficient_stat_sum(family: str, data: np.ndarray, *, obs_var: float) -> np.ndarray:
    """Sum over the data of each observation's contribution to the posterior natural parameter."""
    family = _validated_family(family)
    obs_var = _validated_obs_var(obs_var)
    x = _validated_data(family, data)
    n = len(x)
    if family == "normal_normal":  # unknown mean, known obs variance obs_var
        return np.array([x.sum() / obs_var, -n / (2.0 * obs_var)])
    if family == "beta_bernoulli":  # x in {0,1}
        k = float(x.sum())
        return np.array([k, n - k])
    if family == "gamma_poisson":  # x nonneg counts; natural stats of Poisson in (log lam, lam) are (x, -1)
        return np.array([x.sum(), -float(n)])
    raise AssertionError("validated family was not handled")


def _prior_natural(family: str, prior: tuple[float, float]) -> np.ndarray:
    """Natural parameter of the prior for each family."""
    family = _validated_family(family)
    a, b = _validated_prior(family, prior)
    if family == "normal_normal":  # prior N(m0, 1/lambda0); prior = (m0, lambda0)
        m0, lam0 = a, b
        return np.array([lam0 * m0, -lam0 / 2.0])
    if family == "beta_bernoulli":  # prior Beta(a0, b0); natural = (a0 - 1, b0 - 1)
        return np.array([a - 1.0, b - 1.0])
    if family == "gamma_poisson":  # prior Gamma(shape a0, rate b0); natural = (a0 - 1, -b0)
        return np.array([a - 1.0, -b])
    raise AssertionError("validated family was not handled")


def _natural_to_params(family: str, eta: np.ndarray) -> tuple[float, float]:
    """Invert the natural parameter back to the family's usual parameters."""
    family = _validated_family(family)
    eta = np.asarray(eta, dtype=float)
    if eta.shape != (2,) or not np.all(np.isfinite(eta)):
        raise ValueError("natural parameters must be a finite length-two vector.")
    if family == "normal_normal":  # eta = (lambda m, -lambda/2)
        lam = -2.0 * eta[1]
        parameters = (float(eta[0] / lam), float(lam))  # (mean, precision)
    elif family == "beta_bernoulli":
        parameters = (float(eta[0] + 1.0), float(eta[1] + 1.0))  # (a, b)
    else:
        parameters = (float(eta[0] + 1.0), float(-eta[1]))  # (shape, rate)
    if not np.all(np.isfinite(parameters)) or parameters[1] <= 0.0:
        raise ValueError("natural parameters do not define a proper posterior.")
    if family != "normal_normal" and parameters[0] <= 0.0:
        raise ValueError("natural parameters do not define a proper posterior.")
    return parameters


def cvi_step(
    family: str, prior: tuple[float, float], data: Any, *, rho: float = 1.0, obs_var: float = 1.0
) -> tuple[float, float]:
    """One natural-gradient CVI step from ``prior`` given ``data``; returns the updated parameters.

    The step is ``eta <- eta_prior + rho * (sum of data sufficient statistics)``. With ``rho = 1``
    this is exactly the conjugate posterior update (the EM M-step for a conjugate leaf).
    """
    family = _validated_family(family)
    prior = _validated_prior(family, prior)
    rho = _validated_rho(rho)
    obs_var = _validated_obs_var(obs_var)
    values = _validated_data(family, data)
    eta = _prior_natural(family, prior) + rho * _sufficient_stat_sum(family, values, obs_var=obs_var)
    return _natural_to_params(family, eta)


def conjugate_posterior(
    family: str, prior: tuple[float, float], data: Any, *, obs_var: float = 1.0
) -> tuple[float, float]:
    """The closed-form conjugate posterior parameters (the reference the CVI step must reproduce)."""
    family = _validated_family(family)
    prior = _validated_prior(family, prior)
    obs_var = _validated_obs_var(obs_var)
    x = _validated_data(family, data)
    n = len(x)
    if family == "normal_normal":
        m0, lam0 = prior
        lam_n = lam0 + n / obs_var
        m_n = (lam0 * m0 + x.sum() / obs_var) / lam_n
        return float(m_n), float(lam_n)
    if family == "beta_bernoulli":
        a0, b0 = prior
        k = float(x.sum())
        return float(a0 + k), float(b0 + n - k)
    if family == "gamma_poisson":
        a0, b0 = prior
        return float(a0 + x.sum()), float(b0 + n)
    raise AssertionError("validated family was not handled")


def damped_to_convergence(
    family: str, prior: tuple[float, float], data: Any, *, rho: float = 0.3, iters: int = 200, obs_var: float = 1.0
) -> tuple[float, float]:
    """Repeated damped natural-gradient steps toward the posterior (rho<1 converges to the same fixed point).

    Each step moves the natural parameter a fraction ``rho`` of the way from the current value to the
    full posterior natural parameter -- a convex combination that converges to the posterior.
    """
    family = _validated_family(family)
    prior = _validated_prior(family, prior)
    rho = _validated_rho(rho)
    obs_var = _validated_obs_var(obs_var)
    if isinstance(iters, (bool, np.bool_)) or not isinstance(iters, (int, np.integer)) or int(iters) <= 0:
        raise ValueError("iters must be a positive integer.")
    values = _validated_data(family, data)
    eta_post = _prior_natural(family, prior) + _sufficient_stat_sum(family, values, obs_var=obs_var)
    eta = _prior_natural(family, prior)
    for _ in range(int(iters)):
        eta = (1.0 - rho) * eta + rho * eta_post
    return _natural_to_params(family, eta)
