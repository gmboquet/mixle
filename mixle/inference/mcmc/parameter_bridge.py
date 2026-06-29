"""High-level parameter-posterior sampling for ``mixle.stats`` families.

Given a prototype distribution that fixes the model family, a dataset, and a
prior over parameters, this samples
``p(theta | data) proportional to exp(sum_i log p(x_i | theta)) * prior(theta)``
by running Metropolis-Hastings or Hamiltonian Monte Carlo in an unconstrained
reparameterization (log for positive scales, stick-breaking for probability
simplices) and mapping the retained samples back to parameter space (or to
rebuilt distribution objects).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .proposals import Proposal, RandomWalkProposal
from .samplers import LogTarget, MCMCResult, hamiltonian_monte_carlo, metropolis_hastings, nuts

# ---------------------------------------------------------------------------
# Parameter posterior: flatten/unflatten bridge + samplers
# ---------------------------------------------------------------------------
#
# Sampling p(theta | data) needs three things for each model family:
#   * a way to read theta off a prototype distribution as a flat vector,
#   * an unconstrained reparameterization phi = T(theta) so random-walk / HMC
#     proposals never leave the parameter domain (log for positive scales,
#     stick-breaking for probability simplices), with the log |det dtheta/dphi|
#     Jacobian so the sampled density is the correct posterior in theta-space,
#   * a constructor that rebuilds a fresh distribution from a proposed theta so
#     the data log-likelihood can be evaluated with seq_encode/seq_log_density.
#
# ``ParameterBridge`` packages those operations.  ``build_parameter_bridge``
# resolves the bridge from the prototype's type; unsupported shapes raise a
# clear NotImplementedError naming the type.


@dataclass(frozen=True)
class ParameterBridge:
    """Map between a distribution's parameters and an unconstrained vector.

    Attributes:
        dim: Length of the unconstrained vector ``phi``.
        to_unconstrained: ``theta -> phi`` for the prototype's parameters.
        from_unconstrained: ``phi -> theta`` (parameter-space value).
        log_abs_det_jacobian: ``phi -> log|det dtheta/dphi|`` so a flat prior in
            theta-space maps to the right density in phi-space.
        build: ``theta -> distribution`` rebuilds a model from parameters.
        param_names: Human-readable names for each theta block (diagnostics).
        initial_theta: The prototype's parameters in theta-space (chain start).
    """

    dim: int
    to_unconstrained: Callable[[Any], np.ndarray]
    from_unconstrained: Callable[[np.ndarray], Any]
    log_abs_det_jacobian: Callable[[np.ndarray], float]
    build: Callable[[Any], Any]
    param_names: tuple[str, ...]
    initial_theta: Any = None


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _stick_breaking_forward(phi: np.ndarray) -> np.ndarray:
    """Map ``k-1`` unconstrained reals to a length-``k`` probability simplex.

    Uses the standard logistic stick-breaking transform (as in Stan).  The
    returned vector is strictly positive and sums to one.
    """
    k = phi.shape[0] + 1
    p = np.empty(k, dtype=float)
    remaining = 1.0
    for i in range(k - 1):
        # offset keeps a uniform-ish base measure but is irrelevant to validity
        z = 1.0 / (1.0 + np.exp(-(phi[i] - np.log(float(k - 1 - i)))))
        p[i] = remaining * z
        remaining = remaining - p[i]
    p[k - 1] = remaining
    return p


def _stick_breaking_log_det(phi: np.ndarray) -> float:
    """Return ``log|det d p / d phi|`` for the stick-breaking transform.

    Only the first ``k-1`` coordinates of ``p`` are free; the Jacobian of the
    map from ``phi`` to those coordinates is lower triangular, so the log abs
    determinant is the sum of the diagonal log terms.
    """
    k = phi.shape[0] + 1
    remaining = 1.0
    total = 0.0
    for i in range(k - 1):
        z = 1.0 / (1.0 + np.exp(-(phi[i] - np.log(float(k - 1 - i)))))
        # d p_i / d phi_i = remaining * z * (1 - z); off-diagonal terms below it
        total += np.log(remaining) + np.log(z) + np.log1p(-z)
        remaining = remaining * (1.0 - z)
    return float(total)


def _seq_log_density_sum(dist: Any, encoded: Any) -> float:
    """Return ``sum_i log p(x_i | dist)`` for a stats distribution."""
    return float(np.sum(dist.seq_log_density(encoded)))


def _encode_data(prototype: Any, data: Any) -> tuple[Any, Callable[[Any], Any]]:
    """Encode ``data`` once and return (encoded, encode_fn) for the family.

    ``stats`` distributions encode through ``dist_to_encoder().seq_encode``.
    The returned ``encode_fn`` lets re-encoding happen if a rebuilt distribution
    needs it (categorical encodings depend only on the data, so the cached
    encoding is reused across proposals).
    """
    encoder = prototype.dist_to_encoder()
    return encoder.seq_encode(data), encoder.seq_encode


def _make_builder(prototype: Any, ctor_kwargs: dict[str, Any]) -> Callable[..., Any]:
    cls = type(prototype)

    def build(*args: Any) -> Any:
        return cls(*args, **ctor_kwargs)

    return build


def build_parameter_bridge(prototype: Any) -> ParameterBridge:
    """Build a :class:`ParameterBridge` for a prototype distribution.

    Supported ``mixle.stats`` families:

    * Gaussian ``(mu, sigma2)`` -> ``(mu, log sigma2)``
    * Gamma ``(k, theta)`` -> ``(log k, log theta)``
    * Exponential ``beta``/``lam`` (positive scalar) -> ``log``
    * Poisson ``lam`` -> ``log lam``
    * Bernoulli ``p`` -> ``logit p``
    * Beta ``(a, b)`` -> ``(log a, log b)``
    * Categorical probability map -> stick-breaking over the simplex

    Raises:
        NotImplementedError: if the family or a parameter shape is unsupported.
    """
    cls_name = type(prototype).__name__
    name = getattr(prototype, "name", None)
    keys = getattr(prototype, "keys", None)

    # carry name (and keys when present) so the rebuilt model matches the
    # prototype but never the prior/posterior bookkeeping that would change the
    # likelihood surface.
    kw: dict[str, Any] = {}
    if name is not None and "name" in _ctor_param_names(prototype):
        kw["name"] = name

    if cls_name == "GaussianDistribution":

        def to_u(theta):
            mu, sigma2 = theta
            return np.asarray([float(mu), float(np.log(sigma2))], dtype=float)

        def from_u(phi):
            return (float(phi[0]), float(np.exp(phi[1])))

        build = _make_builder(prototype, kw)
        return ParameterBridge(
            dim=2,
            to_unconstrained=to_u,
            from_unconstrained=from_u,
            log_abs_det_jacobian=lambda phi: float(phi[1]),  # d sigma2 / d log sigma2
            build=lambda theta: build(theta[0], theta[1]),
            param_names=("mu", "sigma2"),
            initial_theta=(float(prototype.mu), float(prototype.sigma2)),
        )

    if cls_name in ("GammaDistribution", "BetaDistribution"):
        a0, a1 = ("k", "theta") if cls_name == "GammaDistribution" else ("a", "b")
        build = _make_builder(prototype, kw)

        def to_u(theta):
            return np.asarray([float(np.log(theta[0])), float(np.log(theta[1]))], dtype=float)

        def from_u(phi):
            return (float(np.exp(phi[0])), float(np.exp(phi[1])))

        return ParameterBridge(
            dim=2,
            to_unconstrained=to_u,
            from_unconstrained=from_u,
            log_abs_det_jacobian=lambda phi: float(phi[0] + phi[1]),
            build=lambda theta: build(theta[0], theta[1]),
            param_names=(a0, a1),
            initial_theta=(float(getattr(prototype, a0)), float(getattr(prototype, a1))),
        )

    if cls_name == "ExponentialDistribution":
        positive_attr = "beta" if hasattr(prototype, "beta") else "lam"
        build = _make_builder(prototype, kw)
        return ParameterBridge(
            dim=1,
            to_unconstrained=lambda theta: np.asarray([float(np.log(theta))], dtype=float),
            from_unconstrained=lambda phi: float(np.exp(phi[0])),
            log_abs_det_jacobian=lambda phi: float(phi[0]),
            build=lambda theta: build(theta),
            param_names=(positive_attr,),
            initial_theta=float(getattr(prototype, positive_attr)),
        )

    if cls_name == "PoissonDistribution":
        build = _make_builder(prototype, kw)
        return ParameterBridge(
            dim=1,
            to_unconstrained=lambda theta: np.asarray([float(np.log(theta))], dtype=float),
            from_unconstrained=lambda phi: float(np.exp(phi[0])),
            log_abs_det_jacobian=lambda phi: float(phi[0]),
            build=lambda theta: build(theta),
            param_names=("lam",),
            initial_theta=float(prototype.lam),
        )

    if cls_name == "BernoulliDistribution":
        build = _make_builder(prototype, kw)

        def from_u(phi):
            return float(1.0 / (1.0 + np.exp(-phi[0])))

        return ParameterBridge(
            dim=1,
            to_unconstrained=lambda theta: np.asarray([float(np.log(theta) - np.log1p(-theta))], dtype=float),
            from_unconstrained=from_u,
            # d p / d logit = p (1 - p); log = -softplus(-phi) - softplus(phi)
            log_abs_det_jacobian=lambda phi: float(-_softplus(-phi[0]) - _softplus(phi[0])),
            build=lambda theta: build(theta),
            param_names=("p",),
            initial_theta=float(prototype.p),
        )

    if cls_name == "CategoricalDistribution":
        prob_map = (
            prototype.get_parameters()
            if hasattr(prototype, "get_parameters")
            else getattr(prototype, "pmap", getattr(prototype, "prob_map", None))
        )
        if not isinstance(prob_map, Mapping):
            raise NotImplementedError("CategoricalDistribution parameter posterior requires a probability map.")
        labels = tuple(prob_map.keys())
        k = len(labels)
        if k < 2:
            raise NotImplementedError("CategoricalDistribution parameter posterior needs at least two categories.")
        default_value = float(getattr(prototype, "default_value", 0.0))
        # build keyword set: stats CategoricalDistribution takes default_value/name
        cat_kw = dict(kw)
        if "default_value" in _ctor_param_names(prototype):
            cat_kw["default_value"] = default_value
        build = _make_builder(prototype, cat_kw)

        def to_u(theta):
            p = np.asarray([float(theta[label]) for label in labels], dtype=float)
            p = np.clip(p, 1.0e-12, None)
            p = p / p.sum()
            # invert stick-breaking
            phi = np.empty(k - 1, dtype=float)
            remaining = 1.0
            for i in range(k - 1):
                z = p[i] / remaining
                z = min(max(z, 1.0e-12), 1.0 - 1.0e-12)
                phi[i] = np.log(z) - np.log1p(-z) + np.log(float(k - 1 - i))
                remaining = remaining - p[i]
            return phi

        def from_u(phi):
            p = _stick_breaking_forward(np.asarray(phi, dtype=float))
            return {label: float(p[i]) for i, label in enumerate(labels)}

        return ParameterBridge(
            dim=k - 1,
            to_unconstrained=to_u,
            from_unconstrained=from_u,
            log_abs_det_jacobian=lambda phi: _stick_breaking_log_det(np.asarray(phi, dtype=float)),
            build=lambda theta: build(theta),
            param_names=tuple(str(label) for label in labels),
            initial_theta={label: float(prob_map[label]) for label in labels},
        )

    raise NotImplementedError(
        "sample_parameter_posterior does not support %s; supported families are "
        "Gaussian, Gamma, Exponential, Poisson, Bernoulli, Beta, and Categorical." % cls_name
    )


def _ctor_param_names(prototype: Any) -> tuple[str, ...]:
    import inspect

    try:
        sig = inspect.signature(type(prototype).__init__)
        return tuple(sig.parameters.keys())
    except (TypeError, ValueError):
        return ()


def _coerce_prior_logpdf(prior: Any, bridge: ParameterBridge) -> Callable[[Any], float]:
    """Return ``theta -> log prior(theta)`` from a flexible ``prior`` argument.

    Accepts: ``None`` (flat / improper prior, returns 0), a callable taking the
    parameter-space ``theta``, or a mixle distribution exposing
    ``log_density`` whose support matches the bridge's theta representation.
    """
    if prior is None:
        return lambda theta: 0.0
    if callable(prior) and not hasattr(prior, "log_density"):
        return lambda theta: float(prior(theta))
    if hasattr(prior, "log_density"):
        return lambda theta: float(prior.log_density(theta))
    raise TypeError("prior must be None, a callable, or a distribution with log_density.")


def sample_parameter_posterior(
    prototype_dist: Any,
    data: Any,
    prior: Any = None,
    sampler: str = "mh",
    steps: int = 2000,
    burn_in: int = 500,
    thin: int = 1,
    seed: int | None = None,
    proposal: Proposal | None = None,
    initial: Any = None,
    step_size: float = 0.05,
    num_steps: int = 20,
    grad_log_target: Callable[[Any], Any] | None = None,
    return_distributions: bool = False,
) -> MCMCResult:
    """Sample the parameter posterior ``p(theta | data)`` of a distribution.

    The model family is fixed by ``prototype_dist``; its parameters define the
    sampled space.  The unnormalized log target is the data log-likelihood
    (rebuilding the distribution per proposed ``theta`` and summing
    ``seq_log_density``) plus a prior log-density and the reparameterization
    Jacobian.  Sampling runs in an unconstrained space (see
    :func:`build_parameter_bridge`) so proposals stay in-domain, and retained
    samples are mapped back to parameter space.

    Args:
        prototype_dist: A distribution instance fixing the model family.
        data: Observations accepted by the family's encoder.
        prior: ``None`` (flat), a callable ``theta -> log p(theta)``, or a
            distribution with ``log_density`` over the parameter representation.
        sampler: ``'mh'`` (Metropolis-Hastings), ``'hmc'`` (Hamiltonian Monte
            Carlo), or ``'nuts'`` (No-U-Turn Sampler, self-tuning). HMC/NUTS use
            ``grad_log_target`` if given, else a finite-difference gradient.
        steps: Number of retained posterior samples.
        burn_in: Number of initial transitions to discard.
        thin: Keep one sample every ``thin`` transitions.
        seed: Seed for the RandomState.
        proposal: Optional MH proposal in the unconstrained space; defaults to a
            random walk (MH only).
        initial: Optional starting ``theta``; defaults to the prototype's
            parameters.
        step_size, num_steps: HMC leapfrog controls (NUTS self-tunes step size).
        grad_log_target: Optional exact gradient of ``log_target`` in the
            unconstrained space (e.g. from
            :func:`mixle.inference.mcmc.gradients.torch_gradient`). Replaces the
            finite-difference gradient for HMC/NUTS -- one backward pass per step
            instead of ``O(dim)`` target evaluations.
        return_distributions: If True, ``samples`` are rebuilt distribution
            objects instead of parameter-space values.

    Returns:
        MCMCResult whose ``samples`` are parameter-space values (or rebuilt
        distributions) and whose diagnostics come from the underlying driver.
    """
    bridge = build_parameter_bridge(prototype_dist)
    encoded, _ = _encode_data(prototype_dist, data)
    prior_logpdf = _coerce_prior_logpdf(prior, bridge)
    rng = np.random.RandomState(seed)

    def log_target(phi: Any) -> float:
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        if not np.all(np.isfinite(phi)):
            return -np.inf
        theta = bridge.from_unconstrained(phi)
        try:
            dist = bridge.build(theta)
            ll = _seq_log_density_sum(dist, encoded)
            lp = prior_logpdf(theta)
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            return -np.inf
        rv = ll + lp + bridge.log_abs_det_jacobian(phi)
        return rv if np.isfinite(rv) else -np.inf

    theta0 = bridge.initial_theta if initial is None else initial
    phi0 = bridge.to_unconstrained(theta0)

    sampler = sampler.lower()
    if sampler == "mh":
        if proposal is None:
            proposal = RandomWalkProposal(scale=0.1 * np.ones(bridge.dim, dtype=float))
        raw = metropolis_hastings(
            log_target,
            initial=phi0 if bridge.dim > 1 else float(phi0[0]),
            proposal=proposal,
            num_samples=steps,
            burn_in=burn_in,
            thin=thin,
            rng=rng,
        )
    elif sampler == "hmc":
        grad = grad_log_target if grad_log_target is not None else _finite_difference_gradient(log_target)
        raw = hamiltonian_monte_carlo(
            log_target,
            grad_log_target=grad,
            initial=phi0 if bridge.dim > 1 else float(phi0[0]),
            num_samples=steps,
            step_size=step_size,
            num_steps=num_steps,
            burn_in=burn_in,
            thin=thin,
            rng=rng,
        )
    elif sampler == "nuts":
        grad = grad_log_target if grad_log_target is not None else _finite_difference_gradient(log_target)
        raw = nuts(
            log_target,
            grad_log_target=grad,
            initial=phi0 if bridge.dim > 1 else float(phi0[0]),
            num_samples=steps,
            warmup=burn_in,
            thin=thin,
            rng=rng,
        )
    else:
        raise ValueError("sampler must be 'mh', 'hmc', or 'nuts'.")

    mapped: list[Any] = []
    for phi in raw.samples:
        phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
        theta = bridge.from_unconstrained(phi_arr)
        mapped.append(bridge.build(theta) if return_distributions else theta)

    return MCMCResult(
        samples=mapped, log_probs=raw.log_probs, accepted=raw.accepted, transition_labels=raw.transition_labels
    )


def _finite_difference_gradient(log_target: LogTarget, eps: float = 1.0e-5) -> Callable[[Any], Any]:
    """Return a central finite-difference gradient of ``log_target``."""

    def grad(x: Any) -> Any:
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        g = np.zeros_like(arr)
        for i in range(arr.size):
            step = eps * (1.0 + abs(arr[i]))
            up = arr.copy()
            down = arr.copy()
            up[i] += step
            down[i] -= step
            f_up = log_target(up if arr.size > 1 else float(up[0]))
            f_down = log_target(down if arr.size > 1 else float(down[0]))
            if not (np.isfinite(f_up) and np.isfinite(f_down)):
                g[i] = 0.0
            else:
                g[i] = (f_up - f_down) / (2.0 * step)
        return float(g[0]) if np.isscalar(x) or np.ndim(x) == 0 else g

    return grad
