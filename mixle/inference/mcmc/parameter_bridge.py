"""High-level parameter-posterior sampling for ``mixle.stats`` families.

Given a prototype distribution that fixes the model family, a dataset, and a
prior over parameters, this samples
``p(theta | data) proportional to exp(sum_i log p(x_i | theta)) * prior(theta)``
by running Metropolis-Hastings or Hamiltonian Monte Carlo in an unconstrained
reparameterization (log for positive scales, stick-breaking for probability
simplices) and mapping the retained samples back to parameter space (or to
rebuilt distribution objects).

Seven families (Gaussian, Gamma, Beta, Exponential, Poisson, Bernoulli,
Categorical) get a hand-tuned bridge below.  Every other family that declares a
:class:`~mixle.stats.compute.declarations.DistributionDeclaration` -- the same
per-parameter ``constraint``/``differentiable`` metadata
:mod:`mixle.inference.gradient_fit` already reads for autograd fitting -- is
bridged generically by :func:`_generic_declared_bridge`, so adding a new
supported family here is a matter of the family declaring itself, not editing
this module.  See :func:`build_parameter_bridge`'s docstring for exactly which
constraints have a generic reparameterization and why some declared families
(a covariance matrix, a coupled bound, a natural/scoring parameterization that
differs from the constructor) are still excluded.
"""

from __future__ import annotations

import inspect
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


def _stick_breaking_inverse(p: np.ndarray) -> np.ndarray:
    """Return the ``phi`` that :func:`_stick_breaking_forward` maps to ``p``.

    ``p`` must be a length-``k`` probability vector (strictly positive,
    summing to one); the returned vector has length ``k - 1``.
    """
    k = p.shape[0]
    phi = np.empty(k - 1, dtype=float)
    remaining = 1.0
    for i in range(k - 1):
        z = p[i] / remaining
        z = min(max(z, 1.0e-12), 1.0 - 1.0e-12)
        phi[i] = np.log(z) - np.log1p(-z) + np.log(float(k - 1 - i))
        remaining = remaining - p[i]
    return phi


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

    Seven families get a hand-tuned bridge below:

    * Gaussian ``(mu, sigma2)`` -> ``(mu, log sigma2)``
    * Gamma ``(k, theta)`` -> ``(log k, log theta)``
    * Exponential ``beta``/``lam`` (positive scalar) -> ``log``
    * Poisson ``lam`` -> ``log lam``
    * Bernoulli ``p`` -> ``logit p``
    * Beta ``(a, b)`` -> ``(log a, log b)``
    * Categorical probability map -> stick-breaking over the simplex

    Every other family falls through to :func:`_generic_declared_bridge`, which reads the
    same :class:`~mixle.stats.compute.declarations.DistributionDeclaration` that
    :mod:`mixle.inference.gradient_fit` already uses for autograd fitting. A declared
    parameter is sampled -- included in ``theta`` (a ``{name: value}`` dict here, unlike the
    tuple/scalar shapes above) -- when it is ``differentiable`` and its ``constraint`` is one
    of ``real`` (identity), ``positive``/``positive_vector`` (log), ``unit_interval`` (logit),
    ``real_vector`` (identity), or ``simplex_vector``/``simplex_map`` (stick-breaking);
    non-differentiable declared parameters (e.g. a Binomial's ``n``) are carried as fixed
    constructor keywords taken from ``prototype``, the same nuisance-parameter treatment
    :mod:`mixle.stats.bayes.conjugate` uses. This covers 33 families as of this writing
    (the 7 above plus 26 more -- see ``mcmc_test.py``'s ``GenericParameterBridgeTestCase``
    for the current list).

    Still unsupported, deliberately: families with no declaration at all (compositional/
    structured models -- mixtures, HMMs, copulas, ... -- for which a single flat parameter
    vector isn't the right shape); families whose declaration describes an exponential-family
    natural/scoring parameterization rather than the constructor's (e.g. von Mises' ``eta1``/
    ``eta2``/``log_const`` vs. its ``mu``/``kappa`` constructor -- the declared names aren't
    constructor keywords, so there's nothing safe to dispatch on); and declared-but-exotic
    constraints with no generic reparameterization yet (a covariance/precision matrix's
    ``positive_matrix``, or a coupled ``greater_than:``/``less_than:`` bound such as Uniform's
    ``high``). These need a family-specific bridge (or a matrix/coupled-bound generalization)
    rather than a generic one and are out of scope here.

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
            return _stick_breaking_inverse(p)

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

    generic = _generic_declared_bridge(prototype, kw)
    if generic is not None:
        return generic

    raise NotImplementedError(
        "sample_parameter_posterior does not support %s. Explicitly hand-tuned: Gaussian, Gamma, "
        "Exponential, Poisson, Bernoulli, Beta, Categorical. Every other distribution that declares a "
        "mixle.stats.compute.declarations.DistributionDeclaration is bridged automatically as long as "
        "every declared parameter name is also a constructor keyword and every differentiable "
        "parameter's constraint is one of real/positive/unit_interval/real_vector/positive_vector/"
        "simplex_vector/simplex_map (see build_parameter_bridge's docstring). %s has none of a "
        "declaration, or fails one of those two conditions -- most likely because it has no "
        "declaration at all, its declaration describes a natural/scoring parameterization rather than "
        "its constructor's, or a differentiable parameter's constraint (e.g. a covariance matrix or a "
        "coupled bound) has no generic reparameterization yet." % (cls_name, cls_name)
    )


def _ctor_param_names(prototype: Any) -> tuple[str, ...]:
    try:
        sig = inspect.signature(type(prototype).__init__)
        return tuple(sig.parameters.keys())
    except (TypeError, ValueError):
        return ()


# ---------------------------------------------------------------------------
# Generic bridge: dispatch against the mixle.stats declaration registry
# ---------------------------------------------------------------------------
#
# mixle.stats.compute.declarations.DistributionDeclaration already records, per family,
# the constructor parameters generated scoring kernels need: a name, a `constraint` (its
# domain -- "positive", "unit_interval", a simplex, ...), and a `differentiable` flag that
# mixle.inference.gradient_fit already uses to separate free parameters from cached/nuisance
# ones for autograd fitting. That is exactly the metadata this module's hand-written 7-family
# table was duplicating per family, so a declared family is bridged generically instead of
# needing its own branch above.


@dataclass(frozen=True)
class _ParamBlock:
    """One sampled parameter's slice of the combined unconstrained ``phi`` vector."""

    name: str
    dim: int
    to_unconstrained: Callable[[Any], np.ndarray]
    from_unconstrained: Callable[[np.ndarray], Any]
    log_abs_det_jacobian: Callable[[np.ndarray], float]


def _real_block(name: str, length: int | None) -> _ParamBlock:
    if length is None:
        return _ParamBlock(
            name=name,
            dim=1,
            to_unconstrained=lambda v: np.asarray([float(v)], dtype=float),
            from_unconstrained=lambda phi: float(phi[0]),
            log_abs_det_jacobian=lambda phi: 0.0,
        )
    return _ParamBlock(
        name=name,
        dim=length,
        to_unconstrained=lambda v: np.asarray(v, dtype=float).reshape(-1),
        from_unconstrained=lambda phi: np.asarray(phi, dtype=float).copy(),
        log_abs_det_jacobian=lambda phi: 0.0,
    )


def _positive_block(name: str, length: int | None) -> _ParamBlock:
    if length is None:
        return _ParamBlock(
            name=name,
            dim=1,
            to_unconstrained=lambda v: np.asarray([float(np.log(v))], dtype=float),
            from_unconstrained=lambda phi: float(np.exp(phi[0])),
            log_abs_det_jacobian=lambda phi: float(phi[0]),
        )
    return _ParamBlock(
        name=name,
        dim=length,
        to_unconstrained=lambda v: np.log(np.asarray(v, dtype=float)),
        from_unconstrained=lambda phi: np.exp(np.asarray(phi, dtype=float)),
        log_abs_det_jacobian=lambda phi: float(np.sum(phi)),
    )


def _unit_interval_block(name: str) -> _ParamBlock:
    def to_u(v: Any) -> np.ndarray:
        v = float(v)
        return np.asarray([np.log(v) - np.log1p(-v)], dtype=float)

    def from_u(phi: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-phi[0])))

    def log_det(phi: np.ndarray) -> float:
        # d p / d logit = p (1 - p); log = -softplus(-phi) - softplus(phi)
        return float(-_softplus(-phi[0]) - _softplus(phi[0]))

    return _ParamBlock(name=name, dim=1, to_unconstrained=to_u, from_unconstrained=from_u, log_abs_det_jacobian=log_det)


def _simplex_vector_block(name: str, length: int) -> _ParamBlock | None:
    if length < 2:
        return None

    def to_u(v: Any) -> np.ndarray:
        p = np.clip(np.asarray(v, dtype=float), 1.0e-12, None)
        p = p / p.sum()
        return _stick_breaking_inverse(p)

    return _ParamBlock(
        name=name,
        dim=length - 1,
        to_unconstrained=to_u,
        from_unconstrained=lambda phi: _stick_breaking_forward(np.asarray(phi, dtype=float)),
        log_abs_det_jacobian=lambda phi: _stick_breaking_log_det(np.asarray(phi, dtype=float)),
    )


def _simplex_map_block(name: str, labels: tuple[Any, ...]) -> _ParamBlock | None:
    if len(labels) < 2:
        return None

    def to_u(prob_map: Any) -> np.ndarray:
        p = np.asarray([float(prob_map[label]) for label in labels], dtype=float)
        p = np.clip(p, 1.0e-12, None)
        p = p / p.sum()
        return _stick_breaking_inverse(p)

    def from_u(phi: np.ndarray) -> dict[Any, float]:
        p = _stick_breaking_forward(np.asarray(phi, dtype=float))
        return {label: float(p[i]) for i, label in enumerate(labels)}

    return _ParamBlock(
        name=name,
        dim=len(labels) - 1,
        to_unconstrained=to_u,
        from_unconstrained=from_u,
        log_abs_det_jacobian=lambda phi: _stick_breaking_log_det(np.asarray(phi, dtype=float)),
    )


def _declared_parameter_block(name: str, constraint: str, value: Any) -> _ParamBlock | None:
    """Return the unconstrained-space block for one declared, differentiable parameter.

    Returns ``None`` when ``constraint`` has no generic reparameterization yet (a matrix, a
    coupled ``greater_than:``/``less_than:`` bound, or anything else outside the set below).
    """
    if constraint == "real":
        return _real_block(name, None)
    if constraint == "positive":
        return _positive_block(name, None)
    if constraint == "unit_interval":
        return _unit_interval_block(name)
    if constraint == "real_vector":
        return _real_block(name, len(value))
    if constraint == "positive_vector":
        return _positive_block(name, len(value))
    if constraint == "simplex_vector":
        return _simplex_vector_block(name, len(value))
    if constraint == "simplex_map":
        if not isinstance(value, Mapping):
            return None
        return _simplex_map_block(name, tuple(value.keys()))
    return None


def _generic_declared_bridge(prototype: Any, kw: dict[str, Any]) -> ParameterBridge | None:
    """Build a :class:`ParameterBridge` from ``prototype``'s ``DistributionDeclaration``, or
    return ``None`` (never raise) when the family cannot be bridged generically.

    A family is bridged when (1) it has a declaration, (2) every declared parameter name is
    also a constructor keyword of ``type(prototype)`` -- ruling out families whose declaration
    describes an exponential-family natural/scoring parameterization instead (e.g. von Mises
    declares ``eta1``/``eta2``/``log_const`` for generated scoring kernels, but its constructor
    takes ``mu``/``kappa``: nothing here would be safe to dispatch on), (3) every constructor
    argument the declaration doesn't cover has a default, and (4) every *differentiable*
    declared parameter's constraint has a generic transform (see
    :func:`_declared_parameter_block`). Non-differentiable declared parameters (e.g. a
    Binomial's ``n``, an ErdosRenyiGraph's ``directed``) are carried as fixed constructor
    keywords taken from ``prototype`` rather than sampled -- the same treatment
    :mod:`mixle.stats.bayes.conjugate` gives a known nuisance parameter.
    """
    from mixle.stats.compute.declarations import declaration_for

    cls = type(prototype)
    declaration = declaration_for(prototype)
    if declaration is None:
        return None

    try:
        ctor_params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return None

    declared_names = {spec.name for spec in declaration.parameters}
    for pname, param in ctor_params.items():
        if pname == "self" or param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        if param.kind == inspect.Parameter.POSITIONAL_ONLY and pname in declared_names:
            return None  # can't pass a positional-only constructor argument by keyword
        if param.default is inspect.Parameter.empty and pname not in declared_names and pname != "name":
            return None  # a required constructor argument the declaration doesn't cover

    fixed_kwargs: dict[str, Any] = {}
    blocks: list[_ParamBlock] = []
    for spec in declaration.parameters:
        if spec.name not in ctor_params:
            return None  # declaration describes a natural/scoring parameterization, not the ctor
        value = getattr(prototype, spec.name)
        if not spec.differentiable:
            fixed_kwargs[spec.name] = value
            continue
        block = _declared_parameter_block(spec.name, spec.constraint, value)
        if block is None:
            return None
        blocks.append(block)

    if not blocks:
        return None  # nothing to sample a posterior over

    offsets: list[int] = []
    total = 0
    for block in blocks:
        offsets.append(total)
        total += block.dim

    def to_u(theta: dict[str, Any]) -> np.ndarray:
        parts = [block.to_unconstrained(theta[block.name]) for block in blocks]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

    def from_u(phi: np.ndarray) -> dict[str, Any]:
        phi = np.asarray(phi, dtype=float)
        return {
            block.name: block.from_unconstrained(phi[offset : offset + block.dim])
            for block, offset in zip(blocks, offsets)
        }

    def log_det(phi: np.ndarray) -> float:
        phi = np.asarray(phi, dtype=float)
        return float(
            sum(block.log_abs_det_jacobian(phi[offset : offset + block.dim]) for block, offset in zip(blocks, offsets))
        )

    def build(theta: dict[str, Any]) -> Any:
        return cls(**fixed_kwargs, **theta, **kw)

    return ParameterBridge(
        dim=total,
        to_unconstrained=to_u,
        from_unconstrained=from_u,
        log_abs_det_jacobian=log_det,
        build=build,
        param_names=tuple(block.name for block in blocks),
        initial_theta={block.name: getattr(prototype, block.name) for block in blocks},
    )


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
