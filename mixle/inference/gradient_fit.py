"""Gradient-based (autograd) maximum-likelihood and MAP fitting.

``fit_mle`` / ``fit_map`` optimize a distribution's parameters by gradient descent through a Torch
backend (constraint reparameterization, optional declaration-backed priors), returning a
``GradientFitResult``. This is the gradient counterpart of the EM drivers in ``estimation.py``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Any, TypeVar

import numpy as np

from mixle.inference.priors import as_prior_dict
from mixle.stats.compute.gradient import GradientFitError
from mixle.stats.compute.pdist import SequenceEncodableProbabilityDistribution

T = TypeVar("T")
E0 = TypeVar("E0")


@dataclass
class GradientFitResult:
    """Optimization result for generic autograd MLE/MAP fitting."""

    model: SequenceEncodableProbabilityDistribution
    value: float
    iterations: int
    history: tuple[float, ...] = ()
    converged: bool = False
    initial_value: float | None = None
    final_delta: float | None = None
    log_likelihood: float | None = None
    log_prior: float | None = None
    prior_strength: float = 0.0
    tag: str = "MLE"
    best_value: float | None = None
    best_iteration: int | None = None
    final_gradient_norm: float | None = None

    def as_tuple(self) -> tuple[SequenceEncodableProbabilityDistribution, float]:
        """Return the historical ``(model, objective)`` shape."""
        return self.model, self.value

    @property
    def objective_change(self) -> float | None:
        """Return the signed objective change from the start of optimization."""
        if self.initial_value is None:
            return None
        return self.value - self.initial_value

    @property
    def improvement(self) -> float | None:
        """Return the maximization improvement from the start objective."""
        return self.objective_change

    @property
    def best_improvement(self) -> float | None:
        """Return best improvement seen during optimization."""
        if self.initial_value is None or self.best_value is None:
            return None
        return self.best_value - self.initial_value

    @property
    def prior_sensitivity(self) -> float | None:
        """Return the magnitude fraction of the final objective coming from the prior."""
        if self.log_likelihood is None or self.log_prior is None:
            return None
        likelihood = abs(float(self.log_likelihood))
        prior = abs(float(self.log_prior))
        total = likelihood + prior
        return 0.0 if total == 0.0 else prior / total


def _torch_for_gradient_fit(engine, precision: Any | None = None):
    try:
        import torch
    except ImportError as e:
        raise ImportError("fit_mle/fit_map require torch for autograd-backed engines.") from e

    if engine is None:
        from mixle.engines import TorchEngine

        engine = TorchEngine(dtype=precision or torch.float64)
    elif precision is not None:
        from mixle.engines import engine_with_precision

        engine = engine_with_precision(engine, precision)
    if not getattr(engine, "supports_autograd", False):
        raise ValueError("fit_mle/fit_map require an engine with supports_autograd=True.")
    return torch, engine


def _tensor_param(value, engine, torch, transform=None):
    tensor = engine.asarray(value, dtype=getattr(engine, "dtype", None))
    tensor = tensor.clone().detach()
    eps = 1.0e-8
    if transform == "log":
        tensor = torch.log(torch.clamp(tensor, min=eps))
    elif transform == "logit":
        tensor = torch.logit(torch.clamp(tensor, min=eps, max=1.0 - eps))
    elif transform == "logits":
        tensor = torch.log(torch.clamp(tensor, min=eps))
    tensor.requires_grad_(True)
    return tensor


def _gradient_raw_state(dist, engine, torch, leaves):
    from mixle.stats.compute.declarations import declaration_for

    hook = getattr(dist, "gradient_fit_state", None)
    if callable(hook):
        state = hook(engine, torch, leaves, _gradient_raw_state, _tensor_param)
        if state is not None:
            return state

    declaration = declaration_for(dist)
    if declaration is None or not callable(getattr(dist, "backend_seq_log_density", None)):
        return ("fixed", dist)
    if not declaration.differentiable:
        return ("fixed", dist)

    raw = {}
    fixed = {}
    for spec in declaration.parameters:
        value = getattr(dist, spec.name)
        if not spec.differentiable:
            fixed[spec.name] = value
            continue
        if _is_ordered_bound_constraint(spec.constraint):
            anchor = _ordered_bound_anchor(spec.constraint)
            delta = _ordered_bound_delta(getattr(dist, spec.name), getattr(dist, anchor), spec.constraint)
            raw_name = _coupled_raw_name(spec.name, anchor, spec.constraint)
            raw[raw_name] = _tensor_param(delta, engine, torch, transform="log")
            leaves.append(raw[raw_name])
            continue
        raw_name, transform = _raw_name_and_transform(spec.name, spec.constraint)
        raw[raw_name] = _tensor_param(value, engine, torch, transform=transform)
        leaves.append(raw[raw_name])
    return ("leaf", dist, declaration, raw, fixed)


def _raw_name_and_transform(name: str, constraint: str) -> tuple[str, str | None]:
    if _is_ordered_bound_constraint(constraint):
        return _coupled_raw_name(name, _ordered_bound_anchor(constraint), constraint), "log"
    if constraint in ("positive", "positive_vector", "positive_matrix"):
        return "log_" + name, "log"
    if constraint == "unit_interval":
        return "logit_" + name, "logit"
    if constraint in ("simplex", "simplex_vector", "row_simplex_matrix", "column_simplex_matrix"):
        return name + "_logits", "logits"
    return name, None


def _canonical_value(name: str, spec_name: str, constraint: str, raw: dict, torch):
    if constraint in ("positive", "positive_vector", "positive_matrix"):
        return torch.exp(raw["log_" + spec_name])
    if constraint == "unit_interval":
        return torch.sigmoid(raw["logit_" + spec_name])
    if constraint in ("simplex", "simplex_vector"):
        return torch.softmax(raw[spec_name + "_logits"], dim=0)
    if constraint == "row_simplex_matrix":
        return torch.softmax(raw[spec_name + "_logits"], dim=1)
    if constraint == "column_simplex_matrix":
        return torch.softmax(raw[spec_name + "_logits"], dim=0)
    return raw[name]


def _is_greater_than_constraint(constraint: str) -> bool:
    return str(constraint).startswith("greater_than:")


def _is_less_than_constraint(constraint: str) -> bool:
    return str(constraint).startswith("less_than:")


def _is_ordered_bound_constraint(constraint: str) -> bool:
    return _is_greater_than_constraint(constraint) or _is_less_than_constraint(constraint)


def _ordered_bound_anchor(constraint: str) -> str:
    anchor = str(constraint).split(":", 1)[1] if ":" in str(constraint) else ""
    if not anchor:
        raise ValueError("%s constraint requires an anchor parameter." % constraint)
    return anchor


def _ordered_bound_delta(value: Any, anchor_value: Any, constraint: str) -> Any:
    if _is_greater_than_constraint(constraint):
        delta = value - anchor_value
    else:
        delta = anchor_value - value
    delta_arr = np.asarray(delta, dtype=np.float64)
    if np.any(delta_arr <= 0.0) or not np.all(np.isfinite(delta_arr)):
        raise ValueError("Initial value for %s must satisfy its ordered bound." % constraint)
    return delta


def _coupled_raw_name(name: str, anchor: str, constraint: str) -> str:
    return "log_" + _ordered_bound_delta_name(name, anchor, constraint)


def _ordered_bound_delta_name(name: str, anchor: str, constraint: str) -> str:
    if _is_greater_than_constraint(constraint):
        return "%s_minus_%s" % (name, anchor)
    return "%s_minus_%s" % (anchor, name)


def _gradient_shadow_state(state, torch):
    shadow_fn = getattr(state, "shadow", None)
    if callable(shadow_fn):
        return shadow_fn(torch, _gradient_shadow_state)
    kind = state[0]
    if kind == "leaf":
        _, template, declaration, raw, fixed = state
        shadow = object.__new__(type(template))
        shadow.__dict__.update(getattr(template, "__dict__", {}))
        params = {}
        for spec in declaration.parameters:
            if spec.name in fixed:
                params[spec.name] = fixed[spec.name]
            elif _is_ordered_bound_constraint(spec.constraint):
                anchor = _ordered_bound_anchor(spec.constraint)
                anchor_value = params.get(anchor, getattr(template, anchor))
                delta = torch.exp(raw[_coupled_raw_name(spec.name, anchor, spec.constraint)])
                if _is_greater_than_constraint(spec.constraint):
                    params[spec.name] = anchor_value + delta
                else:
                    params[spec.name] = anchor_value - delta
            else:
                raw_name, _ = _raw_name_and_transform(spec.name, spec.constraint)
                params[spec.name] = _canonical_value(raw_name, spec.name, spec.constraint, raw, torch)
            setattr(shadow, spec.name, params[spec.name])
        if "p_vec" in params:
            shadow.log_p_vec = torch.log(params["p_vec"])
        return shadow
    if kind == "fixed":
        return state[1]
    raise GradientFitError("Unknown gradient fit state %s." % kind)


def _gradient_enc_chunks(enc):
    """Normalize an encoded payload into a list of per-chunk payloads to score.

    ``mixle.stats.seq_encode`` returns the chunked form ``[(size, payload), ...]``;
    an encoder's own ``seq_encode`` returns a single bare payload. The gradient
    objective sums log densities over observations, so either form reduces to a
    list of payloads whose per-chunk score sums are added together.
    """
    if (
        isinstance(enc, list)
        and enc
        and all(isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], (int, np.integer)) for c in enc)
    ):
        return [payload for _, payload in enc]
    return [enc]


def _gradient_score_state(state, enc, engine, torch):
    from mixle.stats.compute.backend import backend_seq_log_density

    score_fn = getattr(state, "score", None)
    if callable(score_fn):
        return score_fn(enc, engine, torch, _gradient_score_state)
    kind = state[0]
    if kind == "leaf":
        return backend_seq_log_density(_gradient_shadow_state(state, torch), enc, engine)
    if kind == "fixed":
        return engine.asarray(state[1].seq_log_density(enc))
    raise GradientFitError("Unknown gradient fit state %s." % kind)


def _detach_value(x):
    if hasattr(x, "detach"):
        arr = x.detach().cpu().numpy()
        return float(arr) if np.ndim(arr) == 0 else arr
    return x


def _tensor_scalar(x) -> float:
    return float(x.detach().cpu().item())


def _gradient_best_entry(history: Sequence[float]) -> tuple[float, int]:
    values = np.asarray(history, dtype=np.float64)
    idx = int(np.nanargmax(values))
    return float(values[idx]), idx


def _gradient_objective_norm(torch, leaves: Sequence[Any], objective) -> float:
    for leaf in leaves:
        if getattr(leaf, "grad", None) is not None:
            leaf.grad = None
    value = objective()
    value.backward()
    total = None
    for leaf in leaves:
        grad = getattr(leaf, "grad", None)
        if grad is None:
            continue
        term = torch.sum(grad.detach() * grad.detach())
        total = term if total is None else total + term
    for leaf in leaves:
        if getattr(leaf, "grad", None) is not None:
            leaf.grad = None
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach().cpu().item())


def _gradient_build_state(state, torch):
    build_fn = getattr(state, "build", None)
    if callable(build_fn):
        return build_fn(torch, _gradient_build_state, _detach_value)
    kind = state[0]
    if kind == "leaf":
        _, template, declaration, raw, fixed = state
        args = []
        params = {}
        for spec in declaration.parameters:
            if spec.name in fixed:
                value = fixed[spec.name]
            elif _is_ordered_bound_constraint(spec.constraint):
                anchor = _ordered_bound_anchor(spec.constraint)
                anchor_value = params.get(anchor, getattr(template, anchor))
                delta = torch.exp(raw[_coupled_raw_name(spec.name, anchor, spec.constraint)])
                value = anchor_value + delta if _is_greater_than_constraint(spec.constraint) else anchor_value - delta
            else:
                raw_name, _ = _raw_name_and_transform(spec.name, spec.constraint)
                value = _canonical_value(raw_name, spec.name, spec.constraint, raw, torch)
            params[spec.name] = value
            args.append(_detach_value(value))
        kwargs = {}
        if hasattr(template, "name"):
            kwargs["name"] = template.name
        if hasattr(template, "keys"):
            kwargs["keys"] = template.keys
        try:
            return type(template)(*args, **kwargs)
        except TypeError:
            kwargs.pop("keys", None)
            return type(template)(*args, **kwargs)
    if kind == "fixed":
        return state[1]
    raise GradientFitError("Unknown gradient fit state %s." % kind)


def _raw_l2_prior(leaves, initial_leaves, torch, prior_strength: float, engine):
    if prior_strength == 0.0:
        return _prior_zero(torch, engine, leaves[0] if leaves else None)
    penalty = _prior_zero(torch, engine, leaves[0] if leaves else None)
    for cur, start in zip(leaves, initial_leaves):
        delta = cur - start
        penalty = penalty + torch.sum(delta * delta)
    return -0.5 * float(prior_strength) * penalty


def fit_mle(
    enc: Any,
    model: SequenceEncodableProbabilityDistribution,
    engine=None,
    max_its: int = 500,
    lr: float = 0.05,
    optimizer: str = "adam",
    tol: float = 1.0e-7,
    out: IO | None = None,
    print_iter: int = 100,
    precision: Any | None = None,
    return_result: bool = False,
) -> Any:
    """Fit converted models by maximizing backend log likelihood with autograd.

    The generic implementation handles declaration-backed tensor leaves and
    delegates structured model families to distribution-owned
    ``gradient_fit_state`` hooks.
    """
    return _fit_gradient(
        enc,
        model,
        engine,
        max_its,
        lr,
        optimizer,
        tol,
        out,
        print_iter,
        tag="MLE",
        prior_strength=0.0,
        precision=precision,
        return_result=return_result,
    )


def fit_map(
    enc: Any,
    model: SequenceEncodableProbabilityDistribution,
    engine=None,
    prior_strength: float = 1.0,
    priors: Any | None = None,
    max_its: int = 500,
    lr: float = 0.05,
    optimizer: str = "adam",
    tol: float = 1.0e-7,
    out: IO | None = None,
    print_iter: int = 100,
    precision: Any | None = None,
    return_result: bool = False,
) -> Any:
    """Fit converted models with MAP priors over declaration-backed parameters.

    ``prior_strength=0`` is exactly the same objective as ``fit_mle`` when no
    explicit ``priors`` are supplied.  ``priors`` may be a legacy prior dict or
    one of the helpers from ``mixle.inference.priors``.
    """
    return _fit_gradient(
        enc,
        model,
        engine,
        max_its,
        lr,
        optimizer,
        tol,
        out,
        print_iter,
        tag="MAP",
        prior_strength=float(prior_strength),
        priors=priors,
        precision=precision,
        return_result=return_result,
    )


def _fit_gradient(
    enc,
    model,
    engine,
    max_its,
    lr,
    optimizer,
    tol,
    out,
    print_iter,
    tag,
    prior_strength,
    priors=None,
    precision: Any | None = None,
    return_result: bool = False,
):
    torch, engine = _torch_for_gradient_fit(engine, precision=precision)
    if hasattr(enc, "payload"):
        enc = enc.payload
    enc_chunks = _gradient_enc_chunks(enc)

    leaves: list[Any] = []
    state = _gradient_raw_state(model, engine, torch, leaves)
    if not leaves:
        raise GradientFitError("%s has no differentiable parameters." % type(model).__name__)
    initial_leaves_by_id = {id(leaf): leaf.detach().clone() for leaf in leaves}
    priors = as_prior_dict(priors)

    def log_likelihood():
        total = None
        for chunk in enc_chunks:
            part = engine.sum(_gradient_score_state(state, chunk, engine, torch))
            total = part if total is None else total + part
        return total

    def log_prior():
        if priors is not None or prior_strength != 0.0:
            return _gradient_log_prior_state(state, priors, prior_strength, torch, engine, initial_leaves_by_id)
        return _prior_zero(torch, engine, leaves[0])

    def objective():
        return log_likelihood() + log_prior()

    # one gradient-descent loop, shared with objectives.optimize_torch_objective.
    # restore_best=False keeps this fit's "return the final iterate" semantics
    # (the leaves still hold the final values for the diagnostics below).
    from mixle.inference.objectives import optimize_torch_objective

    loop = optimize_torch_objective(
        leaves,
        objective,
        engine=engine,
        max_its=max_its,
        lr=lr,
        optimizer=optimizer,
        tol=tol,
        maximize=True,
        out=out,
        print_iter=print_iter,
        restore_best=False,
        return_result=True,
    )
    history = list(loop.history)
    iterations = loop.iterations
    converged = loop.converged

    final_obj = history[-1]
    final_ll = _tensor_scalar(log_likelihood())
    final_lp = _tensor_scalar(log_prior())
    final_delta = history[-1] - history[-2] if len(history) > 1 else None
    best_value, best_iteration = _gradient_best_entry(history)
    final_gradient_norm = _gradient_objective_norm(torch, leaves, objective)
    result = GradientFitResult(
        _gradient_build_state(state, torch),
        final_obj,
        iterations,
        history=tuple(history),
        converged=converged,
        initial_value=history[0],
        final_delta=final_delta,
        log_likelihood=final_ll,
        log_prior=final_lp,
        prior_strength=float(prior_strength),
        tag=tag,
        best_value=best_value,
        best_iteration=best_iteration,
        final_gradient_norm=final_gradient_norm,
    )
    return result if return_result else result.as_tuple()


def _gradient_log_prior_state(state, priors, prior_strength: float, torch, engine, initial_leaves_by_id):
    """Structured log prior for declaration-backed MAP objectives."""
    prior_fn = getattr(state, "log_prior", None)
    if callable(prior_fn):
        return prior_fn(priors, prior_strength, torch, engine, initial_leaves_by_id, _gradient_log_prior_state)
    kind = state[0]
    if kind == "leaf":
        shadow = _gradient_shadow_state(state, torch)
        declaration, raw = state[2], state[3]
        prior_hook = getattr(shadow, "gradient_log_prior", None)
        if callable(prior_hook):
            hook_lp = prior_hook(priors, prior_strength, torch, engine)
            if hook_lp is not None:
                return hook_lp

        lp = _prior_zero(torch, engine, next(iter(raw.values()), None))
        matched = False

        for spec in declaration.parameters:
            param_prior = _parameter_prior(priors, spec.name)
            if spec.name in state[4]:
                continue
            if _is_ordered_bound_constraint(spec.constraint):
                anchor = _ordered_bound_anchor(spec.constraint)
                delta_name = _ordered_bound_delta_name(spec.name, anchor, spec.constraint)
                param_prior = _parameter_prior(priors, delta_name) or param_prior
                if param_prior is None:
                    continue
                pfam = _prior_family(param_prior)
                if pfam == "gamma":
                    value = torch.exp(raw[_coupled_raw_name(spec.name, anchor, spec.constraint)])
                    shape = engine.asarray(param_prior.get("shape", 1.0))
                    rate = engine.asarray(param_prior.get("rate", 0.0))
                    # Clamp away from 0: a saturated Adam tail can drive exp(raw) to underflow to
                    # exactly 0.0, making log(value) == -inf so (shape-1)*log(value) becomes NaN
                    # (shape == 1) or diverges (shape < 1), poisoning the whole MAP objective.
                    value_safe = torch.clamp(value, min=1.0e-12)
                    lp = lp + torch.sum((shape - 1.0) * torch.log(value_safe) - rate * value)
                    matched = True
                continue
            if param_prior is None:
                continue
            raw_name, _ = _raw_name_and_transform(spec.name, spec.constraint)
            value = _canonical_value(raw_name, spec.name, spec.constraint, raw, torch)
            pfam = _prior_family(param_prior)
            if pfam == "gamma" and spec.constraint in ("positive", "positive_vector", "positive_matrix"):
                shape = engine.asarray(param_prior.get("shape", 1.0))
                rate = engine.asarray(param_prior.get("rate", 0.0))
                # Clamp away from 0: a saturated Adam tail can drive exp(raw) to underflow to
                # exactly 0.0, making log(value) == -inf so (shape-1)*log(value) becomes NaN
                # (shape == 1) or diverges (shape < 1), poisoning the whole MAP objective.
                value_safe = torch.clamp(value, min=1.0e-12)
                lp = lp + torch.sum((shape - 1.0) * torch.log(value_safe) - rate * value)
                matched = True
            elif pfam == "beta" and spec.constraint == "unit_interval":
                alpha = engine.asarray(param_prior.get("alpha", 1.0))
                beta = engine.asarray(param_prior.get("beta", 1.0))
                # Clamp away from {0, 1}: a saturated sigmoid tail (raw >~ 37) gives value == 1.0
                # exactly, making log1p(-value) == -inf and poisoning the whole MAP objective.
                value = torch.clamp(value, 1.0e-12, 1.0 - 1.0e-12)
                lp = lp + torch.sum((alpha - 1.0) * torch.log(value) + (beta - 1.0) * torch.log1p(-value))
                matched = True
            elif pfam == "dirichlet" and spec.constraint in (
                "simplex",
                "simplex_vector",
                "row_simplex_matrix",
                "column_simplex_matrix",
            ):
                alpha = _dirichlet_alpha_tensor(param_prior.get("alpha"), None, value, engine, torch)
                # Clamp away from 0: a saturated softmax tail can drive a component to exactly 0,
                # making log(value) == -inf and poisoning the whole MAP objective.
                value = torch.clamp(value, 1.0e-12, 1.0 - 1.0e-12)
                lp = lp + torch.sum((alpha - 1.0) * torch.log(value))
                matched = True

        if matched:
            return lp
        return _raw_l2_prior(
            _state_leaves(state), _state_initial_leaves(state, initial_leaves_by_id), torch, prior_strength, engine
        )
    if kind == "fixed":
        return _prior_zero(torch, engine)
    return _prior_zero(torch, engine)


def _prior_zero(torch, engine, ref=None):
    if ref is not None:
        return torch.as_tensor(0.0, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(0.0, dtype=engine.dtype, device=engine.device)


def _prior_family(prior):
    return prior.get("family") if isinstance(prior, Mapping) else None


def _prior_parameter_matches(prior, name: str) -> bool:
    return not isinstance(prior, Mapping) or prior.get("parameter") in (None, name)


def _parameter_prior(priors, name: str):
    family = _prior_family(priors)
    if family in ("gamma", "beta", "dirichlet") and _prior_parameter_matches(priors, name):
        return priors
    if isinstance(priors, Mapping):
        if isinstance(priors.get("parameters"), Mapping) and name in priors["parameters"]:
            return as_prior_dict(priors["parameters"][name])
        if name in priors:
            return as_prior_dict(priors[name])
    return None


def _dirichlet_alpha_tensor(alpha, labels, logits, engine, torch):
    if alpha is None:
        alpha = 1.0
    if isinstance(alpha, Mapping):
        if labels is None:
            raise ValueError("Dirichlet alpha mappings require categorical labels.")
        alpha = [alpha.get(label, 1.0) for label in labels]
    alpha_t = engine.asarray(alpha)
    if alpha_t.ndim == 0:
        return alpha_t + torch.zeros_like(logits)
    return alpha_t


def _state_leaves(state):
    kind = state[0]
    if kind == "leaf":
        return list(state[3].values())
    return []


def _state_initial_leaves(state, initial_leaves_by_id):
    return [initial_leaves_by_id[id(leaf)] for leaf in _state_leaves(state)]
