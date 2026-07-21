"""Stable, executable per-parameter optimizer routing for Torch modules.

The statistical runtime can describe richer update contracts, but stable model code cannot depend on
``mixle.experimental``.  This module is the small common execution layer: it accepts either its own
shape/name-derived plan or the experimental runtime's structurally compatible plan and executes the
selected update for each parameter.  AdamW is available as an explicit route, not the automatic
default.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "NeuralOptimizerPlan",
    "NeuralOptimizerRoute",
    "build_auto_optimizer",
    "build_routed_optimizer",
    "plan_neural_optimizer",
    "resolve_neural_optimizer",
    "shard_safe_neural_optimizer_plan",
]


@dataclass(frozen=True)
class NeuralOptimizerRoute:
    """One named parameter's executable update family and selection reason."""

    name: str
    family: str
    reason: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class NeuralOptimizerPlan:
    """Inspectable non-Adam-first routes for a Torch module."""

    routes: tuple[NeuralOptimizerRoute, ...]
    sign_stable: bool | None = None

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({route.family for route in self.routes}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "families": list(self.families),
            "sign_stable": self.sign_stable,
            "routes": [
                {
                    "name": route.name,
                    "family": route.family,
                    "reason": route.reason,
                    "shape": list(route.shape),
                }
                for route in self.routes
            ],
        }


def _automatic_family(
    name: str, shape: tuple[int, ...], numel: int, matrix_min_elements: int, sign_stable: bool
) -> tuple[str, str]:
    lower = name.lower()
    parts = set(lower.replace("_", ".").split("."))
    if "embed" in lower or parts.intersection({"emb", "wte", "wpe"}):
        return "adagrad", "embedding rows benefit from frequency-adaptive diagonal scaling"
    if "router" in lower or "gate" in lower:
        return "adagrad", "gate/router receives diagonal adaptive updates until Fisher state is supplied"
    if "lora" in lower or "adapter" in lower:
        return "adagrad", "low-rank adapter keeps factor-local diagonal state"
    if len(shape) == 2 and numel >= matrix_min_elements and min(shape) >= 16:
        return "muon", "large hidden matrix receives an orthogonalized momentum direction"
    if len(shape) <= 1 or "norm" in lower or lower.endswith("bias") or ".bias" in lower:
        if not sign_stable:
            return "adagrad", "sign-unstable scalar/vector block receives scale-adaptive one-moment updates"
        return "rprop", "full-batch scalar/vector block receives resilient sign-and-step updates"
    if not sign_stable:
        return "sgd_momentum", "sign-unstable small matrix uses momentum without two-moment state"
    return "rprop", "small full-batch block receives resilient sign-and-step updates"


def plan_neural_optimizer(
    module: Any, *, matrix_min_elements: int = 4_096, sign_stable: bool = True
) -> NeuralOptimizerPlan:
    """Plan update families from parameter role and shape without importing Torch at module import time."""

    if matrix_min_elements < 1:
        raise ValueError("matrix_min_elements must be positive.")
    named = getattr(module, "named_parameters", None)
    if not callable(named):
        raise TypeError("module must expose named_parameters().")
    routes = []
    seen: set[int] = set()
    for name, parameter in named():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        shape = tuple(int(value) for value in parameter.shape)
        if not bool(getattr(parameter, "requires_grad", True)):
            family, reason = "frozen", "parameter is frozen"
        else:
            numel = int(parameter.numel())
            family, reason = _automatic_family(str(name), shape, numel, matrix_min_elements, sign_stable)
        routes.append(NeuralOptimizerRoute(str(name), family, reason, shape))
    return NeuralOptimizerPlan(tuple(routes), sign_stable)


def shard_safe_neural_optimizer_plan(plan: NeuralOptimizerPlan) -> NeuralOptimizerPlan:
    """Localize routes whose geometry is not invariant under parameter sharding.

    Muon/Kronecker/natural-gradient directions require a global matrix. Applying
    them independently to rectangular shards is a different algorithm. Until a
    backend declares and executes the required gather/factor reductions, use
    momentum for those blocks and record the reason in the plan.
    """

    global_geometry = {"muon", "kronecker", "natural_gradient"}
    routes = tuple(
        NeuralOptimizerRoute(
            route.name,
            "sgd_momentum" if route.family in global_geometry else route.family,
            (
                route.reason + "; localized to momentum because global geometry is sharded"
                if route.family in global_geometry
                else route.reason
            ),
            route.shape,
        )
        for route in plan.routes
    )
    return NeuralOptimizerPlan(routes, plan.sign_stable)


def _route_name(route: Any) -> str:
    if hasattr(route, "name"):
        return str(route.name)
    return str(route.parameter.name)


def _route_family(route: Any) -> str:
    family = route.family
    return str(getattr(family, "value", family))


def build_auto_optimizer(module: Any, *, lr: float = 1.0e-3, sign_stable: bool = True, **kwargs: Any) -> Any:
    """Plan and build the default non-Adam-first optimizer for ``module``."""

    plan = plan_neural_optimizer(module, sign_stable=sign_stable)
    return build_routed_optimizer(module, plan, lr=lr, **kwargs)


def resolve_neural_optimizer(
    module: Any,
    optimizer: Any = None,
    *,
    lr: float = 1.0e-3,
    sign_stable: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Resolve automatic, named, planned, or callable neural optimizers with an audit receipt."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - Torch is optional
        raise ImportError("neural optimization requires PyTorch.") from error
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("module has no trainable parameters.")
    if optimizer is None or (isinstance(optimizer, str) and optimizer.lower() == "auto"):
        result = build_auto_optimizer(module, lr=lr, sign_stable=sign_stable)
        return result, {"name": "auto", "plan": result.optimizer_plan.as_dict()}
    if isinstance(optimizer, str):
        key = optimizer.lower().replace("-", "_")
        builders = {
            "rprop": lambda: torch.optim.Rprop(parameters, lr=lr),
            "adagrad": lambda: torch.optim.Adagrad(parameters, lr=lr),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
            "sgd_momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
            "adamw": lambda: torch.optim.AdamW(parameters, lr=lr),
            "adam": lambda: torch.optim.Adam(parameters, lr=lr),
        }
        if key not in builders:
            raise ValueError("unsupported neural optimizer %r." % optimizer)
        return builders[key](), {"name": key, "plan": None}
    if hasattr(optimizer, "routes"):
        result = build_routed_optimizer(module, optimizer, lr=lr)
        plan = optimizer.as_dict() if callable(getattr(optimizer, "as_dict", None)) else None
        return result, {"name": "routed_plan", "plan": plan}
    if callable(optimizer):
        return optimizer(parameters), {"name": "custom", "plan": None}
    raise TypeError("optimizer must be a name, optimizer plan, or callable.")


def build_routed_optimizer(
    module: Any,
    plan: Any,
    *,
    lr: float = 1.0e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    muon_backend: str = "newton_schulz",
    muon_steps: int = 7,
    fisher_provider: Callable[[str, Any], Any] | None = None,
    projection_provider: Callable[[str, Any], None] | None = None,
) -> Any:
    """Build a Torch optimizer from stable or experimental structurally compatible routes."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - Torch is optional
        raise ImportError("routed neural optimization requires PyTorch.") from error
    if lr <= 0.0 or eps <= 0.0 or weight_decay < 0.0:
        raise ValueError("optimizer lr/eps must be positive and weight_decay non-negative.")
    if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
        raise ValueError("optimizer betas must contain two values in [0, 1).")
    if precondition_frequency < 1:
        raise ValueError("precondition_frequency must be positive.")
    if muon_backend not in ("newton_schulz", "svd"):
        raise ValueError("muon_backend must be 'newton_schulz' or 'svd'.")
    if muon_steps < 1:
        raise ValueError("muon_steps must be positive.")

    named = dict(module.named_parameters())
    groups = []
    parameter_families: dict[int, str] = {}
    skipped = {"exact", "frozen", "discrete_search"}
    for route in plan.routes:
        name = _route_name(route)
        family = _route_family(route)
        resolved_name = name
        for prefix in ("module.", "_orig_mod.", "_orig_mod.module.", "module._orig_mod."):
            candidate = prefix + name
            if resolved_name not in named and candidate in named:
                resolved_name = candidate
        if resolved_name not in named:
            raise KeyError("optimizer plan parameter is absent from module: %s" % name)
        parameter = named[resolved_name]
        if family in skipped:
            continue
        existing = parameter_families.get(id(parameter))
        if existing is not None and existing != family:
            raise ValueError("shared parameter %s has conflicting optimizer routes." % name)
        parameter_families[id(parameter)] = family
        if family == "natural_gradient" and fisher_provider is None:
            raise ValueError("natural-gradient routes require fisher_provider.")
        if family == "proximal" and projection_provider is None:
            raise ValueError("proximal routes require projection_provider.")
        groups.append(
            {
                "params": [parameter],
                "name": name,
                "family": family,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
        )
    if not groups:
        raise ValueError("optimizer plan has no executable gradient-routed parameters.")
    parameters = [parameter for group in groups for parameter in group["params"]]
    families = {group["family"] for group in groups}
    if families == {"adamw"}:
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        optimizer.optimizer_plan = plan
        return optimizer
    if families == {"sgd_momentum"}:
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=betas[0], weight_decay=weight_decay)
        optimizer.optimizer_plan = plan
        return optimizer
    if families == {"adagrad"}:
        optimizer = torch.optim.Adagrad(parameters, lr=lr, eps=eps, weight_decay=weight_decay)
        optimizer.optimizer_plan = plan
        return optimizer
    if families == {"rprop"} and weight_decay == 0.0:
        optimizer = torch.optim.Rprop(parameters, lr=lr)
        optimizer.optimizer_plan = plan
        return optimizer

    class RoutedNeuralOptimizer(torch.optim.Optimizer):
        def __init__(self) -> None:
            super().__init__(groups, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
            self.optimizer_plan = plan

        @staticmethod
        def _adam_direction(state: dict[str, Any], gradient: Any, beta1: float, beta2: float, epsilon: float) -> Any:
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(gradient)
                state["exp_avg_sq"] = torch.zeros_like(gradient)
            state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            correction1 = 1.0 - beta1 ** state["step"]
            correction2 = 1.0 - beta2 ** state["step"]
            denominator = state["exp_avg_sq"].sqrt().div_(math.sqrt(correction2)).add_(epsilon)
            return state["exp_avg"] / correction1 / denominator

        @staticmethod
        def _momentum(state: dict[str, Any], gradient: Any, beta: float) -> Any:
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(gradient)
            state["momentum"].mul_(beta).add_(gradient)
            return state["momentum"]

        @staticmethod
        def _adagrad(state: dict[str, Any], gradient: Any, epsilon: float) -> Any:
            if "sum_squares" not in state:
                state["sum_squares"] = torch.zeros_like(gradient)
            state["sum_squares"].addcmul_(gradient, gradient)
            return gradient / state["sum_squares"].sqrt().add_(epsilon)

        @staticmethod
        def _rprop(state: dict[str, Any], gradient: Any, initial_step: float) -> Any:
            if "step_sizes" not in state:
                state["step_sizes"] = torch.full_like(gradient, initial_step)
                state["previous_gradient"] = torch.zeros_like(gradient)
            product = gradient * state["previous_gradient"]
            state["step_sizes"].mul_(torch.where(product > 0.0, 1.2, torch.where(product < 0.0, 0.5, 1.0))).clamp_(
                1.0e-6, 50.0
            )
            effective = torch.where(product < 0.0, torch.zeros_like(gradient), gradient)
            state["previous_gradient"].copy_(effective)
            return effective.sign() * state["step_sizes"]

        @staticmethod
        def _polar_direction(momentum: Any) -> Any:
            if momentum.ndim != 2:
                raise ValueError("Muon route requires a matrix parameter gradient.")
            value = momentum.float()
            if not torch.isfinite(value).all():
                raise ValueError("Muon route requires a finite matrix gradient.")
            if muon_backend == "svd":
                left, _, right = torch.linalg.svd(value, full_matrices=False)
                direction = left @ right
            else:
                transposed = value.shape[0] > value.shape[1]
                work = value.T if transposed else value
                work = work / work.norm().clamp_min(1.0e-7)
                for _ in range(muon_steps):
                    gram = work @ work.T
                    work = 1.5 * work - 0.5 * gram @ work
                direction = work.T if transposed else work
            direction = direction.to(dtype=momentum.dtype)
            return direction * math.sqrt(max(1.0, momentum.shape[0] / momentum.shape[1]))

        @staticmethod
        def _inverse_quarter(factor: Any, damping: float) -> Any:
            values, vectors = torch.linalg.eigh(0.5 * (factor + factor.T))
            powers = values.clamp_min(0.0).add(damping).pow(-0.25)
            return (vectors * powers.unsqueeze(0)) @ vectors.T

        def _kronecker_direction(
            self, state: dict[str, Any], gradient: Any, beta1: float, beta2: float, epsilon: float
        ) -> Any:
            if gradient.ndim != 2:
                raise ValueError("Kronecker route requires a matrix parameter gradient.")
            momentum = self._momentum(state, gradient, beta1)
            rows, columns = gradient.shape
            row_observation = gradient.float() @ gradient.float().T / max(columns, 1)
            column_observation = gradient.float().T @ gradient.float() / max(rows, 1)
            if "row_factor" not in state:
                state["row_factor"] = row_observation
                state["column_factor"] = column_observation
            else:
                state["row_factor"].mul_(beta2).add_(row_observation, alpha=1.0 - beta2)
                state["column_factor"].mul_(beta2).add_(column_observation, alpha=1.0 - beta2)
            if "row_inverse" not in state or state["step"] % precondition_frequency == 0:
                state["row_inverse"] = self._inverse_quarter(state["row_factor"], epsilon)
                state["column_inverse"] = self._inverse_quarter(state["column_factor"], epsilon)
            return (state["row_inverse"] @ momentum.float() @ state["column_inverse"]).to(dtype=gradient.dtype)

        @torch.no_grad()
        def step(self, closure: Callable[[], Any] | None = None) -> Any:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                family = group["family"]
                beta1, beta2 = group["betas"]
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    gradient = parameter.grad
                    if gradient.is_sparse:
                        raise ValueError("routed optimizer does not silently densify sparse gradients.")
                    state = self.state[parameter]
                    state["step"] = int(state.get("step", 0)) + 1
                    if family in ("adamw", "diagonal_adaptive", "low_rank_adaptive", "proximal"):
                        direction = self._adam_direction(state, gradient, beta1, beta2, group["eps"])
                    elif family == "adagrad":
                        direction = self._adagrad(state, gradient, group["eps"])
                    elif family == "rprop":
                        direction = self._rprop(state, gradient, group["lr"])
                    elif family == "sgd_momentum":
                        direction = self._momentum(state, gradient, beta1)
                    elif family == "muon":
                        direction = self._polar_direction(self._momentum(state, gradient, beta1))
                    elif family == "kronecker":
                        direction = self._kronecker_direction(state, gradient, beta1, beta2, group["eps"])
                    elif family == "natural_gradient":
                        fisher = torch.as_tensor(
                            fisher_provider(group["name"], parameter), device=gradient.device, dtype=torch.float32
                        )
                        flat = gradient.float().reshape(-1)
                        if fisher.shape != (flat.numel(), flat.numel()):
                            raise ValueError("fisher_provider returned the wrong shape for %s." % group["name"])
                        direction = torch.linalg.solve(
                            0.5 * (fisher + fisher.T) + group["eps"] * torch.eye(flat.numel(), device=flat.device),
                            flat,
                        ).reshape_as(gradient)
                        direction = direction.to(dtype=gradient.dtype)
                    else:
                        raise ValueError("unsupported executable optimizer family %s." % family)
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(direction, alpha=-1.0 if family == "rprop" else -group["lr"])
                    if family == "proximal":
                        projection_provider(group["name"], parameter)
            return loss

    return RoutedNeuralOptimizer()
