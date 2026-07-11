"""Executable PyTorch adapter for statistically typed per-parameter optimizer routes."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from mixle.experimental.typed_runtime.geometry import OptimizerFamily, OptimizerPlan

FisherProvider = Callable[[str, Any], Any]
ProjectionProvider = Callable[[str, Any], None]


def build_routed_torch_optimizer(
    module: Any,
    plan: OptimizerPlan,
    *,
    lr: float = 1.0e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    muon_backend: str = "newton_schulz",
    muon_steps: int = 7,
    fisher_provider: FisherProvider | None = None,
    projection_provider: ProjectionProvider | None = None,
) -> Any:
    """Build a real ``torch.optim.Optimizer`` executing a typed geometry plan.

    Muon routes use a fixed-step Newton-Schulz polar approximation by default;
    ``muon_backend="svd"`` retains the exact polar factor as an audit reference.
    Kronecker routes maintain row/column second-moment factors and refresh inverse
    fourth-roots at ``precondition_frequency``.
    """

    try:
        import torch
    except ImportError as error:
        raise ImportError("build_routed_torch_optimizer requires PyTorch.") from error
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
    parameter_families: dict[int, OptimizerFamily] = {}
    for route in plan.routes:
        if route.parameter.name not in named:
            raise KeyError("optimizer plan parameter is absent from module: %s" % route.parameter.name)
        parameter = named[route.parameter.name]
        if route.family in (OptimizerFamily.EXACT, OptimizerFamily.FROZEN, OptimizerFamily.DISCRETE_SEARCH):
            continue
        existing = parameter_families.get(id(parameter))
        if existing is not None and existing is not route.family:
            raise ValueError("shared parameter %s has conflicting optimizer routes." % route.parameter.name)
        parameter_families[id(parameter)] = route.family
        if route.family is OptimizerFamily.NATURAL_GRADIENT and fisher_provider is None:
            raise ValueError("natural-gradient routes require fisher_provider.")
        if route.family is OptimizerFamily.PROXIMAL and projection_provider is None:
            raise ValueError("proximal routes require projection_provider.")
        groups.append(
            {
                "params": [parameter],
                "name": route.parameter.name,
                "family": route.family.value,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
        )
    if not groups:
        raise ValueError("optimizer plan has no executable gradient-routed parameters.")

    class RoutedTorchOptimizer(torch.optim.Optimizer):
        def __init__(self) -> None:
            super().__init__(groups, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
            self.optimizer_plan = plan

        @staticmethod
        def _adam_direction(state: dict[str, Any], gradient: Any, beta1: float, beta2: float, eps: float) -> Any:
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(gradient)
                state["exp_avg_sq"] = torch.zeros_like(gradient)
            state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            step = state["step"]
            correction1 = 1.0 - beta1**step
            correction2 = 1.0 - beta2**step
            denominator = state["exp_avg_sq"].sqrt().div_(math.sqrt(correction2)).add_(eps)
            return state["exp_avg"] / correction1 / denominator

        @staticmethod
        def _momentum(state: dict[str, Any], gradient: Any, beta: float) -> Any:
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(gradient)
            state["momentum"].mul_(beta).add_(gradient, alpha=1.0 - beta)
            return state["momentum"]

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
                # The normalized singular values lie in (0, 1], where this canonical polar
                # iteration converges to one. Fixed work makes it suitable for compilation and
                # sharding, while the exact-SVD backend remains available for audits.
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
            self,
            state: dict[str, Any],
            gradient: Any,
            beta1: float,
            beta2: float,
            eps: float,
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
                state["row_inverse"] = self._inverse_quarter(state["row_factor"], eps)
                state["column_inverse"] = self._inverse_quarter(state["column_factor"], eps)
            direction = state["row_inverse"] @ momentum.float() @ state["column_inverse"]
            return direction.to(dtype=gradient.dtype)

        @torch.no_grad()
        def step(self, closure: Callable[[], Any] | None = None) -> Any:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                family = OptimizerFamily(group["family"])
                beta1, beta2 = group["betas"]
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    gradient = parameter.grad
                    if gradient.is_sparse:
                        raise ValueError("routed optimizer does not silently densify sparse gradients.")
                    state = self.state[parameter]
                    state["step"] = int(state.get("step", 0)) + 1
                    if family in (
                        OptimizerFamily.ADAMW,
                        OptimizerFamily.DIAGONAL_ADAPTIVE,
                        OptimizerFamily.LOW_RANK_ADAPTIVE,
                        OptimizerFamily.PROXIMAL,
                    ):
                        direction = self._adam_direction(state, gradient, beta1, beta2, group["eps"])
                    elif family is OptimizerFamily.SGD_MOMENTUM:
                        direction = self._momentum(state, gradient, beta1)
                    elif family is OptimizerFamily.MUON:
                        direction = self._polar_direction(self._momentum(state, gradient, beta1))
                    elif family is OptimizerFamily.KRONECKER:
                        direction = self._kronecker_direction(state, gradient, beta1, beta2, group["eps"])
                    elif family is OptimizerFamily.NATURAL_GRADIENT:
                        fisher = fisher_provider(group["name"], parameter)
                        fisher = torch.as_tensor(fisher, device=gradient.device, dtype=torch.float32)
                        flat = gradient.float().reshape(-1)
                        if fisher.shape != (flat.numel(), flat.numel()):
                            raise ValueError("fisher_provider returned the wrong shape for %s." % group["name"])
                        direction = (
                            torch.linalg.solve(
                                0.5 * (fisher + fisher.T) + group["eps"] * torch.eye(flat.numel(), device=flat.device),
                                flat,
                            )
                            .reshape_as(gradient)
                            .to(dtype=gradient.dtype)
                        )
                    else:
                        raise ValueError("unsupported executable optimizer family %s." % family.value)
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(direction, alpha=-group["lr"])
                    if family is OptimizerFamily.PROXIMAL:
                        projection_provider(group["name"], parameter)
            return loss

    return RoutedTorchOptimizer()


__all__ = ["FisherProvider", "ProjectionProvider", "build_routed_torch_optimizer"]
