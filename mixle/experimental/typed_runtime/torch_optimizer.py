"""Executable PyTorch adapter for statistically typed per-parameter optimizer routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mixle.experimental.typed_runtime.geometry import OptimizerPlan
from mixle.models.optimizer_routing import build_routed_optimizer

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

    return build_routed_optimizer(
        module,
        plan,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        precondition_frequency=precondition_frequency,
        muon_backend=muon_backend,
        muon_steps=muon_steps,
        fisher_provider=fisher_provider,
        projection_provider=projection_provider,
    )


__all__ = ["FisherProvider", "ProjectionProvider", "build_routed_torch_optimizer"]
