"""Autograd gradients for MCMC log-targets (optional Torch backend).

Hamiltonian samplers (HMC, MALA, NUTS) need the gradient of the log-target. The default
finite-difference gradient costs ``O(d)`` full target evaluations per step -- the documented
bottleneck for parameter-posterior HMC over higher-dimensional models. When the target is
expressed in Torch, autograd returns the *exact* gradient in a single backward pass,
independent of dimension. This module provides that bridge.

It is entirely optional: import-light, and :func:`torch_gradient` raises a clear error only
if called without Torch. Callers that want a guaranteed-available gradient can fall back to
the finite-difference path in ``parameter_bridge``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def torch_available() -> bool:
    """Return True if Torch can be imported (autograd gradients are available)."""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def torch_gradient(log_target_torch: Callable[[Any], Any], dtype: str = "float64") -> Callable[[Any], Any]:
    """Build an exact ``grad_log_target`` from a Torch-valued log-target via autograd.

    Args:
        log_target_torch: Callable mapping a 1-D Torch tensor of parameters to a scalar Torch
            tensor (the unnormalized log target). Differentiation flows through whatever Torch
            ops it uses, so the gradient is exact -- one backward pass regardless of dimension.
        dtype: Torch float dtype for the parameter tensor ("float64" by default for MCMC
            numerics; use "float32" to match a single-precision model).

    Returns:
        ``grad(x) -> gradient`` accepting and returning numpy (a float for scalar ``x``,
        otherwise an array shaped like ``x``). Suitable as the ``grad_log_target`` argument to
        :func:`hamiltonian_monte_carlo`, :func:`nuts`, or a Langevin proposal.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover - exercised only without Torch
        raise RuntimeError("torch_gradient requires PyTorch; install torch or use the finite-difference path.") from e

    tdtype = getattr(torch, dtype)

    def grad(x: Any) -> Any:
        scalar = np.isscalar(x) or np.ndim(x) == 0
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        t = torch.tensor(arr, dtype=tdtype, requires_grad=True)
        val = log_target_torch(t if not scalar else t[0])
        if not torch.is_tensor(val) or val.ndim != 0:
            val = val.reshape(()) if torch.is_tensor(val) else torch.as_tensor(val, dtype=tdtype)
        (g,) = torch.autograd.grad(val, t)
        gn = np.asarray(g.detach().cpu().numpy(), dtype=float)
        if not np.all(np.isfinite(gn)):
            raise ValueError("torch_gradient produced non-finite values; check the log-target.")
        return float(gn[0]) if scalar else gn

    return grad


def value_and_torch_gradient(
    log_target_torch: Callable[[Any], Any], dtype: str = "float64"
) -> Callable[[Any], tuple[float, Any]]:
    """Like :func:`torch_gradient` but returns ``(value, gradient)`` in one backward pass.

    Useful for samplers that need both the log-target and its gradient at the same point
    (HMC/NUTS leapfrog), avoiding a redundant forward evaluation.
    """
    grad_only = torch_gradient(log_target_torch, dtype=dtype)
    import torch

    tdtype = getattr(torch, dtype)

    def value_and_grad(x: Any) -> tuple[float, Any]:
        scalar = np.isscalar(x) or np.ndim(x) == 0
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        t = torch.tensor(arr, dtype=tdtype)
        with torch.no_grad():
            val = log_target_torch(t if not scalar else t[0])
            value = float(val.reshape(()).item()) if torch.is_tensor(val) else float(val)
        return value, grad_only(x)

    return value_and_grad
