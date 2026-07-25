"""Continual / multi-stage fine-tuning helpers: parameter snapshot + diagonal Fisher + EWC for neural leaves.

Continued pretraining (CPT) without catastrophic forgetting = continue the same module on new data plus an EWC
penalty ``lambda * sum_i F_i (theta_i - theta*_i)^2`` anchoring to the pretrained params ``theta*`` weighted by
the diagonal Fisher ``F`` (how much each parameter mattered for the old task). The Fisher is the same curvature
mixle uses for posterior approximation; here it is per-parameter importance for anti-forgetting. Use it as a
declarative stage in the pipeline::

    pre  = Categorical(logits=Net(out=K)).fit(yA, given={"x": XA})
    F    = fisher_diagonal(pre.dist, XA, yA)
    cpt  = Categorical(logits=Net(out=K)).fit(yB, given={"x": XB}, init=pre, ewc=ewc(snapshot(pre.dist), F, lam=200))
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


class ParameterBundle(list):
    """Tensor list carrying the ordered parameter identity manifest."""

    def __init__(self, values: list[Any], names: tuple[str, ...]) -> None:
        super().__init__(values)
        self.names = names
        if len(self) != len(self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("parameter bundle requires one unique name per tensor.")


def snapshot(leaf_or_module: Any) -> ParameterBundle:
    """Detached clones of the module's parameters -- the anchor ``theta*`` for an EWC penalty."""
    module = getattr(leaf_or_module, "module", leaf_or_module)
    named = list(module.named_parameters())
    if not named:
        raise ValueError("cannot snapshot a module with no parameters.")
    return ParameterBundle([parameter.detach().clone() for _, parameter in named], tuple(name for name, _ in named))


def fisher_diagonal(
    leaf: Any,
    x: Any,
    y: Any,
    *,
    samples: int = 512,
    device: str = "cpu",
    seed: int = 0,
) -> ParameterBundle:
    """Diagonal empirical Fisher computed on an isolated evaluation-mode copy."""
    import torch

    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or int(samples) <= 0:
        raise ValueError("samples must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    x = np.asarray(x, dtype="float32")
    y_raw = np.asarray(y)
    if x.ndim < 2 or y_raw.ndim != 1 or not len(x) or len(x) != len(y_raw):
        raise ValueError("x and y must be non-empty aligned feature and label arrays.")
    if not np.isfinite(x).all() or not np.isfinite(y_raw).all():
        raise ValueError("x and y must contain only finite values.")
    if y_raw.dtype.kind not in "iu" and not np.all(y_raw == np.floor(y_raw)):
        raise ValueError("classification labels must be integers.")
    y = y_raw.astype(int)

    source_module = getattr(leaf, "module", leaf)
    source_named = list(source_module.named_parameters())
    if not source_named:
        raise ValueError("cannot estimate Fisher information for a module with no parameters.")
    module = copy.deepcopy(source_module).to(device)
    module.eval()
    named = list(module.named_parameters())
    if tuple(name for name, _ in named) != tuple(name for name, _ in source_named):
        raise RuntimeError("isolated module copy changed the parameter identity manifest.")
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(len(x), min(int(samples), len(x)), replace=False)
    fisher = [torch.zeros_like(parameter) for _, parameter in named]
    for i in idx:
        module.zero_grad(set_to_none=True)
        logits = module(torch.as_tensor(x[i : i + 1]).to(device))
        if logits.ndim != 2 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
            raise ValueError("classification module must return one finite two-dimensional logits row.")
        if y[i] < 0 or y[i] >= logits.shape[1]:
            raise ValueError(f"classification label {y[i]} is outside logits width {logits.shape[1]}.")
        logp = torch.log_softmax(logits, dim=1)[0, int(y[i])]
        logp.backward()
        for fisher_tensor, (_, parameter) in zip(fisher, named):
            if parameter.grad is not None:
                fisher_tensor += parameter.grad.detach() ** 2
    n = len(idx)
    result = [
        (value / n).detach().to(device=source_parameter.device, dtype=source_parameter.dtype)
        for value, (_, source_parameter) in zip(fisher, source_named)
    ]
    if any(not torch.isfinite(value).all() for value in result):
        raise ValueError("Fisher estimation produced non-finite values.")
    return ParameterBundle(result, tuple(name for name, _ in named))


def ewc(anchor: ParameterBundle, fisher: ParameterBundle, lam: float = 1.0) -> tuple:
    """Bundle ``(anchor, fisher, lambda)`` for ``.fit(..., ewc=...)`` (the EWC anti-forgetting penalty)."""
    import torch

    if not isinstance(anchor, ParameterBundle) or not isinstance(fisher, ParameterBundle):
        raise TypeError("anchor and fisher must be ParameterBundle values with identity manifests.")
    if anchor.names != fisher.names or len(anchor) != len(fisher) or not anchor:
        raise ValueError("anchor and Fisher parameter identity manifests must match exactly.")
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("EWC penalty lambda must be finite and non-negative.")
    for name, anchor_tensor, fisher_tensor in zip(anchor.names, anchor, fisher):
        if not isinstance(anchor_tensor, torch.Tensor) or not isinstance(fisher_tensor, torch.Tensor):
            raise TypeError(f"EWC parameter {name!r} must use tensor anchors and Fisher values.")
        if anchor_tensor.shape != fisher_tensor.shape:
            raise ValueError(f"EWC parameter {name!r} shape mismatch: {anchor_tensor.shape} vs {fisher_tensor.shape}.")
        if anchor_tensor.device != fisher_tensor.device or anchor_tensor.dtype != fisher_tensor.dtype:
            raise ValueError(f"EWC parameter {name!r} anchor and Fisher device/dtype must match.")
        if not torch.isfinite(anchor_tensor).all() or not torch.isfinite(fisher_tensor).all():
            raise ValueError(f"EWC parameter {name!r} contains non-finite values.")
        if torch.any(fisher_tensor < 0.0):
            raise ValueError(f"EWC Fisher values for parameter {name!r} must be non-negative.")
    return (anchor, fisher, float(lam))
