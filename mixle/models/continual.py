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

from typing import Any

import numpy as np


def snapshot(leaf_or_module: Any) -> list:
    """Detached clones of the module's parameters -- the anchor ``theta*`` for an EWC penalty."""
    module = getattr(leaf_or_module, "module", leaf_or_module)
    return [p.detach().clone() for p in module.parameters()]


def fisher_diagonal(leaf: Any, x: Any, y: Any, *, samples: int = 512, device: str = "cpu", seed: int = 0) -> list:
    """Diagonal empirical Fisher of a classification leaf's module on ``(x, y)``: mean of ``(d log p(y|x)/dtheta)^2``."""
    import torch

    module = leaf.module.to(device)
    x = np.asarray(x, dtype="float32")
    y = np.asarray(y, dtype=int)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), min(int(samples), len(x)), replace=False)
    fisher = [torch.zeros_like(p) for p in module.parameters()]
    for i in idx:
        module.zero_grad()
        logits = module(torch.as_tensor(x[i : i + 1]).to(device))
        logp = torch.log_softmax(logits, dim=1)[0, int(y[i])]
        logp.backward()
        for f, p in zip(fisher, module.parameters()):
            if p.grad is not None:
                f += p.grad.detach() ** 2
    n = len(idx)
    return [(f / n).cpu() for f in fisher]


def ewc(anchor: list, fisher: list, lam: float = 1.0) -> tuple:
    """Bundle ``(anchor, fisher, lambda)`` for ``.fit(..., ewc=...)`` (the EWC anti-forgetting penalty)."""
    return (anchor, fisher, float(lam))
