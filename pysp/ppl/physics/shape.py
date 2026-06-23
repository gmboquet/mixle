"""Geometry by level sets: shape optimization and inverse shape inference (phase 3).

A shape is represented implicitly by a level-set field ``phi``: the interior is ``{phi > 0}`` and the
boundary is ``{phi = 0}``. A material property is read off it with a smoothed Heaviside,
``material = outside + (inside - outside) * H(phi)`` (``ops.level_set``), which is differentiable in
``phi`` -- so the same adjoint machinery that recovers a coefficient field recovers a *shape*.

Two uses, both reusing the PDE forward operators and adjoints:

- Inverse shape inference (Bayesian): make ``phi`` a ``GP`` field and a forward model read its level-set
  material; ``joint([...]).fit(...)`` returns a posterior over ``phi``, hence over the shape and its
  uncertainty. No new fitting machinery -- it is a field inverse problem with a Heaviside link.
- Shape optimization (deterministic design): :func:`shape_optimize` minimizes a design objective (drag,
  compliance, misfit-to-target) over ``phi`` by L-BFGS on the adjoint gradients, with an optional
  smoothness prior to keep the boundary well-behaved.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def level_set_material(phi, inside: float, outside: float, *, eps: float = 0.1) -> np.ndarray:
    """The (numpy) material field of a level set, for post-processing a recovered ``phi``: ``outside +
    (inside - outside) * 0.5 (1 + tanh(phi / eps))``. The shape interior is ``phi > 0``."""
    phi = np.asarray(phi, dtype=float)
    return outside + (inside - outside) * 0.5 * (1.0 + np.tanh(phi / eps))


def shape_optimize(
    phi0: np.ndarray,
    objective: Callable,
    *,
    prior_precision: np.ndarray | None = None,
    steps: int = 200,
    lr: float = 0.4,
) -> np.ndarray:
    """Minimize a design ``objective(phi, ops) -> scalar`` over a level-set field by L-BFGS.

    ``objective`` builds the design cost from ``phi`` using the ``ops`` namespace (``ops.level_set``,
    ``ops.sparse_solve``, ``ops.matvec``, ...), so its gradient flows through the PDE adjoint.
    ``prior_precision`` (n x n), if given, adds a smoothness prior ``0.5 phi^T P phi`` (e.g. a
    :class:`~pysp.ppl.RandomWalk` precision) that regularizes the boundary. Returns the optimized ``phi``.
    """
    import torch

    from pysp.ppl.physics.ops import make_ops

    ops = make_ops()
    phi = torch.tensor(np.asarray(phi0, dtype=float), requires_grad=True)
    P = None if prior_precision is None else torch.as_tensor(np.asarray(prior_precision, dtype=float))
    opt = torch.optim.LBFGS([phi], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective(phi, ops)
        if P is not None:
            loss = loss + 0.5 * (phi @ (P @ phi))
        loss.backward()
        return loss

    opt.step(closure)
    return phi.detach().numpy()
