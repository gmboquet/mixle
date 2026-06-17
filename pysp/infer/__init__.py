"""Bring-your-own-target inference facade.

A small public surface so external consumers can run pysp's samplers / VI on an *arbitrary*
differentiable target without reaching into ``pysp.ppl`` internals:

* :func:`nuts` — fast (fused ``value_and_grad``) NUTS over a user log-target, multi-chain, with
  R-hat / ESS diagnostics returned alongside the draws.
* :func:`advi` — automatic-differentiation VI over a user *batched* torch target.
* :mod:`pysp.infer.diagnostics` (re-exported here as :func:`rhat` / :func:`ess`) — convergence
  diagnostics over plain ``(n_chains, n_draws, d)`` arrays.

These wrap :func:`pysp.utils.mcmc.nuts` and the ADVI core in :mod:`pysp.ppl.autograd`; the
samplers themselves stay model-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .diagnostics import ess, rhat

__all__ = ["NutsResult", "AdviResult", "nuts", "advi", "rhat", "ess"]


@dataclass(frozen=True)
class NutsResult:
    """Draws and diagnostics from a multi-chain :func:`nuts` run.

    ``samples`` is ``(chains * draws, d)`` pooled draws; ``chains`` is ``(n_chains, draws, d)``.
    """

    samples: np.ndarray
    chains: np.ndarray
    rhat: np.ndarray
    ess: np.ndarray
    num_target_evals: int
    step_size: float
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AdviResult:
    """Result of :func:`advi`: posterior draws plus the fitted variational parameters."""

    samples: np.ndarray
    mean: np.ndarray
    scale: np.ndarray


def nuts(
    value_and_grad: Callable[[np.ndarray], tuple[float, Any]],
    *,
    dim: int | None = None,
    init: Any = None,
    num_samples: int = 1000,
    warmup: int = 1000,
    chains: int = 1,
    mass: Any = 1.0,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    thin: int = 1,
    rng: np.random.RandomState | int | None = None,
) -> NutsResult:
    """No-U-Turn Sampler over an arbitrary differentiable log-target.

    Args:
        value_and_grad: fused callable ``theta -> (logp, grad)`` returning the (unnormalized) log
            target and its gradient in one shot. The fused contract halves forward passes and lets
            the sampler cache endpoint gradients across the leapfrog/tree.
        dim: parameter dimension. Required unless ``init`` is given.
        init: initial state, shape ``(dim,)`` or ``(chains, dim)`` for per-chain starts. Defaults
            to zeros. A 1-D ``init`` is reused (jittered after the first) across chains.
        num_samples: retained post-warmup draws per chain.
        warmup: step-size adaptation / burn-in iterations per chain.
        chains: number of independent chains (>= 2 to get a meaningful R-hat).
        mass, target_accept, max_tree_depth, thin: forwarded to the sampler.
        rng: seed / RandomState for reproducibility.

    Returns:
        :class:`NutsResult` with pooled ``samples`` ``(chains*draws, d)``, per-chain ``chains``
        ``(chains, draws, d)``, per-dimension ``rhat`` and ``ess``, the total target-evaluation
        count and the adapted ``step_size``.
    """
    from pysp.utils.mcmc import nuts as _nuts

    if dim is None and init is None:
        raise ValueError("pass dim= or init=.")
    rng = _as_rng(rng)
    inits = _chain_inits(init, dim, chains, rng)
    d = inits.shape[1]

    chain_arrays: list[np.ndarray] = []
    total_evals = 0
    last_step = float("nan")
    for c in range(chains):
        seed = int(rng.randint(1, 2**31 - 1))
        res = _nuts(
            value_and_grad=value_and_grad,
            initial=inits[c],
            num_samples=num_samples,
            warmup=warmup,
            mass=mass,
            target_accept=target_accept,
            max_tree_depth=max_tree_depth,
            thin=thin,
            rng=np.random.RandomState(seed),
        )
        arr = np.asarray(res.samples, dtype=float).reshape(len(res.samples), d)
        chain_arrays.append(arr)
        total_evals += int(getattr(res, "num_target_evals", 0))
        last_step = float(getattr(res, "step_size", float("nan")))

    n = min(a.shape[0] for a in chain_arrays)
    stacked = np.stack([a[:n] for a in chain_arrays], axis=0)  # (chains, n, d)
    pooled = stacked.reshape(-1, d)
    rh = rhat(stacked) if chains >= 2 else np.full(d, np.nan)
    es = ess(stacked)
    return NutsResult(
        samples=pooled,
        chains=stacked,
        rhat=rh,
        ess=es,
        num_target_evals=total_evals,
        step_size=last_step,
    )


def advi(
    target_batch: Callable[[Any], Any],
    u0,
    s0,
    *,
    samples: int = 1000,
    mc: int = 16,
    steps: int = 2000,
    lr: float = 0.05,
    batch_size: int | None = None,
    family: str = "meanfield",
    alpha: float = 1.0,
    rng: np.random.RandomState | int | None = None,
) -> AdviResult:
    """Automatic-differentiation VI over a user *batched* torch target.

    Args:
        target_batch: ``target_batch(U: Tensor(B, d)) -> Tensor(B,)`` returning the (unnormalized)
            joint log-target for each of ``B`` parameter draws. The caller owns any data
            minibatching/rescaling inside this callable; ``batch_size`` here is unused unless the
            caller wires it in (kept for signature parity with the internal ADVI).
        u0, s0: initial variational mean and (marginal) scale in the unconstrained space.
        samples: number of posterior draws to return.
        mc, steps, lr: Monte-Carlo samples per step, optimizer steps, Adam learning rate.
        family: ``'meanfield'`` (diagonal) or ``'fullrank'`` (Cholesky covariance).
        alpha: Renyi/tilted objective exponent (``1.0`` = standard KL-ELBO).
        rng: seed / RandomState.

    Returns:
        :class:`AdviResult` with ``samples`` ``(samples, d)`` drawn from the fitted Gaussian q (in
        the *same* space ``target_batch`` consumes), plus the fitted ``mean`` and ``scale``.
    """
    from pysp.ppl.autograd import _advi_optimize, torch_available

    if not torch_available():
        raise RuntimeError("pysp.infer.advi requires PyTorch.")
    import torch  # noqa: F401

    rng = _as_rng(rng)
    mean_np, scale_np, U = _advi_optimize(
        torch,
        target_batch,
        u0,
        s0,
        samples=samples,
        mc=mc,
        steps=steps,
        lr=lr,
        rng=rng,
        family=family,
        alpha=alpha,
    )
    return AdviResult(samples=U, mean=mean_np, scale=scale_np)


def _as_rng(rng: np.random.RandomState | int | None) -> np.random.RandomState:
    if rng is None:
        return np.random.RandomState()
    if isinstance(rng, (int, np.integer)):
        return np.random.RandomState(int(rng))
    return rng


def _chain_inits(init: Any, dim: int | None, chains: int, rng: np.random.RandomState) -> np.ndarray:
    """Build a ``(chains, d)`` array of initial states from ``init`` / ``dim``."""
    if init is None:
        return np.zeros((chains, int(dim)), dtype=float)
    arr = np.asarray(init, dtype=float)
    if arr.ndim == 1:
        d = arr.shape[0]
        out = np.tile(arr, (chains, 1))
        if chains > 1:  # over-disperse the extra chains so R-hat is meaningful
            out[1:] = out[1:] + rng.standard_normal((chains - 1, d))
        return out
    if arr.ndim == 2:
        if arr.shape[0] != chains:
            raise ValueError("init with shape (n, d) must have n == chains.")
        return arr
    raise ValueError("init must be 1-D (d,) or 2-D (chains, d).")
