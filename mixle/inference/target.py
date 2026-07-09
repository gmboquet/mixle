"""Bring-your-own-target, engine-agnostic inference facade.

A small public surface so external consumers can run mixle's samplers / VI on an *arbitrary*
differentiable target without reaching into ``mixle.ppl`` internals — and run it on whichever
**engine** fits their hardware and target:

* :func:`nuts` — multi-chain No-U-Turn Sampler with R-hat / ESS diagnostics, dispatched to a
  registered backend (``numpy`` / ``numba`` / ``torch`` / ``jax``) selected by ``backend="auto"``.
  Every backend returns the **same** :class:`NutsResult`, so downstream code is backend-blind.
* :func:`nuts_torch` — the torch-native NUTS, kept as a direct entry point (also the ``"torch"``
  backend).
* :func:`advi` — automatic-differentiation VI over a user *batched* torch target.
* :func:`available_backends` — the installed inference engines.
* :mod:`mixle.inference.diagnostics` (re-exported as :func:`rhat` / :func:`ess`) — convergence
  diagnostics over plain ``(n_chains, n_draws, d)`` arrays.

The backend registry lives in :mod:`mixle.inference.backends` (`register, don't branch`); each backend
self-registers at import, behind a try/except so a missing optional engine never breaks
``import mixle.inference``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backends import (
    InferenceBackend,
    available_backends,
    get_inference_backend,
    register_inference_backend,
    select_backend,
)
from .diagnostics import (
    ess,
    ess_bulk,
    ess_tail,
    folded_split_rhat,
    geweke_z,
    mcmc_summary,
    mcse_mean,
    rhat,
    rhat_max,
    split_rhat,
)

__all__ = [
    "NutsResult",
    "AdviResult",
    "nuts",
    "nuts_torch",
    "advi",
    "rhat",
    "ess",
    "split_rhat",
    "folded_split_rhat",
    "rhat_max",
    "mcse_mean",
    "mcmc_summary",
    "ess_bulk",
    "ess_tail",
    "geweke_z",
    "available_backends",
    "InferenceBackend",
    "register_inference_backend",
]


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
    objective: float | None = None


def nuts(
    target: Callable[..., Any],
    *,
    backend: str = "auto",
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
    parallel: bool | str | None = None,
    **backend_kwargs: Any,
) -> NutsResult:
    """No-U-Turn Sampler over an arbitrary log-target, dispatched to a registered engine.

    The ``target`` contract depends on the backend (the kinds cannot be auto-converted across
    autodiff systems):

    * ``numpy`` / ``numba``: a fused ``value_and_grad(theta) -> (logp, grad)`` (njit-jitted for
      ``numba``). The caller supplies the (analytic) gradient.
    * ``torch`` / ``jax``: a scalar ``logp(theta)``; the backend builds ``value_and_grad`` by
      autodiff (``torch.func`` / NumPyro).

    Args:
        target: the log-target, per the contract above.
        backend: ``"auto"`` (default), or one of :func:`available_backends`. ``"auto"`` honors an
            explicit choice elsewhere, otherwise prefers the always-present ``numpy`` path for a
            plain numpy ``value_and_grad``; pass ``backend="numba"`` to run an ``@njit`` target,
            ``"jax"`` / ``"torch"`` for an autodiff scalar ``logp``.
        dim: parameter dimension. Required unless ``init`` is given.
        init: initial state, shape ``(dim,)`` or ``(chains, dim)`` for per-chain starts. Defaults
            to zeros. A 1-D ``init`` is reused (jittered after the first) across chains.
        num_samples: retained post-warmup draws per chain.
        warmup: step-size adaptation / burn-in iterations per chain.
        chains: number of independent chains (>= 2 to get a meaningful R-hat).
        mass, target_accept, max_tree_depth, thin: forwarded to the sampler.
        rng: seed / RandomState for reproducibility.
        parallel: run the ``chains`` independent chains concurrently. ``None``/``False`` (default)
            runs them in the backend's usual single call; ``"thread"`` uses a thread pool (real
            speedup for the ``numba``/``torch`` backends, whose inner loops release the GIL);
            ``True`` or ``"process"`` uses a process pool (real speedup for the pure-``numpy``
            backend; requires a picklable ``target``). Ignored for the ``jax`` backend, which
            already vectorizes chains internally. Each chain still gets an independent seed, so
            results are valid multi-chain draws (R-hat / ESS across chains).
        **backend_kwargs: forwarded to the chosen backend (e.g. ``compile=``, ``device=`` for
            ``torch``; ``chain_method=`` for ``jax``).

    Returns:
        :class:`NutsResult` with pooled ``samples`` ``(chains*draws, d)``, per-chain ``chains``
        ``(chains, draws, d)``, per-dimension ``rhat`` and ``ess``, the total target-evaluation
        count, the adapted ``step_size``, and ``extra={"backend": name}``.
    """
    if dim is None and init is None:
        raise ValueError("pass dim= or init=.")
    from .backends import _dispatch_target_kind

    kind_hint = _dispatch_target_kind(target, backend_kwargs.pop("target_kind", None))
    name = select_backend(backend, target=kind_hint)
    b = get_inference_backend(name)
    common = dict(
        num_samples=num_samples,
        warmup=warmup,
        mass=mass,
        target_accept=target_accept,
        max_tree_depth=max_tree_depth,
        thin=thin,
    )
    # jax (NumPyro) already vectorizes chains internally, so facade-level parallelism is wasteful.
    if parallel and chains > 1 and name != "jax":
        return _parallel_chains(name, target, dim, init, chains, rng, common, backend_kwargs, parallel)
    return b.nuts(target, dim=dim, init=init, chains=chains, rng=rng, **common, **backend_kwargs)


def _single_chain(name: str, target: Any, init_row: np.ndarray, seed: int, common: dict, backend_kwargs: dict):
    """Run one chain on backend ``name`` and return (samples (n,d), num_target_evals, step_size).

    Module-level so it is picklable for the process-pool path.
    """
    res = get_inference_backend(name).nuts(
        target,
        dim=int(init_row.shape[0]),
        init=init_row,
        chains=1,
        rng=np.random.RandomState(int(seed)),
        **common,
        **backend_kwargs,
    )
    d = int(np.asarray(res.samples).shape[1]) if np.asarray(res.samples).ndim == 2 else int(init_row.shape[0])
    samples = np.asarray(res.samples, dtype=float).reshape(-1, d)
    return samples, int(getattr(res, "num_target_evals", 0)), float(getattr(res, "step_size", float("nan")))


def _parallel_chains(name, target, dim, init, chains, rng, common, backend_kwargs, parallel) -> NutsResult:
    """Run ``chains`` independent single-chain runs concurrently and pool them into a NutsResult."""
    rng = _as_rng(rng)
    inits = _chain_inits(init, dim, chains, rng)
    d = inits.shape[1]
    seeds = [int(rng.randint(1, 2**31 - 1)) for _ in range(chains)]
    args = [(name, target, inits[c], seeds[c], common, backend_kwargs) for c in range(chains)]
    mode = "process" if parallel is True else str(parallel)
    if mode == "process":
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=chains) as ex:
            results = [f.result() for f in [ex.submit(_single_chain, *a) for a in args]]
    elif mode == "thread":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=chains) as ex:
            results = list(ex.map(lambda a: _single_chain(*a), args))
    else:
        raise ValueError("parallel must be None/False, True, 'process', or 'thread'.")
    chain_arrays = [r[0] for r in results]
    total_evals = sum(r[1] for r in results)
    last_step = next((r[2] for r in reversed(results) if np.isfinite(r[2])), float("nan"))
    return _pool_chains(chain_arrays, d, chains, total_evals, last_step, backend=name)


# --------------------------------------------------------------------------------------------
# numpy backend (always available): wraps the existing fused-value_and_grad sampler.
# --------------------------------------------------------------------------------------------
def _nuts_numpy(
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
    from mixle.inference.mcmc import nuts as _nuts

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
        chain_arrays.append(np.asarray(res.samples, dtype=float).reshape(len(res.samples), d))
        total_evals += int(getattr(res, "num_target_evals", 0))
        last_step = float(getattr(res, "step_size", float("nan")))
    return _pool_chains(chain_arrays, d, chains, total_evals, last_step, backend="numpy")


# --------------------------------------------------------------------------------------------
# numba backend: njit sampler over an @njit value_and_grad (analytic gradient, no autodiff).
# --------------------------------------------------------------------------------------------
def _nuts_numba(
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
    from mixle.inference.mcmc.nuts_numba import nuts_numba as _nuts_numba_core

    rng = _as_rng(rng)
    inits = _chain_inits(init, dim, chains, rng)
    d = inits.shape[1]
    chain_arrays: list[np.ndarray] = []
    total_evals = 0
    last_step = float("nan")
    for c in range(chains):
        seed = int(rng.randint(1, 2**31 - 1))
        res = _nuts_numba_core(
            value_and_grad,
            initial=inits[c],
            num_samples=num_samples,
            warmup=warmup,
            mass=mass,
            target_accept=target_accept,
            max_tree_depth=max_tree_depth,
            thin=thin,
            seed=seed,
        )
        chain_arrays.append(np.asarray(res.samples, dtype=float).reshape(len(res.samples), d))
        total_evals += int(getattr(res, "num_target_evals", 0))
        last_step = float(getattr(res, "step_size", float("nan")))
    return _pool_chains(chain_arrays, d, chains, total_evals, last_step, backend="numba")


# --------------------------------------------------------------------------------------------
# torch backend: torch-native NUTS over a scalar logp (autodiff; GPU / large targets).
# --------------------------------------------------------------------------------------------
def nuts_torch(
    logp: Callable[[Any], Any],
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
    compile: bool = True,
    dtype: Any = None,
    device: Any = None,
) -> NutsResult:
    """Torch-native NUTS over a torch scalar ``logp(theta) -> Tensor[()]`` (GPU / large autodiff targets).

    The whole leapfrog/tree runs in torch tensors with a ``torch.compile``d ``value_and_grad`` (no
    numpy round-trip / graph re-trace per gradient). Same multi-chain interface and
    :class:`NutsResult` as :func:`nuts`. Also registered as the ``"torch"`` backend.

    **Performance note (be deliberate about when to use this):** on CPU this is typically *slower*
    than the numpy :func:`nuts` (per-op torch dispatch + host syncs in the tree dominate when the
    target is low-cost). Its value is **GPU** (``device=``) and large autodiff targets. For CPU work
    prefer the numpy backend, numba (analytic gradient), or the jax/NumPyro backend (XLA,
    vectorized multi-chain). Chains run in a Python loop (per-chain latency).
    """
    from mixle.inference.mcmc.nuts_torch import nuts_torch as _nuts_torch

    if dim is None and init is None:
        raise ValueError("pass dim= or init=.")
    rng = _as_rng(rng)
    inits = _chain_inits(init, dim, chains, rng)
    d = inits.shape[1]
    chain_arrays: list[np.ndarray] = []
    total_evals = 0
    last_step = float("nan")
    compiled_any = False
    for c in range(chains):
        seed = int(rng.randint(1, 2**31 - 1))
        res = _nuts_torch(
            logp,
            initial=inits[c],
            num_samples=num_samples,
            warmup=warmup,
            mass=mass,
            target_accept=target_accept,
            max_tree_depth=max_tree_depth,
            thin=thin,
            seed=seed,
            compile=compile,
            dtype=dtype,
            device=device,
        )
        chain_arrays.append(np.asarray(res.samples, dtype=float).reshape(len(res.samples), d))
        total_evals += int(getattr(res, "num_target_evals", 0))
        last_step = float(getattr(res, "step_size", float("nan")))
        compiled_any = compiled_any or bool(getattr(res, "compiled", False))
    return _pool_chains(chain_arrays, d, chains, total_evals, last_step, backend="torch", compiled=compiled_any)


# --------------------------------------------------------------------------------------------
# jax / NumPyro backend: wrap NumPyro's NUTS via potential_fn (autodiff + XLA + vectorized chains).
# --------------------------------------------------------------------------------------------
def _nuts_jax(
    logp: Callable[[Any], Any],
    *,
    dim: int | None = None,
    init: Any = None,
    num_samples: int = 1000,
    warmup: int = 1000,
    chains: int = 1,
    mass: Any = 1.0,  # noqa: ARG001 — NumPyro adapts a (dense/diagonal) mass matrix in warmup itself
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    thin: int = 1,  # noqa: ARG001 — thinning handled by num_samples; kept for signature parity
    rng: np.random.RandomState | int | None = None,
    chain_method: str = "vectorized",
) -> NutsResult:
    """NUTS via NumPyro: autodiff + XLA + vectorized multi-chain; GPU on CUDA (hardware-agnostic).

    The user supplies a jax scalar ``logp(theta)``; we run NumPyro's NUTS over
    ``potential_fn = -logp`` so the dynamic tree / dense-mass warmup come from NumPyro (we do not
    hand-roll the tree). For ``chains > 1`` the chains are vectorized (init shape ``(chains, d)``).
    JAX picks the device automatically (GPU if a CUDA build is installed) — no device special-casing.
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    if dim is None and init is None:
        raise ValueError("pass dim= or init=.")
    rng = _as_rng(rng)
    inits = _chain_inits(init, dim, chains, rng)
    d = inits.shape[1]

    def potential_fn(theta):
        return -logp(theta)

    kernel = NUTS(potential_fn=potential_fn, target_accept_prob=target_accept, max_tree_depth=max_tree_depth)
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=num_samples,
        num_chains=chains,
        chain_method=chain_method,
        progress_bar=False,
    )
    init_params = jnp.asarray(inits if chains > 1 else inits[0])
    seed = int(rng.randint(1, 2**31 - 1))
    mcmc.run(jax.random.PRNGKey(seed), init_params=init_params)
    # group_by_chain=True -> (chains, num_samples, d)
    samples = np.asarray(mcmc.get_samples(group_by_chain=True), dtype=float).reshape(chains, num_samples, d)
    step_size = float("nan")
    try:
        ss = mcmc.last_state.adapt_state.step_size
        step_size = float(np.asarray(ss).reshape(-1)[0])
    except Exception:
        pass
    chain_arrays = [samples[c] for c in range(chains)]
    return _pool_chains(chain_arrays, d, chains, 0, step_size, backend="jax")


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
    from mixle.inference._advi import _advi_optimize
    from mixle.inference.mcmc.gradients import torch_available

    if not torch_available():
        raise RuntimeError("mixle.inference.advi requires PyTorch.")
    import torch  # noqa: F401

    rng = _as_rng(rng)
    mean_np, scale_np, U, objective = _advi_optimize(
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
    return AdviResult(samples=U, mean=mean_np, scale=scale_np, objective=objective)


def _as_rng(rng: np.random.RandomState | int | None) -> np.random.RandomState:
    if rng is None:
        return np.random.RandomState()
    if isinstance(rng, (int, np.integer)):
        return np.random.RandomState(int(rng))
    return rng


def _chain_inits(init: Any, dim: int | None, chains: int, rng: np.random.RandomState) -> np.ndarray:
    """Build a ``(chains, d)`` array of initial states from ``init`` / ``dim``."""
    if init is None:
        if dim is None:
            raise ValueError("pass dim= or init=.")
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


def _pool_chains(
    chain_arrays: list[np.ndarray],
    d: int,
    chains: int,
    total_evals: int,
    last_step: float,
    *,
    backend: str,
    **extra: Any,
) -> NutsResult:
    """Stack per-chain draws into a :class:`NutsResult` with pooled draws + R-hat / ESS."""
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
        extra={"backend": backend, **extra},
    )


# --------------------------------------------------------------------------------------------
# Backend registration (`register, don't branch`). Each guarded so a missing optional engine
# never breaks ``import mixle.inference``.
# --------------------------------------------------------------------------------------------
import importlib.util  # noqa: E402

register_inference_backend(
    InferenceBackend(name="numpy", available=lambda: True, target_kind="numpy_vg", nuts=_nuts_numpy)
)
register_inference_backend(
    InferenceBackend(
        name="numba",
        available=lambda: importlib.util.find_spec("numba") is not None,
        target_kind="njit_vg",
        nuts=_nuts_numba,
    )
)
register_inference_backend(
    InferenceBackend(
        name="torch",
        available=lambda: importlib.util.find_spec("torch") is not None,
        target_kind="torch_logp",
        nuts=nuts_torch,
    )
)
register_inference_backend(
    InferenceBackend(
        name="jax",
        available=lambda: (
            importlib.util.find_spec("numpyro") is not None and importlib.util.find_spec("jax") is not None
        ),
        target_kind="jax_logp",
        nuts=_nuts_jax,
    )
)
