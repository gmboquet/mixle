"""Numba (``@njit``) No-U-Turn Sampler over an analytic ``@njit`` ``value_and_grad``.

A line-for-line port of the numpy NUTS in :mod:`mixle.inference.mcmc.samplers` (recursive tree
doubling, U-turn termination, multinomial proposal, dual-averaging step-size adaptation, the
``_find_reasonable_eps`` heuristic) compiled with ``@njit``. The (unnormalised) log-target is a
PASSED ``@njit`` fused ``value_and_grad(theta) -> (float, ndarray)``; numba supports first-class
jitted functions, so the whole sampler — including the recursive ``build_tree`` and its
heterogeneous return tuple — runs in nopython mode with no Python-level callback per step.

**Contract:** the caller supplies an analytic ``@njit`` ``value_and_grad`` — there is **no
autodiff** here (use the torch or jax backends for that). The win is CPU throughput on
analytic-gradient models (no per-step Python dispatch). ``np.random`` (``seed`` /
``standard_normal`` / ``exponential`` / ``random``) is used inside the kernel and is seeded via
``np.random.seed`` for reproducibility.

The public entry point :func:`nuts_numba` is a thin Python wrapper that loops chains, pools the
draws, computes R-hat / ESS, and returns a :class:`mixle.inference.mcmc.samplers.MCMCResult` — mirroring
the numpy facade's per-chain loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mixle.inference.mcmc.samplers import MCMCResult, _validated_nuts_controls
from mixle.utils.optional_deps import numba

njit = numba.njit


@njit(cache=True)
def _kinetic(r, minv):
    return 0.5 * np.sum(r * r * minv)


@njit(cache=True)
def _no_uturn(tm, tp, rm, rp, minv):
    d = tp - tm
    return (np.dot(d, minv * rm) >= 0.0) and (np.dot(d, minv * rp) >= 0.0)


@njit(cache=True)
def _leapfrog(value_and_grad, theta, r, grad, eps, minv):
    r = r + 0.5 * eps * grad
    theta = theta + eps * (minv * r)
    lp1, grad1 = value_and_grad(theta)
    r = r + 0.5 * eps * grad1
    return theta, r, lp1, grad1


@njit(cache=True)
def _find_reasonable_eps(value_and_grad, theta, grad0, minv, sqrt_m, d):
    eps = 1.0
    r = sqrt_m * np.random.standard_normal(d)
    lp0, _ = value_and_grad(theta)
    joint0 = lp0 - _kinetic(r, minv)

    _t1, r1, lp1, _g1 = _leapfrog(value_and_grad, theta, r, grad0, eps, minv)
    j1 = (lp1 - _kinetic(r1, minv)) if np.isfinite(lp1) else -np.inf
    a = 1.0 if (j1 - joint0) > np.log(0.5) else -1.0
    while np.isfinite(j1) and a * (j1 - joint0) > a * np.log(0.5):
        eps *= 2.0**a
        _t1, r1, lp1, _g1 = _leapfrog(value_and_grad, theta, r, grad0, eps, minv)
        j1 = (lp1 - _kinetic(r1, minv)) if np.isfinite(lp1) else -np.inf
        if eps < 1e-10 or eps > 1e10:
            break
    return eps


@njit(cache=True)
def _build_tree(value_and_grad, theta, r, grad, logu, v, j, eps, joint0, minv, delta_max):
    if j == 0:
        theta1, r1, lp1, grad1 = _leapfrog(value_and_grad, theta, r, grad, v * eps, minv)
        joint1 = lp1 - _kinetic(r1, minv)
        n1 = 1 if logu <= joint1 else 0
        s1 = 1 if ((joint1 - logu) > -delta_max and np.isfinite(joint1)) else 0
        a = min(1.0, np.exp(min(joint1 - joint0, 0.0))) if np.isfinite(joint1) else 0.0
        return theta1, r1, grad1, theta1, r1, grad1, theta1, lp1, grad1, n1, s1, a, 1
    tm, rm, gm, tp, rp, gp, tpr, lpr, gpr, n1, s1, a1, na1 = _build_tree(
        value_and_grad, theta, r, grad, logu, v, j - 1, eps, joint0, minv, delta_max
    )
    if s1 == 1:
        if v == -1:
            tm, rm, gm, _, _, _, t2, lp2, g2, n2, s2, a2, na2 = _build_tree(
                value_and_grad, tm, rm, gm, logu, v, j - 1, eps, joint0, minv, delta_max
            )
        else:
            _, _, _, tp, rp, gp, t2, lp2, g2, n2, s2, a2, na2 = _build_tree(
                value_and_grad, tp, rp, gp, logu, v, j - 1, eps, joint0, minv, delta_max
            )
        if n2 > 0 and np.random.random() < n2 / max(n1 + n2, 1):
            tpr, lpr, gpr = t2, lp2, g2
        a1 += a2
        na1 += na2
        n1 += n2
        s1 = s2 if _no_uturn(tm, tp, rm, rp, minv) else 0
    return tm, rm, gm, tp, rp, gp, tpr, lpr, gpr, n1, s1, a1, na1


@njit(cache=True)
def _nuts_core(value_and_grad, theta0, num_samples, warmup, mass, target_accept, max_tree_depth, thin, seed):
    """Run a single NUTS chain in nopython mode. Returns ``(samples, step_size, num_evals)``."""
    np.random.seed(seed)
    d = theta0.shape[0]
    minv = 1.0 / mass
    sqrt_m = np.sqrt(mass)
    delta_max = 1000.0

    eval_count = 0
    cur = theta0.copy()
    cur_lp, cur_grad = value_and_grad(cur)
    eval_count += 1

    eps = _find_reasonable_eps(value_and_grad, cur, cur_grad, minv, sqrt_m, d)
    mu = np.log(10.0 * eps)
    log_eps_bar = 0.0
    h_bar = 0.0
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75

    n_keep = num_samples  # retained draws; num_samples * thin post-warmup iterations, keep every thin-th
    samples = np.empty((n_keep, d))
    log_probs = np.empty(n_keep)
    accepted = np.zeros(n_keep, dtype=np.bool_)  # did the chain actually move on this draw?
    kept = 0
    total = warmup + num_samples * thin

    for it in range(total):
        r0 = sqrt_m * np.random.standard_normal(d)
        joint0 = cur_lp - _kinetic(r0, minv)
        logu = joint0 - np.random.exponential()
        tm = cur.copy()
        tp = cur.copy()
        rm = r0.copy()
        rp = r0.copy()
        gm = cur_grad.copy()
        gp = cur_grad.copy()
        theta_new = cur.copy()
        lp_new = cur_lp
        grad_new = cur_grad.copy()
        n = 1
        s = 1
        j = 0
        alpha = 0.0
        n_alpha = 0
        moved = False  # did any doubling actually adopt its proposal?
        while s == 1 and j < max_tree_depth:
            v = -1 if np.random.random() < 0.5 else 1
            if v == -1:
                tm, rm, gm, _, _, _, tpr, lpr, gpr, n_p, s_p, a_p, na_p = _build_tree(
                    value_and_grad, tm, rm, gm, logu, v, j, eps, joint0, minv, delta_max
                )
            else:
                _, _, _, tp, rp, gp, tpr, lpr, gpr, n_p, s_p, a_p, na_p = _build_tree(
                    value_and_grad, tp, rp, gp, logu, v, j, eps, joint0, minv, delta_max
                )
            # accept_stat must average the Metropolis probability over every leapfrog step in
            # the trajectory, so each doubling's contribution adds to the running totals rather
            # than replacing them. Mirrors the numpy sampler in mcmc/samplers.py.
            alpha += a_p
            n_alpha += na_p
            eval_count += 2**j  # each doubling adds 2**j leapfrog evaluations
            if s_p == 1 and np.random.random() < min(1.0, n_p / max(n, 1)):
                theta_new = tpr
                lp_new = lpr
                grad_new = gpr
                moved = True
            n += n_p
            s = s_p if _no_uturn(tm, tp, rm, rp, minv) else 0
            j += 1
        cur = theta_new
        cur_lp = lp_new
        cur_grad = grad_new

        accept_stat = alpha / max(n_alpha, 1)
        if it < warmup:
            m1 = it + 1
            h_bar = (1.0 - 1.0 / (m1 + t0)) * h_bar + (target_accept - accept_stat) / (m1 + t0)
            log_eps = mu - np.sqrt(m1) / gamma * h_bar
            eta = m1 ** (-kappa)
            log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
            eps = np.exp(log_eps)
        elif it == warmup and warmup > 0:
            # warmup == 0: `it < warmup` never ran, so log_eps_bar is still its zero-initial value
            # -- exp(0.0) == 1.0 would silently overwrite the properly-tuned eps found before this
            # loop with a fixed constant unrelated to the target's actual scale.
            eps = np.exp(log_eps_bar)

        if it >= warmup and ((it - warmup) % thin == 0) and kept < n_keep:
            samples[kept] = cur
            log_probs[kept] = cur_lp
            accepted[kept] = moved
            kept += 1

    return samples[:kept], log_probs[:kept], accepted[:kept], eps, eval_count


def nuts_numba(
    value_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    initial: np.ndarray,
    num_samples: int = 1000,
    warmup: int = 1000,
    mass: Any = 1.0,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    thin: int = 1,
    seed: int | None = None,
) -> MCMCResult:
    """No-U-Turn Sampler over an ``@njit`` analytic ``value_and_grad``, run in nopython mode.

    Args:
        value_and_grad: an ``@njit``-compiled fused callable ``theta -> (logp, grad)`` returning
            the (unnormalised) log target and its **analytic** gradient. There is no autodiff;
            supply the gradient (njit-jitted) yourself.
        initial: starting state, array-like of shape ``(d,)``.
        num_samples, warmup, thin: retained draws, adaptation iters, thinning.
        mass: diagonal mass matrix (scalar or ``(d,)``).
        target_accept, max_tree_depth: NUTS tuning, as in the numpy sampler.
        seed: seed for the in-kernel ``np.random`` (reproducible per chain).

    Returns:
        :class:`~mixle.inference.mcmc.samplers.MCMCResult` with numpy ``samples``, plus ``step_size``
        and ``num_target_evals`` attributes.
    """
    num_samples, warmup, thin, target_accept, max_tree_depth = _validated_nuts_controls(
        num_samples, warmup, thin, target_accept, max_tree_depth
    )
    theta0 = np.asarray(initial, dtype=np.float64).reshape(-1)
    d = theta0.shape[0]
    mass_arr = np.broadcast_to(np.asarray(mass, dtype=np.float64), (d,)).copy()
    if np.any(mass_arr <= 0.0) or not np.all(np.isfinite(mass_arr)):
        raise ValueError("mass must be finite and positive.")
    lp0, g0 = value_and_grad(theta0)
    if not np.isfinite(float(lp0)):
        raise ValueError("initial state has non-finite log target.")
    # The numpy sampler refuses a non-finite gradient at a finite target state; this backend did not,
    # and a NaN gradient there does not stop the chain -- it silently makes every leapfrog step
    # produce NaN positions that then fail the divergence test, so the run "completes" with draws
    # that never left the initial point.
    if not np.all(np.isfinite(np.asarray(g0, dtype=np.float64))):
        raise ValueError("gradient contains non-finite values at a finite target state.")
    seed = int(np.random.randint(1, 2**31 - 1)) if seed is None else int(seed)

    samples, log_probs, accepted, eps, num_evals = _nuts_core(
        value_and_grad,
        theta0,
        num_samples,
        warmup,
        mass_arr,
        target_accept,
        max_tree_depth,
        thin,
        seed,
    )
    sample_list = [samples[i].copy() for i in range(samples.shape[0])]
    res = MCMCResult(
        samples=sample_list,
        log_probs=np.asarray(log_probs, dtype=float),
        # the chain's real movement, not an assumption that NUTS always moves -- see samplers.py
        accepted=np.asarray(accepted, dtype=bool),
        transition_labels=tuple("nuts" for _ in sample_list),
    )
    object.__setattr__(res, "step_size", float(eps))
    object.__setattr__(res, "num_target_evals", int(num_evals))
    return res
