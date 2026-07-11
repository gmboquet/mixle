"""Torch-native No-U-Turn Sampler.

A device-resident port of the numpy NUTS in :mod:`mixle.inference.mcmc.samplers`: identical
algorithm (recursive tree doubling, U-turn termination, multinomial proposal, dual-averaging
step-size adaptation), but the leapfrog trajectory and the target evaluation stay in torch
tensors on the target's device. The (unnormalised) log-target is supplied as a torch scalar
function ``logp(theta) -> Tensor[()]``; its ``value_and_grad`` is built with ``torch.func`` and
``torch.compile``d once, so the autograd graph is traced a single time and reused on every
leapfrog step instead of being rebuilt (and round-tripped through numpy) per gradient call.

Intended for **GPU and large autodiff targets**, where staying on-device with a compiled target
pays off. On CPU this is typically *slower* than the numpy sampler (per-op torch dispatch + the
tree's host syncs dominate when the target is low-cost), so on CPU prefer the numpy / numba / jax
backends. The value here is autodiff without re-tracing the graph every call, plus GPU execution.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from mixle.inference.mcmc.samplers import MCMCResult


def _make_value_and_grad(logp: Callable[[Any], Any], theta0: Any, use_compile: bool):
    """Return ``(value_and_grad, compiled)`` where ``value_and_grad(theta) -> (logp, grad)``.

    Builds the fused value/grad with ``torch.func.grad_and_value`` over ``torch.compile(logp)``
    and verifies it on ``theta0`` (which also triggers compilation); falls back to eager on any
    failure so installs without a working compiler still run.
    """
    import torch

    def _wrap(fn):
        gv = torch.func.grad_and_value(fn)

        def value_and_grad(theta):
            a, b = gv(theta)
            # grad_and_value's tuple order varies across torch versions; the value is the scalar.
            return (a, b) if a.ndim == 0 else (b, a)

        return value_and_grad

    if use_compile:
        try:
            vg = _wrap(torch.compile(logp))
            lp, g = vg(theta0)
            if g.shape == theta0.shape and math.isfinite(float(lp.detach())):
                return vg, True
        except Exception:  # noqa: BLE001
            pass
    vg = _wrap(logp)
    lp, g = vg(theta0)
    if g.shape != theta0.shape:
        raise ValueError("grad shape %s does not match state shape %s." % (tuple(g.shape), tuple(theta0.shape)))
    return vg, False


def nuts_torch(
    logp: Callable[[Any], Any],
    initial: Any,
    num_samples: int = 1000,
    warmup: int = 1000,
    mass: Any = 1.0,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    thin: int = 1,
    seed: int | None = None,
    *,
    compile: bool = True,
    dtype: Any = None,
    device: Any = None,
) -> MCMCResult:
    """No-U-Turn Sampler over a torch scalar log-target, run entirely on-device.

    Args:
        logp: ``logp(theta: Tensor[d]) -> Tensor[()]`` — the (unnormalised) log target.
        initial: starting state, array-like or tensor of shape ``(d,)``.
        num_samples, warmup, thin: retained draws, adaptation iters, thinning.
        mass: diagonal mass matrix (scalar or ``(d,)``).
        target_accept, max_tree_depth: NUTS tuning, as in the numpy sampler.
        seed: seed for momentum + slice/direction RNG (reproducible).
        compile: ``torch.compile`` the target (falls back to eager if unavailable).
        dtype, device: torch dtype/device for the trajectory (default float64 / the
            initial tensor's device, else CPU).

    Returns:
        :class:`~mixle.inference.mcmc.samplers.MCMCResult` with ``samples`` (numpy), plus
        ``step_size`` and ``num_target_evals`` attributes.
    """
    import torch

    if num_samples < 0 or warmup < 0 or thin <= 0:
        raise ValueError("require num_samples>=0, warmup>=0, thin>0.")
    rng = np.random.RandomState(seed)
    dtype = dtype or torch.float64
    if isinstance(initial, torch.Tensor):
        device = device or initial.device
        cur = initial.detach().to(dtype=dtype, device=device).reshape(-1)
    else:
        device = device or torch.device("cpu")
        cur = torch.as_tensor(np.asarray(initial, dtype=float).reshape(-1), dtype=dtype, device=device)
    (d,) = cur.shape
    shape = cur.shape

    mass_arr = torch.as_tensor(np.broadcast_to(np.asarray(mass, dtype=float), (d,)).copy(), dtype=dtype, device=device)
    minv = 1.0 / mass_arr
    sqrt_m = torch.sqrt(mass_arr)
    delta_max = 1000.0

    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng.randint(1, 2**31 - 1)))

    eval_count = [0]
    vg, compiled = _make_value_and_grad(logp, cur, compile)

    def value_and_grad(theta):
        eval_count[0] += 1
        lp, g = vg(theta)
        return lp, g

    def kinetic(r) -> float:
        return 0.5 * float(torch.sum(r * r * minv))

    def leapfrog(theta, r, grad, eps):
        r = r + 0.5 * eps * grad
        theta = theta + eps * (minv * r)
        lp1, grad1 = value_and_grad(theta)
        r = r + 0.5 * eps * grad1
        return theta, r, lp1, grad1

    def no_uturn(tm, tp, rm, rp) -> bool:
        diff = tp - tm
        return float(torch.dot(diff, minv * rm)) >= 0 and float(torch.dot(diff, minv * rp)) >= 0

    cur_lp_t, cur_grad = value_and_grad(cur)
    cur_lp = float(cur_lp_t.detach())
    if not math.isfinite(cur_lp):
        raise ValueError("initial state has non-finite log target.")
    eps = _find_reasonable_eps(cur, cur_lp, cur_grad, leapfrog, kinetic, sqrt_m, shape, gen, dtype, device)
    mu = math.log(10.0 * eps)
    log_eps_bar, h_bar, gamma, t0, kappa = 0.0, 0.0, 0.05, 10.0, 0.75

    samples: list[Any] = []
    log_probs: list[float] = []
    depths: list[int] = []
    total = warmup + num_samples * thin

    def build_tree(theta, r, grad, logu, v, j, eps, joint0):
        if j == 0:
            theta1, r1, lp1_t, grad1 = leapfrog(theta, r, grad, v * eps)
            lp1 = float(lp1_t.detach())
            joint1 = lp1 - kinetic(r1)
            n1 = 1 if logu <= joint1 else 0
            s1 = 1 if (joint1 - logu) > -delta_max and math.isfinite(joint1) else 0
            a = min(1.0, math.exp(min(joint1 - joint0, 0.0))) if math.isfinite(joint1) else 0.0
            return theta1, r1, grad1, theta1, r1, grad1, theta1, lp1, grad1, n1, s1, a, 1
        tm, rm, gm, tp, rp, gp, tpr, lpr, gpr, n1, s1, a1, na1 = build_tree(theta, r, grad, logu, v, j - 1, eps, joint0)
        if s1 == 1:
            if v == -1:
                tm, rm, gm, _, _, _, t2, lp2, g2, n2, s2, a2, na2 = build_tree(tm, rm, gm, logu, v, j - 1, eps, joint0)
            else:
                _, _, _, tp, rp, gp, t2, lp2, g2, n2, s2, a2, na2 = build_tree(tp, rp, gp, logu, v, j - 1, eps, joint0)
            if n2 > 0 and rng.random_sample() < n2 / max(n1 + n2, 1):
                tpr, lpr, gpr = t2, lp2, g2
            a1 += a2
            na1 += na2
            n1 += n2
            s1 = s2 if no_uturn(tm, tp, rm, rp) else 0
        return tm, rm, gm, tp, rp, gp, tpr, lpr, gpr, n1, s1, a1, na1

    for it in range(total):
        r0 = sqrt_m * torch.randn(shape, generator=gen, dtype=dtype, device=device)
        joint0 = cur_lp - kinetic(r0)
        logu = joint0 - rng.exponential()
        tm = tp = cur
        rm = rp = r0
        gm = gp = cur_grad
        theta_new, lp_new, grad_new, n, s, j = cur, cur_lp, cur_grad, 1, 1, 0
        alpha, n_alpha = 0.0, 1
        while s == 1 and j < max_tree_depth:
            v = -1 if rng.random_sample() < 0.5 else 1
            if v == -1:
                tm, rm, gm, _, _, _, tpr, lpr, gpr, n_p, s_p, alpha, n_alpha = build_tree(
                    tm, rm, gm, logu, v, j, eps, joint0
                )
            else:
                _, _, _, tp, rp, gp, tpr, lpr, gpr, n_p, s_p, alpha, n_alpha = build_tree(
                    tp, rp, gp, logu, v, j, eps, joint0
                )
            if s_p == 1 and rng.random_sample() < min(1.0, n_p / max(n, 1)):
                theta_new, lp_new, grad_new = tpr, lpr, gpr
            n += n_p
            s = s_p if no_uturn(tm, tp, rm, rp) else 0
            j += 1
        cur, cur_lp, cur_grad = theta_new, lp_new, grad_new

        accept_stat = alpha / max(n_alpha, 1)
        if it < warmup:
            m1 = it + 1
            h_bar = (1.0 - 1.0 / (m1 + t0)) * h_bar + (target_accept - accept_stat) / (m1 + t0)
            log_eps = mu - math.sqrt(m1) / gamma * h_bar
            eta = m1 ** (-kappa)
            log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
            eps = math.exp(log_eps)
        elif it == warmup:
            eps = math.exp(log_eps_bar)

        if it >= warmup and ((it - warmup) % thin == 0):
            samples.append(cur.detach().cpu().numpy())
            log_probs.append(cur_lp)
            depths.append(j)

    res = MCMCResult(
        samples=samples,
        log_probs=np.asarray(log_probs, dtype=float),
        accepted=np.ones(len(samples), dtype=bool),
        transition_labels=tuple("nuts" for _ in samples),
    )
    object.__setattr__(res, "tree_depth", np.asarray(depths, dtype=int))
    object.__setattr__(res, "step_size", float(eps))
    object.__setattr__(res, "num_target_evals", int(eval_count[0]))
    object.__setattr__(res, "compiled", bool(compiled))
    return res


def _find_reasonable_eps(theta, lp0, grad0, leapfrog, kinetic, sqrt_m, shape, gen, dtype, device) -> float:
    """Heuristic initial step size (Hoffman & Gelman Algorithm 4), torch-tensor variant."""
    import torch

    eps = 1.0
    r = sqrt_m * torch.randn(shape, generator=gen, dtype=dtype, device=device)
    joint0 = lp0 - kinetic(r)

    def joint_after(step):
        _t1, r1, lp1_t, _g1 = leapfrog(theta, r, grad0, step)
        lp1 = float(lp1_t.detach())
        return (lp1 - kinetic(r1)) if math.isfinite(lp1) else -math.inf

    j1 = joint_after(eps)
    a = 1 if (j1 - joint0) > math.log(0.5) else -1
    while a * (joint_after(eps) - joint0) > a * math.log(0.5):
        eps *= 2.0**a
        if eps < 1e-10 or eps > 1e10:
            break
    return eps
