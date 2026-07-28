"""Reusable ADVI core: fit a Gaussian variational posterior by reparameterized-MC SGVB (Adam).

Lives in ``mixle.inference`` (not ``mixle.ppl``) so the core ``mixle.inference.advi`` facade and the
optional ``mixle.ppl`` autograd layer share ONE implementation without ``mixle.inference`` depending
upward on ``mixle.ppl``. The optimizer is generic -- it takes ``torch`` and a batched log-target
callable and has no dependency on the PPL graph types -- so it belongs in the core inference layer.
"""

from __future__ import annotations

import math

import numpy as np


def _exact_positive(name: str, value) -> int:
    """``value`` as an exact positive integer count.

    ``bool`` is rejected on purpose: ``steps=True`` would run a one-step "optimization" and return a
    fitted-looking result. Fractional and non-positive counts are rejected for the same reason --
    ``steps=-1`` performed no optimization at all yet still returned a finite objective and samples,
    and ``mc=0`` updated the variational scale without evaluating a single target draw.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an exact positive integer, got {value!r}")
    count = int(value)
    if count < 1:
        raise ValueError(f"{name} must be positive, got {count}")
    return count


def _validate_init(u0, s0) -> tuple[np.ndarray, np.ndarray]:
    """The variational initialization as aligned finite 1-D arrays with strictly positive scales.

    An initial scale of zero produced ``scale=0`` and objective ``-inf``; a negative scale or a NaN
    anywhere produced all-NaN parameters and objective with no error at all. Neither is a posterior.
    """
    mean = np.asarray(u0, dtype=float)
    scale = np.asarray(s0, dtype=float)
    if mean.ndim != 1 or scale.ndim != 1:
        raise ValueError(f"u0 and s0 must be one-dimensional, got shapes {mean.shape} and {scale.shape}")
    if mean.shape != scale.shape:
        raise ValueError(f"u0 and s0 must be aligned, got shapes {mean.shape} and {scale.shape}")
    if mean.size == 0:
        raise ValueError("u0 and s0 must be non-empty: there is nothing to infer over zero dimensions")
    if not np.isfinite(mean).all():
        raise ValueError("u0 must contain only finite values")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("s0 must contain only finite, strictly positive scales")
    return mean, scale


def _checked_log_p(log_p_fn, u, batch_size: int, *, require_grad: bool):
    """Evaluate the batched log-target and enforce its documented ``(batch_size,)`` contract.

    The contract is ``log_p_fn(U[batch, d]) -> Tensor[batch]``, one log-density per Monte Carlo draw,
    but nothing checked it. A target returning a single scalar sum over the whole batch was accepted
    and optimized as though it were the documented ELBO -- its contribution scaled and coupled across
    draws instead of being averaged -- and a broadcastable wrong-width output silently changed the
    Renyi importance weights. A target detached from the variational parameters is rejected too:
    with no gradient path, the target term contributes nothing to the update and only the entropy is
    actually being optimized.
    """
    log_p = log_p_fn(u)
    shape = tuple(getattr(log_p, "shape", ()))
    if shape != (batch_size,):
        raise ValueError(
            f"log_p_fn must return one log-density per Monte Carlo draw -- a tensor of shape "
            f"({batch_size},) -- but returned shape {shape}"
        )
    if require_grad and not bool(getattr(log_p, "requires_grad", False)):
        raise ValueError(
            "log_p_fn returned a target detached from the variational parameters; ADVI cannot "
            "optimize a target it has no gradient path through"
        )
    return log_p


def _advi_optimize(
    torch,
    log_p_fn,
    u0,
    s0,
    *,
    samples: int,
    mc: int,
    steps: int,
    lr: float,
    rng,
    family: str = "meanfield",
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reusable ADVI core: fit a Gaussian variational posterior by reparameterized-MC SGVB (Adam).

    ``log_p_fn(U: Tensor(mc, d)) -> Tensor(mc,)`` is the (unconstrained) batched joint log-target;
    it owns any data minibatching/rescaling. This is the family/objective machinery shared by
    :meth:`GradTarget.advi` and the public :func:`mixle.inference.advi` facade, with no dependency on
    ``GradTarget``'s slots or data. Returns ``(mean_u, scale_u, U_draws, objective)`` with the
    unconstrained mean/scale, the draws ``(samples, d)``, and the final variational objective value
    (the ELBO for ``alpha=1``, otherwise the tilted Renyi bound).

    Every control is validated before any optimization happens, and the fitted parameters and final
    objective must be finite before a result is handed back: an ``AdviResult`` is a claim that a
    posterior was fitted, and an unvalidated run could satisfy that claim having performed no valid
    optimization at all.

    Raises:
        TypeError: if ``samples``, ``mc``, or ``steps`` is not an exact integer.
        ValueError: for a non-positive count, a non-finite or non-positive learning rate, an
            unsupported ``alpha`` or ``family``, a misaligned/non-finite/non-positive-scale
            initialization, a target that violates its ``(mc,)`` output contract, or a fit that did
            not converge to finite variational parameters and objective.
    """
    samples = _exact_positive("samples", samples)
    mc = _exact_positive("mc", mc)
    steps = _exact_positive("steps", steps)
    lr = float(lr)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"lr must be a finite positive learning rate, got {lr!r}")
    alpha = float(alpha)
    # alpha=1 is the KL-ELBO and alpha=0 the importance-weighted (IWAE) bound; the tilted Renyi
    # family is defined for non-negative alpha. NaN used to propagate straight into all-NaN output.
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"alpha must be finite and >= 0 (1.0 = KL-ELBO, 0.0 = IWAE), got {alpha!r}")
    if family not in ("meanfield", "fullrank"):
        raise ValueError(f"unknown variational family {family!r}; use 'meanfield' or 'fullrank'.")
    u0_arr, s0_arr = _validate_init(u0, s0)

    d = int(u0_arr.size)
    half_d_log2pi = 0.5 * d * math.log(2.0 * math.pi)
    entropy_const = 0.5 * d * (1.0 + math.log(2.0 * math.pi))
    gen = torch.Generator().manual_seed(int(rng.randint(1, 2**31)))
    mean = torch.tensor(u0_arr, dtype=torch.float64, requires_grad=True)
    if family == "fullrank":
        # L_raw holds the Cholesky factor: strict-lower entries free, diagonal in log-space.
        l_raw = torch.tensor(np.diag(np.log(s0_arr)), dtype=torch.float64, requires_grad=True)
        params = [mean, l_raw]
    else:
        log_std = torch.tensor(np.log(s0_arr), dtype=torch.float64, requires_grad=True)
        params = [mean, log_std]
    opt = torch.optim.Adam(params, lr=lr)

    def variational(eps):
        # -> (U draws (mc,d), log q(U) per sample, exact entropy H[q])
        if family == "fullrank":
            chol = torch.tril(l_raw, -1) + torch.diag(torch.exp(torch.diagonal(l_raw)))
            u = mean + eps @ chol.T
            log_diag = torch.diagonal(l_raw)  # = log of chol's diagonal
        else:
            chol = torch.exp(log_std)
            u = mean + chol * eps
            log_diag = log_std
        log_q = -0.5 * (eps * eps).sum(dim=1) - log_diag.sum() - half_d_log2pi
        return u, log_q, log_diag.sum() + entropy_const

    for _ in range(steps):
        opt.zero_grad()
        eps = torch.randn((mc, d), dtype=torch.float64, generator=gen)
        u, log_q, entropy = variational(eps)
        log_p = _checked_log_p(log_p_fn, u, mc, require_grad=True)
        if alpha == 1.0:  # standard ELBO with the exact (low-variance) entropy term
            obj = log_p.mean() + entropy
        else:  # tilted Renyi-alpha bound: tilt the importance weights w=p/q by (1-alpha)
            log_w = log_p - log_q
            obj = (torch.logsumexp((1.0 - alpha) * log_w, dim=0) - math.log(mc)) / (1.0 - alpha)
        (-obj).backward()
        opt.step()

    # final objective at the fitted q, estimated with extra MC samples for a low-variance value
    with torch.no_grad():
        n_eval = max(mc, 256)
        eps = torch.randn((n_eval, d), dtype=torch.float64, generator=gen)
        u, log_q, entropy = variational(eps)
        log_p = _checked_log_p(log_p_fn, u, n_eval, require_grad=False)
        if alpha == 1.0:
            final_obj = float((log_p.mean() + entropy).item())
        else:
            log_w = log_p - log_q
            final_obj = float(
                ((torch.logsumexp((1.0 - alpha) * log_w, dim=0) - math.log(n_eval)) / (1.0 - alpha)).item()
            )

    mean_np = mean.detach().numpy()
    z = rng.standard_normal((samples, d))
    if family == "fullrank":
        chol = (torch.tril(l_raw, -1) + torch.diag(torch.exp(torch.diagonal(l_raw)))).detach().numpy()
        U = mean_np + z @ chol.T
        scale_np = np.sqrt(np.sum(chol * chol, axis=1))  # marginal std per dim
    else:
        scale_np = torch.exp(log_std).detach().numpy()
        U = mean_np + scale_np * z

    # Convergence is an explicit postcondition, not an assumption: returning non-finite variational
    # parameters or a non-finite objective would hand back a fitted-looking artifact from a run that
    # produced no usable posterior.
    if not math.isfinite(final_obj):
        raise ValueError(
            f"ADVI did not converge to a finite objective (got {final_obj}); the fit produced no "
            "usable variational posterior"
        )
    if not (np.isfinite(mean_np).all() and np.isfinite(scale_np).all() and np.all(scale_np > 0.0)):
        raise ValueError("ADVI did not converge to finite variational parameters with positive scales")
    return mean_np, scale_np, U, final_obj
