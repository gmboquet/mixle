"""Reusable ADVI core: fit a Gaussian variational posterior by reparameterized-MC SGVB (Adam).

Lives in ``mixle.inference`` (not ``mixle.ppl``) so the core ``mixle.inference.advi`` facade and the
optional ``mixle.ppl`` autograd layer share ONE implementation without ``mixle.inference`` depending
upward on ``mixle.ppl``. The optimizer is generic -- it takes ``torch`` and a batched log-target
callable and has no dependency on the PPL graph types -- so it belongs in the core inference layer.
"""

from __future__ import annotations

import math

import numpy as np

# Below this distance from alpha = 1 the tilted-Renyi estimator is numerically WORSE than the
# ELBO it converges to: the objective divides logsumexp((1-alpha) log_w) by (1-alpha), so float64
# round-off in the logsumexp is amplified by 1/(1-alpha) -- measured absolute error on a fixed
# log_w vector: 5e-8 at 1-alpha=1e-8, 4e-7 at 1e-10, 3.9e-4 at 1e-12, 2.3e-3 at 1e-13 and growing.
# Meanwhile the MATHEMATICAL gap |L_alpha - ELBO| ~ (1-alpha) Var(log w)/2 is ~2e-6 at 1e-6.
# Inside the band the exact-entropy ELBO branch is therefore both the better estimator and within
# noise of what was asked for. An exact `alpha == 1.0` test left the cliff reachable from below.
_ELBO_ALPHA_TOL = 1e-6


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
    ``GradTarget``'s slots or data. Returns ``(mean_u, scale_u, U_draws, objective, n_eval,
    cholesky)`` with the unconstrained mean/scale, the draws ``(samples, d)``, the final variational
    objective value, the number of Monte Carlo draws that value was estimated with, and -- for
    ``family='fullrank'`` only -- the fitted lower-triangular Cholesky factor of the covariance
    (``None`` for meanfield). Returning the factor is what makes a full-rank fit usable: the
    correlations it exists to learn live in the off-diagonals, and reporting marginal scales alone
    hands back a posterior indistinguishable from the meanfield one the caller opted out of.

    What ``objective`` is -- and is not. For ``|alpha - 1| <= 1e-6`` it is the K-draw MC estimate of
    the ELBO (unbiased in the mean term, exact entropy). Otherwise it is the K-SAMPLE tilted Renyi
    bound at ``K = n_eval = max(mc, 256)``: its expectation DEPENDS ON K (for ``alpha < 1`` it rises
    toward the true bound as K grows, IWAE-style), so two fits' objectives are comparable only at
    the same ``n_eval``, and a bigger objective from a bigger ``mc`` is partly just K. For
    ``alpha > 1`` the finite-K estimate is upward-biased (Jensen flips with the negative
    ``1 - alpha``) and is NOT a lower bound on the evidence -- it can exceed ``log Z``.

    Optimization runs a FIXED number of Adam steps: no convergence criterion is evaluated, and
    nothing here claims stationarity. The postcondition enforced before returning is FINITENESS --
    fitted parameters and objective that are actually numbers -- which rules out a fitted-looking
    artifact from a numerically failed run but says nothing about optimization quality; judge that
    by whether the objective has stopped improving across ``steps``/``lr`` choices.

    Raises:
        TypeError: if ``samples``, ``mc``, or ``steps`` is not an exact integer.
        ValueError: for a non-positive count, a non-finite or non-positive learning rate, an
            unsupported ``alpha`` or ``family``, a misaligned/non-finite/non-positive-scale
            initialization, a target that violates its ``(mc,)`` output contract, or a run that
            ended in non-finite variational parameters or objective.
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
    # Tolerance band, not an exact float test (see _ELBO_ALPHA_TOL): alpha just below 1 reaches the
    # Renyi branch whose 1/(1-alpha) division amplifies logsumexp round-off past the ELBO gap.
    use_elbo = abs(alpha - 1.0) <= _ELBO_ALPHA_TOL
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
        if use_elbo:  # standard ELBO with the exact (low-variance) entropy term
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
        if use_elbo:
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
        chol = None  # meanfield has no off-diagonal structure to report; `scale` is the whole story
        scale_np = torch.exp(log_std).detach().numpy()
        U = mean_np + scale_np * z

    # FINITENESS postcondition -- deliberately not called "convergence": the loop above ran a fixed
    # number of Adam steps and checked no stationarity criterion, so finite output only certifies
    # that the run produced numbers, not that they are an optimum. Non-finite output would hand back
    # a fitted-looking artifact from a numerically failed run.
    if not math.isfinite(final_obj):
        raise ValueError(
            f"ADVI ended with a non-finite objective (got {final_obj}); the run produced no usable "
            "variational posterior"
        )
    if not (np.isfinite(mean_np).all() and np.isfinite(scale_np).all() and np.all(scale_np > 0.0)):
        raise ValueError("ADVI ended with non-finite variational parameters (or non-positive scales)")
    return mean_np, scale_np, U, final_obj, n_eval, chol
