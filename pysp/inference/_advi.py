"""Reusable ADVI core: fit a Gaussian variational posterior by reparameterized-MC SGVB (Adam).

Lives in ``pysp.inference`` (not ``pysp.ppl``) so the core ``pysp.inference.advi`` facade and the
optional ``pysp.ppl`` autograd layer share ONE implementation without ``pysp.inference`` depending
upward on ``pysp.ppl``. The optimizer is generic -- it takes ``torch`` and a batched log-target
callable and has no dependency on the PPL graph types -- so it belongs in the core inference layer.
"""

from __future__ import annotations

import math

import numpy as np


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
    :meth:`GradTarget.advi` and the public :func:`pysp.inference.advi` facade, with no dependency on
    ``GradTarget``'s slots or data. Returns ``(mean_u, scale_u, U_draws, objective)`` with the
    unconstrained mean/scale, the draws ``(samples, d)``, and the final variational objective value
    (the ELBO for ``alpha=1``, otherwise the tilted Renyi bound)."""
    d = int(len(np.asarray(u0, dtype=float)))
    half_d_log2pi = 0.5 * d * math.log(2.0 * math.pi)
    entropy_const = 0.5 * d * (1.0 + math.log(2.0 * math.pi))
    gen = torch.Generator().manual_seed(int(rng.randint(1, 2**31)))
    mean = torch.tensor(np.asarray(u0, dtype=float), dtype=torch.float64, requires_grad=True)
    if family == "fullrank":
        # L_raw holds the Cholesky factor: strict-lower entries free, diagonal in log-space.
        l_raw = torch.tensor(np.diag(np.log(np.asarray(s0, dtype=float))), dtype=torch.float64, requires_grad=True)
        params = [mean, l_raw]
    elif family == "meanfield":
        log_std = torch.tensor(np.log(np.asarray(s0, dtype=float)), dtype=torch.float64, requires_grad=True)
        params = [mean, log_std]
    else:
        raise ValueError(f"unknown variational family {family!r}; use 'meanfield' or 'fullrank'.")
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
        log_p = log_p_fn(u)
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
        log_p = log_p_fn(u)
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
    return mean_np, scale_np, U, final_obj
