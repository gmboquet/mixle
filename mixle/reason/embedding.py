"""A rate-adaptive common embedding whose active dimension scales with information content.

A fixed-width embedding wastes capacity on low-information inputs and truncates high-information
ones. This encoder learns a *shared* latent code with a **variational** per-coordinate posterior
``q(z_k | x) = N(m_k(x), s_k(x)^2)`` and an ARD (automatic relevance determination) gate: a
coordinate whose posterior stays at its prior (``KL(q || p) ~ 0``) carries no information and is
*inactive*. The **active dimension of an input** is therefore ``#{k : KL(q(z_k|x) || p) > tau}`` --
it grows, per input and per modality, with the mutual information between the input and the latent.

Training is a rate--distortion (beta-VAE) objective: reconstruct the input subject to a rate budget
on the total KL. The ``beta`` knob sets bits-per-embedding; the data decides how those bits are
spent across coordinates, so a dense high-entropy input lights up more coordinates than a sparse one.
Because all inputs share one ordered coordinate system, the codes are comparable across modalities
(a *common* embedding), and can index a :class:`mixle.reason.CrossModalStore`.

Torch is imported lazily; :mod:`mixle.reason` exposes this via a deferred attribute.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _torch() -> Any:
    import torch

    return torch


class ScaledEmbedding:
    """A beta-VAE-style common embedding with an ARD gate giving a data-dependent active dimension.

    Args:
        in_dim: input feature width.
        max_dim: the embedding's maximum width (upper bound on active dimension).
        hidden: hidden widths shared by the encoder and decoder trunks.
        beta: rate weight in the ELBO (larger -> tighter rate budget -> fewer active dims).
        kl_tau: per-coordinate KL threshold (nats) above which a coordinate counts as active.
        seed: torch RNG seed.
    """

    def __init__(
        self,
        in_dim: int,
        max_dim: int = 16,
        *,
        hidden: tuple[int, ...] = (64,),
        beta: float = 1.0,
        kl_tau: float = 1e-2,
        seed: int = 0,
    ) -> None:
        torch = _torch()
        torch.manual_seed(int(seed))
        self.in_dim = int(in_dim)
        self.max_dim = int(max_dim)
        self.beta = float(beta)
        self.kl_tau = float(kl_tau)

        def mlp(sizes: list[int]) -> Any:
            layers: list[Any] = []
            for a, b in zip(sizes[:-1], sizes[1:]):
                layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
            return layers[:-1]  # drop trailing ReLU

        h = list(hidden)
        self._enc = torch.nn.Sequential(*mlp([self.in_dim, *h, 2 * self.max_dim])).double()
        self._dec = torch.nn.Sequential(*mlp([self.max_dim, *h[::-1], self.in_dim])).double()
        self._x_mean = np.zeros(self.in_dim)
        self._x_scale = np.ones(self.in_dim)
        self._fitted = False

    # -- internals ----------------------------------------------------------------------------
    def _encode_std(self, x_std: Any) -> tuple[Any, Any]:
        torch = _torch()
        out = self._enc(x_std)
        mu = out[..., : self.max_dim]
        logvar = out[..., self.max_dim :].clamp(-10.0, 10.0)
        return mu, logvar

    @staticmethod
    def _kl(mu: Any, logvar: Any) -> Any:
        # KL(N(mu, e^logvar) || N(0, 1)) per coordinate.
        return 0.5 * (mu**2 + logvar.exp() - 1.0 - logvar)

    # -- training -----------------------------------------------------------------------------
    def fit(self, X: Any, *, epochs: int = 400, lr: float = 3e-3, weight_decay: float = 0.0) -> ScaledEmbedding:
        """Train the embedding on unlabeled inputs ``X`` (``(n, in_dim)``) by the beta-VAE ELBO."""
        torch = _torch()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self._x_mean = X.mean(axis=0)
        self._x_scale = X.std(axis=0) + 1e-8
        xt = torch.as_tensor((X - self._x_mean) / self._x_scale, dtype=torch.float64)
        params = list(self._enc.parameters()) + list(self._dec.parameters())
        opt = torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
        for _ in range(int(epochs)):
            opt.zero_grad()
            mu, logvar = self._encode_std(xt)
            eps = torch.randn_like(mu)
            z = mu + eps * (0.5 * logvar).exp()  # reparameterized sample
            recon = self._dec(z)
            distortion = ((recon - xt) ** 2).sum(dim=-1).mean()
            rate = self._kl(mu, logvar).sum(dim=-1).mean()
            (distortion + self.beta * rate).backward()
            opt.step()
        self._fitted = True
        return self

    # -- inference ----------------------------------------------------------------------------
    def encode(self, X: Any) -> np.ndarray:
        """The embedding means ``(n, max_dim)`` -- the common code (use with a store's keys)."""
        torch = _torch()
        Xs = (np.atleast_2d(np.asarray(X, dtype=float)) - self._x_mean) / self._x_scale
        with torch.no_grad():
            mu, _ = self._encode_std(torch.as_tensor(Xs, dtype=torch.float64))
        return mu.cpu().numpy()

    def coordinate_kl(self, X: Any) -> np.ndarray:
        """Per-coordinate KL from the prior, ``(n, max_dim)`` (nats) -- how much each coord encodes."""
        torch = _torch()
        Xs = (np.atleast_2d(np.asarray(X, dtype=float)) - self._x_mean) / self._x_scale
        with torch.no_grad():
            mu, logvar = self._encode_std(torch.as_tensor(Xs, dtype=torch.float64))
            kl = self._kl(mu, logvar)
        return kl.cpu().numpy()

    def active_dim(self, X: Any) -> np.ndarray:
        """Per-input active dimension: number of coordinates whose KL exceeds ``kl_tau``."""
        return (self.coordinate_kl(X) > self.kl_tau).sum(axis=-1)

    def rate_nats(self, X: Any) -> np.ndarray:
        """Per-input total rate (sum of per-coordinate KL, nats) -- the information the code carries."""
        return self.coordinate_kl(X).sum(axis=-1)
