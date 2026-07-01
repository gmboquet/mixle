"""Amortized modality encoders: raw features -> a Gaussian expert about the latent.

A generative proxy models ``p(x | z)`` and is exact but expensive for high-dimensional modalities
(images, spectra, long series). The scalable alternative is *amortized*: learn a network
``enc(x) -> N(mu(x), diag(sigma^2(x)))`` that maps a modality's features directly to a Gaussian
belief about the shared latent -- a **soft observation**, fused with other modalities as a product
of experts (:meth:`mixle.inference.belief.GaussianBelief.fuse`) and consumed by
:func:`mixle.reason.reason` as evidence.

The encoder is **heteroscedastic**: it learns to report a *smaller* variance on informative inputs
and a *larger* one on ambiguous inputs, so product-of-experts fusion automatically down-weights a
modality exactly where it does not know -- the behavior a fixed-noise (homoscedastic) head cannot
express. Training is amortized probabilistic regression (Gaussian negative log-likelihood over
``(x, z)`` pairs); at inference, encoding is a single forward pass.

Torch is imported lazily inside this module, so the encoder's network is only built when an encoder
is actually constructed (``mixle.reason`` exposes it via a deferred attribute). Domain-neutral:
application-specific encoders (a seismic-trace encoder, a spectra encoder) subclass or configure
this in the ``mixle_pde`` layer, not here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.inference.belief import GaussianBelief
from mixle.reason.core import LinearGaussianEvidence


def _torch() -> Any:
    import torch

    return torch


def _build_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, torch: Any) -> Any:
    layers: list[Any] = []
    prev = in_dim
    for h in hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
        prev = h
    layers.append(torch.nn.Linear(prev, out_dim))
    return torch.nn.Sequential(*layers)


class AmortizedEncoder:
    """A learned encoder mapping modality features to a diagonal-Gaussian belief about the latent.

    Args:
        in_dim: width of the input feature vector.
        latent_dim: dimension of the (sub-)latent this encoder informs.
        hidden: hidden-layer widths of the MLP trunk.
        min_sd: floor on the predicted standard deviation (in latent units), preventing an
            over-confident zero-variance expert.
        seed: torch RNG seed for reproducible initialization/training.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        *,
        hidden: tuple[int, ...] = (64,),
        min_sd: float = 1e-3,
        seed: int = 0,
    ) -> None:
        torch = _torch()
        torch.manual_seed(int(seed))
        self.in_dim = int(in_dim)
        self.latent_dim = int(latent_dim)
        self.min_sd = float(min_sd)
        self._net = _build_mlp(self.in_dim, tuple(hidden), 2 * self.latent_dim, torch).double()
        # standardization stats (filled by fit); identity until then.
        self._x_mean = np.zeros(self.in_dim)
        self._x_scale = np.ones(self.in_dim)
        self._z_mean = np.zeros(self.latent_dim)
        self._z_scale = np.ones(self.latent_dim)
        self._fitted = False

    # -- internals ----------------------------------------------------------------------------
    def _forward_std(self, x_std: Any) -> tuple[Any, Any]:
        """Network forward in standardized space -> (mean_std, var_std) as torch tensors."""
        torch = _torch()
        out = self._net(x_std)
        mu = out[..., : self.latent_dim]
        raw = out[..., self.latent_dim :]
        min_var_std = (self.min_sd / self._z_scale) ** 2  # floor, in standardized units
        floor = torch.as_tensor(min_var_std, dtype=out.dtype)
        var = floor + torch.nn.functional.softplus(raw)
        return mu, var

    def _encode_std(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Batched encode -> (means, vars) in *original* latent units, as numpy arrays."""
        torch = _torch()
        Xs = (np.atleast_2d(np.asarray(X, dtype=float)) - self._x_mean) / self._x_scale
        with torch.no_grad():
            mu_std, var_std = self._forward_std(torch.as_tensor(Xs, dtype=torch.float64))
        mu = mu_std.cpu().numpy() * self._z_scale + self._z_mean
        var = var_std.cpu().numpy() * self._z_scale**2
        return mu, var

    # -- training -----------------------------------------------------------------------------
    def fit(
        self,
        X: Any,
        Z: Any,
        *,
        epochs: int = 300,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
    ) -> AmortizedEncoder:
        """Train the encoder on ``(X, Z)`` pairs by heteroscedastic Gaussian negative log-likelihood.

        ``X`` is ``(n, in_dim)`` modality features, ``Z`` is ``(n, latent_dim)`` latent targets.
        Inputs and targets are standardized internally for stable optimization.
        """
        torch = _torch()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if X.shape[0] != Z.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but Z has {Z.shape[0]}")
        self._x_mean = X.mean(axis=0)
        self._x_scale = X.std(axis=0) + 1e-8
        self._z_mean = Z.mean(axis=0)
        self._z_scale = Z.std(axis=0) + 1e-8
        Xs = (X - self._x_mean) / self._x_scale
        Zs = (Z - self._z_mean) / self._z_scale
        xt = torch.as_tensor(Xs, dtype=torch.float64)
        zt = torch.as_tensor(Zs, dtype=torch.float64)
        opt = torch.optim.Adam(self._net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        for _ in range(int(epochs)):
            opt.zero_grad()
            mu, var = self._forward_std(xt)
            # Gaussian NLL: 0.5 (log var + (z-mu)^2 / var), averaged.
            nll = 0.5 * (torch.log(var) + (zt - mu) ** 2 / var).sum(dim=-1).mean()
            nll.backward()
            opt.step()
        self._fitted = True
        return self

    # -- inference ----------------------------------------------------------------------------
    def encode(self, x: Any) -> GaussianBelief:
        """Encode one input into a diagonal-Gaussian belief ``N(mu(x), diag(sigma^2(x)))``."""
        mu, var = self._encode_std(np.atleast_2d(np.asarray(x, dtype=float)))
        return GaussianBelief(mu[0], np.diag(var[0]))

    def encode_batch(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Encode a batch -> ``(means (n, d), variances (n, d))`` in latent units."""
        return self._encode_std(X)

    def evidence(self, x: Any, *, onto: Any = None, name: str = "") -> LinearGaussianEvidence:
        """A :class:`LinearGaussianEvidence` for :func:`mixle.reason.reason` from encoding ``x``.

        The encoder's Gaussian output ``N(mu, diag(var))`` is a direct observation of its target
        latent. ``onto`` optionally maps a *larger* shared latent onto this encoder's target space
        (a readout / selector matrix, shape ``(latent_dim, full_dim)``); by default ``H = I`` (the
        encoder targets the whole latent).
        """
        mu, var = self._encode_std(np.atleast_2d(np.asarray(x, dtype=float)))
        H = np.eye(self.latent_dim) if onto is None else np.atleast_2d(np.asarray(onto, dtype=float))
        return LinearGaussianEvidence(H, mu[0], np.diag(var[0]), name or "encoder")
