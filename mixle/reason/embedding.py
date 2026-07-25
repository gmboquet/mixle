"""A rate-adaptive embedding whose active dimension scales with information content.

A fixed-width embedding wastes capacity on low-information inputs and truncates high-information
ones. This encoder learns a latent code with a **variational** per-coordinate posterior
``q(z_k | x) = N(m_k(x), s_k(x)^2)`` and an ARD (automatic relevance determination) gate: a
coordinate whose posterior stays at its prior (``KL(q || p) ~ 0``) carries no information and is
*inactive*. The **active dimension of an input** is therefore ``#{k : KL(q(z_k|x) || p) > tau}`` --
it grows with the mutual information between the input and the latent.

Training is a rate--distortion (beta-VAE) objective: reconstruct the input subject to a rate budget
on the total KL. The ``beta`` knob sets bits-per-embedding; the data decides how those bits are
spent across coordinates, so a dense high-entropy input lights up more coordinates than a sparse one.

CONTRACT (MXR-080-0282): one instance is a plain autoencoder over ONE fixed input feature space. All
inputs it is fit and later queried on share that one instance's ordered coordinate system, so codes
produced by THE SAME FITTED INSTANCE are comparable to each other and can index a
:class:`mixle.reason.CrossModalStore` as one corpus's retrieval keys. This class has no modality
identity, no paired views, no contrastive/alignment objective, and no modality-specific front ends --
it does NOT align codes across *different* instances. Two independently-fit ``ScaledEmbedding``s
(e.g. one per modality) converge to arbitrary, unrelated latent rotations, so comparing a code from
one instance against a code from another (cosine similarity, nearest neighbor, shared coordinates,
...) is meaningless, even when both share the same ``max_dim``. For genuine cross-modal comparison,
use a mechanism built for it -- e.g. :class:`mixle.reason.CrossModalJoint` (a shared mixture-latent
joint over named modalities) or Bayesian evidence fusion via :func:`mixle.reason.reason` -- not
independently-fit embeddings.

Torch is imported lazily; :mod:`mixle.reason` exposes this via a deferred attribute.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _torch() -> Any:
    import torch

    return torch


def _require_positive_int(value: Any, name: str) -> int:
    """Validate ``value`` is an exact positive ``int`` (MXR-080-0280).

    A bare ``int(value)`` silently truncates a fractional width/dimension (``2.7`` quietly becomes
    ``2``) and accepts ``bool``; this rejects both, and any non-positive count.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    ivalue = int(value)
    if ivalue < 1:
        raise ValueError(f"{name} must be a positive int (>= 1), got {ivalue!r}")
    return ivalue


def _require_finite_positive_float(value: Any, name: str) -> float:
    """Validate ``value`` is a finite, strictly positive real number (MXR-080-0280)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}: {value!r}")
    fvalue = float(value)
    if not np.isfinite(fvalue):
        raise ValueError(f"{name} must be finite, got {fvalue!r}")
    if fvalue <= 0:
        raise ValueError(f"{name} must be positive, got {fvalue!r}")
    return fvalue


def _require_finite_nonnegative_float(value: Any, name: str) -> float:
    """Validate ``value`` is a finite, non-negative real number (MXR-080-0280)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}: {value!r}")
    fvalue = float(value)
    if not np.isfinite(fvalue):
        raise ValueError(f"{name} must be finite, got {fvalue!r}")
    if fvalue < 0:
        raise ValueError(f"{name} must be non-negative, got {fvalue!r}")
    return fvalue


def _require_2d_finite(X: Any, name: str, width: int) -> np.ndarray:
    """Validate ``X`` is a non-empty, finite, exactly ``width``-wide 2-D array (MXR-080-0280).

    Before this check, a bare ``np.atleast_2d(np.asarray(X, dtype=float))`` accepted anything: zero
    rows silently produced NaN standardization statistics (mean/std of an empty slice) instead of
    raising, a feature width that did not match the declared architecture blew up deep inside the
    network with a cryptic torch shape-mismatch error instead of a clear one, and NaN/inf entries
    silently propagated into NaN losses and NaN-corrupted weights instead of raising up front.
    """
    arr = np.atleast_2d(np.asarray(X, dtype=float))
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if arr.shape[1] != width:
        raise ValueError(f"{name} has width {arr.shape[1]}, expected {width}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    return arr


class ScaledEmbedding:
    """A beta-VAE-style rate-adaptive embedding with an ARD gate giving a data-dependent active dim.

    One instance covers a single fixed input feature space -- see the module docstring's CONTRACT
    (MXR-080-0282): codes are comparable within one fitted instance, never across independently-fit
    instances (e.g. one per modality has its own, unrelated latent rotation).

    Args:
        in_dim: input feature width.
        max_dim: the embedding's maximum width (upper bound on active dimension).
        hidden: hidden widths shared by the encoder and decoder trunks.
        beta: rate weight in the ELBO (larger -> tighter rate budget -> fewer active dims). Must be
            finite and non-negative.
        kl_tau: per-coordinate KL threshold (nats) above which a coordinate counts as active. Must be
            finite and non-negative.
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
        in_dim = _require_positive_int(in_dim, "in_dim")
        max_dim = _require_positive_int(max_dim, "max_dim")
        beta = _require_finite_nonnegative_float(beta, "beta")
        kl_tau = _require_finite_nonnegative_float(kl_tau, "kl_tau")
        torch = _torch()
        torch.manual_seed(int(seed))
        self.in_dim = in_dim
        self.max_dim = max_dim
        self.beta = beta
        self.kl_tau = kl_tau

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
        # No _fitted guard here: fit()'s own training loop calls this every step, before _fitted is
        # set -- the guard belongs on the public inference entry points (encode/coordinate_kl) below.
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
        """Train the embedding on unlabeled inputs ``X`` (``(n, in_dim)``) by the beta-VAE ELBO.

        ``X`` must be non-empty, finite, and exactly ``in_dim`` wide (MXR-080-0280). ``epochs`` must
        be a positive int: at least one optimizer step must run before the network's (otherwise still
        at its random initialization) weights are certified ``fitted``.
        """
        torch = _torch()
        X = _require_2d_finite(X, "X", self.in_dim)
        epochs = _require_positive_int(epochs, "epochs")
        lr = _require_finite_positive_float(lr, "lr")
        weight_decay = _require_finite_nonnegative_float(weight_decay, "weight_decay")
        self._x_mean = X.mean(axis=0)
        self._x_scale = X.std(axis=0) + 1e-8
        xt = torch.as_tensor((X - self._x_mean) / self._x_scale, dtype=torch.float64)
        params = list(self._enc.parameters()) + list(self._dec.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
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
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "ScaledEmbedding.encode/coordinate_kl/active_dim/rate_nats called before .fit(X) -- the "
                "network is still at its random initialization, so the returned code would be meaningless."
            )

    def encode(self, X: Any) -> np.ndarray:
        """The embedding means ``(n, max_dim)`` -- this instance's code (use as a store's keys).

        Only comparable to other codes from THIS fitted instance -- see the module docstring's
        CONTRACT (MXR-080-0282).
        """
        self._require_fitted()
        torch = _torch()
        Xs = (np.atleast_2d(np.asarray(X, dtype=float)) - self._x_mean) / self._x_scale
        with torch.no_grad():
            mu, _ = self._encode_std(torch.as_tensor(Xs, dtype=torch.float64))
        return mu.cpu().numpy()

    def coordinate_kl(self, X: Any) -> np.ndarray:
        """Per-coordinate KL from the prior, ``(n, max_dim)`` (nats) -- how much each coord encodes."""
        self._require_fitted()
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
