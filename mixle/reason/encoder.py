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
    raising; a feature width that did not match the declared architecture either blew up deep inside
    the network with a cryptic torch shape-mismatch error, or -- when the mismatched axis happened to
    be size 1 -- silently broadcast against a differently-sized target instead of raising; and NaN/inf
    entries silently propagated into NaN losses and NaN-corrupted weights instead of raising up front.
    """
    arr = np.atleast_2d(np.asarray(X, dtype=float))
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if arr.shape[1] != width:
        raise ValueError(f"{name} has width {arr.shape[1]}, expected {width}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    return arr


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
        in_dim = _require_positive_int(in_dim, "in_dim")
        latent_dim = _require_positive_int(latent_dim, "latent_dim")
        # min_sd floors the predicted sd (see _forward_std); zero or negative defeats its documented
        # purpose of "preventing an over-confident zero-variance expert" (MXR-080-0280).
        min_sd = _require_finite_positive_float(min_sd, "min_sd")
        torch = _torch()
        torch.manual_seed(int(seed))
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.min_sd = min_sd
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
        if not self._fitted:
            raise RuntimeError(
                "AmortizedEncoder.encode/encode_batch/evidence called before .fit(X, Z) -- the network "
                "is still at its random initialization, so the returned belief would be meaningless."
            )
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

        ``X`` is ``(n, in_dim)`` modality features, ``Z`` is ``(n, latent_dim)`` latent targets --
        both must be non-empty, finite, and exactly the declared width (MXR-080-0280: a ``Z`` whose
        width did not match ``latent_dim`` used to be accepted silently -- and, when its width was 1,
        silently broadcast against the network's ``latent_dim``-wide output instead of raising).
        Inputs and targets are standardized internally for stable optimization. ``epochs`` must be a
        positive int: at least one optimizer step must run before the network's (otherwise still at
        its random initialization) weights are certified ``fitted``.
        """
        torch = _torch()
        X = _require_2d_finite(X, "X", self.in_dim)
        Z = _require_2d_finite(Z, "Z", self.latent_dim)
        if X.shape[0] != Z.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but Z has {Z.shape[0]}")
        epochs = _require_positive_int(epochs, "epochs")
        lr = _require_finite_positive_float(lr, "lr")
        weight_decay = _require_finite_nonnegative_float(weight_decay, "weight_decay")
        self._x_mean = X.mean(axis=0)
        self._x_scale = X.std(axis=0) + 1e-8
        self._z_mean = Z.mean(axis=0)
        self._z_scale = Z.std(axis=0) + 1e-8
        Xs = (X - self._x_mean) / self._x_scale
        Zs = (Z - self._z_mean) / self._z_scale
        xt = torch.as_tensor(Xs, dtype=torch.float64)
        zt = torch.as_tensor(Zs, dtype=torch.float64)
        opt = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
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
        encoder targets the whole latent). ``onto``'s row count must equal ``latent_dim`` exactly, so
        ``H`` produces one row per coordinate of this encoder's Gaussian output, matching ``y = H z``
        (MXR-080-0280: an inconsistent ``onto`` used to be accepted unchecked, surfacing -- if at all
        -- as a cryptic shape error deep inside belief assimilation instead of here, at its source).
        """
        mu, var = self._encode_std(np.atleast_2d(np.asarray(x, dtype=float)))
        if onto is None:
            H = np.eye(self.latent_dim)
        else:
            H = np.atleast_2d(np.asarray(onto, dtype=float))
            if not np.isfinite(H).all():
                raise ValueError("onto must be finite (no NaN or inf)")
            if H.shape[0] != self.latent_dim:
                raise ValueError(
                    f"onto must have shape (latent_dim, full_dim) with latent_dim={self.latent_dim}; "
                    f"got shape {H.shape}"
                )
            if H.shape[1] < 1:
                raise ValueError(f"onto must select from a non-empty latent (at least one column), got shape {H.shape}")
        return LinearGaussianEvidence(H, mu[0], np.diag(var[0]), name or "encoder")
