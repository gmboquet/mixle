"""Trainable cross-modal reasoning model with a shared latent.

This module learns per-modality encoders and decoders jointly from unlabeled
multimodal records. The latent is not observed; training maximizes a
Product-of-Experts variational lower bound for a multimodal VAE-style model.

* Each modality ``m`` has an encoder ``q_m(z | x_m) = N(mu_m, diag(sig_m^2))``.
* The belief given any *subset* of modalities is the **product of experts** with the prior:
  precisions add, so more modalities produce a sharper belief. This matches
  :meth:`mixle.inference.belief.GaussianBelief.fuse` with learned experts.
* Each modality has a decoder ``p(x_m | z)``; training reconstructs every modality from the fused
  latent. Modality-subset subsampling lets inference work from one modality,
  all modalities, or any subset between them.

After training: ``belief(obs)`` returns ``q(z | available modalities)`` as a
:class:`~mixle.inference.belief.GaussianBelief`, and ``predict(obs, target)``
generates a missing modality from the available ones. Uncertainty remains part
of the object: the returned belief is a distribution, sharpened by each
modality in proportion to its learned precision.

Torch is imported lazily; :mod:`mixle.reason` exposes this via a deferred attribute.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.inference.belief import GaussianBelief


def _torch() -> Any:
    import torch

    return torch


def _mlp(sizes: list[int], torch: Any) -> Any:
    layers: list[Any] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
    return torch.nn.Sequential(*layers[:-1]).double()  # drop trailing ReLU


class _Modality:
    def __init__(self, name: str, in_dim: int, latent_dim: int, hidden: tuple[int, ...], torch: Any) -> None:
        self.name = name
        self.in_dim = in_dim
        self.encoder = _mlp([in_dim, *hidden, 2 * latent_dim], torch)  # -> (mu, logvar)
        self.decoder = _mlp([latent_dim, *hidden[::-1], in_dim], torch)
        self.mean = np.zeros(in_dim)
        self.scale = np.ones(in_dim)


class CrossModalModel:
    """A multimodal Product-of-Experts VAE: one shared latent learned from many modalities, unsupervised.

    Args:
        latent_dim: dimension of the shared latent ``z``.
        seed: torch RNG seed.
    """

    def __init__(self, latent_dim: int, *, seed: int = 0) -> None:
        torch = _torch()
        torch.manual_seed(int(seed))
        self.latent_dim = int(latent_dim)
        self._mods: dict[str, _Modality] = {}
        self._fitted = False
        self._conformal: dict[str, tuple[float, np.ndarray, float]] = {}  # target -> (alpha, scale, q)

    def add_modality(self, name: str, in_dim: int, *, hidden: tuple[int, ...] = (64,)) -> CrossModalModel:
        """Register a modality with learned encoder ``q(z|x)`` and decoder ``p(x|z)``."""
        self._mods[name] = _Modality(name, int(in_dim), self.latent_dim, tuple(hidden), _torch())
        return self

    # -- posterior over the latent (product of experts) ---------------------------------------
    def _expert(self, mod: _Modality, x_std: Any) -> tuple[Any, Any]:
        out = mod.encoder(x_std)
        mu = out[..., : self.latent_dim]
        logvar = out[..., self.latent_dim :].clamp(-10.0, 10.0)
        return mu, logvar

    def _poe(self, experts: list[tuple[Any, Any]]) -> tuple[Any, Any]:
        """Fuse Gaussian experts with a unit-Gaussian prior in precision space -> (mu, var)."""
        torch = _torch()
        # prior N(0, I): precision 1, precision-weighted mean 0
        prec = None
        pmean = None
        for mu, logvar in experts:
            p = torch.exp(-logvar)
            prec = p if prec is None else prec + p
            pmean = mu * p if pmean is None else pmean + mu * p
        one = torch.ones_like(prec) if prec is not None else None
        prec = one + prec  # add prior precision
        var = 1.0 / prec
        mean = var * pmean
        return mean, var

    # -- training -----------------------------------------------------------------------------
    def fit(
        self,
        data: dict[str, Any],
        *,
        epochs: int = 600,
        lr: float = 3e-3,
        beta: float = 0.5,
        subsample: bool = True,
    ) -> CrossModalModel:
        """Train encoders and decoders jointly on unlabeled multimodal data.

        ``data`` maps each registered modality name to an ``(N, in_dim)`` array (all modalities share
        the same ``N`` rows -- row ``i`` is one record's several views). ``beta`` weights the KL rate;
        with ``subsample=True`` the ELBO is also evaluated on each single-modality subset so the model
        can infer ``z`` from any one modality alone (the MVAE training trick).
        """
        torch = _torch()
        names = list(self._mods)
        if set(data) != set(names):
            raise ValueError(f"data modalities {sorted(data)} != registered {sorted(names)}")
        n = len(next(iter(data.values())))
        tensors: dict[str, Any] = {}
        for name in names:
            X = np.atleast_2d(np.asarray(data[name], dtype=float))
            mod = self._mods[name]
            mod.mean = X.mean(axis=0)
            mod.scale = X.std(axis=0) + 1e-8
            tensors[name] = torch.as_tensor((X - mod.mean) / mod.scale, dtype=torch.float64)

        params: list[Any] = []
        for mod in self._mods.values():
            params += list(mod.encoder.parameters()) + list(mod.decoder.parameters())
        opt = torch.optim.Adam(params, lr=float(lr))

        # subsets to train on: the full set, plus each singleton (so unimodal inference is learned).
        subsets: list[list[str]] = [names]
        if subsample and len(names) > 1:
            subsets += [[name] for name in names]

        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = torch.zeros((), dtype=torch.float64)
            for subset in subsets:
                experts = [self._expert(self._mods[m], tensors[m]) for m in subset]
                mean, var = self._poe(experts)
                eps = torch.randn_like(mean)
                z = mean + eps * var.sqrt()  # reparameterized sample
                # reconstruct EVERY modality from this subset's latent (cross-modal generation)
                recon = torch.zeros((), dtype=torch.float64)
                for name in names:
                    xhat = self._mods[name].decoder(z)
                    recon = recon + ((xhat - tensors[name]) ** 2).sum(dim=-1).mean()
                kl = (-0.5 * (1.0 + var.log() - mean**2 - var)).sum(dim=-1).mean()
                loss = loss + recon + float(beta) * kl
            (loss / len(subsets)).backward()
            opt.step()
        self._fitted = True
        self._n_train = n
        return self

    # -- inference ----------------------------------------------------------------------------
    def belief(self, obs: dict[str, Any]) -> GaussianBelief:
        """The belief ``q(z | available modalities)`` as a :class:`GaussianBelief` (product of experts)."""
        torch = _torch()
        if not obs:
            return GaussianBelief(np.zeros(self.latent_dim), np.eye(self.latent_dim))
        experts = []
        for name, x in obs.items():
            mod = self._mods[name]
            xs = (np.atleast_2d(np.asarray(x, dtype=float)) - mod.mean) / mod.scale
            with torch.no_grad():
                experts.append(self._expert(mod, torch.as_tensor(xs, dtype=torch.float64)))
        with torch.no_grad():
            mean, var = self._poe(experts)
        m = mean.cpu().numpy()[0]
        v = var.cpu().numpy()[0]
        return GaussianBelief(m, np.diag(v))

    def encode(self, obs: dict[str, Any]) -> np.ndarray:
        """The posterior-mean latent code for ``obs`` (a shared-space embedding usable as store keys)."""
        return self.belief(obs).mean()

    def predict(self, obs: dict[str, Any], target: str) -> np.ndarray:
        """Generate the ``target`` modality from the modalities in ``obs`` (cross-modal generation)."""
        torch = _torch()
        if target not in self._mods:
            raise KeyError(f"unknown modality {target!r}")
        z = self.belief(obs).mean()
        mod = self._mods[target]
        with torch.no_grad():
            xhat = mod.decoder(torch.as_tensor(z[None, :], dtype=torch.float64)).cpu().numpy()[0]
        return xhat * mod.scale + mod.mean

    # -- distribution-free (conformal) calibration --------------------------------------------
    def calibrate(self, cal_data: dict[str, Any], target: str, *, alpha: float = 0.1) -> CrossModalModel:
        """Calibrate cross-modal prediction of ``target`` for finite-sample coverage (split conformal).

        On a held-out calibration set, predict ``target`` from the *other* modalities, normalize the
        per-dimension residuals, and take the ``ceil((n+1)(1-alpha))``-th largest max-normalized
        residual as the conformal radius. Using the *max* over dimensions makes the guarantee
        **simultaneous**: :meth:`predict_interval` returns a box whose *joint* coverage over the whole
        target vector is ``>= 1 - alpha`` -- distribution-free, regardless of model specification
        (unlike the Gaussian posterior interval).
        """
        if target not in self._mods:
            raise KeyError(f"unknown modality {target!r}")
        others = [m for m in self._mods if m != target]
        if not others:
            raise ValueError("need at least one other modality to predict the target from")
        y = np.atleast_2d(np.asarray(cal_data[target], dtype=float))
        n = len(y)
        preds = np.array([self.predict({o: cal_data[o][i] for o in others}, target) for i in range(n)])
        resid = np.abs(y - preds)  # (n, dim)
        scale = resid.std(axis=0) + 1e-8  # per-dim normalization so no dimension dominates the box
        scores = (resid / scale).max(axis=1)  # (n,) max-normalized nonconformity -> simultaneous cover
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        q = float(np.sort(scores)[min(k, n) - 1]) if k <= n else float(scores.max())
        self._conformal[target] = (float(alpha), scale, q)
        return self

    def predict_interval(self, obs: dict[str, Any], target: str) -> tuple[np.ndarray, np.ndarray]:
        """A conformally-calibrated prediction box ``(lower, upper)`` for ``target`` given ``obs``.

        Requires a prior :meth:`calibrate` call for ``target``. Coverage is distribution-free and
        *simultaneous*: ``P(y in box) >= 1 - alpha`` jointly over the whole target vector.
        """
        if target not in self._conformal:
            raise RuntimeError(f"call calibrate(..., target={target!r}) before predict_interval")
        _, scale, q = self._conformal[target]
        yhat = self.predict(obs, target)
        radius = q * scale
        return yhat - radius, yhat + radius

    @property
    def modalities(self) -> Sequence[str]:
        """Return modality names known to the cross-modal model."""
        return list(self._mods)
