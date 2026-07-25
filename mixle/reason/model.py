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


def _require_positive_int(value: Any, name: str) -> int:
    """Validate ``value`` is an exact, positive :class:`int` (MXR-080-0277).

    Mirrors this codebase's ``_require_count`` convention (e.g. ``mixle.substrate.multihop``): a
    plain Python ``int``, never a ``bool`` (a ``bool`` is an ``int`` subclass and would otherwise
    silently mean 0 or 1) and never a float (even a whole-valued one like ``2.0``, silently
    truncated by a bare ``int()``). Unlike ``_require_count`` (which allows 0 for a "how many"
    count), zero or negative is rejected outright here: training for zero or a negative number of
    epochs never updates the randomly-initialized encoder/decoder weights at all, so silently
    accepting it and still marking the model ``_fitted`` would certify random noise as trained.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


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
        self._n_train: int | None = None
        self._conformal: dict[str, tuple[float, np.ndarray, float]] = {}  # target -> (alpha, scale, q)

    def _invalidate_fitted_state(self) -> None:
        """Clear every fitted/calibration artifact after a structural mutation (MXR-080-0276).

        ``_fitted`` describes the WHOLE model: every registered modality is (supposedly) trained
        jointly through the shared latent, and every stored conformal radius was computed against
        the modalities' encoders/decoders *as they existed at calibration time*. Registering or
        replacing a modality changes that structure, so anything that depended on the old structure
        -- the fitted flag, training metadata, and every target's conformal radius, not just the
        touched modality's -- must be cleared together, in the same operation as the mutation. A
        caller must :meth:`fit` (and re-:meth:`calibrate` any target it cares about) again before
        beliefs/predictions/intervals are trustworthy.
        """
        self._fitted = False
        self._n_train = None
        self._conformal = {}

    def add_modality(self, name: str, in_dim: int, *, hidden: tuple[int, ...] = (64,)) -> CrossModalModel:
        """Register a NEW modality with a learned encoder ``q(z|x)`` and decoder ``p(x|z)``.

        MXR-080-0276: a duplicate ``name`` is rejected rather than silently replaced. Silently
        replacing a registered modality's encoder/decoder previously left ``_fitted`` and any
        stored conformal radii intact, so a later prediction could combine a fresh, untrained
        replacement encoder/decoder with calibration computed against the OLD one and still
        advertise calibrated coverage. Use :meth:`replace_modality` to deliberately replace an
        existing modality -- it invalidates fitted/calibration state as part of the same operation.

        Raises:
            ValueError: ``name`` is already registered.
        """
        if name in self._mods:
            raise ValueError(
                f"modality {name!r} is already registered (in_dim={self._mods[name].in_dim}); "
                f"use replace_modality({name!r}, ...) to deliberately replace it"
            )
        self._mods[name] = _Modality(name, int(in_dim), self.latent_dim, tuple(hidden), _torch())
        self._invalidate_fitted_state()
        return self

    def replace_modality(self, name: str, in_dim: int, *, hidden: tuple[int, ...] = (64,)) -> CrossModalModel:
        """Deliberately replace a registered modality with a fresh, untrained encoder/decoder pair.

        MXR-080-0276: unlike :meth:`add_modality` (which rejects a duplicate name outright), this
        method exists specifically to allow overwriting an existing modality's encoder/decoder --
        but doing so invalidates every dependent fitted/calibration artifact in the SAME operation
        (see :meth:`_invalidate_fitted_state`): ``_fitted`` is cleared, training metadata is
        cleared, and every stored conformal radius is cleared, since each was computed against this
        modality's prior encoder/decoder. Call :meth:`fit` (and re-:meth:`calibrate` any target)
        again before relying on beliefs/predictions/intervals.

        Raises:
            KeyError: ``name`` is not already registered.
        """
        if name not in self._mods:
            raise KeyError(f"modality {name!r} is not registered; use add_modality({name!r}, ...) instead")
        self._mods[name] = _Modality(name, int(in_dim), self.latent_dim, tuple(hidden), _torch())
        self._invalidate_fitted_state()
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
    def _validate_training_table(self, data: dict[str, Any], names: list[str]) -> dict[str, np.ndarray]:
        """Validate the COMPLETE aligned training table before any state mutation (MXR-080-0277).

        Every modality's data must be non-empty, match its registered ``in_dim``, and be finite;
        every modality must share the SAME row count (rows are aligned records: row ``i`` of every
        modality is that record's several views, so a length mismatch means there is no such
        alignment). Previously ``n`` was read from one arbitrary modality and never checked against
        the rest, so a mismatch either crashed deep inside torch with an opaque shape error, or --
        when shapes happened to be broadcast-compatible (e.g. one modality with a single row) --
        silently trained against the WRONG pairing of rows instead of failing. Everything is
        checked here, up front, against a single reference row count, so a misaligned or malformed
        table is rejected with a clear error naming the modality and the failed check, before
        ``fit`` changes any per-modality normalization statistic or the model's fitted state.
        """
        arrays: dict[str, np.ndarray] = {}
        n: int | None = None
        ref_name = names[0]
        for name in names:
            X = np.atleast_2d(np.asarray(data[name], dtype=float))
            mod = self._mods[name]
            if X.shape[0] == 0:
                raise ValueError(f"modality {name!r} has no rows (empty training data)")
            if X.shape[1] != mod.in_dim:
                raise ValueError(
                    f"modality {name!r} declared in_dim={mod.in_dim} but training data has width {X.shape[1]}"
                )
            if not np.all(np.isfinite(X)):
                raise ValueError(f"modality {name!r} training data contains non-finite values (NaN/Inf)")
            if n is None:
                n, ref_name = X.shape[0], name
            elif X.shape[0] != n:
                raise ValueError(
                    f"modality {name!r} has {X.shape[0]} rows but modality {ref_name!r} has {n} rows; "
                    "all modalities must share the same row count (aligned records)"
                )
            arrays[name] = X
        return arrays

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

        MXR-080-0277: the complete aligned training table -- equal row counts across every
        modality, each modality's declared ``in_dim`` matching its data's actual width, non-empty
        data, and finite values -- is validated up front, before any per-modality normalization
        statistic or the model's fitted state is touched. ``epochs`` is validated as a genuine
        positive ``int``: a zero or negative value previously still set ``_fitted = True`` on
        encoders/decoders that were never actually updated (their random init weights, dressed up
        as a "trained" model).

        Raises:
            ValueError: ``data``'s modalities don't match the registered set; a modality's data is
                empty, has the wrong width, contains non-finite values, or has a row count that
                disagrees with the other modalities; or ``epochs`` is not positive.
            TypeError: ``epochs`` is not a plain ``int`` (a ``bool`` or a float, whole-valued or
                not, is rejected).
        """
        torch = _torch()
        names = list(self._mods)
        if not names:
            raise ValueError("fit() requires at least one registered modality; call add_modality() first")
        if set(data) != set(names):
            raise ValueError(f"data modalities {sorted(data)} != registered {sorted(names)}")
        epochs = _require_positive_int(epochs, "epochs")
        arrays = self._validate_training_table(data, names)

        n = arrays[names[0]].shape[0]  # validated equal across every modality above
        tensors: dict[str, Any] = {}
        for name in names:
            X = arrays[name]
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

        for _ in range(epochs):
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
        """The belief ``q(z | available modalities)`` as a :class:`GaussianBelief` (product of experts).

        Requires a prior :meth:`fit` call. ``_fitted`` was tracked but never checked: a freshly
        constructed model's encoders carry their random init weights, so every downstream consumer
        of this method -- :meth:`encode`, :meth:`predict`, :meth:`calibrate`, :meth:`predict_interval`
        -- could silently return a meaningless, randomly-initialized "belief" with no indication
        anything was wrong. Checked here once, since all four route through this method.

        ``obs`` carries exactly ONE observation per modality: each value must be a 1-D array of
        length ``in_dim`` (MXR-080-0278). Previously ``np.atleast_2d`` silently accepted a
        ``(B, in_dim)`` batch, fused it through the product-of-experts as if it were legitimate --
        modalities with unequal batch sizes could broadcast into unintended cross-row pairings --
        and then returned only row 0, discarding the rest with no signal anything was dropped.
        No caller in this codebase passes a genuine batch here; call this once per observation.

        Raises:
            ValueError: some modality's observation is not a 1-D array of the registered width.
        """
        if not self._fitted:
            raise RuntimeError("CrossModalModel.belief() called before fit(): the encoders are untrained.")
        torch = _torch()
        if not obs:
            return GaussianBelief(np.zeros(self.latent_dim), np.eye(self.latent_dim))
        experts = []
        for name, x in obs.items():
            mod = self._mods[name]
            xa = np.asarray(x, dtype=float)
            if xa.shape != (mod.in_dim,):
                raise ValueError(
                    f"belief() takes exactly one observation per modality (a 1-D array of length "
                    f"in_dim); modality {name!r} has in_dim={mod.in_dim} but got an array of shape "
                    f"{xa.shape} -- pass a single row, not a batch"
                )
            xs = (np.atleast_2d(xa) - mod.mean) / mod.scale
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
        """Generate the ``target`` modality from the modalities in ``obs`` (cross-modal generation).

        ``obs`` takes exactly one observation per modality; see :meth:`belief` (MXR-080-0278) --
        this method inherits that contract entirely through its call below.
        """
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

        MXR-080-0279: every split-conformal precondition is validated and the method fails closed
        -- raises without storing anything -- before a single score is computed. Previously,
        ``alpha`` outside ``(0, 1)`` could select a negative-indexed or otherwise unrelated order
        statistic instead of raising; an empty or wrong-width holdout produced an opaque
        broadcasting ``ValueError`` (or worse, silently proceeded on a mismatched pairing); and
        non-finite holdout values propagated into a NaN radius stored exactly like a valid one. A
        stored radius is later presented, via :meth:`predict_interval`, as carrying a genuine
        finite-sample coverage guarantee, so it must never come from anything but a valid
        computation on a valid holdout.

        Raises:
            KeyError: ``target`` is not registered, or ``cal_data`` is missing a modality needed to
                predict it.
            ValueError: ``alpha`` is not strictly between 0 and 1; the holdout is empty; a
                modality's holdout width doesn't match its registered ``in_dim``; a modality's
                holdout row count disagrees with the others; or any holdout value is non-finite.
        """
        if target not in self._mods:
            raise KeyError(f"unknown modality {target!r}")
        others = [m for m in self._mods if m != target]
        if not others:
            raise ValueError("need at least one other modality to predict the target from")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
        missing = [m for m in (*others, target) if m not in cal_data]
        if missing:
            raise KeyError(f"cal_data is missing modalities {missing} needed to calibrate target {target!r}")

        target_mod = self._mods[target]
        y = np.atleast_2d(np.asarray(cal_data[target], dtype=float))
        if y.shape[0] == 0:
            raise ValueError(f"calibration holdout for target {target!r} is empty")
        if y.shape[1] != target_mod.in_dim:
            raise ValueError(
                f"target {target!r} declared in_dim={target_mod.in_dim} but holdout data has width {y.shape[1]}"
            )
        if not np.all(np.isfinite(y)):
            raise ValueError(f"calibration holdout for target {target!r} contains non-finite values (NaN/Inf)")
        n = y.shape[0]

        for name in others:
            mod = self._mods[name]
            X = np.atleast_2d(np.asarray(cal_data[name], dtype=float))
            if X.shape[0] != n:
                raise ValueError(
                    f"modality {name!r} holdout has {X.shape[0]} rows but target {target!r} holdout "
                    f"has {n} rows; all modalities must share the same row count"
                )
            if X.shape[1] != mod.in_dim:
                raise ValueError(
                    f"modality {name!r} declared in_dim={mod.in_dim} but holdout data has width {X.shape[1]}"
                )
            if not np.all(np.isfinite(X)):
                raise ValueError(f"modality {name!r} holdout data contains non-finite values (NaN/Inf)")

        preds = np.array([self.predict({o: cal_data[o][i] for o in others}, target) for i in range(n)])
        resid = np.abs(y - preds)  # (n, dim)
        scale = resid.std(axis=0) + 1e-8  # per-dim normalization so no dimension dominates the box
        scores = (resid / scale).max(axis=1)  # (n,) max-normalized nonconformity -> simultaneous cover
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        # finite-sample split conformal: when ceil((n+1)(1-alpha)) exceeds n, no calibration score
        # certifies the level -- the radius is +inf (the box is unbounded), not the max score. Mirrors
        # mixle.scientist.study()'s identical k > n handling for the same split-conformal edge case.
        # (alpha is now guaranteed in (0, 1) and n >= 1 above, so k >= 1 always -- never <= 0.)
        q = float(np.sort(scores)[k - 1]) if k <= n else float("inf")
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
