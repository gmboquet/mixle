"""Generative objective for fitting embeddings and optional codebooks.

Rather than tune the encoder to a label, train it to reconstruct its input as
an autoencoder over units. The shared-space vector must retain enough
information to rebuild the unit, so the representation has an explicit
generative objective. Add a
:class:`~mixle.represent.quantize.VectorQuantizer` and the model becomes a
VQ-VAE: encode -> quantize (straight-through) -> decode, with the codebook
periodically refit on the current embeddings. The learned vocabulary is then
selected by reconstruction quality instead of being fixed by a tokenizer chosen
outside the model.

``fit_autoencoder`` returns the trained encoder + decoder (+ codebook) and the reconstruction-loss history. It is
modality-agnostic: feed it the unit-feature array from any continuous segmenter (patches, windows, atoms, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.represent.embed import FeatureEmbedding, _positive_dimension
from mixle.represent.quantize import VectorQuantizer


class AutoencoderFitError(RuntimeError):
    """Reconstruction training did not complete, so no representation is returned.

    Carries the losses recorded before the failure and the epoch it happened on, so a caller can see
    where the fit diverged instead of receiving an ``AutoencoderResult`` that looks trained.
    """

    def __init__(self, message: str, losses: list[float], epoch: int) -> None:
        super().__init__(message)
        self.losses = losses
        self.epoch = epoch


def _finite_units(units: Any) -> Any:
    """``units`` as a rectangular, non-empty, finite ``(N, in_features)`` array.

    A one-epoch fit containing NaN used to return ``losses=[nan]`` and an encoder producing
    non-finite output, with no failed-fit state -- an artifact that could then enter the shared
    representation space as though reconstruction training had succeeded.
    """
    array = np.asarray(units)
    if array.dtype == object:
        raise ValueError("units must be a rectangular numeric array, not a ragged/object array")
    if array.ndim != 2:
        raise ValueError(f"units must be a 2-D (n_units, in_features) array, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"units must be non-empty in both dimensions, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("units must contain only finite values; a non-finite unit cannot be reconstructed")
    return array


@dataclass
class AutoencoderResult:
    """A reconstruction-trained representation with encoder, decoder, optional codebook, and loss curve.

    Only returned for a fit that ran every requested epoch to a finite loss: an untrained or diverged
    encoder is an :class:`AutoencoderFitError`, never a result. That sentence was previously a claim
    the class did not enforce (MXR-080-1906) -- with no ``__post_init__``,
    ``AutoencoderResult(encoder=FeatureEmbedding(6, 3), decoder=None, quantizer=None, losses=[])``
    was accepted and is indistinguishable from a fitted result, which is the exact state
    :class:`AutoencoderFitError` exists to keep out of the shared representation space.

    Two things are now bound rather than assumed:

    * ``losses`` must be a non-empty run of finite numbers. It is the only evidence that training
      happened at all, so a result carrying none is not a trained representation.
    * a ``quantizer``, when present, must be fitted. ``_train`` refits the codebook at epoch 0
      unconditionally, so every quantizer that reaches a real result has one.

    ``losses`` is also converted to a tuple. ``fit_autoencoder`` used to hand over the SAME list
    object the training loop was appending to, so the fitted loss curve stayed writable through a
    reference the result did not know about: ``res.losses.append(-999.0)`` rewrote recorded evidence
    (verified). Note ``AutoencoderFitError.losses`` stays a ``list`` -- it is type-tested as one by
    ``represent_generative_test`` and is already a defensive copy.

    Deliberately NOT enforced: that ``losses`` actually descends, or that the final loss is what
    ``encoder``/``decoder`` would reproduce on the training units. A legitimate fit may plateau or
    bounce, and re-running the forward pass to check would make constructing a result cost a full
    inference pass over data the result does not retain.
    """

    encoder: FeatureEmbedding
    decoder: Any
    quantizer: VectorQuantizer | None
    losses: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.encoder is None or self.decoder is None:
            raise ValueError("AutoencoderResult requires both a fitted encoder and a fitted decoder")
        try:
            curve = tuple(float(value) for value in self.losses)
        except (TypeError, ValueError) as exc:
            raise ValueError("losses must be a sequence of finite reconstruction losses") from exc
        if not curve:
            raise ValueError(
                "AutoencoderResult requires a non-empty loss curve: an empty one means no epoch ever "
                "ran, and an untrained representation must be an AutoencoderFitError, not a result"
            )
        if not all(np.isfinite(value) for value in curve):
            raise ValueError("losses must all be finite; a non-finite loss is a diverged fit, not a result")
        self.losses = curve
        if self.quantizer is not None and getattr(self.quantizer, "codebook", None) is None:
            raise ValueError(
                "AutoencoderResult was given an unfitted quantizer (no codebook). Training refits the "
                "codebook on the current embeddings before the first epoch ends, so a result's "
                "quantizer is always fitted."
            )

    def encode(self, units: np.ndarray) -> np.ndarray:
        """Encode units through the trained autoencoder encoder."""
        import torch

        with torch.no_grad():
            return self.encoder.module()(torch.as_tensor(np.asarray(units), dtype=torch.float32)).cpu().numpy()


def fit_autoencoder(
    units: np.ndarray,
    dim: int,
    *,
    hidden: tuple[int, ...] = (),
    quantizer: VectorQuantizer | None = None,
    epochs: int = 200,
    lr: float = 1e-2,
    refit_codebook_every: int = 25,
    commitment: float = 0.25,
    seed: int = 0,
) -> AutoencoderResult:
    """Train an encoder+decoder to reconstruct ``units`` ``(N, in_features)`` with an optional VQ bottleneck.

    Without ``quantizer`` this is a standard autoencoder. With one, it is a
    VQ-VAE: the encoder's vectors are quantized (straight-through) before
    decoding and the codebook is refit every ``refit_codebook_every`` epochs on
    the current embeddings. ``commitment`` weights the VQ codebook-commitment
    term.

    Architecture and training controls are validated rather than truncated: ``epochs=0``, ``-2``, and
    ``.9`` each used to return an ordinary result carrying a randomly initialized encoder and an
    empty loss history -- an untrained representation indistinguishable from a trained one. The
    global Torch RNG is seeded for reproducibility and restored on the way out, so fitting a
    representation does not silently reseed the caller's other Torch randomness.

    Raises:
        ValueError: for non-finite/ragged/empty units or a non-exact-positive control.
        AutoencoderFitError: if the loss goes non-finite -- training did not complete, so there is no
            trained representation to return.
    """
    import torch
    import torch.nn as nn

    array = _finite_units(units)
    dim = _positive_dimension("dim", dim)
    epochs = _positive_dimension("epochs", epochs)
    refit_codebook_every = _positive_dimension("refit_codebook_every", refit_codebook_every)
    lr = float(lr)
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"lr must be a finite positive learning rate, got {lr!r}")
    commitment = float(commitment)
    if not np.isfinite(commitment) or commitment < 0.0:
        raise ValueError(f"commitment must be finite and non-negative, got {commitment!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an exact integer, got {seed!r}")

    x = torch.as_tensor(array, dtype=torch.float32)
    in_features = x.shape[1]
    rng_state = torch.get_rng_state()
    torch.manual_seed(int(seed))

    encoder = FeatureEmbedding(in_features, dim, hidden=hidden)
    enc = encoder.module()
    dec_dims = [dim, *hidden, in_features]
    dec_layers: list = []
    for i in range(len(dec_dims) - 1):
        dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
        if i < len(dec_dims) - 2:
            dec_layers.append(nn.ReLU())
    decoder = nn.Sequential(*dec_layers)

    opt = torch.optim.Adam(list(enc.parameters()) + list(decoder.parameters()), lr=lr)
    losses: list[float] = []
    try:
        _train(
            x,
            enc,
            decoder,
            opt,
            losses,
            epochs=epochs,
            quantizer=quantizer,
            refit_codebook_every=refit_codebook_every,
            commitment=commitment,
            torch=torch,
        )
    finally:
        torch.set_rng_state(rng_state)

    return AutoencoderResult(encoder=encoder, decoder=decoder, quantizer=quantizer, losses=losses)


def _train(
    x: Any,
    enc: Any,
    decoder: Any,
    opt: Any,
    losses: list[float],
    *,
    epochs: int,
    quantizer: VectorQuantizer | None,
    refit_codebook_every: int,
    commitment: float,
    torch: Any,
) -> None:
    """Run the reconstruction loop, stopping the moment the objective stops being a number."""
    for epoch in range(epochs):
        opt.zero_grad()
        z = enc(x)  # (N, dim)
        if quantizer is not None:
            if quantizer.codebook is None or (epoch % refit_codebook_every == 0):
                quantizer.fit(z.detach().cpu().numpy())  # refit the codebook on the current embeddings
            zq = quantizer.straight_through(z)
            recon = decoder(zq)
            commit = commitment * torch.mean(
                (z - torch.as_tensor(quantizer.dequantize(quantizer.quantize(z.detach().cpu().numpy())), dtype=z.dtype))
                ** 2
            )
        else:
            recon = decoder(z)
            commit = torch.zeros((), dtype=z.dtype)
        loss = torch.mean((recon - x) ** 2) + commit
        value = float(loss.detach())
        if not np.isfinite(value):
            raise AutoencoderFitError(
                f"reconstruction loss became {value} at epoch {epoch}; training diverged and there is "
                "no trained representation to return",
                list(losses),
                epoch,
            )
        loss.backward()
        opt.step()
        losses.append(value)
