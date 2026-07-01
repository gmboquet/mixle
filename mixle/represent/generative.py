"""Generative objective -- fit the embedding (and its codebook) to *model* the data, so tokenization is inferred.

The self-supervised half of "fit to the objective": rather than tune the encoder to a label, train it to
*reconstruct* its input -- an autoencoder over units. The shared-space vector must retain enough to rebuild the
unit, so the representation is a generative one (no collapse). Turn on a :class:`~mixle.represent.quantize.VectorQuantizer`
and it becomes a VQ-VAE: encode -> quantize (straight-through) -> decode, with the codebook periodically refit on
the current embeddings. Now the *vocabulary* is chosen to best reconstruct the data -- tokenization inferred under
a generative objective, exactly the thing a hardcoded BPE cannot do.

``fit_autoencoder`` returns the trained encoder + decoder (+ codebook) and the reconstruction-loss history. It is
modality-agnostic: feed it the unit-feature array from any continuous segmenter (patches, windows, atoms, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.represent.embed import FeatureEmbedding
from mixle.represent.quantize import VectorQuantizer


@dataclass
class AutoencoderResult:
    """A generatively-trained representation: the encoder, its decoder, an optional codebook, and the loss curve."""

    encoder: FeatureEmbedding
    decoder: Any
    quantizer: VectorQuantizer | None
    losses: list[float] = field(default_factory=list)

    def encode(self, units: np.ndarray) -> np.ndarray:
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
    """Train an encoder+decoder to reconstruct ``units`` ``(N, in_features)``; optionally through a VQ bottleneck.

    Without ``quantizer`` this is a plain autoencoder (the encoder becomes a generative representation). With one,
    it is a VQ-VAE: the encoder's vectors are quantized (straight-through) before decoding and the codebook is
    refit every ``refit_codebook_every`` epochs on the current embeddings, so the learned vocabulary adapts to the
    representation. ``commitment`` weights the VQ codebook-commitment term.
    """
    import torch
    import torch.nn as nn

    x = torch.as_tensor(np.asarray(units), dtype=torch.float32)
    in_features = x.shape[1]
    torch.manual_seed(seed)

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
    for epoch in range(int(epochs)):
        opt.zero_grad()
        z = enc(x)  # (N, dim)
        if quantizer is not None:
            if quantizer.codebook is None or (epoch % max(1, refit_codebook_every) == 0):
                quantizer.fit(z.detach().cpu().numpy())  # refit the vocabulary on the current embeddings
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
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    return AutoencoderResult(encoder=encoder, decoder=decoder, quantizer=quantizer, losses=losses)
