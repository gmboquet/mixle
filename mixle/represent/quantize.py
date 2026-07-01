"""``VectorQuantizer`` -- learn a discrete vocabulary IN the shared embedding space, don't guess it upstream.

Discrete tokens, when you want them (compression, transfer, a fixed vocabulary), come *after* embedding, not
before segmentation: fit a codebook to the continuous vectors and each vector's nearest code is its token id. The
codebook is a *learned* model (k-means / a mixture), so the vocabulary is inferred from data rather than assumed
-- and because every modality is embedded into the same space, one codebook is a **cross-modal vocabulary**
(an image patch and a word can share a token id when they land near the same centroid).

``fit``/``quantize``/``dequantize`` are the codec; ``straight_through`` gives the VQ-VAE gradient so the codebook
and the encoders can be trained end to end under a generative or downstream objective. This is the *only* place
discreteness lives -- the segmenter and embedding stay vocabulary-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class VectorQuantizer:
    """A learned codebook over ``R^dim``: nearest-centroid quantization of embedding vectors into discrete ids."""

    def __init__(self, num_codes: int, dim: int, *, seed: int = 0) -> None:
        self.num_codes = int(num_codes)
        self.dim = int(dim)
        self.seed = int(seed)
        self.codebook: np.ndarray | None = None  # (num_codes, dim)

    def fit(self, vectors: np.ndarray, *, iters: int = 25) -> VectorQuantizer:
        """Fit the codebook by k-means (Lloyd) on ``vectors`` ``(n, dim)`` -- the vocabulary is learned, not assumed."""
        x = np.asarray(vectors, dtype=np.float64)
        rng = np.random.RandomState(self.seed)
        k = min(self.num_codes, len(x))
        centers = x[rng.choice(len(x), size=k, replace=False)].copy()
        for _ in range(int(iters)):
            ids = self._assign(x, centers)
            new = np.stack([x[ids == j].mean(axis=0) if np.any(ids == j) else centers[j] for j in range(len(centers))])
            if np.allclose(new, centers):
                centers = new
                break
            centers = new
        self.codebook = centers
        return self

    @staticmethod
    def _assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2 ; the data terms drop out of the argmin
        d = -2.0 * x @ centers.T + np.sum(centers**2, axis=1)[None, :]
        return d.argmin(axis=1)

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """Nearest-code id for each vector -- the discrete token stream ``(n,)``."""
        if self.codebook is None:
            raise RuntimeError("call fit(...) before quantize(...)")
        return self._assign(np.asarray(vectors, dtype=np.float64), self.codebook)

    def dequantize(self, ids: np.ndarray) -> np.ndarray:
        """Codebook vectors for token ids ``(n,)`` -> ``(n, dim)`` (the reconstruction / de-tokenization)."""
        if self.codebook is None:
            raise RuntimeError("call fit(...) before dequantize(...)")
        return self.codebook[np.asarray(ids, dtype=np.int64)]

    def reconstruction_error(self, vectors: np.ndarray) -> float:
        """Mean squared quantization error -- the codebook's fidelity (a codebook-size / bitrate knob)."""
        v = np.asarray(vectors, dtype=np.float64)
        return float(np.mean(np.sum((v - self.dequantize(self.quantize(v))) ** 2, axis=1)))

    def straight_through(self, vectors: Any) -> Any:
        """VQ-VAE straight-through estimator: return quantized vectors but pass gradients to ``vectors`` unchanged.

        Lets the encoders and (with a codebook-commitment loss) the codebook train end to end through the discrete
        bottleneck. ``vectors`` is a torch tensor ``(n, dim)``.
        """
        import torch

        if self.codebook is None:
            raise RuntimeError("call fit(...) before straight_through(...)")
        cb = torch.as_tensor(self.codebook, dtype=vectors.dtype, device=vectors.device)
        d = torch.cdist(vectors, cb)
        ids = d.argmin(dim=1)
        q = cb[ids]
        return vectors + (q - vectors).detach()  # identity in the backward pass
