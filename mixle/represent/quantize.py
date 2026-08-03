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


def _positive_int(value: Any, name: str) -> int:
    """Exact positive integer or a ``ValueError`` (a Boolean is not a count)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _seed_int(value: Any, name: str) -> int:
    """Exact seed in ``RandomState``'s accepted range, or a ``ValueError`` (MXR-080-1906).

    ``int(seed)`` truncated: ``seed=2.9`` and ``seed=2`` selected the SAME random stream while
    reading as two different declarations (verified -- both produce a bit-identical codebook), and
    ``seed=True`` silently became ``1``. A seed identifies a draw; rounding one is running a
    different experiment than the one that was written down. The range is checked here rather than
    left to ``np.random.RandomState``, which only rejects a negative seed at ``fit`` time -- long
    after the constructor accepted it.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an exact integer, got {value!r}")
    result = int(value)
    if not 0 <= result < 2**32:
        raise ValueError(f"{name} must lie in [0, 2**32), got {result!r}")
    return result


class VectorQuantizer:
    """A learned codebook over ``R^dim``: nearest-centroid quantization of embedding vectors into discrete ids."""

    def __init__(self, num_codes: int, dim: int, *, seed: int = 0) -> None:
        self.num_codes = _positive_int(num_codes, "num_codes")
        self.dim = _positive_int(dim, "dim")
        self.seed = _seed_int(seed, "seed")
        self._codebook: np.ndarray | None = None  # (num_codes, dim), owned and frozen once fitted

    @property
    def codebook(self) -> np.ndarray | None:
        """The fitted ``(num_codes, dim)`` centroids -- read-only, or ``None`` before :meth:`fit`.

        The codebook IS the learned vocabulary: every token id this codec emits or decodes is defined
        by it, so it is fitted state, not a scratch buffer. It used to be a plain public attribute,
        and a single post-fit write silently redefined the vocabulary (MXR-080-1906). Verified: after
        ``vq.fit(x)``, setting ``vq.codebook[0, 0] = nan`` collapsed ``quantize`` from ``[0, 2, 0, 0,
        2]`` to ``[0, 0, 0, 0, 0]`` -- every vector assigned to code 0, because a NaN distance never
        wins an ``argmin`` -- and turned ``reconstruction_error`` into ``nan`` rather than an error.
        Rebinding was just as open: ``vq.codebook = np.zeros((99, 7))`` was accepted beside an
        unchanged ``dim=2`` and ``num_codes=3``.

        The array is frozen with ``writeable = False`` and reachable only through this property, which
        matches :class:`mixle.engines.formats.CodebookFormat`, whose codebook is copied and sealed the
        same way. ``dequantize`` returns ``codebook[ids]``, and NumPy fancy indexing copies, so
        decoded vectors are still ordinary writable arrays.
        """
        return self._codebook

    def _as_vectors(self, vectors: Any, what: str) -> np.ndarray:
        """``(n, dim)`` finite float view of ``vectors``, or a ``ValueError`` naming the geometry violated.

        ``dim`` is the codec's declared geometry, not a hint: fitting a ``(2, 3)`` input under
        ``VectorQuantizer(2, 2)`` used to publish a ``(2, 3)`` codebook while ``dim`` still read ``2``,
        and a NaN sample produced a NaN codebook (and a NaN reconstruction error) that quantized every
        later vector to an arbitrary code.
        """
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError(f"{what} requires a non-empty (n, {self.dim}) array of vectors, got shape {x.shape}")
        if x.shape[1] != self.dim:
            raise ValueError(f"{what} requires vectors of declared width dim={self.dim}, got width {x.shape[1]}")
        if not np.isfinite(x).all():
            raise ValueError(f"{what} requires finite vectors; a non-finite sample poisons the whole codebook")
        return x

    def fit(self, vectors: np.ndarray, *, iters: int = 25) -> VectorQuantizer:
        """Fit the codebook by k-means (Lloyd) on ``vectors`` ``(n, dim)`` -- the vocabulary is learned, not assumed.

        ``num_codes`` is a cap, not a guarantee: fitting on fewer than ``num_codes`` samples yields one
        center per sample, and ``num_codes`` is updated in place to that actual count so it -- and every
        bounds check derived from it (``dequantize``, a caller building a one-hot over the vocabulary) --
        stays honest about the codebook's real capacity instead of quietly overstating it.

        ``iters`` must be a positive integer: ``iters=0`` used to publish the random initialization
        centers as a fitted codec, with no Lloyd step ever run.
        """
        x = self._as_vectors(vectors, "fit")
        iters = _positive_int(iters, "iters")
        rng = np.random.RandomState(self.seed)
        k = min(self.num_codes, len(x))
        centers = x[rng.choice(len(x), size=k, replace=False)].copy()
        for _ in range(iters):
            ids = self._assign(x, centers)
            new = np.stack([x[ids == j].mean(axis=0) if np.any(ids == j) else centers[j] for j in range(len(centers))])
            if np.allclose(new, centers):
                centers = new
                break
            centers = new
        centers.setflags(write=False)  # the fitted vocabulary is evidence, not a scratch buffer
        self._codebook = centers
        self.num_codes = len(centers)  # honest post-fit count; may be < the originally requested cap
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
        return self._assign(self._as_vectors(vectors, "quantize"), self.codebook)

    def dequantize(self, ids: np.ndarray) -> np.ndarray:
        """Codebook vectors for token ids ``(n,)`` -> ``(n, dim)`` (the reconstruction / de-tokenization).

        Ids must already be integral. The cast to ``int64`` used to happen *before* the range check, so a
        fractional id such as ``0.9`` truncated to the ordinary in-range token ``0`` and decoded as if it
        had been produced by :meth:`quantize`.
        """
        if self.codebook is None:
            raise RuntimeError("call fit(...) before dequantize(...)")
        raw = np.asarray(ids)
        if raw.dtype.kind == "b":
            raise ValueError("code ids must be integers, not Booleans")
        if raw.dtype.kind == "f":
            if not np.isfinite(raw).all() or np.any(raw != np.rint(raw)):
                raise ValueError("code ids must be exact integers; a fractional id is not a token")
        elif raw.dtype.kind not in "iu":
            raise ValueError(f"code ids must be integers, got dtype {raw.dtype}")
        ids_arr = raw.astype(np.int64)
        bad = (ids_arr < 0) | (ids_arr >= len(self.codebook))
        if np.any(bad):
            bad_id = int(np.asarray(ids_arr[bad]).flat[0])
            raise IndexError(f"code id {bad_id} out of range for a {len(self.codebook)}-code codebook.")
        return self.codebook[ids_arr]

    def reconstruction_error(self, vectors: np.ndarray) -> float:
        """Mean squared quantization error -- the codebook's fidelity (a codebook-size / bitrate knob)."""
        v = self._as_vectors(vectors, "reconstruction_error")
        return float(np.mean(np.sum((v - self.dequantize(self.quantize(v))) ** 2, axis=1)))

    def straight_through(self, vectors: Any) -> Any:
        """VQ-VAE straight-through estimator: return quantized vectors but pass gradients to ``vectors`` unchanged.

        Lets the encoders and (with a codebook-commitment loss) the codebook train end to end through the discrete
        bottleneck. ``vectors`` is a torch tensor ``(n, dim)``.
        """
        import torch

        if self._codebook is None:
            raise RuntimeError("call fit(...) before straight_through(...)")
        # `torch.tensor` (not `as_tensor`): the fitted codebook is frozen non-writable, and
        # `as_tensor` would try to SHARE that buffer and warn that PyTorch cannot honour
        # non-writable memory. Copying is the correct answer rather than the warning's other
        # suggestion of unfreezing -- the codebook must stay sealed (MXR-080-1906). The copy is
        # (num_codes, dim), negligible beside the cdist on the next line.
        cb = torch.tensor(self._codebook, dtype=vectors.dtype, device=vectors.device)
        d = torch.cdist(vectors, cb)
        ids = d.argmin(dim=1)
        q = cb[ids]
        return vectors + (q - vectors).detach()  # identity in the backward pass
