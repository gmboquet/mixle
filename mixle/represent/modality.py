"""Modality leaves -- turn an image or a signal into a fixed-dim vector the graph can relate (C2).

The cross-modal graph (workstream C1) made a fixed-length vector a first-class node: a
conditional-linear-Gaussian factor can drive it or be driven by it. C2 gives the OTHER modalities a
way in. :func:`vectorize` maps a raw item of any modality to a fixed ``dim`` vector:

  * ``text`` / ``record`` -> the learned embedding (:func:`mixle.represent.fit_embedder`);
  * ``image`` (a 2-D or 3-D numeric array) -> grid-pooled intensities (a coarse, deterministic,
    torch-free descriptor that captures brightness / spatial layout);
  * ``signal`` (a 1-D numeric array) -> per-window statistics (mean, energy, range) across the trace.

The image/signal descriptors are the honest v1 "featurizer" tier -- deterministic and dependency-free,
the same role ``HashedNGram`` plays for text. A learned encoder (the heterogeneous
segmenter+embedding stack) is a drop-in upgrade behind the same ``vectorize`` surface. Once vectorized,
an image or signal field is just a vector node: it participates in structure discovery, is scored,
sampled, and carries UQ like any other.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def image_features(img: Any, dim: int = 16, *, grid: int | None = None) -> np.ndarray:
    """A fixed ``dim`` descriptor of an image: mean intensity over a ``g x g`` grid of cells.

    ``img`` is ``(H, W)`` or ``(H, W, C)``; channels are averaged. The grid side ``g`` is chosen so
    ``g*g`` covers ``dim`` (then truncated/padded to exactly ``dim``), giving a coarse spatial-layout
    vector -- enough for an image field to correlate with structured fields in a discovered graph.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1) if a.ndim > 2 else np.atleast_2d(a)
    g = grid or max(1, int(np.ceil(np.sqrt(dim))))
    h, w = a.shape
    rows = np.array_split(np.arange(h), min(g, h))
    cols = np.array_split(np.arange(w), min(g, w))
    cells = [a[np.ix_(r, c)].mean() for r in rows for c in cols if len(r) and len(c)]
    v = np.asarray(cells, dtype=np.float64)
    return _fit_dim(v, dim)


def signal_features(sig: Any, dim: int = 16, *, windows: int | None = None) -> np.ndarray:
    """A fixed ``dim`` descriptor of a 1-D signal: (mean, energy, range) over evenly-spaced windows."""
    a = np.asarray(sig, dtype=np.float64).ravel()
    nwin = windows or max(1, dim // 3)
    feats: list[float] = []
    for w in np.array_split(a, min(nwin, len(a))) if len(a) else [a]:
        if len(w):
            feats += [float(w.mean()), float(np.mean(w * w)), float(w.max() - w.min())]
    return _fit_dim(np.asarray(feats, dtype=np.float64), dim)


def _fit_dim(v: np.ndarray, dim: int) -> np.ndarray:
    """Truncate or zero-pad ``v`` to exactly ``dim`` components."""
    if v.size >= dim:
        return v[:dim]
    return np.concatenate([v, np.zeros(dim - v.size)])


def vectorize(item: Any, kind: str, *, dim: int = 16, embedder: Any = None) -> np.ndarray:
    """Map a raw ``item`` of modality ``kind`` to a fixed ``dim`` vector (see module docstring).

    Args:
        item: the raw item (a string, a record, an image array, a signal array).
        kind: ``'text'`` | ``'record'`` | ``'image'`` | ``'signal'``.
        dim: output vector dimension.
        embedder: for ``text``/``record``, a fitted :class:`~mixle.represent.Embedder` to reuse
            (else a small one is fit on the single item -- pass one for consistency across a corpus).
    """
    if kind == "image":
        return image_features(item, dim)
    if kind == "signal":
        return signal_features(item, dim)
    if kind in ("text", "record"):
        if embedder is not None:
            return np.asarray(embedder.transform(item), dtype=np.float64)
        from mixle.represent import fit_embedder

        emb = fit_embedder([item, item, item, item], dim=dim, kind=kind, epochs=20)
        return np.asarray(emb.transform(item), dtype=np.float64)
    raise ValueError(f"unknown modality {kind!r}; expected text/record/image/signal")


def vectorize_all(items: Any, kind: str, *, dim: int = 16) -> np.ndarray:
    """Vectorize a sequence of same-modality items to an ``(n, dim)`` array (one shared embedder for text)."""
    items = list(items)
    if kind in ("text", "record"):
        from mixle.represent import fit_embedder

        emb = fit_embedder(items if len(items) >= 4 else items * 4, dim=dim, kind=kind, epochs=40)
        return np.asarray(emb.transform(items), dtype=np.float64)
    return np.stack([vectorize(it, kind, dim=dim) for it in items])
