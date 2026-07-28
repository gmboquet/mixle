"""Deterministic modality vectorization helpers.

:func:`vectorize` maps a raw item to a fixed-length vector that can be used by
structure-learning and heterogeneous Bayesian-network workflows:

  * ``text`` / ``record`` -> the learned embedding (:func:`mixle.represent.fit_embedder`);
  * ``image`` (a 2-D or 3-D numeric array) -> grid-pooled intensities (a coarse, deterministic,
    torch-free descriptor that captures brightness / spatial layout);
  * ``signal`` (a 1-D numeric array) -> per-window statistics (mean, energy, range) across the trace.

The image and signal descriptors are deterministic and dependency-free. They
are intended as a baseline vectorization layer; learned encoders can be placed
behind the same ``vectorize`` surface when a workflow needs richer features.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _positive_int(value: Any, name: str) -> int:
    """Exact positive integer or a ``ValueError`` (a Boolean is not a count)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _require_measured(a: np.ndarray, what: str) -> np.ndarray:
    """Reject absent / non-finite evidence instead of encoding it as an ordinary descriptor.

    A missing modality has no fixed-length descriptor: zero-padding an empty trace produces exactly the
    vector a genuinely measured all-zero trace produces, and NaN samples propagate into features that
    downstream graph/cross-modal fusion consumes as if they were observations. There is no missingness
    channel in this representation, so absence has to be refused at the boundary and handled by the
    caller (drop the record, or model the modality as explicitly missing).
    """
    if a.size == 0:
        raise ValueError(f"{what} requires a non-empty measurement; an absent modality has no descriptor")
    if not np.isfinite(a).all():
        raise ValueError(f"{what} requires finite samples; non-finite input is missing data, not evidence")
    return a


def _cell_grid(dim: int) -> tuple[int, int]:
    """The most nearly square ``(rows, cols)`` with ``rows * cols == dim`` exactly (``rows <= cols``).

    Exact factorization is what makes the descriptor a *partition* of the image. A ``ceil(sqrt(dim))``
    square grid overshoots whenever ``dim`` is not a perfect square, and the surplus cells then have to
    go somewhere; truncating them deletes a fixed region of the image (see :func:`image_features`).
    For a prime ``dim`` this degenerates to a single strip of ``dim`` cells -- coarser along one axis,
    but still covering every pixel, which is the property that matters.
    """
    rows = 1
    for r in range(1, int(np.sqrt(dim)) + 1):
        if dim % r == 0:
            rows = r
    return rows, dim // rows


def _axis_parts(n: int, k: int) -> list[np.ndarray]:
    """``k`` non-empty index groups covering ``range(n)``; groups repeat when ``n < k``.

    Repeating a group is a coarser (duplicated) reading of a small axis. The alternative -- emitting
    fewer cells and zero-padding out to ``dim`` -- would fabricate all-zero image regions that are
    indistinguishable from measured black ones, exactly what :func:`_require_measured` refuses at the
    other boundary.
    """
    groups = np.array_split(np.arange(n), min(k, n))
    return [groups[i * len(groups) // k] for i in range(k)]


def image_features(img: Any, dim: int = 16, *, grid: int | None = None) -> np.ndarray:
    """A fixed ``dim`` descriptor of an image: mean intensity over a ``rows x cols`` partition of cells.

    ``img`` is ``(H, W)`` or ``(H, W, C)``; channels are averaged. The cell grid is the most nearly
    square factorization ``rows * cols == dim`` (:func:`_cell_grid`), so the cells *tile* the image:
    every pixel lies in exactly one cell and therefore influences exactly one output coordinate. That
    is what makes this a spatial-layout vector -- enough for an image field to correlate with
    structured fields in a discovered graph.

    This used to choose a square grid with ``ceil(sqrt(dim))`` cells per side and truncate the
    row-major cell list to ``dim``. For any ``dim`` that is not a perfect square that is not pooling
    to a coarser resolution -- it permanently deletes the bottom/right cells. At the default grid for
    ``dim=2``, changing the entire bottom half of a 4x4 image from 0 to 100 left the descriptor exactly
    ``[0, 0]``; at ``dim=3`` the whole bottom-right quadrant was invisible. A descriptor presented as
    capturing spatial layout must not have location-dependent blind spots.

    ``grid`` overrides the automatic choice with an explicit square side and must satisfy
    ``grid * grid == dim``, so an explicit grid partitions the image too.

    The image must be non-empty and finite (see :func:`_require_measured`).
    """
    dim = _positive_int(dim, "dim")
    if grid is None:
        gr, gc = _cell_grid(dim)
    else:
        gr = gc = _positive_int(grid, "grid")
        if gr * gc != dim:
            raise ValueError(
                f"image_features(grid={grid!r}, dim={dim!r}) needs grid*grid == dim so the {gr}x{gc} cells "
                f"partition the image; {gr * gc} cells cannot be reported as {dim} without dropping or "
                "fabricating a region. Pass dim=grid*grid, or omit grid to use the automatic partition."
            )
    a = _require_measured(np.asarray(img, dtype=np.float64), "image_features")
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1) if a.ndim > 2 else np.atleast_2d(a)
    h, w = a.shape
    rows = _axis_parts(h, gr)
    cols = _axis_parts(w, gc)
    return np.asarray([a[np.ix_(r, c)].mean() for r in rows for c in cols], dtype=np.float64)


def signal_features(sig: Any, dim: int = 16, *, windows: int | None = None) -> np.ndarray:
    """A fixed ``dim`` descriptor of a 1-D signal: (mean, energy, range) over evenly-spaced windows.

    The trace must be non-empty and finite (see :func:`_require_measured`): an empty signal used to
    return the all-zero vector, which is indistinguishable from a measured zero-energy trace, and NaN
    samples used to travel through as NaN features padded out to ``dim``.
    """
    dim = _positive_int(dim, "dim")
    if windows is not None:
        windows = _positive_int(windows, "windows")
    a = _require_measured(np.asarray(sig, dtype=np.float64).ravel(), "signal_features")
    nwin = windows or max(1, dim // 3)
    feats: list[float] = []
    for w in np.array_split(a, min(nwin, len(a))):
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
        embedder: **required** for ``text``/``record``: a fitted :class:`~mixle.represent.Embedder`
            defining the shared coordinate system the vector lives in.

    Raises:
        ValueError: for ``text``/``record`` without an ``embedder``. There is no safe default. The
            fallback used to fit a fresh autoencoder on four copies of the single item being
            transformed, so every call returned coordinates from a *different* learned basis while
            presenting them as a common fixed-length vector. Independently vectorized ``"alpha alpha"``
            and ``"beta beta"`` came out at cosine ``+0.81``; fitting the same two items in one shared
            corpus space gave ``-0.77``. Neither sign nor magnitude survived, so any downstream
            correlation/graph/distance built from such vectors measured the fitting noise of
            single-item autoencoders. Fit one space with :func:`~mixle.represent.fit_embedder` and pass
            it here, or vectorize the whole corpus at once with :func:`vectorize_all`, which fits and
            applies a single shared embedder.
    """
    if kind == "image":
        return image_features(item, dim)
    if kind == "signal":
        return signal_features(item, dim)
    if kind in ("text", "record"):
        if embedder is None:
            raise ValueError(
                f"vectorize(kind={kind!r}) requires a fitted embedder: it defines the coordinate system the "
                "returned vector lives in, and vectors from separately fitted spaces are not comparable. "
                "Fit one space -- fit_embedder(corpus, dim=..., kind=...) -- and pass embedder=..., or call "
                "vectorize_all(items, kind) to vectorize a whole corpus through a single shared embedder."
            )
        return np.asarray(embedder.transform(item), dtype=np.float64)
    raise ValueError(f"unknown modality {kind!r}; expected text/record/image/signal")


def vectorize_all(items: Any, kind: str, *, dim: int = 16) -> np.ndarray:
    """Vectorize a sequence of same-modality items to an ``(n, dim)`` array (one shared embedder for text)."""
    items = list(items)
    if kind in ("text", "record"):
        from mixle.represent import fit_embedder

        emb = fit_embedder(items if len(items) >= 4 else items * 4, dim=dim, kind=kind, epochs=40)
        return np.asarray(emb.transform(items), dtype=np.float64)
    return np.stack([vectorize(it, kind, dim=dim) for it in items])
