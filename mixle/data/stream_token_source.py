"""A streaming token source: yield ``(context-window, next-token)`` micro-batches from an IN-MEMORY token-id
array WITHOUT ever materializing the ``(N, block)`` window matrix -- each batch's windows are built on the fly
and discarded.

This is the data half of the non-buffering streaming estimator. The standard encoder builds and buffers every
``(window -> next)`` observation (``O(corpus x block)`` host RAM -- the materialization wall); this keeps only the
token array resident (``O(corpus)``) plus ``O(batch x block)`` ephemeral per batch. Two honest caveats: the token
array must fit in memory (this is not an out-of-core reader), and ``shuffle=True`` materializes one full-length
``O(corpus)`` permutation for the epoch order (still ``O(corpus)``, but a real extra array; ``shuffle=False``
avoids it).

The same generator SHAPE extends to a true out-of-core corpus -- reading windows from a memory-mapped / sharded
token file, where a checkpoint would be just the cursor position -- but that out-of-core / resumable-cursor
version is not implemented here.

Token ids are validated (finite, exact-integer, in a lossless ``int64`` range) and stay ``int64`` all the way
through to the yielded batches -- context windows are never downcast to ``float32`` internally, since that
silently loses integer identity above ``2**24`` (MXR-080-0062). A model boundary that genuinely wants a float
context (e.g. to ride a shared float input path) casts explicitly on its own side; that is a downstream,
deliberate choice, not something this data layer should do silently on a caller's behalf.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _validate_token_ids(token_ids: Any, name: str = "token_ids") -> np.ndarray:
    """Validate and coerce ``token_ids`` to a 1-D, lossless-integer (``int64``) array.

    Rejects non-1-D input, non-finite values, fractional values, and anything outside the ``int64`` range --
    a blind ``.astype(int)`` would otherwise silently truncate a fractional id (``2.8`` -> ``2``) or corrupt
    an out-of-range one instead of failing loudly (MXR-080-0062). Finiteness is checked before any
    equality/comparison against the array so a NaN never reaches a comparison it would silently lose
    (``NaN != NaN`` is always ``True``, which would otherwise misclassify a NaN as "fractional" instead of
    "non-finite").
    """
    ids = np.asarray(token_ids)
    if ids.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array of token ids, got shape {ids.shape}")
    if ids.size == 0:
        return ids.astype(np.int64)
    if np.issubdtype(ids.dtype, np.integer):
        if not np.can_cast(ids.dtype, np.int64, casting="safe"):
            lo, hi = int(ids.min()), int(ids.max())
            if lo < _INT64_MIN or hi > _INT64_MAX:
                raise ValueError(f"{name} contains values outside the int64 range: min={lo}, max={hi}")
        return ids.astype(np.int64)
    if not np.issubdtype(ids.dtype, np.floating):
        raise TypeError(f"{name} must be an integer or floating array of token ids, got dtype {ids.dtype}")
    if not np.all(np.isfinite(ids)):
        raise ValueError(f"{name} contains non-finite values (NaN/inf); token ids must be finite integers.")
    if not np.array_equal(ids, np.trunc(ids)):
        raise ValueError(f"{name} contains non-integer (fractional) values; token ids must be exact integers.")
    if bool(np.any(ids < _INT64_MIN)) or bool(np.any(ids > _INT64_MAX)):
        raise ValueError(f"{name} contains values outside the int64 range.")
    return ids.astype(np.int64)


def stream_token_source(
    token_ids: Any, block: int, batch_size: int, *, epochs: int = 1, shuffle: bool = True, seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(context_windows (b, block) int64, next_tokens (b,) int64)`` micro-batches; never buffers windows.

    The token array -- and, with ``shuffle=True``, one ``O(len(token_ids))`` permutation for the epoch order --
    is the only resident data; each micro-batch's windows are built on the fly and discarded.

    ``token_ids`` is validated (finite, exact-integer, lossless ``int64`` range) and stays ``int64`` through
    every yielded batch -- it is never downcast to ``float32`` here, since that silently loses integer
    identity for ids at or above ``2**24`` (MXR-080-0062); a consumer that wants float input casts
    explicitly on its own side.
    """
    ids = _validate_token_ids(token_ids)
    n = len(ids) - int(block)
    if n <= 0:
        return
    rng = np.random.RandomState(seed)
    for _ in range(int(epochs)):
        order = rng.permutation(n) if shuffle else np.arange(n)
        for k in range(0, n, int(batch_size)):
            idx = order[k : k + int(batch_size)]
            ctx = np.stack([ids[i : i + int(block)] for i in idx])  # int64, matches ids -- built, then discarded
            nxt = ids[idx + int(block)]
            yield (ctx, nxt)
