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
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


def stream_token_source(
    token_ids: Any, block: int, batch_size: int, *, epochs: int = 1, shuffle: bool = True, seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(context_windows (b, block) float32, next_tokens (b,) int)`` micro-batches; never buffers windows.

    The token array -- and, with ``shuffle=True``, one ``O(len(token_ids))`` permutation for the epoch order --
    is the only resident data; each micro-batch's windows are built on the fly and discarded.
    """
    ids = np.asarray(token_ids)
    n = len(ids) - int(block)
    if n <= 0:
        return
    rng = np.random.RandomState(seed)
    for _ in range(int(epochs)):
        order = rng.permutation(n) if shuffle else np.arange(n)
        for k in range(0, n, int(batch_size)):
            idx = order[k : k + int(batch_size)]
            ctx = np.stack([ids[i : i + int(block)] for i in idx]).astype("float32")  # per-batch, then discarded
            nxt = ids[idx + int(block)].astype(int)
            yield (ctx, nxt)
