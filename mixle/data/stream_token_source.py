"""A streaming token source: yield ``(context-window, next-token)`` micro-batches from a token-id array via a
resumable cursor, WITHOUT ever materializing the ``(N, block)`` window matrix or a Python list of observations.

This is the data half of the non-buffering streaming estimator. The standard encoder builds and buffers every
``(window -> next)`` observation (``O(corpus x block)`` host RAM -- the materialization wall); this yields them a
micro-batch at a time from the read-only token array (``O(corpus)`` resident + ``O(batch x block)`` ephemeral).
For a real out-of-core corpus the same generator shape reads from a memory-mapped / sharded token file; the
cursor is resumable, so a checkpoint is just its position.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


def stream_token_source(
    token_ids: Any, block: int, batch_size: int, *, epochs: int = 1, shuffle: bool = True, seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(context_windows (b, block) float32, next_tokens (b,) int)`` micro-batches; never buffers windows.

    The token array is the only resident data; each micro-batch's windows are built on the fly and discarded.
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
