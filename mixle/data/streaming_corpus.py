"""Streaming, sharded, packed, deterministic token-corpus pipeline (roadmap F3).

Scope note: this builds the *pipeline* -- sharded corpus streaming with per-rank residency, sequence
packing, and deterministic global ordering given ``(seed, epoch)`` -- over a pre-tokenized corpus (plain
integer token-id arrays, one array per document). A real BPE/tokenizer library is not a dependency of this
codebase and is out of scope here; tokenization is a separate, later concern. F1 (the distributed trainer
this pipeline would ultimately feed "at target tokens/s") does not exist yet either, so "saturates F1" is
not measurable from this module alone -- everything else in the F3 acceptance list (per-rank sharding,
packing efficiency, determinism, curriculum hooks) is real and tested here.

Composes with existing machinery rather than duplicating it:

- Per-rank residency extends :class:`~mixle.utils.parallel.multiprocessing.MPEncodedData`'s existing
  sharding contract: that handle splits an input sequence round-robin, rank ``i`` keeping
  ``data[i], data[i + world_size], data[i + 2*world_size], ...`` (disjoint, complete-coverage).
  :func:`shard_documents_for_rank` applies the identical round-robin split, but to the *shuffled* global
  document order from :func:`global_document_order` rather than to the raw input order -- so shuffling
  changes which rank sees which document while the sharding contract itself (disjoint, complete, index
  mod world_size) is unchanged and remains bitwise reproducible.
- The packed output rows are ``block + 1`` tokens: ``row[:-1]`` is the model input and ``row[1:]`` is the
  shifted next-token target at every position, matching the dense all-position teacher-forcing shape
  ``mixle.models.language_model._forward_all_positions`` / ``LM.fit_pairs`` already consume (as opposed to
  ``mixle.data.stream_token_source``'s one-target-per-window shape, which is the right shape for a single
  unbounded sliding stream but wastes a factor of ``block`` of compute once sequences are packed).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

__all__ = [
    "global_document_order",
    "shard_documents_for_rank",
    "pack_documents",
    "PackedCorpus",
    "StreamingCorpus",
]

SequenceSelector = Callable[[np.ndarray, int, int], np.ndarray]


def _epoch_seed(seed: int, epoch: int) -> int:
    """Fold ``(seed, epoch)`` into one 32-bit seed via ``SeedSequence`` -- independent, reproducible streams
    per epoch from a single user-facing ``seed`` (the standard "reshuffle differently each epoch, but
    deterministically" pattern)."""
    return int(np.random.SeedSequence([int(seed), int(epoch)]).generate_state(1)[0])


def global_document_order(
    num_documents: int,
    *,
    seed: int,
    epoch: int,
    sequence_selector: SequenceSelector | None = None,
) -> np.ndarray:
    """The single, authoritative order documents are consumed in across ALL ranks combined for this epoch.

    Same ``(seed, epoch)`` -> bitwise-identical order on every call/process (pure function of its inputs,
    ``RandomState`` seeded deterministically). Different ``epoch`` with the same ``seed`` -> a different,
    still fully deterministic order.

    ``sequence_selector`` is the curriculum hook for E7 ("a bandit over length buckets" rationing
    ultra-long examples, per the roadmap): an optional ``(order, seed, epoch) -> order`` callback that may
    reorder, filter, or otherwise transform the base permutation before it is handed out to ranks. Left
    ``None``, the base shuffle passes through unchanged. This module does not implement any curriculum
    policy itself -- only the extension point.
    """
    rng = np.random.RandomState(_epoch_seed(seed, epoch))
    order = rng.permutation(int(num_documents))
    if sequence_selector is not None:
        order = np.asarray(sequence_selector(order, int(seed), int(epoch)))
    return order


def shard_documents_for_rank(order: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    """Round-robin split of a (possibly shuffled/filtered) global document order across ranks.

    Rank ``i`` gets ``order[i], order[i + world_size], order[i + 2*world_size], ...`` -- the same
    round-robin residency contract :class:`~mixle.utils.parallel.multiprocessing.MPEncodedData` already
    uses for its worker shards, so the two compose: disjoint across ranks, complete coverage of ``order``.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not (0 <= rank < world_size):
        raise ValueError("rank must satisfy 0 <= rank < world_size, got rank=%r world_size=%r" % (rank, world_size))
    return np.asarray(order)[rank::world_size]


class PackedCorpus:
    """Result of :func:`pack_documents`: the packed rows plus the measured packing efficiency."""

    __slots__ = ("rows", "packing_efficiency", "real_tokens", "total_tokens")

    def __init__(self, rows: np.ndarray, real_tokens: int, total_tokens: int) -> None:
        self.rows = rows
        self.real_tokens = int(real_tokens)
        self.total_tokens = int(total_tokens)
        self.packing_efficiency = (self.real_tokens / self.total_tokens) if self.total_tokens else 1.0

    def __len__(self) -> int:
        return int(self.rows.shape[0])


def pack_documents(
    documents: Sequence[Any],
    indices: Sequence[int],
    block: int,
    *,
    pad_id: int = 0,
    boundary_id: int | None = None,
) -> PackedCorpus:
    """Concatenate ``documents[indices]`` (in the given order) into one token stream and chunk it into
    fixed-length ``block + 1`` rows -- the standard "concatenate-and-chunk with document boundaries"
    packing scheme used in LM pretraining.

    ``row[:-1]`` (length ``block``) is the model input; ``row[1:]`` (length ``block``) is the shifted
    next-token target at every position. Padding (``pad_id``) is only ever needed to fill out the final
    row once the stream runs out, so waste is bounded by ``block`` tokens total, not ``block`` tokens per
    document -- packing efficiency (the real-token fraction) climbs toward 1.0 as the corpus grows relative
    to ``block``. ``boundary_id`` (e.g. an EOS id), if given, is inserted between consecutive documents so
    the model can see document edges within a packed row; it counts as a real (non-pad) token.
    """
    if block <= 0:
        raise ValueError("block must be positive")
    unit = int(block) + 1
    pieces: list[np.ndarray] = []
    for idx in indices:
        doc = np.asarray(documents[idx]).reshape(-1)
        if doc.size:
            pieces.append(doc)
        if boundary_id is not None:
            pieces.append(np.asarray([boundary_id]))
    if not pieces:
        return PackedCorpus(np.zeros((0, unit), dtype=np.int64), real_tokens=0, total_tokens=0)

    flat = np.concatenate(pieces).astype(np.int64)
    n_full = flat.size // unit
    remainder = flat.size - n_full * unit

    full_rows = flat[: n_full * unit].reshape(n_full, unit) if n_full else np.zeros((0, unit), dtype=np.int64)
    pad_count = 0
    if remainder > 0:
        last_row = np.full(unit, pad_id, dtype=np.int64)
        last_row[:remainder] = flat[n_full * unit :]
        rows = np.vstack([full_rows, last_row[None, :]])
        pad_count = unit - remainder
    else:
        rows = full_rows

    total_tokens = int(rows.size)
    real_tokens = total_tokens - pad_count
    return PackedCorpus(rows, real_tokens=real_tokens, total_tokens=total_tokens)


class StreamingCorpus:
    """Per-rank streaming view over a sharded, tokenized corpus: shuffled deterministically by
    ``(seed, epoch)``, packed into fixed-length dense-teacher-forcing rows, and batched.

    ``documents`` is the full corpus (every rank sees the same list; only its OWN shard is ever packed or
    batched -- no gather, no materializing another rank's tokens). For a real out-of-core corpus,
    ``documents`` is a lazy/mmap-backed sequence of per-shard-file document lists; the contract here is
    unchanged since sharding and packing only ever index it, never buffer the whole thing.
    """

    def __init__(
        self,
        documents: Sequence[Any],
        *,
        rank: int,
        world_size: int,
        block: int,
        batch_size: int,
        seed: int = 0,
        pad_id: int = 0,
        boundary_id: int | None = None,
        sequence_selector: SequenceSelector | None = None,
    ) -> None:
        self.documents = documents
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.block = int(block)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.pad_id = int(pad_id)
        self.boundary_id = boundary_id
        self.sequence_selector = sequence_selector
        self.last_packing_efficiency: float | None = None

    def rank_document_indices(self, epoch: int) -> np.ndarray:
        """This rank's document indices for ``epoch``, in the exact order they will be packed."""
        order = global_document_order(
            len(self.documents), seed=self.seed, epoch=epoch, sequence_selector=self.sequence_selector
        )
        return shard_documents_for_rank(order, self.rank, self.world_size)

    def epoch_batches(self, epoch: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(context (b, block) float32, targets (b, block) int64)`` micro-batches for this rank.

        Deterministic given ``(seed, epoch, rank, world_size)``: re-running with the same inputs yields
        bitwise-identical batches. Sets :attr:`last_packing_efficiency` as a side effect (the real-token
        fraction of the rows this call packed).
        """
        indices = self.rank_document_indices(epoch)
        packed = pack_documents(self.documents, indices, self.block, pad_id=self.pad_id, boundary_id=self.boundary_id)
        self.last_packing_efficiency = packed.packing_efficiency
        rows = packed.rows
        for start in range(0, rows.shape[0], self.batch_size):
            chunk = rows[start : start + self.batch_size]
            yield chunk[:, :-1].astype(np.float32), chunk[:, 1:].astype(np.int64)
