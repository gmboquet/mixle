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
- Padding is structural, not training signal. The final packed row may be padded with ``pad_id`` to fill
  out ``block + 1`` tokens once the stream runs out -- bounded waste of at most one row, ever, never more.
  :func:`pack_documents` returns a per-position ``loss_mask`` alongside the rows, and
  :meth:`StreamingCorpus.epoch_batches` threads it through every micro-batch. Callers MUST apply that mask
  (or otherwise exclude its ``False`` positions) before averaging next-token loss -- otherwise the
  fabricated pad targets are silently trained on as if they were real observed labels.
- Document tokens are validated before packing -- finite, exact-integer values in a lossless ``int64``
  range -- and stay ``int64`` throughout: :func:`pack_documents`/:meth:`StreamingCorpus.epoch_batches`
  never downcast context to ``float32``, which would silently lose integer identity for ids at or above
  ``2**24``.
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

_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _require_exact_int(value: Any, name: str) -> int:
    """Coerce ``value`` to a plain ``int``, rejecting bools, non-numeric types, non-finite floats, and any
    value with a fractional part.

    A blind ``int(value)`` truncation (e.g. ``pad_id=0.5`` -> ``0``) silently changes behavior instead of
    failing loudly, which is exactly the MXR-080-0062/MXR-080-0063 failure mode this guards against.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got bool {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        truncated = int(value)
        if value != truncated:
            raise ValueError(f"{name} must be an exact integer, got {value!r}")
        return truncated
    raise TypeError(f"{name} must be an int, got {value!r} ({type(value).__name__})")


def _validate_document_tokens(doc: Any, *, doc_index: Any = None) -> np.ndarray:
    """Validate and coerce one document's tokens to a 1-D, lossless-integer (``int64``) array.

    Rejects a non-1-D document, non-finite values, fractional values, and anything outside the ``int64``
    range -- the same lossless-integer contract
    :func:`~mixle.data.stream_token_source._validate_token_ids` enforces for the sliding-window path,
    applied per document here since documents are validated and concatenated one at a time (MXR-080-0062).
    Finiteness is checked before any equality/comparison against the array so a NaN never reaches a
    comparison it would silently lose.
    """
    label = "documents" if doc_index is None else f"documents[{doc_index!r}]"
    arr = np.asarray(doc)
    if arr.ndim > 1:
        raise ValueError(f"{label} must be a 1-D array of token ids, got shape {arr.shape}")
    arr = arr.reshape(-1)
    if arr.size == 0:
        return arr.astype(np.int64)
    if np.issubdtype(arr.dtype, np.integer):
        if not np.can_cast(arr.dtype, np.int64, casting="safe"):
            lo, hi = int(arr.min()), int(arr.max())
            if lo < _INT64_MIN or hi > _INT64_MAX:
                raise ValueError(f"{label} contains values outside the int64 range: min={lo}, max={hi}")
        return arr.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"{label} must be an integer or floating array of token ids, got dtype {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains non-finite values (NaN/inf); token ids must be finite integers.")
    if not np.array_equal(arr, np.trunc(arr)):
        raise ValueError(f"{label} contains non-integer (fractional) values; token ids must be exact integers.")
    if bool(np.any(arr < _INT64_MIN)) or bool(np.any(arr > _INT64_MAX)):
        raise ValueError(f"{label} contains values outside the int64 range.")
    return arr.astype(np.int64)


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
    """Result of :func:`pack_documents`: the packed rows, the measured packing efficiency, and a per-target
    ``loss_mask`` marking which target positions are real observed next-token labels vs. fabricated
    padding (see :func:`pack_documents` for the exact boundary policy)."""

    __slots__ = ("rows", "packing_efficiency", "real_tokens", "total_tokens", "loss_mask")

    def __init__(self, rows: np.ndarray, *, real_tokens: int, total_tokens: int, loss_mask: np.ndarray) -> None:
        self.rows = rows
        self.real_tokens = int(real_tokens)
        self.total_tokens = int(total_tokens)
        self.packing_efficiency = (self.real_tokens / self.total_tokens) if self.total_tokens else 1.0
        self.loss_mask = loss_mask

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

    Returns a :class:`PackedCorpus` whose ``loss_mask`` (shape ``(n_rows, block)``, boolean) marks which
    TARGET positions (``row[1:]``) are real observed next-token labels vs. fabricated padding. Boundary
    policy, decided explicitly here: padding never appears anywhere but the tail of the FINAL row (see
    above), so ``loss_mask`` is ``True`` everywhere except possibly that one tail; a ``boundary_id`` token
    is real signal like any other token and is never masked out. This function does not implement
    SFT-style prompt masking (weight-0 on prompt tokens) -- that is an orthogonal concern this codebase
    already handles elsewhere via per-sample weights; ``loss_mask`` here answers exactly one question, "is
    this target a fabricated pad position." Callers MUST apply this mask (or otherwise exclude its
    ``False`` positions) before averaging next-token loss, or the fabricated pad targets are silently
    trained on as if they were real labels (MXR-080-0061).

    Every document's tokens are validated before packing -- finite, exact-integer values in a lossless
    ``int64`` range; fractional, non-finite, or out-of-range ids raise rather than being silently cast
    (MXR-080-0062). ``pad_id`` and ``boundary_id`` are validated the same way, since both become real
    token-stream values.
    """
    if block <= 0:
        raise ValueError("block must be positive")
    pad_id = _require_exact_int(pad_id, "pad_id")
    if boundary_id is not None:
        boundary_id = _require_exact_int(boundary_id, "boundary_id")

    unit = int(block) + 1
    pieces: list[np.ndarray] = []
    for idx in indices:
        doc = _validate_document_tokens(documents[idx], doc_index=idx)
        if doc.size:
            pieces.append(doc)
        if boundary_id is not None:
            pieces.append(np.asarray([boundary_id], dtype=np.int64))
    if not pieces:
        empty_mask = np.ones((0, unit - 1), dtype=bool)
        return PackedCorpus(np.zeros((0, unit), dtype=np.int64), real_tokens=0, total_tokens=0, loss_mask=empty_mask)

    flat = np.concatenate(pieces)  # every piece already validated int64 -- no blind whole-corpus cast needed
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

    # loss_mask over TARGET positions (row[1:] of each row): True where the target is a real observed
    # next-token label, False where it is fabricated padding. Padding is only ever added to the tail of the
    # FINAL row (never mid-stream, never any other row -- see docstring), so every row except possibly the
    # last is entirely real. In the last row, target[i] = row[i+1] is real iff i+1 < remainder, i.e. the
    # real target run has length `remainder - 1` (a remainder of exactly 1 real token in the row means ZERO
    # real targets: the sole real token has no successor to predict, so its own target position is pad).
    n_rows = rows.shape[0]
    loss_mask = np.ones((n_rows, unit - 1), dtype=bool)
    if remainder > 0:
        valid_targets = max(remainder - 1, 0)
        loss_mask[-1, valid_targets:] = False

    return PackedCorpus(rows, real_tokens=real_tokens, total_tokens=total_tokens, loss_mask=loss_mask)


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

    def epoch_batches(self, epoch: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield ``(context (b, block) int64, targets (b, block) int64, loss_mask (b, block) bool)``
        micro-batches for this rank.

        ``loss_mask[i, j]`` is True iff ``targets[i, j]`` is a real observed next-token label, and False
        only for fabricated padding introduced to fill out the final row of the packed stream once real
        tokens run out (see :func:`pack_documents` for the exact boundary policy) -- at most one row per
        call carries any ``False`` at all, regardless of corpus size. Callers MUST multiply the
        per-position loss by this mask (or otherwise exclude ``False`` positions) rather than average the
        raw per-position loss unconditionally: doing so silently teaches the model that ``pad_id`` is a
        real next-token target (MXR-080-0061). ``context``/``targets`` stay ``int64`` -- never downcast to
        ``float32`` here, which would silently lose integer identity for ids at or above ``2**24``
        (MXR-080-0062); a consumer that wants float input casts explicitly on its own side.

        Deterministic given ``(seed, epoch, rank, world_size)``: re-running with the same inputs yields
        bitwise-identical batches. Sets :attr:`last_packing_efficiency` as a side effect (the real-token
        fraction of the rows this call packed).
        """
        indices = self.rank_document_indices(epoch)
        packed = pack_documents(self.documents, indices, self.block, pad_id=self.pad_id, boundary_id=self.boundary_id)
        self.last_packing_efficiency = packed.packing_efficiency
        rows = packed.rows
        mask = packed.loss_mask
        for start in range(0, rows.shape[0], self.batch_size):
            chunk = rows[start : start + self.batch_size]
            mask_chunk = mask[start : start + self.batch_size]
            # .copy(): context/targets are overlapping column-shifted views of the SAME `rows` array (and
            # of each other) -- copy so a caller mutating one batch in place can never corrupt `rows`,
            # another already-yielded batch, or (via the context/targets overlap) each other.
            yield chunk[:, :-1].copy(), chunk[:, 1:].copy(), mask_chunk.copy()
