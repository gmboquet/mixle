"""What a retrieval result is relative to, and the exact-count check its ``k`` goes through.

A retriever returns ``(index, score)`` pairs. An index is meaningless on its own: it only names a
record given the corpus it indexes, and the score is only interpretable given the model that
produced it. Neither the corpus nor the model used to be recorded anywhere on the represent
retrievers, so a result could outlive, or be compared across, the thing it was computed against
(MXR-080-1906). :class:`RetrievalIdentity` is that missing record.

Shared by :class:`mixle.represent.api.Embedder` and
:class:`mixle.represent.posterior.PosteriorRetriever` so the two retrieval surfaces describe
themselves the same way.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["RetrievalIdentity", "exact_count", "records_digest", "vectors_digest"]


@dataclass(frozen=True)
class RetrievalIdentity:
    """The model and corpus a retriever's indices and scores refer to.

    Frozen, and built once at construction from state the retriever owns, so it is stable for the
    instance's lifetime: every result the retriever returns is attributable to this identity without
    the result having to carry a copy of it.

    ``corpus_digest`` is a change detector, not an authenticator -- it says whether two retrievers
    were built over the same corpus content, and nothing about provenance. It is ``None`` when the
    corpus holds records that have no canonical encoding (see :func:`records_digest`); absent is
    reported honestly rather than substituted with a digest that would not be reproducible.
    """

    model: str
    corpus_size: int
    corpus_digest: str | None = None

    def matches(self, other: RetrievalIdentity) -> bool:
        """Whether two retrievers ranked the same corpus with the same kind of model.

        Returns ``False`` when either digest is absent: an unknown corpus is not evidence of a
        matching one.
        """
        if not isinstance(other, RetrievalIdentity):
            return False
        if self.corpus_digest is None or other.corpus_digest is None:
            return False
        return (self.model, self.corpus_size, self.corpus_digest) == (
            other.model,
            other.corpus_size,
            other.corpus_digest,
        )


def exact_count(value: Any, name: str) -> int:
    """Return ``value`` as a non-negative ``int`` count, refusing values that only look like one.

    Retrieval ``k`` used to be checked as ``kf = float(k)`` then ``kf < 0 or kf != round(kf)``, which
    admits a Boolean: ``float(True) == 1.0``, so ``k=True`` passed the check and quietly returned
    exactly one neighbour (MXR-080-1906). A Boolean is not a count, exactly as ``VectorQuantizer``
    already refuses ``num_codes=True``.

    An integral float such as ``5.0`` is still ACCEPTED, deliberately. The previous contract allowed
    it, computed counts arrive as floats from ordinary arithmetic, and nothing is lost or truncated
    in the conversion -- refusing it would reject values callers legitimately produce, which is not
    the defect being fixed here. A fractional ``k`` remains an error rather than being truncated.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be a non-negative integer, got {value!r}. A Boolean is not a count: "
            f"float(True) == 1.0, so it would silently mean 'retrieve exactly one'."
        )
    if isinstance(value, (int, np.integer)):
        count = int(value)
    else:
        try:
            as_float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}") from exc
        if not np.isfinite(as_float) or as_float != round(as_float):
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        count = int(round(as_float))
    if count < 0:
        # Never silently accepted: Python's `[:k]` reads a negative k as "all but the last |k|",
        # not empty and not an error, which is never what a negative retrieval count intended.
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return count


def records_digest(records: Any) -> str | None:
    """SHA-256 over a sequence of heterogeneous records, or ``None`` if they cannot be encoded.

    Uses :func:`mixle.data.hashing._canonical`, whose encoding is self-delimiting and closed over the
    types it supports, so two different corpora cannot land on the same bytes. That closed schema
    RAISES on a record type it does not cover -- an arbitrary custom object, say -- and a retrieval
    corpus is explicitly allowed to hold arbitrary heterogeneous payloads. Rather than reject such a
    corpus (the retriever works fine on it) or fingerprint it with ``repr``, whose output embeds
    memory addresses and would differ run to run for equal content, this returns ``None`` and the
    identity records that its corpus digest is unavailable.
    """
    from mixle.data.hashing import _canonical

    digest = hashlib.sha256()
    try:
        for record in records:
            digest.update(_canonical(record))
    except (TypeError, ValueError):
        return None
    return digest.hexdigest()


def vectors_digest(vectors: np.ndarray) -> str:
    """SHA-256 over a numeric ``(n, dim)`` block, tagged with its shape and dtype.

    Used where the corpus is already a fitted array rather than raw records. The shape and dtype are
    hashed alongside the bytes so a reshaped or re-typed view of the same buffer is a different
    corpus, which is what it is for retrieval.
    """
    array = np.ascontiguousarray(vectors)
    digest = hashlib.sha256()
    digest.update(f"{array.shape}|{array.dtype.str}|".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()
