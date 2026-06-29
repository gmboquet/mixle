"""The quantized seek / unrank index over an exact probability-ordered enumeration.

``QuantizedEnumerationIndex`` precomputes how many support values fall in each quantized
log-probability bin, so an arbitrary rank can be resolved and unranked without walking the prefix;
``LazyQuantizedEnumerationIndex`` builds it on demand and ``QuantizedCrossIndex`` indexes the product
of two such enumerations. This is the *finite/materialized* seek index; its exponential-support
counterpart -- the structural count-budget index -- lives in :mod:`mixle.enumeration.quantization.core`.
"""

import bisect
import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np


class QuantizedEnumerationIndex:
    """Bounded, indexable view of an exact probability-ordered enumeration.

    Items are grouped into log-probability bins measured in bits:

        bits(x) = -log2(p(x))
        bin(x)  = floor(bits(x) / bin_width_bits)

    Only items whose bit cost is at most max_bits are indexed. The returned
    (value, log_prob) pairs always carry the original exact log probability; the
    quantized bins are used only for counting and rank lookup.

    The generic implementation can build from an exact enumerator stream, so it works
    for every currently enumerable discrete stats model. Simple distributions may also
    build directly from scored support items, and distribution-specific dynamic programs
    can produce the same bin layout without relying on an exact global stream.
    """

    def __init__(
        self,
        bins: Sequence[tuple[int, list[tuple[Any, float]]]],
        bin_width_bits: float,
        max_bits: float,
        truncated: bool,
    ) -> None:
        self._bins = [(int(b), list(items)) for b, items in bins]
        self.bin_width_bits = float(bin_width_bits)
        self.max_bits = float(max_bits)
        self.truncated = bool(truncated)
        self.counts: dict[int, int] = {b: len(items) for b, items in self._bins}
        self._bin_lookup: dict[int, list[tuple[Any, float]]] = {b: items for b, items in self._bins}
        self._starts: dict[int, int] = {}
        self._cum_starts: list[int] = []
        self._cum_bins: list[int] = []
        pos = 0
        for b, items in self._bins:
            self._starts[b] = pos
            self._cum_starts.append(pos)
            self._cum_bins.append(b)
            pos += len(items)
        self.total_count = pos

    @staticmethod
    def bin_for_log_prob(log_prob: float, bin_width_bits: float = 1.0) -> int:
        """Return the quantized bit bin for a log probability."""
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")
        bits = max(0.0, -float(log_prob) / math.log(2.0))
        return int(math.floor(bits / bin_width_bits + 1.0e-12))

    @classmethod
    def from_enumerator(
        cls, enum: Iterator[tuple[Any, float]], max_bits: float, bin_width_bits: float = 1.0
    ) -> "QuantizedEnumerationIndex":
        """Build an index from an exact non-increasing-probability enumerator.

        Args:
            enum: Iterator yielding (value, log_prob) pairs in exact descending order.
            max_bits: Include only values with -log2(p) <= max_bits.
            bin_width_bits: Width of each quantized probability bin in bits.

        Returns:
            QuantizedEnumerationIndex over the bounded prefix/domain slice.

        """
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        bins: dict[int, list[tuple[Any, float]]] = {}
        truncated = False
        limit = float(max_bits) + 1.0e-12

        for value, log_prob in enum:
            if log_prob == -np.inf:
                continue
            bits = max(0.0, -float(log_prob) / math.log(2.0))
            if bits > limit:
                truncated = True
                break
            b = int(math.floor(bits / bin_width_bits + 1.0e-12))
            bins.setdefault(b, []).append((value, float(log_prob)))

        return cls([(b, bins[b]) for b in sorted(bins)], bin_width_bits, max_bits, truncated)

    @classmethod
    def from_items(
        cls,
        items: Sequence[tuple[Any, float]],
        max_bits: float,
        bin_width_bits: float = 1.0,
        sorted_items: bool = False,
        truncated: bool | None = None,
    ) -> "QuantizedEnumerationIndex":
        """Build an index from known support items and exact log probabilities.

        Args:
            items: Finite collection of (value, log_prob) pairs. Zero-probability
                items with log_prob == -inf are ignored.
            max_bits: Include only values with -log2(p) <= max_bits.
            bin_width_bits: Width of each quantized probability bin in bits.
            sorted_items: If True, preserve the input order. Otherwise, stable-sort
                included items by descending log probability.
            truncated: Optional explicit truncation flag. If omitted, it is True
                when any finite-probability item was outside the bit bound.

        Returns:
            QuantizedEnumerationIndex over the bounded finite item set.

        """
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        scored = []
        excluded = False
        limit = float(max_bits) + 1.0e-12
        for value, log_prob in items:
            lp = float(log_prob)
            if lp == -np.inf:
                continue
            bits = max(0.0, -lp / math.log(2.0))
            if bits > limit:
                excluded = True
                continue
            scored.append((value, lp))

        if not sorted_items:
            scored.sort(key=lambda u: -u[1])

        bins: dict[int, list[tuple[Any, float]]] = {}
        for value, lp in scored:
            b = cls.bin_for_log_prob(lp, bin_width_bits)
            bins.setdefault(b, []).append((value, lp))

        is_truncated = excluded if truncated is None else bool(truncated)
        return cls([(b, bins[b]) for b in sorted(bins)], bin_width_bits, max_bits, is_truncated)

    def __len__(self) -> int:
        return self.total_count

    def bin_for_index(self, index: int) -> tuple[int, int]:
        """Return (bin_id, offset_within_bin) for a bounded quantized rank."""
        if index < 0:
            raise IndexError("index must be non-negative.")
        if index >= self.total_count:
            raise IndexError("index %d outside indexed range of %d items." % (index, self.total_count))
        j = bisect.bisect_right(self._cum_starts, index) - 1
        return self._cum_bins[j], index - self._cum_starts[j]

    def get(self, index: int) -> tuple[Any, float]:
        """Return the indexed (value, exact_log_prob) pair."""
        b, offset = self.bin_for_index(index)
        return self._bin_lookup[b][offset]

    def slice(self, start: int, k: int) -> list[tuple[Any, float]]:
        """Return up to k indexed pairs starting at start."""
        if start < 0:
            raise IndexError("start must be non-negative.")
        if k < 0:
            raise ValueError("k must be non-negative.")
        return list(itertools.islice(self.iter_from(start), k))

    def iter_from(self, start: int = 0) -> Iterator[tuple[Any, float]]:
        """Iterate indexed pairs from start to the end of the bounded index."""
        if start < 0:
            raise IndexError("start must be non-negative.")
        if start >= self.total_count:
            return
        pos = 0
        for _, items in self._bins:
            n = len(items)
            if start >= pos + n:
                pos += n
                continue
            local = max(0, start - pos)
            yield from items[local:]
            pos += n

    def bin_items(self, bin_id: int) -> list[tuple[Any, float]]:
        """Return all indexed items in a quantized probability bin."""
        return list(self._bin_lookup.get(bin_id, []))

    def summary(self) -> dict[str, Any]:
        """Return a compact description of the bounded index."""
        return {
            "max_bits": self.max_bits,
            "bin_width_bits": self.bin_width_bits,
            "total_count": self.total_count,
            "num_bins": len(self._bins),
            "truncated": self.truncated,
            "counts": dict(self.counts),
        }


def quantized_index(
    enum: Iterator[tuple[Any, float]], max_bits: float, bin_width_bits: float = 1.0
) -> QuantizedEnumerationIndex:
    """Convenience wrapper for QuantizedEnumerationIndex.from_enumerator."""
    return QuantizedEnumerationIndex.from_enumerator(enum, max_bits=max_bits, bin_width_bits=bin_width_bits)


class LazyQuantizedEnumerationIndex(QuantizedEnumerationIndex):
    """Quantized index whose bin counts are precomputed but items are unranked lazily.

    This supports compositional distributions where a dynamic program can count how
    many values fall in each quantized log-density bin without materializing every
    Cartesian-product value. The getter receives a bin id and offset within that bin
    and returns the exact (value, log_prob) pair for that quantized rank.
    """

    def __init__(
        self,
        counts: dict[int, int],
        bin_width_bits: float,
        max_bits: float,
        truncated: bool,
        getter: Callable[[int, int], tuple[Any, float]],
    ) -> None:
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        self.bin_width_bits = float(bin_width_bits)
        self.max_bits = float(max_bits)
        self.truncated = bool(truncated)
        self.counts: dict[int, int] = {int(b): int(n) for b, n in sorted(counts.items()) if int(n) > 0}
        self._getter = getter
        self._bins = [(b, self.counts[b]) for b in sorted(self.counts)]
        self._starts: dict[int, int] = {}
        self._cum_starts: list[int] = []
        self._cum_bins: list[int] = []
        pos = 0
        for b, n in self._bins:
            self._starts[b] = pos
            self._cum_starts.append(pos)
            self._cum_bins.append(b)
            pos += n
        self.total_count = pos

    def bin_for_index(self, index: int) -> tuple[int, int]:
        """Return (bin_id, offset_within_bin) for a bounded quantized rank."""
        if index < 0:
            raise IndexError("index must be non-negative.")
        if index >= self.total_count:
            raise IndexError("index %d outside indexed range of %d items." % (index, self.total_count))
        j = bisect.bisect_right(self._cum_starts, index) - 1
        return self._cum_bins[j], index - self._cum_starts[j]

    def get(self, index: int) -> tuple[Any, float]:
        """Return the indexed (value, exact_log_prob) pair."""
        b, offset = self.bin_for_index(index)
        return self._getter(b, offset)

    def iter_from(self, start: int = 0) -> Iterator[tuple[Any, float]]:
        """Iterate indexed pairs from start to the end of the bounded index."""
        if start < 0:
            raise IndexError("start must be non-negative.")
        if start >= self.total_count:
            return
        pos = 0
        for b, n in self._bins:
            if start >= pos + n:
                pos += n
                continue
            local = max(0, start - pos)
            for offset in range(local, n):
                yield self._getter(b, offset)
            pos += n

    def bin_items(self, bin_id: int) -> list[tuple[Any, float]]:
        """Return all indexed items in a quantized probability bin."""
        n = self.counts.get(bin_id, 0)
        return [self._getter(bin_id, i) for i in range(n)]


class QuantizedCrossIndex:
    """Aligned support rows for multiple distributions under quantized bit bounds.

    Each row is (value, log_probs), where log_probs[j] is the exact log-density of
    value under component j, or -inf when that component assigns zero mass. Rows are
    included when any component's bit cost is within its requested bound. The joint
    bin counts expose the cross-bin alignment that mixtures need and marginal bin
    counts cannot provide.
    """

    def __init__(
        self,
        items: Sequence[tuple[Any, Sequence[float]]],
        max_bits: Sequence[float],
        bin_width_bits: float,
        truncated: bool = False,
    ) -> None:
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")
        self.items = [(v, tuple(float(lp) for lp in lps)) for v, lps in items]
        self.max_bits = tuple(float(b) for b in max_bits)
        self.bin_width_bits = float(bin_width_bits)
        self.truncated = bool(truncated)
        self.num_components = len(self.max_bits)
        self.counts: dict[tuple[int | None, ...], int] = {}
        for _, lps in self.items:
            bins = tuple(
                None if lp == -np.inf else QuantizedEnumerationIndex.bin_for_log_prob(lp, self.bin_width_bits)
                for lp in lps
            )
            self.counts[bins] = self.counts.get(bins, 0) + 1
        self.total_count = len(self.items)

    @staticmethod
    def bits_for_log_prob(log_prob: float) -> float:
        """Return -log2(p), using inf for zero probability."""
        lp = float(log_prob)
        return np.inf if lp == -np.inf else max(0.0, -lp / math.log(2.0))

    @classmethod
    def from_items(
        cls,
        items: Sequence[tuple[Any, Sequence[float]]],
        max_bits: Sequence[float],
        bin_width_bits: float = 1.0,
        truncated: bool = False,
    ) -> "QuantizedCrossIndex":
        """Build a cross index from exact aligned support rows."""
        if isinstance(max_bits, np.ndarray):
            max_bits_tuple = tuple(float(x) for x in max_bits.tolist())
        elif isinstance(max_bits, (list, tuple)):
            max_bits_tuple = tuple(float(x) for x in max_bits)
        else:
            max_bits_tuple = (float(max_bits),)
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")

        filtered = []
        excluded = False
        limits = tuple(b + 1.0e-12 for b in max_bits_tuple)
        for value, log_probs in items:
            lps = tuple(float(lp) for lp in log_probs)
            if len(lps) != len(max_bits_tuple):
                raise ValueError("log_probs length does not match max_bits length.")
            bits = tuple(cls.bits_for_log_prob(lp) for lp in lps)
            if any(bits[i] <= limits[i] for i in range(len(bits))):
                filtered.append((value, lps))
            else:
                excluded = True

        filtered.sort(key=lambda u: min(cls.bits_for_log_prob(lp) for lp in u[1]))
        return cls(filtered, max_bits_tuple, bin_width_bits, truncated=truncated or excluded)

    def iter_items(self) -> Iterator[tuple[Any, tuple[float, ...]]]:
        """Iterate aligned support rows."""
        return iter(self.items)

    def summary(self) -> dict[str, Any]:
        """Return a compact description of the cross index."""
        return {
            "max_bits": self.max_bits,
            "bin_width_bits": self.bin_width_bits,
            "total_count": self.total_count,
            "num_joint_bins": len(self.counts),
            "truncated": self.truncated,
            "counts": dict(self.counts),
        }
