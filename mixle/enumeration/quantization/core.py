"""Structural quantized enumeration: count the support without enumerating it.

The smart-enumeration index (see :mod:`mixle.enumeration.algorithms` and
:class:`mixle.stats.compute.pdist.DistributionEnumerator`) is a *seek structure* over a
distribution's exact descending-probability enumeration: it precomputes how many
support values fall in each quantized log-probability bin so that an arbitrary rank
can be resolved and unranked without walking the prefix.

For exponential-support families (sequences, Markov chains, ...) the number of
values within a probability budget is astronomically large, so the index must be
built from per-bin **counts computed structurally** -- by quantizing and binning the
*complete* model probability ``Q(ln p(x))`` and lifting the model's own likelihood
recursion into a count/histogram semiring -- never by materializing the domain.

This module provides that semiring:

  - :class:`Quantizer` maps an exact ``log_prob`` to a fine integer bucket of
    accumulated bits and a coarse output bin.
  - :class:`CountHistogram` is the semiring value: counts indexed by fine bucket,
    with ``shift`` (add a constant log-prob term), ``convolve`` (independent additive
    composition), ``power`` (L-fold self-convolution), and ``add`` (pool alternatives).
    Counts are exact Python integers and may exceed 2**128.
  - :class:`CountIndex` pairs a histogram with a structural *unranker*
    ``get_in_bucket(fine_bucket, offset) -> (value, exact_log_prob)``.
  - :func:`leaf_count_index` builds a :class:`CountIndex` from any exact enumerator,
    bounded by depth (efficient for small/closed-form leaves).
  - :func:`convolve_indices` composes child indices (the Composite reference case),
    and :func:`build_budget_index` accumulates coarse bins until the cumulative count
    reaches the requested ``2**budget_bits`` budget, returning a
    :class:`mixle.enumeration.algorithms.LazyQuantizedEnumerationIndex`.

The bin assignment is the only approximation (intermediate fine-bucket rounding shifts
items by at most ``num_terms / oversample`` coarse bins); the *value set*, *total
count*, and the *exact log probability* of every unranked item are exact, because the
unranker returns the value and the index re-evaluates ``log_density`` on it.
"""

import bisect
import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from mixle.utils.optional_deps import gmpy2

_LOG2 = math.log(2.0)
_TOL = 1.0e-9

# Above this many scalar products (len(a) * len(b)) a convolution switches from the direct
# double loop to Kronecker substitution. Below it, the big-integer pack/unpack overhead is not
# worth it (and delta-shaped operands convolve trivially). Tuned from microbenchmarks.
_KRONECKER_MIN_PRODUCT = 2048

# Above this packed-operand bit length, route the Kronecker multiply through gmpy2/GMP (FFT-based)
# when available. GMP's FFT crossover over Karatsuba is a few thousand bits; 50k bits keeps the mpz
# round-trip off small multiplies while capturing the ~100x win on the multi-megabyte operands that
# wide deep-sequence convolutions produce.
_GMPY2_MIN_BITS = 50_000

# --- exact NTT backend (Pollard 1971): vectorized number-theoretic transform + CRT -----------------
# Two NTT-friendly primes p = c*2^k + 1 with known primitive roots, both < 2^31 so every butterfly
# product fits uint64 EXACTLY (a*b < 2^62), and their product (~2^61.66) bounds the largest output
# coefficient the two-prime CRT reconstructs exactly. An earlier prototype concluded "NTT loses to
# Kronecker" — that verdict was an implementation artifact: it reduced/reconstructed the coefficients
# with per-element Python big-int arithmetic, which dominated the C-level transform. When the
# coefficients FIT machine words (the common depth-bounded-budget case), the whole pipeline —
# reduction, transforms, pointwise multiply, CRT — stays inside vectorized uint64 numpy with no
# Python-level big-int work at all, and the classical algorithm wins as it should.
_NTT_P1, _NTT_G1 = 2013265921, 31  # 15 * 2^27 + 1
_NTT_P2, _NTT_G2 = 1811939329, 13  # 27 * 2^26 + 1
# (prime, primitive root, max transform length = its 2-adic capacity). All primes < 2^31 so every
# butterfly product fits uint64; roots verified primitive (g^((p-1)/q) != 1 for all q | p-1).
_NTT_PRIMES = (
    (_NTT_P1, _NTT_G1, 1 << 27),
    (_NTT_P2, _NTT_G2, 1 << 26),
    (469762049, 3, 1 << 26),  # 7 * 2^26 + 1
    (754974721, 11, 1 << 24),  # 45 * 2^24 + 1
    (998244353, 3, 1 << 23),  # 119 * 2^23 + 1
    (167772161, 3, 1 << 25),  # 5 * 2^25 + 1
    (1004535809, 3, 1 << 21),  # 479 * 2^21 + 1
)
_NTT_CRT_INV = pow(_NTT_P1, _NTT_P2 - 2, _NTT_P2)  # p1^{-1} mod p2
_NTT_MAX_COEFF = _NTT_P1 * _NTT_P2  # exclusive bound on outputs the pure-uint64 two-prime CRT covers
_NTT_MAX_COEFF_FULL = math.prod(p for p, _g, _n in _NTT_PRIMES)  # ~2^207: the full-ladder ceiling
_NTT_ENABLED = True  # test hook: force the Kronecker backend by flipping this off
_ntt_tables_cache: dict[tuple[int, int], tuple[Any, Any, Any, int]] = {}


def _ntt_tables(p: int, g: int, n: int) -> tuple[Any, Any, Any, int]:
    """Cached (bit-reversal index, per-stage forward twiddles, per-stage inverse twiddles, n^-1 mod p)."""
    key = (p, n)
    tab = _ntt_tables_cache.get(key)
    if tab is not None:
        return tab
    bits = n.bit_length() - 1
    idx = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for _ in range(bits):
        rev = (rev << 1) | (idx & 1)
        idx >>= 1
    pp = np.uint64(p)
    fwd: list[Any] = []
    inv: list[Any] = []
    length = 2
    while length <= n:
        half = length >> 1
        root = pow(g, (p - 1) // length, p)
        for stages, r in ((fwd, root), (inv, pow(root, p - 2, p))):
            w = np.empty(half, dtype=np.uint64)
            w[0] = 1
            m, cur = 1, r
            while m < half:  # doubling: w[m:2m] = w[:m] * r^m — O(log half) vector ops
                step = min(m, half - m)
                w[m : m + step] = (w[:step] * np.uint64(cur)) % pp
                cur = (cur * cur) % p
                m <<= 1
            stages.append(w)
        length <<= 1
    tab = (rev, fwd, inv, pow(n, p - 2, p))
    _ntt_tables_cache[key] = tab
    return tab


def _ntt_transform(x: Any, p: int, stages: Any, rev: Any) -> Any:
    """In-place-style iterative Cooley-Tukey NTT of ``x`` (length a power of two) modulo ``p``."""
    x = x[rev]
    pp = np.uint64(p)
    length, s = 2, 0
    n = len(x)
    while length <= n:
        half = length >> 1
        xv = x.reshape(-1, length)
        u = xv[:, :half]
        v = (xv[:, half:] * stages[s]) % pp
        hi = (u + pp - v) % pp  # both outputs read the ORIGINAL u before either write
        lo = (u + v) % pp
        xv[:, :half] = lo
        xv[:, half:] = hi
        length <<= 1
        s += 1
    return x


def _packed_bits_estimate(a: list[int], b: list[int]) -> int:
    """The Kronecker packed-operand bit length this multiply would produce (backend routing only)."""
    bit_l = max(a).bit_length() + max(b).bit_length() + min(len(a), len(b)).bit_length() + 1
    return 8 * ((bit_l + 7) // 8) * max(len(a), len(b))


def _kept_coeff_bound(a: list[int], b: list[int], width: int, delta_bits: float | None) -> int:
    """A sound MEASURED bound on every kept output coefficient (index < ``width``).

    Baseline: the arithmetic bound ``min(len)*max(a)*max(b)``. When the caller supplies the
    histogram's bucket pitch ``delta_bits`` (bits per fine bucket), sharpen it with the measured
    exponential envelopes ``a_i <= 2^(C_a + i*delta)`` where ``C_a = max_i(bitlen(a_i) - i*delta)``
    (and likewise ``C_b``): then ``out[k] = sum_i a_i b_{k-i} <= min(len) * 2^(C_a + C_b + k*delta)``.
    Count-DP histograms hug that envelope (counts grow with the bucket's bits), so the kept low-order
    outputs get a bound near their true size even when the raw maxima sit far beyond the CRT range —
    with nothing assumed: both envelopes are computed from the data.
    """
    bound = min(len(a), len(b)) * max(a) * max(b)
    if delta_bits is None or delta_bits <= 0:
        return bound
    c_a = max(v.bit_length() - i * delta_bits for i, v in enumerate(a) if v)
    c_b = max(v.bit_length() - i * delta_bits for i, v in enumerate(b) if v)
    env_bits = c_a + c_b + delta_bits * (width - 1) + math.log2(min(len(a), len(b), width)) + 1.0
    if env_bits < bound.bit_length():
        return 1 << int(math.ceil(env_bits))
    return bound


def _convolve_ntt(a: list[int], b: list[int], width: int, bucket_delta_bits: float | None = None) -> list[int] | None:
    """Exact integer convolution via a multi-prime NTT + CRT, vectorized in uint64.

    Returns ``None`` when a KEPT output coefficient (index < ``width``) could exceed the prime
    ladder's CRT capacity (~2^207) — the caller falls back to Kronecker substitution, which is exact
    for arbitrarily large counts. Eligibility and the number of primes come from
    :func:`_kept_coeff_bound` (measured, assumption-free). A wrapped coefficient beyond ``width`` is
    harmless: the transform length covers the full linear convolution (no cyclic aliasing), each
    coefficient reconstructs independently, and the out-of-range ones are exactly the ones sliced
    away. Within capacity the result is bit-identical to the other backends.
    """
    max_a = max(a)
    max_b = max(b)
    if max_a == 0 or max_b == 0:
        return [0] * width
    bound = _kept_coeff_bound(a, b, width, bucket_delta_bits)
    if not _NTT_ENABLED or bound >= _NTT_MAX_COEFF_FULL:
        return None
    n_primes = 1
    cap = _NTT_PRIMES[0][0]
    while cap <= bound:
        cap *= _NTT_PRIMES[n_primes][0]
        n_primes += 1

    full = len(a) + len(b) - 1
    n = 1
    while n < full:
        n <<= 1
    if any(n > max_n for _p, _g, max_n in _NTT_PRIMES[:n_primes]):
        return None  # transform longer than a used prime's 2-adic capacity
    if max_a.bit_length() < 64 and max_b.bit_length() < 64:
        fa = np.zeros(n, dtype=np.uint64)
        fb = np.zeros(n, dtype=np.uint64)
        fa[: len(a)] = a
        fb[: len(b)] = b
        reduce = lambda arr, pp: arr % pp  # noqa: E731 - vectorized residues
    else:
        fa, fb = a, b  # counts above 2^64: reduce per prime with Python int mod (O(n), low-overhead vs the multiply)

        def reduce(arr: Any, pp: Any) -> Any:
            out = np.zeros(n, dtype=np.uint64)
            out[: len(arr)] = [c % int(pp) for c in arr]
            return out

    residues = []
    for p, g, _max_n in _NTT_PRIMES[:n_primes]:
        rev, fwd, inv, n_inv = _ntt_tables(p, g, n)
        pp = np.uint64(p)
        ha = _ntt_transform(reduce(fa, pp), p, fwd, rev)
        hb = _ntt_transform(reduce(fb, pp), p, fwd, rev)
        hc = _ntt_transform((ha * hb) % pp, p, inv, rev)
        residues.append(((hc * np.uint64(n_inv)) % pp)[:width])

    if n_primes == 1:
        return residues[0].tolist()
    if n_primes == 2:
        a1, a2 = residues
        p2 = np.uint64(_NTT_P2)
        t = ((a2 + p2 - a1 % p2) * np.uint64(_NTT_CRT_INV)) % p2
        return (a1 + np.uint64(_NTT_P1) * t).tolist()  # < p1*p2 < 2^62: exact in uint64

    # 3+ primes: Garner mixed-radix digits, every intermediate in uint64 (digits and moduli < 2^31,
    # so d*p products < 2^62); only the final positional assembly leaves machine words, one low-overhead
    # Python big-int expression per KEPT coefficient (the same order of per-element work as
    # Kronecker's from_bytes unpack).
    digits = [residues[0]]
    for j in range(1, n_primes):
        pj = np.uint64(_NTT_PRIMES[j][0])
        acc = residues[j]
        scale = np.uint64(1)
        for i in range(j):
            di = digits[i] % pj
            acc = (acc + pj - (di * scale) % pj) % pj
            scale = (scale * np.uint64(_NTT_PRIMES[i][0])) % pj
        digits.append((acc * np.uint64(pow(int(scale), _NTT_PRIMES[j][0] - 2, _NTT_PRIMES[j][0]))) % pj)
    lists = [d.tolist() for d in digits]
    prefix = [1]
    for p, _g, _max_n in _NTT_PRIMES[: n_primes - 1]:
        prefix.append(prefix[-1] * p)
    return [sum(dj * pj for dj, pj in zip(row, prefix)) for row in zip(*lists)]


def _convolve_kronecker(a: list[int], b: list[int], width: int) -> list[int]:
    """Exact integer convolution of ``a`` and ``b`` via one big-integer multiply.

    Pack each histogram into a single integer whose base-``2**(8*slot)`` digits are the coefficients
    (``slot`` bytes wide), multiply (CPython uses Karatsuba once the operands are large), and read
    back the low ``width`` digits. ``slot`` is chosen so no output coefficient overflows its digit
    -- the max output coefficient is ``< min(len a, len b) * max(a) * max(b)`` -- so digits never
    carry into one another and the result is exact for arbitrarily large counts.

    Packing and unpacking go through ``int.from_bytes`` / ``int.to_bytes`` so they are linear in the
    output size; a digit-at-a-time shift/or would be quadratic and would erase the multiply's win.
    """
    max_a = max(a)
    max_b = max(b)
    if max_a == 0 or max_b == 0:
        return [0] * width
    terms = min(len(a), len(b))
    bit_l = max_a.bit_length() + max_b.bit_length() + terms.bit_length() + 1
    slot = (bit_l + 7) // 8  # bytes per coefficient (byte-aligned for clean unpacking)

    buf_a = bytearray(slot * len(a))
    for i, c in enumerate(a):
        if c:
            buf_a[i * slot : i * slot + slot] = c.to_bytes(slot, "little")
    buf_b = bytearray(slot * len(b))
    for i, c in enumerate(b):
        if c:
            buf_b[i * slot : i * slot + slot] = c.to_bytes(slot, "little")

    int_a = int.from_bytes(buf_a, "little")
    int_b = int.from_bytes(buf_b, "little")
    # Route the multiply through GMP's FFT-based bignum multiply when gmpy2 is installed and the
    # operands are large: CPython's int multiply is Karatsuba (O(n^1.585)), which degrades into tens
    # of seconds once the packed operands reach a few megabytes, whereas GMP uses Schoenhage-Strassen
    # (O(n log n)) and is ~100x faster there. The result is bit-identical (exact integer multiply);
    # below the threshold the mpz round-trip is not worth it, so plain int is used.
    if gmpy2 is not None and min(int_a.bit_length(), int_b.bit_length()) > _GMPY2_MIN_BITS:
        product = int(gmpy2.mpz(int_a) * gmpy2.mpz(int_b))
    else:
        product = int_a * int_b
    prod_bytes = product.to_bytes((product.bit_length() + 7) // 8 or 1, "little")

    out = [0] * width
    avail = len(prod_bytes)
    for k in range(width):
        lo = k * slot
        if lo >= avail:
            break
        out[k] = int.from_bytes(prod_bytes[lo : lo + slot], "little")
    return out


class Quantizer:
    """Maps exact log probabilities to fine buckets and coarse bins.

    bits(x) = -log2 p(x) >= 0. The fine bucket is floor(bits * oversample / bin_width);
    the coarse bin is fine_bucket // oversample, which equals floor(bits / bin_width) for
    a single value (so leaf coarse bins match the existing QuantizedEnumerationIndex
    convention). ``oversample`` (R) controls how finely accumulated log-probabilities are
    tracked through convolutions, bounding the requantization error.
    """

    __slots__ = ("bin_width_bits", "oversample", "executor", "count_mode")

    def __init__(
        self, bin_width_bits: float = 1.0, oversample: int = 8, executor=None, count_mode: str = "exact"
    ) -> None:
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")
        if int(oversample) < 1:
            raise ValueError("oversample must be a positive integer.")
        if count_mode not in ("exact", "float"):
            raise ValueError("count_mode must be 'exact' or 'float'")
        self.bin_width_bits = float(bin_width_bits)
        self.oversample = int(oversample)
        # Optional convolution executor (see mixle.enumeration.quantization.parallel). Lives only in the
        # building process; the count-DP routes its heavy convolutions through it when present.
        self.executor = executor
        # 'exact' carries counts as arbitrary-precision integers; 'float' carries them as float64 --
        # C-speed numpy convolutions, exact below 2**53 and ~1e-16 relative error per operation beyond
        # (see CountHistogram.convolve_float). The approximate-counting knob for deep budgets.
        self.count_mode = count_mode

    def convolve(
        self, a: "CountHistogram", b: "CountHistogram", max_fine_bucket: int | None = None
    ) -> "CountHistogram":
        """Convolve two histograms, using the attached parallel executor when present."""
        if self.count_mode == "float":
            return a.convolve_float(b, max_fine_bucket=max_fine_bucket)
        if self.executor is not None:
            return self.executor.convolve(a, b, max_fine_bucket)
        return a.convolve(b, max_fine_bucket=max_fine_bucket, bucket_delta_bits=self.bin_width_bits / self.oversample)

    def bits(self, log_prob: float) -> float:
        """Information content -log2 p in bits (>= 0)."""
        return max(0.0, -float(log_prob) / _LOG2)

    def fine_bucket(self, log_prob: float) -> int:
        """Fine integer bucket of a log probability."""
        b = self.bits(log_prob) * self.oversample / self.bin_width_bits
        return int(math.floor(b + _TOL))

    def coarse_bin(self, fine_bucket: int) -> int:
        """Coarse output bin containing a fine bucket."""
        return int(fine_bucket) // self.oversample

    def fine_per_bit(self) -> float:
        """Fine buckets per bit of information."""
        return self.oversample / self.bin_width_bits


class CountHistogram:
    """Counts of support values indexed by fine bucket of accumulated bits.

    ``data[i]`` is the (exact, possibly huge) number of values whose fine bucket is
    ``base + i``. The histogram is dense over ``[base, base + len(data))`` with implicit
    zeros outside. This is the value type of the count semiring.
    """

    __slots__ = ("base", "data")

    def __init__(self, base: int, data: list[int]) -> None:
        self.base = int(base)
        self.data = list(data)
        self._normalize()

    def _normalize(self) -> None:
        d = self.data
        # Trim leading zeros (advancing base) and trailing zeros.
        lo = 0
        n = len(d)
        while lo < n and d[lo] == 0:
            lo += 1
        if lo == n:
            self.base = 0
            self.data = []
            return
        hi = n
        while hi > lo and d[hi - 1] == 0:
            hi -= 1
        if lo != 0 or hi != n:
            self.base += lo
            self.data = d[lo:hi]

    @classmethod
    def empty(cls) -> "CountHistogram":
        """Return an empty histogram."""
        return cls(0, [])

    @classmethod
    def delta(cls, fine_bucket: int, count: int = 1) -> "CountHistogram":
        """A single bucket with the given count (the multiplicative identity when count=1, bucket=0)."""
        return cls(fine_bucket, [int(count)])

    def is_empty(self) -> bool:
        """Return whether the histogram has no stored counts."""
        return not self.data

    def total(self) -> int:
        """Return the total count across buckets."""
        return sum(self.data)

    def max_bucket(self) -> int | None:
        """Return the largest occupied fine bucket, if any."""
        return None if not self.data else self.base + len(self.data) - 1

    def count_at(self, fine_bucket: int) -> int:
        """Return the count at a fine bucket."""
        i = int(fine_bucket) - self.base
        return self.data[i] if 0 <= i < len(self.data) else 0

    def shift(self, k: int) -> "CountHistogram":
        """Return a copy with every bucket moved by k (adds a constant log-prob term)."""
        if not self.data:
            return CountHistogram.empty()
        return CountHistogram(self.base + int(k), list(self.data))

    def truncate(self, max_fine_bucket: int) -> "CountHistogram":
        """Drop buckets strictly beyond ``max_fine_bucket`` (depth bound)."""
        if not self.data:
            return CountHistogram.empty()
        hi = int(max_fine_bucket) - self.base + 1
        if hi <= 0:
            return CountHistogram.empty()
        if hi >= len(self.data):
            return CountHistogram(self.base, list(self.data))
        return CountHistogram(self.base, self.data[:hi])

    def add(self, other: "CountHistogram") -> "CountHistogram":
        """Pointwise sum (pool of mutually exclusive alternatives, e.g. different lengths)."""
        if not self.data:
            return CountHistogram(other.base, list(other.data))
        if not other.data:
            return CountHistogram(self.base, list(self.data))
        base = min(self.base, other.base)
        end = max(self.base + len(self.data), other.base + len(other.data))
        out = [0] * (end - base)
        for i, c in enumerate(self.data):
            if c:
                out[self.base + i - base] += c
        for i, c in enumerate(other.data):
            if c:
                out[other.base + i - base] += c
        return CountHistogram(base, out)

    def convolve(
        self, other: "CountHistogram", max_fine_bucket: int | None = None, bucket_delta_bits: float | None = None
    ) -> "CountHistogram":
        """Discrete convolution: counts of sums of two independent additive log-prob terms.

        Optionally drop output buckets beyond ``max_fine_bucket`` during accumulation so a
        depth bound keeps the histogram width fixed regardless of how large the counts grow.

        Three backends produce identical results: a direct double loop for small operands, a
        vectorized multi-prime number-theoretic transform (:func:`_convolve_ntt`) when every KEPT
        output coefficient provably fits the prime ladder's CRT capacity (``bucket_delta_bits`` — the
        quantizer's bits-per-fine-bucket pitch — lets the measured-envelope bound engage; see
        :func:`_kept_coeff_bound`), and Kronecker substitution (:func:`_convolve_kronecker`)
        otherwise. The latter packs each integer histogram into a single big integer and multiplies
        once, pushing the inner loop into CPython's sub-quadratic big-integer multiply -- exact for
        arbitrarily large counts (they can exceed the ladder's ~2^207, so an unchecked
        floating-point FFT is never an option).
        """
        a, b = self.data, other.data
        if not a or not b:
            return CountHistogram.empty()
        base = self.base + other.base
        width = len(a) + len(b) - 1
        if max_fine_bucket is not None:
            cap = int(max_fine_bucket) - base + 1
            if cap <= 0:
                return CountHistogram.empty()
            width = min(width, cap)
        # Only output buckets [0, width) are kept, and out[k] (k < width) draws solely on
        # a[i], b[j] with i, j < width, so trimming both inputs to that prefix bounds the work.
        if len(a) > width:
            a = a[:width]
        if len(b) > width:
            b = b[:width]
        if len(a) * len(b) > _KRONECKER_MIN_PRODUCT:
            # Backend choice, measured: the vectorized NTT beats CPython-bigint Kronecker 3-14x across
            # the eligible range, but GMP's FFT multiply still wins once the packed operands are large
            # enough to clear the gmpy2 threshold (~0.6-0.9x there) — so with gmpy2 installed AND a
            # huge multiply ahead, keep Kronecker; otherwise try the NTT first.
            out = None
            if gmpy2 is None or _packed_bits_estimate(a, b) <= _GMPY2_MIN_BITS:
                out = _convolve_ntt(a, b, width, bucket_delta_bits)  # exact when outputs fit the CRT range
            if out is None:
                out = _convolve_kronecker(a, b, width)
        else:
            out = [0] * width
            for i, ai in enumerate(a):
                if not ai:
                    continue
                hi = width - i
                if hi <= 0:
                    break
                for j in range(min(len(b), hi)):
                    bj = b[j]
                    if bj:
                        out[i + j] += ai * bj
        return CountHistogram(base, out)

    def convolve_float(self, other: "CountHistogram", max_fine_bucket: int | None = None) -> "CountHistogram":
        """Approximate convolution carrying counts as float64 -- the quantized-counting fast path.

        One :func:`numpy.convolve` at C speed replaces the exact big-integer machinery. Counts are
        **exact while they stay below 2**53** (float64 integers) and carry a relative error of at most
        ``~width * 2**-53`` per convolution beyond that -- negligible against the bin-assignment
        approximation the count-DP already makes. Counts above ``~2**1000`` would overflow float64, so
        such results fall back to the exact backend (the histograms are identical below the cliff).
        """
        a, b = self.data, other.data
        if not a or not b:
            return CountHistogram.empty()
        base = self.base + other.base
        width = len(a) + len(b) - 1
        if max_fine_bucket is not None:
            cap = int(max_fine_bucket) - base + 1
            if cap <= 0:
                return CountHistogram.empty()
            width = min(width, cap)
        fa = np.asarray(a[:width] if len(a) > width else a, dtype=np.float64)
        fb = np.asarray(b[:width] if len(b) > width else b, dtype=np.float64)
        if not (np.isfinite(fa).all() and np.isfinite(fb).all()):
            return self.convolve(other, max_fine_bucket=max_fine_bucket)  # counts too large for float64
        out = np.convolve(fa, fb)[:width]
        if not np.isfinite(out).all():
            return self.convolve(other, max_fine_bucket=max_fine_bucket)
        return CountHistogram(base, out.tolist())


class CountIndex:
    """A fine-bucket count histogram paired with a structural unranker.

    ``get_in_bucket(fine_bucket, offset)`` returns ``(value, exact_log_prob)`` for the
    ``offset``-th value (0-based) whose fine bucket is ``fine_bucket``. The ordering within
    a bucket is deterministic but otherwise unspecified.
    """

    __slots__ = ("hist", "_getter", "dropped_upper")

    def __init__(
        self, hist: CountHistogram, getter: Callable[[int, int], tuple[Any, float]], dropped_upper: float = 0.0
    ) -> None:
        self.hist = hist
        self._getter = getter
        # A sound upper bound on in-budget values EXCLUDED from the histogram by an explicit
        # approximation knob (e.g. the autoregressive ``branch_cap``). 0.0 for exhaustive indices.
        # Distinct from budget truncation: deepening cannot recover these, so ``truncated`` stays
        # False for them and count queries report the bracket [total, total + dropped_upper].
        self.dropped_upper = float(dropped_upper)

    def total(self) -> int:
        """Return total count represented by the index."""
        return self.hist.total()

    def get_in_bucket(self, fine_bucket: int, offset: int) -> tuple[Any, float]:
        """Return the item at an offset within a fine bucket."""
        if offset < 0 or offset >= self.hist.count_at(fine_bucket):
            raise IndexError("offset %d outside fine bucket %d" % (offset, fine_bucket))
        return self._getter(int(fine_bucket), int(offset))


def leaf_count_index(
    enum: Iterator[tuple[Any, float]], quantizer: Quantizer, max_fine_bucket: int, max_items: int | None = None
) -> tuple[CountIndex, bool]:
    """Build a CountIndex from an exact descending-probability enumerator, bounded by depth.

    Pulls items until their fine bucket exceeds ``max_fine_bucket`` (or, if ``max_items`` is given,
    until that many items have been taken). Efficient for closed-form or small-support leaves (a
    geometric/Poisson has ~depth items within a depth bound); the ``max_items`` cap is the brake for
    the enumerate-and-bin fallback over exponential-support families that cannot count structurally.
    Returns ``(index, truncated)`` where ``truncated`` is True if in-bound items were left untaken.
    """
    by_bucket: dict[int, list[tuple[Any, float]]] = {}
    truncated = False
    taken = 0
    for value, log_prob in enum:
        if log_prob == -math.inf:
            continue
        fb = quantizer.fine_bucket(log_prob)
        if fb > max_fine_bucket:
            truncated = True
            break
        by_bucket.setdefault(fb, []).append((value, float(log_prob)))
        taken += 1
        if max_items is not None and taken >= max_items:
            truncated = True
            break

    if not by_bucket:
        return CountIndex(CountHistogram.empty(), lambda fb, off: (_ for _ in ()).throw(IndexError())), truncated

    lo = min(by_bucket)
    hi = max(by_bucket)
    data = [0] * (hi - lo + 1)
    for fb, items in by_bucket.items():
        data[fb - lo] = len(items)
    hist = CountHistogram(lo, data)

    def getter(fb: int, off: int) -> tuple[Any, float]:
        return by_bucket[fb][off]

    return CountIndex(hist, getter), truncated


def child_count_index(child, path: str, quantizer: Quantizer, max_fine_bucket: int) -> tuple[CountIndex, bool]:
    """Build child.quantized_count_index(...), annotating EnumerationError with the child's path."""
    from mixle.stats.compute.pdist import EnumerationError

    try:
        return child.quantized_count_index(quantizer, max_fine_bucket)
    except EnumerationError as e:
        new_path = path if not e.path else "%s -> %s" % (path, e.path)
        raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None


def convolve_indices(children: Sequence[CountIndex], quantizer: Quantizer, max_fine_bucket: int) -> CountIndex:
    """Compose independent child indices into their additive (convolution) product.

    The joint histogram is the convolution of the child histograms (capped at the depth
    bound). Unranking resolves the per-child fine buckets that sum to the target, then the
    per-child offsets via mixed-radix decomposition using suffix-convolution counts. This is
    the Composite reference case; the empty product is the single empty tuple at bucket 0.
    """
    n = len(children)
    if n == 0:
        empty_hist = CountHistogram.delta(0, 1)
        return CountIndex(empty_hist, lambda fb, off: ((), 0.0))

    # Suffix convolutions: suffix[i] = conv(children[i].hist, ..., children[n-1].hist).
    suffix: list[CountHistogram] = [None] * (n + 1)  # type: ignore
    suffix[n] = CountHistogram.delta(0, 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = quantizer.convolve(children[i].hist, suffix[i + 1], max_fine_bucket=max_fine_bucket)

    joint = suffix[0]

    def getter(fb: int, off: int) -> tuple[Any, float]:
        values: list[Any] = []
        log_prob = 0.0
        remaining = int(fb)
        o = int(off)
        for i in range(n):
            child = children[i]
            tail = suffix[i + 1]
            # Iterate this child's buckets in increasing order; completions for the
            # remaining children at (remaining - b) come from the suffix histogram.
            chosen = None
            for b in range(child.hist.base, child.hist.base + len(child.hist.data)):
                ci = child.hist.count_at(b)
                if not ci:
                    continue
                rem = remaining - b
                m = tail.count_at(rem)
                if not m:
                    continue
                block = ci * m
                if o < block:
                    local = o // m  # index of this child's item within bucket b
                    o = o % m  # offset into the remaining-children block
                    cval, clp = child.get_in_bucket(b, local)
                    values.append(cval)
                    log_prob += clp
                    remaining = rem
                    chosen = b
                    break
                o -= block
            if chosen is None:
                raise IndexError("offset outside convolution bucket %d" % fb)
        return tuple(values), float(log_prob)

    return CountIndex(joint, getter)


# ---------------------------------------------------------------------------
# Budget-driven coarse index (the new count-budget mode).
# ---------------------------------------------------------------------------


def build_budget_index(
    index: CountIndex,
    quantizer: Quantizer,
    budget_bits: float,
    value_combine: Callable[[Any], Any] | None = None,
    exact_log_density: Callable[[Any], float] | None = None,
    truncated: bool = False,
):
    """Wrap a CountIndex as a budget-bounded LazyQuantizedEnumerationIndex.

    Accumulates coarse bins (fine buckets grouped by ``quantizer.oversample``) in
    descending-probability order until the cumulative count reaches ``2**budget_bits``, then
    stops. The returned getter maps a coarse (bin, offset) to a fine (bucket, offset),
    unranks the structural value, optionally maps it with ``value_combine`` (e.g. tuple->list),
    and reports the exact log density (recomputed via ``exact_log_density`` when supplied,
    otherwise the structurally accumulated log probability).
    """
    from mixle.enumeration.quantization.seek import LazyQuantizedEnumerationIndex

    R = quantizer.oversample
    bw = quantizer.bin_width_bits
    hist = index.hist
    budget = None if budget_bits is None else _two_pow(budget_bits)

    # Group fine buckets into coarse bins, in increasing depth (descending probability).
    coarse_counts: dict[int, int] = {}
    # For each coarse bin: ordered list of (fine_bucket, count) and the cumulative offset
    # boundaries, so a within-bin offset maps to a fine bucket in O(log #fine).
    coarse_layout: dict[int, tuple[list[int], list[int], list[int]]] = {}
    cumulative = 0
    covered_truncated = truncated
    if hist.data:
        # Walk coarse bins in order; within a coarse bin walk its fine buckets in order.
        by_coarse: dict[int, list[tuple[int, int]]] = {}
        for i, c in enumerate(hist.data):
            if not c:
                continue
            fb = hist.base + i
            cb = fb // R
            by_coarse.setdefault(cb, []).append((fb, c))
        stop = False
        for cb in sorted(by_coarse):
            if stop:
                covered_truncated = True
                break
            fine_buckets = []
            fine_counts = []
            starts = []
            running = 0
            bin_total = 0
            for fb, c in by_coarse[cb]:
                fine_buckets.append(fb)
                fine_counts.append(c)
                starts.append(running)
                running += c
                bin_total += c
            coarse_counts[cb] = bin_total
            coarse_layout[cb] = (fine_buckets, starts, fine_counts)
            cumulative += bin_total
            if budget is not None and cumulative >= budget:
                stop = True  # budget met; include this bin, then stop adding deeper bins.
        else:
            # Loop finished without breaking: did we exhaust the histogram?
            covered_truncated = covered_truncated  # leave as-is (depth bound handled by caller)

    def getter(bin_id: int, offset: int) -> tuple[Any, float]:
        layout = coarse_layout.get(bin_id)
        if layout is None:
            raise IndexError("coarse bin %d not indexed" % bin_id)
        fine_buckets, starts, fine_counts = layout
        # Find the fine bucket whose [start, start+count) range contains offset.
        j = bisect.bisect_right(starts, offset) - 1
        if j < 0 or offset >= starts[j] + fine_counts[j]:
            raise IndexError("offset %d outside coarse bin %d" % (offset, bin_id))
        fb = fine_buckets[j]
        value, lp = index.get_in_bucket(fb, offset - starts[j])
        if value_combine is not None:
            value = value_combine(value)
        if exact_log_density is not None:
            lp = float(exact_log_density(value))
        return value, lp

    max_bin = max(coarse_counts) if coarse_counts else 0
    return LazyQuantizedEnumerationIndex(
        coarse_counts, bin_width_bits=bw, max_bits=float(max_bin) * bw, truncated=covered_truncated, getter=getter
    )


def count_budget_index(
    dist,
    budget_bits: float,
    bin_width_bits: float = 1.0,
    oversample: int = 8,
    max_depth_bits: float = 4096.0,
    num_workers: int | None = None,
    count_mode: str = "exact",
):
    """Driver for the count-budget mode: deepen until the budget is covered, then build the index.

    Calls ``dist.quantized_count_index(quantizer, max_fine_bucket)`` at geometrically increasing
    depths until the structural count reaches ``2**budget_bits`` or the support is exhausted (no
    further truncation), then wraps the result with :func:`build_budget_index`. The exact log
    density of each unranked value is reported via ``dist.log_density``.

    When ``num_workers`` is greater than 1, the heavy count-histogram convolutions are computed on
    a process pool (parallel quantization); the parallel result is identical to the serial one.
    ``count_mode='float'`` carries counts as float64 (C-speed convolutions, exact below 2**53,
    ~1e-16 relative error per operation beyond) -- the approximate-counting mode for deep budgets.
    """
    executor = None
    if num_workers is not None and int(num_workers) > 1:
        from mixle.enumeration.quantization.parallel import ConvolutionExecutor

        executor = ConvolutionExecutor(num_workers=num_workers)
    try:
        if executor is not None:
            executor.__enter__()
        q = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample, executor=executor, count_mode=count_mode)
        budget = _two_pow(budget_bits)
        depth_bits = max(float(bin_width_bits), float(budget_bits))
        index = None
        truncated = True
        fine_per_bit = q.fine_per_bit()
        while True:
            max_fb = int(math.ceil(depth_bits * fine_per_bit))
            index, truncated = dist.quantized_count_index(q, max_fb)
            if index.total() >= budget or not truncated:
                break
            if depth_bits >= max_depth_bits:
                break
            depth_bits = min(depth_bits * 2.0, max_depth_bits)
        return build_budget_index(index, q, budget_bits, exact_log_density=dist.log_density, truncated=truncated)
    finally:
        if executor is not None:
            executor.close()


def distinct_budget_stream(
    dist,
    budget_bits: float,
    bin_width_bits: float = 1.0,
    oversample: int = 8,
    dedup: str = "canonical",
    start: int = 0,
    stop: int | None = None,
    max_entries: int = 1 << 16,
    num_workers: int | None = None,
) -> Iterator[tuple[Any, float]]:
    """Yield DISTINCT (value, exact_log_prob) from a count-budget index, in approx descending order.

    Builds the budget index and removes repeats by one of two modes (see
    ``ProbabilityDistribution.count_budget_distinct`` for the full contract):

      - ``dedup='canonical'``: stateless per-item predicate (``dist.is_canonical_copy``), O(1)
        memory and random-accessible -- ``start``/``stop`` choose a STRUCTURAL rank range, so the
        distinct enumeration can begin anywhere and partition across workers with no shared state.
      - ``dedup='window'``: a bounded O(max_entries) LRU over the stream (sequential; ``start`` must
        be 0).

    Exact-count families never duplicate, so either mode is a pass-through.
    """
    index = count_budget_index(
        dist, budget_bits, bin_width_bits=bin_width_bits, oversample=oversample, num_workers=num_workers
    )
    n = index.total_count
    stop = n if stop is None else min(int(stop), n)
    start = max(0, int(start))

    if dedup == "window":
        if start != 0:
            raise ValueError("dedup='window' is sequential; start must be 0 (use 'canonical' to seek)")
        from mixle.enumeration.quantization.semiring import bounded_dedup_stream

        raw = (index.get(i) for i in range(start, stop))
        return bounded_dedup_stream(raw, max_entries=max_entries)

    if dedup != "canonical":
        raise ValueError("dedup must be 'canonical' or 'window'")
    quantizer = Quantizer(bin_width_bits=bin_width_bits, oversample=oversample)

    def gen():
        for i in range(start, stop):
            coarse_bin, _off = index.bin_for_index(i)
            value, lp = index.get(i)
            if dist.is_canonical_copy(value, coarse_bin, quantizer):
                yield value, lp

    return gen()


def logit_error_bucket_slack(eps_nats: float, steps: int, quantizer: Quantizer) -> int:
    """Sound bucket-shift bound when step log-probabilities carry up to ``eps_nats`` of error each.

    The license to run enumeration forwards on a **quantized model** (int8/int4 inference, a distilled
    twin): enumeration never needs logits more accurate than the fine-bucket width. If every step
    log-probability is within ``eps_nats`` of the true model's, a ``steps``-long value's accumulated
    fine bucket shifts by at most ``ceil(steps * eps_nats / ln2 * fine_per_bit)`` buckets -- this bound.
    It composes additively with the existing ``steps``-floor structural smear, so every count / rank /
    mass bracket in this package widens by exactly this many buckets and remains sound.

    Practical reading: with the default quantizer (1/8-bit fine buckets) a per-step logit error of
    ~0.01 nats (typical of int8 inference) costs ``ceil(steps * 0.115)`` buckets -- about one bucket per
    nine steps -- while the forwards run 2-4x faster.
    """
    if eps_nats < 0:
        raise ValueError("eps_nats must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return int(math.ceil(float(steps) * (float(eps_nats) / _LOG2) * quantizer.fine_per_bit() - _TOL))


def _two_pow(bits: float) -> int:
    """2**bits as an exact integer ceiling (budget is a count threshold)."""
    b = float(bits)
    if b <= 0:
        return 1
    fl = int(math.floor(b))
    frac = b - fl
    base = 1 << fl
    if frac <= 0:
        return base
    return int(math.ceil(base * (2.0**frac)))
