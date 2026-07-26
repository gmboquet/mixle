"""Unified per-tensor quantization surface with a method picker (roadmap I1).

Unifies TWO existing, independently-landed quantization mechanisms behind one interface:

1. :mod:`mixle.task.quantize` -- int8/int4 per-tensor symmetric quantization
   (:func:`~mixle.task.quantize.quantize_dequantize_array`, the exact core of ``quantize_mlp``) and LNS
   (log-number-system) compute quantization (:class:`~mixle.engines.lns.LogNumberSystem`). Both are
   "already in-tree and load-bearing" (roadmap context); this module wraps them, it does not
   reimplement their arithmetic.
2. :mod:`mixle.models.sorted_profile_quantizer` (roadmap G4) -- head-exact + parametric-tail
   per-tensor storage, honestly scoped to optimizer-states / KV-tails / anomaly-detection, NOT a
   general weight quantizer (see that module's docstring).

:func:`quantize_tensor` is the single per-tensor entry point: explicit ``method=`` values
(``"int8"``, ``"int4"``, ``"lns"``, ``"sorted_profile"``) dispatch directly to the corresponding
underlying primitive; ``method="auto"`` runs a small picker -- reusing this codebase's existing
:class:`~mixle.task.bandit.UCB1` discrete-arm machinery (the D5/ConditionalJIT "small
learned/bandit controller picks an action per context" pattern, replicated here at per-tensor
scale: the "context" is one tensor, the "arms" are the four methods, the "reward" is measured
reconstruction quality at a matched byte budget) -- to choose, PER TENSOR, whichever method gives
the best reconstruction at the requested size budget.

Every :class:`QuantizedTensor` -- explicit or auto-picked -- carries a :class:`QuantizationReceipt`:
the chosen method, its measured bytes/error/compression ratio, and (for auto-pick) the SAME real
numbers for every method that was considered and rejected, so no choice is silently unexplained.

**Matched-size protocol.** All four methods are compared at a shared byte budget
``target_bytes = ceil(n * bits / 8)`` (``bits`` defaults to 8, i.e. the int8 rate). int8/int4 hit
their fixed rate by construction (8 or 4 bits/element); LNS quantizes log-magnitude to the same
integer width (nibble-packed at 4 bits, reusing :func:`mixle.task.quantize._pack_nibbles`) plus a
packed sign bit per element (``mixle.models.unified_quantizer`` does not compress the sign, so LNS
carries a small, honestly-reported ``n/8``-byte overhead on top of the magnitude bits); sorted-profile
has no size KNOB tied to ``bits`` at all -- its rate is set by the tensor's own permutation-index
dtype (:func:`mixle.models.sorted_profile_quantizer._index_dtype`), so at some tensor sizes it will
not fit the budget at all. A method whose ACTUAL measured ``nbytes`` exceeds ``target_bytes`` is
marked ``eligible=False`` in the receipt and is never auto-picked, even if its reconstruction error
would otherwise be the best -- "matched size" is enforced on real measured bytes, not assumed ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np

from mixle.engines.lns import LogNumberSystem
from mixle.models.sorted_profile_quantizer import (
    DEFAULT_GOF_THRESHOLD,
    SortedProfileEncoding,
)
from mixle.models.sorted_profile_quantizer import fit_sorted_profile as _fit_sorted_profile
from mixle.models.sorted_profile_quantizer import reconstruct as _reconstruct_sorted_profile
from mixle.task.bandit import UCB1
from mixle.task.quantize import _QMAX, _pack_nibbles, _unpack_nibbles
from mixle.task.quantize import dequantize_symmetric as _dequantize_symmetric
from mixle.task.quantize import quantize_dequantize_array as _symmetric_quantize

__all__ = [
    "METHODS",
    "QuantizationReceipt",
    "MethodCandidate",
    "QuantizedTensor",
    "QuantizationBudgetError",
    "SymmetricQuantPayload",
    "LNSTensorPayload",
    "quantize_tensor",
    "lns_quantize_array",
    "lns_dequantize_array",
]

METHODS: tuple[str, ...] = ("int8", "int4", "lns", "sorted_profile")

_LNS_EPS = 1e-12

class QuantizationBudgetError(ValueError):
    """No candidate satisfies the caller's literal serialized-byte budget."""

    def __init__(self, target_bytes: int, candidates: dict[str, MethodCandidate]) -> None:
        self.target_bytes = target_bytes
        self.candidates = candidates
        measured = ", ".join(f"{name}={candidate.nbytes}B" for name, candidate in candidates.items())
        super().__init__(f"no quantization method fits the {target_bytes}-byte budget ({measured})")


def _as_source_array(tensor: Any) -> np.ndarray:
    """Return a finite non-empty real array while preserving source dtype and shape."""
    if hasattr(tensor, "detach"):  # torch.Tensor
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    if array.dtype.kind not in {"i", "u", "f"} or array.dtype.kind == "b":
        raise TypeError("quantize_tensor requires a real integer or floating tensor")
    if array.size == 0:
        raise ValueError("quantize_tensor requires a non-empty tensor")
    if not np.all(np.isfinite(array)):
        raise ValueError("quantize_tensor requires finite values")
    return array


def _exact_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


# --- payload wrappers for the two mixle.task.quantize primitives (int8/int4, LNS) -------------------


@dataclass(frozen=True)
class SymmetricQuantPayload:
    """Validated serialized symmetric payload: int8 codes or true two-per-byte packed int4 codes."""

    codes: np.ndarray
    scale: float
    bits: int
    n: int

    def __post_init__(self) -> None:
        bits = _exact_int(self.bits, "bits")
        if bits not in _QMAX:
            raise ValueError(f"bits must be one of {sorted(_QMAX)}")
        n = _exact_int(self.n, "n")
        codes = np.asarray(self.codes)
        expected = n if bits == 8 else (n + 1) // 2
        expected_dtype = np.int8 if bits == 8 else np.uint8
        if codes.shape != (expected,) or codes.dtype != expected_dtype:
            raise ValueError(
                f"int{bits} codes must have shape {(expected,)} and dtype {np.dtype(expected_dtype)}"
            )
        scale = np.float64(self.scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("quantization scale must be positive and finite")
        codes = np.ascontiguousarray(codes).copy()
        codes.setflags(write=False)
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "n", n)

    @property
    def wq(self) -> np.ndarray:
        """Return unpacked integer codes for compatibility with the symmetric primitive."""
        if self.bits == 4:
            return _unpack_nibbles(self.codes, (self.n,))
        return self.codes.copy()

    def nbytes(self) -> int:
        return self.codes.nbytes + np.dtype(np.float64).itemsize

    def reconstruct(self) -> np.ndarray:
        return _dequantize_symmetric(self.wq, self.scale)


@dataclass(frozen=True)
class LNSTensorPayload:
    """The LNS payload: log-magnitude quantized via :class:`mixle.engines.lns.LogNumberSystem`
    (the SAME class ``mixle.task.quantize.lns_classifier`` uses for its integer log-space
    inference), plus a packed sign bit per element (LNS strips sign when it takes ``log|v|``).

    ``codes`` stores the quantized log-magnitude: a raw int8 array at ``bits=8``, or -- reusing
    :func:`mixle.task.quantize._pack_nibbles`, the SAME nibble packer ``QuantizedMLP`` uses for its
    int4 weights -- two-per-byte packed at ``bits=4``, so LNS's on-disk rate genuinely matches
    int4's, not just its accounting.
    """

    codes: np.ndarray  # int8 (bits=8) or nibble-packed uint8 (bits=4)
    sign_bits: np.ndarray  # np.packbits of (value >= 0)
    step: float
    center: float
    bits: int
    n: int

    def __post_init__(self) -> None:
        bits = _exact_int(self.bits, "bits")
        if bits not in _QMAX:
            raise ValueError(f"bits must be one of {sorted(_QMAX)}")
        n = _exact_int(self.n, "n")
        codes = np.asarray(self.codes)
        expected_codes = n if bits == 8 else (n + 1) // 2
        expected_dtype = np.int8 if bits == 8 else np.uint8
        if codes.shape != (expected_codes,) or codes.dtype != expected_dtype:
            raise ValueError(
                f"LNS int{bits} codes must have shape {(expected_codes,)} and dtype {np.dtype(expected_dtype)}"
            )
        sign_bits = np.asarray(self.sign_bits)
        if sign_bits.shape != ((n + 7) // 8,) or sign_bits.dtype != np.uint8:
            raise ValueError("LNS sign_bits must be a canonical packed uint8 vector")
        step = float(self.step)
        center = float(self.center)
        if not np.isfinite(step) or step <= 0.0 or not np.isfinite(center):
            raise ValueError("LNS step must be positive finite and center must be finite")
        codes = np.ascontiguousarray(codes).copy()
        sign_bits = np.ascontiguousarray(sign_bits).copy()
        codes.setflags(write=False)
        sign_bits.setflags(write=False)
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "sign_bits", sign_bits)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "n", n)

    def nbytes(self) -> int:
        return self.codes.nbytes + self.sign_bits.nbytes + 16  # codes + sign + float64 step/center

    def reconstruct(self) -> np.ndarray:
        return lns_dequantize_array(self)


def lns_quantize_array(flat: np.ndarray, bits: int = 8) -> LNSTensorPayload:
    """Quantize a flat float array in the log-magnitude domain via :class:`LogNumberSystem`.

    ``mixle.engines.lns.LogNumberSystem`` was built to quantize log-DENSITIES (already-log-domain,
    already-positive-support values); a general tensor has both sign and a linear-domain magnitude.
    This function is the honest generalization: split ``v = sign(v) * |v|``, quantize
    ``log(|v| + eps)`` with :class:`LogNumberSystem` (the exact same ``quantize``/``dequantize``
    integer machinery ``lns_classifier`` uses -- not reimplemented here), and pack the sign
    separately (1 bit/element via :func:`numpy.packbits`). This is the natural fit for
    multiplicative-scale / heavy-tailed data (LNS's whole reason to exist), and a poor fit for
    already-near-zero-centered, additive-scale data -- exactly where int8's LINEAR quantization
    should (and, per the model-zoo test, does) win instead.
    """
    bits = _exact_int(bits, "bits")
    if bits not in _QMAX:
        raise ValueError(f"bits must be one of {sorted(_QMAX)}, got {bits}")
    flat = np.asarray(flat, dtype=np.float64)
    if flat.ndim != 1 or flat.size == 0 or not np.all(np.isfinite(flat)):
        raise ValueError("lns_quantize_array requires a non-empty finite one-dimensional array")
    n = flat.size
    qmax = _QMAX[bits]
    sign = np.where(flat < 0, -1.0, 1.0)
    log_mag = np.log(np.abs(flat) + _LNS_EPS)
    lo, hi = float(log_mag.min()) if n else 0.0, float(log_mag.max()) if n else 0.0
    center = (hi + lo) / 2.0
    span = (hi - lo) / 2.0
    step = (span / qmax) or 1.0
    lns = LogNumberSystem(step=step)
    k = np.clip(lns.quantize(log_mag - center), -qmax, qmax).astype(np.int8)
    codes = _pack_nibbles(k) if bits == 4 else k
    sign_bits = np.packbits((sign > 0).astype(np.uint8))
    return LNSTensorPayload(codes=codes, sign_bits=sign_bits, step=step, center=center, bits=bits, n=n)


def lns_dequantize_array(payload: LNSTensorPayload) -> np.ndarray:
    """Inverse of :func:`lns_quantize_array`."""
    k = _unpack_nibbles(payload.codes, (payload.n,)) if payload.bits == 4 else payload.codes
    lns = LogNumberSystem(step=payload.step)
    log_mag = lns.dequantize(k) + payload.center
    mag = np.exp(log_mag) - _LNS_EPS
    mag = np.maximum(mag, 0.0)
    sign_bits = np.unpackbits(payload.sign_bits)[: payload.n]
    sign = sign_bits.astype(np.float64) * 2.0 - 1.0
    return sign * mag


# --- receipts -----------------------------------------------------------------------------------


@dataclass
class MethodCandidate:
    """Real, measured numbers for one method considered for one tensor -- the raw material every
    :class:`QuantizationReceipt` (chosen or rejected) is built from."""

    method: str
    nbytes: int
    reconstruction_error: float  # normalized MSE: mean((v - v_hat)**2) / mean(v**2)
    compression_ratio: float
    eligible: bool  # nbytes <= target_bytes
    reward: float  # what the picker actually compared (−inf-ish for ineligible)


@dataclass
class QuantizationReceipt:
    """Explains why ``method`` was used for one tensor: its own measured numbers, plus -- for
    auto-pick -- the same real numbers for every OTHER method that was considered and rejected."""

    method: str
    auto: bool
    nbytes: int
    reconstruction_error: float
    compression_ratio: float
    original_bytes: int
    source_dtype: str
    target_bytes: int
    candidates: dict[str, MethodCandidate] = field(default_factory=dict)
    notes: str = ""

    def rejected(self) -> dict[str, MethodCandidate]:
        """The candidates NOT chosen (empty for an explicit, non-auto dispatch)."""
        return {m: c for m, c in self.candidates.items() if m != self.method}


@dataclass
class QuantizedTensor:
    """Unified result: whichever underlying encoding was produced, with a ``.reconstruct()`` that
    works regardless of which method was actually used, plus the receipt explaining the choice."""

    method: str
    shape: tuple[int, ...]
    source_dtype: str
    payload: SymmetricQuantPayload | LNSTensorPayload | SortedProfileEncoding
    receipt: QuantizationReceipt

    def reconstruct(self) -> np.ndarray:
        if isinstance(self.payload, SortedProfileEncoding):
            flat = _reconstruct_sorted_profile(self.payload).reshape(-1)
        else:
            flat = self.payload.reconstruct()
        return np.asarray(flat, dtype=np.float64).reshape(self.shape)


# --- per-method run + score -----------------------------------------------------------------------


def _reconstruction_error(flat: np.ndarray, flat_hat: np.ndarray) -> float:
    denom = float(np.mean(flat.astype(np.float64) ** 2))
    mse = float(np.mean((flat.astype(np.float64) - flat_hat.astype(np.float64)) ** 2))
    return mse / denom if denom > 0 else mse


def _run_int(flat: np.ndarray, bits: int, clip_percentile: float | None) -> tuple[SymmetricQuantPayload, np.ndarray]:
    wq, scale = _symmetric_quantize(flat, bits=bits, clip_percentile=clip_percentile)
    codes = _pack_nibbles(wq) if bits == 4 else np.asarray(wq, dtype=np.int8)
    payload = SymmetricQuantPayload(codes=codes, scale=np.float64(scale), bits=bits, n=wq.size)
    return payload, payload.reconstruct()


def _run_lns(flat: np.ndarray, bits: int) -> tuple[LNSTensorPayload, np.ndarray]:
    payload = lns_quantize_array(flat, bits=bits)
    return payload, payload.reconstruct()


def _run_sorted_profile(
    tensor: Any, top_k: int, tail_family: Any, gof_threshold: float
) -> tuple[SortedProfileEncoding, np.ndarray]:
    encoding = _fit_sorted_profile(tensor, top_k=top_k, tail_family=tail_family, gof_threshold=gof_threshold)
    return encoding, _reconstruct_sorted_profile(encoding).reshape(-1)


def _candidate(
    method: str, payload: Any, flat: np.ndarray, flat_hat: np.ndarray, target_bytes: int, original_bytes: int
) -> tuple[Any, MethodCandidate]:
    nbytes = payload.nbytes()
    err = _reconstruction_error(flat, flat_hat)
    if not np.isfinite(err) or err < 0.0:
        raise ValueError(f"{method} produced an invalid reconstruction error")
    eligible = nbytes <= target_bytes
    reward = -err if eligible else float("-inf")
    ratio = original_bytes / nbytes if nbytes > 0 else float("inf")
    return payload, MethodCandidate(
        method=method,
        nbytes=nbytes,
        reconstruction_error=err,
        compression_ratio=ratio,
        eligible=eligible,
        reward=reward,
    )


def _run_all_methods(
    tensor: Any,
    flat: np.ndarray,
    bits: int,
    target_bytes: int,
    original_bytes: int,
    top_k: int,
    tail_family: Any,
    gof_threshold: float,
    clip_percentile: float | None,
) -> dict[str, tuple[Any, MethodCandidate]]:
    out: dict[str, tuple[Any, MethodCandidate]] = {}
    p8, hat8 = _run_int(flat, 8, clip_percentile)
    out["int8"] = _candidate("int8", p8, flat, hat8, target_bytes, original_bytes)
    p4, hat4 = _run_int(flat, 4, clip_percentile)
    out["int4"] = _candidate("int4", p4, flat, hat4, target_bytes, original_bytes)
    lns_bits = 8 if bits >= 8 else 4
    plns, hat_lns = _run_lns(flat, lns_bits)
    out["lns"] = _candidate("lns", plns, flat, hat_lns, target_bytes, original_bytes)
    psp, hat_sp = _run_sorted_profile(tensor, top_k, tail_family, gof_threshold)
    out["sorted_profile"] = _candidate("sorted_profile", psp, flat, hat_sp, target_bytes, original_bytes)
    return out


# --- the public entry point -------------------------------------------------------------------------


def quantize_tensor(
    tensor: Any,
    method: str = "auto",
    *,
    bits: int = 8,
    target_compression: int | None = None,
    top_k: int = 0,
    tail_family: Any = None,
    gof_threshold: float = DEFAULT_GOF_THRESHOLD,
    clip_percentile: float | None = None,
    seed: int | None = None,
) -> QuantizedTensor:
    """The single per-tensor quantization entry point (roadmap I1).

    Args:
        tensor: numpy array or torch tensor of any shape.
        method: ``"auto"`` (default, runs the picker) or one of :data:`METHODS`
            (``"int8"``, ``"int4"``, ``"lns"``, ``"sorted_profile"``) to dispatch directly to the
            corresponding underlying primitive.
        bits: exact target bits/element for the hard matched-size budget
            (``target_bytes = ceil(n*bits/8)``). Explicit int8/int4 methods retain their named code
            widths; LNS uses int4 below 8 target bits and int8 at or above 8. ``target_compression``
            (if given) derives an exact target rate from the actual source dtype width, and is rejected
            when that division is not integral.
        top_k, tail_family, gof_threshold: forwarded to
            :func:`mixle.models.sorted_profile_quantizer.fit_sorted_profile`.
        clip_percentile: forwarded to :func:`mixle.task.quantize.quantize_dequantize_array` for the
            int8/int4 methods.
        seed: seed for the auto-pick :class:`~mixle.task.bandit.UCB1` picker (deterministic either way,
            kept for interface symmetry with the rest of mixle's bandit call sites).

    Returns:
        QuantizedTensor
    """
    source = _as_source_array(tensor)
    source_bits = source.dtype.itemsize * 8
    bits = _exact_int(bits, "bits")
    if target_compression is not None:
        target_compression = _exact_int(target_compression, "target_compression")
        if source_bits % target_compression:
            raise ValueError(
                f"target_compression={target_compression} does not produce an exact bit rate "
                f"from {source_bits}-bit source values"
            )
        bits = source_bits // target_compression
        if bits < 1:
            raise ValueError("target_compression produces a sub-bit target rate")
    if bits > source_bits:
        raise ValueError(f"bits cannot exceed the source precision ({source_bits})")
    if clip_percentile is not None:
        clip_percentile = _finite_real(
            clip_percentile,
            "clip_percentile",
            minimum=np.nextafter(0.0, 1.0),
            maximum=100.0,
        )
    if seed is not None:
        seed = _exact_int(seed, "seed", minimum=0)
        if seed > np.iinfo(np.uint32).max:
            raise ValueError("seed must fit in an unsigned 32-bit integer")
    flat = source.reshape(-1).astype(np.float64)
    n = flat.size
    shape = tuple(source.shape)
    source_dtype = source.dtype.str
    original_bytes = source.nbytes
    target_bytes = int(np.ceil(n * bits / 8))

    if not isinstance(method, str) or (method != "auto" and method not in METHODS):
        raise ValueError(f"method must be 'auto' or one of {METHODS}, got {method!r}")

    if method == "int8" or method == "int4":
        req_bits = 8 if method == "int8" else 4
        payload, flat_hat = _run_int(flat, req_bits, clip_percentile)
        payload, cand = _candidate(method, payload, flat, flat_hat, target_bytes, original_bytes)
    elif method == "lns":
        payload, flat_hat = _run_lns(flat, 8 if bits >= 8 else 4)
        payload, cand = _candidate(method, payload, flat, flat_hat, target_bytes, original_bytes)
    elif method == "sorted_profile":
        payload, flat_hat = _run_sorted_profile(tensor, top_k, tail_family, gof_threshold)
        payload, cand = _candidate(method, payload, flat, flat_hat, target_bytes, original_bytes)
    elif method == "auto":
        results = _run_all_methods(
            tensor, flat, bits, target_bytes, original_bytes, top_k, tail_family, gof_threshold, clip_percentile
        )
        picked_method, payload, candidates = _auto_pick(results, target_bytes=target_bytes, seed=seed)
        receipt = QuantizationReceipt(
            method=picked_method,
            auto=True,
            nbytes=candidates[picked_method].nbytes,
            reconstruction_error=candidates[picked_method].reconstruction_error,
            compression_ratio=candidates[picked_method].compression_ratio,
            original_bytes=original_bytes,
            source_dtype=source_dtype,
            target_bytes=target_bytes,
            candidates=candidates,
            notes=f"UCB1 evaluated all {len(candidates)} methods once (real measured reward); "
            f"picked {picked_method!r} (reward={candidates[picked_method].reward:.6g}) over "
            + ", ".join(
                f"{m}(reward={c.reward:.6g}, eligible={c.eligible})"
                for m, c in candidates.items()
                if m != picked_method
            ),
        )
        return QuantizedTensor(
            method=picked_method,
            shape=shape,
            source_dtype=source_dtype,
            payload=payload,
            receipt=receipt,
        )
    else:  # pragma: no cover - guarded above
        raise ValueError(f"unknown method {method!r}")

    if not cand.eligible:
        raise QuantizationBudgetError(target_bytes, {method: cand})

    receipt = QuantizationReceipt(
        method=method,
        auto=False,
        nbytes=cand.nbytes,
        reconstruction_error=cand.reconstruction_error,
        compression_ratio=cand.compression_ratio,
        original_bytes=original_bytes,
        source_dtype=source_dtype,
        target_bytes=target_bytes,
        candidates={method: cand},
        notes=f"explicit method={method!r}: no picker was run.",
    )
    return QuantizedTensor(
        method=method,
        shape=shape,
        source_dtype=source_dtype,
        payload=payload,
        receipt=receipt,
    )


def _auto_pick(
    results: dict[str, tuple[Any, MethodCandidate]], target_bytes: int, seed: int | None = None
) -> tuple[str, Any, dict[str, MethodCandidate]]:
    """The D5/ConditionalJIT pattern at micro scale: a small bandit controller picks an action
    (quantization method) per context (one tensor).

    Reuses :class:`mixle.task.bandit.UCB1` -- the codebase's existing discrete-arm picker -- rather
    than a bespoke learned-picker framework. Because the "reward" for every arm (method) is fully,
    cheaply computable up front (real measured reconstruction error at the matched byte budget,
    already computed by :func:`_run_all_methods`), over-budget arms are removed before selection
    (and an empty eligible set raises :class:`QuantizationBudgetError`). This is UCB1's cold-start regime taken to
    completion: ``select()`` plays each never-pulled arm once IN ORDER (its documented behavior when
    ``pulls`` is all zero), so calling it once per method sweeps every arm exactly once; ``update``
    then records the real reward. After the sweep, ``ucb1.means`` holds each arm's single observed
    (real) reward, and the eligible arm with the highest mean is the pick. This is the same
    select/update loop the rest of mixle's bandit call sites use; it is simply run to convergence in
    one sweep because, unlike a serving-time bandit, every arm's true reward is already known here.
    """
    order = list(results.keys())
    candidates = {m: c for m, (_p, c) in results.items()}
    eligible_order = [method for method in order if candidates[method].eligible]
    if not eligible_order:
        raise QuantizationBudgetError(target_bytes, candidates)
    order = eligible_order
    n_arms = len(order)
    if n_arms == 1:
        best_method = order[0]
        return best_method, results[best_method][0], candidates
    bandit = UCB1(n_arms=n_arms, seed=seed)
    for _ in range(n_arms):
        arm = bandit.select()
        method = order[arm]
        reward = results[method][1].reward
        bandit.update(arm, reward)
    best_arm = int(np.argmax(bandit.means))
    best_method = order[best_arm]
    payload = results[best_method][0]
    return best_method, payload, candidates
