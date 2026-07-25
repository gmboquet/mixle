"""Sorted-profile (permutation x monotone) quantizer (roadmap G4): head-exact + parametric-tail per tensor.

Per the R1 copula note (roadmap doc, R1 -> G4, F6, I2, H4), any flattened tensor value vector ``v``
decomposes as ``v = P . s``: a sorted **profile** ``s`` (the empirical quantile function) composed with a
**permutation** ``P`` (the arrangement mapping sorted rank back to original position). H4's
``mixle/experimental/tying_discovery.py`` (``tensor_profile`` / ``profile_distance``, see that module's
docstring) already uses the marginal half of this decomposition -- a fixed-length RESAMPLED profile -- as a
tying-discovery signal, and deliberately throws the permutation away. G4 keeps BOTH halves and turns the
decomposition into an actual per-tensor storage format:

* ``s`` is not stored as a raw sorted array -- it is FIT as a parametric mixle distribution (reusing this
  codebase's real ``mixle.stats``/``mixle.inference.estimate`` machinery, not a hand-rolled curve fit), so
  the non-outlier bulk of the tensor collapses to a handful of distribution parameters instead of one float
  per element;
* ``P`` is stored as literal permutation indices (an integer array) -- per the R1 note's honest
  acknowledgment that "arbitrary permutations are gather ops": there is no closed-form compact encoding of
  an arbitrary permutation short of ``n*log2(n)`` bits, so this module does not pretend otherwise. The sort
  itself is an exact, free (deterministic, non-iterative) operation -- unlike G2's
  :func:`mixle.models.sigma_weighted_projection.sigma_weighted_permutation`, no Sinkhorn/OT solver is
  needed here, because there is nothing to OPTIMIZE: sorting a tensor's own values against itself has one
  unambiguous answer. (G2's Sinkhorn permutation solver is for the DIFFERENT problem of matching one
  tensor's rows to another's under a Sigma-weighted cost -- not reused here.)
* the head (top-``k`` largest-magnitude values) is carved out and stored EXACTLY before any of the above,
  because outliers are exactly where a smooth parametric quantile fit is worst -- this is the "head-exact"
  half of "head-exact + parametric-tail";
* a per-tensor goodness-of-fit RECEIPT (a real, computed Kolmogorov-Smirnov statistic, reusing
  :func:`mixle.utils.evaluation.ks_test` rather than a hand-rolled discrepancy measure) is attached to every
  encoding, and a bad receipt triggers a DENSE FALLBACK rather than silently accepting a bad lossy fit.

Honest scope (do not read this module as a general weight quantizer): the roadmap doc scopes G4 to exactly
three use cases --

1. optimizer states (F6) -- e.g. Adam's second-moment buffer, which is positive, heavy-tailed, and mostly
   smooth (a good match for a Gamma/log-normal-family tail fit); this module builds the mechanism generically
   enough to apply there without F6 itself existing yet;
2. KV-cache tails (E2/I2) -- same story, not built here;
3. anomaly detection (:func:`detect_anomaly`) -- the goodness-of-fit receipt IS the anomaly signal: a tensor
   that suddenly stops matching its own historical value-profile family is itself worth flagging.

Hardware reality (R1): arbitrary permutations are memory-bound gather ops with no FLOP savings, so this
scheme is honestly a STORAGE/regularization/receipt-structure win (real when the permutation indices fit in
fewer bits than the values they replace -- e.g. ``uint16`` indices against ``float32`` values for tensors
under 65536 elements) rather than a speed win, unless restricted to block forms that map to tensor cores
(not attempted here).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np

from mixle.inference import estimate
from mixle.stats import GaussianEstimator
from mixle.utils.evaluation import ks_test

__all__ = [
    "SortedProfileEncoding",
    "AnomalyReport",
    "fit_sorted_profile",
    "reconstruct",
    "detect_anomaly",
]

# Default dense-fallback trigger: the Kolmogorov-Smirnov D-statistic between the fitted tail distribution
# and the actual non-outlier values. D is bounded in [0, 1] and, for a WELL-SPECIFIED family, shrinks
# towards 0 as the sample grows (D ~ O(1/sqrt(n)) for a true fit); for a genuinely mismatched family
# (multi-modal data against a unimodal fit, say) D does NOT shrink with n -- it sits at a roughly constant,
# much larger bias. 0.05 is comfortably above the finite-sample noise floor for tensors of a few thousand
# elements or more (see the module tests for measured D values on both sides of this line) while still
# comfortably below the D observed for genuinely bad fits.
DEFAULT_GOF_THRESHOLD = 0.05

# Default anomaly-detection margin: how much WORSE (in absolute KS-D terms, relative to the reference
# encoding's own receipt) a new tensor's fit against the reference's tail-distribution FAMILY has to get
# before it is flagged. A pure ratio threshold breaks down when the reference D is already tiny (any noise
# doubles it), so this combines a relative ratio with an absolute floor -- see :func:`detect_anomaly`.
DEFAULT_ANOMALY_RATIO = 2.0
DEFAULT_ANOMALY_ABS_MARGIN = 0.02

def _index_dtype(n: int) -> np.dtype:
    """Smallest unsigned integer dtype that can address ``n`` distinct positions -- the honest per-index
    storage cost of a literal permutation array (R1: "arbitrary permutations are gather ops", stored as
    literal indices, ``n*log2(n)`` bits in the worst case; we round up to the nearest whole byte width
    numpy actually offers rather than hand-rolling bit-packing).
    """
    if n <= 0:
        return np.dtype(np.uint8)
    if n <= 2**8:
        return np.dtype(np.uint8)
    if n <= 2**16:
        return np.dtype(np.uint16)
    if n <= 2**32:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


@dataclass(frozen=True)
class SortedProfileEncoding:
    """Storage format for one tensor's sorted-profile (permutation x monotone) encoding.

    Either the ``used_dense_fallback=False`` branch (``top_k_*`` / ``tail_distribution`` /
    ``permutation_indices`` populated, ``dense_values=None``) or the ``used_dense_fallback=True`` branch
    (``dense_values`` populated, the rest ``None``/empty) is populated -- never both -- so
    :func:`reconstruct` can dispatch on the flag alone.

    Attributes:
        shape (tuple[int, ...]): Original tensor shape (reconstruction reshapes back to this).
        top_k_values (np.ndarray | None): Exact values of the top-``k`` largest-magnitude entries
            ("head-exact"). ``None``/empty when ``used_dense_fallback``.
        top_k_indices (np.ndarray | None): Flat indices (into the original tensor, C order) the
            ``top_k_values`` came from.
        tail_distribution (Any | None): A fitted ``mixle.stats`` distribution object (exposing ``.cdf`` and
            ``.quantile``) over the non-outlier ("tail") values -- the parametric replacement for storing
            those values directly.
        permutation_indices (np.ndarray | None): Length-``n_tail`` array of flat original indices, ordered
            so that ``permutation_indices[r]`` is where the ``r``-th smallest non-outlier value belongs.
            This IS the permutation ``P`` in ``v = P . s``.
        goodness_of_fit (float): KS D-statistic between the fitted ``tail_distribution`` and the actual
            non-outlier values (0 = perfect fit; see :data:`DEFAULT_GOF_THRESHOLD`). Set even when
            ``used_dense_fallback`` (it is the receipt that CAUSED the fallback), so the receipt itself is
            never silently thrown away.
        used_dense_fallback (bool): True if the fit was rejected and the tensor is stored densely instead.
        dense_values (np.ndarray | None): The full flattened tensor, only populated when
            ``used_dense_fallback``.
        n_tail (int): Number of non-outlier elements (``= permutation_indices.size`` in the non-fallback
            case; kept explicitly so ``nbytes``/receipts are meaningful in the fallback case too).
    """

    shape: tuple
    top_k_values: np.ndarray | None
    top_k_indices: np.ndarray | None
    tail_distribution: Any | None
    permutation_indices: np.ndarray | None
    goodness_of_fit: float
    used_dense_fallback: bool
    dense_values: np.ndarray | None = None
    n_tail: int = 0
    source_dtype: str = "<f4"
    format_version: int = 1
    _index_dtype: np.dtype = field(default_factory=lambda: np.dtype(np.uint32), repr=False)

    MAGIC = b"MSPQ"
    FORMAT_VERSION = 1
    _HEADER = struct.Struct("<4sBI")

    def __post_init__(self) -> None:
        if self.format_version != self.FORMAT_VERSION:
            raise ValueError(f"unsupported SortedProfileEncoding format_version {self.format_version}")
        if not isinstance(self.shape, (tuple, list)) or any(
            isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or int(size) < 0
            for size in self.shape
        ):
            raise ValueError("shape must contain exact non-negative integer dimensions")
        shape = tuple(int(size) for size in self.shape)
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if size <= 0:
            raise ValueError("SortedProfileEncoding requires a non-empty source shape")
        source_dtype = np.dtype(self.source_dtype)
        if source_dtype.kind not in {"i", "u", "f"} or source_dtype.kind == "b":
            raise TypeError("source_dtype must be a real integer or floating dtype")
        goodness = _bounded_finite(self.goodness_of_fit, "goodness_of_fit", minimum=0.0, maximum=1.0)
        if not isinstance(self.used_dense_fallback, (bool, np.bool_)):
            raise TypeError("used_dense_fallback must be a boolean")
        n_tail = _exact_int(self.n_tail, "n_tail", minimum=0)
        index_dtype = np.dtype(self._index_dtype)
        expected_index_dtype = _index_dtype(size)
        if index_dtype != expected_index_dtype:
            raise ValueError(f"_index_dtype must be {expected_index_dtype} for a source of size {size}")

        if self.used_dense_fallback:
            if self.dense_values is None:
                raise ValueError("dense fallback requires dense_values")
            dense_values = np.asarray(self.dense_values)
            if dense_values.shape != (size,) or dense_values.dtype != source_dtype:
                raise ValueError(
                    f"dense_values must have shape {(size,)} and dtype {source_dtype}; "
                    f"got shape={dense_values.shape}, dtype={dense_values.dtype}"
                )
            if not np.all(np.isfinite(dense_values)):
                raise ValueError("dense_values must be finite")
            if any(
                value is not None
                for value in (
                    self.top_k_values,
                    self.top_k_indices,
                    self.tail_distribution,
                    self.permutation_indices,
                )
            ):
                raise ValueError("dense fallback cannot also contain head/tail fields")
            if n_tail != 0:
                raise ValueError("dense fallback must set n_tail=0")
            object.__setattr__(self, "dense_values", _immutable_copy(dense_values))
        else:
            if self.dense_values is not None or self.tail_distribution is None:
                raise ValueError("parametric encoding requires tail_distribution and no dense_values")
            if n_tail < 2 or n_tail > size:
                raise ValueError("parametric encoding requires 2 <= n_tail <= size")
            top_count = size - n_tail
            top_values = np.asarray(self.top_k_values)
            top_indices = np.asarray(self.top_k_indices)
            permutation = np.asarray(self.permutation_indices)
            if top_values.shape != (top_count,) or top_values.dtype != source_dtype:
                raise ValueError(
                    f"top_k_values must have shape {(top_count,)} and source dtype {source_dtype}"
                )
            if top_indices.shape != (top_count,) or top_indices.dtype != index_dtype:
                raise ValueError(
                    f"top_k_indices must have shape {(top_count,)} and dtype {index_dtype}"
                )
            if permutation.shape != (n_tail,) or permutation.dtype != index_dtype:
                raise ValueError(
                    f"permutation_indices must have shape {(n_tail,)} and dtype {index_dtype}"
                )
            if not np.all(np.isfinite(top_values)):
                raise ValueError("top_k_values must be finite")
            all_indices = np.concatenate((top_indices.astype(np.int64), permutation.astype(np.int64)))
            if (
                np.any(all_indices < 0)
                or np.any(all_indices >= size)
                or np.unique(all_indices).size != size
            ):
                raise ValueError("head and tail indices must form a disjoint permutation of the source")
            if not callable(getattr(self.tail_distribution, "quantile", None)):
                raise TypeError("tail_distribution must provide quantile(probability)")
            if not callable(getattr(self.tail_distribution, "cdf", None)):
                raise TypeError("tail_distribution must provide cdf(value)")
            object.__setattr__(self, "top_k_values", _immutable_copy(top_values))
            object.__setattr__(self, "top_k_indices", _immutable_copy(top_indices))
            object.__setattr__(self, "permutation_indices", _immutable_copy(permutation))

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "source_dtype", source_dtype.str)
        object.__setattr__(self, "goodness_of_fit", goodness)
        object.__setattr__(self, "used_dense_fallback", bool(self.used_dense_fallback))
        object.__setattr__(self, "n_tail", n_tail)
        object.__setattr__(self, "_index_dtype", index_dtype)

    @property
    def size(self) -> int:
        """Total element count of the original tensor."""
        return int(np.prod(self.shape)) if len(self.shape) else 1

    def nbytes(self) -> int:
        """Return the exact size of the versioned :meth:`to_bytes` representation."""
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize this validated encoding without pickle."""
        distribution_json = None
        if self.tail_distribution is not None:
            from mixle.utils.serialization import to_json

            distribution_json = to_json(self.tail_distribution, separators=(",", ":"), sort_keys=True)
        metadata = {
            "shape": self.shape,
            "source_dtype": self.source_dtype,
            "goodness_of_fit": self.goodness_of_fit,
            "used_dense_fallback": self.used_dense_fallback,
            "n_tail": self.n_tail,
            "index_dtype": self._index_dtype.str,
            "tail_distribution": distribution_json,
        }
        metadata_bytes = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(metadata_bytes) > 16 * 1024 * 1024:
            raise ValueError("SortedProfileEncoding metadata exceeds 16 MiB")
        arrays = (
            (self.dense_values,)
            if self.used_dense_fallback
            else (self.top_k_values, self.top_k_indices, self.permutation_indices)
        )
        return (
            self._HEADER.pack(self.MAGIC, self.format_version, len(metadata_bytes))
            + metadata_bytes
            + b"".join(array.tobytes(order="C") for array in arrays)
        )

    @classmethod
    def from_bytes(cls, payload: Any) -> SortedProfileEncoding:
        """Parse an exact-length, versioned encoding payload."""
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("SortedProfileEncoding payload must be bytes-like")
        view = memoryview(payload)
        if len(view) < cls._HEADER.size:
            raise ValueError("SortedProfileEncoding payload is truncated")
        magic, version, metadata_size = cls._HEADER.unpack(view[: cls._HEADER.size])
        if magic != cls.MAGIC or version != cls.FORMAT_VERSION:
            raise ValueError("SortedProfileEncoding payload has invalid magic or version")
        if metadata_size > 16 * 1024 * 1024 or cls._HEADER.size + metadata_size > len(view):
            raise ValueError("SortedProfileEncoding metadata length is invalid")
        metadata_end = cls._HEADER.size + metadata_size
        try:
            metadata = json.loads(bytes(view[cls._HEADER.size : metadata_end]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("SortedProfileEncoding metadata is invalid JSON") from exc
        required = {
            "shape",
            "source_dtype",
            "goodness_of_fit",
            "used_dense_fallback",
            "n_tail",
            "index_dtype",
            "tail_distribution",
        }
        if not isinstance(metadata, dict) or set(metadata) != required:
            raise ValueError("SortedProfileEncoding metadata has an invalid schema")
        shape = tuple(metadata["shape"])
        source_dtype = np.dtype(metadata["source_dtype"])
        index_dtype = np.dtype(metadata["index_dtype"])
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        n_tail = metadata["n_tail"]
        offset = metadata_end
        if metadata["used_dense_fallback"]:
            value_bytes = size * source_dtype.itemsize
            if len(view) != offset + value_bytes:
                raise ValueError("dense SortedProfileEncoding payload has an invalid length")
            dense = np.frombuffer(view[offset:], dtype=source_dtype).copy()
            return cls(
                shape=shape,
                top_k_values=None,
                top_k_indices=None,
                tail_distribution=None,
                permutation_indices=None,
                goodness_of_fit=metadata["goodness_of_fit"],
                used_dense_fallback=True,
                dense_values=dense,
                n_tail=0,
                source_dtype=source_dtype.str,
                format_version=version,
                _index_dtype=index_dtype,
            )

        n_tail = _exact_int(n_tail, "n_tail", minimum=2)
        top_count = size - n_tail
        top_bytes = top_count * source_dtype.itemsize
        top_index_bytes = top_count * index_dtype.itemsize
        permutation_bytes = n_tail * index_dtype.itemsize
        expected = offset + top_bytes + top_index_bytes + permutation_bytes
        if len(view) != expected:
            raise ValueError("parametric SortedProfileEncoding payload has an invalid length")
        top_values = np.frombuffer(view[offset : offset + top_bytes], dtype=source_dtype).copy()
        offset += top_bytes
        top_indices = np.frombuffer(view[offset : offset + top_index_bytes], dtype=index_dtype).copy()
        offset += top_index_bytes
        permutation = np.frombuffer(view[offset:], dtype=index_dtype).copy()
        distribution_json = metadata["tail_distribution"]
        if not isinstance(distribution_json, str):
            raise ValueError("parametric payload requires a serialized tail_distribution")
        from mixle.utils.serialization import from_json

        distribution = from_json(distribution_json)
        return cls(
            shape=shape,
            top_k_values=top_values,
            top_k_indices=top_indices,
            tail_distribution=distribution,
            permutation_indices=permutation,
            goodness_of_fit=metadata["goodness_of_fit"],
            used_dense_fallback=False,
            dense_values=None,
            n_tail=n_tail,
            source_dtype=source_dtype.str,
            format_version=version,
            _index_dtype=index_dtype,
        )


def _immutable_copy(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value).copy()
    result.setflags(write=False)
    return result


def _exact_int(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {bound}")
    return result


def _bounded_finite(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    below = result <= minimum if strict_minimum else result < minimum
    if below or (maximum is not None and result > maximum):
        interval = f"({minimum}, {maximum}]" if strict_minimum else f"[{minimum}, {maximum}]"
        raise ValueError(f"{name} must be in {interval}")
    return result


def _as_source_array(tensor: Any, where: str) -> np.ndarray:
    """Return a finite real numpy view while preserving the source dtype and shape."""
    if hasattr(tensor, "detach"):  # torch.Tensor
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    if array.dtype.kind not in {"i", "u", "f"} or array.dtype.kind == "b":
        raise TypeError(f"{where} requires a real integer or floating tensor")
    if array.size == 0:
        raise ValueError(f"{where} requires a non-empty tensor")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{where} requires finite tensor values")
    return array


def _as_flat_numpy(tensor: Any, where: str) -> np.ndarray:
    return _as_source_array(tensor, where).reshape(-1).astype(np.float64)


def fit_sorted_profile(
    tensor: Any,
    top_k: int = 0,
    tail_family: Any = None,
    gof_threshold: float = DEFAULT_GOF_THRESHOLD,
) -> SortedProfileEncoding:
    """Encode ``tensor`` as head-exact outliers + a fitted parametric tail distribution + permutation.

    Args:
        tensor: A torch tensor or numpy array of any shape.
        top_k (int): Number of largest-magnitude entries to carve out and store EXACTLY ("head-exact"),
            before any fitting happens -- outliers are exactly where a smooth parametric quantile fit is
            worst, so they are never asked to survive the parametric tail model. 0 disables head-exact
            storage entirely (the whole tensor goes through the tail fit).
        tail_family: A ``mixle.stats`` ``ParameterEstimator`` instance (e.g. ``GaussianEstimator()``,
            ``GammaEstimator()``) used to fit the non-outlier values via ``mixle.inference.estimate``.
            Defaults to ``GaussianEstimator()``. Pick a family whose support matches the tensor's actual
            values -- e.g. ``GammaEstimator()`` for a strictly-positive optimizer second-moment buffer,
            per F6's honest scope note (see module docstring); a mismatched family is not silently
            accepted -- it is caught by the goodness-of-fit receipt below and triggers the dense fallback.
        gof_threshold (float): Maximum acceptable KS D-statistic (see :data:`DEFAULT_GOF_THRESHOLD`) before
            falling back to dense storage.

    Returns:
        SortedProfileEncoding: either a populated head/tail/permutation encoding
        (``used_dense_fallback=False``) or a dense fallback (``used_dense_fallback=True``), always carrying
        the real, computed ``goodness_of_fit`` receipt either way.
    """
    if tail_family is None:
        tail_family = GaussianEstimator()

    source = _as_source_array(tensor, "fit_sorted_profile")
    source_flat = source.reshape(-1)
    flat = source_flat.astype(np.float64)
    n = flat.size
    shape = tuple(source.shape)
    top_k = _exact_int(top_k, "top_k", minimum=0, maximum=n)
    gof_threshold = _bounded_finite(
        gof_threshold,
        "gof_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    idx_dtype = _index_dtype(n)

    if top_k > 0:
        abs_vals = np.abs(flat)
        top_k_indices = np.argpartition(-abs_vals, top_k - 1)[:top_k]
        top_k_indices = top_k_indices[np.argsort(-abs_vals[top_k_indices])]
    else:
        top_k_indices = np.array([], dtype=np.int64)

    outlier_mask = np.zeros(n, dtype=bool)
    outlier_mask[top_k_indices] = True
    tail_original_indices = np.nonzero(~outlier_mask)[0]
    tail_values = flat[tail_original_indices]
    n_tail = tail_values.size

    if n_tail < 2:
        # Nothing left to fit a distribution to -- dense fallback is the only honest option.
        encoding = SortedProfileEncoding(
            shape=shape,
            top_k_values=None,
            top_k_indices=None,
            tail_distribution=None,
            permutation_indices=None,
            goodness_of_fit=1.0,
            used_dense_fallback=True,
            dense_values=source_flat.copy(),
            n_tail=0,
            source_dtype=source.dtype.str,
            _index_dtype=idx_dtype,
        )
        encoding.to_bytes()
        return encoding

    tail_distribution = estimate(list(tail_values), tail_family)
    d_stat, _p_value = ks_test(tail_values, tail_distribution)
    d_stat = _bounded_finite(d_stat, "computed goodness_of_fit", minimum=0.0, maximum=1.0)

    order = np.argsort(tail_values)  # ascending: sorted_tail[r] = tail_values[order[r]]
    permutation_indices = tail_original_indices[order].astype(idx_dtype)

    used_dense_fallback = d_stat > gof_threshold
    if used_dense_fallback:
        encoding = SortedProfileEncoding(
            shape=shape,
            top_k_values=None,
            top_k_indices=None,
            tail_distribution=None,
            permutation_indices=None,
            goodness_of_fit=d_stat,
            used_dense_fallback=True,
            dense_values=source_flat.copy(),
            n_tail=0,
            source_dtype=source.dtype.str,
            _index_dtype=idx_dtype,
        )
        encoding.to_bytes()
        return encoding

    encoding = SortedProfileEncoding(
        shape=shape,
        top_k_values=source_flat[top_k_indices].copy() if top_k > 0 else np.array([], dtype=source.dtype),
        top_k_indices=top_k_indices.astype(idx_dtype) if top_k > 0 else np.array([], dtype=idx_dtype),
        tail_distribution=tail_distribution,
        permutation_indices=permutation_indices,
        goodness_of_fit=d_stat,
        used_dense_fallback=False,
        dense_values=None,
        n_tail=n_tail,
        source_dtype=source.dtype.str,
        _index_dtype=idx_dtype,
    )
    encoding.to_bytes()
    return encoding


def reconstruct(encoding: SortedProfileEncoding) -> np.ndarray:
    """Invert a :class:`SortedProfileEncoding` back to an (approximate, or dense-exact) tensor.

    The head (top-k outliers) is EXACT in both branches (either stored verbatim, or -- in the dense
    fallback case -- simply part of the densely-stored tensor). The tail is exact under dense fallback and
    approximate (reconstructed from the fitted parametric quantile function) otherwise.

    Returns:
        np.ndarray: array reshaped to ``encoding.shape`` using the recorded source dtype.
    """
    if not isinstance(encoding, SortedProfileEncoding):
        raise TypeError("encoding must be a SortedProfileEncoding")
    if encoding.used_dense_fallback:
        return encoding.dense_values.copy().reshape(encoding.shape)

    n = encoding.size
    out = np.zeros(n, dtype=np.float64)

    n_tail = encoding.n_tail
    # Reconstruct the sorted tail profile from the fitted quantile function at the midpoint of each rank's
    # probability mass -- the standard "plotting position" for turning n ranks into n quantile queries.
    ranks = (np.arange(n_tail, dtype=np.float64) + 0.5) / n_tail
    sorted_tail_hat = np.asarray([encoding.tail_distribution.quantile(float(q)) for q in ranks], dtype=np.float64)
    if sorted_tail_hat.shape != (n_tail,) or not np.all(np.isfinite(sorted_tail_hat)):
        raise ValueError("tail_distribution returned invalid reconstruction quantiles")
    out[encoding.permutation_indices.astype(np.int64)] = sorted_tail_hat

    if encoding.top_k_values is not None and encoding.top_k_values.size > 0:
        out[encoding.top_k_indices.astype(np.int64)] = encoding.top_k_values

    return out.astype(np.dtype(encoding.source_dtype)).reshape(encoding.shape)


@dataclass(frozen=True)
class AnomalyReport:
    """Result of scoring a new tensor against a reference encoding's tail-distribution family.

    Attributes:
        ks_statistic (float): KS D-statistic of the new tensor's non-outlier values against the
            REFERENCE encoding's fitted ``tail_distribution`` (the family is held fixed; only the data
            changes -- this is a re-SCORING, not a re-fit).
        reference_goodness_of_fit (float): The reference encoding's own receipt, for context.
        is_anomaly (bool): Whether ``ks_statistic`` has degraded significantly relative to
            ``reference_goodness_of_fit`` (see :func:`detect_anomaly` for the exact rule).
    """

    ks_statistic: float
    reference_goodness_of_fit: float
    is_anomaly: bool


def detect_anomaly(
    tensor: Any,
    reference_encoding: SortedProfileEncoding,
    ratio_threshold: float = DEFAULT_ANOMALY_RATIO,
    abs_margin: float = DEFAULT_ANOMALY_ABS_MARGIN,
) -> AnomalyReport:
    """Anomaly-detection use of the goodness-of-fit receipt (roadmap G4, use case 3).

    A tensor that historically fit ``reference_encoding.tail_distribution``'s family well and suddenly stops
    fitting it -- a burst of extreme values, a distribution shift -- is itself an anomaly signal, independent
    of whatever downstream task the tensor feeds. This function re-SCORES ``tensor`` against the reference's
    ALREADY-FITTED family (it does not fit a new distribution to ``tensor``), then compares the resulting
    KS D-statistic to the reference's own receipt.

    The new tensor's outliers are excluded using the reference encoding's own top-k COUNT (not its specific
    indices, which belong to a different tensor) so the comparison is apples-to-apples with how the reference
    receipt itself was computed.

    Flagging rule: ``is_anomaly`` fires when the new D-statistic exceeds
    ``max(ratio_threshold * reference_goodness_of_fit, reference_goodness_of_fit + abs_margin)`` -- a ratio
    threshold alone breaks down when the reference D is already tiny (sampling noise alone can double it), so
    it is combined with an absolute floor. Both directions are meaningful test cases: a similarly-distributed
    new draw should score close to (or even below) the reference's own receipt; a genuinely shifted or
    outlier-contaminated tensor should score well past the combined threshold.

    Returns:
        AnomalyReport
    """
    if not isinstance(reference_encoding, SortedProfileEncoding):
        raise TypeError("reference_encoding must be a SortedProfileEncoding")
    ratio_threshold = _bounded_finite(
        ratio_threshold,
        "ratio_threshold",
        minimum=0.0,
        strict_minimum=True,
    )
    abs_margin = _bounded_finite(abs_margin, "abs_margin", minimum=0.0)
    if reference_encoding.used_dense_fallback:
        raise ValueError(
            "detect_anomaly requires a reference_encoding with a fitted tail_distribution "
            "(reference_encoding.used_dense_fallback was True, so there is no fitted family to score against)"
        )

    flat = _as_flat_numpy(tensor, "detect_anomaly")
    n = flat.size

    top_k_ref = reference_encoding.top_k_values.size if reference_encoding.top_k_values is not None else 0
    top_k = int(max(0, min(top_k_ref, n - 2)))  # keep >= 2 non-outlier values to score against

    if top_k > 0:
        abs_vals = np.abs(flat)
        outlier_indices = np.argpartition(-abs_vals, top_k - 1)[:top_k]
        outlier_mask = np.zeros(n, dtype=bool)
        outlier_mask[outlier_indices] = True
        tail_values = flat[~outlier_mask]
    else:
        tail_values = flat

    d_stat, _p_value = ks_test(tail_values, reference_encoding.tail_distribution)
    d_stat = _bounded_finite(d_stat, "computed anomaly ks_statistic", minimum=0.0, maximum=1.0)
    ref_d = reference_encoding.goodness_of_fit
    threshold = max(ratio_threshold * ref_d, ref_d + abs_margin)
    is_anomaly = d_stat > threshold

    return AnomalyReport(
        ks_statistic=d_stat,
        reference_goodness_of_fit=ref_d,
        is_anomaly=is_anomaly,
    )
