"""Floating-point precision normalization helpers for compute engines.

The functions convert user-facing precision names into NumPy or Torch dtypes and
reject storage-only quantization formats that are not valid compute precisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_ALIASES = {
    "16": "float16",
    "half": "float16",
    "fp16": "float16",
    "float16": "float16",
    "32": "float32",
    "single": "float32",
    "float": "float32",
    "fp32": "float32",
    "float32": "float32",
    "64": "float64",
    "double": "float64",
    "fp64": "float64",
    "float64": "float64",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


# Quantization and sub-byte format names that are not supported compute precisions on the
# NumPy/numba path. Numba cannot compile below float32 and sub-byte formats have no native CPU
# arithmetic, so they require packed storage plus specialized dequantization kernels.
# Reject them with an actionable message instead of a cryptic ``np.dtype`` TypeError.
_UNSUPPORTED_LOW_PRECISION = frozenset(
    {
        "fp8", "float8", "e4m3", "e5m2",
        "fp6", "float6", "e2m3", "e3m2",
        "fp4", "float4", "e2m1", "nf4",
        "mxfp", "mxfp4", "mxfp6", "mxfp8",
        "float2", "float3", "float5", "float7",
        "int4", "int8",
    }
)  # fmt: skip


def precision_name(precision: Any) -> str:
    """Return a readable canonical precision name."""
    if precision is None:
        return "default"
    text = str(precision).replace("torch.", "").replace("numpy.", "").replace("np.", "")
    text = text.replace("<class '", "").replace("'>", "")
    text = text.split(".")[-1].lower()
    return _ALIASES.get(text, text)


def normalize_numpy_dtype(precision: Any) -> np.dtype | None:
    """Normalize a precision specifier to a NumPy floating dtype."""
    if precision is None:
        return None
    name = precision_name(precision)
    if name in _UNSUPPORTED_LOW_PRECISION:
        raise ValueError(
            "precision %r is not a supported compute precision. Sub-byte / FP8 / microscaling / "
            "codebook formats have no native CPU arithmetic (numba cannot compile below float32) and "
            "would require packed storage + GPU dequant kernels. Supported compute precisions: "
            "float32 (reduced) and float64 (default); float16/bfloat16 are Torch/GPU-only." % (precision,)
        )
    if name == "bfloat16":
        raise ValueError("NumPyEngine does not support bfloat16 precision (Torch/GPU-only).")
    try:
        dtype = np.dtype(name)
    except TypeError:
        dtype = np.dtype(precision)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("precision must be a floating-point dtype, got %r." % (precision,))
    return dtype


def normalize_torch_dtype(precision: Any, torch_module: Any) -> Any:
    """Normalize a precision specifier to a Torch floating dtype."""
    if precision is None:
        return None
    if torch_module is not None and isinstance(precision, torch_module.dtype):
        dtype = precision
    else:
        name = precision_name(precision)
        lookup = {
            "float16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "float32": torch_module.float32,
            "float64": torch_module.float64,
        }
        if name not in lookup:
            raise ValueError("Unknown Torch floating precision %r." % (precision,))
        dtype = lookup[name]
    if not dtype.is_floating_point:
        raise ValueError("precision must be a floating-point dtype, got %r." % (precision,))
    return dtype


def engine_with_precision(engine: Any, precision: Any) -> Any:
    """Return ``engine`` adjusted to the requested floating precision."""
    if precision is None:
        return engine
    if engine is None:
        from mixle.engines.numpy_engine import NumpyEngine

        return NumpyEngine(dtype=precision)
    fn = getattr(engine, "with_precision", None)
    if not callable(fn):
        raise TypeError("%s does not support precision adjustment." % type(engine).__name__)
    return fn(precision)


def _is_gpu_engine(engine: Any) -> bool:
    """True only for a Torch engine placed on a non-CPU device (where float32 actually pays off)."""
    if engine is None or getattr(engine, "name", None) != "torch":
        return False
    device = str(getattr(engine, "device", "cpu")).lower()
    return "cpu" not in device


def _numeric_data_sample(data: Any, sample_size: int = 512) -> np.ndarray | None:
    """Flatten an evenly-spaced sample of ``sample_size`` observations to a float array, or None if
    not numeric.

    Handles scalars, sequences/arrays of scalars, and (nested) tuples of those -- enough to read the
    magnitude/dynamic-range of continuous data. Structured/categorical/None observations yield None,
    in which case the caller stays at the safe default precision.

    Strided across the full dataset rather than the leading ``sample_size`` rows: naturally-ordered
    data (sorted, appended-to over time, grouped by source) can concentrate extreme-magnitude values
    later in the sequence, which a plain prefix would never see -- silently recommending float32 for
    data that is not actually well-conditioned for it. ``list(data)`` already materializes every row
    to index into, so striding costs nothing extra over slicing a prefix.
    """
    if data is None:
        return None
    try:
        rows = list(data)
    except TypeError:
        return None
    if not rows:
        return None
    if len(rows) > sample_size:
        step = len(rows) / sample_size
        head = [rows[int(i * step)] for i in range(sample_size)]
    else:
        head = rows
    out: list[float] = []

    def _collect(obj: Any) -> bool:
        if obj is None or isinstance(obj, (str, bytes, bool)):
            return False
        if isinstance(obj, (int, float, np.integer, np.floating)):
            out.append(float(obj))
            return True
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind not in "fiu" or obj.size == 0:
                return False
            out.extend(np.asarray(obj, dtype=np.float64).ravel().tolist())
            return True
        if isinstance(obj, (list, tuple)):
            ok = False
            for el in obj:
                ok = _collect(el) or ok
            return ok
        return False

    any_numeric = False
    for obs in head:
        any_numeric = _collect(obs) or any_numeric
    if not any_numeric or not out:
        return None
    return np.asarray(out, dtype=np.float64)


def auto_precision(data: Any = None, *, engine: Any = None, sample_size: int = 512) -> str:
    """Recommend ``'float32'`` or ``'float64'`` from the data and the target hardware.

    float32 only helps on a GPU Torch engine (on CPU/NumPy it is a no-op or slower), and even there
    only when the data is well conditioned for single precision. Sufficient-statistic *accumulation*
    is already float64-safe (see ``ComputeEngine.accumulator_dtype``), so this guards the remaining
    risk -- the ~7 significant digits of float32 *scoring* -- by inspecting the data's magnitude and
    dynamic range. Returns ``'float64'`` whenever a numeric sample is unavailable or looks risky.

    Args:
        data: A representative sample of the raw observations (or an iterable of them).
        engine: The target compute engine; float32 is only recommended for a GPU Torch engine.
        sample_size: How many leading observations to inspect.

    Returns:
        ``'float32'`` or ``'float64'``.
    """
    if not _is_gpu_engine(engine):
        return "float64"
    sample = _numeric_data_sample(data, sample_size)
    if sample is None or sample.size == 0:
        return "float64"
    amax = float(np.max(np.abs(sample)))
    spread = float(np.std(sample))
    # Large magnitude or wide dynamic range exceeds float32's ~7 significant digits in scoring.
    if amax >= 1.0e4:
        return "float64"
    if spread > 0.0 and amax / spread >= 1.0e3:
        return "float64"
    return "float32"
