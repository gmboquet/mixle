"""NumPy and SciPy implementation of the ``ComputeEngine`` protocol.

The engine is the default host execution path for local scoring, sufficient
statistics, generated kernels, and optional fused NumPy/Numba mixture kernels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.special

from mixle.engines.base import ComputeEngine
from mixle.engines.precision import normalize_numpy_dtype


class NumpyEngine(ComputeEngine):
    """Host NumPy/SciPy engine used by the default local execution path."""

    name = "numpy"
    supports_autograd = False
    # NumPy is the host array library: numba / pure-numpy kernels apply, and the E-step accumulates
    # on the host (no engine-resident round-trip to avoid).
    supports_numba = True
    resident_estep = False
    # Opt-in: route a fusible composite/mixture of low-cost leaves to the source-generated single-pass
    # fused numba kernel (mixle.stats.compute.fused_codegen). Off on the default engine so the established
    # kernels and their tight host-parity contracts are unchanged; ``FUSED_NUMPY_ENGINE`` flips it on.
    prefer_fused = False

    device = "cpu"

    def __init__(self, dtype: Any = None, prefer_fused: bool = False) -> None:
        self.dtype = normalize_numpy_dtype(dtype)
        self.prefer_fused = prefer_fused

    @property
    def accumulator_dtype(self) -> Any:
        """High-precision dtype for sufficient-statistic reductions (always float64).

        Reductions that aggregate over observations (the sum/index-add forming sufficient
        statistics) accumulate in this dtype even when scoring runs in reduced precision, so a
        float32 fit does not drift on large N (e.g. catastrophic cancellation in a variance).
        """
        return np.float64

    def with_precision(self, precision: Any) -> NumpyEngine:
        """Return a NumPy engine using ``precision`` for floating arrays."""
        return NumpyEngine(dtype=precision, prefer_fused=self.prefer_fused)

    def asarray(self, x: Any, dtype: Any = None) -> np.ndarray:
        """Convert ``x`` to a NumPy ndarray under the engine dtype policy."""
        arr = np.asarray(x)
        if dtype is None and self.dtype is not None and arr.dtype.kind == "f":
            return arr.astype(self.dtype, copy=False)
        return np.asarray(arr, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = None) -> np.ndarray:
        """Allocate a NumPy zero array using the configured float dtype."""
        return np.zeros(shape, dtype=dtype or self.dtype)

    def empty(self, shape: Any, dtype: Any = None) -> np.ndarray:
        """Allocate an uninitialized NumPy array using the configured float dtype."""
        return np.empty(shape, dtype=dtype or self.dtype)

    def arange(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return ``np.arange`` with float ranges honoring the precision policy."""
        if self.dtype is not None and "dtype" not in kwargs and _has_float_arg(args):
            kwargs["dtype"] = self.dtype
        return np.arange(*args, **kwargs)

    def to_numpy(self, x: Any) -> np.ndarray:
        """Return ``x`` as a host NumPy array."""
        return np.asarray(x)

    def stack(self, arrays: Any, axis: int = 0) -> np.ndarray:
        """Stack arrays with ``np.stack`` along the requested axis."""
        return np.stack(arrays, axis=axis)

    log = staticmethod(np.log)
    exp = staticmethod(np.exp)
    sqrt = staticmethod(np.sqrt)
    abs = staticmethod(np.abs)
    where = staticmethod(np.where)
    maximum = staticmethod(np.maximum)
    clip = staticmethod(np.clip)
    floor = staticmethod(np.floor)
    isnan = staticmethod(np.isnan)
    isinf = staticmethod(np.isinf)

    def sum(self, a: Any, *args: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Reduce with ``np.sum``, accumulating floats in ``accumulator_dtype`` by default.

        A float32 ``np.sum`` drifts on large N; when the caller passes no explicit
        ``dtype`` we promote floating inputs to the engine's high-precision
        accumulator (float64) so correctness no longer relies on every caller
        threading ``dtype=accumulator_dtype``.  Integer inputs and explicit-dtype
        calls keep NumPy's default behavior.
        """
        if dtype is None:
            arr = np.asarray(a)
            if arr.dtype.kind == "f":
                dtype = self.accumulator_dtype
        return np.sum(a, *args, dtype=dtype, **kwargs)

    max = staticmethod(np.max)
    dot = staticmethod(np.dot)
    matmul = staticmethod(np.matmul)
    cumsum = staticmethod(np.cumsum)
    logsumexp = staticmethod(scipy.special.logsumexp)
    bincount = staticmethod(np.bincount)
    unique = staticmethod(np.unique)
    searchsorted = staticmethod(np.searchsorted)
    gammaln = staticmethod(scipy.special.gammaln)
    digamma = staticmethod(scipy.special.digamma)
    betaln = staticmethod(scipy.special.betaln)
    erf = staticmethod(scipy.special.erf)
    # optional trig tier (not in REQUIRED_OPS): directional families use these where the engine has them
    cos = staticmethod(np.cos)
    sin = staticmethod(np.sin)
    arctan2 = staticmethod(np.arctan2)
    i0e = staticmethod(scipy.special.i0e)
    erfcx = staticmethod(scipy.special.erfcx)

    def index_add(self, out: np.ndarray, index: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Add ``values`` into ``out`` at ``index`` using NumPy's ``add.at`` semantics."""
        np.add.at(out, index, values)
        return out


NUMPY_ENGINE = NumpyEngine()
FUSED_NUMPY_ENGINE = NumpyEngine(prefer_fused=True)  # opt-in single-pass fused numba kernels


def _has_float_arg(args: Any) -> bool:
    return any(isinstance(x, (float, np.floating)) for x in args)
