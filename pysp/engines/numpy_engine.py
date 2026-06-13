"""NumPy implementation of the ComputeEngine protocol."""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.special

from pysp.engines.base import ComputeEngine
from pysp.engines.precision import normalize_numpy_dtype


class NumpyEngine(ComputeEngine):
    """Host NumPy/SciPy engine used by the default local execution path."""

    name = 'numpy'
    supports_autograd = False

    device = 'cpu'

    def __init__(self, dtype: Any = None) -> None:
        self.dtype = normalize_numpy_dtype(dtype)

    def with_precision(self, precision: Any) -> 'NumpyEngine':
        """Return a NumPy engine using ``precision`` for floating arrays."""
        return NumpyEngine(dtype=precision)

    def asarray(self, x: Any, dtype: Any = None) -> np.ndarray:
        """Convert ``x`` to a NumPy ndarray under the engine dtype policy."""
        arr = np.asarray(x)
        dt = dtype
        if dt is None and self.dtype is not None and arr.dtype.kind == 'f':
            dt = self.dtype
        return np.asarray(x, dtype=dt)

    def zeros(self, shape: Any, dtype: Any = None) -> np.ndarray:
        """Allocate a NumPy zero array using the configured float dtype."""
        return np.zeros(shape, dtype=dtype or self.dtype)

    def empty(self, shape: Any, dtype: Any = None) -> np.ndarray:
        """Allocate an uninitialized NumPy array using the configured float dtype."""
        return np.empty(shape, dtype=dtype or self.dtype)

    def arange(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return ``np.arange`` with float ranges honoring the precision policy."""
        if self.dtype is not None and 'dtype' not in kwargs and _has_float_arg(args):
            kwargs['dtype'] = self.dtype
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
    sum = staticmethod(np.sum)
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

    def index_add(self, out: np.ndarray, index: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Add ``values`` into ``out`` at ``index`` using NumPy's ``add.at`` semantics."""
        np.add.at(out, index, values)
        return out


NUMPY_ENGINE = NumpyEngine()


def _has_float_arg(args: Any) -> bool:
    return any(isinstance(x, (float, np.floating)) for x in args)
