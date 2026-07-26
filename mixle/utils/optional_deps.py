"""Optional-dependency shims so the base install works without the heavy extras.

numba: when installed, the real module is re-exported. When missing, a no-op
stand-in is provided whose jit/njit decorators return the function unchanged
and whose prange is range - the jitted code paths then run as pure Python
(correct, but slow). Install the accelerated paths with:

    pip install mixle[numba]

pyspark: `pyspark` is None when missing and RDD_TYPES is an empty tuple, so
`isinstance(data, RDD_TYPES)` is simply False and the estimation helpers fall
through to their local implementations. Install with:

    pip install mixle[spark]

zarr / h5py: the array-store data sources (``mixle.data.sources.array_source``) read lazily from
on-disk zarr and HDF5 volumes without materializing them. Both are ``None`` when missing and
``HAS_ZARR`` / ``HAS_H5PY`` are ``False``, so the connectors raise the standard ``require(...)`` message
instead of an ``ImportError`` on import. numpy-memmap volumes need no extra dependency. Install with:

    pip install mixle[arrays]

pandas: ``mixle.data.sources.pandas_source`` reads a caller-supplied DataFrame by duck-typing (it never
imports pandas itself), but the write-side result-egress methods (``ParameterPosterior.to_dataframe``,
``CalibrationReport.to_dataframe``, ``MarkovChainLatentPosterior.to_dataframe``, and each type's
``to_parquet``) construct a real ``pandas.DataFrame``, so they need the library itself. ``pandas`` is
``None`` when missing and ``HAS_PANDAS`` is ``False``, so those methods raise the standard
``require(...)`` message at call time instead of an ``ImportError`` on import. Install with:

    pip install mixle[pandas]
"""

from importlib import import_module
from types import ModuleType

__all__ = [
    "numba",
    "HAS_NUMBA",
    "pyspark",
    "HAS_PYSPARK",
    "RDD_TYPES",
    "gmpy2",
    "HAS_GMPY2",
    "zarr",
    "HAS_ZARR",
    "h5py",
    "HAS_H5PY",
    "pandas",
    "HAS_PANDAS",
    "MPI",
    "HAS_MPI4PY",
    "require",
]


def require(name: str, extra: str):
    """Raise a helpful error for a feature that needs an uninstalled extra."""
    raise ImportError("%s is required for this feature; install it with pip install mixle[%s]" % (name, extra))


def _import_optional(module_name: str) -> ModuleType | None:
    """Import an optional top-level target without hiding a broken install.

    Only absence of the exact requested module is converted to ``None``.
    ``ImportError``/``ModuleNotFoundError`` for an internal dependency, binary
    extension, or other import-time failure propagates to the caller.
    """
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise


# gmpy2: when installed, the structural count-DP routes its large histogram convolutions through GMP's
# FFT-based big-integer multiply (Schoenhage-Strassen), ~100x faster than CPython's Karatsuba on the
# multi-megabyte operands that wide deep-sequence convolutions produce. When missing, gmpy2 is None and
# the convolution falls back to the exact CPython big-int path. Install with: pip install mixle[gmpy2]
gmpy2 = _import_optional("gmpy2")
HAS_GMPY2 = gmpy2 is not None


numba = _import_optional("numba")
HAS_NUMBA = numba is not None
if numba is None:
    HAS_NUMBA = False

    class _NumbaShim:
        prange = staticmethod(range)

        @staticmethod
        def _decorate(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def deco(f):
                return f

            return deco

        njit = _decorate
        jit = _decorate

    numba = _NumbaShim()


pyspark = _import_optional("pyspark")
if pyspark is not None:
    rdd = _import_optional("pyspark.rdd")
    if rdd is None:
        raise ImportError("installed pyspark package is missing its required pyspark.rdd module")
    HAS_PYSPARK = True
    RDD_TYPES = (rdd.RDD,)
else:
    HAS_PYSPARK = False
    RDD_TYPES = ()


zarr = _import_optional("zarr")
HAS_ZARR = zarr is not None


h5py = _import_optional("h5py")
HAS_H5PY = h5py is not None


pandas = _import_optional("pandas")
HAS_PANDAS = pandas is not None


# mpi4py: the "mpi" distributed backend (mixle.utils.parallel.mpi) needs an actual MPI runtime to do
# anything useful, so MPI is None and HAS_MPI4PY is False when missing rather than a no-op shim -- the
# backend raises via require(...) at its entry points instead of silently pretending to coordinate
# ranks. Install with: pip install mixle[mpi]
_mpi4py = _import_optional("mpi4py")
if _mpi4py is not None:
    MPI = _import_optional("mpi4py.MPI")
    if MPI is None:
        raise ImportError("installed mpi4py package is missing its required mpi4py.MPI module")
    HAS_MPI4PY = True
else:
    MPI = None
    HAS_MPI4PY = False
