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
"""

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
    "MPI",
    "HAS_MPI4PY",
    "require",
]


def require(name: str, extra: str):
    """Raise a helpful error for a feature that needs an uninstalled extra."""
    raise ImportError("%s is required for this feature; install it with pip install mixle[%s]" % (name, extra))


# gmpy2: when installed, the structural count-DP routes its large histogram convolutions through GMP's
# FFT-based big-integer multiply (Schoenhage-Strassen), ~100x faster than CPython's Karatsuba on the
# multi-megabyte operands that wide deep-sequence convolutions produce. When missing, gmpy2 is None and
# the convolution falls back to the exact CPython big-int path. Install with: pip install mixle[gmpy2]
try:
    import gmpy2

    HAS_GMPY2 = True
except ImportError:
    gmpy2 = None
    HAS_GMPY2 = False


try:
    import numba

    HAS_NUMBA = True
except ImportError:
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


try:
    import pyspark
    import pyspark.rdd

    HAS_PYSPARK = True
    RDD_TYPES = (pyspark.rdd.RDD,)
except ImportError:
    pyspark = None
    HAS_PYSPARK = False
    RDD_TYPES = ()


try:
    import zarr

    HAS_ZARR = True
except ImportError:
    zarr = None
    HAS_ZARR = False


try:
    import h5py

    HAS_H5PY = True
except ImportError:
    h5py = None
    HAS_H5PY = False


# mpi4py: the "mpi" distributed backend (mixle.utils.parallel.mpi) needs an actual MPI runtime to do
# anything useful, so MPI is None and HAS_MPI4PY is False when missing rather than a no-op shim -- the
# backend raises via require(...) at its entry points instead of silently pretending to coordinate
# ranks. Install with: pip install mixle[mpi]
try:
    from mpi4py import MPI

    HAS_MPI4PY = True
except ImportError:
    MPI = None
    HAS_MPI4PY = False
