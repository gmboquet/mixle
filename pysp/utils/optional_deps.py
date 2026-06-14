"""Optional-dependency shims so the base install works without the heavy extras.

numba: when installed, the real module is re-exported. When missing, a no-op
stand-in is provided whose jit/njit decorators return the function unchanged
and whose prange is range - the jitted code paths then run as pure Python
(correct, but slow). Install the accelerated paths with:

    pip install pysparkplug[numba]

pyspark: `pyspark` is None when missing and RDD_TYPES is an empty tuple, so
`isinstance(data, RDD_TYPES)` is simply False and the estimation helpers fall
through to their local implementations. Install with:

    pip install pysparkplug[spark]
"""

__all__ = ["numba", "HAS_NUMBA", "pyspark", "HAS_PYSPARK", "RDD_TYPES", "require"]


def require(name: str, extra: str):
    """Raise a helpful error for a feature that needs an uninstalled extra."""
    raise ImportError("%s is required for this feature; install it with pip install pysparkplug[%s]" % (name, extra))


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
