"""Array compute-engine protocol used by backend-neutral kernels."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class ComputeEngine(ABC):
    """Small array-backend interface for numpy/torch/etc.

    Engines own arithmetic policy: array library, device, dtype, and optional
    compilation.  Distribution and kernel code should depend only on this
    surface when it wants backend-neutral arrays.
    """

    name = "base"
    supports_autograd = False
    dtype = None
    device = "cpu"

    # Canonical array-op surface that backend-neutral kernels duck-type on the engine.  Historically
    # only the handful of allocation ops below were ``@abstractmethod``, while kernels reached for
    # ~25 more (``log``, ``where``, ``logsumexp``, ``gammaln``, ``index_add`` ...) that each engine
    # provided informally -- the root cause of the "present on numpy, missing/divergent on torch or
    # symbolic" bug class.  Declaring the contract here and checking it in ``__init_subclass__`` makes
    # a missing op fail loudly at engine *class definition* rather than deep inside a kernel.
    #
    # These are the elementwise/reduction/special-function ops every numeric engine must supply.
    # Optional capabilities (autograd, device placement, precision adjustment, symbolic comparison
    # masks) are deliberately NOT listed -- they are not part of the universal kernel surface.
    REQUIRED_OPS: tuple[str, ...] = (
        # allocation / conversion
        "asarray",
        "zeros",
        "empty",
        "arange",
        "to_numpy",
        "stack",
        # elementwise math
        "log",
        "exp",
        "sqrt",
        "abs",
        "where",
        "maximum",
        "clip",
        "floor",
        "isnan",
        "isinf",
        # reductions / linear algebra
        "sum",
        "max",
        "dot",
        "matmul",
        "cumsum",
        "logsumexp",
        # indexing / set ops
        "bincount",
        "unique",
        "searchsorted",
        "index_add",
        # special functions
        "gammaln",
        "digamma",
        "betaln",
        "erf",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Enforce the :attr:`REQUIRED_OPS` contract on every concrete engine subclass.

        Abstract subclasses (those still carrying ``@abstractmethod`` placeholders) are exempt, so
        intermediate bases can be declared incrementally.  A concrete engine missing any required op
        raises here -- at import/class-definition time -- instead of failing inside a kernel.
        """
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        missing = tuple(op for op in cls.REQUIRED_OPS if getattr(cls, op, None) is None)
        if missing:
            raise TypeError("%s does not provide required compute ops: %s" % (cls.__name__, ", ".join(missing)))

    # Mathematical constants are part of the engine's arithmetic policy: a numeric engine returns
    # plain floats, but an exact/symbolic engine overrides these so that e.g. ``pi`` stays a symbolic
    # ``pi`` (and ``half`` an exact 1/2) instead of collapsing to a float.  ``pysp.engines.arithmetic`` reads
    # them from the active engine so call sites can be backend-neutral.
    pi = math.pi
    e = math.e
    euler_gamma = 0.5772156649015328606
    inf = math.inf
    zero = 0.0
    one = 1.0
    two = 2.0
    half = 0.5

    def constant(self, value: Any) -> Any:
        """Return ``value`` in this engine's scalar representation (identity for numeric engines)."""
        return value

    # Capability flags for kernel/E-step dispatch -- routed on these instead of the engine name so
    # new backends opt in by setting flags rather than by editing core dispatch ("register, don't
    # branch"). ``supports_numba``: the engine operates on host numpy arrays, so numba-compiled /
    # pure-numpy kernels and the numpy ``seq_log_density`` fallback apply (numpy sets this True).
    # ``resident_estep``: prefer an engine-resident ``seq_update_engine`` over round-tripping the
    # E-step through host numpy (every non-host engine, e.g. torch/jax, wants this -- the default).
    supports_numba = False
    resident_estep = True

    @property
    def accumulator_dtype(self) -> Any:
        """High-precision dtype for sufficient-statistic reductions, or ``None`` when not applicable.

        Numeric engines override this with their float64 accumulator so a reduced-precision fit does
        not drift on large N (see ``NumpyEngine``/``TorchEngine``).  The base returns ``None`` --
        meaning "no separate accumulator dtype", which is the correct policy for engines that never
        drive the numeric accumulate path (e.g. the symbolic engine, where reductions are exact
        expression trees).  ``None`` is also a valid ``dtype=`` argument to ``sum`` (NumPy's default).
        """
        return None

    @property
    def precision(self) -> str:
        """Return the engine dtype policy as a stable user-facing name."""
        from pysp.engines.precision import precision_name

        return precision_name(self.dtype)

    def with_precision(self, precision: Any) -> ComputeEngine:
        """Return an equivalent engine with a different floating-point policy."""
        raise TypeError("%s does not support precision adjustment." % type(self).__name__)

    @abstractmethod
    def asarray(self, x: Any, dtype: Any = None) -> Any:
        """Convert ``x`` into this engine's array/tensor representation."""
        ...

    @abstractmethod
    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        """Allocate a zero-filled array on this engine."""
        ...

    @abstractmethod
    def empty(self, shape: Any, dtype: Any = None) -> Any:
        """Allocate an uninitialized array on this engine."""
        ...

    @abstractmethod
    def arange(self, *args: Any, **kwargs: Any) -> Any:
        """Return an evenly spaced one-dimensional array on this engine."""
        ...

    @abstractmethod
    def to_numpy(self, x: Any) -> Any:
        """Move an engine array back to a NumPy/host representation."""
        ...

    @abstractmethod
    def stack(self, arrays: Any, axis: int = 0) -> Any:
        """Stack a sequence of arrays along ``axis``."""
        ...

    def requires_grad(self, x: Any) -> bool:
        """Return whether ``x`` participates in this engine's autograd graph."""
        return False

    def compile(self, fn: Callable) -> Callable:
        """Optionally compile ``fn``; engines without a compiler return it unchanged."""
        return fn

    def replicate(self, x: Any) -> Any:
        """Return ``x`` in the engine's replicated placement, when applicable."""
        return self.asarray(x)

    def place_component_axis(self, x: Any, axis: int = 0) -> Any:
        """Return ``x`` with a component-axis placement, when the engine supports it."""
        return self.asarray(x)
