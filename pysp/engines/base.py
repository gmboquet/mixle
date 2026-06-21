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

    # Mathematical constants are part of the engine's arithmetic policy: a numeric engine returns
    # plain floats, but an exact/symbolic engine overrides these so that e.g. ``pi`` stays a symbolic
    # ``pi`` (and ``half`` an exact 1/2) instead of collapsing to a float.  ``pysp.arithmetic`` reads
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
