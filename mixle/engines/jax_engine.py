"""JAX implementation of the ComputeEngine protocol -- XLA arrays, functional autograd, GPU/TPU.

JAX's ``jax.numpy`` mirrors the NumPy API, so most ops alias straight through (no ``axis``->``dim``
translation as Torch needs). Three JAX-isms are handled here:

* **float64 is opt-in, and it is the caller's global setting, not this module's.** JAX defaults to
  float32 and treats ``jax_enable_x64`` as a process-wide flag that JAX itself only documents as
  reliable when set before any JAX computation runs -- so this module never calls
  ``jax.config.update(...)`` itself; doing that on import would silently reconfigure unrelated JAX
  code sharing the process, including code that already initialized JAX before Mixle was imported.
  Instead the engine reads the ambient ``jax.config.jax_enable_x64`` at construction: when the caller
  enabled it themselves (e.g. ``jax.config.update("jax_enable_x64", True)`` at application start,
  before touching JAX), the engine's default dtype and :attr:`accumulator_dtype` are float64 as
  before; when it is off, both fall back to float32 -- the precision JAX would silently truncate a
  float64 request to anyway -- so the engine never claims a precision the runtime cannot deliver.
  Pass ``dtype=`` explicitly to override either way.
* **device placement is explicit and validated.** ``device`` resolves to a concrete ``jax.Device`` at
  construction time (default ``"cpu"``; also accepts ``"gpu"``, ``"tpu"``, or ``"platform:index"``).
  A platform or index that does not exist in the current JAX runtime raises ``ValueError``
  immediately. Every allocation/conversion method places its result on that resolved device via
  ``jax.device_put``, so a requested device is never just a label JAX's own runtime default silently
  overrides.
* **arrays are immutable.** ``index_add`` uses the functional ``arr.at[idx].add(...)`` update and returns
  the new array (return-value-only, like the Torch engine).

Autograd is *functional* in JAX (``jax.grad`` / ``value_and_grad``), not tensor-tagged, so
:meth:`requires_grad` is always False even though :attr:`supports_autograd` is True. Like every non-host
engine it keeps ``resident_estep=True`` / ``supports_numba=False``: scoring runs on JAX arrays, and the
E-step round-trips through host NumPy unless an engine-resident kernel is registered.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mixle.engines.base import ComputeEngine
from mixle.engines.precision import normalize_numpy_dtype
from mixle.utils.optional_deps import require

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as jsp
    # Deliberately no `jax.config.update("jax_enable_x64", True)` here -- see the module docstring:
    # jax_enable_x64 is a process-wide, caller-owned setting, so JaxEngine reads it instead of forcing
    # it (MXR-080-0147). Do not reintroduce a global config write at import time.
except ImportError:  # pragma: no cover - exercised when optional extra is absent
    jax = None
    jnp = None
    jsp = None


def _resolve_jax_device(device: Any) -> Any:
    """Resolve ``device`` to a concrete, validated ``jax.Device`` (MXR-080-0146).

    ``device`` may be ``None`` (defaults to ``"cpu"``), a platform name (``"cpu"``, ``"gpu"``,
    ``"tpu"``), a ``"platform:index"`` string selecting a specific device of that platform, or an
    already-concrete device object (returned unchanged -- e.g. re-threaded from
    :meth:`JaxEngine.with_precision`). A platform with no devices in the current JAX runtime, or an
    out-of-range index, raises ``ValueError`` immediately instead of silently falling back to
    whatever device JAX would otherwise pick as its own runtime default.
    """
    if device is not None and not isinstance(device, str):
        return device
    spec = device or "cpu"
    platform, sep, index = spec.partition(":")
    try:
        available = jax.devices(platform)
    except RuntimeError as exc:
        present = sorted({d.platform for d in jax.devices()})
        raise ValueError(
            "JaxEngine device %r requests platform %r, which has no devices in this JAX runtime "
            "(platforms present: %s). Install the matching JAX plugin (e.g. jax[cuda]) or request a "
            "platform that is actually present." % (spec, platform, present)
        ) from exc
    idx = 0
    if sep:
        try:
            idx = int(index)
        except ValueError:
            raise ValueError("JaxEngine device %r has a non-integer index %r." % (spec, index)) from None
    if not 0 <= idx < len(available):
        raise ValueError(
            "JaxEngine device %r requests index %d on platform %r, but only %d device(s) are "
            "available there." % (spec, idx, platform, len(available))
        )
    return available[idx]


class JaxEngine(ComputeEngine):
    """JAX array engine: XLA-compiled ops, float64, optional ``jax.jit`` compilation, GPU/TPU via JAX."""

    name = "jax"
    supports_autograd = True

    def __init__(self, device: str | None = None, dtype: Any = None, compile: bool = False) -> None:
        """Construct a JAX engine bound to a concrete, validated device.

        ``device`` defaults to ``"cpu"`` (matching :class:`~mixle.engines.torch_engine.TorchEngine`'s
        default) and accepts a platform name, a ``"platform:index"`` string, or an already-resolved
        ``jax.Device`` (see :func:`_resolve_jax_device`). ``dtype`` defaults to float64 only when the
        ambient ``jax.config.jax_enable_x64`` is already enabled by the caller; otherwise it falls
        back to float32, since JAX would silently truncate an unenabled float64 request anyway.
        """
        if jnp is None:
            require("jax", "jax")
        self.device = _resolve_jax_device(device)
        # jax_enable_x64 is process-wide and caller-owned (see module docstring): float64 is only
        # actually available when the caller enabled it themselves, so an unenabled runtime demotes
        # the engine's default dtype and accumulator to float32 -- the same precision JAX would
        # silently truncate to anyway -- rather than claiming float64 and lying about it. Mirrors
        # TorchEngine's `_no_f64` MPS handling (no float64 hardware there either).
        self._no_f64 = not bool(getattr(jax.config, "jax_enable_x64", False))
        if isinstance(dtype, np.dtype) and not np.issubdtype(dtype, np.floating):
            # A concrete non-floating NumPy dtype (e.g. int32/bool, as surfaced by engine discovery
            # reading a JAX array's own storage dtype -- JAX dtypes are plain np.dtype instances) is
            # not a meaningful floating-point policy value -- treat it like "no override" so engine
            # construction/discovery succeeds instead of raising (this used to make engine_of() raise
            # ValueError for ordinary integer and Boolean JAX arrays, breaking indexing, masks, and
            # categorical payload dispatch -- mirrors TorchEngine's identical fix, MXR-080-0122). A
            # *named* precision that resolves to non-floating (e.g. dtype="int64") is a genuine caller
            # mistake and still raises below, inside normalize_numpy_dtype.
            dtype = None
        # Whether the floating-point policy below is a real, caller-requested opinion (True) or just an
        # implicit default because no floating dtype was given/applicable (False). mixle.engines.engine_of's
        # mixed-engine check (_engines_compatible) reads this so an engine discovered from a non-floating
        # leaf (an integer index or Boolean mask array) never conflicts with a genuinely floating sibling
        # leaf purely because of this filled-in default. Mirrors TorchEngine.dtype_explicit.
        self.dtype_explicit = dtype is not None
        if dtype is not None:
            self.dtype = normalize_numpy_dtype(dtype)
            if self._no_f64 and self.dtype == np.float64:
                self.dtype = np.float32
        else:
            self.dtype = np.float32 if self._no_f64 else np.float64
        self.compile_enabled = bool(compile)

    @property
    def accumulator_dtype(self) -> Any:
        """High-precision dtype for sufficient-statistic reductions (float64, or float32 when the
        ambient ``jax_enable_x64`` is off -- see the module docstring; JAX would silently truncate a
        float64 accumulator request to float32 there anyway, so this avoids claiming a precision the
        runtime cannot actually give)."""
        return np.float32 if self._no_f64 else np.float64

    def with_precision(self, precision: Any) -> JaxEngine:
        """Return a JAX engine with the same placement and a new dtype policy."""
        return JaxEngine(device=self.device, dtype=precision, compile=self.compile_enabled)

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        """Convert ``x`` to a JAX array on this engine's device. Float inputs are force-cast to the
        engine dtype (float64 by default) unless ``dtype`` is given -- matching the Torch engine's
        contract, not NumPy's."""
        if jnp is None:
            require("jax", "jax")
        a = x if isinstance(x, jax.Array) else np.asarray(x)
        if dtype is not None:
            dt = dtype
        elif a.dtype.kind == "f":
            dt = self.dtype
        elif a.dtype.kind == "b":
            dt = jnp.bool_
        else:
            dt = jnp.int64
        return jax.device_put(jnp.asarray(a, dtype=dt), self.device)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        """Allocate a zero array with this engine's dtype, placed on this engine's device."""
        return jax.device_put(jnp.zeros(shape, dtype=dtype or self.dtype), self.device)

    def empty(self, shape: Any, dtype: Any = None) -> Any:
        """Allocate an array on this engine's device (JAX has no uninitialized ``empty``; zeros is
        the safe equivalent)."""
        return jax.device_put(jnp.zeros(shape, dtype=dtype or self.dtype), self.device)

    def arange(self, *args: Any, **kwargs: Any) -> Any:
        """Return ``jnp.arange`` placed on this engine's device; float arguments select the engine
        float dtype."""
        if "dtype" not in kwargs and any(isinstance(v, (float, np.floating)) for v in args):
            kwargs["dtype"] = self.dtype
        return jax.device_put(jnp.arange(*args, **kwargs), self.device)

    def to_numpy(self, x: Any) -> np.ndarray:
        """Move a JAX array back to a host NumPy array."""
        return np.asarray(x)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        """Stack arrays with ``jnp.stack``, placed on this engine's device."""
        return jax.device_put(jnp.stack(tuple(arrays), axis=axis), self.device)

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        """Join arrays with ``jnp.concatenate`` on this engine's device."""
        return jax.device_put(jnp.concatenate(tuple(arrays), axis=axis), self.device)

    def requires_grad(self, x: Any) -> bool:
        """Always False: JAX autograd is functional (``jax.grad``), not tensor-tagged."""
        return False

    def compile(self, fn: Callable) -> Callable:
        """Compile ``fn`` with ``jax.jit`` when enabled."""
        if self.compile_enabled and jax is not None:
            return jax.jit(fn)
        return fn

    # jax.numpy mirrors numpy's signatures, so these alias directly (lambdas keep the class body import-safe
    # when JAX is absent -- the body is never evaluated until an op is actually called).
    log = staticmethod(lambda x: jnp.log(x))
    exp = staticmethod(lambda x: jnp.exp(x))
    sqrt = staticmethod(lambda x: jnp.sqrt(x))
    abs = staticmethod(lambda x: jnp.abs(x))
    where = staticmethod(lambda *args: jnp.where(*args))
    maximum = staticmethod(lambda x, y: jnp.maximum(x, y))
    clip = staticmethod(lambda x, a_min=None, a_max=None: jnp.clip(x, a_min, a_max))
    floor = staticmethod(lambda x: jnp.floor(x))
    isnan = staticmethod(lambda x: jnp.isnan(x))
    isinf = staticmethod(lambda x: jnp.isinf(x))
    sum = staticmethod(lambda x, *args, **kwargs: jnp.sum(x, *args, **kwargs))
    max = staticmethod(lambda x, *args, **kwargs: jnp.max(x, *args, **kwargs))
    dot = staticmethod(lambda x, y: jnp.dot(x, y))
    matmul = staticmethod(lambda x, y: jnp.matmul(x, y))
    cumsum = staticmethod(lambda x, *args, **kwargs: jnp.cumsum(x, *args, **kwargs))
    logsumexp = staticmethod(lambda x, *args, **kwargs: jsp.logsumexp(x, *args, **kwargs))
    bincount = staticmethod(lambda x, *args, **kwargs: jnp.bincount(x, *args, **kwargs))
    unique = staticmethod(lambda x, *args, **kwargs: jnp.unique(x, *args, **kwargs))
    searchsorted = staticmethod(lambda x, y, *args, **kwargs: jnp.searchsorted(x, y, *args, **kwargs))
    gammaln = staticmethod(lambda x: jsp.gammaln(x))
    digamma = staticmethod(lambda x: jsp.digamma(x))
    betaln = staticmethod(lambda x, y: jsp.gammaln(x) + jsp.gammaln(y) - jsp.gammaln(x + y))
    erf = staticmethod(lambda x: jsp.erf(x))
    # optional trig tier (not in REQUIRED_OPS): directional families use these where the engine has them
    cos = staticmethod(lambda x: jnp.cos(x))
    sin = staticmethod(lambda x: jnp.sin(x))
    arctan2 = staticmethod(lambda x, y: jnp.arctan2(x, y))
    i0e = staticmethod(lambda x: jsp.i0e(x))

    def index_add(self, out: Any, index: Any, values: Any) -> Any:
        """Add ``values`` into ``out`` along axis 0 via the functional ``.at[idx].add`` update,
        placed on this engine's device.

        Contract: return-value-only -- JAX arrays are immutable, so this returns a new array; callers
        must use the return value (the same contract the Torch engine documents)."""
        idx = index if isinstance(index, jax.Array) else jnp.asarray(index, dtype=jnp.int64)
        return jax.device_put(out.at[idx].add(values), self.device)
