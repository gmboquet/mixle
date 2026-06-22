"""Backend-dispatched arithmetic helpers used by pysparkplug classes.

Array *operations* dispatch on their arguments' engine (``engine_of``); scalar *constants* have no
arguments to dispatch on, so they resolve from the **active engine** instead -- ``NUMPY_ENGINE`` by
default (plain floats, the historical behaviour).  Select a different one to make the constants come
from it, e.g. the symbolic engine, so that ``pi`` stays a symbolic ``pi`` and ``half`` an exact 1/2
instead of collapsing to floats during exact arithmetic::

    import pysp.arithmetic as arith
    with arith.using_engine("symbolic"):
        arith.pi      # SymbolicExpression "pi", not 3.141592...

This is the seam for letting users control the arithmetic backend without switching pysparkplug to a
full computer-algebra system.  ``maxint`` / ``maxrandint`` / ``eps`` are implementation limits and
tolerances (not mathematical constants), so they stay engine-independent.
"""

from __future__ import annotations

import builtins
from contextlib import contextmanager
from numbers import Number
from typing import Any

from pysp.engines import NUMPY_ENGINE, SYMBOLIC_ENGINE, ComputeEngine, engine_of

__all__ = [
    "asarray",
    "zeros",
    "empty",
    "arange",
    "to_numpy",
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
    # NOTE: ``sum`` and ``max`` are dispatched and importable by explicit name
    # (``from pysp.arithmetic import sum``) but are deliberately kept OUT of
    # ``__all__``: many leaf/latent modules do ``from pysp.arithmetic import *``
    # and call the *builtins* ``max``/``sum`` inside numba ``nopython`` kernels.
    # Exporting the dispatch wrappers would shadow those builtins in the module
    # namespace and break numba type inference (Untyped global name 'max').
    "dot",
    "matmul",
    "cumsum",
    "logsumexp",
    "stack",
    "bincount",
    "index_add",
    "unique",
    "searchsorted",
    "gammaln",
    "digamma",
    "betaln",
    "erf",
    "pi",  # noqa: F822 -- resolved from the active engine via module __getattr__ (PEP 562)
    "e",  # noqa: F822
    "euler_gamma",  # noqa: F822
    "maxint",
    "maxrandint",
    "one",  # noqa: F822
    "zero",  # noqa: F822
    "two",  # noqa: F822
    "half",  # noqa: F822
    "inf",  # noqa: F822
    "eps",
    "constant",
    "get_default_engine",
    "set_default_engine",
    "using_engine",
]

# Engine-provided mathematical constants resolve from the active engine via __getattr__ below.
_ENGINE_CONSTANTS = frozenset({"pi", "e", "euler_gamma", "one", "zero", "two", "half", "inf"})

_NAMED_ENGINES = {"numpy": NUMPY_ENGINE, "symbolic": SYMBOLIC_ENGINE}

_default_engine: ComputeEngine = NUMPY_ENGINE


def _resolve_engine(engine: ComputeEngine | str) -> ComputeEngine:
    if isinstance(engine, ComputeEngine):
        return engine
    try:
        return _NAMED_ENGINES[engine]
    except KeyError:
        raise ValueError(
            "unknown engine %r; pass a ComputeEngine or one of %s" % (engine, sorted(_NAMED_ENGINES))
        ) from None


def get_default_engine() -> ComputeEngine:
    """Return the engine that supplies scalar constants and the operation-dispatch default."""
    return _default_engine


def set_default_engine(engine: ComputeEngine | str) -> ComputeEngine:
    """Set the active engine (a :class:`ComputeEngine` or ``"numpy"``/``"symbolic"``); return the previous one."""
    global _default_engine
    prev = _default_engine
    _default_engine = _resolve_engine(engine)
    return prev


@contextmanager
def using_engine(engine: ComputeEngine | str):
    """Context manager that makes ``engine`` the active engine for its block, then restores the previous one."""
    prev = set_default_engine(engine)
    try:
        yield _default_engine
    finally:
        set_default_engine(prev)


def constant(value: Any) -> Any:
    """Wrap ``value`` in the active engine's scalar representation (identity for numeric engines)."""
    return _default_engine.constant(value)


def _dispatch(name):
    def fn(*args, **kwargs):
        engine = engine_of(args, default=_default_engine)
        return getattr(engine, name)(*args, **kwargs)

    fn.__name__ = name
    return fn


asarray = _dispatch("asarray")
zeros = _dispatch("zeros")
empty = _dispatch("empty")
arange = _dispatch("arange")
to_numpy = _dispatch("to_numpy")

log = _dispatch("log")
exp = _dispatch("exp")
sqrt = _dispatch("sqrt")
abs = _dispatch("abs")
where = _dispatch("where")
maximum = _dispatch("maximum")
clip = _dispatch("clip")
floor = _dispatch("floor")
isnan = _dispatch("isnan")
isinf = _dispatch("isinf")

sum = _dispatch("sum")


def max(*args, **kwargs):
    """Return Python scalar ``max`` or dispatch array reductions to the active engine."""
    if len(args) > 1 and not kwargs and all(isinstance(arg, Number) for arg in args):
        return builtins.max(*args)
    engine = engine_of(args, default=_default_engine)
    return engine.max(*args, **kwargs)


dot = _dispatch("dot")
matmul = _dispatch("matmul")
cumsum = _dispatch("cumsum")
logsumexp = _dispatch("logsumexp")
stack = _dispatch("stack")

bincount = _dispatch("bincount")
index_add = _dispatch("index_add")
unique = _dispatch("unique")
searchsorted = _dispatch("searchsorted")

gammaln = _dispatch("gammaln")
digamma = _dispatch("digamma")
betaln = _dispatch("betaln")
erf = _dispatch("erf")


# Implementation limits / tolerances -- engine-independent (not mathematical constants).
maxint = 2**31 - 1
maxrandint = 2**31 - 1
eps = 1.0e-8


def __getattr__(name: str) -> Any:
    # Mathematical constants resolve from the active engine (PEP 562); everything else is a real
    # module global handled before this hook fires.
    if name in _ENGINE_CONSTANTS:
        return getattr(_default_engine, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
