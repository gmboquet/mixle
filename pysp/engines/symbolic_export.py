"""Lowering of :class:`SymbolicExpression` trees to sympy and sagemath.

The native :mod:`pysp.engines.symbolic_engine` produces a dependency-free
expression tree (e.g. from ``backend_seq_log_density`` under
``SYMBOLIC_ENGINE``).  This module converts those trees into sympy or sage
expressions so they can be simplified, rendered as LaTeX, differentiated
symbolically, or fed into a code generator.

Both backends are imported lazily so importing :mod:`pysp.engines` never
requires sympy or sage.  Array inputs (NumPy object arrays of expression
nodes) are mapped elementwise into a NumPy object array of backend
expressions, preserving shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from pysp.engines.symbolic_engine import SymbolicExpression

#: Ops that have no symbolic meaning (data-dependent index/tabulation kernels).
_NON_SYMBOLIC_OPS = frozenset({"bincount", "unique", "searchsorted", "index_add"})


def _import_sympy():
    try:
        import sympy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without sympy
        raise ImportError(
            "sympy is required for to_sympy/to_latex; install it with "
            "`pip install sympy` (or `pip install pysparkplug[sympy]`)."
        ) from exc
    return sympy


def _import_sage():
    # Full SageMath exposes the unified ``sage.all``; the pip-installable modular distribution
    # (``pip install passagemath-symbolics``) exposes the same symbolic surface under
    # ``sage.all__sagemath_symbolics`` instead, so accept either.
    import importlib

    for module_name in ("sage.all", "sage.all__sagemath_symbolics"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    raise ImportError(  # pragma: no cover - exercised only without any sage
        "sagemath is required for to_sage; install full SageMath, or `pip install "
        "passagemath-symbolics` for the symbolic subset."
    )


def _is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.dtype == object


def _map_array(x: np.ndarray, convert: Callable[[Any], Any]) -> np.ndarray:
    """Apply ``convert`` to every element, returning an object array of exprs."""
    out = np.empty(x.shape, dtype=object)
    for idx in np.ndindex(x.shape):
        out[idx] = convert(x[idx])
    return out


# ---------------------------------------------------------------------------
# sympy backend
# ---------------------------------------------------------------------------


def _sympy_ops(sympy) -> dict[str, Callable[..., Any]]:
    return {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "pow": lambda a, b: a**b,
        "neg": lambda a: -a,
        "lt": lambda a, b: sympy.Lt(a, b),
        "le": lambda a, b: sympy.Le(a, b),
        "gt": lambda a, b: sympy.Gt(a, b),
        "ge": lambda a, b: sympy.Ge(a, b),
        "eq": lambda a, b: sympy.Eq(a, b),
        "ne": lambda a, b: sympy.Ne(a, b),
        "and": lambda a, b: sympy.And(a, b),
        "or": lambda a, b: sympy.Or(a, b),
        "invert": lambda a: sympy.Not(a),
        "log": sympy.log,
        "exp": sympy.exp,
        "sqrt": sympy.sqrt,
        "abs": sympy.Abs,
        "floor": sympy.floor,
        "gammaln": sympy.loggamma,
        "digamma": sympy.digamma,
        "erf": sympy.erf,
        "betaln": lambda a, b: sympy.loggamma(a) + sympy.loggamma(b) - sympy.loggamma(a + b),
        "isnan": lambda a: sympy.Function("isnan")(a),
        "isinf": lambda a: sympy.Function("isinf")(a),
        "max": lambda *xs: sympy.Max(*xs),
        "where": lambda cond, a, b: sympy.Piecewise((a, cond), (b, True)),
        "clip": lambda x, a_min, a_max: _sympy_clip(sympy, x, a_min, a_max),
        # nullary named math constants -- stay exact (pi, not 3.14159...)
        "pi": lambda: sympy.pi,
        "e": lambda: sympy.E,
        "euler_gamma": lambda: sympy.EulerGamma,
        "inf": lambda: sympy.oo,
    }


def _sympy_clip(sympy, x, a_min, a_max):
    if a_min is not None:
        x = sympy.Max(x, a_min)
    if a_max is not None:
        x = sympy.Min(x, a_max)
    return x


def to_sympy(expr: Any) -> Any:
    """Convert a :class:`SymbolicExpression` (or object array of them) to sympy.

    Scalar nodes become a sympy expression; object arrays map elementwise into
    a NumPy object array of sympy expressions (shape preserved).  Genuinely
    non-symbolic kernels (``bincount``/``unique``/``searchsorted``/
    ``index_add``) raise :class:`NotImplementedError`.  Raises
    :class:`ImportError` if sympy is unavailable.
    """
    sympy = _import_sympy()
    ops = _sympy_ops(sympy)

    def convert(node: Any) -> Any:
        if isinstance(node, SymbolicExpression):
            if node.op == "symbol":
                return sympy.Symbol(node.args[0])
            if node.op == "const":
                return _sympy_const(sympy, node.args[0])
            if node.op in _NON_SYMBOLIC_OPS:
                raise NotImplementedError("symbolic op %r has no sympy representation" % node.op)
            handler = ops.get(node.op)
            if handler is None:
                raise NotImplementedError("symbolic op %r is not supported by to_sympy" % node.op)
            return handler(*[convert(arg) for arg in node.args])
        if _is_array(node):
            return _map_array(node, convert)
        # raw python/numpy scalar embedded in the tree
        return _sympy_const(sympy, node)

    if _is_array(expr):
        return _map_array(expr, convert)
    return convert(expr)


def _sympy_const(sympy, value: Any) -> Any:
    if isinstance(value, bool):
        return sympy.true if value else sympy.false
    if isinstance(value, (int, np.integer)):
        return sympy.Integer(int(value))
    if isinstance(value, (float, np.floating)):
        return sympy.Float(float(value))
    # fall back to sympify for anything else (e.g. already-sympy objects)
    return sympy.sympify(value)


def to_latex(expr: Any) -> str:
    """Return a LaTeX string for ``expr`` via sympy."""
    sympy = _import_sympy()
    return sympy.latex(to_sympy(expr))


# ---------------------------------------------------------------------------
# sage backend
# ---------------------------------------------------------------------------


def _sage_ops(sage) -> dict[str, Callable[..., Any]]:
    return {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "pow": lambda a, b: a**b,
        "neg": lambda a: -a,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "and": lambda a, b: a & b,
        "or": lambda a, b: a | b,
        "invert": lambda a: ~a,
        "log": sage.log,
        "exp": sage.exp,
        "sqrt": sage.sqrt,
        "abs": sage.abs_symbolic,
        "floor": sage.floor,
        "gammaln": sage.log_gamma,
        "digamma": sage.psi,
        "erf": sage.erf,
        "betaln": lambda a, b: sage.log_gamma(a) + sage.log_gamma(b) - sage.log_gamma(a + b),
        "isnan": lambda a: sage.function("isnan")(a),
        "isinf": lambda a: sage.function("isinf")(a),
        "max": lambda *xs: sage.max_symbolic(*xs),
        "where": lambda cond, a, b: _sage_where(sage, cond, a, b),
        "clip": lambda x, a_min, a_max: _sage_clip(sage, x, a_min, a_max),
        "pi": lambda: sage.pi,
        "e": lambda: sage.e,
        "euler_gamma": lambda: sage.euler_gamma,
        "inf": lambda: sage.oo,
    }


def _sage_where(sage, cond, a, b):
    # Sage lacks a Piecewise-over-relations primitive comparable to sympy's;
    # encode where(cond, a, b) as a*[cond] + b*[!cond] via Heaviside-free
    # indicator using the boolean's truth as 0/1 through `cond.subs`-friendly
    # form.  We use a symbolic indicator built from the relation.
    indicator = sage.SR(cond)
    return a * indicator + b * (1 - indicator)


def _sage_clip(sage, x, a_min, a_max):
    if a_min is not None:
        x = sage.max_symbolic(x, a_min)
    if a_max is not None:
        x = sage.min_symbolic(x, a_max)
    return x


def to_sage(expr: Any) -> Any:
    """Convert a :class:`SymbolicExpression` (or object array of them) to sage.

    Mirrors :func:`to_sympy` against ``sage.all``.  Raises
    :class:`ImportError` if sage is unavailable, and
    :class:`NotImplementedError` for non-symbolic kernels.
    """
    sage = _import_sage()
    ops = _sage_ops(sage)

    def convert(node: Any) -> Any:
        if isinstance(node, SymbolicExpression):
            if node.op == "symbol":
                return sage.var(node.args[0])
            if node.op == "const":
                return sage.SR(node.args[0])
            if node.op in _NON_SYMBOLIC_OPS:
                raise NotImplementedError("symbolic op %r has no sage representation" % node.op)
            handler = ops.get(node.op)
            if handler is None:
                raise NotImplementedError("symbolic op %r is not supported by to_sage" % node.op)
            return handler(*[convert(arg) for arg in node.args])
        if _is_array(node):
            return _map_array(node, convert)
        return sage.SR(node)

    if _is_array(expr):
        return _map_array(expr, convert)
    return convert(expr)
