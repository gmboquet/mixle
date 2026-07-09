"""A small dependency-free symbolic compute engine.

The symbolic engine is for expression tracing, generated-kernel inspection, and
lightweight algebraic experiments.  Numeric execution remains the job of
NumPy/Torch engines; arrays here are NumPy object arrays of scalar expression
nodes so generated kernels can be inspected without a separate runtime.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.special

from mixle.engines.base import ComputeEngine


@dataclass(frozen=True)
class SymbolicExpression:
    """A compact immutable symbolic expression tree."""

    op: str
    args: tuple[Any, ...] = ()

    __array_priority__ = 1000
    __pysp_engine__ = None

    @staticmethod
    def symbol(name: str) -> SymbolicExpression:
        """Create a named symbolic variable node."""
        return SymbolicExpression("symbol", (str(name),))

    @staticmethod
    def constant(value: Any) -> SymbolicExpression:
        """Create a symbolic constant node."""
        return SymbolicExpression("const", (value,))

    @staticmethod
    def call(op: str, *args: Any) -> SymbolicExpression:
        """Create an operation node with symbolic arguments."""
        return SymbolicExpression(op, tuple(_sym(arg) for arg in args))

    def evaluate(self, values: dict[str, Any]) -> Any:
        """Evaluate the expression with a mapping from symbol names to values."""
        if self.op == "symbol":
            return values[self.args[0]]
        if self.op == "const":
            return self.args[0]
        vals = [arg.evaluate(values) if isinstance(arg, SymbolicExpression) else arg for arg in self.args]
        return _EVAL_OPS[self.op](*vals)

    def symbols(self) -> tuple[str, ...]:
        """Return sorted symbolic variable names referenced by this expression."""
        return tuple(sorted(_collect_symbols(self)))

    def op_counts(self) -> dict[str, int]:
        """Return operation counts for this expression tree."""
        counts: Counter = Counter()
        _collect_op_counts(self, counts)
        return dict(counts)

    def depth(self) -> int:
        """Return expression-tree depth, counting this node."""
        return _expression_depth(self)

    def node_count(self) -> int:
        """Return the total number of expression nodes in this tree."""
        return sum(self.op_counts().values())

    def __str__(self) -> str:
        if self.op == "symbol":
            return self.args[0]
        if self.op == "const":
            return repr(self.args[0])
        if not self.args:  # nullary named constant (pi, e, euler_gamma, inf)
            return self.op
        if self.op in _INFIX and len(self.args) == 2:
            return "(%s %s %s)" % (self.args[0], _INFIX[self.op], self.args[1])
        if self.op == "neg":
            return "(-%s)" % self.args[0]
        return "%s(%s)" % (self.op, ", ".join(str(arg) for arg in self.args))

    __repr__ = __str__

    def __add__(self, other):
        return SymbolicExpression.call("add", self, other)

    def __radd__(self, other):
        return SymbolicExpression.call("add", other, self)

    def __sub__(self, other):
        return SymbolicExpression.call("sub", self, other)

    def __rsub__(self, other):
        return SymbolicExpression.call("sub", other, self)

    def __mul__(self, other):
        return SymbolicExpression.call("mul", self, other)

    def __rmul__(self, other):
        return SymbolicExpression.call("mul", other, self)

    def __truediv__(self, other):
        return SymbolicExpression.call("div", self, other)

    def __rtruediv__(self, other):
        return SymbolicExpression.call("div", other, self)

    def __pow__(self, other):
        return SymbolicExpression.call("pow", self, other)

    def __rpow__(self, other):
        return SymbolicExpression.call("pow", other, self)

    def __neg__(self):
        return SymbolicExpression.call("neg", self)

    def __lt__(self, other):
        return SymbolicExpression.call("lt", self, other)

    def __le__(self, other):
        return SymbolicExpression.call("le", self, other)

    def __gt__(self, other):
        return SymbolicExpression.call("gt", self, other)

    def __ge__(self, other):
        return SymbolicExpression.call("ge", self, other)

    def __eq__(self, other):
        return SymbolicExpression.call("eq", self, other)

    def __ne__(self, other):
        return SymbolicExpression.call("ne", self, other)

    def __and__(self, other):
        return SymbolicExpression.call("and", self, other)

    def __rand__(self, other):
        return SymbolicExpression.call("and", other, self)

    def __or__(self, other):
        return SymbolicExpression.call("or", self, other)

    def __ror__(self, other):
        return SymbolicExpression.call("or", other, self)

    def __invert__(self):
        return SymbolicExpression.call("invert", self)

    def __bool__(self):
        raise TypeError("symbolic expressions cannot be used as Python booleans.")


class SymbolicEngine(ComputeEngine):
    """Small symbolic expression engine over scalar nodes and object arrays."""

    name = "symbolic"
    supports_autograd = False

    # Exact/symbolic constants: pi, e and euler_gamma are named nodes (they lower to sympy.pi etc.
    # and never collapse to a float); the small numbers are symbolic constants, with half kept as an
    # exact 1/2 rather than the float 0.5.
    pi = SymbolicExpression("pi", ())
    e = SymbolicExpression("e", ())
    euler_gamma = SymbolicExpression("euler_gamma", ())
    inf = SymbolicExpression("inf", ())
    zero = SymbolicExpression.constant(0)
    one = SymbolicExpression.constant(1)
    two = SymbolicExpression.constant(2)
    half = SymbolicExpression("div", (SymbolicExpression.constant(1), SymbolicExpression.constant(2)))

    def with_precision(self, precision: Any) -> SymbolicEngine:
        """Return this engine unchanged: symbolic expressions carry no float precision policy.

        The numeric engines swap their float dtype here, but symbolic nodes are exact expression
        trees with no reduced-precision representation, so precision adjustment is a no-op rather than
        an error -- this lets backend-neutral code call ``with_precision`` uniformly across engines.
        """
        return self

    def constant(self, value: Any) -> SymbolicExpression:
        """Return ``value`` as a symbolic constant node."""
        return SymbolicExpression.constant(value)

    def symbol(self, name: str) -> SymbolicExpression:
        """Return a named symbolic expression variable."""
        return SymbolicExpression.symbol(name)

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        """Convert scalars/arrays/strings into symbolic expression objects."""
        if isinstance(x, SymbolicExpression):
            return x
        if isinstance(x, str):
            return self.symbol(x)
        arr = np.asarray(x, dtype=dtype)
        if arr.shape == ():
            return SymbolicExpression.constant(arr.item())
        return np.vectorize(_sym, otypes=[object])(arr)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        """Return an object array filled with symbolic zero constants."""
        return np.full(shape, SymbolicExpression.constant(0.0), dtype=object)

    def empty(self, shape: Any, dtype: Any = None) -> Any:
        """Return an uninitialized object array for symbolic expressions."""
        return np.empty(shape, dtype=object)

    def arange(self, *args: Any, **kwargs: Any) -> Any:
        """Return symbolic constants corresponding to ``np.arange`` values."""
        return np.asarray([SymbolicExpression.constant(v) for v in np.arange(*args, **kwargs)], dtype=object)

    def to_numpy(self, x: Any) -> Any:
        """Return ``x`` as a NumPy object array without numeric evaluation."""
        return np.asarray(x, dtype=object)

    def evaluate(self, x: Any, values: dict[str, Any]) -> Any:
        """Evaluate a scalar expression or object-array expression tree."""
        if isinstance(x, SymbolicExpression):
            return x.evaluate(values)
        arr = np.asarray(x, dtype=object)
        if arr.shape == ():
            value = arr.item()
            return value.evaluate(values) if isinstance(value, SymbolicExpression) else value
        return np.vectorize(
            lambda value: value.evaluate(values) if isinstance(value, SymbolicExpression) else value,
            otypes=[object],
        )(arr)

    def symbols(self, x: Any) -> tuple[str, ...]:
        """Return sorted symbolic variable names referenced by ``x``."""
        names = set()
        for expr in _iter_expressions(x):
            names.update(expr.symbols())
        return tuple(sorted(names))

    def op_counts(self, x: Any) -> dict[str, int]:
        """Return aggregate operation counts over a scalar or array expression."""
        counts: Counter = Counter()
        for expr in _iter_expressions(x):
            _collect_op_counts(expr, counts)
        return dict(counts)

    def diagnostics(self, x: Any) -> dict[str, Any]:
        """Return a compact diagnostic summary for generated-kernel inspection."""
        expressions = tuple(_iter_expressions(x))
        counts: Counter = Counter()
        max_depth = 0
        names = set()
        for expr in expressions:
            _collect_op_counts(expr, counts)
            max_depth = max(max_depth, expr.depth())
            names.update(expr.symbols())
        return {
            "num_expressions": len(expressions),
            "symbols": tuple(sorted(names)),
            "op_counts": dict(counts),
            "max_depth": max_depth,
        }

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        """Stack symbolic arrays with NumPy object-array semantics."""
        return np.stack(tuple(arrays), axis=axis)

    log = staticmethod(lambda x: _elementwise_call("log", x))
    exp = staticmethod(lambda x: _elementwise_call("exp", x))
    sqrt = staticmethod(lambda x: _elementwise_call("sqrt", x))
    abs = staticmethod(lambda x: _elementwise_call("abs", x))
    floor = staticmethod(lambda x: _elementwise_call("floor", x))
    gammaln = staticmethod(lambda x: _elementwise_call("gammaln", x))
    digamma = staticmethod(lambda x: _elementwise_call("digamma", x))
    erf = staticmethod(lambda x: _elementwise_call("erf", x))

    @staticmethod
    def sum(x: Any, axis: Any = None, *args: Any, **kwargs: Any) -> Any:
        """Return a symbolic sum reduction over ``axis``."""
        return _reduce_symbolic(x, _sum_values, axis=axis)

    @staticmethod
    def logsumexp(x: Any, axis: Any = None, *args: Any, **kwargs: Any) -> Any:
        """Return a symbolic log-sum-exp reduction over ``axis``."""
        return _reduce_symbolic(x, _logsumexp_values, axis=axis)

    @staticmethod
    def where(cond: Any, x: Any, y: Any) -> Any:
        """Return a symbolic elementwise conditional expression."""
        return _elementwise_call("where", cond, x, y)

    @staticmethod
    def maximum(x: Any, y: Any) -> Any:
        """Return a symbolic elementwise maximum expression."""
        return _elementwise_call("max", x, y)

    @staticmethod
    def less(x: Any, y: Any) -> Any:
        """Return a symbolic less-than comparison."""
        return _elementwise_call("lt", x, y)

    @staticmethod
    def less_equal(x: Any, y: Any) -> Any:
        """Return a symbolic less-than-or-equal comparison."""
        return _elementwise_call("le", x, y)

    @staticmethod
    def greater(x: Any, y: Any) -> Any:
        """Return a symbolic greater-than comparison."""
        return _elementwise_call("gt", x, y)

    @staticmethod
    def greater_equal(x: Any, y: Any) -> Any:
        """Return a symbolic greater-than-or-equal comparison."""
        return _elementwise_call("ge", x, y)

    @staticmethod
    def equal(x: Any, y: Any) -> Any:
        """Return a symbolic equality comparison."""
        return _elementwise_call("eq", x, y)

    @staticmethod
    def not_equal(x: Any, y: Any) -> Any:
        """Return a symbolic inequality comparison."""
        return _elementwise_call("ne", x, y)

    @staticmethod
    def logical_and(x: Any, y: Any) -> Any:
        """Return a symbolic elementwise logical-and expression."""
        return _elementwise_call("and", x, y)

    @staticmethod
    def logical_or(x: Any, y: Any) -> Any:
        """Return a symbolic elementwise logical-or expression."""
        return _elementwise_call("or", x, y)

    @staticmethod
    def logical_not(x: Any) -> Any:
        """Return a symbolic elementwise logical-not expression."""
        return _elementwise_call("invert", x)

    @staticmethod
    def clip(x: Any, a_min: Any = None, a_max: Any = None) -> Any:
        """Return a symbolic clipped expression."""
        return _elementwise_call("clip", x, a_min, a_max)

    isnan = staticmethod(lambda x: _elementwise_call("isnan", x))
    isinf = staticmethod(lambda x: _elementwise_call("isinf", x))
    max = staticmethod(lambda x, axis=None, *args, **kwargs: _reduce_symbolic(x, _max_values, axis=axis))
    dot = staticmethod(lambda x, y: np.dot(_sym_array(x), _sym_array(y)))
    matmul = staticmethod(lambda x, y: np.matmul(_sym_array(x), _sym_array(y)))
    cumsum = staticmethod(lambda x, axis=None, *args, **kwargs: np.cumsum(_sym_array(x), axis=axis))
    bincount = staticmethod(lambda x, *args, **kwargs: SymbolicExpression.call("bincount", x))
    unique = staticmethod(lambda x, *args, **kwargs: SymbolicExpression.call("unique", x))
    searchsorted = staticmethod(lambda x, y, *args, **kwargs: SymbolicExpression.call("searchsorted", x, y))
    betaln = staticmethod(lambda x, y: _elementwise_call("betaln", x, y))

    def index_add(self, out: Any, index: Any, values: Any) -> Any:
        """Return a symbolic index-add operation node."""
        return SymbolicExpression.call("index_add", out, index, values)

    @staticmethod
    def to_sympy(x: Any) -> Any:
        """Lower a symbolic expression (or object array) to a sympy expression."""
        from mixle.engines.symbolic_export import to_sympy

        return to_sympy(x)

    @staticmethod
    def to_sage(x: Any) -> Any:
        """Lower a symbolic expression (or object array) to a sage expression."""
        from mixle.engines.symbolic_export import to_sage

        return to_sage(x)

    @staticmethod
    def to_latex(x: Any) -> str:
        """Return a LaTeX string for a symbolic expression via sympy."""
        from mixle.engines.symbolic_export import to_latex

        return to_latex(x)


def _sum_values(values: Any) -> SymbolicExpression:
    values = np.asarray(values, dtype=object).reshape(-1)
    rv = SymbolicExpression.constant(0.0)
    for value in values:
        rv = rv + value
    return rv


def _max_values(values: Any) -> SymbolicExpression:
    values = np.asarray(values, dtype=object).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot reduce an empty symbolic array.")
    rv = _sym(values[0])
    for value in values[1:]:
        rv = SymbolicExpression.call("max", rv, value)
    return rv


def _logsumexp_values(values: Any) -> SymbolicExpression:
    values = np.asarray(values, dtype=object).reshape(-1)
    # Build the max-shifted form ``m + log(sum(exp(x - m)))`` so the numeric
    # evaluate path is stable (a naive ``log(sum(exp(x)))`` overflows, e.g.
    # ``[1000, 1000]``) while staying algebraically equivalent for export.
    shift = _max_values(values)
    terms = [SymbolicExpression.call("exp", _sym(value) - shift) for value in values]
    total = SymbolicExpression.constant(0.0)
    for term in terms:
        total = total + term
    return shift + SymbolicExpression.call("log", total)


def _reduce_symbolic(x: Any, reducer: Callable[[Any], SymbolicExpression], axis: Any = None) -> Any:
    arr = _sym_array(x)
    if axis is None:
        return reducer(arr.reshape(-1))
    if isinstance(axis, tuple):
        rv = arr
        for one_axis in sorted(axis, reverse=True):
            rv = _reduce_symbolic(rv, reducer, axis=one_axis)
        return rv
    return np.apply_along_axis(lambda values: reducer(values), int(axis), arr)


def _elementwise_call(op: str, *args: Any) -> Any:
    arrays = [np.asarray(arg, dtype=object) for arg in args]
    if all(arr.shape == () for arr in arrays):
        return SymbolicExpression.call(op, *[arr.item() for arr in arrays])
    bcast = np.broadcast_arrays(*arrays)
    out = np.empty(bcast[0].shape, dtype=object)
    for idx in np.ndindex(out.shape):
        out[idx] = SymbolicExpression.call(op, *[arr[idx] for arr in bcast])
    return out


def _sym_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=object)
    if arr.shape == ():
        return np.asarray(_sym(arr.item()), dtype=object)
    return np.vectorize(_sym, otypes=[object])(arr)


def _sym(x: Any) -> SymbolicExpression:
    if isinstance(x, SymbolicExpression):
        return x
    if isinstance(x, str):
        return SymbolicExpression.symbol(x)
    return SymbolicExpression.constant(x)


def _iter_expressions(x: Any) -> Iterable[SymbolicExpression]:
    if isinstance(x, SymbolicExpression):
        yield x
        return
    arr = np.asarray(x, dtype=object)
    for value in arr.reshape(-1):
        if isinstance(value, SymbolicExpression):
            yield value


def _collect_symbols(expr: Any) -> set:
    if not isinstance(expr, SymbolicExpression):
        return set()
    if expr.op == "symbol":
        return {expr.args[0]}
    names = set()
    for arg in expr.args:
        names.update(_collect_symbols(arg))
    return names


def _collect_op_counts(expr: Any, counts: Counter) -> None:
    if not isinstance(expr, SymbolicExpression):
        return
    counts[expr.op] += 1
    for arg in expr.args:
        _collect_op_counts(arg, counts)


def _expression_depth(expr: Any) -> int:
    if not isinstance(expr, SymbolicExpression) or not expr.args:
        return 1
    child_depths = [_expression_depth(arg) for arg in expr.args if isinstance(arg, SymbolicExpression)]
    return 1 + (max(child_depths) if child_depths else 0)


_INFIX = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "pow": "**",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
    "and": "&",
    "or": "|",
}


def _clip_value(x: Any, a_min: Any, a_max: Any) -> Any:
    if a_min is not None:
        x = max(x, a_min)
    if a_max is not None:
        x = min(x, a_max)
    return x


_EVAL_OPS: dict[str, Callable[..., Any]] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "pow": lambda x, y: x**y,
    "neg": lambda x: -x,
    "lt": lambda x, y: x < y,
    "le": lambda x, y: x <= y,
    "gt": lambda x, y: x > y,
    "ge": lambda x, y: x >= y,
    "eq": lambda x, y: x == y,
    "ne": lambda x, y: x != y,
    "and": lambda x, y: bool(x) and bool(y),
    "or": lambda x, y: bool(x) or bool(y),
    "invert": lambda x: not bool(x),
    "log": math.log,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "abs": abs,
    "floor": math.floor,
    "max": lambda *xs: max(xs),
    "where": lambda cond, x, y: x if bool(cond) else y,
    "clip": _clip_value,
    "gammaln": math.lgamma,
    "digamma": scipy.special.digamma,
    "erf": math.erf,
    "isnan": math.isnan,
    "isinf": math.isinf,
    "betaln": lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y),
    # nullary named constants
    "pi": lambda: math.pi,
    "e": lambda: math.e,
    "euler_gamma": lambda: 0.5772156649015328606,
    "inf": lambda: math.inf,
}


#: Shared symbolic engine; arithmetic on symbolic nodes/object arrays dispatches here.
SYMBOLIC_ENGINE = SymbolicEngine()

# Tag scalar expression nodes so mixle.engines.arithmetic recovers the symbolic engine.
SymbolicExpression.__pysp_engine__ = SYMBOLIC_ENGINE


def is_symbolic_payload(x: Any) -> bool:
    """Return True for a symbolic node or a NumPy object array of symbolic nodes."""
    if isinstance(x, SymbolicExpression):
        return True
    if isinstance(x, np.ndarray) and x.dtype == object and x.size:
        return isinstance(x.flat[0], SymbolicExpression)
    return False
