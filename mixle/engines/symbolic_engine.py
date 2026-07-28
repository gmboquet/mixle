"""A small dependency-free symbolic compute engine.

The symbolic engine is for expression tracing, generated-kernel inspection, and
lightweight algebraic experiments.  Numeric execution remains the job of
NumPy/Torch engines; arrays here are NumPy object arrays of scalar expression
nodes so generated kernels can be inspected without a separate runtime.
"""

from __future__ import annotations

import functools
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

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        """Join symbolic arrays with NumPy object-array semantics."""
        return np.concatenate(tuple(arrays), axis=axis)

    log = staticmethod(lambda x: _elementwise_call("log", x))
    exp = staticmethod(lambda x: _elementwise_call("exp", x))
    sqrt = staticmethod(lambda x: _elementwise_call("sqrt", x))
    abs = staticmethod(lambda x: _elementwise_call("abs", x))
    floor = staticmethod(lambda x: _elementwise_call("floor", x))
    gammaln = staticmethod(lambda x: _elementwise_call("gammaln", x))
    digamma = staticmethod(lambda x: _elementwise_call("digamma", x))
    erf = staticmethod(lambda x: _elementwise_call("erf", x))

    @staticmethod
    def sum(
        x: Any,
        axis: Any = None,
        dtype: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = None,
    ) -> Any:
        """Return a symbolic sum reduction over ``axis``, matching NumPy's ``sum`` contract.

        ``keepdims``, ``initial``, and ``where`` are implemented exactly as ``np.sum`` defines them:
        reduced axes are kept as size-1 dimensions, ``initial`` seeds each reduction group (the
        additive identity 0.0 when omitted), and ``where`` excludes masked-out elements by
        substituting the additive identity in their place before folding. ``dtype`` has no symbolic
        analogue -- an expression node carries no numeric storage to cast -- so a non-``None`` value
        is rejected rather than silently ignored (matching the "reject unsupported arguments" half
        of the declared reduction contract, since honoring it is not meaningful here).
        """
        if dtype is not None:
            raise NotImplementedError(
                "the symbolic engine does not support dtype casting in reductions (got dtype=%r); "
                "a symbolic expression carries no numeric dtype to cast into." % (dtype,)
            )
        if where is not None:
            x = _mask_with_identity(_sym_array(x), where, SymbolicExpression.constant(0.0))
        return _reduce_symbolic(x, lambda values: _sum_values(values, initial=initial), axis=axis, keepdims=keepdims)

    @staticmethod
    def logsumexp(x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        """Return a symbolic log-sum-exp reduction over ``axis``, matching SciPy's ``logsumexp``.

        ``keepdims`` is implemented exactly as ``scipy.special.logsumexp`` defines it. SciPy's own
        ``logsumexp`` (which ``NumpyEngine.logsumexp`` forwards to directly) has no
        ``dtype``/``initial``/``where`` parameter either, so there is no declared behavior to honor
        for them here; passing them raises the same ``TypeError`` SciPy's own function would, for the
        same reason (an unexpected keyword argument). An all-``-inf`` input (every term has zero
        probability in log-space) correctly evaluates to ``-inf``, not ``NaN`` -- see
        :func:`_logsumexp_values`. An empty reduction is likewise defined as ``-inf`` (the identity:
        the log of a sum of zero terms), matching SciPy.
        """
        return _reduce_symbolic(x, _logsumexp_values, axis=axis, keepdims=keepdims)

    @staticmethod
    def max(
        x: Any,
        axis: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = None,
    ) -> Any:
        """Return a symbolic max reduction over ``axis``, matching NumPy's ``max`` contract.

        ``keepdims`` and ``initial`` are implemented exactly as ``np.max`` defines them (``np.max``
        itself has no ``dtype`` parameter, so none is declared here either). Max has no universal
        identity element, so -- exactly like ``np.max`` -- passing ``where`` without also passing
        ``initial`` is rejected rather than silently reducing an under-specified group. An empty
        reduction with no ``initial`` raises (there is nothing to seed it with), matching ``np.max``;
        with ``initial`` it returns ``initial``.
        """
        arr = _sym_array(x)
        if where is not None:
            if initial is None:
                raise ValueError(
                    "the symbolic 'max' reduction has no identity element, so using 'where' also "
                    "requires an explicit 'initial' (matching np.max's own restriction)."
                )
            arr = _mask_with_identity(arr, where, _sym(initial))
        return _reduce_symbolic(arr, lambda values: _max_values(values, initial=initial), axis=axis, keepdims=keepdims)

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
    dot = staticmethod(lambda x, y: np.dot(_sym_array(x), _sym_array(y)))
    matmul = staticmethod(lambda x, y: np.matmul(_sym_array(x), _sym_array(y)))
    bincount = staticmethod(lambda x, *args, **kwargs: SymbolicExpression.call("bincount", x))
    unique = staticmethod(lambda x, *args, **kwargs: SymbolicExpression.call("unique", x))
    searchsorted = staticmethod(lambda x, y, *args, **kwargs: SymbolicExpression.call("searchsorted", x, y))
    betaln = staticmethod(lambda x, y: _elementwise_call("betaln", x, y))

    @staticmethod
    def cumsum(x: Any, axis: Any = None, dtype: Any = None, out: Any = None) -> Any:
        """Return a symbolic cumulative sum over ``axis``, matching NumPy's ``cumsum`` contract.

        ``axis`` behaves exactly as ``np.cumsum`` defines it (``None`` flattens first). ``dtype`` has
        no symbolic analogue -- an expression node carries no numeric storage to cast into, the same
        reason :meth:`sum` rejects it -- and ``out`` cannot be honored either, because the result is a
        freshly built object array of expression nodes rather than a buffer that can be written into.
        Both are therefore rejected rather than silently discarded (MXR-080-1567: the previous
        implementation was a lambda that swallowed ``dtype``, ``out``, and any other argument into
        unused ``*args``/``**kwargs`` and passed only ``axis`` through, so a caller asking for a
        float32 accumulation or an output buffer got neither and no signal). An unexpected keyword now
        raises ``TypeError`` from ordinary signature binding, as the other engines' ``cumsum`` does.
        """
        if dtype is not None:
            raise NotImplementedError(
                "the symbolic engine does not support dtype casting in cumsum (got dtype=%r); "
                "a symbolic expression carries no numeric dtype to cast into." % (dtype,)
            )
        if out is not None:
            raise NotImplementedError(
                "the symbolic engine does not support cumsum(out=...) (got out=%r); the result is a "
                "newly built object array of expression nodes, not a writable numeric buffer." % (out,)
            )
        return np.cumsum(_sym_array(x), axis=axis)

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


def _sum_values(values: Any, initial: Any = None) -> SymbolicExpression:
    """Fold ``values`` with ``+``, seeded by ``initial`` (the additive identity 0.0 when omitted).

    Sum always has a well-defined identity, so -- unlike :func:`_max_values` -- an empty ``values``
    is never an error: it just returns the seed unchanged, exactly as ``np.sum(np.array([]))`` (0.0)
    or ``np.sum(np.array([]), initial=x)`` (x) does.
    """
    values = np.asarray(values, dtype=object).reshape(-1)
    rv = SymbolicExpression.constant(0.0) if initial is None else _sym(initial)
    for value in values:
        rv = rv + value
    return rv


def _max_values(values: Any, initial: Any = None) -> SymbolicExpression:
    """Fold ``values`` with elementwise ``max``, optionally seeded by ``initial``.

    Max has no universal identity, so an empty ``values`` raises unless ``initial`` was given to
    seed the fold (matching ``np.max``'s own restriction: ``np.max([])`` raises, ``np.max([],
    initial=x)`` returns ``x``). When ``initial`` is omitted and ``values`` is non-empty, this is
    byte-identical to the pre-fix code: seed from the first element, fold over the rest.
    """
    values = np.asarray(values, dtype=object).reshape(-1)
    if values.size == 0:
        if initial is None:
            raise ValueError("cannot reduce an empty symbolic array.")
        return _sym(initial)
    if initial is None:
        rv, rest = _sym(values[0]), values[1:]
    else:
        rv, rest = _sym(initial), values
    for value in rest:
        rv = SymbolicExpression.call("max", rv, value)
    return rv


def _logsumexp_values(values: Any) -> SymbolicExpression:
    """Return the symbolic max-shifted log-sum-exp of ``values``: ``m + log(sum(exp(x - m)))``.

    The shift keeps the numeric evaluate path stable (a naive ``log(sum(exp(x)))`` overflows, e.g.
    ``[1000, 1000]``) while staying algebraically equivalent for export. An empty ``values`` is
    defined as ``-inf`` (the log of a sum of zero terms), matching ``scipy.special.logsumexp``.

    An all-``-inf`` input is the degenerate case where the shift ``m`` ITSELF is ``-inf`` (every
    term has zero probability in log-space): naively evaluating the shifted formula then computes
    ``-inf - (-inf)`` for every term, an indeterminate form that is ``NaN`` in IEEE float
    arithmetic, poisoning the whole reduction to ``NaN`` instead of the correct ``-inf``. Guard
    this by wrapping the naive formula in a symbolic ``where(m == -inf, -inf, naive)``: evaluate()
    still computes the naive branch eagerly (it is not short-circuited), so the "poisoned" NaN IS
    computed in that case, but it is then discarded by the selection rather than propagated,
    since only the ``-inf`` branch is actually returned. This composes correctly with an outer
    ``keepdims``/``axis`` reduction (handled entirely in :func:`_reduce_symbolic`, agnostic to what
    an individual reducer call returns) and leaves the ordinary (finite-shift) case's expression
    structure -- and therefore its op_counts()/diagnostics()/to_sympy() lowering -- unchanged: the
    "where"/"eq" wrapper only adds nodes around the untouched naive branch, it does not alter it.
    """
    values = np.asarray(values, dtype=object).reshape(-1)
    neg_inf = SymbolicExpression.call("neg", SymbolicExpression("inf", ()))
    if values.size == 0:
        return neg_inf
    shift = _max_values(values)
    terms = [SymbolicExpression.call("exp", _sym(value) - shift) for value in values]
    total = SymbolicExpression.constant(0.0)
    for term in terms:
        total = total + term
    naive = shift + SymbolicExpression.call("log", total)
    shift_is_neg_inf = SymbolicExpression.call("eq", shift, neg_inf)
    return SymbolicExpression.call("where", shift_is_neg_inf, neg_inf, naive)


def _mask_with_identity(arr: np.ndarray, where: Any, identity: SymbolicExpression) -> np.ndarray:
    """Return ``arr`` with elements outside ``where`` (broadcast to ``arr``'s shape) replaced by
    ``identity``, so a masked-out position folds into a reduction without affecting its result."""
    mask = np.broadcast_to(np.asarray(where, dtype=bool), arr.shape)
    out = np.empty(arr.shape, dtype=object)
    for idx in np.ndindex(arr.shape):
        out[idx] = arr[idx] if mask[idx] else identity
    return out


def _reduce_symbolic(
    x: Any,
    reducer: Callable[[Any], SymbolicExpression],
    axis: Any = None,
    keepdims: bool = False,
) -> Any:
    """Reduce ``x`` with ``reducer`` over ``axis``, matching NumPy's ``axis``/``keepdims`` contract.

    ``reducer`` folds a flat 1-D slice of symbolic values into a single :class:`SymbolicExpression`
    (see :func:`_sum_values`/:func:`_max_values`/:func:`_logsumexp_values`); this wrapper supplies
    the NumPy-style axis bookkeeping around it -- including, when ``keepdims`` is requested,
    reinserting the reduced axes as size-1 dimensions rather than dropping them (the pre-fix
    behavior, which silently ignored ``keepdims`` entirely and always returned the reduced-rank
    result).
    """
    arr = _sym_array(x)
    result = _reduce_over_axis(arr, reducer, axis=axis)
    if not keepdims:
        return result
    ndim = arr.ndim
    if axis is None:
        reduced_axes: tuple[int, ...] = tuple(range(ndim))
    elif isinstance(axis, tuple):
        # Same normalization the reduction itself used (MXR-080-1566), so keepdims reinserts the
        # dimensions that were actually reduced.
        reduced_axes = tuple(sorted(_normalize_reduction_axes(axis, ndim))) if ndim else (0,) * len(axis)
    else:
        reduced_axes = ((int(axis) % ndim) if ndim else 0,)
    out = np.asarray(result, dtype=object)
    for one_axis in reduced_axes:
        out = np.expand_dims(out, axis=one_axis)
    return out


def _normalize_reduction_axes(axis: tuple, ndim: int) -> tuple[int, ...]:
    """Resolve a tuple of reduction axes against ``ndim``, matching NumPy's own contract.

    MXR-080-1566: a negative axis names a position relative to the rank it is applied at, and the
    sequential fold in :func:`_reduce_over_axis` drops a dimension per pass -- so every axis must be
    resolved against the ORIGINAL rank before any of them is used, or the second pass reduces a
    different dimension than NumPy says it does. Duplicates are rejected exactly as NumPy does,
    rather than silently reducing two different dimensions.
    """
    normalized: list[int] = []
    for one_axis in axis:
        value = int(one_axis)
        resolved = value + ndim if value < 0 else value
        if not 0 <= resolved < ndim:
            raise np.exceptions.AxisError(value, ndim)
        if resolved in normalized:
            raise ValueError("duplicate value in 'axis'")
        normalized.append(resolved)
    return tuple(normalized)


def _reduce_over_axis(arr: np.ndarray, reducer: Callable[[Any], SymbolicExpression], axis: Any = None) -> Any:
    if axis is None:
        return reducer(arr.reshape(-1))
    if isinstance(axis, tuple):
        rv = arr
        for one_axis in sorted(_normalize_reduction_axes(axis, arr.ndim), reverse=True):
            rv = _reduce_over_axis(rv, reducer, axis=one_axis)
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
    if _any_vector_valued(x, a_min, a_max):
        # MXR-080-1564: builtin max/min on an array operand force an ambiguous bool() conversion.
        if a_min is not None:
            x = np.maximum(x, a_min)
        if a_max is not None:
            x = np.minimum(x, a_max)
        return x
    if a_min is not None:
        x = max(x, a_min)
    if a_max is not None:
        x = min(x, a_max)
    return x


def _is_vector_valued(x: Any) -> bool:
    """Return whether ``x`` is an array with at least one axis (as opposed to a Python scalar
    or a 0-d array), i.e. a value ``bool(x)`` cannot always convert unambiguously."""
    return isinstance(x, np.ndarray) and x.ndim > 0


def _any_vector_valued(*values: Any) -> bool:
    """Whether any evaluated operand is a genuine (ndim > 0) array."""
    return any(_is_vector_valued(value) for value in values)


def _array_aware(scalar_fn: Callable[..., Any], array_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Dispatch to ``array_fn`` when any evaluated operand is a genuine array, else ``scalar_fn``.

    MXR-080-1564: :meth:`SymbolicExpression.evaluate` advertises array-valued symbol bindings, but the
    scalar ``math``/builtin implementations of these ops cannot accept one -- ``math.log`` raises
    "only 0-dimensional arrays can be converted to Python scalars" and builtin ``max``/``min`` raise
    the ambiguous-truth error. Only genuine ``ndim > 0`` arrays take the NumPy/SciPy path, so scalar
    and 0-d results stay byte-identical to the pre-fix behavior (notably ``math.floor``'s ``int`` and
    ``math.isnan``'s ``bool``, which their NumPy counterparts do not reproduce).
    """

    def evaluate(*values: Any) -> Any:
        return array_fn(*values) if _any_vector_valued(*values) else scalar_fn(*values)

    return evaluate


def _max_of(*values: Any) -> Any:
    """Elementwise/variadic max, array-aware (MXR-080-1564)."""
    if _any_vector_valued(*values):
        return functools.reduce(np.maximum, values)
    return max(values)


def _betaln_scalar(x: Any, y: Any) -> Any:
    return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)


def _eval_and(x: Any, y: Any) -> Any:
    if _is_vector_valued(x) or _is_vector_valued(y):
        return np.logical_and(x, y)
    return bool(x) and bool(y)


def _eval_or(x: Any, y: Any) -> Any:
    if _is_vector_valued(x) or _is_vector_valued(y):
        return np.logical_or(x, y)
    return bool(x) or bool(y)


def _eval_invert(x: Any) -> Any:
    if _is_vector_valued(x):
        return np.logical_not(x)
    return not bool(x)


def _eval_where(cond: Any, x: Any, y: Any) -> Any:
    if _is_vector_valued(cond) or _is_vector_valued(x) or _is_vector_valued(y):
        return np.where(cond, x, y)
    return x if bool(cond) else y


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
    # Symbol values may be bound to vector (array) data at evaluate() time, not just scalars, so
    # these must not force a scalar bool() conversion: numpy raises "the truth value of an array
    # with more than one element is ambiguous" for any >1-element array, which broke every
    # vector-valued and/or/not/where evaluation outright (MXR-080-0152). The scalar/0-d path keeps
    # the exact original bool()-based logic (byte-identical behavior); only genuine ndim>0 arrays
    # take the array-aware numpy path.
    "and": _eval_and,
    "or": _eval_or,
    "invert": _eval_invert,
    # Likewise (MXR-080-1564): the scalar `math` implementations of these ops cannot accept an
    # array-bound symbol at all -- math.log/exp/sqrt/floor/lgamma/erf/isnan/isinf raise "only
    # 0-dimensional arrays can be converted to Python scalars", and the builtin max/min behind
    # `max`/`clip` raise the ambiguous-truth error -- so an array binding is routed to the
    # NumPy/SciPy equivalent. `abs` and `digamma` already accept arrays unchanged.
    "log": _array_aware(math.log, np.log),
    "exp": _array_aware(math.exp, np.exp),
    "sqrt": _array_aware(math.sqrt, np.sqrt),
    "abs": abs,
    "floor": _array_aware(math.floor, np.floor),
    "max": _max_of,
    "where": _eval_where,
    "clip": _clip_value,
    "gammaln": _array_aware(math.lgamma, scipy.special.gammaln),
    "digamma": scipy.special.digamma,
    "erf": _array_aware(math.erf, scipy.special.erf),
    "isnan": _array_aware(math.isnan, np.isnan),
    "isinf": _array_aware(math.isinf, np.isinf),
    "betaln": _array_aware(_betaln_scalar, scipy.special.betaln),
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
    """Return True for a symbolic node or a NumPy object array of ALL-symbolic nodes.

    Every element of an object array is checked, not just the first: classifying the whole
    array from a single element lets a mixed-ownership payload -- some elements genuine
    ``SymbolicExpression`` nodes, others unrelated plain objects -- be misrouted based on
    whichever kind happens to occupy index 0 (numeric-first routes the symbolic elements
    through NumPy, which was never built to handle them; symbolic-first routes the unrelated
    elements through the symbolic engine instead). Mixed ownership is rejected outright rather
    than silently guessed at from one element.

    Scans left to right and stops at the first element whose kind disagrees with the first
    element's, so a genuinely uniform array (the overwhelmingly common case) still only pays for
    a single early-exit scan, while a mixed one is still caught regardless of which kind happens
    to sit at index 0.
    """
    if isinstance(x, SymbolicExpression):
        return True
    if isinstance(x, np.ndarray) and x.dtype == object and x.size:
        flat = x.reshape(-1)
        first_is_symbolic = isinstance(flat[0], SymbolicExpression)
        for value in flat:
            if isinstance(value, SymbolicExpression) != first_is_symbolic:
                raise TypeError(
                    "object array has mixed ownership: element %r is %ssymbolic while element %r is "
                    "%ssymbolic; a payload must be entirely SymbolicExpression nodes or entirely "
                    "non-symbolic, not a mix of the two."
                    % (
                        flat[0],
                        "" if first_is_symbolic else "not ",
                        value,
                        "" if not first_is_symbolic else "not ",
                    )
                )
        return first_is_symbolic
    return False
