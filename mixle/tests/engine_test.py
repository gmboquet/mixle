import importlib
import math
import tempfile
import unittest

import numpy as np
import scipy.special

from mixle.engines import (
    NUMPY_ENGINE,
    JaxEngine,
    NumpyEngine,
    SymbolicEngine,
    SymbolicExpression,
    TorchEngine,
    engine_of,
    engine_with_precision,
    precision_name,
    to_numpy,
)
from mixle.engines import arithmetic as ar

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
else:
    torch = None

HAS_JAX = importlib.util.find_spec("jax") is not None
if HAS_JAX:
    import jax
    import jax.numpy as jnp
else:
    jax = None
    jnp = None


def _single_rank_mesh():
    import torch.distributed as dist
    from torch.distributed.tensor import DeviceMesh

    if not dist.is_initialized():
        path = tempfile.NamedTemporaryFile(delete=False).name
        dist.init_process_group("gloo", rank=0, world_size=1, init_method="file://" + path)
    return DeviceMesh("cpu", [0])


class EngineTestCase(unittest.TestCase):
    def test_numpy_engine_recovery_for_nested_encoding(self):
        enc = (np.asarray([1.0, 2.0]), {"x": np.asarray([3])})
        self.assertIsInstance(engine_of(enc), NumpyEngine)

    def test_numpy_arithmetic_matches_numpy(self):
        x = np.asarray([1.0, 4.0, 9.0])
        np.testing.assert_allclose(ar.sqrt(x), np.sqrt(x))
        np.testing.assert_allclose(ar.log(x), np.log(x))
        self.assertAlmostEqual(ar.dot(x, x), np.dot(x, x))

    def test_numpy_engine_precision_policy(self):
        engine = NumpyEngine(dtype="float32")
        x = engine.asarray([1.0, 2.0])

        self.assertEqual(x.dtype, np.dtype("float32"))
        self.assertEqual(engine.zeros(2).dtype, np.dtype("float32"))
        self.assertEqual(engine.arange(0.0, 1.0, 0.25).dtype, np.dtype("float32"))
        self.assertEqual(engine.asarray([1, 2]).dtype, np.dtype("int64"))
        self.assertEqual(engine.precision, "float32")

    def test_engine_with_precision_returns_adjusted_engine(self):
        engine = engine_with_precision(NUMPY_ENGINE, "float32")

        self.assertIsInstance(engine, NumpyEngine)
        self.assertEqual(engine.asarray([1.0]).dtype, np.dtype("float32"))
        self.assertEqual(precision_name(np.float64), "float64")

    def test_symbolic_engine_builds_and_evaluates_scalar_expression(self):
        engine = SymbolicEngine()
        x = engine.symbol("x")
        expr = engine.log(x * x + 1.0)

        self.assertIsInstance(expr, SymbolicExpression)
        self.assertIn("log", str(expr))
        self.assertAlmostEqual(expr.evaluate({"x": 2.0}), np.log(5.0))
        self.assertEqual(expr.symbols(), ("x",))
        self.assertEqual(expr.depth(), 4)
        self.assertEqual(expr.node_count(), 6)
        self.assertEqual(expr.op_counts()["symbol"], 2)
        self.assertEqual(engine.diagnostics(expr)["symbols"], ("x",))

    def test_symbolic_engine_traces_array_expressions(self):
        engine = SymbolicEngine()
        x = engine.symbol("x")
        y = engine.symbol("y")
        arr = engine.asarray([[x, 2.0], [y, 4.0]])

        logged = engine.log(arr + 1.0)
        col_sum = engine.sum(arr, axis=0)
        product = engine.matmul(arr, engine.asarray([1.0, 2.0]))
        row_lse = engine.logsumexp(arr, axis=1)

        np.testing.assert_allclose(
            np.asarray(engine.evaluate(logged, {"x": 1.0, "y": 3.0}), dtype=float),
            np.log(np.asarray([[2.0, 3.0], [4.0, 5.0]])),
        )
        np.testing.assert_allclose(
            np.asarray(engine.evaluate(col_sum, {"x": 1.0, "y": 3.0}), dtype=float), np.asarray([4.0, 6.0])
        )
        np.testing.assert_allclose(
            np.asarray(engine.evaluate(product, {"x": 1.0, "y": 3.0}), dtype=float), np.asarray([5.0, 11.0])
        )
        np.testing.assert_allclose(
            np.asarray(engine.evaluate(row_lse, {"x": 1.0, "y": 3.0}), dtype=float),
            np.log(np.exp(np.asarray([[1.0, 2.0], [3.0, 4.0]])).sum(axis=1)),
        )

        diagnostics = engine.diagnostics(row_lse)
        self.assertEqual(diagnostics["num_expressions"], 2)
        self.assertEqual(diagnostics["symbols"], ("x", "y"))
        self.assertEqual(diagnostics["op_counts"]["log"], 2)
        self.assertEqual(diagnostics["op_counts"]["exp"], 4)
        self.assertGreaterEqual(diagnostics["max_depth"], 4)

    def _eval_all(self, engine, expr):
        """Evaluate every element of a (possibly array-shaped) symbolic result against {}."""
        return np.vectorize(lambda v: v.evaluate({}))(np.asarray(expr, dtype=object))

    def test_symbolic_sum_keepdims_retains_reduced_axes_as_size_one(self):
        # Regression (MXR-080-0150), exact audit repro: a symbolic sum requested with
        # keepdims=True used to return a scalar (the reduced axis silently dropped, since
        # keepdims was accepted into **kwargs and then never read) instead of an array with the
        # reduced axis kept as a size-1 dimension, unlike NumPy/Torch.
        engine = SymbolicEngine()
        arr2d = engine.asarray([[1.0, 2.0], [3.0, 4.0]])

        by_axis = engine.sum(arr2d, axis=1, keepdims=True)
        self.assertEqual(np.asarray(by_axis).shape, (2, 1))
        np.testing.assert_allclose(self._eval_all(engine, by_axis), [[3.0], [7.0]])

        by_none = engine.sum(arr2d, axis=None, keepdims=True)
        self.assertEqual(np.asarray(by_none).shape, (1, 1))
        np.testing.assert_allclose(self._eval_all(engine, by_none), [[10.0]])

        arr3d = engine.asarray(np.arange(24, dtype=float).reshape(2, 3, 4))
        by_tuple = engine.sum(arr3d, axis=(0, 2), keepdims=True)
        self.assertEqual(np.asarray(by_tuple).shape, (1, 3, 1))
        np.testing.assert_allclose(
            self._eval_all(engine, by_tuple),
            np.sum(np.arange(24, dtype=float).reshape(2, 3, 4), axis=(0, 2), keepdims=True),
        )

    def test_symbolic_cumsum_rejects_unsupported_arguments_instead_of_dropping_them(self):
        # Regression (MXR-080-1567): cumsum was a lambda taking *args/**kwargs and forwarding only
        # `axis`, so dtype=, out=, and any unknown argument were silently discarded.
        engine = SymbolicEngine()
        arr = engine.asarray([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(NotImplementedError):
            engine.cumsum(arr, dtype=np.float32)
        with self.assertRaises(NotImplementedError):
            engine.cumsum(arr, out=np.empty(4))
        with self.assertRaises(TypeError):
            engine.cumsum(arr, bogus_argument=1)
        # the supported arguments still behave exactly as np.cumsum defines them
        raw = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        flat = engine.cumsum(arr)
        np.testing.assert_allclose(self._eval_all(engine, flat), np.cumsum(raw))
        by_axis = engine.cumsum(arr, axis=1)
        np.testing.assert_allclose(self._eval_all(engine, by_axis), np.cumsum(raw, axis=1))

    def test_symbolic_tuple_reduction_normalizes_negative_axes_against_original_rank(self):
        # Regression (MXR-080-1566): the tuple fold sorted the RAW axes and reduced them
        # sequentially, so a negative axis resolved against the already-shrunken rank. Reducing
        # (2,3,4) over (-1,-2) returned shape (3,) / [15,19,23] instead of numpy's (2,) / [11,23].
        engine = SymbolicEngine()
        raw = np.arange(24, dtype=float).reshape(2, 3, 4)
        arr = engine.asarray(raw)
        for axes in [(-1, -2), (-2, -1), (0, -1), (0, 2), (-3, -2, -1)]:
            expected = np.max(raw, axis=axes)
            got = np.asarray(engine.max(arr, axis=axes))
            self.assertEqual(got.shape, expected.shape, axes)
            np.testing.assert_allclose(self._eval_all(engine, got), expected, err_msg=str(axes))
        # keepdims must reinsert the same axes the reduction actually removed.
        kept = engine.sum(arr, axis=(-1, -3), keepdims=True)
        self.assertEqual(np.asarray(kept).shape, (1, 3, 1))
        np.testing.assert_allclose(self._eval_all(engine, kept), np.sum(raw, axis=(-1, -3), keepdims=True))

    def test_symbolic_tuple_reduction_rejects_duplicate_and_out_of_range_axes(self):
        # Regression (MXR-080-1566): duplicate axes were accepted and reduced two DIFFERENT
        # dimensions ((0,0) on a (2,3,4) array returned shape (4,)); numpy raises instead.
        engine = SymbolicEngine()
        arr = engine.asarray(np.arange(24, dtype=float).reshape(2, 3, 4))
        for axes in [(0, 0), (-1, 2), (1, -2)]:
            with self.assertRaises(ValueError):
                engine.sum(arr, axis=axes)
        for axes in [(1, 5), (0, -4)]:
            with self.assertRaises(np.exceptions.AxisError):
                engine.sum(arr, axis=axes)

    def test_symbolic_max_and_logsumexp_keepdims_retains_reduced_axes(self):
        # Regression (MXR-080-0150): keepdims was silently ignored identically for max and
        # logsumexp, not just sum.
        engine = SymbolicEngine()
        arr2d = engine.asarray([[1.0, 2.0], [3.0, 4.0]])

        max_kept = engine.max(arr2d, axis=1, keepdims=True)
        self.assertEqual(np.asarray(max_kept).shape, (2, 1))
        np.testing.assert_allclose(self._eval_all(engine, max_kept), [[2.0], [4.0]])

        lse_kept = engine.logsumexp(arr2d, axis=1, keepdims=True)
        self.assertEqual(np.asarray(lse_kept).shape, (2, 1))
        np.testing.assert_allclose(
            self._eval_all(engine, lse_kept),
            np.log(np.exp(np.asarray([[1.0, 2.0], [3.0, 4.0]])).sum(axis=1, keepdims=True)),
        )

    def test_symbolic_sum_and_max_initial_seeds_each_reduction_group(self):
        # Regression (MXR-080-0150): initial was silently ignored -- a symbolic sum/max with an
        # explicit initial seed returned the same result as without one, unlike NumPy, which
        # seeds EACH reduction group with it (not just a single global add).
        engine = SymbolicEngine()
        arr1d = engine.asarray([1.0, 2.0, 3.0])
        self.assertEqual(float(engine.sum(arr1d, initial=100.0).evaluate({})), 106.0)
        self.assertEqual(float(engine.max(arr1d, initial=100.0).evaluate({})), 100.0)
        # an initial below every element must not change the max
        self.assertEqual(float(engine.max(arr1d, initial=-100.0).evaluate({})), 3.0)

        arr2d = engine.asarray([[1.0, 2.0], [3.0, 4.0]])
        per_row = self._eval_all(engine, engine.sum(arr2d, axis=1, initial=10.0))
        np.testing.assert_allclose(per_row, [13.0, 17.0])  # 10 added to EACH row, not once globally

    def test_symbolic_max_where_requires_initial_like_numpy(self):
        # Regression (MXR-080-0150): where was silently ignored entirely. Max has no universal
        # identity, so -- matching np.max's own restriction -- using where without initial must
        # raise rather than silently reduce an under-specified group.
        engine = SymbolicEngine()
        arr1d = engine.asarray([1.0, 2.0, 3.0])
        mask = np.array([True, False, True])
        with self.assertRaises(ValueError):
            engine.max(arr1d, where=mask)
        masked = engine.max(arr1d, where=mask, initial=-1.0e18)
        self.assertEqual(float(masked.evaluate({})), 3.0)  # only elements 0 and 2 (1.0, 3.0) considered

    def test_symbolic_sum_where_excludes_masked_elements_without_requiring_initial(self):
        # Sum's additive identity (0.0) always exists, so -- unlike max -- where alone (no
        # initial) is well-defined and must exclude masked-out elements from the total.
        engine = SymbolicEngine()
        arr1d = engine.asarray([1.0, 2.0, 3.0])
        mask = np.array([True, False, True])
        masked = engine.sum(arr1d, where=mask)
        self.assertEqual(float(masked.evaluate({})), 4.0)  # 1.0 + 3.0, excluding 2.0

    def test_symbolic_sum_rejects_explicit_dtype_but_accepts_the_default(self):
        # Regression (MXR-080-0150): dtype was silently ignored -- a symbolic sum with an
        # explicit dtype= behaved identically to one without, even though a real cast is not
        # meaningful for an exact expression tree with no numeric storage. It must now be
        # rejected explicitly (the "reject unsupported arguments" half of the audit's fix menu)
        # rather than silently accepted and ignored.
        engine = SymbolicEngine()
        arr1d = engine.asarray([1.0, 2.0, 3.0])
        with self.assertRaises(NotImplementedError):
            engine.sum(arr1d, dtype=np.float32)
        # the default (unset) dtype is a no-op, exactly like NumPy's own dtype=None default
        self.assertEqual(float(engine.sum(arr1d, dtype=None).evaluate({})), 6.0)

    def test_symbolic_max_and_logsumexp_reject_unsupported_reduction_kwargs(self):
        # Regression (MXR-080-0150): max and logsumexp caught the SAME **kwargs catch-all as sum,
        # so every unsupported argument was silently swallowed for them too. np.max itself has no
        # dtype parameter at all, and scipy.special.logsumexp (which NumpyEngine.logsumexp
        # forwards to directly) has none of dtype/initial/where -- so the symbolic engine
        # declares the identical, narrower parameter set and lets an unsupported keyword raise
        # Python's own TypeError, exactly as calling the real NumPy/SciPy function would.
        engine = SymbolicEngine()
        arr1d = engine.asarray([1.0, 2.0, 3.0])
        with self.assertRaises(TypeError):
            engine.max(arr1d, dtype=np.float32)
        for kwargs in ({"dtype": np.float32}, {"initial": 1.0}, {"where": np.array([True])}):
            with self.assertRaises(TypeError):
                engine.logsumexp(arr1d, **kwargs)

    def test_symbolic_logsumexp_all_negative_infinity_is_negative_infinity_not_nan(self):
        # Regression (MXR-080-0150), exact audit repro: the stable max-shifted LSE computes
        # m + log(sum(exp(x - m))) with m = max(x). When every input is -inf, m is ALSO -inf, so
        # every shifted term computes -inf - (-inf) = NaN (the classic inf-inf indeterminate
        # form), poisoning the whole reduction to NaN instead of the mathematically correct -inf
        # (every term has probability exactly zero in log-space, so the total probability is
        # zero and its log is -inf).
        engine = SymbolicEngine()
        neg_inf = engine.asarray([float("-inf"), float("-inf")])
        result = engine.logsumexp(neg_inf)
        self.assertEqual(float(result.evaluate({})), float("-inf"))
        self.assertFalse(np.isnan(float(result.evaluate({}))))

        # a MIXED input (not all -inf) must still work exactly as before -- only the genuinely
        # degenerate all-(-inf) case changes
        mixed = engine.asarray([float("-inf"), 2.0, 3.0])
        expected = float(np.logaddexp(np.logaddexp(-np.inf, 2.0), 3.0))
        self.assertAlmostEqual(float(engine.logsumexp(mixed).evaluate({})), expected, places=10)

    def test_symbolic_sum_max_logsumexp_empty_reduction_identities(self):
        # Regression (MXR-080-0150): the audit asked for the empty/all-impossible identities to
        # be defined explicitly, matching NumPy/SciPy: sum([]) is the additive identity 0.0;
        # max([]) has no identity and raises unless initial is given (in which case it returns
        # initial, unchanged from the pre-fix behavior for the no-initial case -- max([]) already
        # raised before this fix); logsumexp([]) is -inf (the log of a sum of zero terms).
        engine = SymbolicEngine()
        empty = engine.asarray(np.array([], dtype=object))
        self.assertEqual(float(engine.sum(empty).evaluate({})), 0.0)
        with self.assertRaises(ValueError):
            engine.max(empty)
        self.assertEqual(float(engine.max(empty, initial=-5.0).evaluate({})), -5.0)
        self.assertEqual(float(engine.logsumexp(empty).evaluate({})), float("-inf"))

    def test_symbolic_reductions_without_special_arguments_are_unaffected(self):
        # Negative control for MXR-080-0150: ordinary axis-only reductions (the overwhelmingly
        # common case, and the only case exercised before this fix) must return exactly the same
        # shapes and values as before -- keepdims/dtype/initial/where must be strictly additive.
        engine = SymbolicEngine()
        arr2d = engine.asarray([[1.0, 2.0], [3.0, 4.0]])

        plain_sum = engine.sum(arr2d, axis=1)
        self.assertEqual(np.asarray(plain_sum).shape, (2,))
        np.testing.assert_allclose(self._eval_all(engine, plain_sum), [3.0, 7.0])

        plain_max = engine.max(arr2d, axis=0)
        self.assertEqual(np.asarray(plain_max).shape, (2,))
        np.testing.assert_allclose(self._eval_all(engine, plain_max), [3.0, 4.0])

        # the pre-existing overflow-safety case (E3) must still be exact
        plain_lse = engine.logsumexp(engine.asarray([1000.0, 1000.0]))
        self.assertAlmostEqual(float(plain_lse.evaluate({})), 1000.0 + np.log(2.0))

    def test_symbolic_engine_traces_comparison_masks(self):
        engine = SymbolicEngine()
        x = engine.symbol("x")
        y = engine.symbol("y")

        mask = (x >= 0.0) & (y < 2.0)
        expr = engine.where(mask, x + y, x - y)

        self.assertAlmostEqual(engine.evaluate(expr, {"x": 1.0, "y": 1.5}), 2.5)
        self.assertAlmostEqual(engine.evaluate(expr, {"x": -1.0, "y": 1.5}), -2.5)

        with self.assertRaises(TypeError):
            bool(mask)

        diagnostics = engine.diagnostics(expr)
        self.assertEqual(diagnostics["symbols"], ("x", "y"))
        self.assertEqual(diagnostics["op_counts"]["where"], 1)
        self.assertEqual(diagnostics["op_counts"]["ge"], 1)
        self.assertEqual(diagnostics["op_counts"]["lt"], 1)
        self.assertEqual(diagnostics["op_counts"]["and"], 1)

        arr = engine.asarray([x, y])
        arr_mask = engine.logical_and(
            engine.greater_equal(arr, 1.0),
            engine.less_equal(arr, 2.0),
        )
        arr_expr = engine.where(arr_mask, arr, 0.0)
        np.testing.assert_allclose(
            np.asarray(engine.evaluate(arr_expr, {"x": 1.5, "y": 3.0}), dtype=float), np.asarray([1.5, 0.0])
        )

    def test_symbolic_evaluate_and_or_not_where_are_elementwise_for_vector_bindings(self):
        # Regression (MXR-080-0152), exact audit repro: a SINGLE expression tree evaluated with
        # an array-BOUND symbol used to convert "and"/"or"/"where" conditions with Python's
        # bool(), which raises numpy's "truth value of an array with more than one element is
        # ambiguous" for any >1-element array -- so a vector-valued expression could not use
        # these logical ops at all, despite the engine's promised elementwise semantics. This is
        # distinct from test_symbolic_engine_traces_comparison_masks's array case above: that
        # test evaluates an ARRAY OF SEPARATE per-element expressions (each bound to its own
        # scalar); this evaluates ONE expression tree whose symbol is bound to array-valued data.
        engine = SymbolicEngine()
        x = engine.symbol("x")
        y = engine.symbol("y")
        mask = (x >= 0.0) & (y < 2.0)
        expr = engine.where(mask, x + y, x - y)

        # four rows, cross-checked against an independent raw-numpy computation of the identical
        # mask-and-select formula, so this verifies genuine per-element correctness (not just
        # "the call doesn't crash") without relying on hand-computed expected literals
        xv = np.array([1.0, -1.0, 5.0, -5.0])
        yv = np.array([1.5, 1.5, 5.0, 0.5])
        mask_np = (xv >= 0.0) & (yv < 2.0)
        # confirm the mask genuinely varies across the array -- not a degenerate all-one-value case
        self.assertIn(True, mask_np.tolist())
        self.assertIn(False, mask_np.tolist())
        result = np.asarray(engine.evaluate(expr, {"x": xv, "y": yv}), dtype=float)
        np.testing.assert_allclose(result, np.where(mask_np, xv + yv, xv - yv))

        and_expr = engine.logical_and(x, y)
        np.testing.assert_array_equal(
            engine.evaluate(and_expr, {"x": np.array([True, False, True]), "y": np.array([True, True, False])}),
            [True, False, False],
        )
        or_expr = engine.logical_or(x, y)
        np.testing.assert_array_equal(
            engine.evaluate(or_expr, {"x": np.array([True, False, True]), "y": np.array([True, True, False])}),
            [True, True, True],
        )
        not_expr = engine.logical_not(x)
        np.testing.assert_array_equal(
            engine.evaluate(not_expr, {"x": np.array([True, False, True])}), [False, True, False]
        )

    def test_symbolic_where_selects_elementwise_for_plain_list_bindings(self):
        # Regression (MXR-080-1565), exact audit repro: vector detection tested only
        # isinstance(np.ndarray), so a condition bound to an ordinary Python list took the scalar
        # path -- bool([True, False]) is True (a nonempty list), so where(c, [1,2], [3,4])
        # returned [1, 2] wholesale instead of selecting [1, 4] elementwise.
        engine = SymbolicEngine()
        c = engine.symbol("c")
        x = engine.symbol("x")
        y = engine.symbol("y")
        expr = engine.where(c, x, y)

        from_list = np.asarray(expr.evaluate({"c": [True, False], "x": [1, 2], "y": [3, 4]}))
        np.testing.assert_array_equal(from_list, [1, 4])
        from_array = np.asarray(
            expr.evaluate({"c": np.array([True, False]), "x": np.array([1, 2]), "y": np.array([3, 4])})
        )
        np.testing.assert_array_equal(from_list, from_array)

        # tuples are array-likes too, and the same normalization makes and/or/not elementwise
        np.testing.assert_array_equal(
            np.asarray(engine.logical_and(x, y).evaluate({"x": (True, False, True), "y": (True, True, False)})),
            [True, False, False],
        )
        np.testing.assert_array_equal(
            np.asarray(engine.logical_not(x).evaluate({"x": [True, False, True]})), [False, True, False]
        )

    def test_symbolic_where_does_not_evaluate_the_unselected_scalar_branch(self):
        # Regression (MXR-080-1565): both branches were evaluated before the condition selected
        # one, so guarding an undefined case -- the main reason to write a conditional -- did not
        # work: the unselected 1/z branch raised ZeroDivisionError at z=0 anyway.
        engine = SymbolicEngine()
        c = engine.symbol("c")
        z = engine.symbol("z")
        guarded = engine.where(c, engine.constant(0.0), engine.constant(1.0) / z)

        self.assertEqual(guarded.evaluate({"c": True, "z": 0.0}), 0.0)
        self.assertEqual(guarded.evaluate({"c": False, "z": 4.0}), 0.25)
        with self.assertRaises(ZeroDivisionError):
            guarded.evaluate({"c": False, "z": 0.0})  # genuinely selected: must still raise

        # a vector condition still selects elementwise (np.where semantics, both branches evaluated)
        both = engine.where(c, z + 1.0, z - 1.0)
        np.testing.assert_allclose(
            np.asarray(both.evaluate({"c": np.array([True, False]), "z": np.array([10.0, 10.0])}), dtype=float),
            [11.0, 9.0],
        )

    def test_symbolic_evaluate_and_or_not_where_scalar_case_is_unaffected(self):
        # Negative control for MXR-080-0152: the scalar/0-d condition path must behave EXACTLY
        # as before the fix -- same values AND the same plain Python types, not just "close
        # enough" (the fix branches on ndim>0 specifically so this path is untouched code).
        engine = SymbolicEngine()
        x = engine.symbol("x")
        y = engine.symbol("y")
        mask = (x >= 0.0) & (y < 2.0)
        expr = engine.where(mask, x + y, x - y)

        result = expr.evaluate({"x": 1.0, "y": 1.5})
        self.assertEqual(result, 2.5)
        self.assertIs(type(result), float)

        self.assertIs(engine.logical_and(x, y).evaluate({"x": True, "y": False}), False)
        self.assertIs(engine.logical_or(x, y).evaluate({"x": False, "y": True}), True)
        self.assertIs(engine.logical_not(x).evaluate({"x": True}), False)

        # a 0-d array (ndim == 0, as opposed to ndim > 0) is scalar-like and must not raise either
        result_0d = expr.evaluate({"x": np.array(1.0), "y": np.array(1.5)})
        self.assertAlmostEqual(float(result_0d), 2.5)

    def test_symbolic_transcendental_and_minmax_ops_accept_array_bindings(self):
        # Regression (MXR-080-1564): evaluate() advertises array-valued symbol bindings, but most
        # ops were scalar `math` functions / builtin max/min -- log and gammaln raised
        # "only 0-dimensional arrays can be converted to Python scalars", maximum and clip raised
        # numpy's ambiguous-truth error -- so array-bound numeric math was unusable outright.
        engine = SymbolicEngine()
        x = engine.symbol("x")
        y = engine.symbol("y")
        xv = np.array([0.25, 1.0, 2.5, 4.0])
        yv = np.array([0.5, 2.0, 3.0, 1.5])
        values = {"x": xv, "y": yv}

        # each cross-checked against the independent raw-numpy/scipy computation, not a literal
        cases = [
            (engine.log(x), np.log(xv)),
            (engine.exp(x), np.exp(xv)),
            (engine.sqrt(x), np.sqrt(xv)),
            (engine.floor(x), np.floor(xv)),
            (engine.abs(engine.symbol("x") - 2.0), np.abs(xv - 2.0)),
            (engine.gammaln(x), scipy.special.gammaln(xv)),
            (engine.digamma(x), scipy.special.digamma(xv)),
            (engine.erf(x), scipy.special.erf(xv)),
            (engine.betaln(x, y), scipy.special.betaln(xv, yv)),
            (engine.maximum(x, 1.0), np.maximum(xv, 1.0)),
            (engine.maximum(x, y), np.maximum(xv, yv)),
            (engine.clip(x, 0.5, 3.0), np.clip(xv, 0.5, 3.0)),
        ]
        for expr, expected in cases:
            got = np.asarray(expr.evaluate(values), dtype=float)
            np.testing.assert_allclose(got, np.asarray(expected, dtype=float), err_msg=str(expr))

        # boolean predicates stay elementwise too
        nan_values = {"x": np.array([1.0, np.nan, np.inf])}
        np.testing.assert_array_equal(engine.isnan(x).evaluate(nan_values), [False, True, False])
        np.testing.assert_array_equal(engine.isinf(x).evaluate(nan_values), [False, False, True])

    def test_symbolic_numeric_ops_scalar_results_are_unchanged(self):
        # Negative control for MXR-080-1564: the scalar path must keep its exact pre-fix Python
        # types, which the numpy equivalents do NOT reproduce (np.floor returns a float, np.isnan
        # a numpy bool_). The dispatch branches on ndim>0 specifically so this path is untouched.
        engine = SymbolicEngine()
        x = engine.symbol("x")
        y = engine.symbol("y")

        floored = engine.floor(x).evaluate({"x": 2.7})
        self.assertIs(type(floored), int)
        self.assertEqual(floored, 2)
        self.assertIs(engine.isnan(x).evaluate({"x": 1.0}), False)
        self.assertIs(engine.isinf(x).evaluate({"x": float("inf")}), True)
        self.assertAlmostEqual(engine.log(x).evaluate({"x": 2.0}), math.log(2.0))
        self.assertAlmostEqual(engine.maximum(x, y).evaluate({"x": 1.0, "y": 3.0}), 3.0)
        self.assertAlmostEqual(engine.clip(x, 0.0, 1.0).evaluate({"x": 2.0}), 1.0)
        self.assertAlmostEqual(
            engine.betaln(x, y).evaluate({"x": 2.0, "y": 3.0}), float(scipy.special.betaln(2.0, 3.0))
        )
        # 0-d arrays are scalar-like and keep taking the scalar path without raising
        self.assertAlmostEqual(float(engine.log(x).evaluate({"x": np.array(2.0)})), math.log(2.0))

    def test_symbolic_payloads_dispatch_through_arithmetic(self):
        from mixle.engines import SYMBOLIC_ENGINE

        # a scalar node and an object array of nodes both recover the symbolic
        # engine through engine_of, while ordinary numpy arrays stay numpy
        node = SymbolicExpression.symbol("x")
        self.assertIs(engine_of(node), SYMBOLIC_ENGINE)
        arr = SYMBOLIC_ENGINE.asarray(["x", "y"])
        self.assertIs(engine_of(arr), SYMBOLIC_ENGINE)
        self.assertEqual(engine_of(np.array([1.0, 2.0])).name, "numpy")

        # mixle.engines.arithmetic dispatches symbolic inputs to the symbolic engine
        expr = ar.log(ar.exp(node) + 1.0)
        self.assertIsInstance(expr, SymbolicExpression)
        self.assertAlmostEqual(float(SYMBOLIC_ENGINE.evaluate(expr, {"x": 0.0})), np.log(2.0))

    def test_is_symbolic_payload_rejects_mixed_ownership_numeric_first(self):
        # Regression (MXR-080-0151), exact audit repro: an object array used to be classified
        # by x.flat[0] alone, so a mixed-ownership array's routing depended entirely on which
        # kind happened to sit at index 0. An array whose first element is NOT symbolic used to
        # silently route its embedded symbolic element through NumPy (which was never built to
        # handle SymbolicExpression objects).
        from mixle.engines import is_symbolic_payload

        class Unrelated:
            pass

        mixed = np.empty(2, dtype=object)
        mixed[0] = Unrelated()
        mixed[1] = SymbolicExpression.symbol("x")
        with self.assertRaises(TypeError):
            is_symbolic_payload(mixed)
        with self.assertRaises(TypeError):
            engine_of(mixed)  # end-to-end through mixle.engines' own dispatch

    def test_is_symbolic_payload_rejects_mixed_ownership_symbolic_first(self):
        # Regression (MXR-080-0151), the reverse ordering: an array whose first element IS
        # symbolic used to silently route its unrelated element through the symbolic engine.
        # Both orderings must be caught -- not just whichever one was "lucky" before.
        from mixle.engines import is_symbolic_payload

        class Unrelated:
            pass

        mixed = np.empty(2, dtype=object)
        mixed[0] = SymbolicExpression.symbol("x")
        mixed[1] = Unrelated()
        with self.assertRaises(TypeError):
            is_symbolic_payload(mixed)
        with self.assertRaises(TypeError):
            engine_of(mixed)

    def test_is_symbolic_payload_uniform_arrays_are_unaffected(self):
        # Negative control: a genuinely uniform object array -- all-symbolic or all-other --
        # must still classify exactly as before. Only genuinely mixed ownership is new.
        from mixle.engines import SYMBOLIC_ENGINE, is_symbolic_payload

        class Unrelated:
            pass

        all_symbolic = np.array([SymbolicExpression.symbol("x"), SymbolicExpression.symbol("y")], dtype=object)
        self.assertTrue(is_symbolic_payload(all_symbolic))
        self.assertIs(engine_of(all_symbolic), SYMBOLIC_ENGINE)

        all_other = np.array([Unrelated(), Unrelated()], dtype=object)
        self.assertFalse(is_symbolic_payload(all_other))

        # an empty object array has no elements to disagree, so it is not a symbolic payload
        self.assertFalse(is_symbolic_payload(np.array([], dtype=object)))

    def test_direct_engine_rejects_invalid_pysp_engine_tag(self):
        # Regression (MXR-080-0120): __pysp_engine__ used to be trusted at face value, so an
        # object tagging itself with a plain string (or any non-ComputeEngine value) made
        # engine_of silently return that string as if it were a legitimate engine.
        class Junk:
            __pysp_engine__ = "junk"

        with self.assertRaises(TypeError):
            engine_of(Junk())

    def test_direct_engine_accepts_a_real_computeengine_tag(self):
        # Positive control for the above: a genuine ComputeEngine tag is still trusted -- this
        # is the exact mechanism SymbolicExpression relies on to route to SYMBOLIC_ENGINE.
        class Tagged:
            __pysp_engine__ = NUMPY_ENGINE

        self.assertIs(engine_of(Tagged()), NUMPY_ENGINE)

    def test_engine_of_resolves_most_specific_registered_subclass(self):
        # Regression (MXR-080-0120): registry lookup used to scan _ARRAY_ENGINE_REGISTRY in
        # insertion order and return the first isinstance match, so an np.ndarray SUBCLASS
        # registered to a different engine was still shadowed by the earlier generic
        # np.ndarray -> NumpyEngine rule. Resolution must walk the MRO and prefer the most
        # specific registered ancestor, regardless of registration order.
        import mixle.engines as engines_module
        from mixle.engines import SYMBOLIC_ENGINE, register_array_type

        class MyArraySubclass(np.ndarray):
            pass

        register_array_type(MyArraySubclass, SYMBOLIC_ENGINE)
        self.addCleanup(engines_module._ARRAY_ENGINE_REGISTRY.pop, MyArraySubclass, None)

        x = np.asarray([1.0, 2.0]).view(MyArraySubclass)
        self.assertIs(engine_of(x), SYMBOLIC_ENGINE)
        # the generic rule is unaffected for plain (non-subclassed) ndarrays
        self.assertIsInstance(engine_of(np.asarray([1.0, 2.0])), NumpyEngine)

    def test_registration_requires_a_type_engine_and_explicit_collision_policy(self):
        import mixle.engines as engines_module
        from mixle.engines import SYMBOLIC_ENGINE, register_array_type

        class Payload:
            pass

        with self.assertRaises(TypeError):
            register_array_type("not a type", NUMPY_ENGINE)
        with self.assertRaises(TypeError):
            register_array_type(Payload, "not an engine")
        with self.assertRaises(ValueError):
            register_array_type(object, NUMPY_ENGINE)
        register_array_type(Payload, NUMPY_ENGINE)
        self.addCleanup(engines_module._ARRAY_ENGINE_REGISTRY.pop, Payload, None)
        with self.assertRaises(ValueError):
            register_array_type(Payload, SYMBOLIC_ENGINE)
        register_array_type(Payload, SYMBOLIC_ENGINE, override=True)
        self.assertIs(engine_of(Payload()), SYMBOLIC_ENGINE)

    def test_cyclic_containers_raise_stable_boundary_errors(self):
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaisesRegex(ValueError, "cyclic container"):
            engine_of(cyclic)
        with self.assertRaisesRegex(ValueError, "cyclic container"):
            to_numpy(cyclic)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_arithmetic_dispatch_discovers_keyword_and_mixed_engine_operands(self):
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
        converted = ar.asarray(x=tensor)
        self.assertIsInstance(converted, torch.Tensor)
        with self.assertRaisesRegex(TypeError, "mixed compute engines"):
            ar.where(cond=np.array([True, False]), x=tensor, y=tensor)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_arithmetic_max_discovers_keyword_operands(self):
        # Residual of MXR-080-1526: `max` is the one dispatched wrapper in this module written by
        # hand (for the scalar fast path) rather than generated by _dispatch, and it kept dispatching
        # on `args` alone after _dispatch was fixed. With no positional operand, engine_of saw an
        # empty tuple and fell back to the default NumPy engine, so `max(x=tensor)` sent a Torch
        # tensor to np.max -- which forwards to the tensor's own .max with NumPy's `axis=`/`out=`
        # keywords and raises, instead of the value the identical positional call returns.
        tensor = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float64)
        by_keyword = ar.max(x=tensor)
        self.assertIsInstance(by_keyword, torch.Tensor)
        self.assertEqual(float(by_keyword), 5.0)
        self.assertEqual(float(ar.max(tensor)), float(by_keyword))
        # Negative controls: the scalar builtins fast path and plain NumPy dispatch are untouched.
        self.assertEqual(ar.max(3, 7), 7)
        self.assertEqual(float(ar.max(np.array([1.0, 4.0]))), 4.0)

    def test_compute_engine_rejects_noncallable_required_operation(self):
        with self.assertRaisesRegex(TypeError, "callable.*log"):

            class InvalidEngine(NumpyEngine):
                log = 42

    def test_to_numpy_flat_array_still_converts_correctly(self):
        # Negative control for the recursive to_numpy fix (MXR-080-0123): a plain, non-nested
        # array must still convert exactly as before.
        x = np.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(to_numpy(x), x)

    def test_to_numpy_plain_nested_list_without_engine_values_is_unchanged(self):
        # Negative control: nested Python data with no engine-owned leaf inside is still handed
        # to the resolved engine as ONE unit (a single 2D array), not walked leaf by leaf.
        out = to_numpy([[1.0, 2.0], [3.0, 4.0]])
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_allclose(out, [[1.0, 2.0], [3.0, 4.0]])

    def test_to_numpy_recurses_ragged_list_of_arrays(self):
        # Regression (MXR-080-0123): to_numpy used to resolve ONE engine for the whole container
        # and hand it the entire container as-is -- np.asarray on a ragged list of differently
        # shaped arrays raises ValueError. Each leaf must convert independently, preserving the
        # list structure (this reproduces the finding's "ragged list" failure mode without
        # needing torch: np.asarray(ragged) already raises for the same structural reason
        # np.asarray(list_of_torch_tensors) does).
        ragged = [np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0, 3.0])]
        with self.assertRaises(ValueError):
            np.asarray(ragged)  # confirms the old single-shot conversion really would fail

        out = to_numpy(ragged)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        np.testing.assert_allclose(out[0], [1.0, 2.0])
        np.testing.assert_allclose(out[1], [1.0, 2.0, 3.0])

    def test_to_numpy_recurses_dict_values(self):
        # Regression (MXR-080-0123): to_numpy used to hand the whole dict to a single engine's
        # to_numpy (np.asarray on a dict wraps it in a useless 0-d object array instead of
        # touching its values); dict VALUES must each convert in place, preserving keys.
        payload = {"a": np.asarray([1.0, 2.0]), "b": np.asarray([3.0, 4.0, 5.0])}
        out = to_numpy(payload)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"a", "b"})
        np.testing.assert_allclose(out["a"], [1.0, 2.0])
        np.testing.assert_allclose(out["b"], [3.0, 4.0, 5.0])

    def test_to_numpy_preserves_tuple_container_type(self):
        payload = (np.asarray([1.0]), np.asarray([2.0, 3.0]))
        out = to_numpy(payload)
        self.assertIsInstance(out, tuple)
        np.testing.assert_allclose(out[0], [1.0])
        np.testing.assert_allclose(out[1], [2.0, 3.0])

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_engine_recovery_and_arithmetic(self):
        x = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        eng = engine_of(x)
        self.assertIsInstance(eng, TorchEngine)
        y = ar.sqrt(x)
        self.assertTrue(isinstance(y, torch.Tensor))
        self.assertTrue(torch.allclose(y, torch.sqrt(x)))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_engine_precision_policy(self):
        engine = TorchEngine(dtype="float32")
        x = engine.asarray([1.0, 2.0])
        y = engine.asarray([1, 2])

        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.int64)
        self.assertEqual(engine.zeros(2).dtype, torch.float32)
        self.assertEqual(engine.arange(0.0, 1.0, 0.25).dtype, torch.float32)
        self.assertEqual(engine.with_precision("float64").dtype, torch.float64)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_asarray_rejects_complex_input(self):
        # Regression (MXR-080-0148), exact audit repro: complex NumPy input used to fall through the
        # non-floating/non-Boolean fallback straight to torch.int64 -- engine.asarray([1+2j]) silently
        # returned tensor([1]), destroying the imaginary component AND collapsing the real part to an
        # integer, with only Torch's own internal cast warning (never a mixle-raised error) along the way.
        engine = TorchEngine(dtype="float64")
        with self.assertRaises(ValueError):
            engine.asarray([1 + 2j])
        with self.assertRaises(ValueError):
            engine.asarray(np.asarray([1 + 2j], dtype=np.complex128))
        with self.assertRaises(ValueError):
            engine.asarray(np.asarray([1 + 2j], dtype=np.complex64))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_asarray_accepts_complex_input_with_explicit_dtype(self):
        # A caller that passes dtype= explicitly is opting into a specific policy with full knowledge --
        # that path already bypasses the kind-based inference entirely (dtype is not None short-circuits
        # before any arr.dtype.kind check), so it must keep working and round-trip both components exactly.
        engine = TorchEngine(dtype="float64")
        out = engine.asarray([1 + 2j], dtype=torch.complex128)
        self.assertEqual(out.dtype, torch.complex128)
        self.assertEqual(complex(out[0]), 1 + 2j)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_asarray_real_int_float_bool_input_unaffected(self):
        # Negative control: ordinary real-valued NumPy kinds (the vast majority of callers) must still
        # convert exactly as before -- the complex rejection must not touch any other kind's dispatch.
        engine = TorchEngine(dtype="float64")
        ints = engine.asarray([1, 2, 3])
        self.assertEqual(ints.dtype, torch.int64)
        self.assertTrue(torch.equal(ints, torch.tensor([1, 2, 3], dtype=torch.int64)))
        floats = engine.asarray([1.0, 2.5])
        self.assertEqual(floats.dtype, torch.float64)
        self.assertTrue(torch.allclose(floats, torch.tensor([1.0, 2.5], dtype=torch.float64)))
        bools = engine.asarray([True, False])
        self.assertEqual(bools.dtype, torch.bool)
        self.assertTrue(torch.equal(bools, torch.tensor([True, False])))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_engine_accepts_concrete_non_floating_dtype_as_no_override(self):
        # Regression (MXR-080-0122): a concrete non-floating torch.dtype (as engine discovery
        # reads off an integer/Boolean tensor's own storage) must fall back to the engine's
        # default float policy rather than raising.
        engine = TorchEngine(dtype=torch.int64)
        self.assertTrue(engine.dtype.is_floating_point)
        self.assertFalse(engine.dtype_explicit)
        engine_bool = TorchEngine(dtype=torch.bool)
        self.assertTrue(engine_bool.dtype.is_floating_point)
        self.assertFalse(engine_bool.dtype_explicit)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_engine_rejects_invalid_named_precision(self):
        # Negative control: only a concrete non-floating *dtype object* is treated as "no
        # override" above -- an invalid named precision string is still a genuine caller
        # mistake and must still raise (unchanged from before the fix).
        with self.assertRaises(ValueError):
            TorchEngine(dtype="int64")

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_mps_engine_falls_back_to_float32(self):
        # MPS has no float64; an UNSET/default dtype must still downgrade silently so torch-ready
        # families run on Apple-silicon GPUs without every caller having to opt in explicitly -- this
        # is the low-stakes "no opinion was expressed" case, not the MXR-080-0149 bug (see below).
        # torch.device("mps") is constructible regardless of whether MPS is actually available, so this
        # exercises the policy on any host (incl. CPU-only CI).
        mps = TorchEngine(device="mps")
        self.assertEqual(mps.dtype, torch.float32)
        self.assertEqual(mps.accumulator_dtype, torch.float32)
        self.assertFalse(mps.dtype_explicit)
        # CPU/CUDA keep full precision
        self.assertEqual(TorchEngine(device="cpu").accumulator_dtype, torch.float64)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_mps_engine_rejects_explicit_float64(self):
        # Regression (MXR-080-0149): an EXPLICIT float64 request on MPS used to be silently replaced
        # with float32 (same code path exercised via TorchEngine(device="mps", dtype="float64") in the
        # old version of test_mps_engine_falls_back_to_float32 above), so a caller relying on float64 for
        # precision allocation or a scientific tolerance could receive a policy that cannot meet it with
        # no infeasibility signal at all. It must now fail closed instead. torch.device("mps") is
        # constructible on any host, so this needs no real MPS hardware.
        with self.assertRaises(ValueError):
            TorchEngine(device="mps", dtype="float64")
        with self.assertRaises(ValueError):
            TorchEngine(device="mps", dtype=torch.float64)
        with self.assertRaises(ValueError):
            TorchEngine(device="mps").with_precision("float64")

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_mps_engine_accepts_explicit_float32(self):
        # Negative control: an explicit request for a precision MPS DOES support must keep working --
        # only the unmeetable explicit float64 request is rejected.
        engine = TorchEngine(device="mps", dtype="float32")
        self.assertEqual(engine.dtype, torch.float32)
        self.assertTrue(engine.dtype_explicit)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_explicit_float64_still_works_on_float64_capable_device(self):
        # Negative control: explicit float64 on a device that DOES support it (e.g. CPU) is unaffected.
        engine = TorchEngine(device="cpu", dtype="float64")
        self.assertEqual(engine.dtype, torch.float64)
        self.assertTrue(engine.dtype_explicit)
        self.assertEqual(engine.with_precision("float64").dtype, torch.float64)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_sum_promotes_float32_input_to_accumulator_dtype(self):
        # Regression: TorchEngine.sum was a bare torch.sum passthrough with no accumulator_dtype
        # promotion (unlike NumpyEngine.sum, which already promotes), so a float32-precision fit on
        # this engine accumulated sufficient statistics in float32 and silently drifted on large N --
        # exactly the risk accumulator_dtype exists to guard against. Must now match numpy exactly.
        from mixle.engines.numpy_engine import NumpyEngine

        raw = np.ones(5_000_000, dtype=np.float32) * 0.1
        engine = TorchEngine(device="cpu", dtype="float32")
        result = engine.sum(engine.asarray(raw))
        self.assertEqual(result.dtype, torch.float64)
        # compare against numpy's accumulator_dtype-promoted sum of the SAME float32-rounded values --
        # the true reference (this many np.float32(0.1) values do not sum to exactly 500000.0 because
        # 0.1 has no exact float32 representation; float32 *accumulation* of that error compounds far
        # worse than float64 accumulation of the same per-element bias).
        reference = float(NumpyEngine(dtype="float32").sum(raw))
        self.assertAlmostEqual(float(result), reference, places=3)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_sum_respects_an_explicit_dtype_override(self):
        engine = TorchEngine(device="cpu", dtype="float32")
        x = engine.asarray(np.ones(10, dtype=np.float32))
        result = engine.sum(x, dtype=torch.float32)
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_sum_matches_numpy_engine_accumulation_accuracy(self):
        from mixle.engines.numpy_engine import NumpyEngine

        ne = NumpyEngine(dtype="float32")
        te = TorchEngine(device="cpu", dtype="float32")
        raw = np.random.RandomState(0).randn(2_000_000).astype(np.float32) * 3.0 + 10.0
        true_sum = float(np.sum(raw.astype(np.float64)))
        np_sum = float(ne.sum(ne.asarray(raw)))
        torch_sum = float(te.sum(te.asarray(raw)))
        self.assertAlmostEqual(np_sum, true_sum, places=1)
        self.assertAlmostEqual(torch_sum, true_sum, places=1)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_max_tuple_axes_normalize_negatives_against_original_rank(self):
        # Regression (MXR-080-1562): the tuple fold sorted the raw axes, so negative axes were
        # applied against the SHRINKING rank instead of the original one. Reducing (2,3,4) over
        # (-1,-2) returned shape (3,) / [15,19,23]; numpy's contract is shape (2,) / [11,23].
        te = TorchEngine(device="cpu", dtype="float64")
        raw = np.arange(24.0).reshape(2, 3, 4)
        t = te.asarray(raw)
        for axes in [(-1, -2), (-2, -1), (0, -1), (0, 1), (1, 2), (-3, -2, -1)]:
            expected = np.max(raw, axis=axes)
            got = np.asarray(te.max(t, axis=axes).cpu())
            self.assertEqual(got.shape, expected.shape, axes)
            np.testing.assert_allclose(got, expected, err_msg=str(axes))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_max_tuple_axes_reject_duplicates_and_out_of_range(self):
        # Regression (MXR-080-1562): duplicate axes silently reduced two DIFFERENT dimensions
        # ((0,0) on a (2,3,4) tensor returned shape (4,)); numpy raises instead.
        te = TorchEngine(device="cpu", dtype="float64")
        t = te.asarray(np.arange(24.0).reshape(2, 3, 4))
        for axes in [(0, 0), (-1, 2), (1, -2)]:
            with self.assertRaises(ValueError):
                te.max(t, axis=axes)
        for axes in [(1, 5), (0, -4)]:
            with self.assertRaises(IndexError):
                te.max(t, axis=axes)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_mixed_engine_payload_fails(self):
        payload = (np.asarray([1.0]), torch.tensor([1.0]))
        with self.assertRaises(TypeError):
            engine_of(payload)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_mixed_device_torch_payload_fails(self):
        # Regression (MXR-080-0121): mixed-engine detection used to compare only the Python
        # class, so two Torch tensors on different devices passed as "homogeneous" and would
        # dispatch through whichever was encountered first. "meta" is a real, always-available
        # torch device (no GPU/MPS hardware required) that differs from the default "cpu".
        payload = (torch.tensor([1.0]), torch.tensor([1.0], device="meta"))
        with self.assertRaises(TypeError):
            engine_of(payload)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_mixed_precision_policy_torch_payload_fails(self):
        # Regression (MXR-080-0121): two Torch tensors with explicitly different floating
        # dtypes used to pass the homogeneity check (same TorchEngine class, precision ignored).
        payload = (torch.tensor([1.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float64))
        with self.assertRaises(TypeError):
            engine_of(payload)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_homogeneous_torch_payload_still_resolves(self):
        # Negative control: matching device + dtype across a nested payload must still resolve
        # cleanly -- the stricter check must not false-positive on genuinely homogeneous data.
        payload = (torch.tensor([1.0], dtype=torch.float64), {"x": torch.tensor([2.0, 3.0], dtype=torch.float64)})
        eng = engine_of(payload)
        self.assertIsInstance(eng, TorchEngine)
        self.assertEqual(str(eng.device), "cpu")

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_boolean_mask_with_float_data_does_not_falsely_conflict(self):
        # Interaction between MXR-080-0121 and MXR-080-0122: a Boolean mask tensor alongside
        # explicit float32 data (the ar.where(mask, a, b) shape, extremely common) must NOT be
        # treated as a precision-policy conflict -- the mask's engine carries no real dtype
        # opinion of its own, so it must not clash with the data arrays' genuine float32 policy.
        mask = torch.tensor([True, False, True])
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        out = ar.where(mask, a, b)
        self.assertTrue(torch.equal(out, torch.tensor([1.0, 20.0, 3.0], dtype=torch.float32)))

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_engine_of_integer_tensor_no_longer_raises(self):
        # Regression (MXR-080-0122), exact audit repro: engine_of(torch.tensor([1])) used to
        # construct TorchEngine(dtype=torch.int64), whose constructor only accepted floating
        # dtypes, raising ValueError and breaking indexing, masks, and categorical dispatch.
        eng = engine_of(torch.tensor([1, 2, 3]))
        self.assertIsInstance(eng, TorchEngine)
        self.assertTrue(eng.dtype.is_floating_point)  # engine keeps its own float POLICY...
        self.assertFalse(eng.dtype_explicit)  # ...but it is a default, not the tensor's own dtype

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_engine_of_bool_tensor_no_longer_raises(self):
        # Regression (MXR-080-0122), Boolean equivalent of the audit repro.
        eng = engine_of(torch.tensor([True, False, True]))
        self.assertIsInstance(eng, TorchEngine)
        self.assertFalse(eng.dtype_explicit)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_engine_of_floating_tensor_dtype_policy_unchanged(self):
        # Negative control: discovery from a genuinely floating tensor keeps tracking that
        # tensor's own dtype exactly as before the fix.
        eng32 = engine_of(torch.tensor([1.0], dtype=torch.float32))
        self.assertEqual(eng32.dtype, torch.float32)
        self.assertTrue(eng32.dtype_explicit)
        eng64 = engine_of(torch.tensor([1.0], dtype=torch.float64))
        self.assertEqual(eng64.dtype, torch.float64)
        self.assertTrue(eng64.dtype_explicit)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_indexing_and_masking_dispatch_through_fixed_discovery(self):
        # Regression (MXR-080-0122): integer index tensors and Boolean masks are ordinary,
        # extremely common Torch usage that engine_of used to break entirely.
        values = torch.tensor([10.0, 20.0, 30.0])
        idx = torch.as_tensor([0, 2])
        self.assertIsInstance(engine_of(idx), TorchEngine)  # discovery on the index itself
        self.assertTrue(torch.equal(values[idx], torch.tensor([10.0, 30.0])))

        mask = torch.tensor([True, False, True])
        self.assertIsInstance(engine_of(mask), TorchEngine)  # discovery on the mask itself
        self.assertTrue(torch.equal(values[mask], torch.tensor([10.0, 30.0])))

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_jax_engine_accepts_concrete_non_floating_dtype_as_no_override(self):
        # Regression: JaxEngine had the same dtype-crash bug class as pre-fix TorchEngine
        # (MXR-080-0122) -- a concrete non-floating NumPy dtype (as engine discovery reads off an
        # integer/Boolean JAX array's own storage; JAX dtypes are plain np.dtype instances) must fall
        # back to the engine's default float policy rather than raising.
        engine = JaxEngine(dtype=np.dtype("int64"))
        self.assertTrue(np.issubdtype(engine.dtype, np.floating))
        self.assertFalse(engine.dtype_explicit)
        engine_bool = JaxEngine(dtype=np.dtype("bool"))
        self.assertTrue(np.issubdtype(engine_bool.dtype, np.floating))
        self.assertFalse(engine_bool.dtype_explicit)

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_jax_engine_rejects_invalid_named_precision(self):
        # Negative control: only a concrete non-floating *dtype object* is treated as "no override"
        # above -- an invalid named precision string is still a genuine caller mistake and must still
        # raise (matches TorchEngine's identical precedent, unchanged from before the fix).
        with self.assertRaises(ValueError):
            JaxEngine(dtype="int64")

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_engine_of_integer_jax_array_no_longer_raises(self):
        # Regression, exact audit repro: engine_of(jnp.array([1, 2, 3])) used to unconditionally
        # construct JaxEngine(dtype=x.dtype) with the array's own integer dtype, whose constructor
        # (via normalize_numpy_dtype) only accepted floating dtypes, raising ValueError -- breaking
        # indexing (integer index arrays), masks (Boolean arrays), and categorical payload dispatch.
        eng = engine_of(jnp.array([1, 2, 3]))
        self.assertIsInstance(eng, JaxEngine)
        self.assertTrue(np.issubdtype(eng.dtype, np.floating))  # engine keeps its own float POLICY...
        self.assertFalse(eng.dtype_explicit)  # ...but it is a default, not the array's own dtype

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_engine_of_bool_jax_array_no_longer_raises(self):
        # Regression, Boolean equivalent of the audit repro.
        eng = engine_of(jnp.array([True, False, True]))
        self.assertIsInstance(eng, JaxEngine)
        self.assertFalse(eng.dtype_explicit)

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_engine_of_floating_jax_array_dtype_policy_unchanged(self):
        # Negative control: discovery from a genuinely floating JAX array keeps tracking that array's
        # own dtype exactly as before the fix.
        eng32 = engine_of(jnp.array([1.0], dtype=jnp.float32))
        self.assertEqual(eng32.dtype, np.dtype("float32"))
        self.assertTrue(eng32.dtype_explicit)

        # float64 is only actually achievable with jax_enable_x64 on (see JaxEngine's own module
        # docstring) -- scope the ambient config change to this test and restore it unconditionally.
        orig_x64 = bool(jax.config.jax_enable_x64)
        self.addCleanup(jax.config.update, "jax_enable_x64", orig_x64)
        jax.config.update("jax_enable_x64", True)
        eng64 = engine_of(jnp.array([1.0], dtype=jnp.float64))
        self.assertEqual(eng64.dtype, np.dtype("float64"))
        self.assertTrue(eng64.dtype_explicit)

    def test_jax_array_placement_reads_every_accessor_jax_versions_expose(self):
        # Regression (MXR-080-1561): engine discovery built JaxEngine(dtype=...) with no device, so
        # every JAX array resolved to the default CPU device regardless of where it really lived.
        # This half needs no jax installed -- placement recovery is plain attribute access, and it
        # must cope with the property (jax >= 0.4.27), the older method, and the devices() set.
        from mixle.engines.jax_engine import jax_array_placement

        class _Modern:
            device = "gpu:1"

        class _Legacy:
            def device(self):
                return "gpu:0"

        class _DevicesOnly:
            device = None

            def devices(self):
                return {"tpu:3"}

        class _Sharded:
            device = None

            def devices(self):
                return {"gpu:0", "gpu:1"}  # a sharding, not one device

        class _NoAccessor:
            pass

        self.assertEqual(jax_array_placement(_Modern()), "gpu:1")
        self.assertEqual(jax_array_placement(_Legacy()), "gpu:0")
        self.assertEqual(jax_array_placement(_DevicesOnly()), "tpu:3")
        # unknown placement falls back to the constructor default rather than guessing one device
        self.assertIsNone(jax_array_placement(_Sharded()))
        self.assertIsNone(jax_array_placement(_NoAccessor()))

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_engine_of_jax_array_preserves_its_actual_device(self):
        # Regression (MXR-080-1561): _direct_engine dropped the array's placement, so a GPU/TPU
        # array was relabeled as the default CPU device and _engines_compatible (which compares
        # str(engine.device)) could not tell two placements apart.
        devices = jax.devices()
        for device in devices:
            arr = jax.device_put(jnp.array([1.0, 2.0], dtype=jnp.float32), device)
            self.assertEqual(str(engine_of(arr).device), str(device))
        if len(devices) < 2:
            self.skipTest("only one JAX device available; no cross-device conflict to detect")
        a = jax.device_put(jnp.array([1.0], dtype=jnp.float32), devices[0])
        b = jax.device_put(jnp.array([1.0], dtype=jnp.float32), devices[1])
        with self.assertRaises(TypeError):
            engine_of((a, b))

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_jax_engine_rejects_explicit_float64_when_x64_is_disabled(self):
        # Regression (MXR-080-1559): an explicit dtype="float64" was silently rewritten to float32
        # when the ambient jax_enable_x64 was off, so a caller relying on float64 for a scientific
        # tolerance got a float32 policy with no signal the requirement went unmet. TorchEngine
        # already fails closed for the identical case (float64 on MPS); JAX now matches.
        orig_x64 = bool(jax.config.jax_enable_x64)
        self.addCleanup(jax.config.update, "jax_enable_x64", orig_x64)
        jax.config.update("jax_enable_x64", False)

        for requested in ("float64", np.dtype("float64"), np.float64):
            with self.assertRaises(ValueError):
                JaxEngine(dtype=requested)
        with self.assertRaises(ValueError):
            JaxEngine().with_precision("float64")

        # the unrequested default still accommodates quietly, and float32 is honored either way
        self.assertEqual(JaxEngine().dtype, np.dtype("float32"))
        self.assertFalse(JaxEngine().dtype_explicit)
        self.assertEqual(JaxEngine(dtype="float32").dtype, np.dtype("float32"))

        # with x64 actually enabled the explicit request is satisfiable and must be honored
        jax.config.update("jax_enable_x64", True)
        self.assertEqual(JaxEngine(dtype="float64").dtype, np.dtype("float64"))

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_boolean_mask_with_float_jax_data_does_not_falsely_conflict(self):
        # Interaction test mirroring test_boolean_mask_with_float_data_does_not_falsely_conflict for
        # Torch: a Boolean mask array alongside explicit float32 data (the ar.where(mask, a, b) shape,
        # extremely common) must NOT be treated as a precision-policy conflict -- the mask's engine
        # carries no real dtype opinion of its own, so it must not clash with the data arrays' genuine
        # float32 policy. This specifically needs jax_enable_x64=True: with x64 off, the only floating
        # dtype JAX can actually produce is float32, so an implicit-default mask engine and an explicit
        # float32 data engine would coincidentally compare equal even WITHOUT the dtype_explicit fix --
        # x64 on is what makes the mask engine's implicit default (float64) genuinely diverge from the
        # data's explicit float32, so this is the case that actually exercises the fix.
        orig_x64 = bool(jax.config.jax_enable_x64)
        self.addCleanup(jax.config.update, "jax_enable_x64", orig_x64)
        jax.config.update("jax_enable_x64", True)

        mask = jnp.array([True, False, True])
        a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        b = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)
        out = ar.where(mask, a, b)
        np.testing.assert_allclose(np.asarray(out), [1.0, 20.0, 3.0])

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_jax_indexing_and_masking_dispatch_through_fixed_discovery(self):
        # Regression: integer index arrays and Boolean masks are ordinary, extremely common JAX usage
        # that engine_of used to break entirely.
        values = jnp.array([10.0, 20.0, 30.0])
        idx = jnp.array([0, 2])
        self.assertIsInstance(engine_of(idx), JaxEngine)  # discovery on the index itself
        np.testing.assert_allclose(np.asarray(values[idx]), [10.0, 30.0])

        mask = jnp.array([True, False, True])
        self.assertIsInstance(engine_of(mask), JaxEngine)  # discovery on the mask itself
        np.testing.assert_allclose(np.asarray(values[mask]), [10.0, 30.0])

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_to_numpy_recurses_ragged_list_of_torch_tensors(self):
        # Regression (MXR-080-0123), exact audit scenario: a ragged list of Torch tensors used
        # to be handed whole to one engine's to_numpy, which falls through to
        # np.asarray(list_of_tensors) and fails on the mismatched shapes.
        ragged = [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0, 3.0])]
        out = to_numpy(ragged)
        self.assertIsInstance(out, list)
        np.testing.assert_allclose(out[0], [1.0, 2.0])
        np.testing.assert_allclose(out[1], [1.0, 2.0, 3.0])

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_to_numpy_recurses_dict_of_torch_tensors(self):
        # Regression (MXR-080-0123): dictionary VALUES used to remain un-converted tensors.
        payload = {"obs": torch.tensor([1.0, 2.0]), "count": torch.tensor([3, 4, 5])}
        out = to_numpy(payload)
        self.assertIsInstance(out["obs"], np.ndarray)
        self.assertIsInstance(out["count"], np.ndarray)
        np.testing.assert_allclose(out["obs"], [1.0, 2.0])
        np.testing.assert_allclose(out["count"], [3, 4, 5])

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_to_numpy_transfers_non_cpu_tensor_to_host(self):
        # Device-transfer half of MXR-080-0123. Prefers a real accelerator (CUDA, then MPS) when
        # available; otherwise this still exercises the structural fix on CPU, since
        # TorchEngine.to_numpy always routes through .detach().cpu().numpy() -- the host-transfer
        # call is on the path (as a no-op) even without real device hardware in this environment.
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        payload = {"x": torch.tensor([1.0, 2.0, 3.0], device=dev)}
        out = to_numpy(payload)
        self.assertIsInstance(out["x"], np.ndarray)
        np.testing.assert_allclose(out["x"], [1.0, 2.0, 3.0])

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_engine_mesh_replicates_and_component_shards(self):
        from torch.distributed.tensor import DTensor, Replicate, Shard

        mesh = _single_rank_mesh()
        engine = TorchEngine(dtype=torch.float64, mesh=mesh, shard="components")

        replicated = engine.asarray([1.0, 2.0, 3.0])
        sharded = engine.place_component_axis(replicated, axis=0)

        self.assertIsInstance(replicated, DTensor)
        self.assertIsInstance(sharded, DTensor)
        self.assertIsInstance(replicated.placements[0], Replicate)
        self.assertIsInstance(sharded.placements[0], Shard)
        self.assertEqual(sharded.placements[0].dim, 0)
        np.testing.assert_allclose(engine.to_numpy(sharded), np.asarray([1.0, 2.0, 3.0]))
        self.assertIsInstance(engine_of(sharded), TorchEngine)


class ActiveEngineConcurrencyTest(unittest.TestCase):
    """Regression: using_active_engine used to back its state with threading.local, which isolates OS
    threads (still verified below) but not concurrent asyncio tasks sharing one thread -- one task
    could observe another's active engine mid-block. Fixed by switching to contextvars.ContextVar."""

    def test_concurrent_asyncio_tasks_do_not_see_each_others_engine(self):
        import asyncio

        from mixle.engines.base import active_engine, using_active_engine

        seen = {}

        async def worker(name, delay1, delay2):
            with using_active_engine(name):
                await asyncio.sleep(delay1)
                seen[name] = active_engine()
                await asyncio.sleep(delay2)

        async def main():
            await asyncio.gather(worker("X", 0.03, 0.03), worker("Y", 0.0, 0.06))

        asyncio.run(main())
        self.assertEqual(seen, {"X": "X", "Y": "Y"})
        self.assertIsNone(active_engine())  # cleared outside every block

    def test_concurrent_threads_still_isolated(self):
        import threading

        from mixle.engines.base import active_engine, using_active_engine

        seen = {}

        def worker(name, delay):
            with using_active_engine(name):
                import time

                time.sleep(delay)
                seen[name] = active_engine()

        t1 = threading.Thread(target=worker, args=("A", 0.03))
        t2 = threading.Thread(target=worker, args=("B", 0.0))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(seen, {"A": "A", "B": "B"})

    def test_exception_inside_the_block_still_restores_the_previous_engine(self):
        from mixle.engines.base import active_engine, using_active_engine

        with using_active_engine("outer"):
            with self.assertRaises(ValueError):
                with using_active_engine("inner"):
                    raise ValueError("boom")
            self.assertEqual(active_engine(), "outer")
        self.assertIsNone(active_engine())


class EngineFeatureFlagTypeTest(unittest.TestCase):
    """MXR-080-1563 -- feature flags that select an execution/compilation/numerical path must require
    an actual Boolean.

    `bool("false")` is `True`, so a flag that round-tripped through YAML/JSON/an environment variable
    as the string `"false"` used to switch the path ON -- the exact inversion of what it says.
    """

    def test_numpy_fused_kernel_flag_rejects_non_bool(self):
        for bad in ("false", "0", "", 0, 1, None, 1.0):
            with self.subTest(prefer_fused=bad), self.assertRaises(TypeError):
                NumpyEngine(prefer_fused=bad)

    def test_numpy_fused_kernel_flag_accepts_bools(self):
        self.assertFalse(NumpyEngine(prefer_fused=False).prefer_fused)
        self.assertTrue(NumpyEngine(prefer_fused=True).prefer_fused)
        self.assertIs(NumpyEngine(prefer_fused=np.bool_(True)).prefer_fused, True)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_compile_flag_rejects_non_bool(self):
        for bad in ("false", "0", "", 0, 1, None, 1.0):
            with self.subTest(compile=bad), self.assertRaises(TypeError):
                TorchEngine(compile=bad)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_compile_flag_accepts_bools(self):
        self.assertFalse(TorchEngine(compile=False).compile_enabled)
        self.assertTrue(TorchEngine(compile=True).compile_enabled)
        # with_precision re-threads the stored flag, so it must stay an accepted Boolean
        self.assertTrue(TorchEngine(compile=True).with_precision("float32").compile_enabled)

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_jax_compile_flag_rejects_non_bool(self):
        for bad in ("false", "0", "", 0, 1, None, 1.0):
            with self.subTest(compile=bad), self.assertRaises(TypeError):
                JaxEngine(compile=bad)

    @unittest.skipUnless(HAS_JAX, "jax is not installed")
    def test_jax_compile_flag_accepts_bools(self):
        self.assertFalse(JaxEngine(compile=False).compile_enabled)
        self.assertTrue(JaxEngine(compile=True).compile_enabled)
        self.assertTrue(JaxEngine(compile=True).with_precision("float32").compile_enabled)


if __name__ == "__main__":
    unittest.main()
