import importlib
import tempfile
import unittest

import numpy as np

from mixle.engines import (
    NUMPY_ENGINE,
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


if __name__ == "__main__":
    unittest.main()
