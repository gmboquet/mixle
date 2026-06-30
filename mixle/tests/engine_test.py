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
    def test_mps_engine_falls_back_to_float32(self):
        # MPS has no float64; the engine must downgrade so torch-ready families run on Apple-silicon GPUs.
        # torch.device("mps") is constructible regardless of whether MPS is actually available, so this
        # exercises the policy on any host (incl. CPU-only CI).
        mps = TorchEngine(device="mps")
        self.assertEqual(mps.dtype, torch.float32)
        self.assertEqual(mps.accumulator_dtype, torch.float32)
        self.assertEqual(TorchEngine(device="mps", dtype="float64").dtype, torch.float32)  # explicit f64 downgraded
        # CPU/CUDA keep full precision
        self.assertEqual(TorchEngine(device="cpu").accumulator_dtype, torch.float64)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_mixed_engine_payload_fails(self):
        payload = (np.asarray([1.0]), torch.tensor([1.0]))
        with self.assertRaises(TypeError):
            engine_of(payload)

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


if __name__ == "__main__":
    unittest.main()
