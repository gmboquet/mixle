"""Cross-engine parity for the canonical ComputeEngine op surface.

The ``ComputeEngine`` ABC declares :attr:`~pysp.engines.base.ComputeEngine.REQUIRED_OPS` -- the
array ops backend-neutral kernels duck-type on the engine.  Historically these were provided
informally per engine, so an op could be present on numpy but missing or divergent on torch/symbolic
(finding D1).  This test converts that bug class into a guarded invariant:

* every required op exists on every engine (already enforced at class-definition by
  ``__init_subclass__``; re-asserted here as an explicit, readable contract), and
* on small representative inputs each engine produces matching numeric results.

The symbolic engine is exercised where the op is numerically evaluable (its reductions/elementwise
ops build expression trees that ``evaluate`` to the same numbers); ops that have no scalar-evaluable
symbolic form (``index_add``, ``bincount``, ``unique``, ``searchsorted``) are checked numpy-vs-torch
only -- documented per-op below rather than silently skipped.
"""

import importlib
import unittest

import numpy as np

from pysp.engines import NumpyEngine, SymbolicEngine, TorchEngine
from pysp.engines.base import ComputeEngine

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
else:  # pragma: no cover - torch is installed in the dev env
    torch = None


def _to_np(engine, value):
    """Materialize an engine result as a host float array for comparison."""
    if isinstance(engine, SymbolicEngine):
        return np.asarray(engine.to_numpy(value), dtype=object)
    return np.asarray(engine.to_numpy(value), dtype=np.float64)


# Each case: name -> (callable(engine) -> result, evaluable_symbolically).  The callable builds the op
# on the engine from that engine's own ``asarray`` inputs so dtype/placement policy is respected.
def _cases():
    a = [1.0, 4.0, 9.0]
    b = [2.0, 1.0, 0.5]
    mat = [[1.0, 2.0], [3.0, 4.0]]
    vec = [1.0, 2.0]
    pos = [0.5, 1.5, 2.5]

    return {
        "log": (lambda e: e.log(e.asarray(a)), True),
        "exp": (lambda e: e.exp(e.asarray(b)), True),
        "sqrt": (lambda e: e.sqrt(e.asarray(a)), True),
        "abs": (lambda e: e.abs(e.asarray([-1.0, 2.0, -3.0])), True),
        "where": (lambda e: e.where(e.asarray([True, False, True]), e.asarray(a), e.asarray(b)), True),
        "maximum": (lambda e: e.maximum(e.asarray(a), e.asarray(b)), True),
        "clip": (lambda e: e.clip(e.asarray(a), 2.0, 5.0), True),
        "floor": (lambda e: e.floor(e.asarray([1.2, 2.8, 3.5])), True),
        "isnan": (lambda e: e.isnan(e.asarray([0.0, 1.0, 2.0])), True),
        "isinf": (lambda e: e.isinf(e.asarray([0.0, 1.0, 2.0])), True),
        # Reductions take ``axis`` uniformly (numpy flattens with no axis, torch requires a dim, so a
        # cross-engine call passes the axis explicitly -- this is the documented engine contract).
        "sum": (lambda e: e.sum(e.asarray(a), axis=0), True),
        "max": (lambda e: e.max(e.asarray(a), axis=0), True),
        "dot": (lambda e: e.dot(e.asarray(vec), e.asarray(vec)), True),
        "matmul": (lambda e: e.matmul(e.asarray(mat), e.asarray(vec)), True),
        "cumsum": (lambda e: e.cumsum(e.asarray(a), axis=0), True),
        "logsumexp": (lambda e: e.logsumexp(e.asarray(a), axis=0), True),
        "gammaln": (lambda e: e.gammaln(e.asarray(pos)), True),
        "digamma": (lambda e: e.digamma(e.asarray(pos)), True),
        "betaln": (lambda e: e.betaln(e.asarray(pos), e.asarray(pos)), True),
        "erf": (lambda e: e.erf(e.asarray([0.0, 0.5, 1.0])), True),
        # Integer / indexing ops have no scalar-evaluable symbolic form: numpy-vs-torch only.
        "bincount": (lambda e: e.bincount(e.asarray(np.array([0, 1, 1, 2, 2, 2], dtype=np.int64))), False),
        "unique": (lambda e: e.unique(e.asarray(np.array([3, 1, 2, 1, 3], dtype=np.int64))), False),
        "searchsorted": (
            lambda e: e.searchsorted(e.asarray(np.array([1.0, 3.0, 5.0])), e.asarray(np.array([2.0, 4.0]))),
            False,
        ),
    }


class EngineOpParityTestCase(unittest.TestCase):
    def setUp(self):
        self.numpy = NumpyEngine(dtype="float64")
        self.symbolic = SymbolicEngine()
        self.torch = TorchEngine(device="cpu", dtype="float64") if HAS_TORCH else None

    def test_required_ops_present_on_every_engine(self):
        engines = [self.numpy, self.symbolic] + ([self.torch] if self.torch is not None else [])
        for engine in engines:
            with self.subTest(engine=engine.name):
                missing = [op for op in ComputeEngine.REQUIRED_OPS if getattr(engine, op, None) is None]
                self.assertEqual(missing, [], "%s missing required ops: %s" % (engine.name, missing))

    def test_required_ops_covered_by_parity_cases(self):
        # Guard against the canonical list and the parity matrix drifting apart: every required *math*
        # op must have a parity case.  Pure allocation/conversion ops are covered by engine_test.py.
        allocation = {"asarray", "zeros", "empty", "arange", "to_numpy", "stack", "index_add"}
        math_ops = set(ComputeEngine.REQUIRED_OPS) - allocation
        self.assertEqual(math_ops - set(_cases()), set(), "REQUIRED_OPS math op without a parity case")

    def test_accumulator_dtype_present(self):
        # The prior audit gap: symbolic lacked accumulator_dtype.  It must now resolve on every engine.
        self.assertIs(self.numpy.accumulator_dtype, np.float64)
        self.assertIsNone(self.symbolic.accumulator_dtype)
        if self.torch is not None:
            self.assertIs(self.torch.accumulator_dtype, torch.float64)

    def test_op_results_match_across_engines(self):
        cases = _cases()
        for name, (build, symbolic_ok) in cases.items():
            ref = _to_np(self.numpy, build(self.numpy)).astype(np.float64)
            if self.torch is not None:
                with self.subTest(op=name, engine="torch"):
                    got = _to_np(self.torch, build(self.torch)).astype(np.float64)
                    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6, err_msg="torch op %s" % name)
            if symbolic_ok:
                with self.subTest(op=name, engine="symbolic"):
                    expr = build(self.symbolic)
                    evaluated = self.symbolic.evaluate(expr, {})
                    got = np.asarray(self.symbolic.to_numpy(evaluated), dtype=np.float64)
                    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6, err_msg="symbolic op %s" % name)

    def test_index_add_parity(self):
        # index_add carries an engine-specific in-place/return contract; check numpy-vs-torch parity.
        out_np = self.numpy.zeros(3)
        res_np = self.numpy.index_add(
            out_np, np.array([0, 1, 1, 2], dtype=np.int64), self.numpy.asarray([1.0, 2.0, 3.0, 4.0])
        )
        ref = np.asarray(res_np, dtype=np.float64)
        np.testing.assert_allclose(ref, np.array([1.0, 5.0, 4.0]))
        if self.torch is not None:
            out_t = self.torch.zeros(3)
            res_t = self.torch.index_add(
                out_t, np.array([0, 1, 1, 2], dtype=np.int64), self.torch.asarray([1.0, 2.0, 3.0, 4.0])
            )
            np.testing.assert_allclose(self.torch.to_numpy(res_t), ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
