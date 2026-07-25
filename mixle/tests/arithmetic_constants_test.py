"""Engine-aware mathematical constants in mixle.engines.arithmetic (numeric by default, symbolic on request)."""

import math
import unittest

import numpy as np

import mixle.engines.arithmetic as arith
from mixle.engines import NUMPY_ENGINE, SYMBOLIC_ENGINE, SymbolicExpression, to_sympy

try:
    import sympy

    HAS_SYMPY = True
except ImportError:  # pragma: no cover
    HAS_SYMPY = False


class ArithmeticConstantsTest(unittest.TestCase):
    def tearDown(self):
        arith.set_default_engine(NUMPY_ENGINE)  # never leak the active engine across tests

    def test_default_is_numpy_floats(self):
        self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)
        self.assertEqual(arith.pi, math.pi)
        self.assertIsInstance(arith.pi, float)
        self.assertEqual((arith.zero, arith.one, arith.two, arith.half), (0.0, 1.0, 2.0, 0.5))

    def test_limits_are_engine_independent(self):
        with arith.using_engine("symbolic"):
            self.assertEqual(arith.maxrandint, 2**31 - 1)  # implementation limits, not math constants
            self.assertEqual(arith.eps, 1.0e-8)

    def test_max_keeps_python_scalar_semantics(self):
        self.assertEqual(arith.max(1.0, 2.5, -1.0), 2.5)
        self.assertEqual(arith.max(np.float64(0.4), 0.0, np.float64(0.6)), np.float64(0.6))

    def test_max_still_dispatches_array_reductions(self):
        values = np.asarray([[1.0, 3.0], [4.0, 2.0]])
        np.testing.assert_allclose(arith.max(values, axis=1), np.asarray([3.0, 4.0]))

    def test_symbolic_constants_are_symbolic(self):
        with arith.using_engine(SYMBOLIC_ENGINE):
            self.assertIsInstance(arith.pi, SymbolicExpression)
            self.assertEqual(str(arith.pi), "pi")
            self.assertIsInstance(arith.two, SymbolicExpression)
            c = arith.constant(7)  # == is overloaded on SymbolicExpression, so compare structurally
            self.assertEqual((c.op, c.args), ("const", (7,)))

    def test_using_engine_restores_previous(self):
        self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)
        with arith.using_engine("symbolic"):
            self.assertIs(arith.get_default_engine(), SYMBOLIC_ENGINE)
        self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)  # restored
        self.assertEqual(arith.pi, math.pi)

    def test_set_default_engine_returns_previous(self):
        prev = arith.set_default_engine("symbolic")
        self.assertIs(prev, NUMPY_ENGINE)
        self.assertIsInstance(arith.pi, SymbolicExpression)

    def test_unknown_engine_name_raises(self):
        with self.assertRaises(ValueError):
            arith.set_default_engine("quantum")

    @unittest.skipUnless(HAS_SYMPY, "sympy not installed")
    def test_symbolic_constants_stay_exact_through_sympy(self):
        with arith.using_engine("symbolic"):
            self.assertEqual(to_sympy(arith.pi), sympy.pi)  # not 3.14159...
            self.assertEqual(to_sympy(arith.e), sympy.E)
            self.assertEqual(to_sympy(arith.euler_gamma), sympy.EulerGamma)
            self.assertEqual(to_sympy(arith.half), sympy.Rational(1, 2))  # exact 1/2, not 0.5
            expr = arith.two * arith.pi
            self.assertEqual(to_sympy(expr), 2 * sympy.pi)


class ArithmeticToNumpyTest(unittest.TestCase):
    """Regression tests for ``mixle.engines.arithmetic.to_numpy`` -- a SEPARATE dispatch-wrapper
    function from ``mixle.engines.to_numpy`` (fixed by MXR-080-0123), built (until this fix) via
    this module's generic ``_dispatch`` machinery. ``_dispatch`` resolves ONE engine for the whole
    payload (correctly, recursing through ``engine_of``) but then hands that single engine the
    ORIGINAL, unconverted container -- exactly the "container passed whole to one engine's conversion
    routine" bug MXR-080-0123 fixed in the other ``to_numpy``. Confirmed with a numpy-only repro (no
    torch/jax needed): a dict of arrays converted to a useless 0-d object array wrapping the dict, and
    a ragged list of arrays raised ValueError, exactly mirroring the pre-fix package-level bug.
    """

    def tearDown(self):
        arith.set_default_engine(NUMPY_ENGINE)  # never leak the active engine across tests

    def test_flat_array_still_converts_correctly(self):
        # Negative control: a plain, non-nested array must still convert exactly as before.
        x = np.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(arith.to_numpy(x), x)

    def test_plain_nested_list_without_engine_values_is_unchanged(self):
        # Negative control: nested Python data with no engine-owned leaf inside is still handed to the
        # resolved engine as ONE unit (a single 2D array), not walked leaf by leaf.
        out = arith.to_numpy([[1.0, 2.0], [3.0, 4.0]])
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_allclose(out, [[1.0, 2.0], [3.0, 4.0]])

    def test_recurses_ragged_list_of_arrays(self):
        # Regression, exact audit repro: arith.to_numpy used to resolve ONE engine for the whole list
        # (correctly, via engine_of's own recursion) but then hand NumpyEngine the RAGGED LIST ITSELF --
        # np.asarray on a ragged list of differently shaped arrays raises ValueError. Each leaf must
        # convert independently, preserving list structure.
        ragged = [np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0, 3.0])]
        with self.assertRaises(ValueError):
            np.asarray(ragged)  # confirms the old single-shot conversion really would fail

        out = arith.to_numpy(ragged)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        np.testing.assert_allclose(out[0], [1.0, 2.0])
        np.testing.assert_allclose(out[1], [1.0, 2.0, 3.0])

    def test_recurses_dict_values(self):
        # Regression, exact audit repro: arith.to_numpy used to hand the whole dict to a single
        # engine's to_numpy (np.asarray on a dict wraps it in a useless 0-d object array instead of
        # touching its values); dict VALUES must each convert in place, preserving keys.
        payload = {"a": np.asarray([1.0, 2.0]), "b": np.asarray([3.0, 4.0, 5.0])}
        out = arith.to_numpy(payload)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"a", "b"})
        np.testing.assert_allclose(out["a"], [1.0, 2.0])
        np.testing.assert_allclose(out["b"], [3.0, 4.0, 5.0])

    def test_preserves_tuple_container_type(self):
        payload = (np.asarray([1.0]), np.asarray([2.0, 3.0]))
        out = arith.to_numpy(payload)
        self.assertIsInstance(out, tuple)
        np.testing.assert_allclose(out[0], [1.0])
        np.testing.assert_allclose(out[1], [2.0, 3.0])

    def test_uses_active_engine_for_plain_data_with_no_engine_owned_value(self):
        # arithmetic.to_numpy's distinguishing contract vs. mixle.engines.to_numpy (see this module's
        # docstring): a payload with no engine-owned value anywhere resolves against the ACTIVE engine
        # -- this module's whole extension point -- not a hardcoded NumpyEngine. A naive
        # `to_numpy = mixle.engines.to_numpy` alias would silently drop this for plain data.
        with arith.using_engine("symbolic"):
            out = arith.to_numpy(5.0)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, object)  # SymbolicEngine.to_numpy: np.asarray(x, dtype=object)

    def test_engine_owned_array_ignores_active_engine_default(self):
        # Negative control: a genuine NumPy array resolves through its OWN registered engine
        # (NumpyEngine) regardless of the active default -- the active-engine fallback only applies
        # when NOTHING in the payload is directly engine-owned.
        with arith.using_engine("symbolic"):
            out = arith.to_numpy(np.asarray([1.0, 2.0, 3.0]))
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.float64)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


class DefaultEngineConcurrencyTest(unittest.TestCase):
    """Regression (MXR-080-0124): ``using_engine`` used to back its state with a bare module-global
    (``_default_engine``), which -- unlike the estimation engine context in ``mixle.engines.base``
    (``_ACTIVE`` / ``using_active_engine``) -- has no isolation between concurrent asyncio tasks or
    threads: two overlapping ``using_engine(...)`` blocks would stomp on each other's active engine, so
    a symbolic task could observe "numpy" while its own symbolic scope was still notionally active.
    Fixed by switching to ``contextvars.ContextVar`` with token-based restoration, mirroring ``_ACTIVE``.
    """

    def tearDown(self):
        arith.set_default_engine(NUMPY_ENGINE)  # never leak the active engine across tests

    def test_concurrent_asyncio_tasks_do_not_see_each_others_engine(self):
        import asyncio

        seen = {}

        async def worker(name, engine, delay1, delay2):
            with arith.using_engine(engine):
                await asyncio.sleep(delay1)
                # symbolic_task sleeps first (yielding control) so numpy_task can enter and stomp the
                # old bare global before symbolic_task resumes and samples it mid-block.
                seen[name] = arith.get_default_engine()
                await asyncio.sleep(delay2)

        async def main():
            await asyncio.gather(
                worker("symbolic_task", "symbolic", 0.03, 0.03),
                worker("numpy_task", "numpy", 0.0, 0.06),
            )

        asyncio.run(main())
        self.assertIs(seen["symbolic_task"], SYMBOLIC_ENGINE)
        self.assertIs(seen["numpy_task"], NUMPY_ENGINE)
        self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)  # cleared outside every block

    def test_concurrent_threads_do_not_see_each_others_engine(self):
        import threading

        # Fixed sleep delays are not a reliable way to force the overlap for OS threads (scheduling
        # isn't deterministic the way asyncio's cooperative await points are -- a naive sleep-timed
        # version of this test can pass "by accident" even against the buggy bare-global code, because
        # thread B's restore-on-exit happens to land back on the value thread A expects). Synchronize
        # explicitly instead, so thread A samples the shared default WHILE thread B's
        # using_engine("numpy") block is still open -- the actual "observed mid-block" scenario.
        seen = {}
        a_ready = threading.Event()
        b_entered = threading.Event()
        a_sampled = threading.Event()

        def worker_a():
            with arith.using_engine("symbolic"):
                a_ready.set()
                b_entered.wait()
                seen["A"] = arith.get_default_engine()
                a_sampled.set()

        def worker_b():
            a_ready.wait()
            with arith.using_engine("numpy"):
                b_entered.set()
                a_sampled.wait()  # keep this scope open until A has sampled mid-block

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertIs(seen["A"], SYMBOLIC_ENGINE)
        self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)  # cleared outside every block

    def test_nested_scopes_restore_correctly(self):
        self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)
        with arith.using_engine("symbolic"):
            self.assertIs(arith.get_default_engine(), SYMBOLIC_ENGINE)
            with arith.using_engine(NUMPY_ENGINE):
                self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)
            # restored to the outer scope's engine, not some stale process-global value
            self.assertIs(arith.get_default_engine(), SYMBOLIC_ENGINE)
        self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)

    def test_exception_inside_the_block_still_restores_the_previous_engine(self):
        with arith.using_engine("symbolic"):
            with self.assertRaises(ValueError):
                with arith.using_engine(NUMPY_ENGINE):
                    raise ValueError("boom")
            self.assertIs(arith.get_default_engine(), SYMBOLIC_ENGINE)
        self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)

    def test_single_threaded_non_nested_usage_is_unchanged(self):
        # Negative control: ordinary sequential, single-scope usage -- no concurrency, no nesting --
        # must behave exactly as it did before the ContextVar fix.
        self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)
        self.assertEqual(arith.pi, math.pi)
        with arith.using_engine("symbolic"):
            self.assertIs(arith.get_default_engine(), SYMBOLIC_ENGINE)
            self.assertIsInstance(arith.pi, SymbolicExpression)
        self.assertIs(arith.get_default_engine(), NUMPY_ENGINE)
        self.assertEqual(arith.pi, math.pi)


if __name__ == "__main__":
    unittest.main()
