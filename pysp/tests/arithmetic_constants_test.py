"""Engine-aware mathematical constants in pysp.arithmetic (numeric by default, symbolic on request)."""

import math
import unittest

import pysp.arithmetic as arith
from pysp.engines import NUMPY_ENGINE, SYMBOLIC_ENGINE, SymbolicExpression, to_sympy

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


if __name__ == "__main__":
    unittest.main()
