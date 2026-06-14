import importlib
import unittest

import numpy as np

from pysp import arithmetic as ar
from pysp.engines import (
    SYMBOLIC_ENGINE,
    SymbolicExpression,
    to_latex,
    to_sage,
    to_sympy,
)

HAS_SYMPY = importlib.util.find_spec("sympy") is not None
HAS_SAGE = importlib.util.find_spec("sage") is not None


def _gaussian_log_density_expr(mu=0.0, sigma2=1.0):
    """Symbolic Gaussian log-density expression in symbol ``x``."""
    from pysp.stats.gaussian import GaussianDistribution

    x = SYMBOLIC_ENGINE.symbol("x")
    return GaussianDistribution(mu, sigma2).backend_seq_log_density(x, SYMBOLIC_ENGINE), x


@unittest.skipUnless(HAS_SYMPY, "sympy is not installed")
class SymbolicSympyExportTestCase(unittest.TestCase):
    def _assert_matches_native(self, expr, samples, places=10):
        import sympy

        sym = to_sympy(expr)
        names = expr.symbols()
        syms = [sympy.Symbol(n) for n in names]
        f = sympy.lambdify(syms, sym, modules=["numpy", "math"])
        for point in samples:
            native = float(expr.evaluate(point))
            via_sympy = float(f(*[point[n] for n in names]))
            self.assertAlmostEqual(native, via_sympy, places=places)

    def test_scalar_expression_roundtrip(self):
        x = SYMBOLIC_ENGINE.symbol("x")
        expr = ar.log(ar.exp(x) + 1.0)
        self._assert_matches_native(expr, [{"x": -2.0}, {"x": 0.0}, {"x": 3.5}])

    def test_special_function_gammaln(self):
        x = SYMBOLIC_ENGINE.symbol("x")
        expr = SYMBOLIC_ENGINE.gammaln(x * x + 2.0)
        self._assert_matches_native(expr, [{"x": 0.5}, {"x": 1.0}, {"x": 2.3}])

    def test_betaln_lowers_to_loggamma(self):
        import sympy

        x = SYMBOLIC_ENGINE.symbol("x")
        y = SYMBOLIC_ENGINE.symbol("y")
        expr = SYMBOLIC_ENGINE.betaln(x, y)
        sym = to_sympy(expr)
        self.assertIn(sympy.loggamma, {a.func for a in sym.atoms(sympy.Function)})
        f = sympy.lambdify([sympy.Symbol("x"), sympy.Symbol("y")], sym, "scipy")
        self.assertAlmostEqual(float(expr.evaluate({"x": 2.0, "y": 3.0})), float(f(2.0, 3.0)), places=10)

    def test_where_lowers_to_piecewise(self):
        import sympy

        x = SYMBOLIC_ENGINE.symbol("x")
        expr = SYMBOLIC_ENGINE.where(x >= 0.0, x + 1.0, x - 1.0)
        sym = to_sympy(expr)
        self.assertIsInstance(sym, sympy.Piecewise)
        self._assert_matches_native(expr, [{"x": -3.0}, {"x": 0.0}, {"x": 2.0}])

    def test_clip_lowers_to_min_max(self):
        x = SYMBOLIC_ENGINE.symbol("x")
        expr = SYMBOLIC_ENGINE.clip(x, 0.0, 5.0)
        self._assert_matches_native(expr, [{"x": -2.0}, {"x": 3.0}, {"x": 9.0}])

    def test_gaussian_density_roundtrip(self):
        expr, _ = _gaussian_log_density_expr(mu=1.5, sigma2=2.0)
        self._assert_matches_native(expr, [{"x": -1.0}, {"x": 0.0}, {"x": 1.5}, {"x": 4.0}])

    def test_gaussian_symbolic_differentiation_is_score(self):
        import sympy

        mu, sigma2 = 1.5, 2.0
        expr, x_node = _gaussian_log_density_expr(mu=mu, sigma2=sigma2)
        x = sympy.Symbol("x")
        score = sympy.diff(to_sympy(expr), x)
        f = sympy.lambdify(x, score, "numpy")
        for xv in (-1.0, 0.0, 1.5, 4.0):
            analytic = -(xv - mu) / sigma2
            self.assertAlmostEqual(float(f(xv)), analytic, places=10)

    def test_array_input_maps_elementwise(self):
        import sympy

        x = SYMBOLIC_ENGINE.symbol("x")
        y = SYMBOLIC_ENGINE.symbol("y")
        arr = SYMBOLIC_ENGINE.asarray([x, y])
        logged = SYMBOLIC_ENGINE.log(arr + 1.0)
        out = to_sympy(logged)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (2,))
        self.assertIsInstance(out[0], sympy.Expr)
        f0 = sympy.lambdify(sympy.Symbol("x"), out[0], "numpy")
        self.assertAlmostEqual(float(f0(3.0)), float(np.log(4.0)), places=10)

    def test_non_symbolic_op_raises(self):
        node = SymbolicExpression.call("bincount", SYMBOLIC_ENGINE.symbol("x"))
        with self.assertRaises(NotImplementedError) as ctx:
            to_sympy(node)
        self.assertIn("bincount", str(ctx.exception))

    def test_to_latex_returns_nonempty_string(self):
        expr, _ = _gaussian_log_density_expr()
        latex = to_latex(expr)
        self.assertIsInstance(latex, str)
        self.assertTrue(latex)

    def test_engine_wrappers(self):
        x = SYMBOLIC_ENGINE.symbol("x")
        expr = ar.log(ar.exp(x) + 1.0)
        self.assertEqual(str(SYMBOLIC_ENGINE.to_sympy(expr)), str(to_sympy(expr)))
        self.assertTrue(SYMBOLIC_ENGINE.to_latex(expr))


@unittest.skipUnless(HAS_SAGE, "sagemath is not installed")
class SymbolicSageExportTestCase(unittest.TestCase):
    def test_gaussian_density_roundtrip(self):
        import sage.all as sage

        expr, _ = _gaussian_log_density_expr(mu=1.5, sigma2=2.0)
        sym = to_sage(expr)
        x = sage.var("x")
        for xv in (-1.0, 0.0, 1.5, 4.0):
            native = float(expr.evaluate({"x": xv}))
            via_sage = float(sym.subs({x: xv}))
            self.assertAlmostEqual(native, via_sage, places=10)


if __name__ == "__main__":
    unittest.main()
