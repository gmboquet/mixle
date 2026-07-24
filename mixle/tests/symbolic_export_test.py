import importlib
import unittest

import numpy as np

from mixle.engines import (
    SYMBOLIC_ENGINE,
    SymbolicExpression,
    to_latex,
    to_sage,
    to_sympy,
)
from mixle.engines import arithmetic as ar

HAS_SYMPY = importlib.util.find_spec("sympy") is not None


def _sage_module_name():
    # full SageMath -> "sage.all"; pip-installable passagemath -> "sage.all__sagemath_symbolics".
    for name in ("sage.all", "sage.all__sagemath_symbolics"):
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            continue
    return None


_SAGE_MODULE = _sage_module_name()
HAS_SAGE = _SAGE_MODULE is not None


def _gaussian_log_density_expr(mu=0.0, sigma2=1.0):
    """Symbolic Gaussian log-density expression in symbol ``x``."""
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

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

    def test_pi_stays_symbolic_not_a_decimal(self):
        # gaussian.py / log_gaussian.py / student_t.py / multivariate_gaussian.py got pi via
        # `from mixle.engines.arithmetic import *` (or, for student_t, `math.pi` directly) inside
        # functions that also take an explicit `engine` argument. `from module import *` binds a
        # STATIC snapshot at the IMPORTING module's own import time (ordinary Python import
        # semantics, not something arithmetic.py's PEP-562 __getattr__ can override), so it was
        # always NUMPY_ENGINE's plain-float pi -- regardless of which engine a given call later
        # passed in. Passing SYMBOLIC_ENGINE explicitly never made it symbolic: the LaTeX literally
        # contained "6.28318530717959" instead of "\pi". Fixed by reading `engine.pi` off the
        # actual passed-in engine (every ComputeEngine already exposes it: math.pi on the numpy/
        # torch engines, a genuine SymbolicExpression on SYMBOLIC_ENGINE) instead of the frozen
        # import.
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
        from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution
        from mixle.stats.univariate.continuous.student_t import StudentTDistribution

        x = SYMBOLIC_ENGINE.symbol("x")
        cases = [
            GaussianDistribution(2.0, 1.5).backend_seq_log_density(x, SYMBOLIC_ENGINE),
            LogGaussianDistribution(0.5, 1.2).backend_seq_log_density(x, SYMBOLIC_ENGINE),
            StudentTDistribution(4.0, 0.0, 1.0).backend_seq_log_density(x, SYMBOLIC_ENGINE),
        ]
        for expr in cases:
            latex = to_latex(expr)
            self.assertIn(r"\pi", latex)
            self.assertNotIn("6.283", latex)  # 2*pi as a decimal
            self.assertNotIn("3.14159", latex)  # pi as a decimal

    def test_pi_numeric_value_unchanged_for_the_default_numpy_engine(self):
        # engine.pi must be a drop-in for the old frozen-import pi on the ordinary (non-symbolic)
        # path -- same float, so no numeric drift for real callers who never touch SYMBOLIC_ENGINE.
        from mixle.engines import NUMPY_ENGINE
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        d = GaussianDistribution(2.0, 1.5)
        x = NUMPY_ENGINE.asarray([-1.0, 0.0, 3.0, 7.0])
        got = d.backend_seq_log_density(x, NUMPY_ENGINE)
        want = np.array([d.log_density(float(v)) for v in [-1.0, 0.0, 3.0, 7.0]])
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_engine_wrappers(self):
        x = SYMBOLIC_ENGINE.symbol("x")
        expr = ar.log(ar.exp(x) + 1.0)
        self.assertEqual(str(SYMBOLIC_ENGINE.to_sympy(expr)), str(to_sympy(expr)))
        self.assertTrue(SYMBOLIC_ENGINE.to_latex(expr))


@unittest.skipUnless(HAS_SAGE, "sagemath is not installed")
class SymbolicSageExportTestCase(unittest.TestCase):
    def test_gaussian_density_roundtrip(self):
        sage = importlib.import_module(_SAGE_MODULE)

        expr, _ = _gaussian_log_density_expr(mu=1.5, sigma2=2.0)
        sym = to_sage(expr)
        x = sage.var("x")
        for xv in (-1.0, 0.0, 1.5, 4.0):
            native = float(expr.evaluate({"x": xv}))
            via_sage = float(sym.subs({x: xv}))
            self.assertAlmostEqual(native, via_sage, places=10)

    def test_engine_constants_lower_to_exact_sage(self):
        sage = importlib.import_module(_SAGE_MODULE)
        with ar.using_engine("symbolic"):
            self.assertTrue(bool(to_sage(ar.pi) == sage.pi))  # exact symbolic pi, not a float
            self.assertTrue(bool(to_sage(ar.e) == sage.e))
            self.assertTrue(bool(to_sage(ar.euler_gamma) == sage.euler_gamma))
            self.assertTrue(bool(to_sage(ar.half) == sage.SR(1) / sage.SR(2)))  # exact 1/2
            expr = ar.two * ar.pi
            self.assertTrue(bool(to_sage(expr) == 2 * sage.pi))

    # MXR-080-0153 regression: the sage `where` lowering used to build an
    # arithmetic 0/1 "indicator" out of `heaviside(lhs - rhs)` and blend
    # `a*indicator + b*(1-indicator)`. Since sage defines `heaviside(0) ==
    # 1/2`, every one of `<`, `<=`, `>`, `>=` evaluated to `0.5*a + 0.5*b`
    # exactly at the comparison boundary (`lhs == rhs`) instead of selecting
    # a single branch -- regardless of whether the comparison was strict or
    # non-strict. `where` must match `numpy.where` and pick exactly one
    # branch for every input, including ties. The fix routes through sage's
    # `cases`, which evaluates the relation's actual boolean truth value.

    def test_where_boundary_selects_exact_branch(self):
        sage = importlib.import_module(_SAGE_MODULE)
        x = SYMBOLIC_ENGINE.symbol("x")
        xs = sage.var("x")
        a_val, b_val = 10.0, 20.0
        blended = 0.5 * (a_val + b_val)  # the give-away value from the old bug
        # (comparison at x == 5.0, expected single branch at that exact tie)
        boundary_cases = [
            ("<", x < 5.0, b_val),  # strict: boundary is FALSE -> else-branch
            ("<=", x <= 5.0, a_val),  # non-strict: boundary is TRUE -> then-branch
            (">", x > 5.0, b_val),  # strict: boundary is FALSE -> else-branch
            (">=", x >= 5.0, a_val),  # non-strict: boundary is TRUE -> then-branch
        ]
        for name, cond, expected in boundary_cases:
            with self.subTest(op=name):
                expr = SYMBOLIC_ENGINE.where(cond, a_val, b_val)
                # sanity: the native (non-sage) reference agrees on the expected branch
                self.assertEqual(float(expr.evaluate({"x": 5.0})), expected)
                sym = to_sage(expr)
                via_sage = float(sym.subs({xs: 5.0}))
                self.assertEqual(via_sage, expected)
                self.assertNotAlmostEqual(via_sage, blended)

    def test_where_selects_correct_branch_away_from_boundary(self):
        sage = importlib.import_module(_SAGE_MODULE)
        x = SYMBOLIC_ENGINE.symbol("x")
        xs = sage.var("x")
        a_val, b_val = 10.0, 20.0
        comparators = {"<": x < 5.0, "<=": x <= 5.0, ">": x > 5.0, ">=": x >= 5.0}
        for name, cond in comparators.items():
            expr = SYMBOLIC_ENGINE.where(cond, a_val, b_val)
            sym = to_sage(expr)
            for xv in (4.0, 6.0):  # strictly below / strictly above the boundary
                with self.subTest(op=name, x=xv):
                    expected = float(expr.evaluate({"x": xv}))
                    via_sage = float(sym.subs({xs: xv}))
                    self.assertEqual(via_sage, expected)

    def test_where_sage_export_has_no_raw_heaviside(self):
        # Symbolic-form check: the exported expression must be a genuine
        # piecewise/indicator (sage's `cases`), never a raw `heaviside` call
        # left over from the old blended-arithmetic lowering.
        x = SYMBOLIC_ENGINE.symbol("x")
        expr = SYMBOLIC_ENGINE.where(x >= 0.0, x + 1.0, x - 1.0)
        text = str(to_sage(expr))
        self.assertNotIn("heaviside", text)
        self.assertIn("cases", text)

    def test_non_conditional_export_unaffected_by_where_fix(self):
        # Negative control: a plain arithmetic expression with no `where` or
        # comparison anywhere in the tree must export exactly as before --
        # this change only touches the `where` lowering.
        sage = importlib.import_module(_SAGE_MODULE)
        x = SYMBOLIC_ENGINE.symbol("x")
        xs = sage.var("x")
        expr = ar.log(ar.exp(x) + 1.0)
        sym = to_sage(expr)
        for xv in (-2.0, 0.0, 3.5):
            native = float(expr.evaluate({"x": xv}))
            via_sage = float(sym.subs({xs: xv}))
            self.assertAlmostEqual(native, via_sage, places=10)


if __name__ == "__main__":
    unittest.main()
