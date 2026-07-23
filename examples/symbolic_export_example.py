"""mixle's symbolic export: closed-form log-densities as LaTeX / SymPy / (optionally) Sage.

``mixle.engines.symbolic_engine.SYMBOLIC_ENGINE`` runs a distribution's own backend log-density math
through a small dependency-free expression tree instead of NumPy/Torch; ``mixle.engines.symbolic_export``
then lowers that tree to SymPy (``to_sympy``, ``to_latex``) or, optionally, Sage (``to_sage``). This is
not a capability every distribution has: the only structural gate is ``backend_seq_log_density``
(``mixle.capability.SupportsBackendScoring``), and that's necessary but not sufficient -- the symbolic
engine implements only a subset of the ops NumPy/Torch do (e.g. no ``erfcx``), so some distributions
(``SkewNormalDistribution`` among them) raise ``AttributeError`` under it. ``GaussianDistribution`` and
``StudentTDistribution`` are used below because both are confirmed to round-trip cleanly.

The point isn't "it can print LaTeX" by itself -- it's using the closed form for real symbolic calculus.
Section 3 differentiates each log-density to get its exact score function (d/dx log f(x)), then shows a
genuine, non-obvious result: the Gaussian's score grows without bound as an observation moves away from
the mean (a distant outlier pulls a fit arbitrarily hard), while the Student-t's score is *redescending*
-- it saturates back toward zero for extreme x (bounded influence, a textbook robustness property, not
a marketing claim).

  1. instance log-density (concrete parameters)         -> LaTeX
  2. general closed-form (parameters left symbolic too)  -> LaTeX
  3. d/dx -> the score function, then the growth-vs-saturation numeric contrast
  4. optional: the same expression lowered to Sage, skipped gracefully if Sage isn't installed

Needs: ``pip install mixle[sympy]`` (sympy is an optional extra, not a base dependency). Section 4
additionally needs ``pip install mixle[sage]``; it degrades gracefully without it.

Run: ``python examples/symbolic_export_example.py``
"""

from __future__ import annotations

import importlib.util

import sympy

from mixle.engines import SYMBOLIC_ENGINE, to_latex, to_sage, to_sympy
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.student_t import StudentTDistribution

GAUSSIAN_MU, GAUSSIAN_SIGMA2 = 2.0, 1.5
STUDENT_T_DF, STUDENT_T_LOC, STUDENT_T_SCALE = 4.0, 0.0, 1.0


def build_distributions():
    """Two showcase distributions, both confirmed to round-trip cleanly through the symbolic engine."""
    gaussian = GaussianDistribution(mu=GAUSSIAN_MU, sigma2=GAUSSIAN_SIGMA2)
    student_t = StudentTDistribution(df=STUDENT_T_DF, loc=STUDENT_T_LOC, scale=STUDENT_T_SCALE)
    return gaussian, student_t


def demo_instance_log_density(gaussian, student_t, x):
    """Section 1: each distribution's own log-density (concrete parameters baked in) as LaTeX."""
    print("1. Instance log-density (concrete parameters) -> LaTeX:")
    g_expr = gaussian.backend_seq_log_density(x, SYMBOLIC_ENGINE)
    t_expr = student_t.backend_seq_log_density(x, SYMBOLIC_ENGINE)
    print(f"   Gaussian(mu={GAUSSIAN_MU}, sigma2={GAUSSIAN_SIGMA2}):")
    print(f"     {to_latex(g_expr)}")
    print(f"   StudentT(df={STUDENT_T_DF}, loc={STUDENT_T_LOC}, scale={STUDENT_T_SCALE}):")
    print(f"     {to_latex(t_expr)}")
    return g_expr, t_expr


def demo_general_formula(x):
    """Section 2: the textbook closed-form formula with every parameter left symbolic, not just x."""
    print("\n2. General closed-form (all parameters symbolic, not tied to any instance) -> LaTeX:")
    mu = SYMBOLIC_ENGINE.symbol("mu")
    sigma2 = SYMBOLIC_ENGINE.symbol("sigma2")
    g_general = GaussianDistribution.backend_log_density_from_params(x, mu, sigma2, SYMBOLIC_ENGINE)
    print(f"   Gaussian(x; mu, sigma2):\n     {to_latex(g_general)}")

    df = SYMBOLIC_ENGINE.symbol("df")
    loc = SYMBOLIC_ENGINE.symbol("loc")
    scale = SYMBOLIC_ENGINE.symbol("scale")
    t_general = StudentTDistribution.backend_log_density_from_params(x, df, loc, scale, SYMBOLIC_ENGINE)
    print(f"   StudentT(x; df, loc, scale):\n     {to_latex(t_general)}")


def demo_score_function(g_expr, t_expr):
    """Section 3: differentiate each log-density to get its exact score function.

    This is the real payoff: a numeric check that the Gaussian's score explodes for an extreme
    observation while the Student-t's score saturates back toward zero.
    """
    print("\n3. Score function d/dx log f(x) -- exact symbolic differentiation:")
    x_sym = sympy.Symbol("x")
    g_score = sympy.diff(to_sympy(g_expr), x_sym).simplify()
    t_score = sympy.diff(to_sympy(t_expr), x_sym).simplify()
    print(f"   Gaussian score:  {sympy.latex(g_score)}")
    print(f"   StudentT score:  {sympy.latex(t_score)}")

    g_score_fn = sympy.lambdify(x_sym, g_score, "numpy")
    t_score_fn = sympy.lambdify(x_sym, t_score, "numpy")
    print("\n   The same distant observation, evaluated on both scores:")
    print(f"   {'x':>8}  {'Gaussian score':>16}  {'StudentT score':>16}")
    for xv in (10.0, 100.0):
        print(f"   {xv:8.1f}  {float(g_score_fn(xv)):16.4f}  {float(t_score_fn(xv)):16.4f}")
    print(
        "   -> Gaussian: |score| keeps growing -- a distant outlier pulls the fit arbitrarily hard.\n"
        "   -> Student-t: |score| shrinks back toward 0 -- bounded, redescending, self-limiting influence."
    )


def _sage_module_name() -> str | None:
    """Return the importable Sage module name, or None if Sage isn't installed.

    Mirrors ``mixle/tests/symbolic_export_test.py``: full SageMath exposes ``sage.all``; the
    pip-installable ``passagemath-symbolics`` distribution exposes the same symbolic surface under
    ``sage.all__sagemath_symbolics`` instead, so either name is accepted.
    """
    for name in ("sage.all", "sage.all__sagemath_symbolics"):
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            continue
    return None


def demo_sage_export(g_expr):
    """Section 4 (optional): lower the same expression to Sage; skipped gracefully if unavailable."""
    print("\n4. Optional Sage export (skipped automatically if Sage isn't installed):")
    if _sage_module_name() is None:
        print("   sage not found -- skipping (pip install mixle[sage] to enable this section).")
        return
    sage_expr = to_sage(g_expr)
    print(f"   to_sage(Gaussian log-density) -> {type(sage_expr).__module__}.{type(sage_expr).__qualname__}")
    print(f"   {sage_expr}")


def main():
    print("# mixle symbolic export: closed-form log-densities as LaTeX / SymPy / Sage\n")
    x = SYMBOLIC_ENGINE.symbol("x")
    gaussian, student_t = build_distributions()
    g_expr, t_expr = demo_instance_log_density(gaussian, student_t, x)
    demo_general_formula(x)
    demo_score_function(g_expr, t_expr)
    demo_sage_export(g_expr)


if __name__ == "__main__":
    main()
