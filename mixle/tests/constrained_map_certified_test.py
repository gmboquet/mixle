"""The constrained MAP fit reaches the constrained optimum or fails, never a nearly constant vector.

0.8.1 adversarial review P04-F01..F03: the Powell penalty ramp converged to within 0.02 of the
isotonic or convex-regression target and then the feasibility repair, whose step is the sum of the
violated constraints' normals, could not clear overlapping tied constraints; its fallback returned the
starting vector (the grand mean), the walled Nelder-Mead polish could not move off the wall, and the
fit came back ``ok`` with a constant mean 221 nats worse than the optimum at ten dimensions. Powell's
own convergence flag was never read, so ``max_iter=1`` returned a wrong point as a success, and at a
fully tied corner the polish stopped 0.005 to 0.03 short of the pooled mean.

Now an inequality-constrained fit whose constraints carry signed margins (the comparison relations and
every shape constraint) is solved by SLSQP from a strictly feasible start; the result must be feasible
and no worse than SLSQP's own value; the penalty ramp remains for constraints without margins and
certifies what it returns against its penalized bound. These tests pin the targets a hand computation
gives: pool-adjacent-violators for monotone fits, an SLSQP convex regression for curvature fits.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np
from scipy.optimize import minimize

from mixle.ppl import DiagGaussian, Normal, concave, convex, decreasing, free, increasing, lipschitz
from mixle.ppl.inference import _step_into_margins


def pool_adjacent_violators(y):
    """Weighted isotonic (non-decreasing) regression of equally weighted ``y``."""
    blocks = [[float(v), 1.0, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-15:
            a, b = blocks[i], blocks[i + 1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for mean, _weight, count in blocks:
        out += [mean] * count
    return np.array(out)


def curvature_regression(y, sign):
    """Least-squares projection of ``y`` onto the vectors with second differences of ``sign``."""
    y = np.asarray(y, dtype=float)
    d = len(y)
    cons = [{"type": "ineq", "fun": (lambda v, k=k: sign * (v[k + 2] - 2 * v[k + 1] + v[k]))} for k in range(d - 2)]
    result = minimize(
        lambda v: np.sum((v - y) ** 2), y, constraints=cons, method="SLSQP", options={"ftol": 1e-14, "maxiter": 5000}
    )
    return result.x


def column_data(locs, *, n=200, sd=0.5, seed=0):
    rng = np.random.RandomState(seed)
    return np.stack([rng.normal(loc, sd, n) for loc in locs], axis=1)


def constrained_fit(X, constraint, *, seed=1, **kw):
    d = X.shape[1]
    v = free(d, name="v")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = DiagGaussian(d, mean=v, var=np.full(d, 0.25)).fit(
            X.tolist(), how="map", constraints=[constraint(v)], rng=np.random.RandomState(seed), **kw
        )
    return np.asarray(fit.params["mean"], dtype=float)


class ReachesTheHandComputedTargetTest(unittest.TestCase):
    def test_isotonic_fits_match_pool_adjacent_violators_up_to_twenty_dimensions(self):
        for d in (2, 5, 8, 10, 15, 20):
            for scenario in ("reversed", "one swap", "random permutation"):
                if scenario == "reversed":
                    locs = np.arange(d)[::-1].astype(float)
                elif scenario == "one swap":
                    locs = np.arange(d).astype(float)
                    if d >= 3:
                        locs[1], locs[2] = locs[2], locs[1]
                else:
                    locs = np.random.RandomState(100 + d).permutation(d).astype(float) * 0.3
                X = column_data(locs, seed=d)
                means = X.mean(axis=0)
                with self.subTest(d=d, scenario=scenario, constraint="increasing"):
                    fit = constrained_fit(X, increasing)
                    np.testing.assert_allclose(fit, pool_adjacent_violators(means), atol=1e-3)
                    self.assertTrue(np.all(np.diff(fit) >= -1e-9))
                with self.subTest(d=d, scenario=scenario, constraint="decreasing"):
                    fit = constrained_fit(X, decreasing)
                    np.testing.assert_allclose(fit, -pool_adjacent_violators(-means), atol=1e-3)
                    self.assertTrue(np.all(np.diff(fit) <= 1e-9))

    def test_curvature_fits_match_the_convex_regression_projection(self):
        for d in (5, 7, 10, 12, 20):
            locs = np.random.RandomState(200 + d).permutation(d).astype(float) * 0.3
            X = column_data(locs, seed=d)
            means = X.mean(axis=0)
            with self.subTest(d=d, constraint="convex"):
                fit = constrained_fit(X, convex)
                np.testing.assert_allclose(fit, curvature_regression(means, 1.0), atol=2e-3)
                self.assertTrue(np.all(np.diff(fit, 2) >= -1e-9))
            with self.subTest(d=d, constraint="concave"):
                fit = constrained_fit(X, concave)
                np.testing.assert_allclose(fit, curvature_regression(means, -1.0), atol=2e-3)
                self.assertTrue(np.all(np.diff(fit, 2) <= 1e-9))

    def test_fully_tied_corner_lands_on_the_pooled_mean(self):
        # P04-F03: decreasing(v) on increasing column means, the optimum is the constant grand-mean
        # vector and every constraint is active; the old polish stopped 2-3 standard errors short.
        for d, seed in ((5, 0), (5, 1), (10, 2)):
            X = column_data(2.0 * np.arange(d), seed=seed)
            fit = constrained_fit(X, decreasing)
            np.testing.assert_allclose(fit, np.full(d, X.mean()), atol=1e-5)

    def test_reviewers_ten_dimensional_case_is_not_the_grand_mean(self):
        rng = np.random.RandomState(101)
        locs = rng.permutation(10).astype(float) * 0.3
        X = np.stack([rng.normal(loc, 0.5, 200) for loc in locs], axis=1)
        fit = constrained_fit(X, increasing)
        target = pool_adjacent_violators(X.mean(axis=0))
        np.testing.assert_allclose(fit, target, atol=1e-3)
        self.assertGreater(np.ptp(fit), 0.4)  # three pooled blocks, not one constant vector

    def test_lipschitz_and_interior_optimum_are_untouched(self):
        X = column_data([0.0, 0.3, 0.5, 0.9, 1.0], seed=3)
        means = X.mean(axis=0)
        fit = constrained_fit(X, increasing)
        np.testing.assert_allclose(fit, means, atol=1e-4)  # already increasing: the MLE
        fit = constrained_fit(X, lambda v: lipschitz(v, 0.2))
        self.assertTrue(np.all(np.abs(np.diff(fit)) <= 0.2 + 1e-7))
        self.assertLess(np.sum((fit - means) ** 2), np.sum((np.full(5, means.mean()) - means) ** 2))


class FailsHonestlyTest(unittest.TestCase):
    def test_exhausted_budget_raises_naming_max_iter(self):
        # P04-F02: the unconstrained path raised at max_iter=1; the constrained path returned a wrong
        # point as a success.
        X = column_data([0.0, 2.0, 1.0, 3.0, 4.0], n=300)
        with self.assertRaises(RuntimeError) as caught:
            constrained_fit(X, increasing, seed=0, max_iter=1)
        self.assertIn("max_iter", str(caught.exception))
        y = list(np.random.RandomState(0).normal(5, 2, 400))
        with self.assertRaises(RuntimeError):
            Normal(Normal(0, 10, name="mu"), free).fit(y, how="map", max_iter=1)


class MarginsTest(unittest.TestCase):
    def test_shape_and_comparison_constraints_carry_signed_margins(self):
        v = free(4, name="v")
        a = Normal(0, 1, name="a")
        b = Normal(0, 1, name="b")
        env = {v: np.array([0.0, 1.0, 3.0, 2.0])}
        np.testing.assert_allclose(increasing(v).margin(env), [1.0, 2.0, -1.0])
        np.testing.assert_allclose(decreasing(v).margin(env), [-1.0, -2.0, 1.0])
        np.testing.assert_allclose(convex(v).margin(env), [1.0, -3.0])
        np.testing.assert_allclose(concave(v).margin(env), [-1.0, 3.0])
        np.testing.assert_allclose(lipschitz(v, 1.5).margin(env), [0.5, -0.5, 2.5, 2.5, 3.5, 0.5])
        np.testing.assert_allclose((a < b).margin({a: 1.0, b: 3.0}), 2.0)
        np.testing.assert_allclose((increasing(v) & (a < b)).margin({**env, a: 1.0, b: 3.0}), [1.0, 2.0, -1.0, 2.0])
        self.assertIsNone((increasing(v) | decreasing(v)).margin)
        self.assertIsNone((~increasing(v)).margin)
        self.assertIsNone(a.eq(b).margin)

    def test_step_into_margins_lifts_every_tied_second_difference_at_once(self):
        # the sum-of-normals repair cannot: adjacent second-difference normals cancel on the interior
        def margins(u):
            return np.diff(u, 2)

        start = np.full(10, 3.0)
        lifted, ok = _step_into_margins(start, margins, 1.0e-7)
        self.assertTrue(ok)
        self.assertTrue(np.all(margins(lifted) >= 0.5e-7))
        self.assertLess(np.max(np.abs(lifted - start)), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
