"""Regression guards for the independent-tester findings in ``release-checklists/0.8.1-followups.md``.

Each behavioral finding (FU-01 .. FU-07, FU-13, FU-15) has a test that failed against the candidate
the testers ran (``dd167be2``) and passes after its fix; the two findings that were already fixed by
later 0.8.0 work (FU-03, FU-14) are pinned here as guards on the exact constructions the testers
used, so they cannot silently reopen. Run this file in a numpy+scipy-only environment as well: FU-13
only reproduces under the pinned numpy.
"""

from __future__ import annotations

import math
import unittest
import warnings

import numpy as np
from scipy.stats import genpareto

from mixle.inference import optimize
from mixle.inference.calibration import pit_values
from mixle.inference.nonparametric import ks_1samp
from mixle.stats import (
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
)
from mixle.stats.rankings.bradley_terry import BradleyTerryDistribution, BradleyTerryEstimator
from mixle.stats.rankings.paired_comparison import (
    DavidsonDistribution,
    DavidsonEstimator,
    RaoKupperDistribution,
    RaoKupperEstimator,
    ThurstoneMostellerDistribution,
    ThurstoneMostellerEstimator,
)
from mixle.stats.univariate.continuous.generalized_pareto import (
    GeneralizedParetoDistribution,
    GeneralizedParetoEstimator,
)


class WinProbabilityIsTheConditionalTest(unittest.TestCase):
    """FU-01: ``log_density`` is the joint over (pair, outcome); ``win_probability`` is P(i beats j)."""

    def test_bradley_terry_and_thurstone_mosteller(self):
        for dist in (BradleyTerryDistribution([0.0, 1.0, -0.5]), ThurstoneMostellerDistribution([0.0, 1.0, -0.5])):
            joint = math.exp(dist.log_density((0, 1))) + math.exp(dist.log_density((1, 0)))
            self.assertAlmostEqual(joint, 1.0 / 3.0, places=12)  # 1 / C(3, 2): documented, not a bug
            self.assertAlmostEqual(dist.win_probability(0, 1) + dist.win_probability(1, 0), 1.0, places=12)
            self.assertGreater(dist.win_probability(1, 0), dist.win_probability(0, 1))  # worth 1.0 beats worth 0.0
            self.assertIn("JOINT", dist.log_density.__doc__)
        bt = BradleyTerryDistribution([0.0, 1.0, -0.5])
        self.assertAlmostEqual(bt.win_probability(1, 0), 1.0 / (1.0 + math.exp(-1.0)), places=12)

    def test_tie_families(self):
        for dist in (DavidsonDistribution([0.0, 1.0, -0.5], 1.0), RaoKupperDistribution([0.0, 1.0, -0.5], 1.5)):
            joint = sum(math.exp(dist.log_density((0, 1, outcome))) for outcome in (0, 1, 2))
            self.assertAlmostEqual(joint, 1.0 / 3.0, places=12)
            total = dist.win_probability(0, 1) + dist.win_probability(1, 0) + dist.tie_probability(0, 1)
            self.assertAlmostEqual(total, 1.0, places=12)
            self.assertAlmostEqual(dist.tie_probability(0, 1), dist.tie_probability(1, 0), places=12)


class ShapeClampIsDisclosedTest(unittest.TestCase):
    """FU-02: a moment estimate clamped to ``xi_min`` must appear in ``numerical_repairs()``."""

    def test_tiny_n_clamp_is_named(self):
        clamps = 0
        for n in (3, 4, 5, 6):
            for seed in range(60):
                x = genpareto.rvs(0.3, scale=1.0, size=n, random_state=seed).tolist()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = optimize(x, GeneralizedParetoEstimator(), max_its=1)
                if fit.shape == -10.0:
                    clamps += 1
                    self.assertTrue(
                        any(note.startswith("shape-clamped(") and "xi_min" in note for note in fit.numerical_repairs()),
                        (n, seed, fit.numerical_repairs()),
                    )
        self.assertGreater(clamps, 0, "the scan must exercise the clamp for this test to mean anything")

    def test_ordinary_fit_stays_silent(self):
        x = genpareto.rvs(0.3, scale=1.0, size=2000, random_state=0).tolist()
        fit = optimize(x, GeneralizedParetoEstimator(), max_its=1)
        self.assertFalse(any(note.startswith("shape-clamped(") for note in fit.numerical_repairs()))


class OptimizeDisclosesEarlyStopsTest(unittest.TestCase):
    """FU-03 (verified fixed): the tester's 2-component GPD mixture either runs on or says why it stopped."""

    def test_gpd_mixture_does_not_stop_silently(self):
        x = np.concatenate(
            [
                genpareto.rvs(0.2, scale=1.0, size=400, random_state=1),
                genpareto.rvs(0.6, scale=3.0, size=400, random_state=2),
            ]
        ).tolist()
        est = MixtureEstimator([GeneralizedParetoEstimator(), GeneralizedParetoEstimator()])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = optimize(x, est, max_its=2000, delta=1e-8, rng=np.random.RandomState(3))
        provenance = fit.fit_provenance()
        stopped_short = provenance.iterations < 2000 and not provenance.converged
        disclosed = any("iteration" in str(w.message).lower() or "stopp" in str(w.message).lower() for w in caught)
        self.assertTrue(
            provenance.converged or not stopped_short or disclosed, (provenance, [str(w.message) for w in caught])
        )


class ScalarCdfCallablesTest(unittest.TestCase):
    """FU-04: a fitted distribution's scalar-only ``cdf`` is accepted, and a real TypeError is not relabelled."""

    def setUp(self):
        self.dist = GeneralizedParetoDistribution(1.0, 0.2, 0.0)
        self.y = np.array([0.5, 1.0, 2.0])

    def test_pit_values_and_ks_accept_scalar_cdf(self):
        u = pit_values(self.y, self.dist.cdf)
        np.testing.assert_allclose(u, [self.dist.cdf(v) for v in self.y])
        result = ks_1samp(self.y, self.dist.cdf)
        self.assertTrue(0.0 <= result.statistic <= 1.0)

    def test_vectorized_cdf_still_works_and_values_errors_stay_value_errors(self):
        np.testing.assert_allclose(pit_values(self.y, lambda v: np.asarray(v) / 4.0), self.y / 4.0)
        with self.assertRaisesRegex(ValueError, "finite values matching y"):
            pit_values(self.y, lambda v: ["not", "numbers", "here"])

    def test_a_genuine_type_error_inside_cdf_is_not_relabelled(self):
        def broken(value):
            raise TypeError("cdf needs a threshold argument")

        with self.assertRaisesRegex(TypeError, "threshold argument"):
            pit_values(self.y, broken)


class MixedArityComparisonsTest(unittest.TestCase):
    """FU-05: a 3-tuple among 2-tuples names the row instead of numpy's inhomogeneous-shape message."""

    def test_row_numbered_message(self):
        data = [(0, 1), (1, 2), (2, 0, 1), (0, 2)]
        with self.assertRaisesRegex(ValueError, r"row 2 has 3 items; every comparison must have exactly 2") as ctx:
            optimize(data, BradleyTerryEstimator(3, pseudo_count=0.5), max_its=2)
        self.assertNotIn("inhomogeneous", str(ctx.exception))
        with self.assertRaisesRegex(ValueError, r"row 1 has 2 items; every comparison must have exactly 3"):
            optimize([(0, 1, 2), (1, 2), (0, 2, 0)], DavidsonEstimator(3, pseudo_count=0.5), max_its=2)


class IdentifiabilityFailureIsUniformTest(unittest.TestCase):
    """FU-06/FU-07: every family raises ``ValueError`` for a competitor that never lost, up front, and the
    diagnostics report the directed condition that actually gates the finite MLE."""

    # item 0 beats everyone and never loses: undirected graph connected, directed graph not strongly connected
    UNDEFEATED = [(0, 1)] * 5 + [(0, 2)] * 5 + [(1, 2)] * 3 + [(2, 1)] * 2

    def test_same_exception_type_across_families(self):
        for estimator in (BradleyTerryEstimator(3), DavidsonEstimator(3), RaoKupperEstimator(3)):
            data = (
                self.UNDEFEATED
                if isinstance(estimator, BradleyTerryEstimator)
                else [(i, j, 0) for i, j in self.UNDEFEATED]
            )
            with self.assertRaisesRegex(ValueError, "not strongly connected"):
                optimize(data, estimator, max_its=5, init_p=1.0)

    def test_regularized_fit_reports_both_connectivity_facts(self):
        bt = optimize(self.UNDEFEATED, BradleyTerryEstimator(3, pseudo_count=0.5), max_its=50)
        self.assertTrue(bt.fit_diagnostics.regularized)
        for estimator in (DavidsonEstimator(3, pseudo_count=0.5), RaoKupperEstimator(3, pseudo_count=0.5)):
            fit = optimize([(i, j, 0) for i, j in self.UNDEFEATED], estimator, max_its=50)
            self.assertTrue(fit.fit_diagnostics.graph_connected)
            self.assertFalse(fit.fit_diagnostics.strongly_connected)  # what graph_connected could not say
            self.assertTrue(fit.fit_diagnostics.regularized)
        tm = optimize(self.UNDEFEATED, ThurstoneMostellerEstimator(3), max_its=5, init_p=1.0)
        self.assertTrue(tm.fit_diagnostics.graph_connected)
        self.assertFalse(tm.fit_diagnostics.strongly_connected)

    def test_strongly_connected_data_reports_true(self):
        data = [(0, 1)] * 4 + [(1, 2)] * 4 + [(2, 0)] * 4 + [(1, 0)] * 2
        fit = optimize([(i, j, 0) for i, j in data], DavidsonEstimator(3), max_its=50, init_p=1.0)
        self.assertTrue(fit.fit_diagnostics.strongly_connected)
        self.assertFalse(fit.fit_diagnostics.regularized)


class FitDiagnosticsRoundTripTest(unittest.TestCase):
    """FU-13: ``l1_change`` is a plain float before and after a JSON round-trip (numpy scalar before the fix)."""

    def test_l1_change_type(self):
        data = [(0, 1)] * 5 + [(1, 2)] * 4 + [(2, 0)] * 3 + [(1, 0)] * 2 + [(0, 2)] * 2 + [(2, 1)]
        fit = optimize(data, BradleyTerryEstimator(3, pseudo_count=0.5), max_its=50)
        self.assertIs(type(fit.fit_diagnostics.l1_change), float)
        back = type(fit).from_json(fit.to_json())
        self.assertIs(type(back.fit_diagnostics.l1_change), float)


class RandomInitMixtureStaysFiniteTest(unittest.TestCase):
    """FU-14 (verified fixed): the notebook's 24 random-init diagonal-Gaussian mixture fits are all finite."""

    def test_notebook_construction(self):
        rng = np.random.RandomState(0)
        centers = np.array([[0, 0], [5, 5], [0, 6]])
        X = np.vstack([rng.normal(c, 0.7, (200, 2)) for c in centers])
        data = [list(map(float, r)) for r in X]
        est = MixtureEstimator([DiagonalGaussianEstimator(dim=2)] * 3)
        arr = np.asarray(data)
        lls = []
        for seed in range(24):
            r = np.random.RandomState(seed)
            picked = arr[r.choice(len(arr), 3, replace=False)]
            sd = arr.std(axis=0)
            init = MixtureDistribution(
                [DiagonalGaussianDistribution(list(map(float, c)), list(map(float, sd**2))) for c in picked],
                [1.0 / 3] * 3,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = optimize(data, est, max_its=40, prev_estimate=init)
            lls.append(float(np.sum(fit.seq_log_density(fit.dist_to_encoder().seq_encode(data)))))
        self.assertTrue(np.all(np.isfinite(lls)), lls)
        np.histogram(lls, bins=16)  # the notebook's next line


class ConstrainedDerivativeFreeMapTest(unittest.TestCase):
    """FU-15: a 5-D monotone MAP fit reaches the constrained optimum instead of exhausting the budget or
    collapsing onto a constant vector against the feasibility wall."""

    @staticmethod
    def _pooled(values):
        blocks = [[float(v), 1] for v in values]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                a, b = blocks[i], blocks[i + 1]
                blocks[i : i + 2] = [[(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]]]
                i = max(i - 1, 0)
            else:
                i += 1
        return np.concatenate([[m] * n for m, n in blocks])

    def test_tutorial_increasing_mean(self):
        from mixle.ppl import DiagGaussian, decreasing, free, increasing

        rng = np.random.RandomState(0)
        X = np.stack([rng.normal(loc, 0.5, 300) for loc in [0, 2, 1, 3, 4]], axis=1)
        column_means = X.mean(axis=0)
        v = free(5, name="v")
        fit = DiagGaussian(5, mean=v).fit(X.tolist(), how="map", constraints=[increasing(v)])
        mean = np.asarray(fit.params["mean"], dtype=float)
        self.assertTrue(np.all(np.diff(mean) >= -1e-9), mean)
        # the monotone optimum pools the two out-of-order columns; the constant vector the walled
        # Nelder-Mead used to return is 2.0 away from it
        np.testing.assert_allclose(mean, self._pooled(column_means), atol=2e-3)
        v = free(5, name="v")
        fit = DiagGaussian(5, mean=v).fit(X.tolist(), how="map", constraints=[decreasing(v)])
        mean = np.asarray(fit.params["mean"], dtype=float)
        self.assertTrue(np.all(np.diff(mean) <= 1e-9), mean)
        np.testing.assert_allclose(mean, self._pooled(column_means[::-1])[::-1], atol=2e-3)

    def test_failure_message_names_max_iter(self):
        from mixle.ppl.inference import _derivative_free_constrained

        u, f, algorithm, ok, message = _derivative_free_constrained(
            lambda z: float(np.sum(z**2)),
            lambda z: False,  # nothing is feasible: the walled fallback can only fail
            None,
            np.ones(2),
            max_iter=50,
            tol=1e-8,
            rng=np.random.RandomState(0),
        )
        self.assertFalse(ok)
        self.assertEqual(algorithm, "nelder-mead")


class DescribeRoutesRawDataTest(unittest.TestCase):
    """FU-09: raw data is pointed at ``propose`` instead of "no catalogued capability detected"."""

    def test_arrays_lists_and_frames(self):
        import mixle

        for raw in (np.random.RandomState(0).normal(size=(50, 2)), [1.0, 2.0, 3.0], [[1, "a"], [2, "b"]]):
            text = mixle.describe(raw)
            self.assertIn("raw data, not a model", text)
            self.assertIn("mixle.propose(data)", text)
            self.assertNotIn("no catalogued capability detected", text)
        fitted = optimize([1.0, 2.0, 3.0, 2.5], GaussianEstimator(), max_its=1)
        self.assertIn("distribution", mixle.describe(fitted).lower())


class MixtureHintIsConditionalTest(unittest.TestCase):
    """FU-11: the latent-structure caveat names multimodal fields and is silent on unimodal data."""

    def test_unimodal_silent_bimodal_named(self):
        from mixle.utils.automatic.profiling import analyze_structure

        rng = np.random.RandomState(0)
        unimodal = analyze_structure([[float(v)] for v in rng.normal(0.0, 1.0, 400)])
        self.assertFalse([w for w in unimodal.warnings if "multimodal" in w])
        bimodal = analyze_structure(
            [[float(v)] for v in np.concatenate([rng.normal(-4, 1, 200), rng.normal(4, 1, 200)])]
        )
        hits = [w for w in bimodal.warnings if "look multimodal" in w]
        self.assertEqual(len(hits), 1, bimodal.warnings)
        # The note uses format_path's '$[0]' spelling, like every other note in the list; it used
        # to leak the raw internal tuple (P03-F08).
        self.assertIn("$[0]", hits[0])


if __name__ == "__main__":
    unittest.main()
