"""The diagonal Gaussian's variance floor must be priced per coordinate, not off the mean (T1-03).

``DiagonalGaussianEstimator`` floored EVERY coordinate at ``1e-6 * mean(var)`` -- a floor relative to
the mean of all the columns' variances rather than to each column's own scale. A diagonal covariance
exists to let coordinates keep their own scales, so a floor priced off their mean couples them: on a
plainly ordinary probability/dollars/count table (variance spread only 9.1e6) the smallest variance
came back 3.03x too wide, and on metre/kilometre-style data 3.2e17x too wide, while
``MultivariateGaussianEstimator`` -- whose ridge was made per-coordinate for exactly this reason
(T1-05) -- returned the maximum-likelihood variances from the identical input.

The consequences pinned here are not cosmetic: the inflated column inverted an AIC model comparison
on data whose columns are independent (the diagonal family is the CORRECT one there, and it lost to
the over-parameterized full covariance by 346 AIC), and it cost a two-component diagonal mixture 68
nats with an empty ``numerical_repairs()`` on the flagship ``optimize()`` path.

Both directions of the safeguard are pinned. A coordinate with no positive variance of its own must
still be lifted -- a zero variance is an infinite density -- with the same manufactured jitter as
before and a ``variance-floored(...)`` record; and a scale-homogeneous fit, an explicit ``min_covar``
and a degenerate column must be bit-identical to the previous behaviour.

Also pinned: the class docstring's shift-equivariance guarantee. It promised the variances of
``x + c`` "to within a few ulps", while the measured worst case in the band just below the
conditioning gate is ~3e-8 relative -- roughly 1e8 ulps. The docstring now states a real relative
bound, and this file holds the measurement to it (T1-07).
"""

import unittest
from fractions import Fraction

import numpy as np

from mixle.inference import optimize
from mixle.stats import (
    DiagonalGaussianEstimator,
    MixtureEstimator,
    MultivariateGaussianEstimator,
)


def _exact_variances(x):
    """Population variance of each column computed exactly in rationals, so the reference is not
    itself a floating-point estimate of the quantity under test."""
    out = []
    for j in range(x.shape[1]):
        values = [Fraction(float(v)) for v in x[:, j]]
        n = len(values)
        mean = sum(values) / n
        out.append(float(sum((v - mean) ** 2 for v in values) / n))
    return np.asarray(out)


def _fit(x, **kwargs):
    """Fit through the accumulate + estimate path (the one an engine and ``optimize`` both drive)."""
    estimator = DiagonalGaussianEstimator(dim=x.shape[1], **kwargs)
    accumulator = estimator.accumulator_factory().make()
    for row in x:
        accumulator.update(row, 1.0, None)
    return estimator.estimate(None, accumulator.value())


def _spread_data(power, n=400, seed=77):
    """Three unit-scale columns re-expressed in units ``1e-power``, 1, ``1e+power``."""
    z = np.random.default_rng(seed).normal(size=(n, 3))
    return z * np.array([10.0**-power, 1.0, 10.0**power])


def _ordinary_table(n=800, seed=2024):
    """A probability, a price and a count: no engineered scales, variance spread only 9.1e6."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.beta(2.0, 5.0, n),
            rng.lognormal(6.0, 0.8, n),
            rng.poisson(30.0, n).astype(float),
        ]
    )


def _loglik(model, rows):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(rows))))


class PerCoordinateVarianceFloorTest(unittest.TestCase):
    """A coordinate with its own positive variance keeps the maximum-likelihood value."""

    def test_a_heterogeneous_unit_fit_returns_the_exact_per_column_mle(self):
        for power in (2, 3, 4, 6):
            with self.subTest(power=power):
                x = _spread_data(power)
                fitted = _fit(x)
                # Before: [3.19e+05x, 1, 1] at power=3 and [3.19e+17x, 3.33e+05x, 1] at power=6.
                np.testing.assert_allclose(np.asarray(fitted.covar), _exact_variances(x), rtol=1e-12)
                self.assertEqual(fitted.numerical_repairs(), ())

    def test_an_ordinary_probability_price_count_table_is_not_inflated(self):
        x = _ordinary_table()
        fitted = _fit(x)
        # Before: the probability column came back 3.02699562x too wide, with a repair recorded.
        np.testing.assert_allclose(np.asarray(fitted.covar), _exact_variances(x), rtol=1e-12)
        self.assertEqual(fitted.numerical_repairs(), ())

    def test_the_optimize_path_agrees_with_the_accumulate_path(self):
        x = _ordinary_table()
        rows = [tuple(row) for row in x]
        fitted = optimize(rows, DiagonalGaussianEstimator(dim=3), max_its=5)
        np.testing.assert_allclose(np.asarray(fitted.covar), _exact_variances(x), rtol=1e-12)
        self.assertEqual(fitted.numerical_repairs(), ())

    def test_the_diagonal_and_full_covariance_estimators_agree_on_the_same_data(self):
        # Two estimators of the same family disagreed by up to 3.2e17x on identical input, so any
        # model comparison between them was decided by the artifact rather than by the data.
        x = _spread_data(6)
        rows = [tuple(row) for row in x]
        diagonal = np.asarray(_fit(x).covar)
        full = np.diag(np.asarray(optimize(rows, MultivariateGaussianEstimator(dim=3), max_its=5).covar))
        # The full covariance adds its own ~1e-6 relative per-coordinate ridge; that is the whole gap.
        np.testing.assert_allclose(diagonal, full / (1.0 + 1.0e-6), rtol=1e-9)

    def test_the_floor_is_equivariant_under_a_single_column_change_of_units(self):
        # The defect in one line: rescaling ONE column moved the OTHER columns' fitted variances.
        rng = np.random.default_rng(19)
        x = rng.standard_normal((300, 3)) * np.array([1.0, 1.0, 1e5])
        scale = 2.0**-40  # a power of two, so the rescaling itself is exact in binary floating point
        rescaled = x * np.array([scale, 1.0, 1.0])
        base = np.asarray(_fit(x).covar)
        moved = np.asarray(_fit(rescaled).covar)
        self.assertEqual(list(moved[1:]), list(base[1:]))  # untouched columns, bit for bit
        self.assertEqual(moved[0], base[0] * scale * scale)


class ModelSelectionTest(unittest.TestCase):
    """The floor decided model comparisons that the data should have decided."""

    def test_the_diagonal_family_wins_on_independent_columns(self):
        x = _ordinary_table()
        rows = [tuple(row) for row in x]
        diagonal = optimize(rows, DiagonalGaussianEstimator(dim=3), max_its=5)
        full = optimize(rows, MultivariateGaussianEstimator(dim=3), max_its=5)
        aic_diagonal = 2 * 6 - 2 * _loglik(diagonal, rows)
        aic_full = 2 * 9 - 2 * _loglik(full, rows)
        # Before: 16967.23 vs 16621.16 -- the over-parameterized model beat the correct family by
        # 346 AIC on columns that were drawn independently, purely because of the floor.
        self.assertLess(aic_diagonal, aic_full)

    def test_a_diagonal_mixture_fit_loses_no_likelihood_to_the_floor(self):
        x = _ordinary_table()
        rows = [tuple(row) for row in x]
        fits = {}
        for label, kwargs in (("default", {}), ("escape", {"ridge": 0.0, "min_covar": 1e-30})):
            estimator = MixtureEstimator([DiagonalGaussianEstimator(dim=3, **kwargs) for _ in range(2)])
            model = optimize(rows, estimator, max_its=60, delta=None, rng=np.random.RandomState(4))
            fits[label] = (_loglik(model, rows), model)
        # Before: the default fit sat 67.7 nats below the escape hatch, with a
        # variance-floored(...) record on one component and no warning from optimize().
        self.assertAlmostEqual(fits["default"][0], fits["escape"][0], places=6)
        self.assertEqual(
            [component.numerical_repairs() for component in fits["default"][1].components],
            [(), ()],
        )


class DegenerateCoordinateStillFlooredTest(unittest.TestCase):
    """The narrow case the mean-priced jitter exists for must keep working, and keep being disclosed."""

    def test_a_constant_column_is_lifted_and_recorded(self):
        rng = np.random.default_rng(3)
        x = np.column_stack([rng.standard_normal(80), np.full(80, 7.0)])
        fitted = _fit(x)
        covar = np.asarray(fitted.covar)
        self.assertGreater(float(covar[1]), 0.0)
        # The jitter is priced off the coordinates that DO have a variance -- unchanged behaviour.
        self.assertAlmostEqual(float(covar[1]), 1.0e-6 * float(covar[0]), delta=1e-18)
        self.assertEqual(
            fitted.numerical_repairs(),
            ("variance-floored(1.18e-06; onto a non-positive variance)",),
        )
        # ... and the same clamp reaches the fit receipt through the ``optimize`` path.
        provenance = optimize([tuple(row) for row in x], DiagonalGaussianEstimator(dim=2), max_its=5).fit_provenance()
        self.assertIsNotNone(provenance)
        self.assertTrue(any("variance-floored" in repair for repair in provenance.repairs), provenance.repairs)

    def test_data_with_no_scale_at_all_falls_back_to_the_absolute_floor(self):
        fitted = _fit(np.zeros((40, 2)))
        np.testing.assert_array_equal(np.asarray(fitted.covar), np.full(2, 1.0e-8))
        self.assertTrue(any("variance-floored" in repair for repair in fitted.numerical_repairs()))

    def test_a_single_point_component_is_lifted_on_every_coordinate(self):
        # The EM case the floor exists for: one observation implies zero spread everywhere.
        fitted = _fit(np.asarray([[3.0, -2.0, 11.0]]))
        self.assertTrue(np.all(np.asarray(fitted.covar) > 0.0))
        self.assertTrue(any("variance-floored" in repair for repair in fitted.numerical_repairs()))

    def test_an_explicit_min_covar_is_still_a_hard_per_coordinate_lower_bound(self):
        rng = np.random.default_rng(23)
        x = rng.standard_normal((200, 2)) * 1.0e-6
        fitted = _fit(x, min_covar=1.0e-3)
        np.testing.assert_array_equal(np.asarray(fitted.covar), np.full(2, 1.0e-3))
        self.assertTrue(any("variance-floored" in repair for repair in fitted.numerical_repairs()))

    def test_an_explicit_min_covar_is_not_raised_by_the_other_columns_scale(self):
        # ``mixle.task`` passes min_covar=1e-3 and asks for a floor of 1e-3. With the mean-priced
        # rule a large column silently raised that floor to 33.2 on both small columns.
        x = _spread_data(2, seed=77) * np.array([1e2, 1e2, 1e2])
        fitted = _fit(x, min_covar=1.0e-3)
        exact = _exact_variances(x)
        np.testing.assert_allclose(np.asarray(fitted.covar), np.maximum(exact, 1.0e-3), rtol=1e-12)


class UnchangedOnOrdinaryDataTest(unittest.TestCase):
    """The scale-homogeneous fit must be exactly what it always was."""

    def test_a_scale_homogeneous_fit_is_the_raw_mle_and_stays_silent(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((200, 3))
        fitted = _fit(x)
        np.testing.assert_allclose(np.asarray(fitted.covar), _exact_variances(x), rtol=1e-13)
        self.assertEqual(fitted.numerical_repairs(), ())

    def test_a_small_scale_homogeneous_fit_is_the_raw_mle(self):
        rng = np.random.default_rng(13)
        x = rng.standard_normal((200, 3)) * 1.0e-6
        fitted = _fit(x)
        np.testing.assert_allclose(np.asarray(fitted.covar), _exact_variances(x), rtol=1e-13)
        self.assertEqual(fitted.numerical_repairs(), ())


class ShiftEquivarianceBoundTest(unittest.TestCase):
    """The docstring's guarantee, held to the number it now states."""

    def _fitted_and_exact(self, x):
        estimator = DiagonalGaussianEstimator(dim=x.shape[1])
        factory = estimator.accumulator_factory()
        accumulator = factory.make()
        encoded = factory.make().acc_to_encoder().seq_encode([tuple(row) for row in x])
        accumulator.seq_update(encoded, np.ones(len(x)), None)
        fitted = np.asarray(estimator.estimate(None, accumulator.value()).covar)
        return fitted, _exact_variances(x)

    def test_the_worst_case_below_the_conditioning_gate_is_within_the_stated_bound(self):
        # mean^2 / variance just under the gate's 4e6: the raw E[x^2] - mu^2 form is still in use.
        base = np.random.default_rng(1).standard_normal((2000, 3)) * 3.0
        fitted, exact = self._fitted_and_exact(base + 3.0 * 1950.0)
        error = float(np.max(np.abs(fitted - exact) / exact))
        self.assertLess(error, 1.0e-7)  # the bound the class docstring states
        # ... and it is emphatically NOT "a few ulps", which is what the docstring used to promise.
        self.assertGreater(error, 1.0e-12)

    def test_the_anchored_track_is_exact_at_epoch_seconds(self):
        base = np.random.default_rng(1).standard_normal((2000, 3)) * 3.0
        for offset in (1.7e9, 1.0e10):
            with self.subTest(offset=offset):
                fitted, exact = self._fitted_and_exact(base + offset)
                self.assertLess(float(np.max(np.abs(fitted - exact) / exact)), 1.0e-13)


class DocumentedPolicyTest(unittest.TestCase):
    """The class docstring documented shift-equivariance and nothing about the ridge it applies.

    A reader comparing it with ``MultivariateGaussianEstimator`` -- which documents its ridge in five
    paragraphs -- would reasonably conclude this estimator has no ridge at all.
    """

    def test_the_class_docstring_states_the_variance_floor_policy(self):
        doc = DiagonalGaussianEstimator.__doc__
        self.assertIn("ridge * var_i", doc)
        self.assertIn("min_covar", doc)
        self.assertIn("variance-floored", doc)

    def test_the_class_docstring_states_a_real_shift_equivariance_bound(self):
        doc = DiagonalGaussianEstimator.__doc__
        self.assertIn("1e-7", doc)
        self.assertNotIn("to within a few\n    ulps", doc)


if __name__ == "__main__":
    unittest.main()
