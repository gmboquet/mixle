"""Second-wave campaign regressions for the stats families: T1-05 and T4-9.

T1-05: ``MultivariateGaussianEstimator``'s default relative ridge was priced off the trace MEAN, so
on heterogeneous-unit data a microscale column paid a jitter sized by the macroscale columns --
measured: the smallest variance of a 300x3 dataset with column scales (1e4, 1, 1e-2) inflated
2.4e5x (~5.4 orders of magnitude), with the (truthful) ``covariance-ridged`` disclosure firing on
every fit. The ridge is now per coordinate (``eps_i = ridge * cov_ii``): a uniform ~1e-6 relative
perturbation whatever the units, silent on full-rank fits, while its actual purpose -- keeping a
rank-deficient scatter factorable -- and its disclosure machinery both survive.

T4-9: likely construction mistakes on the mixture estimators surfaced as bare
IndexError/TypeError/AttributeError/ZeroDivisionError from library internals, some only deep inside
the first EM iteration. Each now fails at the constructor with a message naming the argument, what
arrived, and what is accepted -- and every previously-working legitimate form still constructs.
The packed Gaussian mixture's component contract is enforced too: univariate GaussianEstimator
components are refused at construction with the MixtureEstimator remedy spelled out, other
non-Gaussian families fail the (mu, covar) repack with the contract named instead of a bare
AttributeError, and DiagonalGaussianEstimator components -- refused mid-EM by an overreaching
audit-era isinstance guard -- now fit end to end.
"""

import unittest

import numpy as np

from mixle.inference import optimize
from mixle.stats import (
    DiagonalGaussianEstimator,
    ExponentialEstimator,
    GaussianEstimator,
    GaussianMixtureEstimator,
    MixtureEstimator,
    MultivariateGaussianEstimator,
)


def _heterogeneous_rows(n=300, seed=42):
    """Column scales 1e4, 1, 1e-2: variances span ~12 orders of magnitude -- the T1-05 regime."""
    rng = np.random.default_rng(seed)
    x = np.column_stack(
        [
            1.0e4 * rng.standard_normal(n),
            rng.standard_normal(n),
            1.0e-2 * rng.standard_normal(n),
        ]
    )
    return [tuple(v) for v in x], x


class PerDimensionRidgeTest(unittest.TestCase):
    """T1-05: the default ridge must perturb each variance ~1e-6 RELATIVE to its own scale."""

    def test_heterogeneous_units_no_longer_inflate_the_smallest_variance(self):
        rows, x = _heterogeneous_rows()
        empirical = np.diag(np.cov(x.T, bias=True))
        fitted = optimize(rows, MultivariateGaussianEstimator(dim=3), max_its=5)
        fitted_diag = np.diag(np.asarray(fitted.covar))
        # before the fix: fitted_diag[2] / empirical[2] was ~2.4e5; now every coordinate carries
        # the same ~1e-6 relative jitter, so the ratio is 1 + ridge on all of them
        np.testing.assert_allclose(fitted_diag, empirical * (1.0 + 1.0e-6), rtol=1e-4)
        self.assertLess(float(fitted_diag[2] / empirical[2]), 1.0 + 1e-4)
        # and there is no longer a material repair to disclose
        self.assertEqual(fitted.numerical_repairs(), ())

    def test_fit_is_equivariant_under_a_per_coordinate_change_of_units(self):
        # the property the trace-mean ridge violated: rescaling ONE column must rescale exactly
        # that row/column of the fitted covariance, leaving the others bit-comparable
        rng = np.random.default_rng(7)
        x = rng.standard_normal((250, 3)) @ np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]])
        scale = np.array([1.0, 1.0e-6, 1.0])
        base = optimize([tuple(v) for v in x], MultivariateGaussianEstimator(dim=3), max_its=5)
        scaled = optimize([tuple(v * scale) for v in x], MultivariateGaussianEstimator(dim=3), max_its=5)
        expected = np.asarray(base.covar) * np.outer(scale, scale)
        np.testing.assert_allclose(np.asarray(scaled.covar), expected, rtol=1e-9)

    def test_rank_deficient_scatter_is_still_rescued_and_disclosed(self):
        # the ridge's reason to exist must survive the re-pricing: fewer points than dimensions
        # yields a singular scatter; the per-coordinate ridge must still make it factorable and
        # the repair must still be recorded
        rng = np.random.default_rng(11)
        rows = [tuple(v) for v in rng.standard_normal((2, 3))]
        fitted = optimize(rows, MultivariateGaussianEstimator(dim=3), max_its=1)
        self.assertGreater(float(np.linalg.eigvalsh(np.asarray(fitted.covar)).min()), 0.0)
        self.assertTrue(
            any("covariance-ridged" in r for r in fitted.numerical_repairs()),
            fitted.numerical_repairs(),
        )

    def test_explicit_min_covar_is_a_per_coordinate_absolute_floor_and_is_disclosed(self):
        rows, x = _heterogeneous_rows()
        fitted = optimize(rows, MultivariateGaussianEstimator(dim=3, min_covar=1.0), max_its=5)
        self.assertGreaterEqual(float(np.diag(np.asarray(fitted.covar)).min()), 1.0)
        self.assertTrue(
            any("covariance-ridged" in r for r in fitted.numerical_repairs()),
            fitted.numerical_repairs(),
        )

    def test_a_variance_free_coordinate_gets_data_scaled_jitter_and_is_disclosed(self):
        # a constant column has no scale of its own; its manufactured variance must be priced off
        # the live coordinates (ridge * their mean variance), not an arbitrary absolute constant
        rng = np.random.default_rng(3)
        live = rng.standard_normal(120)
        rows = [(float(v), 5.0) for v in live]
        fitted = optimize(rows, MultivariateGaussianEstimator(dim=2), max_its=1)
        diag = np.diag(np.asarray(fitted.covar))
        live_var = float(np.var(live))
        self.assertGreater(float(diag[1]), 0.0)
        # ~ridge * mean(live variances); generous bounds pin the scaling, not the digits
        self.assertLess(float(diag[1]), 1e-5 * live_var)
        self.assertGreater(float(diag[1]), 1e-8 * live_var)
        self.assertTrue(
            any("onto a non-positive variance" in r for r in fitted.numerical_repairs()),
            fitted.numerical_repairs(),
        )

    def test_ridge_zero_still_recovers_the_exact_mle_on_heterogeneous_units(self):
        rows, x = _heterogeneous_rows()
        fitted = optimize(rows, MultivariateGaussianEstimator(dim=3, ridge=0.0), max_its=5)
        empirical = np.cov(x.T, bias=True)
        np.testing.assert_allclose(np.asarray(fitted.covar), empirical, rtol=1e-9, atol=2e-8)


class MixtureConstructorBoundaryTest(unittest.TestCase):
    """T4-9: constructor mistakes must fail at the boundary, naming argument, got, and accepted."""

    def test_a_single_estimator_is_refused_with_a_wrap_hint(self):
        with self.assertRaises(TypeError) as ctx:
            MixtureEstimator(GaussianEstimator())
        message = str(ctx.exception)
        self.assertIn("estimators", message)
        self.assertIn("GaussianEstimator", message)
        self.assertIn("wrap it in a list", message)

    def test_a_non_sequence_is_refused_naming_the_type(self):
        with self.assertRaises(TypeError) as ctx:
            MixtureEstimator(3)
        self.assertIn("estimators", str(ctx.exception))
        self.assertIn("int", str(ctx.exception))

    def test_an_empty_mixture_is_refused_not_deferred_to_the_e_step(self):
        # used to construct silently and raise a bare IndexError inside the first EM iteration
        with self.assertRaises(ValueError) as ctx:
            MixtureEstimator([])
        self.assertIn("at least one component", str(ctx.exception))

    def test_empty_plus_robust_no_longer_divides_by_zero(self):
        # used to raise a bare ZeroDivisionError from the robust w_min default
        with self.assertRaises(ValueError):
            MixtureEstimator([], robust=True)

    def test_a_non_estimator_element_is_refused_naming_its_index(self):
        with self.assertRaises(TypeError) as ctx:
            MixtureEstimator([GaussianEstimator(), None])
        message = str(ctx.exception)
        self.assertIn("estimators[1]", message)
        self.assertIn("NoneType", message)

    def test_fixed_weights_length_mismatch_is_refused_at_construction(self):
        with self.assertRaises(ValueError) as ctx:
            MixtureEstimator([GaussianEstimator(), GaussianEstimator()], fixed_weights=[0.5, 0.3, 0.2])
        message = str(ctx.exception)
        self.assertIn("fixed_weights", message)
        self.assertIn("exactly 2", message)

    def test_non_numeric_fixed_weights_are_refused_naming_the_argument(self):
        with self.assertRaises(TypeError) as ctx:
            MixtureEstimator([GaussianEstimator(), GaussianEstimator()], fixed_weights="half")
        self.assertIn("fixed_weights", str(ctx.exception))

    def test_negative_fixed_weights_are_refused(self):
        with self.assertRaises(ValueError) as ctx:
            MixtureEstimator([GaussianEstimator(), GaussianEstimator()], fixed_weights=[0.7, -0.3])
        self.assertIn("non-negative", str(ctx.exception))

    def test_negative_pseudo_count_is_refused(self):
        # used to be absorbed: it subtracted mass from every component count and could drive
        # fitted weights negative with no signal anywhere
        with self.assertRaises(ValueError) as ctx:
            MixtureEstimator([GaussianEstimator(), GaussianEstimator()], pseudo_count=-1.0)
        message = str(ctx.exception)
        self.assertIn("pseudo_count", message)
        self.assertIn("non-negative", message)

    def test_non_scalar_pseudo_count_is_refused_at_construction(self):
        # used to construct and raise a bare operand TypeError mid-M-step
        with self.assertRaises(TypeError) as ctx:
            MixtureEstimator([GaussianEstimator(), GaussianEstimator()], pseudo_count=[1.0, 2.0])
        self.assertIn("pseudo_count", str(ctx.exception))

    def test_non_scalar_w_min_is_refused_naming_the_argument(self):
        # used to raise a bare "'<=' not supported between instances of 'str' and 'float'";
        # note a NUMERIC string ("0") is computed with rather than refused, like finite_scalar
        with self.assertRaises(TypeError) as ctx:
            MixtureEstimator([GaussianEstimator(), GaussianEstimator()], w_min="tiny")
        self.assertIn("w_min", str(ctx.exception))
        self.assertEqual(MixtureEstimator([GaussianEstimator(), GaussianEstimator()], w_min="0").w_min, 0.0)

    def test_a_single_string_key_is_refused_with_the_pair_spelled_out(self):
        # keys[1] on a 1-char string used to raise a bare IndexError from the accumulator, and a
        # 2-char string was silently split into two 1-character keys
        for value in ("k", "wc"):
            with self.subTest(keys=value):
                with self.assertRaises(TypeError) as ctx:
                    MixtureEstimator([GaussianEstimator(), GaussianEstimator()], keys=value)
                message = str(ctx.exception)
                self.assertIn("keys", message)
                self.assertIn("(weights_key, components_key)", message)
                self.assertIn(repr(value), message)

    def test_wrong_length_suff_stat_is_refused_at_construction(self):
        # used to construct and raise "can't multiply sequence by non-int of type 'float'" in the
        # first M-step
        with self.assertRaises(ValueError) as ctx:
            MixtureEstimator([GaussianEstimator(), GaussianEstimator()], pseudo_count=1.0, suff_stat=[1.0, 2.0, 3.0])
        self.assertIn("suff_stat", str(ctx.exception))

    def test_gaussian_mixture_estimator_reports_its_own_class_name(self):
        with self.assertRaises(TypeError) as ctx:
            GaussianMixtureEstimator(MultivariateGaussianEstimator(dim=2))
        self.assertIn("GaussianMixtureEstimator", str(ctx.exception))


class GaussianMixtureComponentBoundaryTest(unittest.TestCase):
    """T4-9: wrong component families for the packed Gaussian mixture must fail with guidance.

    ``GaussianMixtureEstimator([GaussianEstimator(), GaussianEstimator()])`` -- the tester's exact
    call, and the natural first attempt at a univariate Gaussian mixture -- used to run a whole
    E-step and then die at the end of the first M-step with a bare ``AttributeError:
    'GaussianDistribution' object has no attribute 'covar'`` from the (mu, covar) repack.
    """

    def test_univariate_gaussian_components_are_refused_at_construction(self):
        with self.assertRaises(TypeError) as ctx:
            GaussianMixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        message = str(ctx.exception)
        self.assertIn("estimators[0]", message)
        self.assertIn("univariate GaussianEstimator", message)
        # the remedy is spelled out: the generic mixture takes scalar components
        self.assertIn("MixtureEstimator([GaussianEstimator(), ...])", message)

    def test_other_non_gaussian_components_fail_the_repack_with_the_contract_named(self):
        # Exponential components sail through EM and used to die in the repack as a bare
        # ``AttributeError: 'ExponentialDistribution' object has no attribute 'mu'``. Unknown
        # estimator types are deliberately NOT blocked at construction (duck-typed vector
        # Gaussians must keep working), so the refusal lands where the incompatibility is proven.
        rng = np.random.default_rng(5)
        data = list(rng.gamma(2.0, 1.0, 80))
        with self.assertRaises(TypeError) as ctx:
            optimize(
                data,
                GaussianMixtureEstimator([ExponentialEstimator(), ExponentialEstimator()]),
                max_its=2,
            )
        message = str(ctx.exception)
        self.assertIn("ExponentialDistribution", message)
        self.assertIn("`mu` and `covar`", message)

    def test_diagonal_gaussian_components_fit_end_to_end(self):
        # The repack hands MultivariateGaussianDistribution components back to the diagonal
        # accumulators on every EM iteration after the first; an audit-era isinstance guard in
        # DiagonalGaussianAccumulator refused that legitimate state, so diagonal components inside
        # a Gaussian mixture died on iteration two with "diagonal Gaussian accumulator estimate
        # must have the configured dimension". The guard is dimension-only now.
        rng = np.random.default_rng(8)
        rows = [
            tuple(v)
            for v in np.vstack(
                [
                    rng.multivariate_normal([-3.0, -3.0], 0.5 * np.eye(2), 80),
                    rng.multivariate_normal([3.0, 3.0], 0.5 * np.eye(2), 80),
                ]
            )
        ]
        fitted = optimize(
            rows,
            GaussianMixtureEstimator([DiagonalGaussianEstimator(dim=2), DiagonalGaussianEstimator(dim=2)]),
            max_its=10,
        )
        means = np.asarray(fitted.mu)[np.argsort(np.asarray(fitted.mu)[:, 0])]
        np.testing.assert_allclose(means, [[-3.0, -3.0], [3.0, 3.0]], atol=0.5)
        np.testing.assert_allclose(np.asarray(fitted.w), [0.5, 0.5], atol=0.1)

    def test_a_genuinely_mismatched_estimate_dimension_is_still_refused(self):
        # the contract the relaxed guard keeps: a wrong-dimension estimate is a real wiring bug
        from mixle.stats.multivariate.diagonal_gaussian import (
            DiagonalGaussianAccumulator,
            DiagonalGaussianDistribution,
        )

        acc = DiagonalGaussianAccumulator(dim=2)
        wrong = DiagonalGaussianDistribution([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        with self.assertRaises(ValueError) as ctx:
            acc.update([1.0, 2.0], 1.0, wrong)
        self.assertIn("configured dimension", str(ctx.exception))
        with self.assertRaises(ValueError):
            acc.seq_update(np.asarray([[1.0, 2.0]]), np.asarray([1.0]), wrong)


class MixtureConstructorNoOverreachTest(unittest.TestCase):
    """The new boundary must refuse nothing that legitimately worked before."""

    def test_every_previously_working_form_still_constructs(self):
        g = GaussianEstimator()
        MixtureEstimator([g, g])
        MixtureEstimator((g, g))  # tuples of estimators
        MixtureEstimator([g, g], fixed_weights=np.array([0.25, 0.75]))
        MixtureEstimator([g, g], fixed_weights=[1, 1])  # ints, unnormalized: used raw downstream
        MixtureEstimator([g, g], pseudo_count=0.0)
        MixtureEstimator([g, g], pseudo_count=2)  # int pseudo-count
        MixtureEstimator([g, g], keys=("w", "c"))
        MixtureEstimator([g, g], keys=["w", None])  # list pair
        MixtureEstimator([g, g], keys=(None, None))
        MixtureEstimator([g, g], w_min=0)
        MixtureEstimator([g, g], w_min=-1.0)  # negative means "no floor", still accepted

    def test_keys_none_means_no_sharing(self):
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()], keys=None)
        self.assertEqual(tuple(est.keys), (None, None))
        acc = est.accumulator_factory().make()
        self.assertIsNone(acc.weight_key)
        self.assertIsNone(acc.comp_key)

    def test_scalar_suff_stat_still_broadcasts_to_a_uniform_prior_count(self):
        rng = np.random.default_rng(0)
        data = list(rng.normal(0.0, 1.0, 60))
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()], pseudo_count=1.0, suff_stat=0.5)
        fitted = optimize(data, est, max_its=3)
        w = np.asarray(fitted.w)
        self.assertEqual(w.shape, (2,))
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)

    def test_matching_fixed_weights_are_used_exactly_as_given(self):
        rng = np.random.default_rng(1)
        data = list(rng.normal(0.0, 1.0, 60))
        fitted = optimize(
            data,
            MixtureEstimator([GaussianEstimator(), GaussianEstimator()], fixed_weights=[0.3, 0.7]),
            max_its=3,
        )
        np.testing.assert_allclose(np.asarray(fitted.w), [0.3, 0.7])


if __name__ == "__main__":
    unittest.main()
