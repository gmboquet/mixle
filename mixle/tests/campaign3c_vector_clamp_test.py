"""Campaign-3c: the vector Gaussians' anchored-scatter clamp must not read GENUINE data as constant.

The reference fix landed in ``mixle/stats/univariate/continuous/gaussian.py``
(``_anchored_pooled_variance``): the anchored scatter is SPLIT into (1) ``core``, the scatter about
the SAMPLE's own mean, which carries all of the real spread at spread scale and keeps only the
pre-existing scale-RELATIVE ``1e-12`` cancellation clamp, and (2) the displacement of the reported
mean from the sample mean -- genuine only under a pseudo-count prior, pure rounding of
``sum_x / count`` on the plain ML path -- which is the ONLY place the data's large magnitude enters,
and is where the ulp-scale clamp (``count * (4*eps*|mean|)^2``) now applies, ALONE.

Before this file's repair, ``diagonal_gaussian.py``'s ``_anchored_pooled_variances``
(per-coordinate, vectorized with ``np.where``) and ``multivariate_gaussian.py``'s
``_anchored_pooled_covariance`` (matrix-valued) each combined core + shift into ONE array and
applied BOTH noise sources to that single combined test -- so the ulp-scale term, an ABSOLUTE floor
set by ``count * (4*eps*|mean|)^2``, had to be crossed by the genuine ``count*spread^2`` scatter too.
At mean ~1e15 a fully representable sd of 0.5 (four ``ulp(1e15)`` steps) sits below that absolute
floor and read as EXACTLY CONSTANT.

Measured on the pre-repair tree (n=1000, mean=1e15, sd=0.5, seed 0):

    diagonal        -> fitted covar collapsed onto the 1e-8 floor (true variance ~0.239)
    full covariance -> fitted covar collapsed onto 1e-8 * I, wiping every off-diagonal entry too
                        (true ~[[0.257, 0.058], [0.058, 0.162]])

Both files now clamp the two noise sources separately. ``core`` keeps the pre-existing scale-relative
test alone -- for the full covariance this stays a WHOLE-MATRIX test (zeroing individual entries of a
matrix that carries real scale could destroy its symmetry/PSD-ness; an all-sub-noise matrix is the
zero matrix, which is PSD, so the whole-matrix test is still the right shape for this piece). The
mean-displacement term is zeroed PER-COORDINATE against the ulp-scale test alone, which is safe even
for the full-covariance matrix specifically because that piece only ever enters as
``outer(gap, gap)``: the outer product of any real vector, partially zeroed or not, is symmetric and
PSD on its own, so a per-coordinate zero there cannot produce the asymmetric/indefinite residue a
per-entry zero of ``core`` could.

Tolerances are stated against an exact-Fraction reference computed here on the same float64 array
that is fitted (mirrors ``campaign3_mvn_test.py``), so nothing incidental to a seed or platform is
pinned.
"""

import unittest
from fractions import Fraction

import numpy as np

from mixle.stats.multivariate.diagonal_gaussian import (
    DiagonalGaussianEstimator,
)
from mixle.stats.multivariate.diagonal_gaussian import (
    _anchored_mean_offset as _diag_anchored_mean_offset,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianEstimator,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    _anchored_mean_offset as _mvn_anchored_mean_offset,
)

_ANCHOR_MEAN_ULP = 8.8817841970012523e-16  # 4 * eps, same constant the production module uses.
MIN_COVAR = 1.0e-8  # DiagonalGaussianEstimator/MultivariateGaussianEstimator default min_covar.


def exact_variance(x: np.ndarray) -> np.ndarray:
    """Population variance of each column of ``x``, in exact rational arithmetic."""
    n, d = x.shape
    out = np.zeros(d)
    for j in range(d):
        col = [Fraction(float(v)) for v in x[:, j]]
        m = sum(col) / n
        out[j] = float(sum((c - m) ** 2 for c in col) / n)
    return out


def exact_covariance(x: np.ndarray) -> np.ndarray:
    """Population covariance of the rows of ``x``, in exact rational arithmetic."""
    n, d = x.shape
    columns = [[Fraction(float(v)) for v in x[:, j]] for j in range(d)]
    means = [sum(c) / n for c in columns]
    out = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            total = sum((a - means[i]) * (b - means[j]) for a, b in zip(columns[i], columns[j]))
            out[i, j] = float(total / n)
    return out


def _pre_fix_diagonal_scatter(anchored: tuple, count: float, mean_offset: np.ndarray) -> np.ndarray:
    """The COMBINED clamp exactly as ``diagonal_gaussian._anchored_pooled_variances`` read before
    this file's repair.

    Kept here, clearly labelled, purely so this regression is self-verifying: it pins that the very
    formula this module used to ship reads the data in
    ``test_genuine_spread_at_extreme_magnitude_is_recovered_not_read_as_constant`` as exactly zero,
    while the current production ``_anchored_pooled_variances`` (imported above) does not.
    """
    anchor, a_sum, a_sum2 = anchored
    centroid_offset = a_sum / count
    gap = mean_offset - centroid_offset
    observed_scatter = (a_sum2 - centroid_offset * a_sum) + count * gap * gap
    noise_scale = np.maximum.reduce(
        (np.abs(a_sum2), np.abs(centroid_offset * a_sum), count * gap * gap, np.full_like(a_sum2, 1.0e-300))
    )
    mean_ulp = _ANCHOR_MEAN_ULP * np.maximum(np.abs(anchor + mean_offset), np.abs(anchor))
    return np.where(
        observed_scatter < np.maximum(1.0e-12 * noise_scale, count * mean_ulp * mean_ulp),
        0.0,
        observed_scatter,
    )


def _pre_fix_full_scatter(anchored: tuple, count: float, mean_offset: np.ndarray) -> np.ndarray:
    """The COMBINED whole-matrix clamp exactly as ``multivariate_gaussian._anchored_pooled_covariance``
    read before this file's repair. See :func:`_pre_fix_diagonal_scatter`.
    """
    anchor, a_sum, a_sum2 = anchored
    centroid_offset = a_sum / count
    gap = mean_offset - centroid_offset
    cross = np.outer(centroid_offset, a_sum)
    observed_scatter = (a_sum2 - cross) + count * np.outer(gap, gap)
    observed_scatter = 0.5 * (observed_scatter + observed_scatter.T)
    noise_scale = max(
        float(np.max(np.abs(a_sum2), initial=0.0)),
        float(np.max(np.abs(cross), initial=0.0)),
        float(count * np.max(gap * gap, initial=0.0)),
        1.0e-300,
    )
    mean_ulp = _ANCHOR_MEAN_ULP * max(
        float(np.max(np.abs(anchor + mean_offset), initial=0.0)), float(np.max(np.abs(anchor), initial=0.0))
    )
    threshold = max(1.0e-12 * noise_scale, count * mean_ulp * mean_ulp)
    if float(np.max(np.abs(observed_scatter), initial=0.0)) < threshold:
        return np.zeros_like(observed_scatter)
    return observed_scatter


class DiagonalExtremeMagnitudeClampTest(unittest.TestCase):
    def test_genuine_spread_at_extreme_magnitude_is_recovered_not_read_as_constant(self):
        rng = np.random.default_rng(0)
        n = 1000
        mean_val = 1.0e15
        x = mean_val + rng.normal(0.0, 0.5, size=(n, 1))
        exact = exact_variance(x)

        est = DiagonalGaussianEstimator(dim=1, ridge=0.0)
        acc = est.accumulator_factory().make()
        acc.seq_update(x, np.ones(n), None)
        fitted = np.asarray(est.estimate(float(n), acc.value()).covar)

        self.assertGreater(fitted[0], 0.01, "genuine spread ~0.24 must not collapse to the min_covar floor")
        np.testing.assert_allclose(fitted, exact, rtol=1.0e-9, atol=0.0)

        # Pin the regression this test guards: the pre-repair formula, fed the SAME anchored
        # moments, reads this genuine spread as exactly zero.
        anchored = (acc._anchor, acc._anchored_sum, acc._anchored_sum2)
        mean_offset = _diag_anchored_mean_offset(anchored, acc.count, None, None)
        pre_fix = _pre_fix_diagonal_scatter(anchored, acc.count, mean_offset)
        np.testing.assert_array_equal(pre_fix, np.zeros_like(pre_fix))

    def test_degenerate_data_still_clamps_to_exactly_zero_at_extreme_magnitude(self):
        n = 400
        row = np.asarray([1.0e15])
        est = DiagonalGaussianEstimator(dim=1, ridge=0.0)
        acc = est.accumulator_factory().make()
        acc.seq_update(np.repeat(row[None, :], n, axis=0), np.ones(n), None)
        fitted = np.asarray(est.estimate(float(n), acc.value()).covar)
        np.testing.assert_array_equal(fitted, np.asarray([MIN_COVAR]))

    def test_accumulator_combine_agrees_exactly_with_a_single_seq_update_on_degenerate_data(self):
        # The accumulator/reweighted-seq_update parity invariant, at the extreme magnitude this file
        # repairs: two algebraically equivalent accumulation paths for a degenerate component must
        # produce the IDENTICAL exactly-zero scatter, or the scale-relative floor downstream reads
        # the +-O(eps) residue as real and the two paths disagree.
        row = np.asarray([1.0e15, -1.0e15])
        est = DiagonalGaussianEstimator(dim=2, ridge=0.0)

        whole = est.accumulator_factory().make()
        whole.seq_update(np.repeat(row[None, :], 800, axis=0), np.ones(800), None)
        a = est.accumulator_factory().make()
        a.seq_update(np.repeat(row[None, :], 300, axis=0), np.ones(300), None)
        b = est.accumulator_factory().make()
        b.seq_update(np.repeat(row[None, :], 500, axis=0), np.ones(500), None)
        combined = a.combine(b.value())

        fitted_whole = np.asarray(est.estimate(800.0, whole.value()).covar)
        fitted_combined = np.asarray(est.estimate(800.0, combined.value()).covar)
        np.testing.assert_array_equal(fitted_whole, fitted_combined)
        np.testing.assert_array_equal(fitted_whole, np.full(2, MIN_COVAR))

    def test_shift_equivariance_at_1_7e9_is_unchanged(self):
        rows = np.random.default_rng(7).normal(0.0, 1.0, size=(500, 2))
        x = rows + 1.7e9
        exact = exact_variance(x)

        est = DiagonalGaussianEstimator(dim=2, ridge=0.0)
        acc = est.accumulator_factory().make()
        acc.seq_update(x, np.ones(len(x)), None)
        fitted = np.asarray(est.estimate(float(len(x)), acc.value()).covar)
        np.testing.assert_allclose(fitted, exact, rtol=1.0e-9, atol=0.0)

    def test_a_genuine_pseudo_count_displacement_at_extreme_magnitude_still_counts(self):
        # With a mean pseudo-count the shift term is genuine, not rounding -- it must not be zeroed
        # just because it rides in the same computation as the (correctly-clamped) core scatter.
        n = 1000
        mean_val = 1.0e15
        x = mean_val + np.random.default_rng(3).normal(0.0, 0.5, size=(n, 1))
        prior_mu = np.asarray([mean_val + 50.0])  # far above the ulp(1e15) ~ 0.125 grid step
        est = DiagonalGaussianEstimator(dim=1, pseudo_count=(4.0, None), suff_stat=(prior_mu, None), ridge=0.0)
        acc = est.accumulator_factory().make()
        acc.seq_update(x, np.ones(n), None)
        fitted = np.asarray(est.estimate(float(n), acc.value()).covar)
        # The pulled mean sits ~50 away from the unpulled sample mean; squared and count-weighted,
        # that displacement alone dwarfs the ~0.24 core scatter -- nowhere near the min_covar floor.
        self.assertGreater(fitted[0], 0.1)


class FullCovarianceExtremeMagnitudeClampTest(unittest.TestCase):
    def test_genuine_spread_and_correlation_at_extreme_magnitude_are_recovered(self):
        rng = np.random.default_rng(1)
        n = 1000
        mean_val = 1.0e15
        cov_true = np.asarray([[0.25, 0.05], [0.05, 0.16]])
        x = mean_val + rng.multivariate_normal(np.zeros(2), cov_true, size=n)
        exact = exact_covariance(x)

        est = MultivariateGaussianEstimator(dim=2, ridge=0.0)
        acc = est.accumulator_factory().make()
        acc.seq_update(x, np.ones(n), None)
        fitted_raw = np.asarray(est.estimate(float(n), acc.value()).covar)
        # ridge=0.0 is the exact MLE up to the documented absolute min_covar jitter on the diagonal
        # (matches the ``ridge=0.0`` convention in campaign3_mvn_test.py's ShiftEquivarianceTest).
        fitted = fitted_raw - MIN_COVAR * np.eye(2)

        self.assertGreater(fitted[0, 0], 0.01, "the WHOLE matrix must not collapse to the min_covar floor")
        self.assertNotEqual(fitted[0, 1], 0.0, "off-diagonal correlation must survive too")
        np.testing.assert_allclose(fitted, exact, rtol=1.0e-8, atol=0.0)
        # Symmetric and PSD, same as any ordinary fit -- the split must not have broken either.
        np.testing.assert_array_equal(fitted, fitted.T)
        np.testing.assert_array_less(-1e-9, np.linalg.eigvalsh(fitted))

        anchored = (acc._anchor, acc._anchored_sum, acc._anchored_sum2)
        mean_offset = _mvn_anchored_mean_offset(anchored, acc.count, None, None)
        pre_fix = _pre_fix_full_scatter(anchored, acc.count, mean_offset)
        np.testing.assert_array_equal(pre_fix, np.zeros_like(pre_fix))

    def test_degenerate_data_still_clamps_the_whole_matrix_to_exactly_zero_at_extreme_magnitude(self):
        n = 400
        row = np.asarray([1.0e15, -3.4e14])
        est = MultivariateGaussianEstimator(dim=2, ridge=0.0)
        acc = est.accumulator_factory().make()
        acc.seq_update(np.repeat(row[None, :], n, axis=0), np.ones(n), None)
        fitted = np.asarray(est.estimate(float(n), acc.value()).covar)
        np.testing.assert_array_equal(fitted, MIN_COVAR * np.eye(2))

    def test_accumulator_combine_agrees_exactly_with_a_single_seq_update_on_degenerate_data(self):
        row = np.asarray([1.0e15, -3.4e14, 7.0e14])
        est = MultivariateGaussianEstimator(dim=3, ridge=0.0)

        whole = est.accumulator_factory().make()
        whole.seq_update(np.repeat(row[None, :], 800, axis=0), np.ones(800), None)
        a = est.accumulator_factory().make()
        a.seq_update(np.repeat(row[None, :], 300, axis=0), np.ones(300), None)
        b = est.accumulator_factory().make()
        b.seq_update(np.repeat(row[None, :], 500, axis=0), np.ones(500), None)
        combined = a.combine(b.value())

        fitted_whole = np.asarray(est.estimate(800.0, whole.value()).covar)
        fitted_combined = np.asarray(est.estimate(800.0, combined.value()).covar)
        np.testing.assert_array_equal(fitted_whole, fitted_combined)
        np.testing.assert_array_equal(fitted_whole, MIN_COVAR * np.eye(3))

    def test_shift_equivariance_at_1_7e9_is_unchanged(self):
        rng = np.random.default_rng(9)
        rows = rng.multivariate_normal(np.zeros(2), [[1.0, 0.4], [0.4, 1.0]], size=500)
        x = rows + 1.7e9
        exact = exact_covariance(x)

        est = MultivariateGaussianEstimator(dim=2, ridge=0.0)
        acc = est.accumulator_factory().make()
        acc.seq_update(x, np.ones(len(x)), None)
        fitted = np.asarray(est.estimate(float(len(x)), acc.value()).covar) - MIN_COVAR * np.eye(2)
        np.testing.assert_allclose(fitted, exact, rtol=1.0e-8, atol=0.0)

    def test_a_genuine_pseudo_count_displacement_at_extreme_magnitude_still_counts(self):
        n = 1000
        mean_val = 1.0e15
        cov_true = np.asarray([[0.25, 0.0], [0.0, 0.16]])
        x = mean_val + np.random.default_rng(11).multivariate_normal(np.zeros(2), cov_true, size=n)
        prior_mu = np.asarray([mean_val + 50.0, mean_val - 30.0])
        prior_covar = np.eye(2)
        est = MultivariateGaussianEstimator(
            dim=2, pseudo_count=(4.0, None), suff_stat=(prior_mu, prior_covar), ridge=0.0
        )
        acc = est.accumulator_factory().make()
        acc.seq_update(x, np.ones(n), None)
        fitted = np.asarray(est.estimate(float(n), acc.value()).covar)
        self.assertGreater(fitted[0, 0], 0.1)
        self.assertGreater(fitted[1, 1], 0.1)
        np.testing.assert_array_equal(fitted, fitted.T)


if __name__ == "__main__":
    unittest.main()
