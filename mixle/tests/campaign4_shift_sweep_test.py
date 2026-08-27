"""Shift-equivariance across EVERY location-scale family, not one family per campaign (T1-01, T1-02).

Four release waves repaired the ``E[x^2] - E[x]^2`` cancellation one family at a time, each wave
covering only the families a tester had just caught. Twice a family that the previous round had
recorded as an accepted limit came back as a blocking finding: campaign three caught the
multivariate/diagonal/generalized-Gaussian/GEV/logistic gap, and campaign four caught Gumbel
(``ValueError`` at offsets 3e8-1.7e9, silently 15.4x wrong at 2**31) and Student-t (scale 1.6e8x too
small at the Unix-epoch offset, ``converged=True``, empty ``numerical_repairs()``, no warning).

So this file does not test Gumbel and Student-t. It sweeps the whole class -- every family in
``mixle/stats/univariate`` and ``mixle/stats/multivariate`` with a location and/or a scale parameter
-- against the same dyadic offset grid, so the next unrepaired sibling fails HERE rather than in a
release candidate. The families repaired in earlier waves are carried along as positive controls:
if one of them regresses, this file says so in the same sweep.

An audit of the same grid at the release candidate's tree found six families failing:
MultivariateStudentT (1.98e4 relative, plus a ``ValueError`` at 1e10), Student-t (5.8e3), Gumbel
(7.8), GeneralizedPareto (shape ``+0.16`` flattened to ``0``, the exponential limit),
SkewNormal (7.0e-4) and ExGaussian (1.3e-4). All six are repaired; the sweep below is what holds
them repaired.

Two further things this file holds closed, both found by widening the same sweep rather than by a
new tester report:

* The RAW-ONLY DISCLOSURE was fail-open in exactly its worst case. Statistics that arrive already
  reduced -- an engine kernel's stacked moments, a legacy artifact, a hand-built tuple -- cannot be
  repaired, so the contract is that ``estimate`` names them. The gate did that only while the
  computed ``E[x^2] - E[x]^2`` was still positive; once cancellation had taken the WHOLE spread it
  fell through silently, which is the case that returns a scale collapsed onto the family floor.
  Measured on the release candidate at sd ~2 and offset 1.7e9: Gumbel and Student-t 1e-8, logistic
  1e-8, GEV 1e-12, generalized-Gaussian alpha 1e-6, Gaussian sigma2 2.89e10 -- every one silent,
  and four of the six never called the disclosure at all. The hole existed in three transcribed
  copies of the gate (scalar contract, vector contract, multivariate Gaussian).
* Two families that difference the moments of ``x**2`` rather than of ``x`` -- Rician and Nakagami
  -- were never in the class's scope and had the identical defect one transform along. Rician lost
  44% of its scale at ``nu/sigma = 1e8`` and, from ``nu/sigma`` ~1e5 up, could not be fitted at all:
  ``scipy.special.ive`` returns NaN past ~1e10 and ``optimize`` reported only "fused EM did not
  produce a finite objective from its non-finite initial model", the same opaque internal error
  Gumbel used to raise.

Two measured limits are pinned here as limits rather than as bugs, because the evidence says they
are representational:

* The conditioning gate deliberately keeps the historical raw single-pass path bit-identical while
  ``abs(mean)/spread`` stays under ~2000, which costs up to ~4e-9 relative. That is family-wide --
  the repaired Gaussian control measures 3.7e-9 in that band, MORE than any family repaired here --
  so it is pinned as a shared design point, not attributed to any one family.
* A full multivariate Student-t EM run is shift-equivariant only to the granularity of its own
  location parameter, because the latent reweighting is a function of ``x - mu`` and ``mu`` is a
  float64 at the data's magnitude. Snapping the un-shifted fit's ``mu`` onto exactly the grid the
  offset forces reproduces the shifted fit to ~3e-14, which is what identifies the mechanism; the
  test below performs that snap so the claim is measured rather than asserted.
"""

import math
import pickle
import unittest
import warnings
from fractions import Fraction

import numpy as np

from mixle.inference import optimize
from mixle.stats import (
    DiagonalGaussianEstimator,
    ExponentiallyModifiedGaussianEstimator,
    GaussianEstimator,
    GeneralizedExtremeValueEstimator,
    GeneralizedGaussianEstimator,
    GeneralizedParetoDistribution,
    GeneralizedParetoEstimator,
    GumbelEstimator,
    InverseGaussianEstimator,
    LaplaceEstimator,
    LogGaussianEstimator,
    LogisticEstimator,
    MultivariateGaussianEstimator,
    MultivariateStudentTEstimator,
    NakagamiEstimator,
    RicianDistribution,
    RicianEstimator,
    SkewNormalEstimator,
    StudentTEstimator,
    UniformEstimator,
)
from mixle.stats.multivariate.multivariate_student_t import MultivariateStudentTDistribution

# Data lives on a dyadic grid so that x + c - c == x EXACTLY in float64 at every offset below:
# the fit on the shifted data is then a fit on the same sample, and any parameter movement is the
# estimator's own loss rather than the data's.
GRID = 2.0**-12

# 3e8 and 1.7e9 are the band where Gumbel raised; 2**31 is where it returned 15.4x silently; 1e12
# is past the point where a float64 can resolve a unit spread at all.
OFFSETS = (0.0, 1.0e6, 1.0e8, 3.0e8, 1.7e9, float(2**31), 1.0e10, 1.0e12)

# The repaired M-steps are exact to a few ulps; 1e-9 is the bar the release states for a scale or
# shape parameter, and every family below clears it by six orders of magnitude or more.
SCALE_TOLERANCE = 1.0e-9

# A location must track the offset, but ``c`` itself only exists on the float64 grid at its own
# magnitude, so the reachable bound is a small multiple of that grid step -- not a relative one.
LOCATION_ULPS = 8.0


def _dyadic(kind, n=600, seed=7, sd=2.0):
    """A sample on the dyadic grid, so every offset in ``OFFSETS`` is exactly representable."""
    rs = np.random.RandomState(seed)
    if kind == "normal":
        raw = rs.normal(0.0, sd, n)
    elif kind == "gumbel":
        raw = rs.gumbel(0.0, sd, n)
    elif kind == "t":
        raw = rs.standard_t(5.0, n) * sd
    elif kind == "skew":
        z0, z1 = rs.randn(n), rs.randn(n)
        raw = sd * (0.8 * np.abs(z0) + math.sqrt(1.0 - 0.64) * z1)
    elif kind == "emg":
        raw = rs.normal(0.0, sd, n) + rs.exponential(1.5, n)
    elif kind == "gpd":
        raw = (2.0 / 0.2) * (np.power(rs.uniform(size=n), -0.2) - 1.0)
    else:  # pragma: no cover - guards a typo in a new case
        raise ValueError(kind)
    return np.round(raw / GRID) * GRID


def _dyadic_matrix(n=600, p=3, seed=11, sd=2.0):
    rs = np.random.RandomState(seed)
    return np.round(rs.normal(0.0, sd, (n, p)) / GRID) * GRID


def _exact_population_variance(values):
    """Population variance in exact rationals, so the reference is not itself a float estimate."""
    numbers = [Fraction(float(v)) for v in values]
    mean = sum(numbers) / len(numbers)
    return float(sum((v - mean) ** 2 for v in numbers) / len(numbers))


def _m_steps(estimator, rows, iterations=1):
    """Run ``iterations`` accumulate-and-estimate cycles, feeding each estimate back into the next."""
    previous = None
    for _ in range(iterations):
        accumulator = estimator.accumulator_factory().make()
        encoded = accumulator.acc_to_encoder().seq_encode(rows)
        weights = np.ones(len(rows))
        if previous is None:
            accumulator.seq_initialize(encoded, weights, np.random.RandomState(0))
        else:
            accumulator.seq_update(encoded, weights, previous)
        previous = estimator.estimate(float(np.sum(weights)), accumulator.value())
    return previous


def _rows(data, multivariate):
    return [list(row) for row in data] if multivariate else list(data)


class ShiftSweepTestCase(unittest.TestCase):
    """The class-wide sweep: family x offset -> parameter movement."""

    def _sweep(
        self,
        label,
        make_estimator,
        data,
        read,
        location_keys,
        iterations=1,
        multivariate=False,
        offsets=OFFSETS,
        scale_tolerance=SCALE_TOLERANCE,
        location_ulps=LOCATION_ULPS,
    ):
        """Assert that ``read``'s scale/shape entries do not move and its location entries track ``c``."""
        base = read(_m_steps(make_estimator(0.0), _rows(data, multivariate), iterations))
        for offset in offsets:
            with self.subTest(family=label, offset=offset):
                shifted = data + offset
                self.assertTrue(
                    np.array_equal(shifted - offset, data),
                    "the offset must be exactly representable or the sweep measures the data, not the fit",
                )
                got = read(_m_steps(make_estimator(offset), _rows(shifted, multivariate), iterations))
                step = float(np.spacing(max(offset, 1.0)))
                for key, reference in base.items():
                    if key in location_keys:
                        moved = abs((got[key] - offset) - reference) / step
                        self.assertLessEqual(
                            moved,
                            location_ulps,
                            "%s: %s moved %.3g grid steps of the offset" % (label, key, moved),
                        )
                    else:
                        moved = abs(got[key] - reference) / max(abs(reference), 1e-300)
                        self.assertLessEqual(
                            moved,
                            scale_tolerance,
                            "%s: %s moved %.3e relative (%.17g -> %.17g)" % (label, key, moved, reference, got[key]),
                        )

    # ---------------------------------------------------------------- positive controls
    # Repaired in earlier waves. If one of these fails, the finding is a REGRESSION and matters
    # more than anything else this file covers.

    def test_gaussian_control_stays_shift_equivariant(self):
        self._sweep(
            "GaussianEstimator",
            lambda c: GaussianEstimator(),
            _dyadic("normal"),
            lambda d,: {"loc": d.mu, "var": d.sigma2},
            {"loc"},
        )

    def test_logistic_control_stays_shift_equivariant(self):
        self._sweep(
            "LogisticEstimator",
            lambda c: LogisticEstimator(),
            _dyadic("normal"),
            lambda d: {"loc": d.loc, "scale": d.scale},
            {"loc"},
        )

    def test_generalized_gaussian_control_stays_shift_equivariant(self):
        self._sweep(
            "GeneralizedGaussianEstimator",
            lambda c: GeneralizedGaussianEstimator(),
            _dyadic("normal"),
            lambda d: {"mu": d.mu, "alpha": d.alpha, "beta": d.beta},
            {"mu"},
        )

    def test_generalized_extreme_value_control_stays_shift_equivariant(self):
        self._sweep(
            "GeneralizedExtremeValueEstimator",
            lambda c: GeneralizedExtremeValueEstimator(),
            _dyadic("gumbel"),
            lambda d: {"loc": d.loc, "scale": d.scale, "shape": d.shape},
            {"loc"},
        )

    def test_multivariate_and_diagonal_gaussian_controls_stay_shift_equivariant(self):
        matrix = _dyadic_matrix()
        self._sweep(
            "MultivariateGaussianEstimator",
            lambda c: MultivariateGaussianEstimator(dim=3),
            matrix,
            lambda d: {"mu0": d.mu[0], "cov00": d.covar[0][0], "cov01": d.covar[0][1]},
            {"mu0"},
            multivariate=True,
        )
        self._sweep(
            "DiagonalGaussianEstimator",
            lambda c: DiagonalGaussianEstimator(dim=3),
            matrix,
            lambda d: {"mu0": d.mu[0], "var0": d.covar[0]},
            {"mu0"},
            multivariate=True,
        )

    def test_order_statistic_families_are_exactly_shift_equivariant(self):
        """Laplace (median/MAD) and Uniform (min/max) never difference raw moments at all."""
        self._sweep(
            "LaplaceEstimator",
            lambda c: LaplaceEstimator(),
            _dyadic("normal"),
            lambda d: {"mu": d.mu, "b": d.b},
            {"mu"},
            scale_tolerance=0.0,
            location_ulps=0.0,
        )
        self._sweep(
            "UniformEstimator",
            lambda c: UniformEstimator(),
            _dyadic("normal"),
            lambda d: {"low": d.low, "high": d.high},
            {"low", "high"},
            location_ulps=0.0,
        )

    # ---------------------------------------------------------------- repaired this wave

    def test_gumbel_is_shift_equivariant(self):
        """T1-02: ValueError at 3e8-1.7e9, and 15.4x too large at 2**31, both from raw moments."""
        self._sweep(
            "GumbelEstimator",
            lambda c: GumbelEstimator(),
            _dyadic("gumbel"),
            lambda d: {"loc": d.loc, "scale": d.scale},
            {"loc"},
        )

    def test_student_t_is_shift_equivariant(self):
        """T1-01: scale collapsed onto the 1e-8 floor at the epoch offset, silently."""
        self._sweep(
            "StudentTEstimator",
            lambda c: StudentTEstimator(),
            _dyadic("t"),
            lambda d: {"loc": d.loc, "scale": d.scale},
            {"loc"},
        )

    def test_skew_normal_is_shift_equivariant(self):
        """Central moments alone were not enough: the batch mean's own rounding biased the third."""
        self._sweep(
            "SkewNormalEstimator",
            lambda c: SkewNormalEstimator(),
            _dyadic("skew"),
            lambda d: {"loc": d.loc, "scale": d.scale, "shape": d.shape},
            {"loc"},
        )

    def test_exgaussian_is_shift_equivariant(self):
        self._sweep(
            "ExponentiallyModifiedGaussianEstimator",
            lambda c: ExponentiallyModifiedGaussianEstimator(),
            _dyadic("emg"),
            lambda d: {"mu": d.mu, "sigma2": d.sigma2, "lam": d.lam},
            {"mu"},
        )

    def test_generalized_pareto_is_equivariant_when_the_threshold_moves_with_the_data(self):
        """Peaks over a threshold at ``loc + c`` are the same exceedances, so the fit must not move."""
        self._sweep(
            "GeneralizedParetoEstimator",
            lambda c: GeneralizedParetoEstimator(loc=c),
            _dyadic("gpd"),
            lambda d: {"loc": d.loc, "scale": d.scale, "shape": d.shape},
            {"loc"},
            location_ulps=0.0,
        )

    def test_multivariate_student_t_m_step_is_shift_equivariant(self):
        """One M-step -- the part that is a pure moment reduction -- must be exact."""
        self._sweep(
            "MultivariateStudentTEstimator (one M-step)",
            lambda c: MultivariateStudentTEstimator(dof=5.0, dim=3),
            _dyadic_matrix(),
            lambda d: {"mu0": d.mu[0], "shape00": d.shape[0][0], "shape01": d.shape[0][1]},
            {"mu0"},
            multivariate=True,
        )

    def test_log_gaussian_is_scale_equivariant(self):
        """A large multiplicative scale on the data IS a large additive offset on ``log(x)``."""
        log_data = np.round(np.random.RandomState(7).normal(0.0, 0.02, 600) / 2.0**-30) * 2.0**-30
        base = _m_steps(LogGaussianEstimator(), list(np.exp(log_data)))
        # 46 is ~1e20; 700 is the float64 ceiling, past which no positive x is representable.
        for log_offset in (46.0, 100.0, 300.0, 700.0):
            with self.subTest(log_offset=log_offset):
                got = _m_steps(LogGaussianEstimator(), list(np.exp(log_data + log_offset)))
                self.assertLessEqual(
                    abs(got.sigma2 - base.sigma2) / base.sigma2,
                    SCALE_TOLERANCE,
                    "log-Gaussian sigma2 moved when the data was scaled by exp(%g)" % log_offset,
                )
                self.assertLessEqual(
                    abs((got.mu - log_offset) - base.mu),
                    LOCATION_ULPS * float(np.spacing(log_offset)),
                )


class DocumentedMomentInversionTestCase(unittest.TestCase):
    """Each repaired family against its OWN documented formula, at every offset.

    An equivariance sweep alone would pass if a family were consistently wrong. These check the
    absolute answer against an exact-arithmetic reference computed from the sample.
    """

    def setUp(self):
        rng = np.random.default_rng(5150)  # the tester's own seed and sample shape
        self.sample = np.round(rng.normal(0.0, 2.0, 600) / GRID) * GRID
        self.variance = _exact_population_variance(self.sample)

    def test_gumbel_matches_its_documented_beta_at_every_offset(self):
        """``GumbelEstimator`` documents ``beta = sqrt(6 * var) / pi``; at 2**31 it returned 15.4x that."""
        reference = math.sqrt(6.0 * self.variance) / math.pi
        for offset in (0.0, 1.0e8, 3.0e8, 1.0e9, 1.7e9, float(2**31), 1.0e10):
            with self.subTest(offset=offset):
                fitted = optimize(list(self.sample + offset), GumbelEstimator(), out=None)
                self.assertAlmostEqual(fitted.scale / reference, 1.0, places=12)
                # loc = mean - beta * gamma, and both halves are the family's own documented form.
                expected_loc = float(np.mean(self.sample)) + offset - reference * 0.5772156649015329
                self.assertLessEqual(
                    abs(fitted.loc - expected_loc),
                    LOCATION_ULPS * float(np.spacing(max(offset, 1.0))),
                )

    def test_student_t_matches_its_moment_target_at_every_offset(self):
        """For df=5, ``Var = scale^2 * df/(df-2)``; at the epoch offset the fit hit the 1e-8 floor."""
        reference = math.sqrt(self.variance * 3.0 / 5.0)
        for offset in (0.0, 1.0e8, 3.0e8, 1.0e9, 1.7e9, float(2**31), 1.0e10):
            with self.subTest(offset=offset):
                fitted = optimize(list(self.sample + offset), StudentTEstimator(), out=None)
                self.assertAlmostEqual(fitted.scale / reference, 1.0, places=12)
                self.assertGreater(
                    fitted.scale,
                    1.0e-6,
                    "a fit on data with spread 2.0 must not land on the min_scale floor",
                )

    def test_gumbel_no_longer_raises_an_internal_em_error_on_epoch_offset_data(self):
        """The 3e8-1.7e9 band raised ``fused EM did not produce a finite objective ...`` instead of fitting."""
        for offset in (3.0e8, 1.0e9, 1.7e9, 1.0e10):
            with self.subTest(offset=offset):
                fitted = optimize(list(self.sample + offset), GumbelEstimator(), out=None)
                self.assertTrue(np.isfinite(fitted.scale) and fitted.scale > 0.0)

    def test_anscombe_series_shifted_to_the_epoch_keeps_its_scale(self):
        """The finding's second data set: 11 points, so nothing here is a large-sample effect."""
        series = {
            "y1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
            "y3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
        }
        for name, values in series.items():
            with self.subTest(series=name):
                base = optimize(values, StudentTEstimator(), out=None)
                shifted = optimize([v + 1.7e9 for v in values], StudentTEstimator(), out=None)
                self.assertAlmostEqual(shifted.scale / base.scale, 1.0, places=7)

    def test_generalized_pareto_keeps_its_shape_over_an_epoch_threshold(self):
        """Peaks over a threshold at epoch seconds fitted ``shape=0`` -- the exponential limit."""
        sample = _dyadic("gpd")
        base = _m_steps(GeneralizedParetoEstimator(loc=0.0), list(sample))
        self.assertGreater(base.shape, 0.05, "the reference fit must have a shape worth losing")
        for offset in (1.0e8, 1.7e9, float(2**31), 1.0e12):
            with self.subTest(offset=offset):
                got = _m_steps(GeneralizedParetoEstimator(loc=offset), list(sample + offset))
                self.assertAlmostEqual(got.shape / base.shape, 1.0, places=12)
                self.assertAlmostEqual(got.scale / base.scale, 1.0, places=12)


class AccumulationPathTestCase(unittest.TestCase):
    """The repair has to survive every route a sufficient statistic takes, not just one seq_update."""

    OFFSET = 1.7e9

    def _paths(self, estimator_factory, rows, read):
        """Fit the same rows through each accumulation route and return the parameter dicts."""

        def one_chunk(rows_):
            accumulator = estimator_factory().accumulator_factory().make()
            encoded = accumulator.acc_to_encoder().seq_encode(rows_)
            accumulator.seq_initialize(encoded, np.ones(len(rows_)), np.random.RandomState(0))
            return estimator_factory().estimate(float(len(rows_)), accumulator.value())

        def many_chunks(rows_):
            estimator = estimator_factory()
            parts = np.array_split(np.asarray(rows_, dtype=float), 7)
            accumulators = []
            for part in parts:
                accumulator = estimator.accumulator_factory().make()
                encoded = accumulator.acc_to_encoder().seq_encode(part.tolist())
                accumulator.seq_initialize(encoded, np.ones(len(part)), np.random.RandomState(0))
                accumulators.append(accumulator)
            pool = accumulators[0]
            for other in accumulators[1:]:
                pool.combine(other.value())
            return estimator.estimate(float(len(rows_)), pool.value())

        def scalar_updates(rows_):
            estimator = estimator_factory()
            accumulator = estimator.accumulator_factory().make()
            for row in rows_:
                accumulator.update(row, 1.0, None)
            return estimator.estimate(float(len(rows_)), accumulator.value())

        def scaled(rows_):
            estimator = estimator_factory()
            accumulator = estimator.accumulator_factory().make()
            encoded = accumulator.acc_to_encoder().seq_encode(rows_)
            accumulator.seq_initialize(encoded, np.ones(len(rows_)), np.random.RandomState(0))
            accumulator.scale(0.25)
            return estimator.estimate(float(len(rows_)) * 0.25, accumulator.value())

        def through_pickle(rows_):
            estimator = estimator_factory()
            accumulator = estimator.accumulator_factory().make()
            encoded = accumulator.acc_to_encoder().seq_encode(rows_)
            accumulator.seq_initialize(encoded, np.ones(len(rows_)), np.random.RandomState(0))
            restored = pickle.loads(pickle.dumps(accumulator.value()))
            merged = estimator.accumulator_factory().make()
            merged.combine(restored)
            return estimator.estimate(float(len(rows_)), merged.value())

        return {
            name: read(route(rows))
            for name, route in (
                ("seq_update", one_chunk),
                ("seven chunks then combine", many_chunks),
                ("scalar update", scalar_updates),
                ("scale(0.25)", scaled),
                ("pickled payload then combine", through_pickle),
            )
        }

    def test_every_accumulation_route_is_shift_equivariant(self):
        """A payload dropped by combine, scale or pickle would silently reinstate the raw path."""
        cases = (
            ("Gumbel", GumbelEstimator, _dyadic("gumbel"), lambda d: (d.loc, d.scale)),
            ("StudentT", StudentTEstimator, _dyadic("t"), lambda d: (d.loc, d.scale)),
        )
        for label, factory, sample, read in cases:
            base = self._paths(factory, list(sample), read)
            shifted = self._paths(factory, list(sample + self.OFFSET), read)
            for route, reference in base.items():
                with self.subTest(family=label, route=route):
                    self.assertAlmostEqual(shifted[route][1] / reference[1], 1.0, places=12)
                    self.assertLessEqual(
                        abs((shifted[route][0] - self.OFFSET) - reference[0]),
                        LOCATION_ULPS * float(np.spacing(self.OFFSET)),
                    )

    def test_generalized_pareto_survives_every_accumulation_route(self):
        sample = _dyadic("gpd")
        base = self._paths(lambda: GeneralizedParetoEstimator(loc=0.0), list(sample), lambda d: (d.scale, d.shape))
        shifted = self._paths(
            lambda: GeneralizedParetoEstimator(loc=self.OFFSET),
            list(sample + self.OFFSET),
            lambda d: (d.scale, d.shape),
        )
        for route, reference in base.items():
            with self.subTest(route=route):
                self.assertAlmostEqual(shifted[route][0] / reference[0], 1.0, places=12)
                self.assertAlmostEqual(shifted[route][1] / reference[1], 1.0, places=12)

    def test_fractional_em_weights_stay_shift_equivariant(self):
        """Mixture responsibilities are fractional; the anchored moments have to be weighted too."""
        sample = _dyadic("t")
        weights = np.round(np.random.RandomState(3).uniform(0.1, 1.0, len(sample)) * 4096) / 4096.0

        def fit(rows):
            estimator = StudentTEstimator()
            accumulator = estimator.accumulator_factory().make()
            encoded = accumulator.acc_to_encoder().seq_encode(rows)
            accumulator.seq_initialize(encoded, weights, np.random.RandomState(0))
            return estimator.estimate(float(np.sum(weights)), accumulator.value())

        base = fit(list(sample))
        shifted = fit(list(sample + self.OFFSET))
        self.assertAlmostEqual(shifted.scale / base.scale, 1.0, places=12)

    def test_multivariate_student_t_survives_chunked_combine(self):
        matrix = _dyadic_matrix()

        def fit(rows):
            estimator = MultivariateStudentTEstimator(dof=5.0, dim=3)
            parts = np.array_split(np.asarray(rows, dtype=float), 5)
            pool = None
            for part in parts:
                accumulator = estimator.accumulator_factory().make()
                encoded = accumulator.acc_to_encoder().seq_encode(part.tolist())
                accumulator.seq_initialize(encoded, np.ones(len(part)), np.random.RandomState(0))
                if pool is None:
                    pool = accumulator
                else:
                    pool.combine(accumulator.value())
            return estimator.estimate(float(len(rows)), pool.value())

        base = fit([list(r) for r in matrix])
        shifted = fit([list(r) for r in (matrix + self.OFFSET)])
        self.assertAlmostEqual(shifted.shape[0][0] / base.shape[0][0], 1.0, places=12)
        self.assertAlmostEqual(shifted.shape[0][1] / base.shape[0][1], 1.0, places=12)


class StatedLimitsTestCase(unittest.TestCase):
    """The two limits this wave states, measured rather than asserted."""

    def test_the_conditioning_gate_costs_the_same_below_it_for_every_family(self):
        """The gate keeps the raw path bit-identical while abs(mean)/spread stays under ~2000.

        That is a deliberate, family-wide design point, so the price it charges is pinned against
        the repaired Gaussian CONTROL rather than attributed to any family repaired this wave: no
        newly repaired family may cost more there than the control already does.
        """
        sample = _dyadic("normal")  # spread 2.0, so the gate activates above abs(mean) ~4000
        below_the_gate = (100.0, 500.0, 1000.0, 2000.0, 3000.0, 3900.0)

        def worst(make_estimator, read):
            base = read(_m_steps(make_estimator(), list(sample)))
            return max(abs(read(_m_steps(make_estimator(), list(sample + c))) - base) / base for c in below_the_gate)

        control = worst(GaussianEstimator, lambda d: d.sigma2)
        self.assertLess(control, 1.0e-8, "the control's own raw-path cost has moved")
        for label, make_estimator, read in (
            ("Gumbel", GumbelEstimator, lambda d: d.scale),
            ("StudentT", StudentTEstimator, lambda d: d.scale),
        ):
            with self.subTest(family=label):
                self.assertLessEqual(worst(make_estimator, read), control)

    def test_multivariate_student_t_em_is_limited_only_by_its_own_location_grid(self):
        """The residual is the granularity of ``mu``, not cancellation -- so snapping ``mu`` reproduces it.

        The EM reweighting is a function of ``x - mu``, and ``mu`` is a float64 at the data's
        magnitude. Fitting the un-shifted data while rounding ``mu`` onto exactly the grid that the
        offset forces reproduces the shifted fit, which no cancellation defect would do.
        """
        matrix = _dyadic_matrix()
        offset = 1.0e12
        estimator = MultivariateStudentTEstimator(dof=5.0, dim=3)

        def run(rows, snap_to=None):
            previous = None
            for _ in range(3):
                accumulator = estimator.accumulator_factory().make()
                encoded = accumulator.acc_to_encoder().seq_encode(rows)
                weights = np.ones(len(rows))
                if previous is None:
                    accumulator.seq_initialize(encoded, weights, np.random.RandomState(0))
                else:
                    accumulator.seq_update(encoded, weights, previous)
                previous = estimator.estimate(float(len(rows)), accumulator.value())
                if snap_to is not None:
                    previous = MultivariateStudentTDistribution(
                        previous.dof, (previous.mu + snap_to) - snap_to, previous.shape
                    )
            return np.asarray(previous.shape)

        shifted = run([list(r) for r in (matrix + offset)])
        snapped = run([list(r) for r in matrix], snap_to=offset)
        residual = float(np.max(np.abs(shifted - snapped) / np.abs(snapped)))
        self.assertLess(
            residual,
            1.0e-11,
            "snapping mu onto the offset's own grid should reproduce the shifted fit exactly; "
            "if it no longer does, the residual is not representational any more",
        )
        # And the residual against the UNSNAPPED fit stays within the granularity that explains it.
        plain = run([list(r) for r in matrix])
        bound = 40.0 * float(np.spacing(offset)) / 2.0  # ulp of the location, in units of the spread
        self.assertLess(float(np.max(np.abs(shifted - plain) / np.abs(plain))), bound)


class GeneralizedParetoPriorTestCase(unittest.TestCase):
    """The one prior in this class encoded as raw moments rather than as parameters.

    Gumbel, Student-t and Logistic all take ``suff_stat=(loc, scale)``, from which the prior
    variance is exact at any magnitude. The generalized Pareto takes ``(mean, second_moment)``, and
    recovering the variance as ``second_moment - mean**2`` destroys it at a large threshold: a
    ``GeneralizedParetoDistribution(2.0, 0.2, loc=1.7e9)`` prior recovered 0.0 from a true 10.4167,
    and at ``loc=1e8`` recovered 10.0 -- 4% low. That moved a fitted shape by 2.7e-2, silently.
    """

    def test_a_library_built_prior_is_threshold_equivariant(self):
        sample = _dyadic("gpd")
        base = _m_steps(GeneralizedParetoDistribution(2.0, 0.2, loc=0.0).estimator(8.0), list(sample))
        for offset in (1.0e8, 1.7e9, float(2**31)):
            with self.subTest(offset=offset), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                got = _m_steps(
                    GeneralizedParetoDistribution(2.0, 0.2, loc=offset).estimator(8.0),
                    list(sample + offset),
                )
                self.assertEqual([str(w.message) for w in caught], [])
                self.assertAlmostEqual(got.scale / base.scale, 1.0, places=12)
                self.assertAlmostEqual(got.shape / base.shape, 1.0, places=12)

    def test_the_prior_payload_survives_pickling(self):
        """Estimators are pickled by the Spark and multiprocessing reducers."""
        sample = _dyadic("gpd")
        estimator = GeneralizedParetoDistribution(2.0, 0.2, loc=1.7e9).estimator(8.0)
        direct = _m_steps(estimator, list(sample + 1.7e9))
        restored = _m_steps(pickle.loads(pickle.dumps(estimator)), list(sample + 1.7e9))
        self.assertEqual(restored.scale, direct.scale)
        self.assertEqual(restored.shape, direct.shape)

    def test_a_hand_built_prior_pair_that_cannot_carry_its_variance_warns(self):
        """The documented raw pair still works, and now says when it has stopped meaning anything."""
        sample = _dyadic("gpd")
        mean0 = 1.7e9 + 2.5
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _m_steps(
                GeneralizedParetoEstimator(loc=1.7e9, pseudo_count=8.0, suff_stat=(mean0, 10.4167 + mean0 * mean0)),
                list(sample + 1.7e9),
            )
        messages = [str(w.message) for w in caught]
        self.assertTrue(messages, "a prior whose variance was destroyed must not stay silent")
        self.assertIn("second_moment", messages[0])

    def test_an_ordinary_hand_built_prior_pair_stays_quiet(self):
        """The warning must name the destroyed-encoding regime only, not every raw pair."""
        sample = _dyadic("gpd")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = _m_steps(
                GeneralizedParetoEstimator(loc=0.0, pseudo_count=8.0, suff_stat=(2.5, 10.4167 + 6.25)),
                list(sample),
            )
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertGreater(fitted.scale, 0.0)

    def test_a_bounded_tail_prior_stays_quiet_even_though_its_variance_is_below_the_mean_squared(self):
        """The floor has to come from ``xi_min``, not from ``m**2``, or ordinary input warns.

        A generalized Pareto with exceedance mean ``m`` has variance ``m**2 / (1 - 2*xi)``, so every
        bounded (``xi < 0``) tail sits BELOW ``m**2`` -- 0.625 of it at ``xi=-0.3``, 0.091 at
        ``xi=-5``. An ``m**2`` floor would have flagged each of these perfectly ordinary priors.
        """
        sample = _dyadic("gpd")
        for shape in (-0.3, -1.0, -5.0):
            distribution = GeneralizedParetoDistribution(2.0, shape, loc=0.0)
            mean0, var0 = distribution.mean(), distribution.variance()
            self.assertLess(var0, mean0 * mean0, "this case only bites when var < m**2")
            with self.subTest(shape=shape), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _m_steps(
                    GeneralizedParetoEstimator(loc=0.0, pseudo_count=8.0, suff_stat=(mean0, var0 + mean0 * mean0)),
                    list(sample),
                )
                self.assertEqual([str(w.message) for w in caught], [])


class RawOnlyStatisticsTestCase(unittest.TestCase):
    """Statistics that arrive already reduced cannot be repaired -- so they must be NAMED."""

    def test_raw_only_statistics_warn_instead_of_returning_a_scale_silently(self):
        """The information cancellation destroyed is not in a reduced tuple any more."""
        sample = _dyadic("t") + 1.7e9
        raw = (float(np.sum(sample)), float(np.sum(sample * sample)), float(len(sample)))
        for label, estimator in (
            ("StudentT", StudentTEstimator()),
            ("Gumbel", GumbelEstimator()),
            ("GeneralizedPareto", GeneralizedParetoEstimator(loc=0.0)),
        ):
            with self.subTest(family=label), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.estimate(float(len(sample)), raw)
                messages = [str(w.message) for w in caught]
                self.assertTrue(messages, "%s returned a scale from uncorrectable moments silently" % label)
                self.assertIn("shift-anchored", messages[0])

    def test_well_conditioned_raw_statistics_do_not_warn(self):
        """The warning must not fire on ordinary data, on one observation, or on all-zero data.

        The ``constant`` case USED to be pinned here as "must stay quiet", on the reasoning that a
        degenerate component is what the scale floor is for. That reasoning is exactly what made the
        worst case the silent one -- see
        :meth:`test_raw_only_statistics_are_loud_when_cancellation_took_the_whole_spread` -- because
        a constant sample at 3.25 and an sd-2 sample at offset 1.7e9 leave raw moments that are
        genuinely indistinguishable: at magnitude ``m`` the raw form resolves no spread below about
        ``1.5e-8 * abs(m)``. Constant data at a nonzero value therefore warns now, and that case
        moved to
        :meth:`test_the_total_loss_message_does_not_claim_the_data_was_ill_conditioned`, where the
        message is checked to make the honest claim instead of the ill-conditioning one. What stays
        quiet is what really has no spread to lose: a single observation, and all-zero data.
        """
        for label, estimator in (("StudentT", StudentTEstimator()), ("Gumbel", GumbelEstimator())):
            for what, sample in (
                ("ordinary", _dyadic("t")),
                ("single-observation", np.full(1, 3.25)),
                ("all-zero", np.zeros(40)),
            ):
                raw = (float(np.sum(sample)), float(np.sum(sample * sample)), float(len(sample)))
                with self.subTest(family=label, data=what), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator.estimate(float(len(sample)), raw)
                    self.assertEqual([str(w.message) for w in caught], [])

    def test_raw_only_statistics_are_loud_when_cancellation_took_the_whole_spread(self):
        """The total-loss regime was the silent one across the WHOLE class, in three copies.

        The shared gate warned only while the computed ``E[x^2] - E[x]^2`` was still positive. Once
        cancellation had taken the entire spread -- the case that returns a scale collapsed onto the
        family floor, i.e. the most wrong answer the family can give -- it fell through and said
        nothing: in the scalar contract, in the vector contract, and in the multivariate Gaussian's
        own transcribed copy. Measured before the repair on sd ~2 data at offset 1.7e9 handed in as
        the declared raw tuple: Gumbel 1e-8 (true 1.4903), Student-t 1e-8 (true 1.4806), logistic
        1e-8 (true 1.0538), GEV 1e-12 (true 1.4903), generalized-Gaussian alpha 1e-6 (true 2.7127),
        Gaussian sigma2 2.89e10 (true 3.6536). Every one silent, and four of those six never called
        the disclosure at all -- while the Gumbel and Student-t docstrings promised that ``estimate``
        "warns rather than returning a scale it cannot stand behind".
        """
        sample = _dyadic("normal") + 1.7e9
        raw3 = (float(np.sum(sample)), float(np.sum(sample * sample)), float(len(sample)))
        cube = float(np.sum(sample**3))
        cases = (
            ("Gumbel", GumbelEstimator(), raw3),
            ("StudentT", StudentTEstimator(), raw3),
            ("Logistic", LogisticEstimator(), raw3),
            ("Gaussian", GaussianEstimator(), raw3 + (raw3[2],)),
            ("GeneralizedExtremeValue", GeneralizedExtremeValueEstimator(), (raw3[0], raw3[1], cube, raw3[2])),
            (
                "GeneralizedGaussian",
                GeneralizedGaussianEstimator(),
                (raw3[2], raw3[0], raw3[1], cube, float(np.sum(sample**4))),
            ),
        )
        for label, estimator, statistic in cases:
            with self.subTest(family=label), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.estimate(float(len(sample)), statistic)
                messages = [str(w.message) for w in caught]
                self.assertTrue(messages, "%s collapsed onto its floor silently" % label)
                joined = " ".join(messages)
                self.assertIn("shift-anchored", joined)
                self.assertIn("subtract a constant origin", joined)

    def test_the_total_loss_message_does_not_claim_the_data_was_ill_conditioned(self):
        """It cannot know that, so it must not say it -- both readings have to be offered.

        A genuinely constant sample and one whose spread cancellation destroyed leave identical raw
        moments. The graduated message ("too ill-conditioned ... mean^2/variance is X") states a
        measurement, and is only correct while a positive variance survives to measure it; the
        total-loss message has to name both possibilities and still point at the same remedy.
        """
        constant = np.full(40, 3.25)
        raw = (float(np.sum(constant)), float(np.sum(constant * constant)), float(len(constant)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GumbelEstimator().estimate(float(len(constant)), raw)
        self.assertEqual(len(caught), 1)
        message = str(caught[0].message)
        self.assertNotIn("too ill-conditioned", message)
        self.assertIn("genuinely constant sample", message)
        self.assertIn("cancellation destroyed", message)
        self.assertIn("not distinguishable", message)

    def test_the_multivariate_contract_closes_the_same_hole_coordinate_wise(self):
        """The vector twin had the identical fail-open, and the full-covariance family a third copy."""
        rows = _dyadic_matrix() + 1.7e9
        count = float(len(rows))
        estimator = MultivariateGaussianEstimator(dim=rows.shape[1])
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_update(rows, np.ones(len(rows)), None)
        stripped = tuple(accumulator.value())  # the declared exchange format: no anchored payload
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                estimator.estimate(count, stripped)
            except ValueError:
                pass  # the PSD guard may still refuse it; the point is that it was named first
        messages = [str(w.message) for w in caught]
        self.assertTrue(messages, "raw multivariate statistics collapsed silently")
        self.assertIn("shift-anchored", messages[0])

    def test_a_well_scaled_multivariate_raw_statistic_stays_quiet(self):
        """The coordinate-wise gate must not fire on the states the library legitimately produces."""
        rows = _dyadic_matrix()
        count = float(len(rows))
        estimator = MultivariateGaussianEstimator(dim=rows.shape[1])
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_update(rows, np.ones(len(rows)), None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.estimate(count, tuple(accumulator.value()))
        self.assertEqual([str(w.message) for w in caught], [])

    def test_an_engine_backed_fit_is_wrong_but_no_longer_silent(self):
        """The raw exchange path is reachable from the ordinary API, not only from a hand-built tuple.

        A compute engine stacks the declared ``StatisticSpec`` moments directly, so the accumulator's
        anchored payload never exists to be carried. On the tree BEFORE this repair -- which already
        had all four earlier waves -- ``optimize(x + 1.7e9, ..., engine=NUMPY_ENGINE)`` on sd ~2 data
        returned a Gaussian variance 7.9e9 times too large and Gumbel/Student-t/logistic scales 5.3%
        wrong, all with ZERO warnings. The wrongness cannot be repaired at the M-step (the engine's
        statistics no longer contain the spread), so what has to hold is that it is named.
        """
        from mixle.engines.numpy_engine import NUMPY_ENGINE

        sample = _dyadic("normal")
        for label, estimator in (
            ("Gaussian", GaussianEstimator),
            ("Gumbel", GumbelEstimator),
            ("StudentT", StudentTEstimator),
            ("Logistic", LogisticEstimator),
        ):
            with self.subTest(family=label), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                optimize(list(sample + 1.7e9), estimator(), engine=NUMPY_ENGINE, out=None)
                messages = [str(w.message) for w in caught]
                self.assertTrue(messages, "%s returned an engine-backed fit silently" % label)
                self.assertTrue(any("shift-anchored" in m for m in messages), messages)

    def test_an_engine_backed_fit_on_ordinary_data_stays_quiet_and_exact(self):
        """The disclosure must not fire on the well-scaled data every engine fit actually sees."""
        from mixle.engines.numpy_engine import NUMPY_ENGINE

        sample = _dyadic("normal")
        for label, estimator, read in (
            ("Gaussian", GaussianEstimator, lambda d: d.sigma2),
            ("Gumbel", GumbelEstimator, lambda d: d.scale),
            ("StudentT", StudentTEstimator, lambda d: d.scale),
            ("Logistic", LogisticEstimator, lambda d: d.scale),
        ):
            with self.subTest(family=label), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                engine_fit = optimize(list(sample), estimator(), engine=NUMPY_ENGINE, out=None)
                self.assertEqual([str(w.message) for w in caught], [])
            plain = optimize(list(sample), estimator(), out=None)
            self.assertAlmostEqual(read(engine_fit), read(plain), places=12)

    def test_a_payload_that_contradicts_its_own_tuple_is_ignored(self):
        """A hand-built statistic must not silently change the estimate its tuple alone supports."""
        sample = _dyadic("t")
        accumulator = StudentTEstimator().accumulator_factory().make()
        encoded = accumulator.acc_to_encoder().seq_encode(list(sample + 1.7e9))
        accumulator.seq_initialize(encoded, np.ones(len(sample)), np.random.RandomState(0))
        value = accumulator.value()
        self.assertIsNotNone(getattr(value, "anchored", None), "the offset data must have anchored")

        honest = StudentTEstimator().estimate(float(len(sample)), value)
        anchor, a_sum, a_sum2 = value.anchored
        value.anchored = (anchor + 1.0e6, a_sum, a_sum2)  # no longer implies the tuple's first moment
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ignored = StudentTEstimator().estimate(float(len(sample)), value)
        # Rejected payload -> the historical raw path, which for this data is exactly the case the
        # disclosure warning exists for. The point is that the contradictory payload was not used.
        self.assertNotAlmostEqual(ignored.scale, honest.scale, places=6)
        self.assertTrue(any("shift-anchored" in str(w.message) for w in caught))


class UnchangedBehaviourTestCase(unittest.TestCase):
    """Ordinary and degenerate data must be untouched by the repair."""

    def test_ordinary_data_never_activates_the_anchored_track(self):
        """A well-conditioned chunk must return the historical PLAIN tuple, byte for byte."""
        for label, estimator, sample in (
            ("Gumbel", GumbelEstimator(), _dyadic("gumbel")),
            ("StudentT", StudentTEstimator(), _dyadic("t")),
            ("GeneralizedPareto", GeneralizedParetoEstimator(loc=0.0), _dyadic("gpd")),
        ):
            with self.subTest(family=label):
                accumulator = estimator.accumulator_factory().make()
                encoded = accumulator.acc_to_encoder().seq_encode(list(sample))
                accumulator.seq_initialize(encoded, np.ones(len(sample)), np.random.RandomState(0))
                value = accumulator.value()
                self.assertIsNone(getattr(value, "anchored", None))
                self.assertEqual(type(value), tuple)
                expected = (
                    float(np.dot(sample, np.ones(len(sample)))),
                    float(np.dot(sample * sample, np.ones(len(sample)))),
                    float(len(sample)),
                )
                self.assertEqual(tuple(float(v) for v in value), expected)

    def test_constant_and_single_observation_data_still_reach_the_documented_floors(self):
        """The anchor activates on constant data by construction; the floors must still apply."""
        constant = [3.25] * 40
        self.assertEqual(_m_steps(GumbelEstimator(), constant).scale, 1.0e-8)
        self.assertEqual(_m_steps(StudentTEstimator(), constant).scale, 1.0e-8)
        self.assertEqual(_m_steps(SkewNormalEstimator(), constant).scale, 1.0e-12)
        self.assertEqual(_m_steps(ExponentiallyModifiedGaussianEstimator(), constant).sigma2, 1.0e-12)
        self.assertEqual(_m_steps(GumbelEstimator(), [7.5]).scale, 1.0e-8)
        self.assertEqual(_m_steps(StudentTEstimator(), [7.5]).scale, 1.0e-8)

    def test_a_generalized_pareto_threshold_at_the_data_is_still_the_degenerate_fit(self):
        """Every exceedance zero: the scale floor, not a negative scale or a raise."""
        fitted = _m_steps(GeneralizedParetoEstimator(loc=3.25), [3.25] * 40)
        self.assertEqual(fitted.scale, 1.0e-12)
        self.assertEqual(fitted.shape, 0.0)

    def test_a_multivariate_student_t_accumulator_can_grow_back_from_an_empty_statistic(self):
        """A starved EM component is a normal state; restoring one and updating it must work.

        ``student_t_moments`` returns ``None`` moment arrays for an empty statistic, so an
        accumulator restored from one keeps its dimension but has no arrays. Both the anchored fold
        and the raw fold need them, and before this the next observation raised
        ``TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'``.
        """
        estimator = MultivariateStudentTEstimator(dof=5.0, dim=2)
        for label, feed in (
            ("scalar update", lambda acc: acc.update([1.0, 2.0], 1.0, None)),
            (
                "seq_update",
                lambda acc: acc.seq_update(acc.acc_to_encoder().seq_encode([[1.0, 2.0], [3.0, 4.0]]), np.ones(2), None),
            ),
        ):
            with self.subTest(route=label):
                accumulator = estimator.accumulator_factory().make()
                accumulator.from_value((0.0, 0.0, None, None))
                self.assertIsNone(accumulator.sum_ux)
                feed(accumulator)
                count, sum_u, sum_ux, sum_uxx = accumulator.value()
                self.assertGreater(count, 0.0)
                self.assertIsNotNone(sum_ux)
                self.assertEqual(np.shape(sum_uxx), (2, 2))

    def test_a_constant_multivariate_student_t_component_is_still_the_ridge(self):
        """A starved or collapsed EM component is a normal state, not a refusal."""
        fitted = _m_steps(MultivariateStudentTEstimator(dof=5.0, dim=2), [[1.0, 2.0]] * 30, iterations=2)
        self.assertEqual(fitted.shape[0][0], 1.0e-12)
        self.assertEqual(fitted.shape[1][1], 1.0e-12)
        np.testing.assert_allclose(fitted.mu, [1.0, 2.0])


def _exact_variance_of_squares(values):
    """``Var(X^2)`` in exact rationals -- the quantity Rician and Nakagami both invert."""
    squares = [Fraction(float(v)) ** 2 for v in values]
    n = len(squares)
    mean = sum(squares) / n
    return sum((y - mean) ** 2 for y in squares) / n, mean


class SquaredMomentFamiliesTestCase(unittest.TestCase):
    """The same defect class, one transform along: families that difference the moments of ``x**2``.

    An additive offset takes a positive-support law out of its own family, so these two have no
    shift-equivariance property to test -- but the enumeration this file works from is "a location
    and/or scale parameter, OR any family whose M-step differences raw moments", and both of these
    invert ``Var(X^2) = E[X^4] - E[X^2]^2``. Their conditioning parameter is ``nu/sigma`` (Rician)
    and ``m`` (Nakagami), and both reach the same cancellation from the other side.
    """

    def _rice(self, ratio, n=2000, seed=31):
        rs = np.random.RandomState(seed)
        a = rs.normal(ratio, 1.0, n)
        b = rs.normal(0.0, 1.0, n)
        return np.sqrt(a * a + b * b)

    def test_rician_scale_matches_exact_arithmetic_at_every_nu_over_sigma(self):
        """The historical form lost ``eps * (nu/sigma)^2``: 5.4e-9 at 1e4, 4.1e-5 at 1e6, 44% at 1e8.

        44% is not a rounding complaint -- at ``nu/sigma = 1e8`` the discriminant went non-positive
        and the estimator returned ``sqrt(m2/2)``, the "no signal component at all" fallback, for
        data whose signal component is the whole story.
        """
        for ratio in (1.0e1, 1.0e4, 1.0e5, 1.0e6, 1.0e8):
            sample = self._rice(ratio)
            variance, mean = _exact_variance_of_squares(sample)
            # sigma^2 = Var(X^2) / (2 (E[X^2] + sqrt(E[X^2]^2 - Var(X^2)))), in exact rationals.
            discriminant = mean * mean - variance
            root = Fraction(math.sqrt(float(discriminant)))
            for _ in range(3):
                root = (root + discriminant / root) / 2
            reference = math.sqrt(float(variance / (2 * (mean + root))))
            with self.subTest(nu_over_sigma=ratio):
                fitted = _m_steps(RicianEstimator(), list(sample))
                self.assertLess(abs(fitted.sigma - reference) / reference, 1.0e-9)

    def test_nakagami_shape_matches_exact_arithmetic_at_every_m(self):
        """``m`` is the reciprocal relative spread of ``X^2``, so the raw form loses ``eps * m``."""
        rs = np.random.RandomState(31)
        for true_m in (1.0, 1.0e4, 1.0e8, 1.0e10):
            sample = np.sqrt(rs.gamma(shape=true_m, scale=1.0 / true_m, size=2000))
            variance, mean = _exact_variance_of_squares(sample)
            reference = float(mean * mean / variance)
            with self.subTest(m=true_m):
                fitted = _m_steps(NakagamiEstimator(), list(sample))
                self.assertLess(abs(fitted.m - reference) / reference, 1.0e-9)

    def test_rician_scoring_is_finite_where_the_scaled_bessel_is_not(self):
        """``ive(0, z)`` is NaN past ~1e10, and ``z = x*nu/sigma^2`` reaches it at nu/sigma ~1e5.

        Every log-density then came back NaN and the caller saw only "fused EM did not produce a
        finite objective from its non-finite initial model" -- the same opaque internal error the
        Gumbel family raised, on data that is genuinely Rician.
        """
        for ratio in (1.0e3, 1.0e5, 1.0e8, 1.0e10):
            model = RicianDistribution(ratio, 1.0)
            sample = np.array([ratio, ratio + 1.0, ratio - 1.0, 0.5 * ratio])
            with self.subTest(nu_over_sigma=ratio):
                scored = model.seq_log_density(model.dist_to_encoder().seq_encode(list(sample)))
                self.assertTrue(np.all(np.isfinite(scored)), scored)
                self.assertTrue(np.isfinite(model.log_density(float(sample[0]))))
                np.testing.assert_allclose(scored[0], model.log_density(float(sample[0])), rtol=1.0e-14)

    def test_the_two_log_bessel_branches_agree_where_both_are_valid(self):
        """The asymptotic branch has to be a continuation, not a second answer."""
        from scipy.special import ive

        # Imported here rather than at module scope: it is the private helper this repair added, and
        # a module-level import would turn "the branch is missing" into a collection error for the
        # whole file instead of a failure on the one test that measures it.
        from mixle.stats.univariate.continuous.rician import _log_i0e

        for z in (1.0e3, 1.0e4, 9.9e4, 1.0e5, 1.1e5, 1.0e6, 1.0e9):
            with self.subTest(z=z):
                np.testing.assert_allclose(_log_i0e(z), float(np.log(ive(0, z))), rtol=1.0e-14)
        for z in (1.0e10, 1.0e12, 1.0e300):
            with self.subTest(z=z):
                self.assertFalse(np.isfinite(ive(0, z)), "this test is pointless if scipy stops failing")
                self.assertTrue(np.isfinite(_log_i0e(z)))

    def test_the_rician_engine_kernel_matches_the_numpy_scorer_past_the_crossover(self):
        """The engine path reaches ``i0e``, which is exact where ``ive`` is not, so it needs no branch.

        Recorded as a measurement rather than an assumption: if a future scipy or torch breaks
        ``i0e`` the way ``ive`` is broken, this is where it surfaces instead of in a fit that
        silently stops converging.
        """
        from mixle.engines.numpy_engine import NUMPY_ENGINE

        for ratio in (1.0e3, 1.0e5, 1.0e8, 1.0e10):
            model = RicianDistribution(ratio, 1.0)
            sample = np.array([ratio, ratio + 1.0, 0.5 * ratio])
            encoded = model.dist_to_encoder().seq_encode(list(sample))
            with self.subTest(nu_over_sigma=ratio):
                through_engine = np.asarray(model.backend_seq_log_density(encoded, NUMPY_ENGINE))
                self.assertTrue(np.all(np.isfinite(through_engine)), through_engine)
                np.testing.assert_allclose(through_engine, model.seq_log_density(encoded), rtol=1.0e-13)

    def test_a_rician_fit_no_longer_raises_an_internal_em_error(self):
        """End to end, through ``optimize``, on the ratios that used to raise."""
        for ratio in (1.0e5, 1.0e6, 1.0e8):
            with self.subTest(nu_over_sigma=ratio):
                fitted = optimize(list(self._rice(ratio)), RicianEstimator(), out=None)
                self.assertLess(abs(fitted.nu - ratio) / ratio, 1.0e-6)
                self.assertGreater(fitted.sigma, 0.5)
                self.assertLess(fitted.sigma, 2.0)

    def test_ordinary_squared_moment_data_never_activates_the_anchored_track(self):
        """Well-conditioned data must return the historical PLAIN tuple, byte for byte."""
        rs = np.random.RandomState(41)
        sample = np.abs(rs.normal(3.0, 1.0, 400)) + 0.5
        squares = sample * sample
        expected = (400.0, float(np.dot(np.ones(400), squares)), float(np.dot(np.ones(400), squares * squares)))
        for label, estimator in (("Rician", RicianEstimator()), ("Nakagami", NakagamiEstimator())):
            with self.subTest(family=label):
                accumulator = estimator.accumulator_factory().make()
                encoded = accumulator.acc_to_encoder().seq_encode(list(sample))
                accumulator.seq_update(encoded, np.ones(400), None)
                value = accumulator.value()
                self.assertIsNone(getattr(value, "anchored", None))
                self.assertEqual(type(value), tuple)
                self.assertEqual(tuple(value), expected)

    def test_the_squared_moment_payload_survives_pickling_and_scaling(self):
        """The payload rides on a tuple subclass, so both routes that drop it are pinned."""
        sample = self._rice(1.0e6, n=200)
        for label, estimator in (("Rician", RicianEstimator()), ("Nakagami", NakagamiEstimator())):
            with self.subTest(family=label):
                accumulator = estimator.accumulator_factory().make()
                encoded = accumulator.acc_to_encoder().seq_encode(list(sample))
                accumulator.seq_update(encoded, np.ones(len(sample)), None)
                value = accumulator.value()
                self.assertIsNotNone(getattr(value, "anchored", None))
                restored = pickle.loads(pickle.dumps(value))
                self.assertIsNotNone(getattr(restored, "anchored", None))
                self.assertEqual(
                    repr(estimator.estimate(float(len(sample)), value)),
                    repr(estimator.estimate(float(len(sample)), restored)),
                )
                accumulator.scale(0.5)
                self.assertIsNotNone(getattr(accumulator.value(), "anchored", None))

    def test_raw_only_squared_moment_statistics_are_named(self):
        """These two joined the class, so they join its disclosure contract too."""
        sample = self._rice(1.0e6, n=200)
        squares = sample * sample
        raw = (float(len(sample)), float(np.sum(squares)), float(np.sum(squares * squares)))
        for label, estimator in (("Rician", RicianEstimator()), ("Nakagami", NakagamiEstimator())):
            with self.subTest(family=label), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.estimate(raw[0], raw)
                messages = [str(w.message) for w in caught]
                self.assertTrue(messages, "%s inverted uncorrectable moments silently" % label)
                self.assertIn("shift-anchored", messages[0])

    def test_a_degenerate_squared_moment_sample_still_reaches_its_documented_floor(self):
        """Constant and single-observation data are normal EM states, not refusals."""
        for label, estimator, read in (
            ("Rician", RicianEstimator(), lambda d: d.sigma),
            ("Nakagami", NakagamiEstimator(), lambda d: d.m),
        ):
            for what, rows in (("constant", [2.5] * 30), ("single", [2.5])):
                with self.subTest(family=label, data=what):
                    fitted = _m_steps(estimator, rows)
                    self.assertTrue(np.isfinite(read(fitted)))
                    self.assertGreater(read(fitted), 0.0)

    def test_an_inverse_gaussian_shape_clamp_is_disclosed_rather_than_silent(self):
        """``lam`` scales with the data, so nanoseconds instead of seconds walk into ``max_param``.

        Measured before: genuine ``IG(mu=2e12, lam=3e12)`` draws fitted ``lam = 1e12``, 3x too small,
        with ``numerical_repairs() == ()`` and no warning -- the same undisclosed-clamp shape that
        made the Student-t ``min_scale`` collapse a finding.
        """
        rs = np.random.RandomState(53)
        clamped = optimize(list(rs.wald(mean=2.0e12, scale=3.0e12, size=2000)), InverseGaussianEstimator(), out=None)
        self.assertEqual(clamped.lam, 1.0e12)
        self.assertTrue(any("lam-clamped" in note for note in clamped.numerical_repairs()), clamped.numerical_repairs())

        ordinary = optimize(list(rs.wald(mean=2.0, scale=3.0, size=2000)), InverseGaussianEstimator(), out=None)
        self.assertEqual((), ordinary.numerical_repairs())
        self.assertLess(abs(ordinary.lam - 3.0) / 3.0, 0.2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
