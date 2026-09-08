"""The library surfaces the notebooks, tutorials and README exercise.

Passes 07 and 08 of the ten adversarial reviews of the 0.8.1 candidate found a PPL fit that was a
silent no-op on an all-constant model, a grouped Bernoulli likelihood that had stopped sampling at
all, a state-space fit that stopped at its budget without a word, a "numba" kernel built and timed
on an install with no numba, an effective sample size larger than the number of draws, sampled
moments printed under the word "exact", a taught argument that does nothing by default, and a
README module the library refused.
"""

from __future__ import annotations

import contextlib
import io
import math
import unittest
import warnings

import numpy as np

import mixle.stats as S
from mixle.inference import optimize
from mixle.ppl import Bernoulli, Beta, Field, LocalLevel, NegativeBinomial, Normal, Poisson, free


def _quiet(callable_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return callable_()


class NothingToFitTest(unittest.TestCase):
    """P07-F03: .fit() on a model whose parameters are all constants was a silent no-op."""

    COUNTS = [float(value) for value in np.random.RandomState(0).poisson(14.8, 164)]

    def test_an_all_constant_model_says_it_had_nothing_to_estimate(self):
        for label, model in (("Poisson", Poisson(2.0)), ("NegativeBinomial", NegativeBinomial(2.0, 0.5))):
            with self.subTest(family=label):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted = model.fit(self.COUNTS)
                self.assertIsNone(fitted.result)
                self.assertTrue(
                    any("nothing to estimate" in str(item.message) for item in caught),
                    [str(item.message) for item in caught],
                )

    def test_a_model_with_a_free_slot_fits_and_stays_quiet(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = Poisson(free).fit(self.COUNTS)
        self.assertAlmostEqual(fitted.dist.lam, float(np.mean(self.COUNTS)), places=9)
        self.assertFalse([item for item in caught if "nothing to estimate" in str(item.message)])


class GroupedBernoulliTest(unittest.TestCase):
    """P07-F04: a Bernoulli likelihood with a Beta group prior no longer sampled at all."""

    HITS = [18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7]
    TRIALS = 45
    A, B = 164.0, 453.9

    def _games(self):
        return [[1.0] * hits + [0.0] * (self.TRIALS - hits) for hits in self.HITS]

    def _exact(self, index):
        return (self.A + self.HITS[index]) / (self.A + self.B + self.TRIALS)

    def test_the_gradient_and_random_walk_samplers_recover_the_conjugate_posterior(self):
        for how, tolerance in (("nuts", 0.004), ("mcmc", 0.02)):
            with self.subTest(how=how):
                fitted = _quiet(
                    lambda h=how: Bernoulli(Beta(self.A, self.B).each()).fit(
                        self._games(), how=h, draws=1000, burn=500, rng=np.random.RandomState(0)
                    )
                )
                summary = fitted.summary()
                for index in (0, len(self.HITS) - 1):
                    self.assertAlmostEqual(summary["p[%d]" % index]["mean"], self._exact(index), delta=tolerance)
                self.assertGreater(summary.get("_acceptance_rate", 1.0), 0.05)

    def test_the_ensemble_sampler_moves_and_converges_with_enough_sweeps(self):
        # Affine-invariant ensemble sampling over eighteen coupled rates needs more sweeps than the
        # gradient samplers to reach the same accuracy; what matters here is that it samples at all.
        fitted = _quiet(
            lambda: Bernoulli(Beta(self.A, self.B).each()).fit(
                self._games(), how="ensemble", draws=6000, burn=3000, rng=np.random.RandomState(0)
            )
        )
        summary = fitted.summary()
        self.assertGreater(summary.get("_acceptance_rate", 0.0), 0.05)
        for index in (0, len(self.HITS) - 1):
            self.assertAlmostEqual(summary["p[%d]" % index]["mean"], self._exact(index), delta=0.02)

    def test_the_rates_are_reported_as_rates_and_shrink_toward_the_prior(self):
        fitted = _quiet(
            lambda: Bernoulli(Beta(self.A, self.B).each()).fit(
                self._games(), how="nuts", draws=800, burn=400, rng=np.random.RandomState(0)
            )
        )
        summary = fitted.summary()
        rates = [summary["p[%d]" % index]["mean"] for index in range(len(self.HITS))]
        self.assertTrue(all(0.0 < rate < 1.0 for rate in rates))
        raw = [hits / self.TRIALS for hits in self.HITS]
        self.assertLess(max(rates) - min(rates), max(raw) - min(raw))  # shrinkage toward the prior

    def test_an_unsupported_grouped_pair_names_both_supported_ones(self):
        with self.assertRaises(NotImplementedError) as caught:
            Poisson(Beta(1.0, 1.0).each()).fit([[1.0, 2.0]], how="mcmc", draws=10, burn=5)
        message = str(caught.exception)
        self.assertIn("Normal", message)
        self.assertIn("Bernoulli", message)


class StateSpaceCapTest(unittest.TestCase):
    """P07-F05: a state-space fit stopped at its budget with no signal at all."""

    @staticmethod
    def _series():
        rng = np.random.RandomState(0)
        level = np.cumsum(rng.normal(0.0, 3.0, 739)) + 100.0
        return (level + rng.normal(0.0, 24.0, 739)).tolist()

    def test_an_unconverged_fit_says_so(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = LocalLevel().fit(self._series()).result
        self.assertFalse(result.converged)
        self.assertEqual(result.termination_reason, "iteration_limit")
        self.assertTrue(
            any("state-space EM stopped at the max_its cap" in str(item.message) for item in caught),
            [str(item.message) for item in caught],
        )


class NumbaKernelTest(unittest.TestCase):
    """P07-F15: a 'numba' kernel was built and timed on an install without numba."""

    @staticmethod
    def _model_and_estimator():
        model = S.MixtureDistribution([S.GaussianDistribution(0.0, 1.0), S.GaussianDistribution(5.0, 1.0)], [0.5, 0.5])
        return model, S.MixtureEstimator([S.GaussianEstimator(), S.GaussianEstimator()])

    def test_an_explicit_numba_request_is_refused_without_numba(self):
        import mixle.stats.compute.kernel as kernel_module
        from mixle.engines import NUMPY_ENGINE
        from mixle.stats.compute.kernel import (
            GeneratedNumbaKernelFactory,
            KernelCapabilityDeclinedError,
            NumbaKernelFactory,
        )

        model, estimator = self._model_and_estimator()
        present = kernel_module.HAS_NUMBA
        try:
            kernel_module.HAS_NUMBA = False
            with self.assertRaises(KernelCapabilityDeclinedError) as caught:
                NumbaKernelFactory().build(model, NUMPY_ENGINE, estimator=estimator)
            self.assertIn("numba is not installed", str(caught.exception))
            # The safe factory keeps its own contract: it declines to the generic kernel.
            fallback = GeneratedNumbaKernelFactory().build(model, NUMPY_ENGINE, estimator=estimator)
            self.assertEqual(type(fallback).__name__, "GenericKernel")
        finally:
            kernel_module.HAS_NUMBA = present


class MarkovChainVocabularyTest(unittest.TestCase):
    """P07-F13: pseudo-count smoothing covers the state set the fit has, and now says so."""

    SEQUENCES = [["a", "b", "a", "b"], ["b", "a", "c"]]

    def test_a_declared_vocabulary_gives_an_unseen_symbol_finite_mass(self):
        from mixle.stats import seq_encode

        inferred = _quiet(lambda: optimize(self.SEQUENCES, S.MarkovChainEstimator(pseudo_count=0.1), max_its=2))
        declared = _quiet(
            lambda: optimize(
                self.SEQUENCES, S.MarkovChainEstimator(pseudo_count=0.1, levels=["a", "b", "c", "z"]), max_its=2
            )
        )
        unseen = [["a", "z"]]
        self.assertEqual(float(inferred.seq_log_density(seq_encode(unseen, model=inferred)[0][1])[0]), -np.inf)
        self.assertTrue(math.isfinite(float(declared.seq_log_density(seq_encode(unseen, model=declared)[0][1])[0])))

    def test_the_docstring_names_the_escape(self):
        self.assertIn("levels", S.MarkovChainEstimator.__init__.__doc__)
        self.assertIn("full vocabulary", S.MarkovChainEstimator.__init__.__doc__)


class EffectiveSampleSizeTest(unittest.TestCase):
    """P08-F08: the reported ESS exceeded the number of draws by up to log10(N)x."""

    def test_ess_never_exceeds_the_draw_count(self):
        data = [float(value) for value in np.random.RandomState(0).normal(5.0, 2.0, 4000)]
        fitted = _quiet(
            lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                data, how="hmc", draws=1000, burn=500, rng=np.random.RandomState(2)
            )
        )
        raw = np.atleast_1d(np.asarray(fitted.result.raw.effective_sample_size(), dtype=float))
        self.assertTrue(np.all(raw <= 1000.0 + 1e-9), raw)
        for value in fitted.result.bulk_ess.values():
            self.assertLessEqual(value, 1000.0 + 1e-9)

    def test_a_correlated_chain_still_reports_far_fewer(self):
        data = [float(value) for value in np.random.RandomState(0).normal(5.0, 2.0, 4000)]
        fitted = _quiet(
            lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                data, how="mcmc", draws=1500, burn=500, rng=np.random.RandomState(2)
            )
        )
        raw = np.atleast_1d(np.asarray(fitted.result.raw.effective_sample_size(), dtype=float))
        self.assertTrue(np.all(raw < 1500.0), raw)


class AffineMomentTest(unittest.TestCase):
    """P08-F14: 'affine maps are exact' beside a sampled mean of 1.02 and variance of 8.95."""

    def test_an_affine_map_has_exact_moments(self):
        affine = 3 * Normal(0, 1) + 1
        self.assertAlmostEqual(affine.mean(), 1.0, places=12)
        self.assertAlmostEqual(affine.var(), 9.0, places=12)
        shifted = 2 * Normal(5, 4) - 3
        self.assertAlmostEqual(shifted.mean(), 7.0, places=12)
        self.assertAlmostEqual(shifted.var(), 64.0, places=12)

    def test_a_non_affine_transform_still_falls_back_to_sampling(self):
        lognormal = Normal(0, 1).exp()
        self.assertAlmostEqual(lognormal.mean(), math.exp(0.5), delta=0.05)


class PrintIterTest(unittest.TestCase):
    """P08-F04: print_iter was inert without out=, which nine tutorial cells relied on.

    The first repair only WARNED, and gated the warning on ``print_iter not in (0, 1)`` -- so
    ``print_iter=1``, the exact spelling those cells use and the one the finding was about, got
    neither output nor a note, while the CHANGELOG said it printed. An argument whose only
    documented purpose is to produce output now produces it.
    """

    DATA = [float(value) for value in np.random.RandomState(0).normal(size=200)]

    def _fit(self, **kwargs):
        """Return ``(stdout_lines, print_iter_warnings)`` for one optimize call."""
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                optimize(
                    self.DATA,
                    S.MixtureEstimator([S.GaussianEstimator()] * 2),
                    max_its=3,
                    rng=np.random.RandomState(0),
                    **kwargs,
                )
        lines = [line for line in captured.getvalue().splitlines() if line.strip()]
        return lines, [str(item.message) for item in caught if "print_iter" in str(item.message)]

    def test_the_tutorial_spelling_prints(self):
        lines, notes = self._fit(print_iter=1)
        self.assertEqual(len(lines), 3)  # one per iteration, the cadence that was asked for
        self.assertEqual(notes, [])

    def test_a_wider_cadence_prints_less_often(self):
        lines, notes = self._fit(print_iter=2)
        self.assertTrue(lines)
        self.assertLess(len(lines), 3)
        self.assertEqual(notes, [])

    def test_an_absent_print_iter_keeps_the_library_quiet(self):
        self.assertEqual(self._fit(), ([], []))

    def test_zero_asks_for_no_lines_and_gets_none(self):
        self.assertEqual(self._fit(print_iter=0), ([], []))

    def test_an_explicit_out_is_still_where_the_lines_go(self):
        buffer = io.StringIO()
        lines, notes = self._fit(print_iter=1, out=buffer)
        self.assertEqual(lines, [])  # not duplicated onto stdout
        self.assertEqual(notes, [])
        self.assertEqual(len([line for line in buffer.getvalue().splitlines() if line.strip()]), 3)

    def test_restarts_do_not_start_printing_on_their_own(self):
        """``best_of`` forwards its own default, so the sentinel has to reach it too."""
        from mixle.inference import best_of

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_of(
                    self.DATA,
                    self.DATA[:50],
                    S.MixtureEstimator([S.GaussianEstimator()] * 2),
                    2,
                    3,
                    1.0,
                    1.0e-8,
                    np.random.RandomState(0),
                )
        self.assertEqual(captured.getvalue().strip(), "")


class TorchModuleShapeTest(unittest.TestCase):
    """P08-F17: the README's own module returned (N, 1) scores and was refused."""

    def test_the_obvious_scalar_density_module_fits(self):
        torch = __import__("torch")

        class ScalarDensity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mu = torch.nn.Parameter(torch.zeros(1))
                self.log_sigma = torch.nn.Parameter(torch.zeros(1))

            def log_density(self, x):
                sigma = torch.exp(self.log_sigma)
                z = (x - self.mu) / sigma
                return -0.5 * z * z - torch.log(sigma) - 0.9189385332046727

        data = [float(value) for value in np.random.RandomState(0).normal(3.0, 2.0, 500)]
        fitted = _quiet(lambda: optimize(data, ScalarDensity(), max_its=20, rng=np.random.RandomState(0)))
        module = fitted.module
        self.assertAlmostEqual(float(module.mu.detach()[0]), float(np.mean(data)), delta=0.05)
        self.assertAlmostEqual(float(torch.exp(module.log_sigma.detach())[0]), float(np.std(data)), delta=0.05)

    def test_a_genuinely_ambiguous_shape_is_still_refused_with_the_fix_named(self):
        torch = __import__("torch")

        class PerDimension(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.zeros(1))

            def log_density(self, x):
                return self.value + torch.zeros((len(x), 3))

        with self.assertRaises(RuntimeError) as caught:
            optimize([0.0, 1.0, 2.0] * 20, PerDimension(), max_its=1)
        self.assertIn("sum(-1)", str(caught.exception))


class RegressionSurfaceTest(unittest.TestCase):
    """P08-F15: a fitted regression printed as unfitted and could not predict at new covariates."""

    @staticmethod
    def _fit():
        rng = np.random.RandomState(0)
        x, z = rng.normal(size=300), rng.normal(size=300)
        y = (2.0 * x - 1.0 * z + 0.5 + rng.normal(0.0, 0.3, 300)).tolist()
        model = Normal(free * Field("x") + free * Field("z") + free, free)
        return model.fit(y, given={"x": x, "z": z}), x, z

    def test_a_fitted_regression_prints_as_fitted(self):
        fitted, _, _ = self._fit()
        printed = repr(fitted)
        self.assertNotIn("bound=None", printed)
        self.assertIn("fitted", printed)
        self.assertIn("x=", printed)

    def test_given_accepts_a_dataframe(self):
        import pandas as pd

        rng = np.random.RandomState(0)
        x, z = rng.normal(size=300), rng.normal(size=300)
        y = (2.0 * x - 1.0 * z + 0.5 + rng.normal(0.0, 0.3, 300)).tolist()
        model = Normal(free * Field("x") + free * Field("z") + free, free)
        from_frame = model.fit(y, given=pd.DataFrame({"x": x, "z": z}))
        from_dict = model.fit(y, given={"x": x, "z": z})
        for name in from_dict.params:
            self.assertAlmostEqual(from_frame.params[name]["mean"], from_dict.params[name]["mean"], places=12)

    def test_predict_takes_the_covariates_to_predict_at(self):
        fitted, _, _ = self._fit()
        drawn = np.asarray(fitted.predict({"x": [1.0, 0.0], "z": [0.0, 1.0]}, rng=np.random.RandomState(0)))
        self.assertEqual(drawn.shape, (2,))
        self.assertAlmostEqual(float(drawn[0]), 2.5, delta=1.5)
        self.assertAlmostEqual(float(drawn[1]), -0.5, delta=1.5)

    def test_a_plain_fit_still_predicts_a_draw_count(self):
        data = [float(value) for value in np.random.RandomState(0).normal(5.0, 2.0, 200)]
        fitted = _quiet(
            lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                data, how="mcmc", draws=200, burn=50, rng=np.random.RandomState(0)
            )
        )
        self.assertEqual(np.asarray(fitted.predict(5, rng=np.random.RandomState(0))).shape, (5,))


if __name__ == "__main__":
    unittest.main()
