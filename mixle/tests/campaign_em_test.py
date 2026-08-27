"""How EM starts: init-strategy validation, the default initialization, and stalled runs.

Campaign findings covered here:

* an unrecognised ``init=`` used to be an exact no-op, so the scikit-learn spellings of the option
  the docstring recommends ("k-means++", "kmeans") silently produced the fit the caller was trying
  to avoid -- and silently disarmed ``robust=True`` when passed alongside it;
* the default Dirichlet initialization started every component at the pooled law, so a mixture over
  visibly separated clusters converged to a single cluster in the empty middle;
* a ``run_em`` whose first step is rejected returns its initial model unchanged, which used to be
  indistinguishable from a fit of the full dataset.
"""

import unittest
import warnings

import numpy as np
from numpy.random import RandomState

from mixle.inference import optimize
from mixle.inference.em import GeneralizedEM, StandardEM, run_em
from mixle.stats import (
    GaussianEstimator,
    MixtureEstimator,
    MultivariateGaussianEstimator,
)
from mixle.stats.compute.sequence import seq_encode, seq_initialize
from mixle.stats.latent.gaussian_mixture import GaussianMixtureEstimator
from mixle.stats.latent.mixture import (
    MIXTURE_INIT_STRATEGIES,
    MixtureAccumulator,
    MixtureAccumulatorFactory,
)
from mixle.stats.univariate.continuous.gaussian import GaussianAccumulator


def _two_clusters(seed=1000, sep=6.0, n=300):
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(0.0, 1.0, n), rng.normal(sep, 1.0, n)].tolist()


def _gaussian_mixture_estimator(**kwargs):
    return MixtureEstimator([GaussianEstimator(), GaussianEstimator()], **kwargs)


class MixtureInitValidationTest(unittest.TestCase):
    """An init strategy the library does not implement must be refused, not absorbed."""

    UNRECOGNISED = ("k-means++", "kmeans", "KMEANS++", "random", "garbage", "", 42, True, None.__class__)

    def test_unrecognised_init_is_rejected_naming_the_accepted_set(self):
        for value in self.UNRECOGNISED:
            with self.subTest(init=repr(value)):
                with self.assertRaises(ValueError) as ctx:
                    _gaussian_mixture_estimator(init=value)
                message = str(ctx.exception)
                self.assertIn("init must be one of", message)
                for accepted in MIXTURE_INIT_STRATEGIES:
                    self.assertIn(accepted, message)

    def test_accepted_init_values_are_the_documented_closed_set(self):
        self.assertEqual(MIXTURE_INIT_STRATEGIES, ("dirichlet", "kmeans++"))
        for value in MIXTURE_INIT_STRATEGIES:
            with self.subTest(init=value):
                self.assertEqual(_gaussian_mixture_estimator(init=value).init, value)

    def test_none_selects_the_default_strategy(self):
        self.assertEqual(_gaussian_mixture_estimator().init, "kmeans++")
        self.assertEqual(_gaussian_mixture_estimator(init=None).init, "kmeans++")
        self.assertEqual(_gaussian_mixture_estimator(robust=True).init, "kmeans++")

    def test_reference_library_spellings_get_a_suggestion(self):
        for value, suggestion in (
            ("k-means++", "kmeans++"),  # sklearn.cluster.KMeans(init=...)
            ("kmeans", "kmeans++"),  # sklearn.mixture.GaussianMixture(init_params=...)
            ("random", "dirichlet"),  # GaussianMixture(init_params="random")
        ):
            with self.subTest(init=value):
                with self.assertRaises(ValueError) as ctx:
                    _gaussian_mixture_estimator(init=value)
                self.assertIn('did you mean "%s"?' % suggestion, str(ctx.exception))

    def test_a_bogus_init_can_no_longer_disarm_robust_mode(self):
        # The worst shape of the old defect: robust=True is the library's advertised safe path, and
        # a redundant misspelled init alongside it silently reverted the fit to the default one.
        with self.assertRaises(ValueError):
            _gaussian_mixture_estimator(robust=True, init="k-means++")

    def test_the_accumulator_layer_policies_init_too(self):
        for factory in (
            lambda value: MixtureAccumulator([GaussianAccumulator(), GaussianAccumulator()], init=value),
            lambda value: MixtureAccumulatorFactory([], init=value),
        ):
            with self.subTest(factory=factory):
                self.assertEqual(factory("dirichlet").init, "dirichlet")
                with self.assertRaises(ValueError):
                    factory("k-means++")


class MixtureDefaultInitTest(unittest.TestCase):
    """The default start must separate ordinary, well-separated clusters inside the default budget."""

    def test_default_fit_separates_two_clusters_within_the_default_budget(self):
        data = _two_clusters()
        model = optimize(data, _gaussian_mixture_estimator(), out=None)
        means = sorted(component.mu for component in model.components)

        # Clusters live at 0 and 6. The Dirichlet start returned both means at ~3 (one cluster in
        # the empty middle) and converged=False at the max_its cap.
        self.assertLess(means[0], 1.5)
        self.assertGreater(means[1], 4.5)
        self.assertTrue(model.fit_provenance().converged)

    def test_default_start_is_never_worse_than_the_classical_dirichlet_start(self):
        for seed in (1000, 1001, 1002):
            for sep in (2.0, 4.0, 6.0):
                with self.subTest(seed=seed, sep=sep):
                    data = _two_clusters(seed=seed, sep=sep)
                    default_fit = optimize(data, _gaussian_mixture_estimator(), out=None)
                    dirichlet_fit = optimize(data, _gaussian_mixture_estimator(init="dirichlet"), out=None)
                    self.assertGreaterEqual(
                        default_fit.fit_provenance().final_objective,
                        dirichlet_fit.fit_provenance().final_objective,
                    )

    def test_dirichlet_remains_available_for_callers_who_want_it(self):
        data = _two_clusters()
        model = optimize(data, _gaussian_mixture_estimator(init="dirichlet"), out=None)
        self.assertEqual(len(model.components), 2)


class GaussianMixtureInitPlumbingTest(unittest.TestCase):
    """``init`` has to survive the Gaussian-mixture specialization's accumulator repack."""

    def _estimator(self, **kwargs):
        return GaussianMixtureEstimator(
            [MultivariateGaussianEstimator(dim=2), MultivariateGaussianEstimator(dim=2)], **kwargs
        )

    def test_configured_init_reaches_the_accumulator(self):
        for value in MIXTURE_INIT_STRATEGIES:
            with self.subTest(init=value):
                factory = self._estimator(init=value).accumulator_factory()
                self.assertEqual(factory.init, value)
                self.assertEqual(factory.make().init, value)

    def test_default_init_reaches_the_accumulator(self):
        self.assertEqual(self._estimator().accumulator_factory().make().init, "kmeans++")

    def test_robust_mode_is_available_on_the_gaussian_specialization(self):
        self.assertGreater(self._estimator(robust=True).w_min, 0.0)


class MixtureScalarInitializationTest(unittest.TestCase):
    """The scalar path cannot k-means++ seed; that limitation has to be visible, not silent."""

    def test_scalar_initialize_uses_the_dirichlet_draw_under_every_strategy(self):
        data = [(0.0,), (5.0,)]
        counts = {}
        for value in MIXTURE_INIT_STRATEGIES:
            acc = MixtureAccumulator([GaussianAccumulator(), GaussianAccumulator()], init=value)
            for observation in data:
                acc.initialize(observation[0], 1.0, RandomState(12))
            counts[value] = acc.value()[0]
        np.testing.assert_allclose(counts["dirichlet"], counts["kmeans++"])


class StalledRunEMTest(unittest.TestCase):
    """A run that accepts no step returns its initialization; that must not present as a fit."""

    def _fixture(self):
        rng = np.random.default_rng(7)
        data = [(float(v),) for v in rng.normal(0.0, 1.0, 200)]
        estimator = MultivariateGaussianEstimator(dim=1)
        enc = seq_encode(data, estimator=estimator)
        initial = seq_initialize(enc_data=enc, estimator=estimator, rng=RandomState(3), p=1.0)
        return enc, estimator, initial

    @staticmethod
    def _worse_model(enc_data, estimator, model, engine):
        """A deliberately likelihood-decreasing candidate, so run_em's own gate rejects it."""
        return type(model)(model.mu + 50.0, model.covar)

    def test_a_run_that_accepts_nothing_warns_and_returns_the_initialization(self):
        enc, estimator, initial = self._fixture()
        strategy = GeneralizedEM(self._worse_model, require_improvement=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = run_em(enc, estimator, initial, strategy=strategy)

        np.testing.assert_array_equal(fitted.mu, initial.mu)
        provenance = fitted.fit_provenance()
        self.assertFalse(provenance.converged)
        self.assertEqual(provenance.iterations, 1)

        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(messages), 1, messages)
        self.assertIn("accepted no EM step", messages[0])
        self.assertIn("initial model unchanged", messages[0])

    def test_an_ordinary_run_stays_quiet(self):
        enc, estimator, initial = self._fixture()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = run_em(enc, estimator, initial, strategy=StandardEM())
        self.assertTrue(fitted.fit_provenance().converged)
        self.assertEqual([str(w.message) for w in caught if issubclass(w.category, UserWarning)], [])

    def test_a_warm_start_already_at_the_fixed_point_stays_quiet(self):
        # The disclosure must not fire on a legitimately finished run: restarting from a converged
        # model produces a first step with (numerically) zero gain, which the tolerance accepts.
        enc, estimator, initial = self._fixture()
        converged = run_em(enc, estimator, initial, strategy=StandardEM())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            refit = run_em(enc, estimator, converged, strategy=StandardEM())
        self.assertTrue(refit.fit_provenance().converged)
        self.assertEqual([str(w.message) for w in caught if issubclass(w.category, UserWarning)], [])


if __name__ == "__main__":
    unittest.main()
