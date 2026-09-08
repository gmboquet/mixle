"""The PPL layer: posteriors that are posteriors, fits that persist, arguments that are checked.

Pass 04 of the ten adversarial reviews of the 0.8.1 candidate (P04-F04..F12) found a constrained
HMC "posterior" that was one point repeated with an acceptance rate of zero, every posterior-bearing
fit unpicklable against a docstring promising the record travels through pickling, a non-finite
penalty answered with an iteration-budget message, a bound ``explain_fit`` that forgot why the
router chose its route, an unfitted vector parameter failing inside numpy, and a grouped optimizer
receipt reporting zero iterations beside ``success: True``.
"""

from __future__ import annotations

import importlib.util
import pickle
import unittest
import warnings

import numpy as np

from mixle.ppl import Bernoulli, Beta, DiagGaussian, Gamma, Mix, Normal, Poisson, free, increasing, potential


def _quiet(callable_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return callable_()


DATA = [float(value) for value in np.random.RandomState(0).normal(5.0, 2.0, 300)]


def _increasing_panel():
    rng = np.random.RandomState(0)
    return np.stack([rng.normal(level, 0.5, 300) for level in (0.0, 2.0, 1.0, 3.0, 4.0)], axis=1).tolist()


class ConstrainedHmcTest(unittest.TestCase):
    """P04-F04: an active hard constraint left HMC accepting nothing, reported as a posterior."""

    def test_a_chain_that_accepted_nothing_is_refused(self):
        mu = Normal(0.0, 10.0, name="mu")
        with self.assertRaises(RuntimeError) as caught:
            _quiet(
                lambda: Normal(mu, free).fit(
                    DATA, how="hmc", draws=400, burn=100, constraints=[mu > 6.0], rng=np.random.RandomState(0)
                )
            )
        message = str(caught.exception)
        self.assertIn("accepted none of its proposals", message)
        self.assertIn("how='mcmc'", message)
        self.assertIn("penalty=", message)

    def test_an_inactive_constraint_and_an_unconstrained_fit_are_unaffected(self):
        mu = Normal(0.0, 10.0, name="mu")
        fitted = _quiet(
            lambda: Normal(mu, free).fit(
                DATA, how="hmc", draws=300, burn=100, constraints=[mu > 0.0], rng=np.random.RandomState(0)
            )
        )
        self.assertGreater(fitted.summary()["mu"]["std"], 0.0)
        self.assertGreater(fitted.result.acceptance_rate, 0.5)
        plain = _quiet(
            lambda: Normal(Normal(0.0, 10.0, name="m"), free).fit(
                DATA, how="hmc", draws=300, burn=100, rng=np.random.RandomState(0)
            )
        )
        self.assertGreater(plain.summary()["m"]["std"], 0.0)


class PosteriorPicklingTest(unittest.TestCase):
    """P04-F05: every posterior-bearing fit was unpicklable, against explain_fit's own promise."""

    def _fits(self):
        yield (
            "mcmc",
            _quiet(
                lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                    DATA, how="mcmc", draws=200, burn=50, rng=np.random.RandomState(0)
                )
            ),
        )
        yield (
            "hmc",
            _quiet(
                lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                    DATA, how="hmc", draws=200, burn=50, rng=np.random.RandomState(0)
                )
            ),
        )
        yield (
            "nuts",
            _quiet(
                lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                    DATA, how="nuts", draws=200, burn=100, rng=np.random.RandomState(0)
                )
            ),
        )
        yield (
            "ensemble",
            _quiet(
                lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                    DATA, how="ensemble", draws=200, burn=50, rng=np.random.RandomState(0)
                )
            ),
        )
        yield (
            "laplace",
            _quiet(
                lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                    DATA, how="laplace", rng=np.random.RandomState(0)
                )
            ),
        )
        yield (
            "vi",
            _quiet(
                lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(DATA, how="vi", rng=np.random.RandomState(0))
            ),
        )
        yield "conjugate-normal", _quiet(lambda: Normal(Normal(0.0, 10.0, name="mu"), 2.0).fit(DATA, how="conjugate"))
        yield "conjugate-nig", _quiet(lambda: Normal(free, free).fit(DATA, how="conjugate"))
        yield (
            "conjugate-beta",
            _quiet(lambda: Bernoulli(Beta(2.0, 2.0, name="p")).fit([1.0] * 40 + [0.0] * 60, how="conjugate")),
        )
        counts = [int(value) for value in np.random.RandomState(0).poisson(3, 200)]
        yield "conjugate-gamma", _quiet(lambda: Poisson(Gamma(2.0, 1.0, name="lam")).fit(counts, how="conjugate"))

    @staticmethod
    def _comparable(summary):
        """A summary with NaNs made comparable (a single chain's split-R-hat is a deliberate NaN)."""
        if isinstance(summary, dict):
            return {key: PosteriorPicklingTest._comparable(value) for key, value in summary.items()}
        if isinstance(summary, float) and np.isnan(summary):
            return "nan"
        if isinstance(summary, np.ndarray):
            return np.where(np.isnan(summary), -12345.0, summary).tolist()
        return summary

    def test_every_route_round_trips_through_pickle_with_its_summary(self):
        for label, fitted in self._fits():
            with self.subTest(route=label):
                restored = pickle.loads(pickle.dumps(fitted))
                self.assertEqual(self._comparable(restored.summary()), self._comparable(fitted.summary()))

    def test_a_restored_conjugate_posterior_still_draws_identically(self):
        fitted = _quiet(lambda: Normal(free, free).fit(DATA, how="conjugate"))
        restored = pickle.loads(pickle.dumps(fitted))
        np.testing.assert_allclose(
            restored.result.samples("mu", 20, np.random.RandomState(0)),
            fitted.result.samples("mu", 20, np.random.RandomState(0)),
        )

    def test_the_model_rebuilding_call_says_why_it_cannot_run(self):
        fitted = _quiet(
            lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                DATA, how="mcmc", draws=200, burn=50, rng=np.random.RandomState(0)
            )
        )
        self.assertEqual(np.asarray(fitted.result.pointwise_log_likelihood(DATA[:5])).shape, (200, 5))
        restored = pickle.loads(pickle.dumps(fitted))
        self.assertEqual(restored.explain_fit()["route"], "mcmc")
        with self.assertRaises(ValueError) as caught:
            restored.result.pointwise_log_likelihood(DATA[:5])
        self.assertIn("restored from a pickle", str(caught.exception))


class PenaltyAndPotentialValidationTest(unittest.TestCase):
    """P04-F07, F11: a non-finite penalty and a string potential failed as budget/attribute errors."""

    def test_penalty_is_validated_like_the_other_controls(self):
        panel = _increasing_panel()
        handle = free(5, name="v")
        for label, weight in (("inf", float("inf")), ("nan", float("nan")), ("zero", 0.0), ("negative", -1.0)):
            with self.subTest(penalty=label):
                with self.assertRaises(ValueError) as caught:
                    DiagGaussian(5, mean=handle, var=np.full(5, 0.25)).fit(
                        panel, how="map", constraints=[increasing(handle)], penalty=weight
                    )
                self.assertIn("penalty must be a finite positive number", str(caught.exception))
        with self.assertRaises(TypeError):
            DiagGaussian(5, mean=handle, var=np.full(5, 0.25)).fit(
                panel, how="map", constraints=[increasing(handle)], penalty="x"
            )

    def test_a_valid_penalty_still_fits(self):
        panel = _increasing_panel()
        handle = free(5, name="v2")
        fitted = _quiet(
            lambda: DiagGaussian(5, mean=handle, var=np.full(5, 0.25)).fit(
                panel, how="map", constraints=[increasing(handle)], penalty=100.0
            )
        )
        self.assertEqual(np.asarray(fitted.summary()["mean"]).shape, (5,))

    def test_potentials_must_be_potentials(self):
        for label, value in (("string", "x"), ("list of strings", ["x"]), ("integer", 3)):
            with self.subTest(potentials=label):
                with self.assertRaises(TypeError) as caught:
                    Normal(Normal(0.0, 10.0, name="mu"), free).fit(DATA, how="map", potentials=value)
                self.assertIn("potentials must be", str(caught.exception))

    def test_a_non_finite_objective_is_reported_as_such(self):
        mu = Normal(0.0, 10.0, name="mu")
        with self.assertRaises(RuntimeError) as caught:
            Normal(mu, free).fit(DATA, how="map", potentials=[potential(lambda m: float("nan"), mu)])
        message = str(caught.exception)
        self.assertIn("not finite at the starting point", message)
        self.assertNotIn("Raise max_iter", message)


class ExplainFitReasonTest(unittest.TestCase):
    """P04-F09: the bound record replaced the routing reason with 'explicit how=...'."""

    def test_an_auto_routed_fit_keeps_the_reason_the_router_gave(self):
        panel = _increasing_panel()
        handle = free(5, name="v")
        model = DiagGaussian(5, mean=handle, var=np.full(5, 0.25))
        before = model.explain_fit(constraints=[increasing(handle)])
        after = _quiet(lambda: model.fit(panel, how="auto", constraints=[increasing(handle)])).explain_fit()
        self.assertEqual(after["route"], before["route"])
        self.assertEqual(after["route_requested"], "auto")
        self.assertIn(before["reason"], after["reason"])

    def test_an_explicit_request_still_says_so(self):
        fitted = _quiet(lambda: Normal(Normal(0.0, 10.0, name="mu"), free).fit(DATA, how="map"))
        self.assertIn("explicit how='map'", fitted.explain_fit()["reason"])


class UnfittedParameterTest(unittest.TestCase):
    """P04-F10: an unfitted VECTOR parameter failed inside numpy instead of saying so."""

    def test_every_unfitted_model_gives_the_same_message(self):
        cases = (
            ("vector", lambda: DiagGaussian(5, mean=free(5, name="v"), var=np.full(5, 0.25)).summary()),
            ("scalar", lambda: Normal(free, free).summary()),
            ("mixture", lambda: Mix([Normal(free, free), Normal(free, free)]).summary()),
        )
        for label, produce in cases:
            with self.subTest(model=label):
                with self.assertRaises(ValueError) as caught:
                    produce()
                self.assertIn("unresolved `free` parameters", str(caught.exception))

    def test_a_fitted_vector_model_still_summarizes(self):
        handle = free(5, name="v")
        fitted = _quiet(
            lambda: DiagGaussian(5, mean=handle, var=np.full(5, 0.25)).fit(
                _increasing_panel(), how="map", constraints=[increasing(handle)]
            )
        )
        self.assertEqual(np.asarray(fitted.summary()["mean"]).shape, (5,))


class SamplerControlTest(unittest.TestCase):
    """P04-F11: draws=1 produced a zero-variance 'posterior'; rng took only one spelling."""

    def test_too_few_draws_to_diagnose_is_refused_for_one_chain_and_two(self):
        for chains in (1, 2):
            with self.subTest(chains=chains):
                with self.assertRaises(ValueError) as caught:
                    Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                        DATA, how="mcmc", draws=1, burn=0, chains=chains, rng=np.random.RandomState(0)
                    )
                self.assertIn("draws must be an integer >= 4", str(caught.exception))

    def test_the_sampler_accepts_the_rng_spellings_predict_accepts(self):
        for value in (0, np.random.default_rng(1), np.random.RandomState(2)):
            with self.subTest(rng=type(value).__name__):
                fitted = _quiet(
                    lambda v=value: Normal(Normal(0.0, 10.0, name="mu"), free).fit(
                        DATA, how="mcmc", draws=100, burn=20, rng=v
                    )
                )
                self.assertGreater(fitted.summary()["mu"]["std"], 0.0)
        with self.assertRaises(TypeError) as caught:
            Normal(Normal(0.0, 10.0, name="mu"), free).fit(DATA, how="mcmc", draws=100, burn=20, rng="x")
        self.assertIn("rng must be", str(caught.exception))


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "the gradient MAP path needs torch; pip install mixle[torch]")
class GroupedOptimizerReceiptTest(unittest.TestCase):
    """P04-F12: a derivative-free grouped MAP reported iterations=0 beside success=True."""

    @staticmethod
    def _groups():
        return [list(np.random.RandomState(k).normal(k - 2.0, 1.0, 30)) for k in range(6)]

    def test_the_derivative_free_path_reports_what_it_actually_counted(self):
        m = Normal(0.0, 100.0, name="m")
        theta = Normal(m, 3.0, name="th").each()
        fitted = _quiet(
            lambda: Normal(theta, free).fit(
                self._groups(), how="map", constraints=[m > 3.0], rng=np.random.RandomState(0)
            )
        )
        optimizer = fitted.summary()["optimizer"]
        self.assertNotIn("iterations", optimizer)
        self.assertGreater(optimizer["objective_evaluations"], 0)
        self.assertTrue(optimizer["success"])

    def test_the_gradient_path_still_reports_iterations(self):
        m = Normal(0.0, 100.0, name="m2")
        theta = Normal(m, 3.0, name="th2").each()
        fitted = _quiet(lambda: Normal(theta, free).fit(self._groups(), how="map", rng=np.random.RandomState(0)))
        optimizer = fitted.summary()["optimizer"]
        self.assertGreaterEqual(optimizer["iterations"], 1)
        self.assertNotIn("objective_evaluations", optimizer)


if __name__ == "__main__":
    unittest.main()
