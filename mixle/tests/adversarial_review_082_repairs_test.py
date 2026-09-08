"""What the 0.8.2 adversarial review found wrong with the 0.8.2 repairs themselves.

Four independent reviewers attacked the repairs this release makes on top of the published 0.8.1.
The defects below are theirs (identifiers ``R<pass>-F<nn>``), and every one is a repair that did not
do what its own CHANGELOG entry said: a guard wired into one route of four, a disclosure that named
the wrong selection, a condition reported differently by two routes to the same model, and a
restored posterior answering a question it could no longer answer.
"""

from __future__ import annotations

import pickle
import unittest
import warnings

import numpy as np

from mixle.inference import learn_bayesian_network, optimize
from mixle.stats import GaussianDistribution, GaussianEstimator
from mixle.stats.compute.sequence import seq_estimate


def _quiet(callable_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return callable_()


class DownhillCapNoteTest(unittest.TestCase):
    """R02-F05: the note claimed the best iterate under the selection that returns the worst."""

    DATA = [float(value) for value in np.random.RandomState(0).normal(3, 2, 200)]

    def _descending_fit(self, track_best):
        calls = {"n": 0}

        def worsening(enc, est, model):
            calls["n"] += 1
            nxt = seq_estimate(enc, est, model)
            if calls["n"] >= 2:
                return GaussianDistribution(nxt.mu + 50.0 * calls["n"], nxt.sigma2)
            return nxt

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(
                self.DATA,
                GaussianEstimator(),
                delta=1.0e-9,
                max_its=5,
                strategy=worsening,
                monotone=False,
                track_best=track_best,
            )
        note = next((str(w.message) for w in caught if "going down" in str(w.message)), "")
        return model, note

    def test_best_seen_selection_says_best(self):
        model, note = self._descending_fit(True)
        self.assertIn("BEST iterate seen", note)
        self.assertAlmostEqual(float(model.mu), 3.0, delta=1.0)  # the good iterate came back
        self.assertIsNotNone(model.fit_provenance().last_accepted_objective)

    def test_last_iterate_selection_says_last_and_points_at_a_receipt_that_exists(self):
        model, note = self._descending_fit(False)
        self.assertIn("LAST iterate", note)
        self.assertNotIn("BEST iterate seen", note)
        self.assertGreater(float(model.mu), 100.0)  # the descending trajectory's endpoint
        # the note must not send the reader to a field that is None on this path
        self.assertIsNone(model.fit_provenance().last_accepted_objective)
        self.assertNotIn("last_accepted_objective", note)


class NetworkImpossibleRecordTest(unittest.TestCase):
    """R02-F09: a record the factors cannot score was -inf one way and NaN the other."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(2)
        n = 800
        a = rng.randint(0, 3, n)
        b = a * 2.0 + rng.normal(0, 0.5, n)
        c = np.where(rng.rand(n) < 0.9, a, rng.randint(0, 3, n))
        d = b + c + rng.normal(0, 0.3, n)
        cls.rows = [(int(a[i]), float(b[i]), int(c[i]), float(d[i])) for i in range(n)]
        cls.net = _quiet(lambda: learn_bayesian_network(cls.rows))

    def _both_routes(self, row):
        scalar = float(self.net.log_density(row))
        vector = float(self.net.seq_log_density(self.net.dist_to_encoder().seq_encode([row]))[0])
        return scalar, vector

    def test_an_unscoreable_field_is_impossible_on_both_routes(self):
        for label, bad in (("nan", float("nan")), ("inf", float("inf")), ("-inf", float("-inf"))):
            with self.subTest(field=label):
                scalar, vector = self._both_routes((0, bad, 0, 0.0))
                self.assertEqual(scalar, float("-inf"))
                self.assertEqual(vector, float("-inf"))

    def test_an_ordinary_record_is_unchanged_and_the_routes_agree(self):
        scalar, vector = self._both_routes(self.rows[0])
        self.assertTrue(np.isfinite(scalar))
        self.assertAlmostEqual(scalar, vector, places=9)

    def test_one_bad_row_does_not_poison_the_batch(self):
        batch = [(0, float("nan"), 0, 0.0), *self.rows[:5]]
        scores = self.net.seq_log_density(self.net.dist_to_encoder().seq_encode(batch))
        self.assertEqual(int(np.sum(np.isfinite(scores))), 5)
        self.assertEqual(scores[0], float("-inf"))


class RestoredPredictiveTest(unittest.TestCase):
    """R02-F04: predict() on a restored posterior answered with the plug-in predictive."""

    ROWS = list(np.random.RandomState(0).normal(5, 2, 3))  # few rows, so the parameter posterior is wide

    def _fitted(self, how):
        from mixle.ppl import Normal

        mu = Normal(0, 10, name="mu")
        kw = {"draws": 600, "burn": 200} if how in ("mcmc", "hmc") else {}
        return _quiet(lambda: Normal(mu, 2.0).fit(self.ROWS, how=how, rng=np.random.RandomState(0), **kw))

    def test_a_live_posterior_integrates_over_its_draws(self):
        live = self._fitted("mcmc")
        spread = float(np.asarray(live.predict(8000, rng=np.random.RandomState(1))).std())
        self.assertGreater(spread, 2.05)  # wider than the plug-in sqrt(4) = 2.0

    def test_a_restored_posterior_refuses_rather_than_dropping_the_uncertainty(self):
        for how in ("mcmc", "laplace"):
            with self.subTest(how=how):
                restored = pickle.loads(pickle.dumps(self._fitted(how)))
                with self.assertRaises(ValueError) as caught:
                    restored.predict(100, rng=np.random.RandomState(1))
                message = str(caught.exception)
                self.assertIn("restored from a pickle", message)
                self.assertIn("plug-in", message)

    def test_everything_the_changelog_says_survives_the_round_trip_still_does(self):
        restored = pickle.loads(pickle.dumps(self._fitted("mcmc")))
        summary = restored.summary()["mu"]
        self.assertGreater(summary["std"], 0.0)
        self.assertGreater(summary["ess_bulk"], 1.0)
        self.assertIsInstance(restored.explain_fit(), dict)


class ZeroAcceptanceEverySamplerTest(unittest.TestCase):
    """R02-F02: the refusal was wired into hmc only, on a guard documenting every sampler."""

    ROWS = list(np.random.RandomState(0).normal(5, 2, 400))

    @staticmethod
    def _model():
        from mixle.ppl import Normal, free

        return Normal(Normal(0, 10, name="mu"), free)

    def test_no_route_returns_a_point_as_a_posterior(self):
        for how, kw in (
            ("mcmc", {"draws": 4, "burn": 0}),
            ("nuts", {"draws": 8, "burn": 0}),
            ("hmc", {"draws": 8, "burn": 0}),
            ("ensemble", {"draws": 8, "burn": 0}),
        ):
            for seed in range(6):
                with self.subTest(how=how, seed=seed):
                    try:
                        fitted = _quiet(
                            lambda how=how, seed=seed, kw=kw: self._model().fit(
                                self.ROWS, how=how, rng=np.random.RandomState(seed), **kw
                            )
                        )
                    except RuntimeError as exc:
                        self.assertIn("accepted none of its proposals", str(exc))
                        continue
                    except ValueError:
                        continue  # a route this install cannot run (no gradient backend)
                    self.assertGreater(fitted.summary()["mu"]["std"], 0.0)

    def test_the_advice_names_the_route_it_is_talking_to(self):
        from mixle.ppl import potential

        wall = potential(lambda m: 0.0 if abs(m - float(np.mean(self.ROWS))) < 1e-9 else float("-inf"), None)
        del wall  # the constructed wall is not needed; the tiny-budget path exercises the same guard
        with self.assertRaises(RuntimeError) as caught:
            _quiet(lambda: self._model().fit(self.ROWS, how="mcmc", draws=4, burn=0, rng=np.random.RandomState(0)))
        message = str(caught.exception)
        self.assertIn("mcmc accepted none", message)
        self.assertNotIn("step_size", message)  # HMC's knob, not a random walk's


if __name__ == "__main__":
    unittest.main()
