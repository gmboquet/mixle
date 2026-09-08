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
from mixle.utils.optional_deps import HAS_PANDAS


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

    def test_a_hard_constraint_is_where_the_collapse_actually_happens(self):
        """R07-F01: the first predicate was the CAUSE, and the worst case reports a healthy rate.

        An ensemble behind a hard constraint projects every walker onto one feasible point; the
        stretch move then returns the current state and is ACCEPTED, so the rate reads 0.89 while
        all 8000 draws are one number -- reported as a posterior with a mean of 30.10 where the
        truncated posterior sits at 20.04. The original test only ever ran the unconstrained target,
        where nothing collapses.
        """
        from mixle.ppl import Normal, free

        for how in ("ensemble", "hmc"):
            with self.subTest(how=how):
                mu = Normal(0, 10, name="mu")
                with self.assertRaises(RuntimeError) as caught:
                    _quiet(
                        lambda mu=mu, how=how: Normal(mu, free).fit(
                            self.ROWS,
                            how=how,
                            draws=600,
                            burn=150,
                            constraints=[mu > 5.30, mu < 5.35],
                            rng=np.random.RandomState(0),
                        )
                    )
                self.assertIn("a point, not a posterior", str(caught.exception))

    def test_the_route_the_refusal_recommends_samples_the_truncated_posterior(self):
        """The refusal must not swallow the route the message itself recommends.

        Which, under a hard constraint, is `mcmc` alone. `nuts` is a gradient route, not one of the
        alternatives. And `ensemble` -- named by the unconstrained message -- cannot do this model:
        the stretch move `X_j + z(X_i - X_j)` cannot leave a point every walker already shares, and
        projecting walkers into a 0.05-wide feasible band is how they come to share one. So the
        recommendation drops it when constraints are active rather than sending the caller from one
        refusal to another.
        """
        from mixle.ppl import Normal, free

        mu = Normal(0, 10, name="mu")
        fitted = _quiet(
            lambda: Normal(mu, free).fit(
                self.ROWS,
                how="mcmc",
                draws=600,
                burn=150,
                constraints=[mu > 5.30, mu < 5.35],
                rng=np.random.RandomState(0),
            )
        )
        row = fitted.summary()["mu"]
        self.assertGreater(row["std"], 0.0)
        self.assertGreater(row["mean"], 5.29)  # inside the band it was constrained to

    def test_the_ensemble_route_on_this_model_is_refused_not_silently_wrong(self):
        """Why the constrained recommendation names one route: the other one collapses, loudly."""
        from mixle.ppl import Normal, free

        mu = Normal(0, 10, name="mu")
        with self.assertRaises(RuntimeError) as caught:
            _quiet(
                lambda: Normal(mu, free).fit(
                    self.ROWS,
                    how="ensemble",
                    draws=600,
                    burn=150,
                    constraints=[mu > 5.30, mu < 5.35],
                    rng=np.random.RandomState(0),
                )
            )
        message = str(caught.exception)
        self.assertIn("every retained draw is the same point", message)
        # And it says why the acceptance rate did not give it away: a stretch move that returns the
        # current state is accepted.
        self.assertIn("accepted without moving", message)

    def test_the_gradient_refusal_drops_the_route_that_cannot_serve_a_constrained_model(self):
        from mixle.ppl.inference import _refuse_a_gradient_route_without_a_gradient

        with self.assertRaises(ValueError) as free_form:
            _refuse_a_gradient_route_without_a_gradient(None, how="nuts")
        self.assertIn("how='mcmc' or how='ensemble'", str(free_form.exception))
        self.assertIn("which are gradient-free", str(free_form.exception))
        with self.assertRaises(ValueError) as constrained:
            _refuse_a_gradient_route_without_a_gradient(None, how="nuts", constrained=True)
        self.assertIn("how='mcmc', which is gradient-free", str(constrained.exception))
        self.assertNotIn("ensemble", str(constrained.exception))
        # A gradient that exists is not this refusal's business, constrained or not.
        self.assertIsNone(_refuse_a_gradient_route_without_a_gradient(lambda u: u, how="nuts", constrained=True))

    def test_a_constant_chain_is_refused_even_with_a_healthy_acceptance_rate(self):
        """The predicate reads the draws, so it does not depend on how the rate was computed."""
        from mixle.ppl.inference import _every_draw_is_one_point

        class Chain:
            def __init__(self, samples, rate):
                self.samples = samples
                self.acceptance_rate = rate

        self.assertTrue(_every_draw_is_one_point([Chain(np.full((500, 2), 3.0), 0.9)]))
        self.assertFalse(_every_draw_is_one_point([Chain(np.random.RandomState(0).normal(size=(500, 2)), 0.0)]))
        self.assertFalse(_every_draw_is_one_point([Chain(np.zeros((0, 2)), 0.0)]))  # no draws is a different failure

    def test_the_advice_names_the_route_it_is_talking_to(self):
        from mixle.ppl import potential

        wall = potential(lambda m: 0.0 if abs(m - float(np.mean(self.ROWS))) < 1e-9 else float("-inf"), None)
        del wall  # the constructed wall is not needed; the tiny-budget path exercises the same guard
        with self.assertRaises(RuntimeError) as caught:
            _quiet(lambda: self._model().fit(self.ROWS, how="mcmc", draws=4, burn=0, rng=np.random.RandomState(0)))
        message = str(caught.exception)
        self.assertIn("mcmc accepted none", message)
        self.assertNotIn("step_size", message)  # HMC's knob, not a random walk's


class UnfittedHandleReprTest(unittest.TestCase):
    """R02-F06: printing a model is how a reader inspects it, and it raised AttributeError."""

    def test_a_vector_free_handle_and_the_models_holding_it_print(self):
        from mixle.ppl import Categorical, DiagGaussian, Normal, free

        handle = free(5, name="v")
        self.assertIn("free(5", repr(handle))
        self.assertIn("name='v'", repr(handle))
        self.assertIn("free(4", repr(free(4)))  # unnamed
        for model in (
            DiagGaussian(5, mean=handle, var=np.full(5, 0.25)),
            Normal(free(1, name="m1"), 1.0),
            Categorical(free(3, name="p")),
        ):
            with self.subTest(model=type(model).__name__):
                text = repr(model)
                self.assertTrue(text.startswith("RV("))
                self.assertIn("free(", text)

    def test_a_fitted_model_prints_as_it_always_did(self):
        from mixle.ppl import DiagGaussian, free

        fitted = _quiet(
            lambda: DiagGaussian(5, mean=free(5, name="v"), var=np.full(5, 0.25)).fit([[0.0] * 5] * 10, how="map")
        )
        self.assertIn("RV(bound=", repr(fitted))


@unittest.skipUnless(HAS_PANDAS, "pandas not installed; pip install mixle[pandas]")
class NetworkFromAFrameTest(unittest.TestCase):
    """R02-F07: a DataFrame iterates as its column NAMES, and the search fitted those."""

    @staticmethod
    def _frame(rows):
        import pandas as pd

        rng = np.random.RandomState(0)
        return pd.DataFrame({"a": rng.randint(0, 3, rows).astype(float), "b": rng.normal(0, 1, rows)})

    def test_an_empty_frame_is_named_as_an_empty_corpus(self):
        import pandas as pd

        with self.assertRaises(ValueError) as caught:
            learn_bayesian_network(pd.DataFrame({"a": [], "b": []}))
        self.assertIn("received no records", str(caught.exception))

    def test_a_frame_is_fitted_as_its_rows_and_matches_the_same_table_as_records(self):
        frame = self._frame(300)
        from_frame = _quiet(lambda: learn_bayesian_network(frame))
        rows = [tuple(row) for row in frame.itertuples(index=False, name=None)]
        from_rows = _quiet(lambda: learn_bayesian_network(rows))
        self.assertEqual(from_frame.fit_provenance().n_observations, 300)
        self.assertEqual(
            [factor.parents for factor in from_frame.factors],
            [factor.parents for factor in from_rows.factors],
        )


@unittest.skipUnless(HAS_PANDAS, "pandas not installed; pip install mixle[pandas]")
class ColumnMappingFrontDoorTest(unittest.TestCase):
    """R06-F01: the mapping route indexed columns by LABEL, so a filtered frame raised KeyError."""

    @staticmethod
    def _columns():
        import pandas as pd

        rng = np.random.RandomState(0)
        frame = pd.DataFrame({"x": rng.normal(0, 1, 300), "k": rng.poisson(3, 300)})
        return frame[frame["x"] > 0.0]  # an ordinary row filter: the surviving index is 1, 4, 7, ...

    def test_every_index_shape_a_pandas_column_can_carry(self):
        import pandas as pd

        rng = np.random.RandomState(1)
        filtered = self._columns()
        cases = {
            "filtered": {"x": filtered["x"], "k": filtered["k"]},
            "string index": {
                "x": pd.Series(rng.normal(size=60), index=["r%d" % i for i in range(60)]),
                "k": pd.Series(rng.poisson(3, 60), index=["r%d" % i for i in range(60)]),
            },
            "duplicated index": {
                "x": pd.Series(rng.normal(size=40), index=[i % 8 for i in range(40)]),
                "k": pd.Series(rng.poisson(3, 40), index=[i % 8 for i in range(40)]),
            },
            "plain lists": {"x": list(rng.normal(size=60)), "k": list(rng.poisson(3, 60))},
        }
        for label, columns in cases.items():
            with self.subTest(columns=label):
                self.assertIsNotNone(_quiet(lambda columns=columns: optimize(columns, max_its=2)))

    def test_the_mapping_and_frame_spellings_of_one_table_fit_the_same_model(self):
        filtered = self._columns()
        mapping = _quiet(
            lambda: optimize({"x": filtered["x"], "k": filtered["k"]}, max_its=3, rng=np.random.RandomState(0))
        )
        frame = _quiet(lambda: optimize(filtered, max_its=3, rng=np.random.RandomState(0)))
        self.assertEqual(str(mapping), str(frame))

    def test_a_column_with_no_row_order_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            optimize({"x": set(range(50)), "k": set(range(50, 100))}, max_its=2)
        message = str(caught.exception)
        self.assertIn("no row order", message)
        self.assertIn("'x'", message)


class ClosedParallelHandleTest(unittest.TestCase):
    """R06-F02: P06-F04's repair shipped with no test, and missed one of the five entry points."""

    @staticmethod
    def _handle_and_estimator():
        import mixle.stats as S
        from mixle.utils.parallel.multiprocessing import MPEncodedData

        rows = [float(v) for v in np.random.RandomState(0).normal(3, 2, 200)]
        estimator = S.GaussianEstimator()
        return MPEncodedData(rows, estimator=estimator, num_workers=2), estimator, rows

    def test_an_open_handle_serves_every_entry_point(self):
        from mixle.inference import seq_estimate, seq_initialize
        from mixle.stats import GaussianDistribution

        handle, estimator, _rows = self._handle_and_estimator()
        try:
            started = seq_initialize(handle, estimator, np.random.RandomState(0), 0.5)
            self.assertGreater(float(started.sigma2), 0.1)  # a real fit, not the variance floor
            fitted = seq_estimate(handle, estimator, GaussianDistribution(0.0, 1.0))
            self.assertAlmostEqual(float(fitted.mu), 3.0, delta=0.5)
        finally:
            handle.close()

    def test_a_closed_handle_refuses_every_entry_point_rather_than_answering(self):
        from mixle.inference import optimize as fit_verb
        from mixle.inference import seq_estimate, seq_initialize
        from mixle.stats import GaussianDistribution

        handle, estimator, _rows = self._handle_and_estimator()
        handle.close()
        routes = {
            "seq_initialize": lambda: seq_initialize(handle, estimator, np.random.RandomState(0), 0.5),
            "seq_estimate": lambda: seq_estimate(handle, estimator, GaussianDistribution(0.0, 1.0)),
            "optimize": lambda: fit_verb(None, estimator, enc_data=handle, max_its=2),
        }
        for name, route in routes.items():
            with self.subTest(route=name):
                with self.assertRaises(RuntimeError) as caught:
                    route()
                self.assertIn("closed", str(caught.exception))

    def test_closing_twice_is_still_the_documented_shutdown(self):
        handle, _estimator, _rows = self._handle_and_estimator()
        handle.close()
        handle.close()  # idempotent: close() is the shutdown, not a request


if __name__ == "__main__":
    unittest.main()


class LifecycleFrontDoorTest(unittest.TestCase):
    """R06-F03: one table, one answer, whichever verb reads it.

    ``_reusable_observations`` -- the normalization and the by-name refusals that replaced a set of
    raw failures during the 0.8.1 campaign -- was wired into ``optimize``/``fit``/``best_of`` only.
    ``Model().fit`` and ``propose`` read their data through ``lifecycle._tabular_records``, which
    handled the three tabular spellings and then fell through to a bare ``list(data)``, so the same
    inputs still reached the failures the front door exists to prevent.
    """

    HOSTILE = {
        "numpy.matrix": (lambda: np.matrix(np.random.RandomState(0).rand(40, 2)), "numpy.matrix"),
        "timedelta64": (lambda: np.arange(40).astype("timedelta64[s]"), "timedelta64"),
        "masked": (lambda: np.ma.masked_array(np.arange(40.0), mask=[0] * 39 + [1]), "masked array"),
        "bare str": (lambda: "hello world hello", "iterates as its individual characters"),
        "0-d array": (lambda: np.array(3.0), "0-dimensional array"),
        "ragged mapping": (lambda: {"x": [1.0, 2.0], "k": [0, 1, 0]}, "lengths differ"),
        "set column": (lambda: {"x": {1.0, 2.0, 3.0}, "k": [0, 1, 0]}, "no row order"),
    }

    def _verbs(self):
        from mixle import Model, propose

        return (("Model.fit()", lambda d: Model().fit(d)), ("propose()", lambda d: propose(d)))

    def test_every_hostile_input_is_refused_by_name_and_names_the_verb(self):
        for name, (build, fragment) in self.HOSTILE.items():
            for entry, call in self._verbs():
                with self.subTest(data=name, verb=entry):
                    with self.assertRaises(ValueError) as caught:
                        _quiet(lambda call=call, build=build: call(build()))
                    message = str(caught.exception)
                    self.assertIn(fragment, message)
                    # The message has to say which verb was called: these are the same refusals the
                    # fit verbs raise, and "optimize() received a str" from a `Model().fit` call
                    # sends the reader to the wrong doorway.
                    self.assertIn(entry, message)

    def test_a_single_observation_still_gets_the_lifecycle_message(self):
        """The fallthrough's own error survives delegation -- it is about the VERB, not the data."""
        for entry, call in self._verbs():
            with self.subTest(verb=entry):
                with self.assertRaises(ValueError) as caught:
                    call(0.5)
                self.assertIn("collection of observation records", str(caught.exception))

    def test_the_normalizing_half_lands_too(self):
        """Refusal is only half the front door; the other half fits what it silently mishandled."""
        from mixle import Model

        rows = [float(v) for v in range(50)]
        baseline = type(_quiet(lambda: Model().fit(rows)).fitted).__name__
        # A one-shot iterator: `list(data)` consumed it during structure inference and encoded zero
        # rows, so the fit ran on nothing at all.
        streamed = _quiet(lambda: Model().fit(float(v) for v in range(50)))
        self.assertEqual(type(streamed.fitted).__name__, baseline)
        # A structured array yields void scalars, which the profiler could not hash.
        table = np.array([(1.0, 2), (3.0, 4)] * 20, dtype=[("x", "f8"), ("k", "i8")])
        self.assertEqual(len(_quiet(lambda: Model().fit(table)).fitted.order), 2)
        # A mapping of columns iterates as its field NAMES.
        mapped = _quiet(lambda: Model().fit({"x": [1.0, 2.0, 3.0] * 10, "k": [0, 1, 0] * 10}))
        self.assertEqual(len(mapped.fitted.dists), 2)


class LookbackSmoothedPosteriorTest(unittest.TestCase):
    """R05-F04: ``seq_posterior`` is the smoothing marginal on every model that defines it.

    The lookback model ran the forward-only kernel unconditionally and returned the FILTERED
    probabilities under the smoothed name -- while the plain HMM's docstring stated the smoothed
    contract "matching every other seq_posterior in the package". Checked against brute-force
    enumeration of all 2**T state paths, which is the definition of both quantities.
    """

    EMISSIONS = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]
    START = [0.6, 0.4]
    TRANSITIONS = [[0.8, 0.2], [0.3, 0.7]]
    SEQUENCE = [0, 2, 2, 1]

    def _model(self, **kw):
        from mixle.stats import CategoricalDistribution, IntegerCategoricalDistribution, SequenceDistribution
        from mixle.stats.latent.lookback_hidden_markov_model import LookbackHiddenMarkovModelDistribution

        topics = [
            SequenceDistribution(IntegerCategoricalDistribution(0, p), len_dist=CategoricalDistribution({1: 1.0}))
            for p in self.EMISSIONS
        ]
        return LookbackHiddenMarkovModelDistribution(topics, w=self.START, transitions=self.TRANSITIONS, lag=0, **kw)

    def _by_enumeration(self):
        """(smoothed, filtered) marginals, summed over every state path."""
        import itertools

        w = np.asarray(self.START)
        a = np.asarray(self.TRANSITIONS)
        b = np.asarray(self.EMISSIONS)
        seq = self.SEQUENCE
        joint = np.zeros((len(seq), 2))
        evidence = 0.0
        for path in itertools.product(range(2), repeat=len(seq)):
            mass = w[path[0]] * b[path[0], seq[0]]
            for t in range(1, len(seq)):
                mass *= a[path[t - 1], path[t]] * b[path[t], seq[t]]
            evidence += mass
            for t, state in enumerate(path):
                joint[t, state] += mass
        smoothed = joint / evidence

        filtered = np.zeros((len(seq), 2))
        step = w * b[:, seq[0]]
        filtered[0] = step / step.sum()
        for t in range(1, len(seq)):
            step = (filtered[t - 1] @ a) * b[:, seq[t]]
            filtered[t] = step / step.sum()
        return smoothed, filtered

    def test_the_default_is_the_smoothing_marginal(self):
        model = self._model()
        encoded = model.dist_to_encoder().seq_encode([self.SEQUENCE])
        smoothed, filtered = self._by_enumeration()
        np.testing.assert_allclose(model.seq_posterior(encoded)[0], smoothed, atol=1e-12)
        # The two quantities agree at the LAST position and nowhere else, which is why returning one
        # under the other's name reads as a plausible answer.
        self.assertGreater(np.abs(smoothed - filtered).max(), 0.1)
        np.testing.assert_allclose(smoothed[-1], filtered[-1], atol=1e-12)

    def test_the_filtered_probabilities_are_still_reachable(self):
        model = self._model()
        encoded = model.dist_to_encoder().seq_encode([self.SEQUENCE])
        _, filtered = self._by_enumeration()
        np.testing.assert_allclose(model.seq_posterior(encoded, filtered=True)[0], filtered, atol=1e-12)

    def test_a_terminal_state_restriction_is_refused_rather_than_ignored(self):
        model = self._model(terminal_states=[1])
        encoded = model.dist_to_encoder().seq_encode([self.SEQUENCE])
        with self.assertRaises(NotImplementedError) as caught:
            model.seq_posterior(encoded)
        self.assertIn("terminal_states", str(caught.exception))
        # The refusal has to name the route that DOES run the restricted recursion.
        self.assertIn("HiddenMarkovModelDistribution.seq_posterior", str(caught.exception))


class MonitorThresholdConstructionTest(unittest.TestCase):
    """R06-F05: a threshold that cannot work is refused where it was written."""

    @staticmethod
    def _fixture():
        from mixle.inference.production.monitor import Monitor

        rows = list(np.random.RandomState(0).normal(0.0, 1.0, 200))
        model = _quiet(lambda: optimize(rows, GaussianEstimator(), max_its=5, out=None))
        return Monitor, model, rows

    def test_a_threshold_that_can_never_pass_is_refused_at_construction(self):
        Monitor, model, rows = self._fixture()
        for kw, fragment in (
            ({"loglik_shift_threshold": 1.0}, "may FALL"),
            ({"psi_threshold": -1.0}, "non-negative"),
            ({"ks_threshold": 2.0}, "in [0, 1]"),
        ):
            with self.subTest(**kw):
                with self.assertRaises(ValueError) as caught:
                    Monitor(model, GaussianEstimator(), rows, **kw)
                self.assertIn(fragment, str(caught.exception))
        with self.assertRaises(TypeError):
            Monitor(model, GaussianEstimator(), rows, ks_threshold="0.5")

    def test_a_workable_threshold_still_constructs_and_checks(self):
        Monitor, model, rows = self._fixture()
        monitor = Monitor(model, GaussianEstimator(), rows, loglik_shift_threshold=-0.5, psi_threshold=0.3)
        self.assertFalse(monitor.check(rows).drift)
        # A negative shift threshold is the meaningful direction and must stay accepted.
        self.assertEqual(monitor.thresholds["loglik_shift_threshold"], -0.5)

    def test_the_monitor_and_the_detector_share_one_rule(self):
        """Not two copies that can drift apart -- the weaker copy is how this got through."""
        from mixle.inference.production import drift as drift_module
        from mixle.inference.production import monitor as monitor_module

        self.assertIs(monitor_module.validate_drift_thresholds, drift_module.validate_drift_thresholds)


class PlackettLuceRowRefusalTest(unittest.TestCase):
    """R06-F07: a row that is not an ordering is refused by name, with its index."""

    ROWS = [[0, 1, 2], [2, 0, 1], [1, 2, 0]] * 5

    def _estimator(self):
        from mixle.stats import PlackettLuceEstimator

        return PlackettLuceEstimator(3)

    def test_a_row_that_is_not_an_ordering_names_the_family_and_the_row(self):
        for label, index, value in (("None", 3, None), ("a scalar", 2, 7), ("a string", 4, "abc")):
            with self.subTest(row=label):
                rows = list(self.ROWS)
                rows[index] = value
                with self.assertRaises(ValueError) as caught:
                    _quiet(lambda rows=rows: optimize(rows, self._estimator(), max_its=2, out=None))
                message = str(caught.exception)
                self.assertIn("Plackett-Luce", message)
                self.assertIn("row %d" % index, message)
                # Not `TypeError: 'NoneType' object is not iterable`, which named neither.
                self.assertNotIn("not iterable", message)

    def test_the_partial_encoder_refuses_the_same_way(self):
        from mixle.stats import PlackettLucePartialDataEncoder

        encoder = PlackettLucePartialDataEncoder(dim=3)
        self.assertEqual(len(encoder.seq_encode([[0, 1], [2, 0], [1, 2, 0]])), 3)
        with self.assertRaises(ValueError) as caught:
            encoder.seq_encode([[0, 1], None])
        self.assertIn("row 1", str(caught.exception))

    def test_well_formed_rankings_still_fit(self):
        fitted = _quiet(lambda: optimize(self.ROWS, self._estimator(), max_its=5, out=None))
        self.assertEqual(len(fitted.log_w), 3)


class UnencodablePromptSeedTest(unittest.TestCase):
    """R07-F03: every prompt without a canonical encoding shared ONE seed.

    ``_derive_seed`` interpolated ``_seed_key``'s ``None`` straight into the digest input, so the
    key was the literal ``"<base_seed>:None"`` -- while both docstrings promised a ``repr`` fallback
    and the warning described the opposite hazard (equal prompts seeding *differently*).
    """

    class Opaque:
        """Unencodable: defining __repr__ proves nothing about what it contains, so _seed_key declines."""

        def __init__(self, tag):
            self.tag = tag

        def __repr__(self):
            return "Opaque(%r)" % self.tag

    def _derive(self, prompt, base=7):
        from mixle.task.calibrated_generator import _derive_seed

        return _quiet(lambda: _derive_seed(base, prompt))

    def test_different_unencodable_prompts_get_different_seeds(self):
        from mixle.task.calibrated_generator import _seed_key

        prompts = [self.Opaque(tag) for tag in ("alpha", "beta", "gamma")]
        for prompt in prompts:
            self.assertIsNone(_seed_key(prompt))  # the precondition: this IS the fallback path
        self.assertEqual(len({self._derive(p) for p in prompts}), len(prompts))

    def test_the_fallback_is_the_documented_one_and_is_stable(self):
        """`repr`, as both docstrings say -- so two prompts with the same repr still agree."""
        import hashlib

        self.assertEqual(self._derive(self.Opaque("alpha")), self._derive(self.Opaque("alpha")))
        expected = hashlib.sha256(b"7:Opaque('alpha')").digest()
        self.assertEqual(self._derive(self.Opaque("alpha")), int.from_bytes(expected[:8], "big") % (2**32))

    def test_it_still_warns_that_the_promise_does_not_cover_this(self):
        from mixle.task.calibrated_generator import _derive_seed

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _derive_seed(7, self.Opaque("alpha"))
        self.assertTrue([e for e in caught if "not reproducible across processes" in str(e.message)])

    def test_canonical_prompts_are_untouched(self):
        """The repair must not move a seed that was already derived from a real encoding."""
        self.assertEqual(self._derive(5), self._derive(5.0))  # 1 == 1.0, so one prompt
        self.assertNotEqual(self._derive(True), self._derive(1))  # ...but a bool is its own kind
        self.assertEqual(self._derive({"a": 1, "b": 2}), self._derive({"b": 2, "a": 1}))
        # And `None` as a prompt is encodable, so it must not collide with the old shared value the
        # `"<base>:None"` key produced.
        self.assertNotEqual(self._derive(None), self._derive(self.Opaque("alpha")))
