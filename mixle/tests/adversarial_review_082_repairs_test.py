"""What the 0.8.2 adversarial review found wrong with the 0.8.2 repairs themselves.

Four independent reviewers attacked the repairs this release makes on top of the published 0.8.1.
The defects below are theirs (identifiers ``R<pass>-F<nn>``), and every one is a repair that did not
do what its own CHANGELOG entry said: a guard wired into one route of four, a disclosure that named
the wrong selection, a condition reported differently by two routes to the same model, and a
restored posterior answering a question it could no longer answer.
"""

from __future__ import annotations

import importlib.util
import pickle
import unittest
import warnings

import numpy as np

from mixle.inference import learn_bayesian_network, optimize
from mixle.stats import GaussianDistribution, GaussianEstimator
from mixle.stats.compute.sequence import seq_estimate
from mixle.utils.optional_deps import HAS_PANDAS

HAS_TORCH = importlib.util.find_spec("torch") is not None


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


class WeibullInfiniteObservationTest(unittest.TestCase):
    """R05-F06: the two scoring routes agree on an observation past the float range.

    ``log_density`` refuses it as impossible (``-inf``, the P01-F04 overflow rule); the vectorized
    route computed ``+inf + -inf`` and returned NaN, which is not "impossible" but "unknown" -- and
    one NaN turns the sum or mean of a whole batch into NaN. Same shape as R02-F09.
    """

    XS = [float("inf"), 1e300, 5.0, 1.0, 0.1, 0.0, -1.0]

    def _routes(self, shape):
        from mixle.stats import WeibullDistribution

        dist = WeibullDistribution(shape=shape, scale=1.0)
        vector = np.asarray(dist.seq_log_density(dist.dist_to_encoder().seq_encode(self.XS)), dtype=float)
        scalar = np.array([float(dist.log_density(x)) for x in self.XS])
        return scalar, vector

    def test_the_two_routes_agree_at_every_shape_regime(self):
        # shape < 1, == 1 and > 1 take three different branches at x == 0, so all three are checked.
        for shape in (0.5, 1.0, 2.0):
            with self.subTest(shape=shape):
                scalar, vector = self._routes(shape)
                self.assertFalse(np.isnan(vector).any(), "vectorized route produced NaN")
                np.testing.assert_allclose(vector, scalar, rtol=1e-12)

    def test_an_infinite_observation_is_impossible_not_unknown(self):
        for shape in (0.5, 1.0, 2.0):
            with self.subTest(shape=shape):
                scalar, vector = self._routes(shape)
                self.assertEqual(vector[0], -np.inf)
                self.assertEqual(scalar[0], -np.inf)

    def test_a_batch_holding_one_is_still_summable(self):
        """The consequence the NaN had: one bad row poisoned every other row's total."""
        from mixle.stats import WeibullDistribution

        dist = WeibullDistribution(shape=2.0, scale=1.0)
        rows = [1.0, 2.0, float("inf"), 0.5]
        scores = np.asarray(dist.seq_log_density(dist.dist_to_encoder().seq_encode(rows)), dtype=float)
        self.assertEqual(float(np.sum(scores)), -np.inf)  # not NaN
        self.assertTrue(np.isfinite(scores[[0, 1, 3]]).all())


class QuantileDomainTest(unittest.TestCase):
    """R05-F07: every family that defines ``quantile`` refuses an out-of-domain ``q`` the same way.

    Four discrete families raised; eight continuous ones let scipy's ``ppf`` answer NaN; and
    ``BernoulliDistribution`` answered with a plausible support point -- ``0.0`` at ``q = -0.5``,
    ``1.0`` at ``q = 1.5`` and at NaN -- which is the only one of the three that is a wrong answer
    rather than a missing one.
    """

    @staticmethod
    def _families():
        import inspect

        from mixle import stats

        candidates = (
            {"shape": 2.0, "scale": 1.0},
            {"k": 2.0, "theta": 1.0},
            {"beta": 1.0},
            {"a": 2.0, "b": 2.0},
            {"mu": 0.0, "sigma2": 1.0},
            {"lam": 1.5},
            {"p": 0.3},
            {"alpha": 2.0, "beta": 2.0},
            {"n": 5, "p": 0.3},
        )
        built = []
        for name in sorted(dir(stats)):
            if not name.endswith("Distribution"):
                continue
            cls = getattr(stats, name, None)
            if not inspect.isclass(cls) or not hasattr(cls, "quantile"):
                continue
            for kwargs in candidates:
                try:
                    built.append((name, cls(**kwargs)))
                    break
                except Exception:  # noqa: BLE001 - this family takes different parameters; try the next shape
                    continue
        return built

    def test_the_survey_reaches_every_family_it_claims_to(self):
        names = [name for name, _ in self._families()]
        self.assertGreaterEqual(len(names), 13)
        self.assertIn("BernoulliDistribution", names)

    def test_an_out_of_domain_q_raises_on_every_family(self):
        for name, dist in self._families():
            for q in (-0.5, 1.5, float("nan"), -np.inf, np.inf):
                with self.subTest(family=name, q=q):
                    with self.assertRaises(ValueError) as caught:
                        dist.quantile(q)
                    self.assertIn("q must be in [0, 1]", str(caught.exception))
                    self.assertIn(name, str(caught.exception))

    def test_the_whole_closed_interval_is_still_answered(self):
        """The refusal must not eat the endpoints, which are the support's own bounds."""
        for name, dist in self._families():
            with self.subTest(family=name):
                for q in (0.0, 0.25, 1.0, 0, 1, np.float64(0.5)):
                    self.assertFalse(np.isnan(float(dist.quantile(q))), "q=%r returned NaN" % (q,))


class ZeroWeightExemptionTest(unittest.TestCase):
    """R05-F05: a zero-weight row must contribute zero, not NaN.

    ``refuse_unsupported_observations`` deliberately exempts a row whose weight is zero -- that
    exemption is what lets a mixture encode one batch against every component (P02-F03). The
    arithmetic did not honour it: the encoders admit out-of-support rows, so the encoded statistic
    holds an infinity there, and ``inf * 0.0`` is NaN. One exempt row turned the running sufficient
    statistic into NaN for every fully-weighted observation in the same chunk, and the fit died
    several frames later on the parameter validator ("requires beta > 0").
    """

    @staticmethod
    def _families():
        from mixle import stats

        return [
            ("Exponential", stats.ExponentialDistribution(beta=1.0), stats.ExponentialEstimator(), [1.0, 2.0, 3.0]),
            ("Gamma", stats.GammaDistribution(k=2.0, theta=1.0), stats.GammaEstimator(), [1.0, 2.0, 3.0]),
            ("Beta", stats.BetaDistribution(a=2.0, b=2.0), stats.BetaEstimator(), [0.2, 0.5, 0.7]),
            ("Weibull", stats.WeibullDistribution(shape=2.0, scale=1.0), stats.WeibullEstimator(), [1.0, 2.0, 3.0]),
            (
                "LogGaussian",
                stats.LogGaussianDistribution(mu=0.0, sigma2=1.0),
                stats.LogGaussianEstimator(),
                [1.0, 2.0, 3.0],
            ),
            (
                "InverseGamma",
                stats.InverseGammaDistribution(alpha=2.0, beta=1.0),
                stats.InverseGammaEstimator(),
                [1.0, 2.0, 3.0],
            ),
            (
                "InverseGaussian",
                stats.InverseGaussianDistribution(mu=1.0, lam=1.0),
                stats.InverseGaussianEstimator(),
                [1.0, 2.0, 3.0],
            ),
            ("HalfNormal", stats.HalfNormalDistribution(sigma=1.0), stats.HalfNormalEstimator(), [1.0, 2.0, 3.0]),
            ("Rayleigh", stats.RayleighDistribution(sigma=1.0), stats.RayleighEstimator(), [1.0, 2.0, 3.0]),
            ("Uniform", stats.UniformDistribution(0.0, 5.0), stats.UniformEstimator(), [1.0, 2.0, 3.0]),
        ]

    @staticmethod
    def _statistics(dist, estimator, rows, weights):
        accumulator = estimator.accumulator_factory().make()
        accumulator.seq_update(dist.dist_to_encoder().seq_encode(rows), np.asarray(weights, dtype=float), None)
        return np.asarray(accumulator.value(), dtype=float)

    def test_an_exempt_row_leaves_the_statistics_exactly_as_if_it_were_absent(self):
        for name, dist, estimator, rows in self._families():
            clean = self._statistics(dist, estimator, rows, [1.0] * len(rows))
            for offending in (float("inf"), float("-inf"), -1.0):
                with self.subTest(family=name, row=offending):
                    got = self._statistics(dist, estimator, [*rows, offending], [*([1.0] * len(rows)), 0.0])
                    self.assertFalse(np.isnan(got).any(), "exempt row produced NaN statistics")
                    np.testing.assert_allclose(got, clean, rtol=1e-12)

    def test_a_weighted_violation_is_still_refused(self):
        """The repair must not turn the exemption into a licence: weight it and it still raises."""
        for name, dist, estimator, rows in self._families():
            with self.subTest(family=name):
                with self.assertRaises(ValueError):
                    self._statistics(dist, estimator, [*rows, float("inf")], [*([1.0] * len(rows)), 1.0])

    def test_the_helper_only_masks_what_the_weight_already_excluded(self):
        from mixle.stats.univariate.continuous._observation_contracts import weighted_statistic_sum

        values = np.array([1.0, 2.0, np.inf])
        self.assertEqual(weighted_statistic_sum(values, np.array([1.0, 1.0, 0.0])), 3.0)
        # A weighted infinity is not masked -- that is a real infinity in the statistic, and the
        # guard above is what refuses it.
        self.assertEqual(weighted_statistic_sum(values, np.array([1.0, 1.0, 1.0])), np.inf)
        # And an ordinary batch is untouched.
        self.assertEqual(weighted_statistic_sum(np.array([1.0, 2.0]), np.array([0.5, 0.5])), 1.5)


class NonFiniteObjectiveDiagnosisTest(unittest.TestCase):
    """R05-F09: name the cause when it is one line away.

    A mixture every one of whose components refuses every observation reported "fused EM did not
    produce a finite objective from its non-finite initial model" -- a message about EM internals
    for what is a data/model mismatch the caller can see and fix.
    """

    def _refusal(self, rows, estimator):
        with self.assertRaises(ValueError) as caught:
            _quiet(lambda: optimize(rows, estimator, max_its=4, out=None, rng=np.random.RandomState(0)))
        return str(caught.exception)

    def test_it_says_that_nothing_is_in_support_and_how_many(self):
        from mixle.stats import BetaEstimator, ExponentialEstimator, MixtureEstimator

        for label, rows, estimator in (
            ("negatives into exponentials", [-1.0, -2.0, -3.0] * 10, MixtureEstimator([ExponentialEstimator()] * 2)),
            ("out-of-unit into betas", [1.5, 2.5, 3.5] * 10, MixtureEstimator([BetaEstimator()] * 2)),
        ):
            with self.subTest(case=label):
                message = self._refusal(rows, estimator)
                self.assertIn("did not produce a finite objective", message)  # the symptom survives
                self.assertIn("30 observation(s) scores -inf", message)  # ...and now names the cause
                self.assertIn("support of any component", message)

    def test_a_healthy_fit_is_untouched(self):
        from mixle.stats import ExponentialEstimator, MixtureEstimator

        rows = list(np.random.RandomState(0).exponential(1.0, 60))
        fitted = _quiet(
            lambda: optimize(
                rows, MixtureEstimator([ExponentialEstimator()] * 2), max_its=4, out=None, rng=np.random.RandomState(0)
            )
        )
        self.assertEqual(len(fitted.components), 2)

    def test_the_diagnosis_does_not_replace_a_more_specific_refusal(self):
        """A single-family fit still refuses at the accumulator, naming the family and the rows."""
        from mixle.stats import ExponentialEstimator

        message = self._refusal([-1.0, -2.0, -3.0] * 10, ExponentialEstimator())
        self.assertIn("ExponentialDistribution has support x >= 0", message)
        self.assertNotIn("did not produce a finite objective", message)


class CapNoteAdviceTest(unittest.TestCase):
    """R07-F06: a note may only advise a knob the verb the caller invoked actually takes.

    ``_caller_stacklevel`` already put the note on the reader's own line, but its TEXT stayed
    written for a direct ``optimize(...)``: routed through a structure verb it still opened with
    "optimize()" and still advised ``delta=None``, which those verbs do not accept.
    """

    CAP_NOTE_MARKERS = ("max_its cap", "delta=None (documented", "rejected update")

    def _cap_notes(self, call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        return [
            str(entry.message)
            for entry in caught
            if any(marker in str(entry.message) for marker in self.CAP_NOTE_MARKERS)
        ]

    def _verbs(self):
        import numpy as np

        from mixle.inference import learn_bayesian_network, optimize
        from mixle.inference.structure import learn_mixture_structure, learn_structure
        from mixle.stats import GaussianEstimator, MixtureEstimator

        pairs = [
            (float(a), float(b))
            for a, b in zip(
                np.random.RandomState(0).normal(0, 1, 200), np.random.RandomState(1).normal(0, 1, 200), strict=True
            )
        ]
        flat = list(np.random.RandomState(0).normal(0.0, 1.0, 300)) + [40.0]
        return [
            (
                "optimize",
                optimize,
                lambda: optimize(
                    flat,
                    MixtureEstimator([GaussianEstimator(), GaussianEstimator()]),
                    max_its=1,
                    out=None,
                    rng=np.random.RandomState(0),
                ),
            ),
            ("learn_structure", learn_structure, lambda: learn_structure(pairs, max_its=1)),
            ("learn_bayesian_network", learn_bayesian_network, lambda: learn_bayesian_network(pairs, max_its=1)),
            (
                "learn_mixture_structure",
                learn_mixture_structure,
                lambda: learn_mixture_structure(pairs, 2, restarts=1, seed=0, max_its=1),
            ),
        ]

    def test_no_note_advises_a_knob_its_verb_does_not_accept(self):
        import inspect

        for name, function, call in self._verbs():
            accepted = set(inspect.signature(function).parameters)
            for note in self._cap_notes(call):
                for knob in ("delta=None", "restarts=", "print_iter=", "track_best="):
                    with self.subTest(verb=name, knob=knob):
                        if knob.split("=")[0] not in accepted:
                            self.assertNotIn(knob, note)

    def test_the_direct_optimize_note_is_unchanged(self):
        """The whole point of the fallback: a direct call still reads exactly as it always did."""
        notes = self._cap_notes(self._verbs()[0][2])
        capped = [note for note in notes if "max_its cap" in note]
        self.assertTrue(capped)
        self.assertTrue(capped[0].startswith("optimize() stopped at the max_its cap"))
        self.assertIn("Raise max_its to fit to convergence, or pass delta=None", capped[0])

    def test_a_forwarded_note_names_the_verb_the_caller_called(self):
        import numpy as np

        from mixle.inference import learn_bayesian_network

        pairs = [
            (float(a), float(b))
            for a, b in zip(
                np.random.RandomState(0).normal(0, 1, 200), np.random.RandomState(1).normal(0, 1, 200), strict=True
            )
        ]
        capped = [n for n in self._cap_notes(lambda: learn_bayesian_network(pairs, max_its=1)) if "max_its cap" in n]
        self.assertTrue(capped)
        self.assertTrue(capped[0].startswith("learn_bayesian_network() stopped"))
        self.assertNotIn("delta=None", capped[0])
        # max_its IS accepted there, so that half of the advice survives.
        self.assertIn("Raise max_its to fit to convergence.", capped[0])


class UnidentifiedComponentRemedyTest(unittest.TestCase):
    """R07-F07: the A-02 note's remedy is addressed to whoever can act on it.

    ``restarts=`` and ``init='dirichlet'`` are knobs on the ESTIMATOR. The README's own
    ``solve(teacher, inputs)`` one-liner fits a four-component mixture internally, so it raised this
    note four to seven times, each advising two knobs the reader had no object to pass them to.
    """

    def _notes(self, call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        return [str(e.message) for e in caught if "less data than they have parameters" in str(e.message)]

    def test_a_caller_who_built_the_estimator_is_told_which_knobs_to_turn(self):
        from mixle.stats import MixtureEstimator

        rows = list(np.random.RandomState(0).normal(0.0, 1.0, 400)) + [40.0]
        notes = self._notes(
            lambda: optimize(
                rows,
                MixtureEstimator([GaussianEstimator(), GaussianEstimator()]),
                max_its=60,
                out=None,
                rng=np.random.RandomState(0),
            )
        )
        self.assertTrue(notes)
        self.assertIn("restarts= or MixtureEstimator(..., init='dirichlet')", notes[0])
        self.assertNotIn("built this mixture itself", notes[0])

    def test_a_caller_who_did_not_is_told_that_instead(self):
        from mixle.task import solve

        words = ["alpha beta", "gamma delta", "beta gamma", "delta alpha"]
        inputs = [words[index % len(words)] for index in range(160)]

        def teacher(text):
            return "first" if text.startswith(("alpha", "beta")) else "second"

        notes = self._notes(lambda: solve(teacher, inputs, seed=0, student="generative"))
        self.assertTrue(notes, "expected solve()'s internal mixture to raise the note")
        self.assertIn("solve() built this mixture itself", notes[0])
        # The reader is pointed at something they can actually do, and not at a bare `restarts=`
        # they have no object to pass.
        self.assertIn("more data, or fitting the mixture directly", notes[0])
        self.assertNotIn("restarts=", notes[0])

    def test_the_observation_half_is_identical_either_way(self):
        """Only the remedy is conditional; what the fit did is reported the same to both callers."""
        from mixle.stats import MixtureEstimator

        rows = list(np.random.RandomState(0).normal(0.0, 1.0, 400)) + [40.0]
        notes = self._notes(
            lambda: optimize(
                rows,
                MixtureEstimator([GaussianEstimator(), GaussianEstimator()]),
                max_its=60,
                out=None,
                rng=np.random.RandomState(0),
            )
        )
        self.assertIn("component_row_mass", notes[0])
        self.assertIn("EM converges to such a solution rather than failing at it", notes[0])


@unittest.skipUnless(HAS_TORCH, "the default Bayesian-optimization surrogate is the torch GP")
class SurrogateRepairDisclosureTest(unittest.TestCase):
    """R07-F05: a repaired surrogate is disclosed on a route the DOE caller can reach.

    The GP records ridging its own covariance under ``numerical_repairs()``, and on a routine
    ``minimize`` run it does so: measured on this objective, 11 of 12 seeds need escalated jitter.
    ``propose_next`` fits the surrogate internally and returns only the point, so that record died
    with the local ``gp`` and ``BayesOptResult`` had no route to it.
    """

    BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]

    @staticmethod
    def _objective(point):
        return float((point[0] - 1.0) ** 2 + (point[1] + 2.0) ** 2)

    def _run(self, seed, **kwargs):
        from mixle.doe import minimize

        settings = {"n_init": 5, "n_iter": 15, "seed": seed}
        settings.update(kwargs)
        return _quiet(lambda: minimize(self._objective, self.BOUNDS, **settings))

    def test_the_repair_the_surrogate_applied_reaches_the_result(self):
        disclosed = [seed for seed in range(12) if self._run(seed).numerical_repairs()]
        # The escalation is routine on this objective, not exotic; the point of the test is that
        # whenever it happens the caller can see it.
        self.assertGreaterEqual(len(disclosed), 8)
        repairs = self._run(disclosed[0]).numerical_repairs()
        self.assertTrue(any("covariance-ridged" in entry for entry in repairs))

    def test_a_run_that_needed_no_repair_reports_none(self):
        """An empty tuple, not a missing attribute -- the channel is always there to be read."""
        from mixle.doe import minimize

        result = _quiet(lambda: minimize(lambda p: float(abs(p[0])), [(-1.0, 1.0)], n_init=3, n_iter=3, seed=0))
        self.assertEqual(result.numerical_repairs(), ())
        self.assertEqual(result.surrogate_repairs, ())

    def test_the_channel_is_spelled_the_way_the_rest_of_the_library_spells_it(self):
        from mixle.doe.bayesopt import BayesOptResult

        result = self._run(0)
        self.assertIsInstance(result, BayesOptResult)
        self.assertEqual(result.numerical_repairs(), result.surrogate_repairs)
        # Duplicates are collapsed: the same ridging reported by every iteration's fit is one fact.
        self.assertEqual(len(set(result.numerical_repairs())), len(result.numerical_repairs()))


class TerminalStateRoutesTest(unittest.TestCase):
    """R05-F01 follow-on: every posterior route honours ``terminal_states``.

    ``seq_posterior`` was repaired for the terminal-state restriction; ``latent_posterior`` (and so
    ``.marginals()``, ``.mode()``, ``.sample()`` and ``posterior_predictive``, which all read the
    object it returns) and ``viterbi`` still ran the unrestricted recursion. Checked against
    enumeration of the admissible paths, which is the definition of both quantities.
    """

    START = [0.6, 0.4]
    TRANSITIONS = [[0.8, 0.2], [0.3, 0.7]]
    TERMINAL = [1]
    SEQUENCES = ([-1.8, 0.1, 2.2, -1.9], [-1.0, 2.0, 2.5, 2.4], [2.0, 2.0, 2.0])

    def _components(self):
        from mixle.stats import GaussianDistribution

        return [GaussianDistribution(mu=-2.0, sigma2=1.0), GaussianDistribution(mu=2.0, sigma2=1.0)]

    def _model(self, terminal):
        from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

        extra = {"terminal_states": self.TERMINAL} if terminal else {}
        return HiddenMarkovModelDistribution(self._components(), w=self.START, transitions=self.TRANSITIONS, **extra)

    def _by_enumeration(self, sequence, terminal):
        """(smoothed marginals, most probable path) over the paths the model actually admits."""
        import itertools

        components = self._components()
        start = np.asarray(self.START)
        transitions = np.asarray(self.TRANSITIONS)
        joint = np.zeros((len(sequence), 2))
        evidence = 0.0
        best_path, best_mass = None, -1.0
        for path in itertools.product(range(2), repeat=len(sequence)):
            # Only the LAST position may be terminal.
            if terminal and any(state in self.TERMINAL for state in path[:-1]):
                continue
            mass = start[path[0]] * np.exp(components[path[0]].log_density(sequence[0]))
            for t in range(1, len(sequence)):
                mass *= transitions[path[t - 1], path[t]] * np.exp(components[path[t]].log_density(sequence[t]))
            evidence += mass
            if mass > best_mass:
                best_mass, best_path = mass, list(path)
            for t, state in enumerate(path):
                joint[t, state] += mass
        return joint / evidence, best_path

    def test_every_route_matches_enumeration_with_and_without_the_restriction(self):
        for terminal in (True, False):
            for sequence in self.SEQUENCES:
                with self.subTest(terminal_states=terminal, sequence=tuple(sequence)):
                    model = self._model(terminal)
                    marginals, best_path = self._by_enumeration(sequence, terminal)
                    posterior = model.latent_posterior(sequence)
                    np.testing.assert_allclose(np.asarray(posterior.marginals()), marginals, atol=1e-12)
                    self.assertEqual(list(posterior.mode()), best_path)
                    self.assertEqual(list(model.viterbi(sequence)), best_path)

    def test_the_restriction_actually_changes_the_answer_here(self):
        """Otherwise the test above would pass on a model that ignores terminal_states."""
        sequence = self.SEQUENCES[0]
        restricted, restricted_path = self._by_enumeration(sequence, True)
        unrestricted, unrestricted_path = self._by_enumeration(sequence, False)
        self.assertGreater(np.abs(restricted - unrestricted).max(), 0.5)
        self.assertNotEqual(restricted_path, unrestricted_path)

    def test_a_sampled_path_never_visits_a_terminal_state_early(self):
        """`.sample()` and `posterior_predictive` read the same object, so the restriction reaches them."""
        model = self._model(True)
        sequence = self.SEQUENCES[0]
        posterior = model.latent_posterior(sequence)
        for seed in range(25):
            path = list(posterior.sample(np.random.RandomState(seed)))
            self.assertEqual(len(path), len(sequence))
            for state in path[:-1]:
                self.assertNotIn(state, self.TERMINAL)
        drawn = model.posterior_predictive(sequence, seed=0)
        self.assertEqual(len(drawn), len(sequence))


class SpellingConsistencyTest(unittest.TestCase):
    """Four pass-02 minors, all the same shape: one spelling of a thing works and its twin does not."""

    def test_an_encoded_batch_is_counted_whether_it_is_a_list_or_a_tuple(self):
        """R02-F10: the zero-row refusal never fired for a tuple of chunks."""
        from mixle.stats import GaussianDistribution, GaussianEstimator

        encoded = GaussianDistribution(mu=0.0, sigma2=1.0).dist_to_encoder().seq_encode([1.0, 2.0, 3.0])
        for empty in ([], ()):
            with self.subTest(spelling=type(empty).__name__):
                with self.assertRaises(ValueError) as caught:
                    optimize(None, GaussianEstimator(), enc_data=empty, max_its=2, out=None)
                self.assertIn("enc_data carries zero rows", str(caught.exception))
        for chunks in ([(3, encoded)], ((3, encoded),)):
            with self.subTest(spelling=type(chunks).__name__):
                fitted = _quiet(
                    lambda chunks=chunks: optimize(None, GaussianEstimator(), enc_data=chunks, max_its=3, out=None)
                )
                # Not just "it fits": the receipt used to say n_observations=None for the tuple.
                self.assertEqual(fitted.fit_provenance().n_observations, 3)

    def test_every_ppl_route_takes_the_same_rng_spellings(self):
        """R02-F12: laplace, vi and map kept a RandomState-only check the four samplers had dropped."""
        from mixle.ppl import Normal, free

        rows = list(np.random.RandomState(0).normal(5.0, 1.0, 60))
        spellings = (7, np.random.default_rng(7), np.random.RandomState(7), None)
        for how in ("laplace", "vi", "map", "mcmc", "ensemble"):
            for rng in spellings:
                with self.subTest(how=how, rng=type(rng).__name__):
                    _quiet(lambda how=how, rng=rng: Normal(free, free).fit(rows, how=how, rng=rng))
        # A bool is still refused everywhere -- it is not an integer seed.
        for how in ("laplace", "map", "mcmc"):
            with self.subTest(how=how, rng="bool"):
                with self.assertRaises(TypeError):
                    _quiet(lambda how=how: Normal(free, free).fit(rows, how=how, rng=True))

    def test_initialize_takes_what_seq_initialize_takes(self):
        """R02-F13: an int or Generator died on a bare AttributeError from inside the loop."""
        from mixle.stats import GaussianEstimator
        from mixle.stats.compute.sequence import initialize

        for rng in (7, np.random.RandomState(7), np.random.default_rng(7)):
            with self.subTest(rng=type(rng).__name__):
                initialize([1.0, 2.0, 3.0], GaussianEstimator(), rng=rng)
        with self.assertRaises(TypeError) as caught:
            initialize([1.0, 2.0, 3.0], GaussianEstimator(), rng="not an rng")
        self.assertIn("initialize()", str(caught.exception))  # names the route, not just the loop

    def test_the_gp_reports_the_largest_repair_not_the_last(self):
        """R02-F14: a later, smaller ridging overwrote a bigger earlier one."""
        from mixle.models.gaussian_process import GaussianProcessRegressor

        model = GaussianProcessRegressor.__new__(GaussianProcessRegressor)
        model._jitter_applied = 0.0
        model.jitter = 1.0e-12
        for applied in (1.0e-2, 1.0e-10):
            model._jitter_applied = max(model._jitter_applied, applied)
        repairs = GaussianProcessRegressor.numerical_repairs(model)
        self.assertTrue(repairs)
        self.assertIn("0.01", repairs[0])


class ProposeAdviceTest(unittest.TestCase):
    """R02-F11: the advised record count was itself refused, at every holdout that binds."""

    ROWS = list(np.random.RandomState(0).normal(0.0, 1.0, 40))

    def _advice(self, n, holdout):
        from mixle import propose

        with self.assertRaises(ValueError) as caught:
            _quiet(lambda: propose(self.ROWS[:n], holdout=holdout))
        return str(caught.exception)

    def test_the_advised_count_is_one_that_actually_fits(self):
        import re

        from mixle import propose

        for holdout in (0.8, 0.5, 0.25, 0.01):
            with self.subTest(holdout=holdout):
                advised = int(re.search(r"at least (\d+) records", self._advice(3, holdout)).group(1))
                _quiet(lambda n=advised, h=holdout: propose(self.ROWS[:n], holdout=h))  # must not raise
                # ...and it is the SMALLEST such count, not merely a safe one.
                with self.assertRaises(ValueError):
                    _quiet(lambda n=advised, h=holdout: propose(self.ROWS[: n - 1], holdout=h))

    def test_lower_holdout_is_offered_only_where_holdout_binds(self):
        """`n_val` is `max(2, round(n*holdout))`, so at a small holdout it is the floor that binds."""
        self.assertIn("lower holdout", self._advice(6, 0.8))
        self.assertNotIn("lower holdout", self._advice(3, 0.01))


class ScalarCdfAndRaggedColumnsTest(unittest.TestCase):
    """R02-F15 and R02-F16: two more raw failures on documented surfaces."""

    Y = np.array([0.1, 0.2, 0.3])

    def test_a_scalar_only_cdf_is_accepted_however_it_wraps_its_answer(self):
        """R02-F15: a length-1 list or array per call died on numpy's own TypeError."""
        from mixle.inference.calibration import pit_values

        for label, cdf in (
            ("a bare float", lambda v: 0.5),
            ("a length-1 list", lambda v: [0.5]),
            ("a length-1 array", lambda v: np.array([0.5])),
            ("a vectorized cdf", lambda v: np.full(np.shape(v), 0.5) if np.ndim(v) else 0.5),
        ):
            with self.subTest(cdf=label):
                np.testing.assert_allclose(np.asarray(pit_values(self.Y, cdf)), [0.5, 0.5, 0.5])

    def test_a_cdf_that_returns_the_wrong_count_is_still_refused(self):
        """The repair accepts a one-element container, not any container."""
        from mixle.inference.calibration import pit_values

        with self.assertRaises((TypeError, ValueError)):
            pit_values(self.Y, lambda v: [0.4, 0.6])

    def test_dependency_gain_refuses_ragged_columns(self):
        """R02-F16: `zip` truncated to the shorter column and scored the remainder as a gain."""
        from mixle.inference.structure import dependency_gain
        from mixle.stats import GaussianEstimator

        with self.assertRaises(ValueError) as caught:
            dependency_gain([1.0, 2.0, 3.0], [0.5, 0.6], GaussianEstimator())
        message = str(caught.exception)
        self.assertIn("3 parent(s) and 2 child value(s)", message)
        # Equal columns still score, unchanged.
        self.assertTrue(np.isfinite(dependency_gain([1.0, 2.0, 3.0], [0.5, 0.6, 0.7], GaussianEstimator())))


class PoissonQuantileTrustTest(unittest.TestCase):
    """The P01-F11 repair fired on NaN alone, and NaN is not the only way scipy gets this wrong.

    Found while running the suite against the minimum supported scipy: at ``lam=1e15`` scipy 1.16.0
    returns a FINITE ``999999999979213.0`` -- about 21000 counts below the median, CDF 0.4997 --
    where 1.17.1 returns NaN. Trusting a finite answer broke ``quantile``'s own contract on
    whichever scipy happened to be installed, and the shipped test asserted the NaN as a precondition
    so it failed on the scipy WITHOUT the defect.
    """

    LAMBDAS = (0.5, 4.0, 100.0, 1.0e6, 1.0e12, 1.0e15)
    LEVELS = (0.01, 0.25, 0.5, 0.9, 0.999)

    def test_the_contract_holds_at_every_rate_whatever_scipy_answers(self):
        from mixle.stats import PoissonDistribution

        for lam in self.LAMBDAS:
            law = PoissonDistribution(lam)
            for q in self.LEVELS:
                with self.subTest(lam=lam, q=q):
                    quantile = law.quantile(q)
                    self.assertTrue(np.isfinite(quantile))
                    # "the smallest k with P(X <= k) >= q" -- both halves.
                    self.assertGreaterEqual(law.cdf(quantile), q)
                    if quantile > 0.0:
                        self.assertLess(law.cdf(quantile - 1.0), q)

    def test_a_wrong_finite_answer_from_scipy_is_rejected_like_a_nan(self):
        """Directly: the validator, not the version, is what decides."""
        from mixle.stats import PoissonDistribution

        law = PoissonDistribution(1.0e15)
        self.assertFalse(law._quantile_holds(999999999979213.0, 0.5))  # the scipy 1.16.0 answer
        self.assertFalse(law._quantile_holds(float("nan"), 0.5))
        self.assertFalse(law._quantile_holds(-1.0, 0.5))
        self.assertTrue(law._quantile_holds(law._quantile_by_bisection(0.5), 0.5))

    def test_small_rates_still_agree_with_scipy_exactly(self):
        """The validator must not push ordinary rates onto the bisection path with a different answer."""
        import scipy.stats as ss

        from mixle.stats import PoissonDistribution

        for lam in (0.5, 4.0, 100.0):
            for q in self.LEVELS:
                with self.subTest(lam=lam, q=q):
                    self.assertEqual(PoissonDistribution(lam).quantile(q), float(ss.poisson.ppf(q, lam)))
