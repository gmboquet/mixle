"""The inference loop: receipts that exist and are true, inputs that are read once, errors that name.

Pass 03 of the ten adversarial reviews of the 0.8.1 candidate (P03-F01..F13) found the flagship
``optimize(data)`` return carrying no receipt at all, a one-shot iterator fitted as zero
observations under a ``converged=True`` receipt, an empty encoding fabricating a model out of the
parameter floors, a receipt naming an objective the run did not maximize, a convergence tolerance
that could never be met, a cap warning whose advice inverted the situation, and several arguments
whose wrong type surfaced as somebody else's ``AttributeError``.
"""

from __future__ import annotations

import math
import random
import unittest
import warnings

import numpy as np

import mixle.stats as S
from mixle import Model, propose
from mixle.inference import learn_bayesian_network, optimize, pit_values, seq_estimate, seq_initialize
from mixle.stats import seq_encode
from mixle.stats.compute.sequence import seq_estimate as _seq_estimate_core
from mixle.utils.optional_deps import HAS_PANDAS


def _quiet(callable_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return callable_()


def _tabular_rows(count: int = 600):
    rng = np.random.RandomState(0)
    plan = rng.choice(["free", "pro"], count)
    usage = np.where(plan == "pro", rng.normal(60.0, 5.0, count), rng.normal(10.0, 3.0, count))
    spend = usage * 2.0 + rng.normal(0.0, 1.0, count)
    return [(str(p), float(u), float(s)) for p, u, s in zip(plan, usage, spend)]


class AutoStructureReceiptTest(unittest.TestCase):
    """P03-F01: the default automatic-structure return carried no receipt at all."""

    def test_every_entry_point_returns_a_model_that_says_how_it_was_fitted(self):
        rows = _tabular_rows()
        for label, produce in (
            ("optimize", lambda: optimize(rows)),
            ("fit", lambda: __import__("mixle.inference", fromlist=["fit"]).fit(rows)),
            ("learn_bayesian_network", lambda: learn_bayesian_network(rows)),
        ):
            with self.subTest(entry=label):
                model = _quiet(produce)
                provenance = model.fit_provenance()
                self.assertIsNotNone(provenance)
                self.assertIn("bayesian-network-structure-search", provenance.algorithm)
                self.assertEqual(provenance.n_observations, len(rows))
                self.assertFalse(provenance.converged)  # a greedy search reaches no certified optimum
                self.assertIsInstance(model.numerical_repairs(), tuple)

    def test_model_fit_reads_the_receipt_it_documents(self):
        model = _quiet(lambda: Model().fit(_tabular_rows()))
        self.assertLessEqual({"n", "n_iter", "converged"}, set(model._fit_info))


class OneShotIteratorTest(unittest.TestCase):
    """P03-F02, P06-F01: an iterator was consumed by estimator inference and fitted as zero rows."""

    DATA = [float(value) for value in np.random.RandomState(0).normal(3.0, 2.0, 200)]

    def test_a_generator_fits_the_observations_it_yields(self):
        from mixle.inference import fit

        for label, produce in (
            ("generator", lambda: optimize(value for value in self.DATA)),
            ("iter(list)", lambda: optimize(iter(self.DATA))),
            ("map", lambda: optimize(map(float, self.DATA))),
            ("fit(generator)", lambda: fit(value for value in self.DATA)),
        ):
            with self.subTest(input=label):
                model = _quiet(produce)
                provenance = model.fit_provenance()
                self.assertEqual(provenance.n_observations, len(self.DATA))
                self.assertTrue(math.isfinite(model.log_density(3.0)))

    def test_an_explicit_estimator_still_agrees(self):
        model = _quiet(lambda: optimize((value for value in self.DATA), S.GaussianEstimator()))
        self.assertEqual(model.fit_provenance().n_observations, len(self.DATA))

    def test_an_empty_generator_is_still_refused(self):
        with self.assertRaises(ValueError) as caught:
            optimize(value for value in [])
        self.assertIn("no observations", str(caught.exception))

    def test_propose_reaches_the_same_recommendation_from_an_iterator(self):
        from_list = _quiet(lambda: propose(list(self.DATA)))
        from_iterator = _quiet(lambda: propose(iter(self.DATA)))
        self.assertEqual(from_iterator.notes, from_list.notes)
        self.assertTrue(any("gaussian" in note for note in from_iterator.notes))


class EmptyEncodedBatchTest(unittest.TestCase):
    """P03-F03: an empty encoding produced a model made entirely of the variance floor."""

    @staticmethod
    def _empty_encoding():
        return seq_encode([], S.GaussianEstimator().accumulator_factory().make().acc_to_encoder())

    def test_optimize_refuses_a_zero_row_encoding(self):
        for encoded in (self._empty_encoding(), []):
            with self.subTest(encoding=type(encoded).__name__):
                with self.assertRaises(ValueError) as caught:
                    optimize(None, S.GaussianEstimator(), enc_data=encoded)
                self.assertIn("zero rows", str(caught.exception))

    def test_the_low_level_primitives_stay_total_but_say_so(self):
        # They must not crash -- three separate repairs (D-0203/D-0204, T4-02) exist to keep them
        # total on an empty corpus -- but the default model they return is disclosed as one.
        encoded = self._empty_encoding()
        for label, produce in (
            ("seq_initialize", lambda: seq_initialize(encoded, S.GaussianEstimator(), np.random.RandomState(0), 0.1)),
            ("seq_estimate", lambda: seq_estimate(encoded, S.GaussianEstimator(), S.GaussianDistribution(0.0, 1.0))),
        ):
            with self.subTest(entry=label):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = produce()
                self.assertIsInstance(model, S.GaussianDistribution)
                self.assertTrue(any("zero rows" in str(item.message) for item in caught))

    def test_a_non_empty_encoding_still_fits(self):
        data = [float(value) for value in np.random.RandomState(0).normal(size=50)]
        encoded = seq_encode(data, S.GaussianEstimator().accumulator_factory().make().acc_to_encoder())
        model = _quiet(lambda: optimize(None, S.GaussianEstimator(), enc_data=encoded))
        self.assertAlmostEqual(model.mu, float(np.mean(data)), places=6)


class ResolvedObjectiveTest(unittest.TestCase):
    """P03-F04: a prior-free container reported 'map' and lost the fused E-step."""

    @staticmethod
    def _rows(count: int = 500):
        rng = np.random.RandomState(0)
        return [(float(a), int(b)) for a, b in zip(rng.normal(0.0, 1.0, count), rng.poisson(3.0, count))]

    def test_a_prior_free_composite_is_maximum_likelihood(self):
        rows = self._rows()
        model = _quiet(lambda: optimize(rows, S.CompositeEstimator([S.GaussianEstimator(), S.PoissonEstimator()])))
        provenance = model.fit_provenance()
        self.assertEqual(provenance.objective, "mle")
        self.assertEqual(provenance.algorithm, "fused-em")
        self.assertAlmostEqual(provenance.final_objective, float(sum(model.log_density(row) for row in rows)), places=6)

    def test_optional_and_sequence_containers_agree(self):
        optional = _quiet(lambda: optimize([1.0, None, 2.0] * 20, S.OptionalEstimator(S.GaussianEstimator())))
        self.assertEqual(optional.fit_provenance().objective, "mle")
        sequence = _quiet(
            lambda: optimize(
                [[1.0, 2.0], [3.0]] * 30,
                S.SequenceEstimator(S.GaussianEstimator(), len_estimator=S.PoissonEstimator()),
            )
        )
        self.assertEqual(sequence.fit_provenance().objective, "mle")

    def test_a_real_prior_still_resolves_to_map(self):
        prior = S.GaussianEstimator(prior=S.NormalGammaDistribution(0.0, 1.0, 2.0, 3.0))
        data = [float(value) for value in np.random.RandomState(0).normal(0.0, 1.0, 200)]
        self.assertEqual(_quiet(lambda: optimize(data, prior)).fit_provenance().objective, "map")
        nested = S.CompositeEstimator([prior, S.PoissonEstimator()])
        self.assertEqual(_quiet(lambda: optimize(self._rows(200), nested)).fit_provenance().objective, "map")


class ConvergenceControlTest(unittest.TestCase):
    """P03-F05, F06: an unsatisfiable tolerance, and advice that inverted a downhill trajectory."""

    DATA = [float(value) for value in np.random.RandomState(0).normal(3.0, 2.0, 200)]

    def test_delta_zero_is_refused_because_it_can_never_be_met(self):
        with self.assertRaises(ValueError) as caught:
            optimize(self.DATA, S.GaussianEstimator(), max_its=5, delta=0)
        self.assertIn("finite positive number", str(caught.exception))

    def test_delta_none_and_a_positive_delta_still_work(self):
        self.assertIsNone(
            _quiet(lambda: optimize(self.DATA, S.GaussianEstimator(), max_its=3, delta=None)).fit_provenance().delta
        )
        self.assertTrue(
            _quiet(lambda: optimize(self.DATA, S.GaussianEstimator(), max_its=30, delta=1e-9))
            .fit_provenance()
            .converged
        )

    def test_a_downhill_best_seen_run_is_not_told_to_raise_max_its(self):
        calls = {"n": 0}

        def worsening(encoded, estimator, model):
            calls["n"] += 1
            step = _seq_estimate_core(encoded, estimator, model)
            if calls["n"] >= 2:
                return S.GaussianDistribution(step.mu + 50.0 * calls["n"], step.sigma2)
            return step

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(
                self.DATA, S.GaussianEstimator(), delta=1e-9, max_its=5, strategy=worsening, monotone=False
            )
        message = next(str(item.message) for item in caught if "max_its cap" in str(item.message))
        self.assertIn("objective going down", message)
        self.assertIn("BEST iterate", message)
        self.assertNotIn("Raise max_its", message)
        self.assertLess(model.fit_provenance().last_accepted_objective, model.fit_provenance().final_objective)

    def test_an_ordinary_capped_run_keeps_its_original_advice(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimize(self.DATA, S.GaussianEstimator(), delta=1e-30, max_its=2)
        message = next(str(item.message) for item in caught if "max_its cap" in str(item.message))
        self.assertIn("Raise max_its", message)


class FieldNamingTest(unittest.TestCase):
    """P03-F08: the multimodality note printed the raw internal path tuple."""

    def test_multimodal_fields_use_the_same_spelling_as_every_other_note(self):
        rng = np.random.RandomState(0)
        bimodal = np.concatenate([rng.normal(-5.0, 1.0, 150), rng.normal(5.0, 1.0, 150)])
        cases = [
            ("scalar", list(bimodal), "$"),
            ("dict", [{"height": float(h), "uni": float(u)} for h, u in zip(bimodal, rng.normal(0, 1, 300))], "$["),
        ]
        if HAS_PANDAS:  # the frame route is one of three; the other two are checked on a base install
            import pandas as pd

            cases.append(("frame", pd.DataFrame({"height": bimodal, "count": rng.poisson(2, 300)}), "$[0]"))
        for label, data, expected in cases:
            with self.subTest(rows=label):
                notes = [note for note in _quiet(lambda d=data: propose(d)).notes if "multimodal" in note]
                self.assertTrue(notes, label)
                self.assertIn("field(s) %s" % expected, notes[0])
                self.assertNotIn("field(s) ()", notes[0])
                self.assertNotIn("field(s) (0,)", notes[0])


class ArgumentTypeTest(unittest.TestCase):
    """P03-F09: wrong-type arguments surfaced as somebody else's AttributeError."""

    DATA = [float(value) for value in np.random.RandomState(0).normal(3.0, 2.0, 100)]

    def test_rng_names_itself_and_the_spellings_it_accepts(self):
        for label, value in (("str", "abc"), ("random.Random", random.Random(1)), ("bool", True)):
            with self.subTest(rng=label):
                with self.assertRaises(TypeError) as caught:
                    optimize(self.DATA, S.GaussianEstimator(), rng=value)
                self.assertIn("rng must be", str(caught.exception))

    def test_the_accepted_rng_spellings_still_work(self):
        for value in (7, np.random.RandomState(1), np.random.default_rng(2)):
            with self.subTest(rng=type(value).__name__):
                self.assertTrue(
                    math.isfinite(
                        _quiet(lambda v=value: optimize(self.DATA, S.GaussianEstimator(), rng=v, max_its=2)).mu
                    )
                )

    def test_prev_estimate_says_it_wants_a_fitted_model(self):
        with self.assertRaises(TypeError) as caught:
            optimize(self.DATA, S.GaussianEstimator(), prev_estimate=S.GaussianEstimator())
        self.assertIn("fitted distribution", str(caught.exception))

    def test_the_low_level_pipeline_takes_the_same_rng_spellings(self):
        encoded = seq_encode(self.DATA, S.GaussianEstimator().accumulator_factory().make().acc_to_encoder())
        with self.assertRaises(TypeError) as caught:
            seq_initialize(encoded, S.GaussianEstimator(), None, 0.1)
        self.assertIn("rng must be", str(caught.exception))
        for value in (3, np.random.default_rng(1), np.random.RandomState(4)):
            with self.subTest(rng=type(value).__name__):
                self.assertIsInstance(
                    seq_initialize(encoded, S.GaussianEstimator(), value, 0.1), S.GaussianDistribution
                )


class ProposeSplitTest(unittest.TestCase):
    """P03-F11: three records trained every candidate on one row and scored the floor."""

    def test_a_single_row_training_split_is_refused_by_name(self):
        # Three records at the default holdout leaves exactly one training row.
        data = [float(value) for value in np.random.RandomState(0).normal(size=3)]
        with self.assertRaises(ValueError) as caught:
            propose(data)
        self.assertIn("variance-floored point mass", str(caught.exception))

    def test_enough_records_still_propose(self):
        for count in (4, 20):
            with self.subTest(records=count):
                data = [float(value) for value in np.random.RandomState(0).normal(size=count)]
                self.assertTrue(any("held-out" in note for note in _quiet(lambda d=data: propose(d)).notes))


class ScalarOnlyCdfTest(unittest.TestCase):
    """P03-F12: a callable that swallowed the array and returned a scalar was a 'shape error'."""

    Y = np.random.RandomState(0).normal(0.0, 1.0, 50)

    def test_a_scalar_returning_callable_is_evaluated_element_wise(self):
        np.testing.assert_allclose(pit_values(self.Y, lambda value: 0.5), np.full(50, 0.5))

    def test_a_non_finite_result_is_named_as_such(self):
        with self.assertRaises(ValueError) as caught:
            pit_values(self.Y, lambda value: float("nan"))
        self.assertIn("finite", str(caught.exception))
        self.assertNotIn("shape", str(caught.exception))

    def test_a_real_cdf_and_a_vectorized_one_are_unchanged(self):
        law = S.GaussianDistribution(0.0, 1.0)
        np.testing.assert_allclose(pit_values(self.Y, law.cdf), [law.cdf(v) for v in self.Y])
        np.testing.assert_allclose(pit_values(self.Y, lambda v: np.full_like(np.asarray(v, float), 0.5)), 0.5)


class LikelihoodFactorMessageTest(unittest.TestCase):
    """P03-F13: 'likelihood factors found' named the mechanism but not the fix."""

    @staticmethod
    def _sequences(count: int = 300):
        rng = np.random.RandomState(0)
        return [list(rng.normal(0.0, 1.0, rng.randint(3, 9))) for _ in range(count)]

    def test_the_message_names_the_family_and_len_estimator(self):
        with self.assertRaises(TypeError) as caught:
            optimize(self._sequences(), S.MixtureEstimator([S.SequenceEstimator(S.GaussianEstimator())] * 2), max_its=3)
        message = str(caught.exception)
        self.assertIn("SequenceDistribution", message)
        self.assertIn("len_estimator", message)
        self.assertIn("length distribution", message)

    def test_the_named_fix_works(self):
        component = S.SequenceEstimator(S.GaussianEstimator(), len_estimator=S.PoissonEstimator())
        model = _quiet(
            lambda: optimize(
                self._sequences(), S.MixtureEstimator([component] * 2), max_its=3, rng=np.random.RandomState(1)
            )
        )
        self.assertEqual(len(model.components), 2)


class CapNoteAttributionTest(unittest.TestCase):
    """P09-F09: a note the caller cannot act on is noise, and one naming a library line is worse.

    ``optimize``'s cap notes hard-coded ``stacklevel=3``, which lands on the caller only for a DIRECT
    ``optimize(...)``. Every forwarded route -- ``fit()``, ``learn_structure()``, the task verbs --
    attributed the note to a line inside the library and advised knobs belonging to a call the reader
    never wrote; one example script printed 44 such lines.
    """

    ROWS = [float(value) for value in np.random.RandomState(0).normal(size=400)]

    @staticmethod
    def _cap_notes(callable_):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            callable_()
        return [entry for entry in caught if "max_its cap" in str(entry.message)]

    def _capped_mixture(self):
        component = S.GaussianEstimator()
        return S.MixtureEstimator([component] * 3)

    def test_a_direct_optimize_still_names_its_own_call_line(self):
        notes = self._cap_notes(
            lambda: optimize(self.ROWS, self._capped_mixture(), max_its=2, rng=np.random.RandomState(0))
        )
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].filename, __file__)

    def test_a_fit_forwarded_note_names_the_fit_call_not_the_forwarding_line(self):
        from mixle.inference.estimation import fit

        notes = self._cap_notes(
            lambda: fit(self.ROWS, self._capped_mixture(), max_its=2, out=None, rng=np.random.RandomState(0))
        )
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].filename, __file__)

    def test_a_structure_search_does_not_narrate_the_models_it_only_scores(self):
        """``dependency_gain`` subtracts two log-likelihoods; neither model is ever handed back."""
        from mixle.inference.structure import dependency_gain

        rng = np.random.RandomState(0)
        parent = [("a" if value > 0 else "b") for value in rng.normal(size=300)]
        child = [rng.normal(2.0 if key == "a" else -2.0, 1.0) for key in parent]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gain = dependency_gain(parent, child, self._capped_mixture(), max_its=2)
        self.assertTrue(math.isfinite(gain))
        optimize_notes = [entry for entry in caught if "optimize() stopped" in str(entry.message)]
        self.assertEqual(optimize_notes, [])
        own = [entry for entry in caught if "dependency_gain scored" in str(entry.message)]
        self.assertEqual(len(own), 1)
        self.assertEqual(own[0].filename, __file__)

    def test_a_structure_search_replays_every_other_warning_the_fits_raise(self):
        from mixle.inference.structure import dependency_gain

        class Loud(S.GaussianEstimator):
            def accumulator_factory(self):
                warnings.warn("a fit-time note the caller should still see", UserWarning, stacklevel=2)
                return super().accumulator_factory()

        rng = np.random.RandomState(0)
        parent = [("a" if value > 0 else "b") for value in rng.normal(size=120)]
        child = [rng.normal(0.0, 1.0) for _ in parent]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dependency_gain(parent, child, Loud(), max_its=2)
        self.assertTrue(any("a fit-time note" in str(entry.message) for entry in caught))


if __name__ == "__main__":
    unittest.main()
