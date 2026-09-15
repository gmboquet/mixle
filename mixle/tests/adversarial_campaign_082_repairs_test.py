"""Repairs for the blocking defects the ten-pass adversarial campaign found on the 0.8.2 candidate.

Each test names the finding it pins and fails on the unrepaired tree. The campaign record and the
independent reproductions live in ``release-checklists/0.8.2-reviews/``.
"""

from __future__ import annotations

import unittest
import warnings
from pathlib import Path

import numpy as np

from mixle.inference import optimize
from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    PoissonDistribution,
    PoissonEstimator,
    TreeHiddenMarkovEstimator,
    TreeHiddenMarkovModelDistribution,
    dump_models,
    load_models,
)
from mixle.utils.optional_deps import HAS_NUMBA


class FittedTreeHmmSerializesTest(unittest.TestCase):
    """Q09-F01: a tree HMM fitted by ``optimize()`` could not be written as JSON at all.

    ``_p_level_cache`` memoizes ``init_prob @ transitions^k``. A constructor leaves it ``None`` and a
    fit warms it, and the decoder compares the artifact's state against a freshly constructed
    object's, so the warm tuple failed a comparison against ``None`` and the whole model was refused.
    The same fit round-tripped on 0.8.1, which makes this a regression the release line introduced.
    """

    def _fitted(self):
        source = TreeHiddenMarkovModelDistribution(
            [GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)],
            [0.5, 0.5],
            [[0.8, 0.2], [0.2, 0.8]],
            len_dist=PoissonDistribution(0.8),
            terminal_level=3,
        )
        data = source.sampler(1).sample(150)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return source, optimize(
                data,
                TreeHiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()], len_estimator=PoissonEstimator()),
                max_its=8,
                rng=np.random.RandomState(1),
                out=None,
            )

    def test_a_fitted_tree_hmm_round_trips_through_json(self):
        source, fitted = self._fitted()
        # The memo really is warm: without that this test would pass on the broken tree too.
        self.assertIsNotNone(fitted._p_level_cache, "the fit should have warmed the level memo")
        self.assertIsNone(source._p_level_cache)
        for label, model in (("constructed", source), ("fitted", fitted)):
            with self.subTest(model=label):
                restored = load_models(dump_models(model))
                self.assertIsNotNone(restored)

    def test_the_memo_is_not_carried_in_the_artifact_and_is_rebuilt_on_use(self):
        """Two models with the same parameters are the same model, warm cache or not."""
        _, fitted = self._fitted()
        text = dump_models(fitted)
        self.assertNotIn("_p_level_cache", text)
        restored = load_models(text)
        model = restored["model"] if isinstance(restored, dict) and "model" in restored else restored
        target = model if hasattr(model, "_get_p_level") else fitted
        self.assertIsNone(getattr(target, "_p_level_cache", None) if target is not fitted else None)
        # Recomputed on demand, and equal to what the warm original holds.
        rebuilt = target._get_p_level(3)
        np.testing.assert_allclose(rebuilt, fitted._get_p_level(3), rtol=1e-12, atol=1e-12)


class QuantileDomainIsUniformTest(unittest.TestCase):
    """Q01-F01 / Q08-F08: the shared out-of-domain refusal held on 15 of 32 families.

    The CHANGELOG says every family that defines ``quantile`` refuses an out-of-domain ``q`` the
    same way, and the migration guide repeats it. It was not so: ``HalfNormal(1).quantile(-0.1)``
    returned ``-0.1257``, ``BetaBinomial(10, 2, 3).quantile(1.1)`` returned the support point
    ``10.0`` -- a plausible-looking answer for a level the caller computed wrong -- and
    Uniform/Laplace/StudentT let scipy answer NaN.

    The source test below is the one that matters. Enumerating families by hand is how the claim
    narrowed in the first place, so it walks the tree instead: every ``def quantile`` under
    ``mixle/stats`` must route through the shared contract, and a family added tomorrow fails until
    it does.
    """

    OUT_OF_DOMAIN = (-0.1, 1.1, float("nan"))

    def test_every_quantile_in_the_tree_routes_through_the_shared_contract(self):
        import ast

        root = Path(__file__).resolve().parents[1] / "stats"
        offenders = []
        checked = 0
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "def quantile" not in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.ClassDef):
                    continue
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "quantile":
                        checked += 1
                        body = ast.get_source_segment(source, item) or ""
                        if "validated_quantile_probability" not in body:
                            offenders.append(f"{path.name}::{node.name}")
        self.assertGreater(checked, 25, "the walk should find the whole quantile surface")
        self.assertEqual(offenders, [], "quantile implementations bypassing the domain contract")

    def test_the_refusal_is_the_same_on_families_that_answered_differently(self):
        """The five the campaign names, plus three that already refused, in one loop."""
        from mixle import stats

        cases = {
            "HalfNormalDistribution": (1.0,),
            "BetaBinomialDistribution": (10, 2.0, 3.0),
            "UniformDistribution": (-1.0, 2.0),
            "LaplaceDistribution": (0.0, 1.0),
            "StudentTDistribution": (5.0,),
            "GaussianDistribution": (0.0, 1.0),
            "PoissonDistribution": (2.0,),
            "ExponentialDistribution": (1.0,),
        }
        for name, args in cases.items():
            family = getattr(stats, name, None)
            if family is None:
                continue
            distribution = family(*args)
            for q in self.OUT_OF_DOMAIN:
                with self.subTest(family=name, q=q):
                    with self.assertRaises(ValueError) as caught:
                        distribution.quantile(q)
                    self.assertIn("q must be in [0, 1]", str(caught.exception))

    def test_an_in_domain_q_still_answers(self):
        """The refusal must not have swallowed the ordinary path."""
        from mixle import stats

        for name, args, expected in (
            ("HalfNormalDistribution", (1.0,), 0.6745),
            ("UniformDistribution", (-1.0, 2.0), 0.5),
            ("GaussianDistribution", (0.0, 1.0), 0.0),
        ):
            family = getattr(stats, name, None)
            if family is None:
                continue
            with self.subTest(family=name):
                self.assertAlmostEqual(family(*args).quantile(0.5), expected, places=3)


class TerminalStateReadoutsAgreeWithTheModelTest(unittest.TestCase):
    """Q02-F01: the readouts enforced half the terminal restriction.

    A terminal-state HMM's length is a stopping time at the first terminal state, so a path is
    admissible only if it passes through no terminal state before the end AND ends in one.
    ``latent_posterior`` and ``viterbi`` masked the first half only, so the last-row marginal came
    back off by 1.0, ``mode()`` returned paths the model scores as impossible, and every FFBS draw
    ended in a non-terminal state. ``seq_posterior`` and ``log_density`` were right the whole time,
    which is why nothing flagged it.

    With both masks the ordinary recursion IS the restricted one: no position before the last can be
    terminal, so no transition can leave a terminal state. That is asserted here against
    ``terminal_forward_backward`` rather than argued.
    """

    def _model(self, means, w, transitions, terminal):
        from mixle.stats import GaussianDistribution, HiddenMarkovModelDistribution

        return HiddenMarkovModelDistribution(
            [GaussianDistribution(float(m), 1.0) for m in means],
            w=w,
            transitions=transitions,
            terminal_states=terminal,
        )

    def test_the_readouts_equal_the_restricted_forward_backward(self):
        from mixle.stats.latent.hidden_markov import (
            _build_emission_encoder,
            _emission_encoders_from_dists,
            terminal_forward_backward,
        )

        rng = np.random.RandomState(0)
        compared = 0
        for _ in range(20):
            k = int(rng.randint(2, 4))
            terminal = sorted(set(rng.choice(k, size=int(rng.randint(1, k)), replace=False).tolist()))
            model = self._model(
                rng.normal(0, 3, k), rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k), size=k), terminal
            )
            x = list(rng.normal(0, 3, int(rng.randint(1, 6))))
            encoder, _ = _build_emission_encoder(_emission_encoders_from_dists(model.topics))
            log_b = model._state_seq_log_densities(encoder.seq_encode(list(x)))
            _, gamma, _ = terminal_forward_backward(model.log_w, model.log_transitions, log_b, model._terminal_mask)
            if gamma is None:
                continue
            compared += 1
            np.testing.assert_allclose(model.latent_posterior(x).marginals(), gamma, atol=1e-10)
        self.assertGreater(compared, 10, "the sweep should compare a useful number of models")

    def test_no_readout_puts_the_chain_in_a_non_terminal_state_at_the_end(self):
        rng = np.random.RandomState(1)
        for length in (1, 2, 4):
            with self.subTest(length=length):
                model = self._model([-3.0, 3.0], [0.5, 0.5], [[0.7, 0.3], [0.4, 0.6]], [1])
                x = [-3.0] * length  # every observation favours the NON-terminal state 0
                posterior = model.latent_posterior(x)
                marginals = posterior.marginals()
                # The admissible last-row marginal is exactly [0, 1]: only state 1 may end the chain.
                np.testing.assert_allclose(marginals[-1], [0.0, 1.0], atol=1e-10)
                self.assertEqual(int(posterior.mode()[-1]), 1)
                self.assertEqual(int(model.viterbi(x)[-1]), 1)
                for seed in range(25):
                    self.assertEqual(int(posterior.sample(np.random.RandomState(seed))[-1]), 1)

    def _seq_viterbi_fixture(self):
        rng = np.random.RandomState(2)
        model = self._model([-3.0, 0.0, 3.0], [0.4, 0.3, 0.3], [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], [2])
        sequences = [[-3.0] * 4, [-3.0], [-3.0, -3.0, 3.0]] + [list(rng.normal(0, 3, n)) for n in (7, 2, 5)]
        return model, sequences

    def test_the_encoded_batch_viterbi_agrees_with_the_single_sequence_one(self):
        """``seq_viterbi`` is a readout too, and the first repair left it on the unrestricted recursion.

        On ``[-3, -3, -3, -3]`` it kept returning ``[0, 0, 0, 0]`` -- a path the model scores as
        impossible -- after ``viterbi`` had been repaired to ``[0, 0, 0, 1]``. This is the
        time-banded layout the model's own encoder produces.
        """
        model, sequences = self._seq_viterbi_fixture()
        banded = model.dist_to_encoder().seq_encode(sequences)
        flat = np.asarray(model.seq_viterbi(banded))
        (_, _, _, _, idx_mat, _, _), _, _ = banded[0]
        for i, x in enumerate(sequences):
            with self.subTest(sequence=i):
                path = flat[idx_mat[i, : len(x)]].tolist()
                self.assertEqual(path, np.asarray(model.viterbi(x)).tolist())
                self.assertEqual(path[-1], 2)

    @unittest.skipUnless(HAS_NUMBA, "a sequence-contiguous encoding is produced only by the numba encoder")
    def test_the_contiguous_layout_is_restricted_the_same_way(self):
        """The same check on the layout a numba-enabled encoder produces for the same data."""
        from mixle.stats import HiddenMarkovModelDistribution

        model, sequences = self._seq_viterbi_fixture()
        twin = HiddenMarkovModelDistribution(model.topics, w=model.w, transitions=model.transitions, use_numba=True)
        contiguous = twin.dist_to_encoder().seq_encode(sequences)
        self.assertIsNotNone(contiguous[1], "the numba encoder should produce the contiguous layout")
        flat = np.asarray(model.seq_viterbi(contiguous))
        offset = 0
        for i, x in enumerate(sequences):
            with self.subTest(sequence=i):
                path = flat[offset : offset + len(x)].tolist()
                self.assertEqual(path, np.asarray(model.viterbi(x)).tolist())
                self.assertEqual(path[-1], 2)
            offset += len(x)

    def test_an_unrestricted_model_is_untouched(self):
        """The masks must only apply when terminal_states is set."""
        model = self._model([-3.0, 3.0], [0.5, 0.5], [[0.7, 0.3], [0.4, 0.6]], None)
        marginals = model.latent_posterior([-3.0, -3.0, -3.0]).marginals()
        self.assertGreater(marginals[-1][0], 0.5)  # free to end where the data points


class OneTableOneAnswerTest(unittest.TestCase):
    """Q05-F01: the production and scoring verbs each had their own idea of what a table is.

    ``optimize`` reads a DataFrame through ``_reusable_observations`` and gets its rows.
    ``fit_with_provenance``, ``Monitor.check``/``update`` and the scoring verbs each carried a
    two-line spelling -- ``records()`` if present, else iterate the object -- which is right for a
    list and wrong for a table. So the same frame fitted as a categorical over its two COLUMN NAMES
    with ``n_records=2`` stamped into the provenance header, a bare string fitted as its 17
    characters, and ``Service.score`` raised a ContractError on a frame ``optimize`` handles.

    One normalizer now, ``mixle.inference.estimation._tabular_records``, which is what makes the
    CHANGELOG's "one table gets one answer whichever verb reads it" true rather than aspirational.
    """

    FRAME = {"x": [1.0, 2.0, 3.0, 4.0], "k": [0, 1, 0, 1]}

    def _frame(self):
        pandas = __import__("pandas")
        return pandas.DataFrame(self.FRAME)

    def setUp(self):
        try:
            __import__("pandas")
        except ImportError:  # pragma: no cover - pandas is a base test dependency
            self.skipTest("pandas is required to express the defect")

    def test_every_verb_reads_the_same_rows_out_of_the_same_frame(self):
        from mixle.inference import optimize
        from mixle.inference.production import fit_with_provenance

        frame = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            by_optimize = optimize(frame, max_its=2)
            by_provenance, header = fit_with_provenance(frame, None, max_its=2)
        self.assertEqual(type(by_provenance).__name__, type(by_optimize).__name__)
        # Four rows, not two column labels. The header records this as fact about the training set.
        self.assertEqual(header.to_dict().get("n_records"), 4)

    def test_a_bare_string_is_refused_by_every_verb_that_takes_data(self):
        from mixle.inference.production import fit_with_provenance

        with self.assertRaises(ValueError) as caught:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_with_provenance("hello world hello", None, max_its=2)
        self.assertIn("iterates as its individual characters", str(caught.exception))

    def test_the_scoring_verbs_score_the_frame_rather_than_refusing_it(self):
        from mixle.inference import optimize
        from mixle.inference.production import Service

        frame = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = optimize(frame, max_its=2)
            scores = Service(model, reference=frame).score(frame)
        self.assertEqual(np.asarray(scores).shape, (4,))

    def test_the_monitor_reads_a_batch_the_way_a_fit_reads_it(self):
        from mixle.inference import optimize
        from mixle.inference.production import Monitor

        frame = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = optimize(frame, max_its=2)
            monitor = Monitor(model, None, frame)
            report = monitor.check(frame)
        self.assertFalse(report.drift, "a frame checked against itself is not drifted")

    def test_the_verbs_share_one_normalizer(self):
        """The duplication was the defect; assert it is gone rather than trusting behaviour alone.

        This check once listed three modules and passed while ``detect_drift`` -- a fourth, named in
        the same finding -- still called ``list(current)``. The behavioural tests below are the
        real guard; this one only keeps the listed modules from growing a private spelling back.
        """
        import inspect

        from mixle.inference import bayesian_network, structure
        from mixle.inference.production import drift, monitor, provenance, serving

        for module in (drift, monitor, provenance, serving, structure, bayesian_network):
            with self.subTest(module=module.__name__.rsplit(".", 1)[-1]):
                self.assertIn("_tabular_records", inspect.getsource(module))

    def _table(self, n=60):
        rng = np.random.RandomState(0)
        x = rng.normal(size=n)
        return {"x": x.tolist(), "k": (x > 0).astype(int).tolist()}

    def test_the_drift_functions_read_a_frame_as_its_rows(self):
        """``detect_drift`` and ``score_drift`` were still ``list(current)``: two column labels, n_current=2."""
        from mixle.inference import optimize
        from mixle.inference.production import detect_drift
        from mixle.inference.production.drift import score_drift

        frame = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = optimize(frame, max_its=2)
            report = detect_drift(model, frame, frame)
            score = score_drift(model, frame, frame)
        self.assertEqual(report.score["n_current"], 4)
        self.assertFalse(report.drift, "a frame checked against itself is not drifted")
        self.assertEqual((score["n_reference"], score["n_current"]), (4, 4))
        with self.assertRaises(ValueError) as caught:
            score_drift(model, frame, "hello world")
        self.assertIn("score_drift(current)", str(caught.exception))

    def test_the_validation_set_meets_the_same_front_door_as_the_data(self):
        """Q06-F01: ``vdata=`` reached the encoder raw on ``optimize``, ``fit`` and ``best_of``."""
        from mixle.inference import best_of, optimize
        from mixle.stats import CategoricalEstimator, CompositeEstimator

        table = self._table()
        rows = list(zip(table["x"], table["k"]))
        estimator = CompositeEstimator([GaussianEstimator(), CategoricalEstimator()])
        pandas = __import__("pandas")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for spelling, vdata in (
                ("mapping of columns", table),
                ("generator", (row for row in rows)),
                ("DataFrame", pandas.DataFrame(table)),
            ):
                with self.subTest(vdata=spelling):
                    model = optimize(rows, estimator, vdata=vdata, max_its=2, out=None)
                    self.assertEqual(type(model).__name__, "CompositeDistribution")
            _, best = best_of(rows, pandas.DataFrame(table), estimator, 1, 2, 0.1, 1e-6, rng=1)
            self.assertEqual(type(best).__name__, "CompositeDistribution")
        with self.assertRaises(ValueError) as caught:
            optimize(rows, estimator, vdata="hello", max_its=2, out=None)
        self.assertIn("optimize(vdata=)", str(caught.exception))

    def test_best_of_reads_a_frame_as_the_rows_it_holds(self):
        """Q06-F02: ``best_of(DataFrame)`` raised "expected 2-tuples, got DataFrame".

        Compared against ``best_of`` on the same rows spelled as tuples, not against ``optimize``:
        ``optimize`` also runs automatic structure search when no estimator is given and ``best_of``
        does not, so their model classes can legitimately differ on correlated columns. What must not
        differ is the data each verb reads, and with one seed the two spellings give the same fit.
        """
        from mixle.inference import best_of

        table = self._table()
        frame = __import__("pandas").DataFrame(table)
        rows = list(zip(table["x"], table["k"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected_ll, expected = best_of(rows, None, None, 1, 2, 0.1, 1e-6, rng=1)
            for spelling, data in (("DataFrame", frame), ("dict.values()", dict(enumerate(rows)).values())):
                with self.subTest(data=spelling):
                    ll, model = best_of(data, None, None, 1, 2, 0.1, 1e-6, rng=1)
                    self.assertEqual(type(model).__name__, type(expected).__name__)
                    self.assertAlmostEqual(ll, expected_ll, places=9)

    def test_the_structure_learners_read_a_table_and_refuse_what_is_not_one(self):
        """Q06-F03 / Q02-F08 / Q03-F04: a mapping of columns was learned as two records of its KEYS."""
        from mixle.inference import learn_bayesian_network, learn_structure

        table = self._table()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for learner in (learn_structure, learn_bayesian_network):
                with self.subTest(learner=learner.__name__, data="mapping of columns"):
                    model = learner(table, max_its=2) if learner is learn_bayesian_network else learner(table)
                    self.assertEqual(model.fit_provenance().n_observations, len(table["x"]))
                with self.subTest(learner=learner.__name__, data="str"):
                    with self.assertRaises(ValueError) as caught:
                        learner("hello world hello")
                    self.assertIn(learner.__name__, str(caught.exception))
                with self.subTest(learner=learner.__name__, data="scalars"):
                    with self.assertRaises(ValueError) as caught:
                        learner([1.0, 2.0, 3.0])
                    self.assertIn("one entry per field", str(caught.exception))

    def test_a_record_model_works_on_every_route_its_fit_does(self):
        """Q06-F05: ``optimize(df, RecordEstimator(...))`` fitted, and no other verb could read that frame."""
        from mixle import Model
        from mixle.inference import optimize
        from mixle.inference.production import Monitor, Service, detect_drift
        from mixle.stats import CategoricalEstimator
        from mixle.stats.combinator.record import RecordEstimator, field

        pandas = __import__("pandas")
        rng = np.random.RandomState(0)
        frame = pandas.DataFrame({"x": rng.normal(size=80), "k": ["a", "b"] * 40})
        rows = frame.to_dict("records")
        estimator = RecordEstimator(
            [field("mean", "x"), field("kind", "k")], [GaussianEstimator(), CategoricalEstimator()]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = optimize(frame, estimator, max_its=3, out=None)
            self.assertTrue(Model(estimator).fit(frame, max_its=2).fitted)
            self.assertEqual(Model(estimator).fit(rows, max_its=2).evaluate(frame)["n"], 80)
            self.assertEqual(np.asarray(Service(model).score(frame)).shape, (80,))
            for spelling, data in (("DataFrame", frame), ("mapping rows", rows)):
                with self.subTest(drift_input=spelling):
                    self.assertFalse(detect_drift(model, data, data).drift)
            self.assertFalse(Monitor(model, estimator, rows).check(frame).drift)


class SupportLimitedComponentsCanBeMixedTest(unittest.TestCase):
    """Q01-F02 / Q02-F02: a mixture refused a model it can express, and blamed the data.

    Two halves, both of them the same asymmetry between what a family refuses and what it tells the
    rest of the library about itself.

    Initialization: four families refused out-of-support rows in ``seq_update`` without exposing
    ``supported_rows``, so a latent model's initializer could not know to withhold those rows. It
    seeded the component from one anyway and the refusal then named the DATA -- a Poisson-plus-
    Geometric fit on counts containing zeros was refused outright, though the Poisson component
    explains the zeros perfectly well.

    Encoding: ``ParetoDataEncoder`` refused non-positive rows, where the contract is that encoders
    admit them so a mixture can encode one batch against every component. The law already agreed
    with itself -- ``seq_log_density`` masks them to -inf and the scalar path returns -inf -- so only
    the encoder disagreed.
    """

    def test_a_poisson_plus_geometric_mixture_fits_counts_containing_zeros(self):
        from mixle.inference import optimize
        from mixle.stats import GeometricEstimator, MixtureEstimator, PoissonEstimator

        rng = np.random.RandomState(0)
        counts = np.concatenate([rng.poisson(2, 200), rng.geometric(0.3, 200)])
        self.assertGreater(int((counts == 0).sum()), 0, "the defect needs zeros in the data")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = optimize(
                counts,
                MixtureEstimator([PoissonEstimator(), GeometricEstimator()]),
                max_its=20,
                rng=np.random.RandomState(1),
            )
        self.assertEqual(len(fitted.w), 2)
        self.assertTrue(all(weight > 0.05 for weight in fitted.w), f"a component was starved: {fitted.w}")

    def test_a_gaussian_plus_pareto_mixture_encodes_one_batch(self):
        from mixle.inference import optimize
        from mixle.stats import GaussianEstimator, MixtureEstimator, ParetoEstimator

        rng = np.random.RandomState(0)
        x = np.concatenate([rng.normal(-3, 1, 200), rng.pareto(3.0, 200) + 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = optimize(
                x,
                MixtureEstimator([GaussianEstimator(), ParetoEstimator()]),
                max_its=20,
                rng=np.random.RandomState(1),
            )
        self.assertEqual(len(fitted.w), 2)

    def test_the_pareto_encoder_admits_what_its_density_already_scores(self):
        from mixle.stats import ParetoDistribution

        law = ParetoDistribution(1.0, 3.0)
        encoded = law.dist_to_encoder().seq_encode([-1.0, 1.5])
        densities = law.seq_log_density(encoded)
        self.assertEqual(float(densities[0]), float("-inf"))
        self.assertEqual(float(law.log_density(-1.0)), float("-inf"))
        self.assertTrue(np.isfinite(densities[1]))

    def test_nan_is_still_refused_as_missing_data(self):
        """Admitting out-of-support rows must not have admitted missing ones."""
        from mixle.stats import ParetoDistribution

        with self.assertRaises(ValueError) as caught:
            ParetoDistribution(1.0, 3.0).dist_to_encoder().seq_encode([float("nan"), 1.5])
        self.assertIn("NaN marks missing data", str(caught.exception))

    def test_a_family_that_refuses_rows_also_declares_which_rows_it_supports(self):
        """The asymmetry itself, asserted: refusing without declaring is what broke initialization."""
        import ast

        root = Path(__file__).resolve().parents[1] / "stats"
        asymmetric = []
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "refuse_unsupported_observations" not in source or "_observation_contracts" in path.name:
                continue
            if "def refuse_unsupported_observations" in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ClassDef):
                    body = ast.get_source_segment(source, node) or ""
                    if "refuse_unsupported_observations(" in body and "def supported_rows" not in body:
                        asymmetric.append(f"{path.name}::{node.name}")
        self.assertEqual(asymmetric, [], "these refuse rows they never declare unsupported")


class EnsembleBurnInPaysForItsInitializationTest(unittest.TestCase):
    """Q04-F11: ``how='ensemble'`` returned a wrong posterior at its default budget, silently.

    ``_ensemble_p0`` deliberately draws HALF the walkers from the prior so one ensemble spans every
    region the prior supports and genuine multimodality blows up the cross-ensemble R-hat instead of
    being averaged away (RR22-12). The burn-in default predates that change and was never revisited,
    so the walkers never contracted and the unburned prior cloud was reported as parameter
    uncertainty -- sds 3.2-9.1x too WIDE on a 5-parameter model, where mcmc, nuts and hmc all
    returned ratios of ~1.0. Nothing said so: split-R-hat is a deliberate NaN for ensembles (walkers
    interact) and the cross-ensemble statistic needs ``chains >= 2``, so the single-chain default has
    no statistic that can flag it.

    Measured one control at a time: burn=500 -> 3.73 posterior sds off at a 9.14x sd ratio;
    burn=1500 -> 0.19 and 1.06. Raising DRAWS does not fix it (burn=500/draws=5000 is still 1.05 off
    at 5.37x), which is also why an ESS threshold would not have caught this: the extra draws lift
    ess_bulk from 31 to 225 while the posterior stays 5x too wide. The sweeps needed grow with the
    number of parameters -- 5 needed 1500, 18 needed ~5400 -- so the default scales with dimension,
    as ``walkers`` already does.
    """

    @staticmethod
    def _fit(burn):
        from mixle.ppl import DiagGaussian, free

        rng = np.random.RandomState(0)
        x = np.stack([rng.normal(loc, 0.5, 300) for loc in (0, 2, 1, 3, 4)], axis=1)
        kw = {} if burn is None else {"burn": burn}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = DiagGaussian(5, mean=free(5, name="v"), var=np.full(5, 0.25)).fit(
                x.tolist(), how="ensemble", rng=np.random.RandomState(0), **kw
            )
        summary = fit.result.summary()
        exact_sd = 0.5 / np.sqrt(300)
        sds = np.array([summary["v%d" % i]["std"] for i in range(5)])
        means = np.array([summary["v%d" % i]["mean"] for i in range(5)])
        return np.abs(means - x.mean(0)).max() / exact_sd, (sds / exact_sd).max()

    def test_the_default_budget_returns_the_posterior_it_claims(self):
        error_sds, sd_ratio = self._fit(None)
        self.assertLess(error_sds, 1.0, "the posterior mean is off by more than one posterior sd")
        self.assertLess(sd_ratio, 1.5, "the reported posterior is far wider than the exact one")

    def test_the_old_default_is_still_wrong_so_the_test_above_can_fail(self):
        """A budget test that passes at any budget proves nothing; pin the defect it was written for."""
        error_sds, sd_ratio = self._fit(500)
        self.assertGreater(sd_ratio, 2.0, "burn=500 should still show the unburned prior cloud")

    def test_the_default_scales_with_the_number_of_parameters(self):
        """Five parameters needed 1500 sweeps and eighteen needed ~5400; a constant covers neither."""
        import inspect

        from mixle.ppl.inference import ensemble_fit

        self.assertIsNone(inspect.signature(ensemble_fit).parameters["burn"].default)
        source = inspect.getsource(ensemble_fit)
        self.assertIn("max(1500, 300 * d)", source)

    def test_an_explicit_budget_is_still_honoured_and_still_validated(self):
        from mixle.ppl import Normal

        rows = list(np.random.RandomState(0).normal(3, 1, 80))
        mu = Normal(0, 10, name="mu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = Normal(mu, 1.0).fit(rows, how="ensemble", rng=np.random.RandomState(0), burn=200)
            self.assertAlmostEqual(fitted.result.summary()["mu"]["mean"], 3.0, delta=0.3)
            with self.assertRaises(ValueError):
                Normal(mu, 1.0).fit(rows, how="ensemble", rng=np.random.RandomState(0), burn=-5)


if __name__ == "__main__":
    unittest.main()
