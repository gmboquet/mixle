"""The improve() driver, auto_select, the ledger, and decision-regret (mixle.evolve)."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from mixle.evolve import (
    EvolutionLedger,
    auto_select,
    crps_objective,
    decision_regret_objective,
    improve,
    nll_objective,
)
from mixle.evolve.operators import Candidate
from mixle.evolve.verify import Verdict
from mixle.inference import bayes_action, posterior
from mixle.inference.estimation import optimize
from mixle.stats import GaussianDistribution


def _fit(data, mu=0.0, sigma2=1.0):
    return optimize(list(data), GaussianDistribution(mu, sigma2).estimator(), out=None)


class ImproveTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(3.0, 2.0, 500))
        self.nll = nll_objective()

    def test_improve_beats_a_bad_champion(self):
        champion = GaussianDistribution(0.0, 1.0)
        ledger = EvolutionLedger()
        result = improve(champion, self.data, objective=self.nll, ledger=ledger, seed=1)
        self.assertTrue(result.verified)
        self.assertGreater(result.delta, 0.0)
        self.assertLessEqual(self.nll.scalar(result.model, self.data), self.nll.scalar(champion, self.data))
        self.assertTrue(len(ledger) >= 1)
        # the ledger is JSON-serializable.
        json.loads(ledger.to_json())

    def test_anti_regression_never_returns_worse_model(self):
        # an already-MLE champion: no operator may produce a verified worse model; the returned model
        # is the unchanged champion (verified=False) or a model that is no worse on the objective.
        champion = _fit(self.data, 3.0, 2.0)
        for seed in range(4):
            result = improve(champion, self.data, objective=self.nll, seed=seed)
            self.assertLessEqual(
                self.nll.scalar(result.model, self.data),
                self.nll.scalar(champion, self.data) + 1e-6,
            )
            if not result.verified:
                self.assertIs(result.model, champion)

    def test_improve_records_every_attempt(self):
        champion = GaussianDistribution(0.0, 1.0)
        ledger = EvolutionLedger()
        improve(champion, self.data, objective=self.nll, ledger=ledger, seed=2, parent_hash="abc")
        self.assertTrue(all(row["parent_hash"] == "abc" for row in ledger))
        self.assertTrue(all("operator" in row and "delta" in row for row in ledger))

    def test_budget_skips_expensive_operators(self):
        champion = GaussianDistribution(0.0, 1.0)
        ledger = EvolutionLedger()
        # budget below AutoSelect.cost_hint (3.0) -> AutoSelect must be skipped (no ledger row for it).
        improve(champion, self.data, objective=self.nll, ledger=ledger, seed=3, budget=1.5)
        self.assertNotIn("auto_select", [row["operator"] for row in ledger])

    def test_simultaneous_challengers_use_one_selection_aware_fwer_gate(self):
        class Operator:
            cost_hint = 1.0

            def __init__(self, name):
                self.name = name

            def applicable(self, model, data, *, ctx):
                return True

            def propose(self, model, data, *, ctx):
                return Candidate(object(), self.name)

        verdicts = [
            Verdict("challenger", 2.0, 0.03, (0.1, 1.0), "unavailable"),
            Verdict("challenger", 1.0, 0.04, (0.1, 1.0), "unavailable"),
        ]
        ledger = EvolutionLedger()
        champion = object()
        with patch("mixle.evolve.improve.challenger_beats_champion", side_effect=verdicts):
            result = improve(
                champion,
                [0.0, 1.0, 2.0, 3.0],
                objective=self.nll,
                operators=[Operator("a"), Operator("b")],
                ledger=ledger,
                require_calibration=False,
            )

        self.assertFalse(result.verified)
        self.assertIs(result.model, champion)
        adjusted = [row["verdict"]["p_value"] for row in ledger]
        self.assertEqual(adjusted, [0.06, 0.06])
        self.assertTrue(all(row["verdict"]["evidence"]["multiplicity"]["method"] == "holm" for row in ledger))


class AutoSelectTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.data = list(rng.normal(2.0, 1.5, 600))

    def test_bic_picks_a_sensible_family(self):
        result = auto_select(self.data, criterion="bic")
        # a single Gaussian field should be recovered as a univariate continuous model that scores well.
        ld = nll_objective().scalar(result.model, self.data)
        ref = nll_objective().scalar(GaussianDistribution(2.0, 1.5**2), self.data)
        self.assertLess(ld, ref + 0.5)

    def test_proper_score_gate_runs(self):
        result = auto_select(self.data, criterion=crps_objective(seed=0), seed=0)
        # returns a fitted model and a verdict either way; never raises.
        self.assertIsNotNone(result.model)
        self.assertIn("family", result.evidence)

    def test_space_is_phase2(self):
        with self.assertRaises(NotImplementedError):
            auto_select(self.data, space=object())


class DecisionRegretTest(unittest.TestCase):
    def test_bayes_action_picks_known_optimum(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 1.0, 500))
        model = _fit(data, 0.0, 1.0)
        actions = list(np.linspace(0.0, 10.0, 21))

        def sq_loss(a, draw):
            return (np.asarray(draw, dtype=float) - a) ** 2

        res = bayes_action(posterior(model, over="predictive"), sq_loss, actions, n=4000, seed=0)
        # the squared-error Bayes action is the predictive mean ~ 5.
        self.assertAlmostEqual(res["action"], 5.0, delta=0.6)

    def test_decision_regret_lower_for_better_model(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(5.0, 1.0, 500))
        good = _fit(data, 0.0, 1.0)
        bad = GaussianDistribution(0.0, 1.0)
        actions = list(np.linspace(0.0, 10.0, 21))

        def sq_loss(a, draw):
            return (np.asarray(draw, dtype=float) - a) ** 2

        obj = decision_regret_objective(sq_loss, actions, seed=0)
        self.assertLess(obj.scalar(good, data), obj.scalar(bad, data))

    def test_bayes_action_requires_samples_contract(self):
        from mixle.capability import CapabilityError

        with self.assertRaises(CapabilityError):
            bayes_action(object(), lambda a, d: 0.0, [1, 2])


class _RecordedPosterior:
    """A three-draw posterior, so loss-call counts are exact and readable."""

    def __init__(self, draws=(0.0, 1.0, 2.0)):
        self._draws = list(draws)

    def samples(self, n, rng):  # noqa: ARG002 -- the fixture ignores n/rng on purpose
        return list(self._draws)


class _CountingLoss:
    """A scalar loss that records how many times it was invoked (a stand-in for any stateful loss)."""

    def __init__(self, vectorized=None):
        self.calls = 0
        if vectorized is not None:
            self.vectorized = vectorized

    def __call__(self, action, draw):
        self.calls += 1
        return float(np.asarray(draw, dtype=float) - action)


class LossDispatchTest(unittest.TestCase):
    """MXR-080-1611: the loss is invoked exactly the number of times its convention requires."""

    def test_declared_scalar_loss_is_not_probed_with_the_whole_draw_array(self):
        loss = _CountingLoss()
        bayes_action(_RecordedPosterior(), loss, [0.0], n=3, seed=0, vectorized=False)
        self.assertEqual(loss.calls, 3)  # one per draw, no speculative array probe

    def test_loss_can_declare_its_convention_by_attribute(self):
        loss = _CountingLoss(vectorized=False)
        bayes_action(_RecordedPosterior(), loss, [0.0], n=3, seed=0)
        self.assertEqual(loss.calls, 3)

    def test_declared_vectorized_loss_is_invoked_once_per_action(self):
        calls = []

        def loss(action, draws):
            calls.append(action)
            return np.asarray(draws, dtype=float) - action

        bayes_action(_RecordedPosterior(), loss, [0.0, 1.0], n=3, seed=0, vectorized=True)
        self.assertEqual(calls, [0.0, 1.0])

    def test_a_vectorized_loss_returning_the_wrong_length_is_reported_not_re_looped(self):
        def bad_loss(action, draws):  # noqa: ARG001
            return 0.0  # one value for three draws

        with self.assertRaisesRegex(ValueError, "one loss per draw"):
            bayes_action(_RecordedPosterior(), bad_loss, [0.0], n=3, seed=0, vectorized=True)

    def test_auto_detection_probes_at_most_once_across_all_actions(self):
        loss = _CountingLoss()
        bayes_action(_RecordedPosterior(), loss, [0.0, 1.0, 2.0], n=3, seed=0)
        # 3 actions x 3 draws == 9 required evaluations, plus a single shared auto-detect probe.
        self.assertEqual(loss.calls, 10)

    def test_scalar_loss_failures_name_the_action_and_draw(self):
        def loss(action, draw):
            if np.asarray(draw, dtype=float).size == 1 and float(draw) == 1.0:
                raise RuntimeError("boom")
            return 0.0

        with self.assertRaises(RuntimeError) as ctx:
            bayes_action(_RecordedPosterior(), loss, ["hold"], n=3, seed=0, vectorized=False)
        self.assertTrue(any("draw #1 of 3" in note and "'hold'" in note for note in ctx.exception.__notes__))


class _ArrayPosterior:
    """A posterior that returns a fixed loss-shaped draw array, so risk maths is checkable by hand."""

    def __init__(self, draws):
        self._draws = np.asarray(draws, dtype=float)

    def samples(self, n, rng):  # noqa: ARG002 -- the fixture returns its fixed batch
        return self._draws


def _identity_loss(action, draws):  # noqa: ARG001
    return np.asarray(draws, dtype=float)


class DecisionInputContractTest(unittest.TestCase):
    """MXR-080-1610: a decision is only reported over an exact draw count and finite losses."""

    def test_a_nan_loss_is_never_reported_as_the_optimal_action(self):
        # np.argmin([nan, 1]) returns the NaN position, so the unscorable action used to be selected
        # and returned with an all-NaN risk profile.
        def loss(action, draw):  # noqa: ARG001
            return float("nan") if action == "a" else 1.0

        with self.assertRaisesRegex(ValueError, "NaN"):
            bayes_action(_ArrayPosterior(np.arange(10.0)), loss, ["a", "b"], n=10, vectorized=False)

    def test_an_infinite_loss_stays_legitimate_and_simply_loses(self):
        # +inf is how an inadmissible action is written; it must not be confused with a failed
        # evaluation, and it can never win an argmin.
        def loss(action, draw):  # noqa: ARG001
            return float("inf") if action == "forbidden" else 1.0

        res = bayes_action(_ArrayPosterior(np.arange(10.0)), loss, ["forbidden", "ok"], n=10, vectorized=False)
        self.assertEqual(res["action"], "ok")

    def test_fractional_and_boolean_draw_counts_are_rejected(self):
        post = _ArrayPosterior(np.arange(10.0))
        for bad in (2.9, True):
            with self.assertRaises(TypeError):
                bayes_action(post, _identity_loss, ["a"], n=bad)

    def test_nonpositive_draw_counts_report_the_count_not_an_empty_quantile(self):
        post = _ArrayPosterior(np.arange(10.0))
        for bad in (0, -3):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                bayes_action(post, _identity_loss, ["a"], n=bad)

    def test_ragged_and_empty_posterior_sample_batches_are_rejected(self):
        class Ragged:
            def samples(self, n, rng):  # noqa: ARG002
                return {"mu": np.zeros(5), "sigma": np.zeros(3)}

        class Empty:
            def samples(self, n, rng):  # noqa: ARG002
                return []

        with self.assertRaisesRegex(ValueError, "ragged"):
            bayes_action(Ragged(), _identity_loss, ["a"], n=5)
        with self.assertRaisesRegex(ValueError, "no draws"):
            bayes_action(Empty(), _identity_loss, ["a"], n=5)

    def test_cvar_alpha_must_be_a_real_tail_mass(self):
        post = _ArrayPosterior(np.arange(10.0))
        for bad in (0.0, -0.1, 1.5, float("nan")):
            with self.assertRaisesRegex(ValueError, "cvar_alpha"):
                bayes_action(post, _identity_loss, ["a"], n=10, cvar_alpha=bad)


class DecisionTailRiskTest(unittest.TestCase):
    """MXR-080-1612: CVaR averages exactly the requested tail mass, atoms included."""

    def test_cvar_at_an_atom_does_not_absorb_the_whole_distribution(self):
        # 80 zeros, ten ones, ten tens: the interpolated VaR at alpha=0.25 lands on the zero atom, and
        # averaging every loss >= VaR used to pull all 100 samples in and report 1.1 instead of 4.4.
        losses = np.concatenate([np.zeros(80), np.ones(10), np.full(10, 10.0)])
        res = bayes_action(_ArrayPosterior(losses), _identity_loss, ["a"], n=100, cvar_alpha=0.25)
        worst_25 = np.sort(losses)[-25:].mean()
        self.assertAlmostEqual(worst_25, 4.4)
        self.assertAlmostEqual(res["risk_profile"]["cvar"], 4.4)

    def test_cvar_matches_the_plain_tail_mean_when_the_tail_is_a_whole_number_of_draws(self):
        losses = np.arange(100.0)
        res = bayes_action(_ArrayPosterior(losses), _identity_loss, ["a"], n=100, cvar_alpha=0.1)
        self.assertAlmostEqual(res["risk_profile"]["cvar"], float(np.arange(90.0, 100.0).mean()))

    def test_full_tail_mass_is_the_mean_and_a_sub_observation_tail_is_the_worst_draw(self):
        losses = np.arange(100.0)
        whole = bayes_action(_ArrayPosterior(losses), _identity_loss, ["a"], n=100, cvar_alpha=1.0)
        self.assertAlmostEqual(whole["risk_profile"]["cvar"], float(losses.mean()))
        sliver = bayes_action(_ArrayPosterior(losses), _identity_loss, ["a"], n=100, cvar_alpha=0.005)
        self.assertAlmostEqual(sliver["risk_profile"]["cvar"], float(losses.max()))

    def test_cvar_never_understates_the_var_it_is_reported_beside(self):
        rng = np.random.RandomState(0)
        losses = np.round(rng.gamma(1.5, 2.0, 500), 1)  # rounding manufactures plenty of ties/atoms
        for alpha in (0.05, 0.1, 0.25, 0.5):
            res = bayes_action(_ArrayPosterior(losses), _identity_loss, ["a"], n=500, cvar_alpha=alpha)
            self.assertGreaterEqual(res["risk_profile"]["cvar"], res["risk_profile"]["var"])


class LedgerIntegrityTest(unittest.TestCase):
    """MXR-080-1761: the ledger is the evidence, so it must own and attest its own rows."""

    def _ledger(self, n=3):
        led = EvolutionLedger()
        for i in range(n):
            led.record(operator=f"op{i}", delta=float(i), verdict={"promote": False}, cost=1.0, parent_hash="p")
        return led

    def test_the_returned_row_is_not_the_stored_row(self):
        led = self._ledger(1)
        returned = led.record(operator="x", delta=1.0, verdict=None, cost=0.0, parent_hash=None)
        returned["delta"] = 999.0
        returned["meta"]["injected"] = True
        self.assertEqual(led.rows[-1]["delta"], 1.0)
        self.assertNotIn("injected", led.rows[-1]["meta"])
        self.assertTrue(led.verify())

    def test_the_rows_view_cannot_rewrite_or_clear_the_trail(self):
        led = self._ledger()
        view = led.rows
        view[0]["delta"] = 999.0
        with self.assertRaises(AttributeError):
            led.rows.clear()
        self.assertEqual(led.rows[0]["delta"], 0.0)
        self.assertEqual(len(led), 3)
        self.assertTrue(led.verify())

    def test_rows_carry_sequence_schema_and_chained_digests(self):
        led = self._ledger()
        rows = led.rows
        self.assertEqual([row["seq"] for row in rows], [0, 1, 2])
        self.assertTrue(all(row["schema_version"] == 1 for row in rows))
        self.assertEqual(rows[1]["prev_hash"], rows[0]["row_hash"])
        self.assertTrue(led.verify())

    def test_verify_detects_edits_deletions_and_reordering(self):
        base = self._ledger().rows

        tampered = [dict(row) for row in base]
        tampered[1]["delta"] = 42.0
        self.assertFalse(EvolutionLedger(tampered).verify())

        self.assertFalse(EvolutionLedger([base[0], base[2]]).verify())  # deletion
        self.assertFalse(EvolutionLedger([base[0], base[2], base[1]]).verify())  # reordering
        self.assertFalse(EvolutionLedger(list(base[1:])).verify())  # front truncation
        self.assertTrue(EvolutionLedger(list(base)).verify())  # intact

    def test_json_round_trip_preserves_verifiability(self):
        led = self._ledger()
        self.assertTrue(EvolutionLedger.from_json(led.to_json()).verify())


class LedgerStrictJsonTest(unittest.TestCase):
    """MXR-080-1762: receipts serialized bare NaN/Infinity tokens -- a Python extension no conforming
    JSON parser will read -- and flattened unknown evidence objects to str() with no warning."""

    def test_a_non_measurement_is_refused_where_it_enters(self):
        led = EvolutionLedger()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                led.record(operator="op", delta=bad, verdict=None, cost=1.0, parent_hash=None)
            with self.assertRaises(ValueError):
                led.record(operator="op", delta=0.0, verdict=None, cost=bad, parent_hash=None)
        with self.assertRaises(ValueError):
            led.record(operator="op", delta=0.0, verdict=None, cost=-1.0, parent_hash=None)
        self.assertEqual(len(led), 0)

    def test_a_scalar_only_verdicts_nan_p_value_survives_a_strict_round_trip(self):
        # A scalar-only Verdict carries p_value=nan as its documented "no paired test was run"
        # sentinel, so the ledger must be able to HOLD one -- and still emit strict JSON.
        led = EvolutionLedger()
        led.record(
            operator="scalar_only",
            delta=0.5,
            verdict={"p_value": float("nan"), "ci": [float("-inf"), float("inf")]},
            cost=1.0,
            parent_hash=None,
        )
        payload = led.to_json()
        self.assertNotIn("NaN,", payload)
        self.assertNotIn("Infinity,", payload)
        # a strict parser (allow_nan=False on the way back in) accepts the document
        json.loads(payload, parse_constant=_no_constants)

        back = EvolutionLedger.from_json(payload)
        self.assertTrue(back.verify())
        row = back.rows[0]
        self.assertTrue(np.isnan(row["verdict"]["p_value"]))
        self.assertEqual(row["verdict"]["ci"], [float("-inf"), float("inf")])

    def test_an_inexactly_encodable_evidence_object_warns_instead_of_flattening_silently(self):
        class Opaque:
            def __str__(self):
                return "opaque"

        led = EvolutionLedger()
        with self.assertWarns(UserWarning):
            led.record(operator="op", delta=0.0, verdict=None, cost=0.0, parent_hash=None, meta={"x": Opaque()})


def _no_constants(name):
    raise ValueError(f"strict JSON parsers reject the bare token {name!r}")


class _MutatingOperator:
    """An operator that damages the champion it is handed and then fails."""

    name = "mutating"
    cost_hint = 1.0

    def applicable(self, model, data, *, ctx):
        del data, ctx
        return True

    def propose(self, model, data, *, ctx):
        del data, ctx
        model.sabotaged = 99
        raise RuntimeError("operator blew up after mutating the champion")


class ChampionIsolationTest(unittest.TestCase):
    """MXR-080-1763: a failed operator must not be able to change the returned champion."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(3.0, 2.0, 200))
        self.nll = nll_objective()

    def test_a_failed_operator_cannot_mutate_the_returned_champion(self):
        champion = _fit(self.data, 3.0, 2.0)
        result = improve(champion, self.data, objective=self.nll, operators=[_MutatingOperator()], seed=0)
        self.assertFalse(result.verified)
        self.assertIs(result.model, champion)
        self.assertFalse(hasattr(champion, "sabotaged"))

    def test_a_working_operator_still_produces_a_verified_improvement(self):
        # Negative control: isolating operators must not break the ordinary proposal path.
        result = improve(GaussianDistribution(0.0, 1.0), self.data, objective=self.nll, seed=0)
        self.assertTrue(result.verified)


if __name__ == "__main__":
    unittest.main()
