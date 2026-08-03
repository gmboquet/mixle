"""System scorecard (mixle.system.scorecard), CARD SCORE-a: evaluate() over a fixed set; catch a worsening round."""

import unittest

from mixle.substrate.core import Substrate, SubstrateItem
from mixle.system import Query, System, SystemConfig, SystemScorecard, detect_regression, evaluate
from mixle.system.scorecard import DEFAULT_SCORER_VERSION, _default_scorer, question_set_identity


class EvaluateTest(unittest.TestCase):
    def test_quality_and_grounded_fraction_on_a_perfect_system(self):
        system = System(SystemConfig(teacher=lambda p: "yes, absolutely"))
        question_set = [(Query("q1"), "yes"), (Query("q2"), "yes")]
        card = evaluate(system, question_set)
        self.assertEqual(card.quality, 1.0)
        self.assertEqual(card.grounded_fraction, 1.0)
        self.assertEqual(card.realized_cost, 2.0)
        self.assertEqual(card.n, 2)

    def test_quality_reflects_wrong_answers(self):
        system = System(SystemConfig(teacher=lambda p: "no idea"))
        question_set = [(Query("q1"), "yes"), (Query("q2"), "yes")]
        card = evaluate(system, question_set)
        self.assertEqual(card.quality, 0.0)

    def test_empty_question_set_is_honest_not_a_div_by_zero_crash(self):
        system = System(SystemConfig(teacher=lambda p: "x"))
        card = evaluate(system, [])
        self.assertEqual(card.n, 0)
        self.assertEqual(card.quality, 0.0)

    def test_teacher_down_answers_are_not_counted_grounded(self):
        def broken_teacher(prompt):
            raise ConnectionError("down")

        store = Substrate()
        store.put(SubstrateItem(kind="text", text="yes it is confirmed"))
        system = System(SystemConfig(teacher=broken_teacher, store=store))
        card = evaluate(system, [(Query("q1"), "yes")])
        self.assertEqual(card.grounded_fraction, 0.0)


class EvaluateDoesNotLeakIntoTrainingTest(unittest.TestCase):
    """CRITICAL: evaluate() must be a genuine read-only snapshot. If it were not, a successful teacher
    answer to a held-out question would land in System._harvest as a side effect of just answering it,
    and a later improve() call could promote it into the captured cache -- after which re-evaluating
    that same "held-out" set would be trivially free and perfect, because the system would have
    memorized the test answers from having been evaluated on them once already."""

    def test_evaluating_a_held_out_question_does_not_populate_the_harvest(self):
        system = System(SystemConfig(teacher=lambda p: "ok"))
        self.assertEqual(system._harvest, {})

        evaluate(system, [(Query("secret-test-question"), "ok")])

        self.assertEqual(system._harvest, {})

    def test_a_generous_improve_after_evaluation_still_has_nothing_to_promote(self):
        system = System(SystemConfig(teacher=lambda p: "ok"))
        evaluate(system, [(Query("secret-test-question"), "ok")])

        report = system.improve(1000)

        self.assertEqual(report["status"], "nothing_to_improve")
        self.assertEqual(system._captured, {})

    def test_reevaluating_the_same_held_out_set_costs_the_same_every_time(self):
        # if evaluation ever leaked into the cache, the SECOND evaluate() over the same set would show
        # realized_cost collapsing towards 0 (served free from the captured cache); it must not move.
        system = System(SystemConfig(teacher=lambda p: "ok"))
        question_set = [(Query("secret-test-question"), "ok")]

        first = evaluate(system, question_set)
        system.improve(1000)  # even a very generous improve() must find nothing to promote
        second = evaluate(system, question_set)

        self.assertEqual(first.realized_cost, 1.0)
        self.assertEqual(second.realized_cost, 1.0)
        self.assertEqual(system.total_spend.to_dict()["frontier_calls"], 0)  # eval never touches this

    # negative control: evaluate() must still faithfully exercise the real answering path -- read-only
    # must not turn evaluation into a no-op that always reports perfect (or always zero) quality
    def test_evaluate_still_reflects_real_answer_quality(self):
        system = System(SystemConfig(teacher=lambda p: "yes, absolutely"))
        card = evaluate(system, [(Query("q1"), "yes")])
        self.assertEqual(card.quality, 1.0)
        self.assertEqual(card.grounded_fraction, 1.0)
        self.assertEqual(card.realized_cost, 1.0)


class RegressionDetectionTest(unittest.TestCase):
    def test_a_deliberately_worsening_round_is_caught(self):
        question_set = [(Query("q1"), "yes"), (Query("q2"), "yes")]

        good_system = System(SystemConfig(teacher=lambda p: "yes, absolutely"))
        baseline = evaluate(good_system, question_set)

        bad_system = System(SystemConfig(teacher=lambda p: "no idea"))
        worse = evaluate(bad_system, question_set)

        report = detect_regression(baseline, worse)
        self.assertTrue(report.regressed)
        self.assertTrue(any("quality regressed" in r for r in report.reasons))

    def test_a_grounded_fraction_drop_is_caught(self):
        question_set = [(Query("q1"), "yes")]
        good_system = System(SystemConfig(teacher=lambda p: "yes"))
        baseline = evaluate(good_system, question_set)

        def broken_teacher(prompt):
            raise ConnectionError("down")

        store = Substrate()
        store.put(SubstrateItem(kind="text", text="yes indeed"))
        degraded_system = System(SystemConfig(teacher=broken_teacher, store=store))
        current = evaluate(degraded_system, question_set)

        report = detect_regression(baseline, current)
        self.assertTrue(report.regressed)
        self.assertTrue(any("grounded_fraction regressed" in r for r in report.reasons))

    def test_a_non_regressing_round_is_not_flagged(self):
        question_set = [(Query("q1"), "yes")]
        system = System(SystemConfig(teacher=lambda p: "yes, absolutely"))
        baseline = evaluate(system, question_set)
        current = evaluate(system, question_set)
        report = detect_regression(baseline, current)
        self.assertFalse(report.regressed)
        self.assertEqual(report.reasons, [])

    def test_an_improving_round_is_not_flagged(self):
        question_set = [(Query("q1"), "yes")]
        bad_system = System(SystemConfig(teacher=lambda p: "no idea"))
        baseline = evaluate(bad_system, question_set)
        good_system = System(SystemConfig(teacher=lambda p: "yes, absolutely"))
        improved = evaluate(good_system, question_set)
        report = detect_regression(baseline, improved)
        self.assertFalse(report.regressed)


class NoFalseCalibrationMetricTest(unittest.TestCase):
    """MXR-080-1691: `calibration` was `quality` copied verbatim -- a deterministic system with no
    confidence at all scored a perfect "calibration" -- and detect_regression never compared it, so a
    1.0 -> 0.0 drop on the field was not a regression. A metric that measures nothing and gates nothing
    must not be reported at all."""

    def test_scorecards_do_not_report_a_calibration_number(self):
        system = System(SystemConfig(teacher=lambda p: "yes"))
        card = evaluate(system, [(Query("q1"), "yes")])
        self.assertNotIn("calibration", card.to_dict())
        self.assertFalse(hasattr(card, "calibration"))

    def test_calibration_is_not_silently_accepted_as_a_constructor_field(self):
        with self.assertRaises(TypeError):
            SystemScorecard(quality=1.0, calibration=1.0, realized_cost=0.0, grounded_fraction=1.0, n=1)


class ComparisonFailsClosedTest(unittest.TestCase):
    """MXR-080-1692: the regression gate failed OPEN on evidence it could not actually compare -- a
    NaN-metric card, a NaN tolerance, and a zero-case card against a 100-case baseline each returned
    `regressed=False`, which reads as "this round is fine"."""

    def _card(self, **kw):
        base = dict(quality=1.0, realized_cost=0.0, grounded_fraction=1.0, n=2, question_set_id="set-a")
        base.update(kw)
        return SystemScorecard(**base)

    def test_nonfinite_and_out_of_range_metrics_are_rejected_at_construction(self):
        for bad in ({"quality": float("nan")}, {"quality": 1.5}, {"grounded_fraction": -0.1}):
            with self.assertRaises(ValueError):
                self._card(**bad)
        with self.assertRaises(ValueError):
            self._card(realized_cost=float("nan"))
        with self.assertRaises(ValueError):
            self._card(realized_cost=-1.0)
        with self.assertRaises(ValueError):
            self._card(n=-1)

    def test_a_nan_tolerance_cannot_silently_disable_the_gate(self):
        baseline = self._card()
        worse = self._card(quality=0.0, realized_cost=99.0, grounded_fraction=0.0)
        self.assertTrue(detect_regression(baseline, worse).regressed)  # sanity: it does catch this
        with self.assertRaises(ValueError):
            detect_regression(baseline, worse, tolerance=float("nan"))
        with self.assertRaises(ValueError):
            detect_regression(baseline, worse, tolerance=-1.0)

    def test_a_card_from_a_different_held_out_set_is_incomparable_not_accepted(self):
        baseline = self._card()
        other_set = self._card(question_set_id="set-b")
        report = detect_regression(baseline, other_set)
        self.assertFalse(report.comparable)
        self.assertTrue(report.regressed)

    def test_a_zero_case_card_does_not_pass_as_a_clean_round(self):
        baseline = self._card(n=100)
        empty = self._card(n=0)
        report = detect_regression(baseline, empty)
        self.assertFalse(report.comparable)
        self.assertTrue(report.regressed)
        self.assertTrue(any("case counts" in r for r in report.reasons))

    def test_unidentified_cards_are_incomparable(self):
        report = detect_regression(self._card(question_set_id=""), self._card(question_set_id=""))
        self.assertFalse(report.comparable)
        self.assertTrue(report.regressed)

    def test_evaluate_binds_the_card_to_its_question_set_and_scorer(self):
        system = System(SystemConfig(teacher=lambda p: "yes"))
        set_a = [(Query("q1"), "yes")]
        set_b = [(Query("q2"), "yes")]
        card_a = evaluate(system, set_a)
        card_b = evaluate(system, set_b)
        self.assertEqual(card_a.question_set_id, question_set_identity(set_a))
        self.assertNotEqual(card_a.question_set_id, card_b.question_set_id)
        self.assertFalse(detect_regression(card_a, card_b).comparable)
        # a laxer judge on the same questions is different evidence too
        lax = evaluate(system, set_a, scorer=lambda reply, expected: True)
        self.assertNotEqual(card_a.question_set_id, lax.question_set_id)
        # and the honest same-set comparison still works
        self.assertTrue(detect_regression(card_a, evaluate(system, set_a)).comparable)


def _identity_under_version(version: str) -> str:
    """question_set_identity for the default judge as if DEFAULT_SCORER_VERSION were ``version``."""
    import mixle.system.scorecard as scorecard_module

    original = scorecard_module.DEFAULT_SCORER_VERSION
    scorecard_module.DEFAULT_SCORER_VERSION = version
    try:
        return question_set_identity([(Query("q1"), "yes")])
    finally:
        scorecard_module.DEFAULT_SCORER_VERSION = original


class DefaultJudgeContractTest(unittest.TestCase):
    """MXR-080-1690: the default scorer accepted an empty reference, making every reply correct."""

    def test_a_blank_reference_answer_is_not_scorable(self):
        system = System(SystemConfig(teacher=lambda p: "anything at all"))
        for blank in ("", "   "):
            with self.subTest(expected=repr(blank)), self.assertRaisesRegex(ValueError, "non-empty reference"):
                evaluate(system, [(Query("q1"), blank)])
        with self.assertRaises(ValueError):
            _default_scorer("anything at all", "")

    def test_the_default_judge_carries_a_version_in_the_card_identity(self):
        # cards judged by different versions of the weak default judge must never be compared
        set_a = [(Query("q1"), "yes")]
        default_id = question_set_identity(set_a)
        custom_id = question_set_identity(set_a, scorer=lambda reply, expected: True)
        self.assertNotEqual(default_id, custom_id)
        self.assertTrue(DEFAULT_SCORER_VERSION)
        # the version participates in the identity, so bumping it invalidates old comparisons
        self.assertNotEqual(default_id, _identity_under_version("substring/v-other"))

    def test_the_default_judge_is_documented_as_unable_to_see_negation(self):
        # the known false positive is retained deliberately (a lexical baseline cannot resolve
        # negation without introducing false negatives) but must be stated, not implied.
        self.assertTrue(_default_scorer("Paris is not the answer", "Paris"))
        doc = _default_scorer.__doc__ or ""
        self.assertIn("negation", doc)
        self.assertIn("task-specific", doc)


class _Fingerprint:
    """A plain caller-supplied fingerprint object. ``Query.fingerprint`` is typed ``Any``, so this is
    an ordinary value to put there -- and it has the default, address-bearing ``repr``."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _Fingerprint) and self.value == other.value

    def __hash__(self):
        return hash(self.value)


class _CallableJudge:
    """A judge that is a callable object rather than a function -- no ``__qualname__``."""

    def __call__(self, reply, expected):
        return reply is not None and expected in reply


def _named_judge(reply, expected):
    return reply is not None and expected in reply


class IdentityIsContentBasedNotReprBasedTest(unittest.TestCase):
    """MXR-080-1902 (High): ``question_set_identity`` digested ``repr(query)`` and
    ``_scorer_identity`` fell back to ``repr(scorer)``. An ordinary object's repr embeds its memory
    address, so two EQUAL question sets (and the same judge in a fresh process) produced different
    identities -- and ``detect_regression`` reads a mismatched identity as
    ``comparable=False, regressed=True``. The gate did not merely weaken; it manufactured a
    regression that never happened."""

    def test_two_equal_question_sets_share_one_identity(self):
        set_a = [(Query("q", task="t", fingerprint=_Fingerprint(1)), "paris")]
        set_b = [(Query("q", task="t", fingerprint=_Fingerprint(1)), "paris")]
        self.assertEqual(set_a, set_b)
        # pre-fix these two digested differently, because repr(_Fingerprint(1)) carries an address
        with self.assertRaisesRegex(ValueError, "no stable identity"):
            question_set_identity(set_a)
        # ...and the same set expressed with content-determined coordinates identifies stably
        set_c = [(Query("q", task="t", fingerprint=[1.0]), "paris")]
        set_d = [(Query("q", task="t", fingerprint=[1.0]), "paris")]
        self.assertEqual(question_set_identity(set_c), question_set_identity(set_d))

    def test_an_improved_round_on_the_same_set_is_not_reported_as_a_regression(self):
        # The end-to-end consequence: the fabricated identity mismatch turned a genuine improvement
        # into ``comparable=False, regressed=True``.
        question_set = [(Query("q1", fingerprint=[0.5, 1.0]), "yes")]
        baseline = SystemScorecard(
            quality=0.5,
            realized_cost=1.0,
            grounded_fraction=1.0,
            n=1,
            question_set_id=question_set_identity(question_set),
        )
        rebuilt = [(Query("q1", fingerprint=[0.5, 1.0]), "yes")]  # the same set, constructed again
        improved = SystemScorecard(
            quality=0.9,
            realized_cost=1.0,
            grounded_fraction=1.0,
            n=1,
            question_set_id=question_set_identity(rebuilt),
        )
        report = detect_regression(baseline, improved)
        self.assertTrue(report.comparable, report.reasons)
        self.assertFalse(report.regressed, report.reasons)

    def test_different_content_still_gets_different_identities(self):
        # Negative control: the digest must still DISTINGUISH genuinely different evidence.
        base = [(Query("q1", fingerprint=[1.0]), "yes")]
        for different in (
            [(Query("q2", fingerprint=[1.0]), "yes")],
            [(Query("q1", fingerprint=[2.0]), "yes")],
            [(Query("q1", task="other", fingerprint=[1.0]), "yes")],
            [(Query("q1", fingerprint=[1.0]), "no")],
            [(Query("q1", fingerprint=[1.0]), "yes"), (Query("q2", fingerprint=[1.0]), "yes")],
        ):
            with self.subTest(different=different):
                self.assertNotEqual(question_set_identity(base), question_set_identity(different))
        # order-sensitive, as the old digest was
        pair = [(Query("q1"), "yes"), (Query("q2"), "yes")]
        self.assertNotEqual(question_set_identity(pair), question_set_identity(list(reversed(pair))))

    def test_a_judge_with_no_stable_name_is_refused_rather_than_addressed(self):
        # Pre-fix: _scorer_identity(_CallableJudge()) returned
        # "module.<module._CallableJudge object at 0x...>" -- a new identity per instance and per
        # process, so a card could never be compared with another measured by the same judge.
        question_set = [(Query("q1"), "yes")]
        system = System(SystemConfig(teacher=lambda p: "yes"))
        with self.assertRaisesRegex(ValueError, "no stable identity"):
            question_set_identity(question_set, scorer=_CallableJudge())
        with self.assertRaisesRegex(ValueError, "no stable identity"):
            evaluate(system, question_set, scorer=_CallableJudge())

    def test_ordinary_named_and_lambda_judges_still_identify(self):
        # Negative control against guard overreach: functions, lambdas and the default judge -- the
        # only judge forms anything in this repo actually passes -- all still work.
        question_set = [(Query("q1"), "yes")]
        default_id = question_set_identity(question_set)
        named_id = question_set_identity(question_set, scorer=_named_judge)
        lambda_id = question_set_identity(question_set, scorer=lambda reply, expected: True)
        self.assertEqual(len({default_id, named_id, lambda_id}), 3)
        self.assertEqual(named_id, question_set_identity(question_set, scorer=_named_judge))


if __name__ == "__main__":
    unittest.main()
