"""sft_planner: a plan-writing LM behind the parse/spec/copy-fidelity gate — never silently wrong."""

import re
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.task.sft_plan import _parse_plan, _plans_match, _serialize_plan


def _teacher(request):
    m = re.search(r"refund order (\d+) for (\w+)", request)
    if m:
        return [
            {"tool": "lookup_order", "args": {"order_id": m.group(1)}},
            {"tool": "notify", "args": {"user": m.group(2)}},
        ]
    m = re.search(r"check status of order (\d+)", request)
    if m:
        return [{"tool": "lookup_order", "args": {"order_id": m.group(1)}}]
    return []


def _requests(n, seed=0):
    rng = np.random.RandomState(seed)
    users = ["bob", "ana", "kim", "raj"]
    out = []
    for _ in range(n):
        oid, user = rng.randint(1000, 9999), users[rng.randint(0, 4)]
        r = rng.rand()
        if r < 0.5:
            out.append(f"please refund order {oid} for {user} as discussed")
        elif r < 0.85:
            out.append(f"can you check status of order {oid} right away")
        else:
            out.append(f"just wanted to say thanks, note {rng.randint(0, 99)}")
    return out


class PlanGrammarTest(unittest.TestCase):
    def test_serialize_parse_round_trip(self):
        plan = [
            {"tool": "lookup_order", "args": {"order_id": "4242"}},
            {"tool": "notify", "args": {"user": "kim"}},
        ]
        self.assertEqual(_parse_plan(_serialize_plan(plan)), plan)
        self.assertEqual(_parse_plan(_serialize_plan([])), [])

    def test_serialize_parse_escapes_structural_and_typed_values(self):
        plan = [
            {
                "tool": "notify",
                "args": {
                    "message": "a; b | c=(d)\nnext",
                    "count": 2,
                    "enabled": False,
                    "empty": "",
                },
            }
        ]
        self.assertEqual(_parse_plan(_serialize_plan(plan)), plan)

    def test_malformed_text_is_rejected_not_guessed(self):
        for bad in ("lookup_order(order_id=", "notify user=kim)", "do(x=1) & do(y=2)", "notify(=kim)"):
            self.assertIsNone(_parse_plan(bad + "\n"))

    def test_plan_agreement_compares_optional_arguments_too(self):
        from mixle.task import ToolSpec

        specs = {
            "notify": ToolSpec(
                "notify",
                ["user", "channel"],
                required=["user"],
            )
        }
        want = [{"tool": "notify", "args": {"user": "kim", "channel": "email"}}]
        wrong = [{"tool": "notify", "args": {"user": "kim", "channel": "sms"}}]
        missing = [{"tool": "notify", "args": {"user": "kim"}}]
        self.assertFalse(_plans_match(wrong, want, specs))
        self.assertFalse(_plans_match(missing, want, specs))

    def test_fixed_values_require_declared_vocabulary_not_tool_name_substrings(self):
        from mixle.task import ToolSpec
        from mixle.task.sft_plan import GenerativePlanner, _CharCodec

        request = "please process this transaction"
        plan = [{"tool": "refund_order", "args": {"kind": "refund"}}]
        undeclared = GenerativePlanner(
            lm=None,
            codec=_CharCodec(["x"]),
            tools={"refund_order": ToolSpec("refund_order", ["kind"])},
            teacher=_teacher,
            plan_agreement=0.0,
        )
        declared = GenerativePlanner(
            lm=None,
            codec=_CharCodec(["x"]),
            tools={
                "refund_order": ToolSpec(
                    "refund_order",
                    ["kind"],
                    vocabulary={"kind": ["refund"]},
                )
            },
            teacher=_teacher,
            plan_agreement=0.0,
        )
        self.assertFalse(undeclared._validate(plan, request))
        self.assertTrue(declared._validate(plan, request))

    def test_teacher_traces_are_validated_once_and_eval_is_split(self):
        from mixle.task import ToolSpec, sft_planner

        calls = 0

        def teacher(_request):
            nonlocal calls
            calls += 1
            return []

        class FakeLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit_pairs(self, pairs, **kwargs):
                self.pairs = list(pairs)

        requests = ["duplicate request"] * 16
        with (
            patch("mixle.models.LM", FakeLM),
            patch(
                "mixle.task.constrained.constrained_plan_decode",
                return_value=("done\n", 0.0),
            ),
        ):
            planner = sft_planner(
                teacher,
                requests,
                [ToolSpec("ping", [])],
                epochs=1,
            )

        self.assertEqual(calls, len(requests))
        self.assertEqual(len(planner.lm.pairs), 12)
        self.assertEqual(planner.calibration_size, 2)
        self.assertEqual(planner.test_size, 2)
        self.assertEqual(planner.calibration_agreement, 1.0)
        self.assertEqual(planner.plan_agreement, 1.0)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class CopyFidelityGateTest(unittest.TestCase):
    def test_copied_values_must_occur_in_the_request(self):
        from mixle.task import ToolSpec
        from mixle.task.sft_plan import GenerativePlanner, _CharCodec

        gp = GenerativePlanner(
            lm=None,
            codec=_CharCodec(["x"]),
            tools={"lookup_order": ToolSpec("lookup_order", ["order_id"])},
            teacher=_teacher,
            plan_agreement=0.0,
        )
        req = "please refund order 4242 for kim as discussed"
        good = [{"tool": "lookup_order", "args": {"order_id": "4242"}}]
        drifted = [{"tool": "lookup_order", "args": {"order_id": "4202"}}]  # the silent copy error
        self.assertTrue(gp._validate(good, req))
        self.assertFalse(gp._validate(drifted, req))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class SftPlannerTest(unittest.TestCase):
    def test_generates_verified_plans_never_silently_wrong(self):
        from mixle.task import ToolSpec, sft_planner

        tools = [ToolSpec("lookup_order", ["order_id"]), ToolSpec("notify", ["user"])]
        # epochs=15/n_layer=1 verified (10+ seeds) to preserve the silent_wrong==0 invariant just as
        # reliably as epochs=40/n_layer=2 while training ~4x faster; n_train=180 is NOT safely
        # reducible -- smaller corpora starve the confidence-floor calibration and let wrong plans
        # through confidently (see sft_planner's holdout calibration in mixle/task/sft_plan.py).
        planner = sft_planner(_teacher, _requests(180), tools, seed=0, epochs=15, d_model=64, n_layer=1)

        specs = {t.name: t for t in tools}
        silent_wrong = 0
        for r in _requests(40, seed=7):
            out = planner(r)
            if not out["escalate"] and not _plans_match(out["plan"], _teacher(r), specs):
                silent_wrong += 1
        self.assertEqual(silent_wrong, 0)  # THE invariant: the gate lets no wrong plan out
        rep = planner.report()
        self.assertEqual(rep["requests"], 40)
        self.assertEqual(rep["harvested_traces"], rep["escalated"])


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ScoreAndSamplePlansTest(unittest.TestCase):
    """workstream C1/C2: a decomposition model you can fit, score, and sample -- a low-probability plan
    is an escalation signal, not a silent guess."""

    @classmethod
    def setUpClass(cls):
        # trained ONCE (seed=0, deterministic) and reused read-only across every test below -- the
        # 4 tests in this class never mutate the planner, so a per-test setUp was retraining the
        # identical model 4 times over for no behavioral difference.
        from mixle.task import ToolSpec, sft_planner

        # epochs=15/n_layer=1 verified (8+ seeds) to keep the same clean score separation between
        # correct/wrong/implausible plans and the calibrated floor as epochs=40/n_layer=2, ~4x faster
        # to train; n_train=180 is load-bearing for calibration quality and left unchanged.
        cls.tools = [ToolSpec("lookup_order", ["order_id"]), ToolSpec("notify", ["user"])]
        cls.planner = sft_planner(_teacher, _requests(180), cls.tools, seed=0, epochs=15, d_model=64, n_layer=1)

    def test_the_teacher_plan_scores_far_above_a_wrong_plan(self):
        from mixle.task import score_plan

        req = "please refund order 5555 for bob as discussed"
        correct = score_plan(self.planner, req, _teacher(req))
        wrong = score_plan(self.planner, req, [{"tool": "notify", "args": {"user": "bob"}}])
        self.assertGreater(correct, wrong)
        self.assertGreater(correct, self.planner.conf_floor)  # the teacher plan clears the escalation floor

    def test_a_low_probability_plan_falls_below_the_calibrated_floor(self):
        from mixle.task import score_plan

        req = "please refund order 5555 for bob as discussed"
        # a plausible-looking but wrong-order plan: notify before the lookup it depends on
        implausible = [
            {"tool": "notify", "args": {"user": "bob"}},
            {"tool": "lookup_order", "args": {"order_id": "5555"}},
        ]
        self.assertLess(score_plan(self.planner, req, implausible), self.planner.conf_floor)

    def test_sample_plans_returns_n_candidates_sorted_by_score(self):
        from mixle.task import sample_plans

        req = "can you check status of order 4242 right away"
        samples = sample_plans(self.planner, req, n=5, temperature=0.7, seed=3)
        self.assertEqual(len(samples), 5)
        scores = [s for _, s in samples]
        self.assertEqual(scores, sorted(scores, reverse=True))  # highest-probability candidate first

    def test_an_unparseable_sample_is_reported_not_guessed(self):
        from mixle.task import sample_plans

        req = "can you check status of order 4242 right away"
        # a very high temperature makes malformed/invalid draws likely -- they must surface as (None, -inf),
        # never as a silently-returned plan that failed to parse or validate
        samples = sample_plans(self.planner, req, n=8, temperature=5.0, seed=9)
        for plan, score in samples:
            if plan is None:
                self.assertEqual(score, float("-inf"))
            else:
                self.assertGreater(score, float("-inf"))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class GenerativePlannerPersistenceTest(unittest.TestCase):
    def test_save_load_plans_identically(self):
        import tempfile

        from mixle.task import GenerativePlanner, ToolSpec, sft_planner

        # This test only checks that save/load reproduces the SAME planner byte-for-byte-behaviorally
        # (identical plans/escalations from a fresh process) -- it makes no claim about plan quality,
        # so the model/corpus can be as small as sft_planner allows (>=16 requests). Verified (6+ seeds)
        # to round-trip identically at this size just as reliably as the original, ~10x+ faster.
        tools = [ToolSpec("lookup_order", ["order_id"]), ToolSpec("notify", ["user"])]
        planner = sft_planner(_teacher, _requests(24), tools, seed=0, epochs=8, d_model=16, n_layer=1)
        fresh = _requests(8, seed=11)
        want = [planner(r) for r in fresh]
        with tempfile.TemporaryDirectory() as d:
            path = planner.save(d + "/gen")
            back = GenerativePlanner.load(path, _teacher)
            got = [back(r) for r in fresh]
        self.assertEqual(got, want)  # identical plans + escalations in a fresh process
        self.assertEqual(back.conf_floor, planner.conf_floor)
        self.assertAlmostEqual(back.plan_agreement, planner.plan_agreement, places=6)


if __name__ == "__main__":
    unittest.main()
