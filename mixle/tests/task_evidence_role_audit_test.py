"""Regression tests for the mixle.task evidence-role / receipt-integrity audit findings.

Every test below names the exact behaviour that was reproduced before the fix, so a future reader can
tell what the assertion is defending rather than guessing from the shape of the check.

Covered: MXR-080-1891 (evaluation evidence leaking into selection), MXR-080-1892 (unreplayable
orchestrator traces and unverified plan harvesting), MXR-080-1893 (silently unenforced schema
keywords, mapping-to-key-list coercion, unvalidated discriminated kinds), MXR-080-1894 (a risk
certificate outliving the policy it certifies), MXR-080-1896 (planning mutating the global torch RNG).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False


# --------------------------------------------------------------------------------------- MXR-080-1891


class DesignSelectionRolesTest(unittest.TestCase):
    """Reproduced: ``design_model`` built the LLM's data profile and the heuristic baseline from ALL
    rows, before the holdout split. A holdout-only extreme reached the prompt as ``max``/``mean``/
    ``var``, and the baseline structure (also the returned fallback) was chosen with the holdout in
    hand -- so the "independent held-out density" the receipt advertises was not independent."""

    def test_profile_and_baseline_see_only_the_training_split(self):
        import mixle.task.recommend as recommend_module
        from mixle.task.design import _split_validation_data, design_model
        from mixle.task.llm import CallableLLM

        rng = np.random.RandomState(0)
        rows = [float(x) for x in rng.randn(40)]
        rows[34] = 1.0e6  # index 34 lands in the holdout under the default validation_seed=17
        train, holdout = _split_validation_data(rows, validate_rows=200, holdout_frac=0.25, validation_seed=17)
        self.assertIn(1.0e6, holdout)
        self.assertNotIn(1.0e6, train)

        seen: dict[str, object] = {}

        def _llm(prompt, system=None):
            seen["profile"] = json.loads(prompt)
            return '{"family":"gaussian"}'

        recommend_rows: list[list[float]] = []
        real_recommend = recommend_module.recommend_model

        def _spy(data, **kw):
            recommend_rows.append(list(data))
            return real_recommend(data, **kw)

        with mock.patch.object(recommend_module, "recommend_model", _spy):
            design_model(rows, CallableLLM(_llm))

        profile = seen["profile"]
        self.assertEqual(profile["n_rows"], len(train))
        # the holdout extreme is gone from every statistic the LLM designs from
        self.assertLess(abs(profile["fields"][0]["mean"]), 1.0e5)
        self.assertNotIn(1.0e6, recommend_rows[0])
        self.assertEqual(len(recommend_rows[0]), len(train))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class SolveHoldoutRolesTest(unittest.TestCase):
    """Reproduced: ``solve()``'s reserved holdout was ONE set. ``improve()`` chose the student with the
    same rows that then set its conformal threshold, and ``holdout_agreement`` was a running maximum
    over that fixed set rather than a held-out estimate -- across unboundedly many rounds, silently."""

    @staticmethod
    def _route(ticket):
        if ticket["amount"] > 500 and ticket["kind"] == "refund":
            return "finance-escalation"
        if ticket["kind"] in ("refund", "billing"):
            return "billing"
        return "support"

    @staticmethod
    def _tickets(n, seed=0):
        rng = np.random.RandomState(seed)
        kinds = ["refund", "billing", "question", "bug"]
        return [
            {
                "kind": kinds[rng.randint(0, 4)],
                "amount": float(rng.gamma(2.0, 150.0)),
                "region": ["us", "eu"][rng.randint(0, 2)],
            }
            for _ in range(n)
        ]

    def test_calibration_and_selection_rows_are_disjoint_and_reuse_is_receipted(self):
        from mixle.task import solve

        sol = solve(self._route, self._tickets(300), alpha=0.15, seed=0, epochs=60)
        cal = {repr(x) for x in sol.cal_inputs}
        sel = {repr(x) for x in sol.sel_inputs}
        self.assertTrue(cal and sel)
        self.assertEqual(cal & sel, set(), "conformal calibration and selection must not share a row")
        train = {repr(x) for x in sol.train_inputs}
        self.assertEqual(train & (cal | sel), set(), "neither reserved role may be trained on")
        # solve() itself is the selection role's first and only use so far
        self.assertEqual(sol.selection_uses, 1)
        self.assertTrue(sol.selection_evidence_is_single_use)
        self.assertTrue(sol.report()["selection_evidence_is_single_use"])

        for t in self._tickets(150, seed=10):
            sol(t)
        if not sol.cascade.stats.escalated_labels:
            self.skipTest("no escalations harvested at this seed; improve() has nothing to fold in")
        sol.improve()
        # the reuse is recorded rather than hidden: the reported agreement is now a selection score
        self.assertEqual(sol.selection_uses, 2)
        self.assertFalse(sol.selection_evidence_is_single_use)
        self.assertFalse(sol.report()["selection_evidence_is_single_use"])
        self.assertEqual(len(sol.selection_receipt), 1)
        entry = sol.selection_receipt[0]
        self.assertFalse(entry["fresh_evidence"])
        self.assertEqual(entry["selection_uses"], 2)
        self.assertEqual(len(entry["evidence_sha256"]), 64)

    def test_fresh_evidence_restores_a_single_use_receipt(self):
        """The escape hatch from the reuse above: a fresh teacher-labeled batch REPLACES both roles,
        so the reported agreement is a genuine single-use held-out number again."""
        from mixle.task import solve

        sol = solve(self._route, self._tickets(300), alpha=0.15, seed=0, epochs=60)
        before = [repr(x) for x in sol.cal_inputs]
        for t in self._tickets(150, seed=10):
            sol(t)
        if not sol.cascade.stats.escalated_labels:
            self.skipTest("no escalations harvested at this seed; improve() has nothing to fold in")
        sol.improve(evidence_inputs=self._tickets(60, seed=99))
        self.assertEqual(sol.selection_uses, 1)
        self.assertTrue(sol.selection_evidence_is_single_use)
        self.assertNotEqual([repr(x) for x in sol.cal_inputs], before)
        self.assertEqual({repr(x) for x in sol.cal_inputs} & {repr(x) for x in sol.sel_inputs}, set())
        self.assertTrue(sol.selection_receipt[-1]["fresh_evidence"])

    def test_invalid_solve_kind_is_refused_instead_of_meaning_record(self):
        """MXR-080-1893: every dispatch spelled ``"text" if kind == "text" else <record>``, so an
        unrecognized kind was never refused -- it silently selected the record path and was stored."""
        from mixle.task import solve

        with self.assertRaisesRegex(ValueError, "kind must be one of"):
            solve(self._route, self._tickets(20), kind="records", seed=0, epochs=5)


# --------------------------------------------------------------------------------------- MXR-080-1892


class OrchestratorTraceReplayTest(unittest.TestCase):
    """Reproduced: every ``TraceStep`` the orchestrator built omitted the RNG states, so
    ``is_bit_identical_replay`` raised ``ValueError: trace step has no captured RNG state`` on the
    orchestrator's own traces -- and a step whose tool raised aborted the replay entirely, so the
    failure paths this loop exists to record were exactly the ones that could not be re-run."""

    class _World:
        failure_atomic = True

        def __init__(self):
            self.n = 0
            self.done = False

        def step(self, action):
            self.n += 1
            if action["args"].get("boom"):
                raise RuntimeError("kaboom")
            return {"ok": action["args"]["i"]}

        def score(self):
            return self.n

        def snapshot(self):
            return {"n": self.n}

    @staticmethod
    def _tool(**kw):
        if kw.get("boom"):
            raise RuntimeError("kaboom")
        return {"ok": kw["i"]}

    def _trace(self):
        from mixle.task.orchestrate import orchestrate

        steps = iter(
            [
                {"tool": "t", "args": {"i": 1}},
                {"tool": "t", "args": {"i": 2, "boom": True}},
                {"tool": "t", "args": {"i": 3}},
                None,
            ]
        )
        return orchestrate("q", lambda q, h: next(steps), self._World(), budget=5).trace

    def test_orchestrated_trace_including_a_failed_step_replays_bit_identically(self):
        from mixle.task.replay import is_bit_identical_replay

        trace = self._trace()
        self.assertEqual(len(trace.steps), 3)
        for step in trace.steps:
            self.assertIsNotNone(step.rng_state_before)
            self.assertIsNotNone(step.rng_state_after)
            self.assertIsInstance(step.succeeded, bool)
        failed = trace.steps[1]
        self.assertFalse(failed.succeeded)
        self.assertEqual(failed.result["error"]["type"], "RuntimeError")
        self.assertTrue(is_bit_identical_replay(trace, {"t": self._tool}))

    def test_a_missing_tool_still_raises_rather_than_recording_a_fake_failure(self):
        """The failure-capturing replay path must not swallow a broken harness."""
        from mixle.task.replay import replay

        with self.assertRaises(KeyError):
            replay(self._trace(), {"other": self._tool})


class TraceHarvestOutcomeBindingTest(unittest.TestCase):
    """Reproduced: a user turn carrying ``tool_result`` blocks was treated as a new request, so the
    harvested plan stopped at the FIRST tool call and the successful retry after it was discarded --
    leaving the errored attempt as the whole "correct" plan taught to the student, with no rejection
    recorded anywhere."""

    @staticmethod
    def _document():
        def block(kind, **kw):
            return {"type": kind, **kw}

        return {
            "id": "c1",
            "messages": [
                {"role": "user", "content": [block("text", text="ship the order")]},
                {"role": "assistant", "content": [block("tool_use", id="tu_1", name="ship", input={"id": "BAD"})]},
                {"role": "user", "content": [block("tool_result", tool_use_id="tu_1", is_error=True, content="no")]},
                {"role": "assistant", "content": [block("tool_use", id="tu_2", name="ship", input={"id": "4242"})]},
                {"role": "user", "content": [block("tool_result", tool_use_id="tu_2", is_error=False, content="ok")]},
                {"role": "assistant", "content": [block("text", text="Shipped.")]},
            ],
        }

    def _harvest(self, document):
        from mixle.task import harvest_agent_traces

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "c1.json").write_text(json.dumps(document))
            return harvest_agent_traces(tmp)

    def test_only_the_call_bound_to_a_successful_result_is_harvested(self):
        traces = self._harvest(self._document())
        self.assertEqual(len(traces), 1)
        trace = traces.traces[0]
        self.assertEqual([step["args"]["id"] for step in trace.plan], ["4242"])
        self.assertTrue(trace.outcomes_verified)
        self.assertEqual(len(traces.rejections), 1)
        self.assertIn("error result", traces.rejections[0].reason)

    def test_an_unbound_call_is_rejected_only_where_outcomes_are_recorded(self):
        document = self._document()
        # drop the successful result: tu_2 now has no bound outcome in a transcript that records them
        document["messages"][4]["content"] = []
        traces = self._harvest(document)
        self.assertEqual(traces.traces[0].plan, [])
        self.assertIn("no bound tool_result", " ".join(r.reason for r in traces.rejections))

        # a transcript with no tool_result blocks at all records no outcomes; its calls are still
        # harvested (that is the shape most stored conversations have) but are not claimed as verified.
        bare = {
            "id": "c2",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "do it"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "go", "input": {}}]},
            ],
        }
        traces = self._harvest(bare)
        self.assertEqual([s["tool"] for s in traces.traces[0].plan], ["go"])
        self.assertFalse(traces.traces[0].outcomes_verified)

    def test_teacher_output_no_longer_aliases_the_harvested_corpus(self):
        """A one-level ``dict(step)`` copy still shared every nested ``args`` value, so a consumer
        editing a plan the teacher handed out rewrote the trace it came from."""
        traces = self._harvest(self._document())
        handed_out = traces.plan_teacher()("ship the order")
        handed_out[0]["args"]["id"] = "MUTATED"
        self.assertEqual(traces.traces[0].plan[0]["args"]["id"], "4242")

        first_call = traces.call_teacher()("ship the order")
        first_call["args"]["id"] = "MUTATED"
        self.assertEqual(traces.traces[0].plan[0]["args"]["id"], "4242")


# --------------------------------------------------------------------------------------- MXR-080-1893


class OutputSchemaVocabularyTest(unittest.TestCase):
    """Reproduced: ``_validate_output_schema`` accepted any keyword outside
    ``{type, enum, properties, required, items}`` and ``_matches_output_schema`` ignored it, so
    ``{"type": "integer", "minimum": 100}`` was accepted at registration and then matched by ``0``."""

    def test_declared_numeric_and_size_bounds_are_enforced(self):
        from mixle.task.catalog_router import _matches_output_schema, _validate_output_schema

        schema = {"type": "integer", "minimum": 100}
        _validate_output_schema(schema)
        self.assertFalse(_matches_output_schema(0, schema))
        self.assertTrue(_matches_output_schema(150, schema))

        self.assertFalse(_matches_output_schema("abcd", {"type": "string", "maxLength": 3}))
        self.assertTrue(_matches_output_schema("ab", {"type": "string", "maxLength": 3}))
        self.assertFalse(_matches_output_schema("11", {"type": "string", "pattern": "^[a-z][0-9]$"}))
        self.assertTrue(_matches_output_schema("x1", {"type": "string", "pattern": "^[a-z][0-9]$"}))
        array_schema = {"type": "array", "items": {"type": "number"}, "minItems": 2}
        self.assertFalse(_matches_output_schema([1.0], array_schema))
        self.assertTrue(_matches_output_schema([1.0, 2.0], array_schema))
        self.assertFalse(_matches_output_schema(1.0, {"type": "number", "exclusiveMinimum": 1.0}))

    def test_a_keyword_outside_the_vocabulary_is_refused_not_dropped(self):
        from mixle.task.catalog_router import _validate_output_schema

        with self.assertRaisesRegex(ValueError, "does not enforce"):
            _validate_output_schema({"type": "integer", "multipleOf": 3})
        with self.assertRaisesRegex(ValueError, "does not enforce"):
            _validate_output_schema({"type": "object", "additionalProperties": False})
        # a keyword valid for another type is still refused for this one
        with self.assertRaisesRegex(ValueError, "does not enforce"):
            _validate_output_schema({"type": "string", "minimum": 1})

    def test_the_existing_vocabulary_still_validates(self):
        """Guard against over-refusal: the shapes the catalog fixtures actually register must pass."""
        from mixle.task.catalog_router import _validate_output_schema

        _validate_output_schema(
            {
                "type": "object",
                "description": "a result",
                "required": ["value"],
                "properties": {
                    "value": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "grade": {"type": "string", "enum": ["a", "b"]},
                },
            }
        )


class BundleRecordCoercionTest(unittest.TestCase):
    """Reproduced: ``list(bundle["items"])`` yielded the KEYS when a bundle stored its records in a
    mapping, so every structured item became a bare string that matched nothing and the gap it should
    have resolved was re-routed as if the evidence had never been supplied."""

    def test_a_mapping_of_records_is_preserved_and_a_scalar_is_refused(self):
        from mixle.task.knowledge_routing import _bundle_records

        item = {"id": "i1", "kind": "artifact"}
        self.assertEqual(_bundle_records({"items": {"i1": item}}, "items"), [item])
        self.assertEqual(_bundle_records({"items": [item]}, "items"), [item])
        self.assertEqual(_bundle_records({}, "items"), [])
        with self.assertRaisesRegex(TypeError, "must be a list of records"):
            _bundle_records({"gaps": "not-a-list"}, "gaps")
        with self.assertRaisesRegex(TypeError, "must be a dictionary"):
            _bundle_records({"items": ["i1"]}, "items")


class AdapterKindRegistrationTest(unittest.TestCase):
    """Reproduced: ``register_adapter`` validated nothing, so ``register_adapter(None, f)`` stored a
    factory under the key ``None`` -- exactly what ``spec.get("kind")`` returns for an ``io`` block
    that declares no kind, so any kind-less spec silently rebuilt as ``f``."""

    def test_a_non_string_kind_is_refused_before_the_registry_is_mutated(self):
        from mixle.task.model import _ADAPTERS, register_adapter

        before = dict(_ADAPTERS)
        for bad in (None, "", "   ", 7):
            with self.assertRaisesRegex(ValueError, "adapter kind must be a non-empty string"):
                register_adapter(bad, lambda spec: "impostor")
        self.assertEqual(dict(_ADAPTERS), before, "a refused registration must leave the registry alone")


# --------------------------------------------------------------------------------------- MXR-080-1894


class GeneratorCertificateBindingTest(unittest.TestCase):
    """Reproduced: ``generate``/``score``/``k``/``seed``/``qhat`` are plain attributes, so swapping the
    generator and the scorer after ``calibrate()`` left the old risk certificate attached and still
    gating -- serving arbitrary output under a bound computed for code that no longer runs, while the
    receipt's own assumption list claimed the policies "remain fixed after calibration"."""

    @staticmethod
    def _calibrated():
        from mixle.task.calibrated_generator import CalibratedGenerator

        def generate(prompt, k):
            return [f"{prompt}-ok"] * k

        def score(candidate):
            return 1.0

        model = CalibratedGenerator(generate, score, alpha=0.2, k=2, seed=0, confidence=0.9)
        return model.calibrate([f"p{i}" for i in range(40)], lambda prompt, candidate: True)

    def test_the_receipt_names_the_policy_it_certifies(self):
        model = self._calibrated()
        policy = model.risk_receipt["policy"]
        self.assertIn("generate", policy["generate"])
        self.assertIn("score", policy["score"])
        self.assertEqual(policy["k"], model.k)
        self.assertEqual(policy["seed"], model.seed)
        self.assertEqual(policy["alpha"], model.alpha)

    def test_serving_under_a_swapped_policy_is_refused(self):
        model = self._calibrated()
        self.assertEqual(model.serve("p0"), "p0-ok")
        model.score = lambda candidate: 99.0
        with self.assertRaisesRegex(RuntimeError, "issued for a different"):
            model.serve("p0")

        model = self._calibrated()
        model.k = model.k + 1
        with self.assertRaisesRegex(RuntimeError, "issued for a different"):
            model.serve("p0")

        model = self._calibrated()
        model.qhat = -1.0e9  # a hand-set threshold is not the certified acceptance rule
        with self.assertRaisesRegex(RuntimeError, "issued for a different"):
            model.serve("p0")

    def test_an_uncertified_generator_is_unaffected(self):
        """Guard against over-refusal: a hand-set ``qhat`` with no certificate is documented as valid."""
        from mixle.task.calibrated_generator import CalibratedGenerator

        model = CalibratedGenerator(lambda p, k: [p] * k, lambda c: 1.0, k=2, qhat=0.0)
        self.assertEqual(model.serve("p0"), "p0")
        model.score = lambda c: 5.0
        self.assertEqual(model.serve("p0"), "p0")


# --------------------------------------------------------------------------------------- MXR-080-1896


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class PlannerGlobalRngTest(unittest.TestCase):
    """Reproduced: ``sft_planner`` called ``torch.manual_seed(seed)``, reseeding the CALLER's global
    torch RNG, so every torch draw after a planner fit silently changed and two planner fits reset an
    unrelated training run's stream."""

    def test_fitting_a_planner_leaves_the_global_torch_rng_where_it_found_it(self):
        import torch

        from mixle.task.sft_plan import sft_planner
        from mixle.task.toolcall import ToolSpec

        def teacher(request):
            return [{"tool": "get", "args": {"city": request.split()[-1]}}]

        requests = [f"weather in city{i}" for i in range(20)]
        tools = [ToolSpec("get", ["city"], ["city"])]

        torch.manual_seed(1234)
        expected = torch.randn(4)

        torch.manual_seed(1234)
        sft_planner(teacher, requests, tools, seed=7, epochs=1, d_model=16, n_layer=1, n_head=1, block=48)
        after = torch.randn(4)
        self.assertTrue(torch.equal(expected, after), "sft_planner must not advance or reseed the caller's RNG")


# --------------------------------------------------------------------------------------- MXR-080-1895


class ExplorationActionExactnessTest(unittest.TestCase):
    """Reproduced: ``ExplorationWorld.step`` coerced the cell with ``int(raw_cell)``, so ``cell="3"``
    was accepted and ``cell=3.7`` was accepted after silently truncating to a DIFFERENT cell than the
    caller named -- both spending real budget on an action the world never validated."""

    def test_only_exact_integer_cells_are_accepted_and_budget_is_untouched_otherwise(self):
        from mixle.task.explore_world import SURVEY_COST, ExplorationWorld

        world = ExplorationWorld(n_cells=10, n_targets=2, budget=40, seed=0)
        for bad in ("3", 3.7, 3.0, True, None, [3]):
            observation = world.step({"type": "survey", "cell": bad})
            self.assertFalse(observation["accepted"], f"cell={bad!r} must be refused, not coerced")
            self.assertEqual(observation["cell"], -1)
        self.assertEqual(world.remaining_budget, 40, "a refused action must not spend budget")

        # guard against over-refusal: exact python and numpy integers are what real policies emit
        for good in (5, np.int64(4)):
            observation = world.step({"type": "survey", "cell": good})
            self.assertTrue(observation["accepted"])
            self.assertEqual(observation["cell"], int(good))
        self.assertEqual(world.remaining_budget, 40 - 2 * SURVEY_COST)


if __name__ == "__main__":
    unittest.main()
