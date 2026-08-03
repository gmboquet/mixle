"""System facade -- the thin shell three verbs (answer/ingest/improve) sit behind (workstream J1/J8)."""

import builtins
import unittest
from unittest.mock import patch

from mixle.substrate.context import ContextBudget, assemble_context
from mixle.substrate.core import Substrate, SubstrateItem
from mixle.substrate.ingest import ingest_documents
from mixle.system import Query, System, SystemConfig


def _fake_teacher(prompt: str) -> str:
    return f"answer to: {prompt}"


class QueryKnowledgeAlignmentTest(unittest.TestCase):
    """Query.from_knowledge_dict is the OTHER half of the mixle-knowledge alignment claim in Query's
    own docstring: build a Query directly from a real assembled ContextPacket (workstream E1's own
    to_knowledge_dict output), not just a claim that the field names happen to match."""

    def test_query_built_from_a_real_assembled_context_packet(self):
        substrate = Substrate()
        ingest_documents(substrate, ["cats are mammals that purr"], source="animal facts")
        pkt = assemble_context(substrate, "mammals", budget=ContextBudget(max_chars=200))
        d = pkt.to_knowledge_dict(id="pkt1", project_id="proj1", target_kind="frontier_llm")

        query = Query.from_knowledge_dict(d, scope="project")
        self.assertEqual(query.task, "mammals")
        self.assertEqual(query.text, pkt.render())
        self.assertEqual(query.scope, "project")

    def test_expected_output_schema_maps_to_expected_output(self):
        d = {"task": "extract", "payload": {"rendered": "hi"}, "expected_output_schema": {"type": "object"}}
        query = Query.from_knowledge_dict(d)
        self.assertEqual(query.expected_output, {"type": "object"})

    def test_missing_expected_output_schema_stays_none(self):
        d = {"task": "chat", "payload": {"rendered": "hi"}}
        query = Query.from_knowledge_dict(d)
        self.assertIsNone(query.expected_output)


class SystemAnswerTest(unittest.TestCase):
    def test_answer_routes_to_teacher_and_returns_a_receipt(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        reply, receipt = system.answer(Query("what is 2+2?", task="qa"))

        self.assertEqual(reply, "answer to: what is 2+2?")
        self.assertEqual(receipt["produced_by"], "teacher")
        # SPEND-a: spend is now the full Spend ledger shape (frontier_calls/oracle_calls/wall_ms/dollars),
        # not the earlier ad hoc {"frontier_calls": 1} dict.
        self.assertEqual(receipt["spend"], {"frontier_calls": 1, "oracle_calls": 0, "wall_ms": 0.0, "dollars": 0.0})
        self.assertEqual(receipt["task"], "qa")
        self.assertFalse(receipt["captured"])

    def test_answer_accepts_a_class_based_llm(self):
        class _LLM:
            def complete(self, prompt, *, system=None, **kwargs):
                return f"llm:{prompt}"

        system = System(SystemConfig(teacher=_LLM()))
        reply, _ = system.answer(Query("hi"))
        self.assertEqual(reply, "llm:hi")

    def test_answer_respects_an_explicit_budget(self):
        system = System(SystemConfig(teacher=_fake_teacher, default_budget=1))
        _, receipt = system.answer(Query("x"), budget=5)
        self.assertEqual(receipt["budget"], 5)


class SystemReadOnlyAnswerTest(unittest.TestCase):
    """CRITICAL: answer(..., read_only=True) is the snapshot path evaluate() relies on -- it must be
    impossible for a read-only call to leave any trace a later improve() (or total_spend) could act on.
    This is what keeps a held-out scorecard from training on its own held-out set (see
    scorecard_test.py's EvaluateDoesNotLeakIntoTrainingTest)."""

    def test_read_only_answer_does_not_populate_the_harvest(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        reply, _ = system.answer(Query("secret-test-question"), read_only=True)
        self.assertEqual(reply, "answer to: secret-test-question")
        self.assertEqual(system._harvest, {})

    def test_read_only_answer_does_not_move_total_spend(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        system.answer(Query("x"), read_only=True)
        self.assertEqual(system.total_spend.to_dict()["frontier_calls"], 0)

    def test_read_only_answer_still_reports_its_hypothetical_spend_on_the_receipt(self):
        # the receipt must still show what the call WOULD have cost (evaluate()'s realized_cost sums
        # this) even though it is not accumulated into total_spend
        system = System(SystemConfig(teacher=_fake_teacher))
        _, receipt = system.answer(Query("x"), read_only=True)
        self.assertEqual(receipt["spend"]["frontier_calls"], 1)
        self.assertEqual(receipt["total_spend"]["frontier_calls"], 0)
        self.assertTrue(receipt["read_only"])

    def test_a_read_only_answer_cannot_be_promoted_by_a_later_improve(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        system.answer(Query("secret-test-question"), read_only=True)
        report = system.improve(1000)
        self.assertEqual(report["status"], "nothing_to_improve")
        self.assertEqual(system._captured, {})

    def test_a_read_only_answer_leaves_no_trace_for_a_later_real_answer(self):
        # a read-only pass must not even leave a trace that changes a later NORMAL call's behavior
        calls = {"n": 0}

        def counting_teacher(prompt):
            calls["n"] += 1
            return f"answer to: {prompt}"

        system = System(SystemConfig(teacher=counting_teacher))
        system.answer(Query("q"), read_only=True)
        system.answer(Query("q"), read_only=True)
        self.assertEqual(calls["n"], 2)  # no caching kicked in from the read-only calls

        reply, receipt = system.answer(Query("q"))  # first REAL call
        self.assertEqual(calls["n"], 3)
        self.assertFalse(receipt["captured"])

    # negative control: a NORMAL (non-read_only) call must still harvest and be promotable -- read_only
    # must only suppress learning for calls that explicitly opt in, not disable capture altogether
    def test_normal_answer_still_harvests_and_is_later_promotable(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        system.answer(Query("public-question"))
        self.assertIn(("public-question", "", "local"), system._harvest)

        report = system.improve(10)
        self.assertEqual(report["status"], "captured")
        self.assertIn(("public-question", "", "local"), system._captured)


class SystemSpendLedgerTest(unittest.TestCase):
    """CARD SPEND-a: budget is a hard ceiling; every call's cost accumulates into System.total_spend."""

    def test_total_spend_accumulates_across_calls(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        for _ in range(3):
            system.answer(Query("x"))
        self.assertEqual(system.total_spend.to_dict()["frontier_calls"], 3)

    def test_over_budget_request_is_refused_not_silently_served(self):
        calls = {"n": 0}

        def counting_teacher(prompt):
            calls["n"] += 1
            return f"answer to: {prompt}"

        system = System(SystemConfig(teacher=counting_teacher))
        reply, receipt = system.answer(Query("x"), budget=0)

        self.assertIsNone(reply)
        self.assertEqual(calls["n"], 0)  # the teacher was never called -- no silent overspend
        self.assertEqual(receipt["status"], "refused")
        self.assertEqual(receipt["shortfall"], 1.0)
        self.assertEqual(receipt["spend"], {"frontier_calls": 0, "oracle_calls": 0, "wall_ms": 0.0, "dollars": 0.0})
        self.assertEqual(system.total_spend.to_dict()["frontier_calls"], 0)

    def test_a_refusal_does_not_perturb_a_later_successful_calls_running_total(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        system.answer(Query("a"))
        system.answer(Query("b"), budget=0)  # refused; must not silently count against total_spend
        system.answer(Query("c"))
        self.assertEqual(system.total_spend.to_dict()["frontier_calls"], 2)

    def test_receipt_carries_both_incremental_and_running_spend(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        system.answer(Query("a"))
        _, receipt = system.answer(Query("b"))
        self.assertEqual(receipt["spend"]["frontier_calls"], 1)
        self.assertEqual(receipt["total_spend"]["frontier_calls"], 2)


class SystemFaultModesTest(unittest.TestCase):
    """CARD FAULT-a: teacher_down / store_down degrade to a named, flagged mode -- never silently."""

    def _broken_teacher(self, prompt: str) -> str:
        raise ConnectionError("teacher endpoint unreachable")

    def test_teacher_down_falls_back_to_store_only_and_flags_it(self):
        store = Substrate()
        store.put(SubstrateItem(kind="text", text="the rollout finished on schedule"))
        system = System(SystemConfig(teacher=self._broken_teacher, store=store))

        reply, receipt = system.answer(Query("rollout status"))
        self.assertIn("degraded: store-only", reply)
        self.assertIn("rollout finished on schedule", reply)
        self.assertEqual(receipt["status"], "answered")
        self.assertEqual(receipt["degraded_mode"], "teacher_down")
        self.assertIn("teacher endpoint unreachable", receipt["degraded_reason"])
        self.assertEqual(receipt["produced_by"], "store")
        # a degraded, store-only answer didn't actually spend a frontier call
        self.assertEqual(receipt["spend"]["frontier_calls"], 0)

    def test_teacher_down_with_no_usable_store_fails_honestly(self):
        system = System(SystemConfig(teacher=self._broken_teacher))
        reply, receipt = system.answer(Query("rollout status"))
        self.assertIsNone(reply)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("teacher unavailable", receipt["reason"])

    def test_store_down_falls_back_to_no_accumulation_and_flags_it(self):
        class _BrokenStore:
            """Every method a store might be asked for raises the SAME unreachable error -- ingest's
            KNOW-a path calls .all() (via assimilate's lookup) before it ever reaches .put()."""

            def put(self, item):
                raise OSError("store unreachable")

            def all(self, *args, **kwargs):
                raise OSError("store unreachable")

            def get(self, *args, **kwargs):
                raise OSError("store unreachable")

        system = System(SystemConfig(teacher=_fake_teacher, store=_BrokenStore()))
        report = system.ingest("the sky is blue", source={"model": "teacher-v1"})

        self.assertEqual(report["status"], "degraded_no_accumulation")
        self.assertFalse(report["assimilated"])
        self.assertEqual(report["degraded_mode"], "store_down")
        self.assertIn("store unreachable", report["degraded_reason"])


class SystemFallbackScopeTest(unittest.TestCase):
    """HIGH: the teacher_down fallback must retrieve using the QUERY's own scope, never
    SystemConfig.scope -- a degraded answer must not silently cross a tenant/evidence scope boundary
    the query itself did not declare, exactly on the path a caller is least likely to double check."""

    def _broken_teacher(self, prompt: str) -> str:
        raise ConnectionError("teacher endpoint unreachable")

    def test_fallback_retrieves_from_the_querys_scope_not_the_configs_default(self):
        store = Substrate()
        store.put(SubstrateItem(kind="text", text="query-scope-secret-material", scope="query-scope"))
        store.put(SubstrateItem(kind="text", text="config-scope-public-material", scope="config-scope"))
        system = System(SystemConfig(teacher=self._broken_teacher, store=store, scope="config-scope"))

        reply, receipt = system.answer(Query("status", scope="query-scope"))

        self.assertIn("query-scope-secret-material", reply)
        self.assertNotIn("config-scope-public-material", reply)
        self.assertEqual(receipt["degraded_mode"], "teacher_down")

    def test_fallback_never_crosses_into_the_configs_scope_even_when_only_it_has_content(self):
        # the sharpest version of the bug: nothing lives in the query's own scope, so the OLD code would
        # silently succeed by reading config-scope material instead; the fixed code must fail honestly
        store = Substrate()
        store.put(SubstrateItem(kind="text", text="config-scope-only-material", scope="config-scope"))
        system = System(SystemConfig(teacher=self._broken_teacher, store=store, scope="config-scope"))

        reply, receipt = system.answer(Query("status", scope="query-scope"))

        self.assertIsNone(reply)
        self.assertEqual(receipt["status"], "failed")

    # negative control: when query.scope == config.scope (the common/default case, also exercised by
    # SystemFaultModesTest above) the fallback must still retrieve normally -- the fix must not simply
    # break retrieval across the board
    def test_fallback_still_retrieves_when_query_and_config_scope_coincide(self):
        store = Substrate()
        store.put(SubstrateItem(kind="text", text="the rollout finished on schedule", scope="team-x"))
        system = System(SystemConfig(teacher=self._broken_teacher, store=store, scope="team-x"))

        reply, receipt = system.answer(Query("rollout status", scope="team-x"))
        self.assertIn("rollout finished on schedule", reply)
        self.assertEqual(receipt["degraded_mode"], "teacher_down")


class SystemIngestTest(unittest.TestCase):
    def test_ingest_with_no_store_is_an_honest_noop(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        report = system.ingest("the sky is blue", source={"model": "teacher-v1"})
        self.assertEqual(report["status"], "no_store")
        self.assertFalse(report["assimilated"])

    def test_ingest_assimilates_via_the_belief_store_when_it_is_importable(self):
        """The primary path now that KNOW-a is built: a real claim, assimilated with real credence."""
        store = Substrate()
        system = System(SystemConfig(teacher=_fake_teacher, store=store))
        report = system.ingest("the sky is blue", source={"model": "teacher-v1"})

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["n_claims"], 1)
        self.assertEqual(len(report["items"]), 1)

    def test_ingest_falls_back_to_a_retrievable_substrate_item_when_no_belief_store_exists(self):
        """The defensive path SYS-a documents ('never a hard import of a card that may not be built
        yet'): still real and worth covering even though mixle.substrate.belief is always importable
        in THIS repo now -- simulate the import genuinely failing, the way it would for a caller
        missing that optional piece."""
        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "mixle.substrate.belief" or name.startswith("mixle.substrate.belief."):
                raise ImportError(f"simulated: {name} not installed")
            return real_import(name, *args, **kwargs)

        store = Substrate()
        system = System(SystemConfig(teacher=_fake_teacher, store=store))
        with patch("builtins.__import__", side_effect=_blocked_import):
            report = system.ingest("the sky is blue", source={"model": "teacher-v1"})

        self.assertEqual(report["status"], "ok_fallback")
        item = store.get(report["item_id"])
        self.assertIsNotNone(item)
        self.assertEqual(item.text, "the sky is blue")
        self.assertEqual(item.provenance, {"model": "teacher-v1"})


class SystemImproveTest(unittest.TestCase):
    def test_improve_on_an_empty_system_is_honest(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        report = system.improve(10)
        self.assertEqual(report["status"], "nothing_to_improve")
        self.assertEqual(report["budget"], 10)


class SystemColdStartCaptureTest(unittest.TestCase):
    """CARD SEED-a: from an empty system, answer then improve, and the second identical answer is free."""

    def _counting_teacher(self):
        calls = {"n": 0}

        def teacher(prompt):
            calls["n"] += 1
            return f"answer to: {prompt}"

        return teacher, calls

    def test_second_identical_query_after_improve_costs_no_frontier_calls(self):
        teacher, calls = self._counting_teacher()
        system = System(SystemConfig(teacher=teacher))
        query = Query("what is the capital of Freedonia?")

        reply1, receipt1 = system.answer(query)
        self.assertEqual(calls["n"], 1)
        self.assertFalse(receipt1["captured"])

        report = system.improve(10)
        self.assertEqual(report["status"], "captured")
        self.assertEqual(report["n_captured"], 1)

        reply2, receipt2 = system.answer(query)
        self.assertEqual(calls["n"], 1)  # no new frontier call -- served from the captured cache
        self.assertEqual(reply2, reply1)
        self.assertTrue(receipt2["captured"])
        self.assertEqual(receipt2["produced_by"], "captured")
        self.assertEqual(receipt2["spend"], {"frontier_calls": 0, "oracle_calls": 0, "wall_ms": 0.0, "dollars": 0.0})

    def test_repeat_query_before_improve_still_pays_for_a_fresh_teacher_call(self):
        teacher, calls = self._counting_teacher()
        system = System(SystemConfig(teacher=teacher))
        query = Query("what is the capital of Freedonia?")
        system.answer(query)
        system.answer(query)
        self.assertEqual(calls["n"], 2)

    def test_capture_is_specific_to_the_captured_query_text(self):
        teacher, calls = self._counting_teacher()
        system = System(SystemConfig(teacher=teacher))
        system.answer(Query("query one"))
        system.improve(10)
        system.answer(Query("query two"))  # a different query -- still a real teacher call
        self.assertEqual(calls["n"], 2)

    def test_improve_with_nothing_harvested_yet_is_still_honest(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        report = system.improve(10)
        self.assertEqual(report["status"], "nothing_to_improve")

    def test_same_text_different_task_does_not_share_a_captured_cache_entry(self):
        # regression: two queries with identical text but a different task (or scope) are different
        # questions and must not silently answer one from the other's captured cache
        teacher, calls = self._counting_teacher()
        system = System(SystemConfig(teacher=teacher))
        system.answer(Query("classify this", task="sentiment"))
        system.improve(10)
        system.answer(Query("classify this", task="topic"))
        self.assertEqual(calls["n"], 2)

    def test_same_text_different_scope_does_not_share_a_captured_cache_entry(self):
        teacher, calls = self._counting_teacher()
        system = System(SystemConfig(teacher=teacher))
        system.answer(Query("classify this", scope="team-a"))
        system.improve(10)
        system.answer(Query("classify this", scope="team-b"))
        self.assertEqual(calls["n"], 2)


class SystemImproveBudgetTest(unittest.TestCase):
    """HIGH: improve() must actually spend its stated budget -- promoting every harvested pair
    regardless of budget makes the parameter a no-op and silently defeats any caller trying to cap
    improvement cost."""

    def _harvest_distinct(self, system, n):
        for i in range(n):
            system.answer(Query(f"q{i}"))

    def test_budget_smaller_than_the_harvest_promotes_only_what_it_affords(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        self._harvest_distinct(system, 5)
        self.assertEqual(len(system._harvest), 5)

        report = system.improve(2)

        self.assertEqual(report["n_captured"], 2)
        self.assertEqual(len(system._captured), 2)
        self.assertEqual(len(system._harvest), 3)  # the rest stay harvested for a later, funded call
        self.assertEqual(report["realized_spend"], 2)
        self.assertEqual(report["budget"], 2)
        # documented priority order: harvest (answer) order, oldest first
        self.assertEqual({k[0] for k in system._captured}, {"q0", "q1"})

    def test_zero_budget_promotes_nothing(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        self._harvest_distinct(system, 1)

        report = system.improve(0)

        self.assertEqual(report["n_captured"], 0)
        self.assertEqual(report["status"], "insufficient_budget")
        self.assertEqual(system._captured, {})
        self.assertEqual(len(system._harvest), 1)  # untouched, not silently dropped

    def test_negative_budget_is_rejected_outright(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        self._harvest_distinct(system, 1)

        with self.assertRaises(ValueError):
            system.improve(-1)
        # rejected outright -- no partial/implicit promotion happened on the way to raising
        self.assertEqual(system._captured, {})
        self.assertEqual(len(system._harvest), 1)

    def test_a_second_improve_call_can_finish_what_the_first_could_not_afford(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        self._harvest_distinct(system, 3)

        system.improve(1)
        self.assertEqual(len(system._captured), 1)
        report2 = system.improve(10)
        self.assertEqual(report2["n_captured"], 2)
        self.assertEqual(len(system._captured), 3)
        self.assertEqual(system._harvest, {})

    # negative control: a budget that comfortably covers the whole harvest still promotes everything,
    # exactly as the pre-fix behavior did in the common case -- the fix must not become more
    # restrictive than necessary when the budget was never actually a binding constraint
    def test_ample_budget_still_promotes_everything(self):
        system = System(SystemConfig(teacher=_fake_teacher))
        self._harvest_distinct(system, 3)

        report = system.improve(1000)

        self.assertEqual(report["status"], "captured")
        self.assertEqual(report["n_captured"], 3)
        self.assertEqual(len(system._captured), 3)
        self.assertEqual(system._harvest, {})


class SystemConfigFromEnvTest(unittest.TestCase):
    def test_from_env_requires_base_url_and_model(self):
        with self.assertRaises(ValueError) as ctx:
            SystemConfig.from_env()
        self.assertIn("MIXLE_TEACHER_BASE_URL", str(ctx.exception))


class IngestBoundaryTest(unittest.TestCase):
    def test_ingestion_stays_inside_the_configured_knowledge_scope(self):
        # MXR-080-1687: the belief path called assimilate() without its scope argument, so it wrote
        # under "local" no matter how the system was configured -- while the import-fallback path did
        # use config.scope, making the boundary depend on optional-module availability.
        substrate = Substrate()
        system = System(SystemConfig(teacher=lambda p: "unused", store=substrate, scope="tenant-A"))

        report = system.ingest("the sky is blue. water is wet.", source={"model": "teacher-v1"})

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["scope"], "tenant-A")
        self.assertTrue(report["items"])  # the assertion below is only meaningful with real items
        for item in report["items"]:
            self.assertEqual(item.scope, "tenant-A")  # was "local" regardless of config.scope

    def test_a_partial_write_is_reported_as_partial_not_as_no_accumulation(self):
        # MXR-080-1688: claims are assimilated one at a time with no transaction. When a later put()
        # raised, with_fallback() converted the whole operation to
        # status="degraded_no_accumulation", assimilated=False, even though earlier claims were
        # already stored -- inviting a retry that duplicates the committed evidence.
        substrate = Substrate()
        real_put = substrate.put
        state = {"calls": 0}

        def flaky_put(item):
            state["calls"] += 1
            if state["calls"] >= 2:
                raise RuntimeError("store unavailable")
            return real_put(item)

        substrate.put = flaky_put
        system = System(SystemConfig(teacher=lambda p: "unused", store=substrate, scope="tenant-A"))

        report = system.ingest("the sky is blue. water is wet. fire is hot.", source={"model": "t"})

        self.assertEqual(report["status"], "partial_accumulation")
        self.assertTrue(report["assimilated"])
        self.assertGreaterEqual(report["n_committed"], 1)
        self.assertEqual(len(report["committed_ids"]), report["n_committed"])
        self.assertLess(report["n_committed"], report["n_claims"])
        self.assertEqual(report["degraded_mode"], "store_down")

    def test_a_failure_before_any_commit_is_still_an_honest_no_accumulation(self):
        # negative control: when nothing was written, degraded_no_accumulation remains correct.
        substrate = Substrate()
        substrate.put = lambda item: (_ for _ in ()).throw(RuntimeError("store unavailable"))
        system = System(SystemConfig(teacher=lambda p: "unused", store=substrate, scope="tenant-A"))

        report = system.ingest("the sky is blue. water is wet.", source={"model": "t"})

        self.assertEqual(report["status"], "degraded_no_accumulation")
        self.assertFalse(report["assimilated"])


class TeacherAnswerContractTest(unittest.TestCase):
    def test_invalid_teacher_output_is_never_a_durable_answered_capability(self):
        # MXR-080-1689: teachers returning None or a dict were charged as successful frontier calls,
        # reported status="answered", harvested, promoted by improve() and then served from the
        # captured cache forever. In the None case the public return was indistinguishable from a
        # refusal or a failure unless every caller also read the receipt.
        for bad in (None, {"answer": "x"}, "", "   "):
            with self.subTest(reply=repr(bad)):
                system = System(SystemConfig(teacher=lambda p, _bad=bad: _bad))
                reply, receipt = system.answer(Query("q"), budget=5)
                self.assertIsNone(reply)
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(system.total_spend.total_units(), 0.0)
                promoted = system.improve(budget=5)
                self.assertEqual(promoted["n_captured"], 0)
                self.assertIsNone(system.answer(Query("q"), budget=5)[0])

    def test_a_valid_teacher_answer_still_flows_through_unchanged(self):
        system = System(SystemConfig(teacher=lambda p: "a real answer"))
        reply, receipt = system.answer(Query("q"), budget=5)
        self.assertEqual(reply, "a real answer")
        self.assertEqual(receipt["status"], "answered")


class BudgetIsAnExactCountTest(unittest.TestCase):
    """MXR-080-1902 (High): ``answer``/``improve`` read their budget ceiling with ``int(budget)``,
    which TRUNCATES rather than validates -- ``budget=1.9`` silently became a one-call ceiling,
    ``budget=True`` became one too (``bool`` is an ``int`` subclass), and ``improve(-0.5)`` truncated
    to ``0`` and returned an ordinary report where ``improve(-1)`` already raised. A budget is
    checked directly against ``Spend.total_units()``, so it is held to the same exact non-Boolean
    nonnegative count contract every ``Spend`` dimension is."""

    def _system(self):
        return System(SystemConfig(teacher=_fake_teacher))

    def test_answer_refuses_a_truncatable_or_boolean_budget(self):
        system = self._system()
        for bad in (1.9, True, False, -1, "5"):
            with self.subTest(budget=bad), self.assertRaisesRegex(ValueError, "budget must be an exact"):
                system.answer(Query("q"), budget=bad)
        # nothing was charged or harvested by the rejected calls
        self.assertEqual(system.total_spend.total_units(), 0.0)
        self.assertEqual(system._harvest, {})

    def test_answer_still_accepts_the_ordinary_integer_ceilings(self):
        # Negative control against guard overreach: 0 (a refusal the suite already asserts) and any
        # positive int are exactly the states the library legitimately produces.
        system = self._system()
        _, refused = system.answer(Query("a"), budget=0)
        self.assertEqual(refused["status"], "refused")
        _, served = system.answer(Query("b"), budget=5)
        self.assertEqual(served["status"], "answered")
        self.assertEqual(served["budget"], 5)
        _, default = system.answer(Query("c"))
        self.assertEqual(default["budget"], system.config.default_budget)

    def test_improve_refuses_a_truncatable_or_boolean_budget(self):
        system = self._system()
        system.answer(Query("a"))
        system.answer(Query("b"))
        harvested = dict(system._harvest)
        self.assertEqual(len(harvested), 2)
        for bad in (1.9, True, -0.5, -1):
            with self.subTest(budget=bad), self.assertRaisesRegex(ValueError, "improve budget must be an exact"):
                system.improve(bad)
        self.assertEqual(system._harvest, harvested, "a rejected improve() promoted something anyway")
        self.assertEqual(system._captured, {})

    def test_improve_still_accepts_zero_and_positive_integer_budgets(self):
        system = self._system()
        system.answer(Query("a"))
        self.assertEqual(system.improve(0)["n_captured"], 0)
        self.assertEqual(system.improve(1)["n_captured"], 1)


if __name__ == "__main__":
    unittest.main()
