"""Harness (R1) + domain templates (R2) + registry (R3): the deployable shell a tiny agent plugs into."""

import unittest

from mixle.substrate import Reasoner, Substrate
from mixle.substrate.act import Action
from mixle.substrate.harness import (
    Harness,
    find_harnesses,
    monitoring_harness,
    register_harness,
    support_triage_harness,
)


def _kb():
    s = Substrate()
    s.add(kind="text", text="Refunds are processed within 30 days of a written request.")
    return s


def _answerer(q, ctx):
    return ctx.splitlines()[0] if ctx else ""


class GatesTest(unittest.TestCase):
    def _harness(self, **kw):
        tickets = []

        def escalate(req, inv):
            tickets.append(req)
            return f"ticket-{len(tickets)}"

        h = support_triage_harness(_kb(), _answerer, escalate=escalate, **kw)
        return h, tickets

    def test_valid_request_is_answered_with_evidence(self):
        h, _ = self._harness()
        r = h.handle("when are refunds processed")
        self.assertEqual(r.status, "answered")
        self.assertIn("30 days", r.answer)
        self.assertIsNotNone(r.investigation)  # the evidence trail travels

    def test_schema_refuses_before_any_model_runs(self):
        h, tickets = self._harness()
        self.assertEqual(h.handle("").status, "refused")
        self.assertEqual(h.handle("x" * 3000).status, "refused")
        self.assertEqual(tickets, [])  # nothing escalated; nothing ran

    def test_input_guardrail_redacts_secrets(self):
        h, _ = self._harness()
        r = h.handle("my key is sk-abcdefghij1234567890XYZ please help with refunds")
        self.assertEqual(r.redactions, 1)  # the secret never reached an action

    def test_abstention_escalates_to_the_policy(self):
        h, tickets = self._harness()
        r = h.handle("what is the meaning of life")
        self.assertEqual(r.status, "escalated")
        self.assertEqual(r.answer, "ticket-1")  # the handler's ticket comes back
        self.assertEqual(tickets, ["what is the meaning of life"])

    def test_whitelist_strips_disallowed_actions(self):
        # triage is retrieve-only: a compute action attached to the reasoner is structurally removed
        reasoner = Reasoner(_answerer, substrate=_kb(), retrieve_min_score=0.2)
        reasoner.add_action(Action("c", "compute", run=lambda q: ["x"], description="compute stuff"))
        h = Harness(reasoner, name="t", allowed_kinds=("retrieve",))
        self.assertEqual({a.kind for a in h.reasoner.actions}, {"retrieve"})

    def test_output_guardrail_redacts_answers(self):
        # a stored secret must not leave through the answer
        s = Substrate()
        s.add(kind="text", text="the refunds api uses token sk-abcdefghij1234567890XYZ internally")
        h = Harness(Reasoner(_answerer, substrate=s, retrieve_min_score=0.2), name="t")
        r = h.handle("refunds api token")
        self.assertEqual(r.status, "answered")
        self.assertNotIn("sk-abcdefghij1234567890XYZ", r.answer)

    def test_ui_hook_sees_every_result_and_cannot_break_the_path(self):
        seen = []

        def hook(res):
            seen.append(res.status)
            raise RuntimeError("ui crashed")  # must not propagate

        h = Harness(Reasoner(_answerer, substrate=_kb(), retrieve_min_score=0.2), name="t", on_result=hook)
        r = h.handle("when are refunds processed")
        self.assertEqual(r.status, "answered")
        self.assertEqual(seen, ["answered"])


class TemplatesAndRegistryTest(unittest.TestCase):
    def test_monitoring_template_allows_compute_and_simulate(self):
        reasoner = Reasoner(_answerer, substrate=_kb(), retrieve_min_score=0.2)
        reasoner.add_action(Action("c", "compute", run=lambda q: ["x"], description="check"))
        reasoner.add_action(Action("d", "delegate", run=lambda q: ["y"], description="remote"))
        h = monitoring_harness(reasoner)
        kinds = {a.kind for a in h.reasoner.actions}
        self.assertIn("compute", kinds)
        self.assertNotIn("delegate", kinds)  # no delegation out of a monitoring shell

    def test_register_and_find_on_the_substrate(self):
        s = _kb()
        h = support_triage_harness(s, _answerer)
        register_harness(s, h, scope="teamA")
        found = find_harnesses(s, "support")
        self.assertEqual(found[0]["harness"], "support-triage")
        self.assertEqual(found[0]["scope"], "teamA")  # P-scoped, shareable under governance
        self.assertEqual(find_harnesses(s, "nonexistent-topic"), [])


if __name__ == "__main__":
    unittest.main()
