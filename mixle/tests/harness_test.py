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

    def test_escalation_receives_the_redacted_request_not_the_raw_secret(self):
        # MXR-080-0271 bug 3: on abstention, the escalation callback used to receive the ORIGINAL,
        # secret-bearing request instead of the guardrail-redacted one.
        h, tickets = self._harness()
        r = h.handle("my key is sk-abcdefghij1234567890XYZ, what is the meaning of life")
        self.assertEqual(r.status, "escalated")
        self.assertEqual(len(tickets), 1)
        self.assertNotIn("sk-abcdefghij1234567890XYZ", tickets[0])
        self.assertIn("[REDACTED", tickets[0])

    def test_escalation_receives_a_redacted_investigation(self):
        # the callback's second argument (the Investigation) must be redacted too, not just the
        # request string -- partial evidence gathered before abstaining can itself carry a secret.
        seen_invs = []

        def escalate(req, inv):
            seen_invs.append(inv)
            return "ticket-1"

        reasoner = Reasoner(_answerer)
        reasoner.add_action(
            Action(
                "probe",
                "compute",
                run=lambda q: ["partial trace token sk-abcdefghij1234567890XYZ"],
                description="",
                base_score=0.05,
            )
        )
        h = Harness(reasoner, name="t", escalate=escalate, min_confidence=0.5)
        r = h.handle("totally unrelated question")
        self.assertEqual(r.status, "escalated")
        self.assertEqual(len(seen_invs), 1)
        self.assertTrue(all("sk-abcdefghij1234567890XYZ" not in f for f in seen_invs[0].evidence))

    def test_whitelist_strips_disallowed_actions(self):
        # triage is retrieve-only: a compute action attached to the reasoner is excluded from this
        # harness's own immutable action view (MXR-080-0271: the harness must not mutate the shared
        # reasoner to enforce this -- see the isolation/bypass tests below).
        reasoner = Reasoner(_answerer, substrate=_kb(), retrieve_min_score=0.2)
        reasoner.add_action(Action("c", "compute", run=lambda q: ["x"], description="compute stuff"))
        h = Harness(reasoner, name="t", allowed_kinds=("retrieve",))
        self.assertEqual({a.kind for a in h.actions}, {"retrieve"})
        # the shared reasoner itself is untouched: any other harness/caller still sees both actions
        self.assertEqual({a.kind for a in reasoner.actions}, {"retrieve", "compute"})

    def test_two_harnesses_sharing_a_reasoner_have_independent_whitelists(self):
        # MXR-080-0271 bug 1: the old code enforced the whitelist by mutating `reasoner._actions` in
        # __init__, so constructing a second harness over the same reasoner silently rewrote the
        # first harness's effective action space. A harness's `.actions` view must be immutable once
        # built, independent of any other harness sharing its reasoner.
        reasoner = Reasoner(_answerer, substrate=_kb(), retrieve_min_score=0.2)
        reasoner.add_action(Action("c", "compute", run=lambda q: ["x"], description="compute stuff"))
        h1 = Harness(reasoner, name="h1", allowed_kinds=("retrieve", "compute"))
        self.assertEqual({a.kind for a in h1.actions}, {"retrieve", "compute"})
        Harness(reasoner, name="h2", allowed_kinds=("compute",))  # built for its side effect only
        self.assertEqual({a.kind for a in h1.actions}, {"retrieve", "compute"})  # h1 unaffected by h2
        self.assertEqual({a.kind for a in reasoner.actions}, {"retrieve", "compute"})  # reasoner untouched

    def test_add_action_after_construction_cannot_bypass_an_existing_harnesss_whitelist(self):
        # MXR-080-0271 bug 2: add_action() on the shared reasoner, called AFTER a harness is built,
        # must not appear in that harness's whitelist view, and must never be reachable via handle().
        reasoner = Reasoner(_answerer, substrate=_kb(), retrieve_min_score=0.2)
        h = Harness(reasoner, name="t", allowed_kinds=("retrieve",))
        fired = []

        def leaky_run(q):
            fired.append(q)
            return ["leaked-action-executed"]

        reasoner.add_action(Action("leak", "delegate", run=leaky_run, description="refunds delegate leak"))
        self.assertEqual({a.kind for a in h.actions}, {"retrieve"})  # structurally still absent
        r = h.handle("refunds delegate leak")
        self.assertEqual(fired, [])  # never executed through this harness
        self.assertNotIn("leaked-action-executed", r.answer or "")

    def test_output_guardrail_redacts_answers(self):
        # a stored secret must not leave through the answer
        s = Substrate()
        s.add(kind="text", text="the refunds api uses token sk-abcdefghij1234567890XYZ internally")
        h = Harness(Reasoner(_answerer, substrate=s, retrieve_min_score=0.2), name="t")
        r = h.handle("refunds api token")
        self.assertEqual(r.status, "answered")
        self.assertNotIn("sk-abcdefghij1234567890XYZ", r.answer)

    def test_retained_investigation_is_redacted_even_when_evidence_bypasses_the_substrate(self):
        # MXR-080-0271 bug 4: the returned Investigation used to retain the RAW answer/evidence even
        # when the top-level answer was masked. Substrate.put() now redacts at the store boundary, so
        # this uses a compute action whose evidence never touches a substrate at all -- the harness
        # itself, not the store, must be the thing that redacts it.
        reasoner = Reasoner(_answerer)
        reasoner.add_action(
            Action(
                "diag",
                "compute",
                run=lambda q: ["live diagnostic dump: internal token sk-abcdefghij1234567890XYZ still valid"],
                description="diagnostic dump token",
            )
        )
        h = Harness(reasoner, name="t")
        r = h.handle("diagnostic dump token")
        self.assertEqual(r.status, "answered")
        self.assertNotIn("sk-abcdefghij1234567890XYZ", r.answer)
        # the retained trace must match what the top-level answer shows -- not the raw evidence beneath it
        self.assertIsNotNone(r.investigation.answer)
        self.assertNotIn("sk-abcdefghij1234567890XYZ", r.investigation.answer)
        self.assertTrue(all("sk-abcdefghij1234567890XYZ" not in f for f in r.investigation.evidence))

    def test_retained_investigation_redacts_a_secret_leaked_through_a_failed_actions_error(self):
        # Step.error/Investigation.error (MXR-080-0256) carry a raised exception's "Type: message" --
        # this postdates _redact_investigation's original fields (answer/fragments), so a secret echoed
        # back by a failing action's own error message was a live gap until this was covered too.
        reasoner = Reasoner(_answerer)

        def _boom(q):
            raise ValueError("could not process request containing sk-abcdefghij1234567890XYZ")

        reasoner.add_action(Action("diag", "compute", run=_boom, description="diagnostic"))
        h = Harness(reasoner, name="t")
        r = h.handle("diagnostic dump token")
        self.assertIsNotNone(r.investigation)
        step = r.investigation.steps[0]
        self.assertTrue(step.failed)
        self.assertNotIn("sk-abcdefghij1234567890XYZ", step.error)
        self.assertIn("ValueError", step.error)  # the error TYPE survives redaction, just not the secret

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
        kinds = {a.kind for a in h.actions}
        self.assertIn("compute", kinds)
        self.assertNotIn("delegate", kinds)  # no delegation out of a monitoring shell
        # the shared reasoner itself is untouched -- delegate is still there for any other caller
        self.assertIn("delegate", {a.kind for a in reasoner.actions})

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
