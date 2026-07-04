"""G: the flagship triage app's claims, each checked — answer/escalate/refuse/redact/monitor/trust."""

import unittest

import numpy as np

from mixle.inference import create
from mixle.inference.nonparametric import ks_2samp
from mixle.pool import PoolJob, submit
from mixle.substrate import (
    Substrate,
    check_factuality,
    safe_text,
    scan_substrate,
    support_triage_harness,
)


def _app():
    sub = Substrate()
    sub.add(kind="text", text=safe_text("Refunds are processed within 30 days of a written request."))
    sub.add(kind="text", text=safe_text("Enterprise support is staffed 24/7; free-tier is business hours."))
    sub.add(kind="trace", text=safe_text("case 4411: internal token sk-abcdefghij1234567890XYZ rotated"))
    tickets = []

    def escalate(req, inv):
        tickets.append(req)
        return f"ticket-{len(tickets)}"

    h = support_triage_harness(sub, lambda q, ctx: ctx.splitlines()[0] if ctx else "", escalate=escalate)
    return sub, h, tickets


class FlagshipTriageTest(unittest.TestCase):
    def test_supported_question_is_answered_and_grounded(self):
        sub, h, _ = _app()
        r = h.handle("when are refunds processed")
        self.assertEqual(r.status, "answered")
        self.assertTrue(check_factuality(sub, r.answer).is_grounded())  # every claim cites the store

    def test_unsupported_question_escalates_never_guesses(self):
        _sub, h, tickets = _app()
        r = h.handle("what is the meaning of life")
        self.assertEqual(r.status, "escalated")
        self.assertEqual(tickets, ["what is the meaning of life"])

    def test_secret_in_request_is_redacted(self):
        _sub, h, _ = _app()
        r = h.handle("my key sk-abcdefghij1234567890XYZ — when are refunds processed")
        self.assertEqual(r.redactions, 1)

    def test_ingested_secret_never_indexed(self):
        sub, _h, _ = _app()
        self.assertEqual(scan_substrate(sub)["n_dirty"], 0)  # safe_text on ingest kept the store clean

    def test_certified_model_and_pool_rails(self):
        spend = [float(x) for x in np.random.RandomState(0).normal(50, 12, 300)]
        art = create(spend, seed=0)
        self.assertGreaterEqual(int(art.guarantee), 4)
        res = submit(PoolJob(run=lambda: 1, kind="verb", reason="test", est_cost=0.0))
        self.assertTrue(res.ok)

    def test_drift_monitor_separates_regimes(self):
        spend = [float(x) for x in np.random.RandomState(0).normal(50, 12, 400)]
        same = [float(x) for x in np.random.RandomState(1).normal(50, 12, 300)]
        shifted = [float(x) for x in np.random.RandomState(2).normal(70, 12, 300)]
        self.assertGreater(ks_2samp(spend, same).pvalue, 0.01)  # no false alarm
        self.assertLess(ks_2samp(spend, shifted).pvalue, 0.01)  # the shift trips


if __name__ == "__main__":
    unittest.main()
