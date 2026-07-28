"""L2: dashboards over receipts — the telemetry stream folded into one auditable summary."""

import unittest

from mixle.telemetry import Telemetry, dashboard, render_dashboard


def _stream():
    t = Telemetry()
    t.record("placement", features={"tflop": 8.0}, choice="pool", outcome={"cost": 2.5})
    t.record("placement", features={"tflop": 0.1}, choice="local", outcome={"cost": 0.0})
    t.record("reason", features={"action": "investigate"}, choice="answer", outcome={"spent": 1.0})
    t.record("reason", features={"action": "investigate"}, choice="abstain", outcome={"spent": 2.0})
    return t


class DashboardTest(unittest.TestCase):
    def test_folds_counts_choices_and_costs(self):
        d = dashboard(_stream())
        self.assertEqual(d["n_events"], 4)
        self.assertEqual(d["by_kind"], {"placement": 2, "reason": 2})
        self.assertEqual(d["choices"]["placement"], {"pool": 1, "local": 1})
        self.assertEqual(d["cost_total"], 5.5)  # cost + spent, summed

    def test_abstention_rate(self):
        self.assertEqual(dashboard(_stream())["abstention_rate"], 0.5)

    def test_no_reason_events_means_no_rate(self):
        t = Telemetry()
        t.record("placement", features={}, choice="local", outcome={"cost": 0.0})
        self.assertIsNone(dashboard(t)["abstention_rate"])  # honest None, not 0%

    def test_render_is_plain_markdown(self):
        md = render_dashboard(_stream())
        self.assertIn("# telemetry receipts", md)
        self.assertIn("abstention rate: 50.0%", md)
        self.assertIn("- pool: 1", md)

    def test_empty_stream(self):
        d = dashboard(Telemetry())
        self.assertEqual(d["n_events"], 0)
        self.assertEqual(d["cost_total"], 0.0)
        self.assertEqual(d["latency_total"], 0.0)

    def test_dollars_and_seconds_are_never_added_together(self):
        # MXR-080-1736: one accumulator took the first of cost/latency/spent, so a $1 cost and a
        # 2-second latency folded into the meaningless number 3 -- and an event carrying both had
        # its latency silently discarded.
        t = Telemetry()
        t.record("route", choice="a", outcome={"cost": 1.0})
        t.record("route", choice="a", outcome={"latency": 2.0})
        t.record("route", choice="a", outcome={"cost": 4.0, "latency": 8.0})
        d = dashboard(t)
        self.assertEqual(d["cost_total"], 5.0)
        self.assertEqual(d["n_costed"], 2)
        self.assertEqual(d["latency_total"], 10.0)
        self.assertEqual(d["n_latency"], 2)

    def test_unaggregatable_amounts_are_counted_not_summed(self):
        # MXR-080-1736: a single NaN produced an ordinary-looking dashboard with cost_total=NaN,
        # and cost=True counted as one dollar.
        t = Telemetry()
        t.record("route", choice="a", outcome={"cost": float("nan")})
        t.record("route", choice="a", outcome={"cost": float("inf")})
        t.record("route", choice="a", outcome={"cost": True})
        t.record("route", choice="a", outcome={"cost": -5.0})
        t.record("route", choice="a", outcome={"cost": 3.0})
        d = dashboard(t)
        self.assertEqual(d["cost_total"], 3.0)
        self.assertEqual(d["n_costed"], 1)
        self.assertEqual(d["n_unaggregatable"], 4)


if __name__ == "__main__":
    unittest.main()
