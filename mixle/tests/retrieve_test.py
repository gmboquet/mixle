"""All-data retrieval (S1): planned cross-kind retrieval that spans modalities."""

import unittest

from mixle.substrate import ContextBudget, Substrate, retrieve
from mixle.telemetry import Telemetry


def _mixed_shard():
    s = Substrate()
    s.add("text", "refunds are processed within 30 days for defective items", tags=["policy"])
    s.add("text", "refund requests over 500 dollars need finance approval", tags=["policy"])
    s.add("text", "the office refund desk is on the third floor")
    s.add(
        "artifact",
        "refund-router solve classifier",
        payload={"ref": "/reg/refund-router"},
        provenance={"source": "registry"},
    )
    s.add("trace", "refund request 900 dollars => finance-escalation", provenance={"source": "harvested"})
    s.add("trace", "refund defective item => billing", provenance={"source": "harvested"})
    return s


class DiversifyTest(unittest.TestCase):
    def test_retrieve_spans_multiple_kinds(self):
        s = _mixed_shard()
        r = retrieve(s, "how do we handle refunds", k=3, diversify=True)
        self.assertGreaterEqual(len(r.kinds()), 2)  # not one kind crowding out the rest
        self.assertLessEqual(len(r), 3)

    def test_flat_merge_can_be_dominated_by_one_kind(self):
        s = _mixed_shard()
        flat = retrieve(s, "refund", k=3, diversify=False)
        div = retrieve(s, "refund", k=3, diversify=True)
        # diversified spans at least as many kinds as the flat merge
        self.assertGreaterEqual(len(div.kinds()), len(flat.kinds()))

    def test_by_kind_groups(self):
        s = _mixed_shard()
        r = retrieve(s, "refund", k=6)
        grouped = r.by_kind()
        self.assertTrue(set(grouped).issubset({"text", "artifact", "trace"}))
        self.assertEqual(sum(len(v) for v in grouped.values()), len(r))


class WeightsAndScopeTest(unittest.TestCase):
    def test_weights_favor_a_kind(self):
        s = _mixed_shard()
        r = retrieve(s, "refund", k=6, weights={"artifact": 5.0})
        self.assertEqual(r.items[0].kind, "artifact")  # boosted to the top

    def test_kinds_filter(self):
        s = _mixed_shard()
        r = retrieve(s, "refund", k=6, kinds=["trace"])
        self.assertTrue(all(i.kind == "trace" for i in r.items))

    def test_scope_filter(self):
        s = Substrate()
        s.add("text", "team-a refund note", scope="team-a")
        s.add("text", "team-b refund note", scope="team-b")
        r = retrieve(s, "refund", k=5, scope="team-a")
        self.assertTrue(all(i.scope == "team-a" for i in r.items))


class HandoffTest(unittest.TestCase):
    def test_to_context_builds_a_packet(self):
        s = _mixed_shard()
        r = retrieve(s, "how do we handle refunds", k=4)
        pkt = r.to_context(budget=ContextBudget(max_chars=300))
        self.assertGreaterEqual(len(pkt), 1)
        self.assertLessEqual(pkt.used_chars, 300)

    def test_provenance_carried(self):
        s = _mixed_shard()
        r = retrieve(s, "refund", k=4)
        prov = r.provenance()
        self.assertEqual(len(prov), len(r))
        self.assertTrue(all("kind" in p and "score" in p for p in prov))


class TelemetryTest(unittest.TestCase):
    def test_retrieve_emits_a_reason_event(self):
        s = _mixed_shard()
        tel = Telemetry()
        retrieve(s, "refund", k=3, telemetry=tel)
        events = list(tel.events(kind="reason"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].features["action"], "retrieve")
        self.assertIn("kinds_covered", events[0].outcome)


if __name__ == "__main__":
    unittest.main()
