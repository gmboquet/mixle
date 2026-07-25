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


def _two_kind_shard():
    # >= 2 items per kind so a k=-1 bug's effect at the per-kind AND the merge stage stay
    # distinguishable from each other (a 1-item kind's "all but the last" is indistinguishable from
    # "none" -- see CountValidationTest.test_retrieve_diversify_and_flat_diverge_on_negative_k).
    s = Substrate()
    s.add("text", "refund policy alpha", tags=["policy"])
    s.add("text", "refund policy beta", tags=["policy"])
    s.add("artifact", "refund-router artifact one", payload={"ref": "/x"})
    s.add("artifact", "refund-router artifact two", payload={"ref": "/y"})
    return s


class CountValidationTest(unittest.TestCase):
    """MXR-080-0236: k (and Retrieval.top's n) must be an exact, non-negative int, with identical
    semantics across Substrate.search and retrieve()'s diversified and flat merge paths."""

    def test_search_rejects_negative_k(self):
        s = _two_kind_shard()
        with self.assertRaises(ValueError):
            s.search("refund", k=-1)

    def test_search_rejects_bool_k(self):
        s = _two_kind_shard()
        with self.assertRaises(TypeError):
            s.search("refund", k=True)

    def test_search_rejects_fractional_k(self):
        s = _two_kind_shard()
        with self.assertRaises(TypeError):
            s.search("refund", k=2.7)

    def test_search_k_zero_returns_nothing(self):
        s = _two_kind_shard()
        self.assertEqual(s.search("refund", k=0), [])

    def test_retrieve_diversify_and_flat_diverge_on_negative_k_pre_fix(self):
        """The audit's own reproduction: pre-fix, diversify's ``while len(merged) < k`` loop treated
        k=-1 as "already satisfied" and returned NOTHING, while the flat path's ``sorted(...)[:k]``
        fell through to Python's negative-index slicing and returned all but a trailing item -- two
        different silent wrong answers for the very same invalid input. Both must now raise the same
        error instead of disagreeing."""
        s = _two_kind_shard()
        with self.assertRaises(ValueError):
            retrieve(s, "refund", k=-1, diversify=True)
        with self.assertRaises(ValueError):
            retrieve(s, "refund", k=-1, diversify=False)

    def test_retrieve_rejects_bool_and_fractional_k(self):
        s = _two_kind_shard()
        with self.assertRaises(TypeError):
            retrieve(s, "refund", k=True)
        with self.assertRaises(TypeError):
            retrieve(s, "refund", k=1.5)

    def test_top_rejects_negative_n(self):
        s = _two_kind_shard()
        r = retrieve(s, "refund", k=4)
        with self.assertRaises(ValueError):
            r.top(-1)

    def test_top_rejects_bool_and_fractional_n(self):
        s = _two_kind_shard()
        r = retrieve(s, "refund", k=4)
        with self.assertRaises(TypeError):
            r.top(True)
        with self.assertRaises(TypeError):
            r.top(1.5)

    def test_top_n_zero_returns_nothing(self):
        s = _two_kind_shard()
        r = retrieve(s, "refund", k=4)
        self.assertEqual(r.top(0), [])


if __name__ == "__main__":
    unittest.main()
