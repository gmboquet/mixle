"""Planned multi-hop retrieval (S2): chain typed hops, keep the evidence path."""

import unittest

from mixle.substrate import ContextBudget, Substrate, SubstrateItem, multihop
from mixle.telemetry import Telemetry

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _lineage_shard():
    """A bug report -> (link) the model artifact -> (link) its training trace. The trace shares NO
    vocabulary with the bug query, so it is reachable ONLY by following links (a genuine hop chain)."""
    s = Substrate()
    trace = SubstrateItem(
        kind="trace",
        text="lineage record alpha bravo charlie delta echo foxtrot",
        provenance={"source": "harvested"},
        id="trc_x",
    )
    art = SubstrateItem(
        kind="artifact",
        text="pricing widget gizmo regressor",
        links=["trc_x"],
        payload={"ref": "/reg/x"},
        provenance={"source": "registry"},
        id="art_x",
    )
    doc = SubstrateItem(
        kind="text",
        text="users report the checkout price is sometimes wrong at high volume",
        links=["art_x"],
        provenance={"source": "tickets"},
        id="doc_x",
    )
    for it in (trace, art, doc):
        s.put(it)  # 3 text-bearing items -> deterministic lexical retrieval (no fuzzy embedder noise)
    return s


class ChainTest(unittest.TestCase):
    def test_chains_across_kinds_by_lineage(self):
        s = _lineage_shard()
        chain = multihop(s, "checkout price is wrong", max_hops=2, seeds=2, branch=2)
        ids = {st.item.id for st in chain.steps}
        self.assertIn("art_x", ids)  # reached the model by following the doc's link
        self.assertIn("trc_x", ids)  # reached the training trace by following the model's link
        self.assertEqual(chain.max_depth(), 2)  # a genuine two-hop chain

    def test_evidence_path_is_a_citable_trace(self):
        s = _lineage_shard()
        chain = multihop(s, "checkout price is wrong", max_hops=2, seeds=2, branch=2)
        path = chain.path_to("trc_x")
        self.assertEqual([p.id for p in path], ["doc_x", "art_x", "trc_x"])  # query -> doc -> model -> trace

    def test_link_hops_are_labeled(self):
        s = _lineage_shard()
        chain = multihop(s, "checkout price is wrong", max_hops=2, seeds=2, branch=2)
        art_step = next(st for st in chain.steps if st.item.id == "art_x")
        self.assertEqual(art_step.via, "link")
        self.assertEqual(art_step.parent_id, "doc_x")


class BudgetTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_TORCH, "30 items crosses the lexical->embedding retrieval threshold")
    def test_max_items_caps_the_chain(self):
        s = Substrate()
        for i in range(30):
            s.add("text", f"document {i} about a common shared topic widget")
        chain = multihop(s, "shared topic widget", max_hops=3, seeds=3, branch=3, max_items=6)
        self.assertLessEqual(len(chain), 6)

    def test_no_new_neighbors_stops_early(self):
        s = Substrate()
        s.add("text", "an isolated document about quokkas with no links or neighbors")
        s.add("text", "totally unrelated content about tax law")
        chain = multihop(s, "quokkas", max_hops=5, seeds=1, branch=2)
        self.assertLessEqual(chain.max_depth(), 2)  # nothing to chain to -> stops, does not spin

    def test_by_depth_grouping(self):
        s = _lineage_shard()
        chain = multihop(s, "checkout price is wrong", max_hops=2, seeds=2, branch=2)
        depths = chain.by_depth()
        self.assertIn(0, depths)  # seeds
        self.assertTrue(any(d > 0 for d in depths))  # and hops out


class HandoffTest(unittest.TestCase):
    def test_to_context(self):
        s = _lineage_shard()
        chain = multihop(s, "checkout price is wrong", max_hops=2)
        pkt = chain.to_context(budget=ContextBudget(max_chars=400))
        self.assertGreaterEqual(len(pkt), 1)

    def test_provenance_records_hop_kind_and_depth(self):
        s = _lineage_shard()
        chain = multihop(s, "checkout price is wrong", max_hops=2)
        prov = chain.provenance()
        self.assertTrue(all({"depth", "via", "parent"} <= set(p) for p in prov))


class TelemetryTest(unittest.TestCase):
    def test_emits_a_reason_event(self):
        s = _lineage_shard()
        tel = Telemetry()
        multihop(s, "checkout price is wrong", max_hops=2, telemetry=tel)
        events = list(tel.events(kind="reason"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].features["action"], "multihop")
        self.assertIn("reached_depth", events[0].outcome)


if __name__ == "__main__":
    unittest.main()
