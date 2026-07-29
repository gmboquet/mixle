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


def _flat_shard(n: int):
    """``n`` unlinked text items that all lexically match "widget" equally (score 1.0, no ties to break
    unpredictably) -- small enough (<8) to stay on the deterministic lexical path, no torch required."""
    s = Substrate()
    for i in range(n):
        s.add("text", f"item {i} about a shared widget topic", id=f"flat_{i}")
    return s


def _link_chain_shard(n: int):
    """A pure LINK chain ``c0 -> c1 -> ... -> c_{n-1}`` (each item links only to the next). Every item's
    text is a single token unique to itself, so CONTENT hops cannot shortcut the chain (zero lexical
    overlap between nodes) -- the only way from ``c0`` to ``c_{n-1}`` is ``n - 1`` LINK hops, one per
    depth, which makes this a deterministic probe for "does the cap hold at every expansion step"."""
    s = Substrate()
    ids = [f"c{i}" for i in range(n)]
    for i in range(n):
        s.put(
            SubstrateItem(
                kind="text",
                text=f"solokeyword{i}",
                links=[ids[i + 1]] if i + 1 < n else [],
                id=ids[i],
            )
        )
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
    @unittest.skipUnless(_HAS_TORCH, "30 items crosses into semantic retrieval, which needs the represent embedder")
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


class ValidationTest(unittest.TestCase):
    """MXR-080-0252: hop/seed/branch/item counts are exact non-negative bounds, and the item cap binds
    seed selection as tightly as it binds hop expansion."""

    def test_seeds_exceeding_max_items_capped_before_hopping(self):
        """Audit scenario: seeds > max_items must not blow the cap -- with max_hops=0 no hop expansion
        runs at all, so any overflow could only have come from seed insertion itself."""
        s = _flat_shard(6)
        chain = multihop(s, "widget", max_hops=0, seeds=6, max_items=2)
        self.assertEqual(len(chain), 2)  # truncated to the cap, not the full 6 seed matches

    def test_seed_cap_boundary_is_exact(self):
        """The cap is an exact boundary: a seed set exactly at max_items keeps every match, and a cap
        one below the natural seed count drops exactly one -- not an off-by-one in either direction."""
        s = _flat_shard(6)
        at_cap = multihop(s, "widget", max_hops=0, seeds=6, max_items=6)
        self.assertEqual(len(at_cap), 6)
        one_under = multihop(s, "widget", max_hops=0, seeds=6, max_items=5)
        self.assertEqual(len(one_under), 5)

    def test_cap_enforced_across_every_expansion_step(self):
        """Audit scenario: the cap must hold at EVERY hop, not just be checked once at the start.

        A 7-node pure LINK chain with branch=1 and max_hops=10 would, uncapped, walk all the way to
        depth 6 (7 items total). max_items=4 must stop it after exactly 3 hops out from the seed.
        """
        s = _link_chain_shard(7)
        chain = multihop(s, "solokeyword0", max_hops=10, seeds=1, branch=1, max_items=4)
        self.assertEqual(len(chain), 4)
        self.assertEqual(chain.max_depth(), 3)  # seed + 3 link hops, not the 6 the chain could reach

    def test_negative_branch_rejected(self):
        s = _lineage_shard()
        with self.assertRaises(ValueError):
            multihop(s, "checkout price is wrong", branch=-1)

    def test_negative_seeds_rejected(self):
        s = _lineage_shard()
        with self.assertRaises(ValueError):
            multihop(s, "checkout price is wrong", seeds=-1)

    def test_negative_max_hops_rejected(self):
        s = _lineage_shard()
        with self.assertRaises(ValueError):
            multihop(s, "checkout price is wrong", max_hops=-1)

    def test_negative_max_items_rejected(self):
        s = _lineage_shard()
        with self.assertRaises(ValueError):
            multihop(s, "checkout price is wrong", max_items=-1)

    def test_fractional_counts_rejected(self):
        s = _lineage_shard()
        for name in ("max_hops", "seeds", "branch", "max_items"):
            with self.subTest(param=repr(name)), self.assertRaises(TypeError):
                multihop(s, "checkout price is wrong", **{name: 1.5})

    def test_boolean_counts_rejected(self):
        s = _lineage_shard()
        for name in ("max_hops", "seeds", "branch", "max_items"):
            with self.subTest(param=repr(name)), self.assertRaises(TypeError):
                multihop(s, "checkout price is wrong", **{name: True})

    def test_valid_traversal_still_works(self):
        """Control: a genuine, well-formed multi-hop call is unaffected by the added validation and
        cap-during-seeding enforcement -- same two-hop lineage chain as ChainTest, asserted end to end."""
        s = _lineage_shard()
        chain = multihop(s, "checkout price is wrong", max_hops=2, seeds=2, branch=2, max_items=12)
        ids = [st.item.id for st in chain.steps]
        self.assertEqual(ids, ["doc_x", "art_x", "trc_x"])
        self.assertEqual(chain.max_depth(), 2)
        self.assertEqual([st.via for st in chain.steps], ["seed", "link", "link"])


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
