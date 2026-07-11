"""ContextPacket + assembly-on-route (O2): budgeted, provenanced views of the substrate."""

import unittest

from mixle.substrate import ContextBudget, Substrate, assemble_context, compress_text, ingest_documents
from mixle.telemetry import Telemetry

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _corpus():
    return [
        "the mitochondria produces ATP energy in cellular respiration",
        "photosynthesis converts sunlight into chemical energy in plants",
        "glycolysis breaks down glucose to release usable energy",
        "the citric acid cycle oxidizes acetyl-CoA to make energy",
        "the moon orbits the earth every twenty seven days",
        "the stock market fell two percent on tuesday afternoon",
    ]


class BudgetTest(unittest.TestCase):
    def test_lexical_assembly_packs_within_budget(self):
        s = Substrate()  # <4 items -> lexical retrieval, deterministic (no embedder)
        s.add("text", "alpha beta gamma delta")
        s.add("text", "beta gamma epsilon")
        pkt = assemble_context(s, "beta gamma", budget=ContextBudget(max_chars=100, shape="passages"))
        self.assertLessEqual(pkt.used_chars, 100)
        self.assertGreaterEqual(len(pkt), 1)
        self.assertIn("beta", pkt.items[0].text)  # the most relevant item leads

    def test_tiny_budget_keeps_at_least_one_item(self):
        s = Substrate()
        s.add("text", "a very long document that easily exceeds a tiny character budget on its own")
        s.add("text", "another document about something else entirely unrelated here")
        pkt = assemble_context(s, "long document", budget=ContextBudget(max_chars=5))
        self.assertEqual(len(pkt), 1)  # always at least the single best item

    @unittest.skipUnless(_HAS_TORCH, "10 items crosses the lexical->embedding retrieval threshold")
    def test_item_cap_is_honored(self):
        s = Substrate()
        for i in range(10):
            s.add("text", f"document number {i} about widgets")
        pkt = assemble_context(s, "widgets", budget=ContextBudget(max_chars=10000, max_items=3))
        self.assertLessEqual(len(pkt), 3)


class ProvenanceTest(unittest.TestCase):
    def test_every_item_carries_provenance(self):
        s = Substrate()
        ingest_documents(s, ["cats are mammals", "dogs are mammals too"], source="animal facts")
        pkt = assemble_context(s, "mammals", budget=ContextBudget(max_chars=200))
        prov = pkt.provenance()
        self.assertEqual(len(prov), len(pkt))
        self.assertTrue(all(p["source"] == "animal facts" and "score" in p for p in prov))

    def test_render_shapes(self):
        s = Substrate()
        s.add("text", "the quick brown fox jumps")
        s.add("text", "quick foxes are clever")
        passages = assemble_context(s, "quick fox", budget=ContextBudget(shape="passages")).render()
        brief = assemble_context(s, "quick fox", budget=ContextBudget(shape="brief")).render(header=False)
        self.assertIn("[text:", passages)  # provenance-tagged passages
        self.assertTrue(brief.startswith("- "))  # bulleted brief


class TelemetryTest(unittest.TestCase):
    def test_assembly_emits_a_context_event(self):
        s = Substrate()
        s.add("text", "some relevant content about topic x")
        tel = Telemetry()
        assemble_context(s, "topic x", budget=ContextBudget(max_chars=200), telemetry=tel)
        events = list(tel.events(kind="context"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].features["budget_chars"], 200)
        self.assertIn("n_selected", events[0].outcome)


class CompressionTest(unittest.TestCase):
    """O3: receipted extractive compression -- fit more sources, measure what's kept."""

    def _shop(self):
        s = Substrate()
        s.add(
            "text",
            "The company was founded in 1998. Our headquarters are in Denver. "
            "The refund policy allows returns within 30 days of purchase. We have 200 employees.",
        )
        s.add(
            "text",
            "Shipping is handled by a third party. Orders ship in 2 business days. "
            "Refunds for defective items are processed immediately without a restocking fee.",
        )
        s.add("text", "Our mascot is a golden retriever named Max. He visits on Fridays.")
        return s

    def test_compression_covers_more_sources_within_budget(self):
        s = self._shop()
        plain = assemble_context(
            s, "refund policy defective items", budget=ContextBudget(max_chars=240), compress=False
        )
        comp = assemble_context(s, "refund policy defective items", budget=ContextBudget(max_chars=240), compress=True)
        self.assertLessEqual(comp.used_chars, 240)
        self.assertGreater(len(comp), len(plain))  # more sources fit once each is summarized
        self.assertTrue(comp.compressed)
        self.assertLess(comp.compression_ratio, 1.0)

    def test_preservation_receipt_keeps_relevant_content(self):
        s = self._shop()
        comp = assemble_context(s, "refund policy defective items", budget=ContextBudget(max_chars=240), compress=True)
        self.assertIn("refund", comp.render().lower())  # the query-relevant sentences survived
        self.assertGreaterEqual(min(comp.preservation()), 0.5)  # each item kept >= half its query terms

    def test_standalone_compressor_prefix_matches_morphology(self):
        out = compress_text("The sky is blue. Refunds are given within 30 days. Cats are cute.", "refund policy", 45)
        self.assertIn("refund", out.lower())  # 'refund' matches 'refunds' by prefix
        self.assertLessEqual(len(out), 45)

    def test_short_text_is_returned_unchanged(self):
        self.assertEqual(compress_text("brief note", "anything", 100), "brief note")


@unittest.skipUnless(_HAS_TORCH, "semantic retrieval needs the represent embedder")
class SemanticAssemblyTest(unittest.TestCase):
    def test_top_item_is_on_topic_and_budget_monotone(self):
        s = Substrate()
        ingest_documents(s, _corpus())
        bio = set(_corpus()[:4])
        big = assemble_context(s, "how do cells generate energy", budget=ContextBudget(max_chars=500))
        self.assertIn(big.items[0].text, bio)  # the single most relevant item is on-topic
        self.assertLessEqual(big.used_chars, 500)
        small = assemble_context(s, "how do cells generate energy", budget=ContextBudget(max_chars=90))
        self.assertLessEqual(len(small), len(big))  # a tighter budget never selects more
        self.assertLessEqual(small.used_chars, 90)


if __name__ == "__main__":
    unittest.main()
