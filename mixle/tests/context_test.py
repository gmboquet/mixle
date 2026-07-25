"""ContextPacket + assembly-on-route (O2): budgeted, provenanced views of the substrate."""

import unittest
from pathlib import Path

from mixle.substrate import (
    ContextBudget,
    ReceiverProfile,
    Substrate,
    SubstrateItem,
    assemble_context,
    assemble_for_receivers,
    compress_text,
    ingest_documents,
)
from mixle.substrate.context import substrate_item_to_knowledge_dict
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

    def test_a_budget_too_small_even_for_the_header_is_rejected(self):
        """Regression (MXR-080-0239): this budget used to still silently return a 1-item packet whose
        actual rendering (header included) was many times over max_chars -- the "always keep >= 1
        item" guarantee previously overrode the hard budget instead of respecting it. No selection can
        make an arbitrarily long task string fit inside a 5-character budget, so this now raises
        immediately instead of doing retrieval work for a budget nothing could ever satisfy."""
        s = Substrate()
        s.add("text", "a very long document that easily exceeds a tiny character budget on its own")
        s.add("text", "another document about something else entirely unrelated here")
        with self.assertRaises(ValueError):
            assemble_context(s, "long document", budget=ContextBudget(max_chars=5))

    def test_tiny_but_feasible_budget_keeps_one_genuinely_truncated_item(self):
        """Once max_chars covers the header and the item's mandatory [kind:id] tag, assembly still
        yields the single most relevant item -- exactly as before -- but now genuinely shrunk to fit
        rather than included at full (possibly overflowing) length."""
        s = Substrate()
        s.add("text", "a very long document that easily exceeds a tiny character budget on its own")
        s.add("text", "another document about something else entirely unrelated here")
        pkt = assemble_context(s, "doc", budget=ContextBudget(max_chars=50))
        self.assertEqual(len(pkt), 1)  # always at least the single best item, whenever at all feasible
        self.assertEqual(pkt.used_chars, len(pkt.render()))  # derived, never an estimate
        self.assertLessEqual(pkt.used_chars, 50)
        # the TEXT USED in the packet is genuinely truncated -- item.text itself (the item's own,
        # untouched full surface) is naturally unaffected and stays long.
        self.assertLess(len(pkt.texts[0]), len(pkt.items[0].text))

    def test_max_chars_must_be_a_positive_whole_number(self):
        """Regression (MXR-080-0239): only max_items was validated -- negative, zero, fractional, and
        non-finite max_chars all sailed through construction and were only discovered (if at all) as a
        silently-overflowing rendering several calls later. bool is technically an int subclass in
        Python but is never a meaningful character budget either."""
        for bad in (-5, 0, 2.5, float("inf"), float("-inf"), float("nan"), True, "100"):
            with self.assertRaises((TypeError, ValueError)):
                ContextBudget(max_chars=bad)

    def test_max_items_must_be_a_whole_number(self):
        for bad in (2.5, float("inf"), float("nan")):
            with self.assertRaises((TypeError, ValueError)):
                ContextBudget(max_items=bad)

    def test_unknown_shape_is_rejected_at_construction(self):
        """Regression (MXR-080-0239): any shape other than "brief" silently fell through to the
        passages/features rendering rather than being validated against the documented closed set."""
        with self.assertRaises(ValueError):
            ContextBudget(shape="bogus")
        for ok in ("passages", "brief", "features"):
            ContextBudget(shape=ok)  # does not raise

    def test_assembled_rendering_always_fits_the_declared_budget(self):
        """The core MXR-080-0239 guarantee, swept from just-barely-feasible to generous: used_chars
        never exceeds max_chars, and never disagrees with what render() actually returns."""
        s = Substrate()
        s.add("text", "a very long document that easily exceeds a tiny character budget on its own")
        s.add("text", "another document about something else entirely unrelated here as well")
        for max_chars in (25, 43, 50, 75, 100, 300, 2000):
            pkt = assemble_context(s, "doc", budget=ContextBudget(max_chars=max_chars))
            self.assertEqual(pkt.used_chars, len(pkt.render()))
            self.assertLessEqual(pkt.used_chars, max_chars)

    def test_compression_per_item_share_is_not_floored_above_a_tight_total_budget(self):
        """Regression (MXR-080-0239): assemble_context's compress=True path used to floor each item's
        fair share of the budget at 40 characters even when the true per-item share (from a small
        overall budget) was smaller -- directly causing the rendering to exceed max_chars. Confirmed
        against the pre-fix code: this exact scenario (max_chars=80, 3 items, compress=True) reported
        used_chars=66 while actually rendering 87 characters, silently over budget."""
        s = Substrate()  # <4 items -> lexical retrieval, deterministic (no embedder)
        s.add("text", "document number zero about widgets and gadgets and things that are relevant to widgets")
        s.add("text", "document number one about widgets and gadgets and things that are relevant to widgets too")
        s.add("text", "document number two about widgets and gadgets and things that are relevant to widgets also")
        pkt = assemble_context(s, "widgets", budget=ContextBudget(max_chars=80, max_items=3), compress=True)
        self.assertEqual(pkt.used_chars, len(pkt.render()))
        self.assertLessEqual(pkt.used_chars, 80)

    @unittest.skipUnless(_HAS_TORCH, "10 items crosses into semantic retrieval, which needs the represent embedder")
    def test_item_cap_is_honored(self):
        s = Substrate()
        for i in range(10):
            s.add("text", f"document number {i} about widgets")
        pkt = assemble_context(s, "widgets", budget=ContextBudget(max_chars=10000, max_items=3))
        self.assertLessEqual(len(pkt), 3)

    def test_max_items_zero_is_rejected_at_construction(self):
        """Regression test: ContextBudget had no validation on max_items, so max_items=0 sailed
        through construction and only blew up later, as a ZeroDivisionError, inside
        assemble_context's compress=True path (n_target = min(0, len(hits)) == 0, then
        max_chars // n_target). Catching it at construction surfaces a clear error at the actual
        mistake, not a confusing crash several calls downstream."""
        with self.assertRaises(ValueError):
            ContextBudget(max_items=0)
        with self.assertRaises(ValueError):
            ContextBudget(max_items=-1)

    def test_compress_with_the_smallest_valid_item_cap_does_not_crash(self):
        s = Substrate()
        for i in range(5):
            s.add("text", f"document number {i} about widgets and gadgets")
        pkt = assemble_context(s, "widgets", budget=ContextBudget(max_chars=500, max_items=1), compress=True)
        self.assertEqual(len(pkt), 1)


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

    def test_output_never_exceeds_max_chars_even_when_the_top_sentence_alone_does(self):
        # The "always keep >= 1 sentence" fallback (so compression never returns empty) previously
        # never capped that first sentence to max_chars -- when the single most query-relevant
        # sentence alone was longer than the budget, the summary could be many times over it.
        text = (
            "The quarterly refund policy changes affect all customers who purchased items in the last "
            "thirty days and want a full refund immediately without any questions asked by support staff. "
            "Cats are cute. The sky is blue today."
        )
        out = compress_text(text, "refund policy", max_chars=20)
        self.assertLessEqual(len(out), 20)
        self.assertGreater(len(out), 0)

    def test_max_chars_0_is_rejected_not_silently_over_budget(self):
        """Regression (MXR-080-0239), the audit's own exact example: compress_text(..., max_chars=0)
        used to return 2 characters (one kept character plus a mandatory ellipsis) -- silently, and
        unboundedly relative to the declared budget of 0. Zero is not a positive budget; this must
        raise, not guess at a 2-character answer nobody asked for."""
        with self.assertRaises(ValueError):
            compress_text("The sky is blue. Refunds are given within 30 days. Cats are cute.", "refund policy", 0)

    def test_max_chars_must_be_a_positive_whole_number(self):
        for bad in (-10, 3.5, float("inf"), float("nan"), True):
            with self.assertRaises((TypeError, ValueError)):
                compress_text("some text of no particular importance here", "task", bad)

    def test_max_chars_1_spends_the_whole_budget_on_the_truncation_marker(self):
        """At the tightest possible positive budget there is no room for the ellipsis AND a kept
        character, so the whole budget (1 char) goes to the ellipsis alone -- never 2 characters."""
        out = compress_text("a much longer sentence than the budget allows by a very large margin", "task", 1)
        self.assertEqual(len(out), 1)

    def test_single_run_on_sentence_with_no_punctuation_still_respects_a_tiny_budget(self):
        """The len(sentences) <= 1 path (no '.', '!', '?', or newline to split on) had the same
        max(max_chars - 1, 1) + ellipsis floor as the multi-sentence path -- exercised separately here
        since it is a distinct branch in _compress."""
        out = compress_text("a single long run on clause with no terminal punctuation at all", "task", 3)
        self.assertLessEqual(len(out), 3)


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


class KnowledgePacketTransferTest(unittest.TestCase):
    """workstream E1/E3: substrate -> a mixle-knowledge-shaped ContextPacket dict -> a different receiver.

    mixle core has no dependency on the mixle-knowledge package (platform contracts depend on core, not
    the other way), so ``to_knowledge_dict`` produces a plain dict field-for-field matching
    ``mixle_knowledge.contracts.ContextPacket`` rather than importing it; the exact field set is pinned
    here as a regression guard against silent contract drift.
    """

    # mirrors mixle_knowledge.contracts.ContextPacket's fields exactly (created_at is defaulted there,
    # so it is intentionally absent from this dict -- everything else must round-trip).
    _KNOWLEDGE_CONTEXT_PACKET_FIELDS = {
        "id",
        "project_id",
        "task",
        "target_kind",
        "target_id",
        "token_budget",
        "byte_budget",
        "evidence_item_ids",
        "constraints",
        "citations",
        "expected_output_schema",
        "payload",
    }

    def _corpus_substrate(self):
        s = Substrate()
        ingest_documents(
            s,
            ["cats are mammals that purr", "dogs are mammals that bark", "the moon orbits the earth"],
            source="animal facts",
        )
        return s

    def test_dict_shape_matches_the_knowledge_contract_field_for_field(self):
        s = self._corpus_substrate()
        pkt = assemble_context(s, "mammals", budget=ContextBudget(max_chars=200))
        d = pkt.to_knowledge_dict(id="pkt1", project_id="proj1", target_kind="frontier_llm")
        self.assertEqual(set(d.keys()), self._KNOWLEDGE_CONTEXT_PACKET_FIELDS)
        self.assertEqual(d["task"], "mammals")
        self.assertEqual(d["evidence_item_ids"], [i.id for i in pkt.items])
        self.assertEqual(len(d["citations"]), len(pkt.items))
        for c in d["citations"]:
            self.assertIn("uri", c)  # the one required SourceRef field

    def test_factuality_receipt_travels_with_the_packet(self):
        from mixle.substrate.factuality import check_factuality

        s = self._corpus_substrate()
        pkt = assemble_context(s, "mammals", budget=ContextBudget(max_chars=200))
        receipt = check_factuality(s, "cats are mammals. cats can fly.")
        d = pkt.to_knowledge_dict(id="pkt2", project_id="proj1", target_kind="local_student", factuality=receipt)
        self.assertIn("factuality", d["payload"])
        self.assertEqual(d["payload"]["factuality"]["grounded_fraction"], receipt.grounded_fraction)
        self.assertLess(receipt.grounded_fraction, 1.0)  # the flight claim is not grounded -- a real receipt

    def test_no_factuality_argument_leaves_payload_without_it(self):
        s = self._corpus_substrate()
        pkt = assemble_context(s, "mammals", budget=ContextBudget(max_chars=200))
        d = pkt.to_knowledge_dict(id="pkt3", project_id="proj1", target_kind="frontier_llm")
        self.assertNotIn("factuality", d["payload"])

    def test_one_packet_two_receivers_get_different_task_conditioned_renderings(self):
        """The E acceptance: one packet, two receivers, per-receiver fidelity reported."""
        s = self._corpus_substrate()
        pkt_llm = assemble_context(s, "mammals", budget=ContextBudget(max_chars=500, shape="passages"))
        pkt_student = assemble_context(s, "mammals", budget=ContextBudget(max_chars=80, shape="features"))
        d_llm = pkt_llm.to_knowledge_dict(id="p_llm", project_id="proj1", target_kind="frontier_llm")
        d_student = pkt_student.to_knowledge_dict(id="p_student", project_id="proj1", target_kind="local_student")
        # same underlying task and substrate, genuinely different per-receiver renderings/budgets
        self.assertEqual(d_llm["task"], d_student["task"])
        self.assertNotEqual(d_llm["target_kind"], d_student["target_kind"])
        self.assertGreater(len(d_llm["payload"]["rendered"]), len(d_student["payload"]["rendered"]))
        self.assertLessEqual(d_student["byte_budget"], 80)
        self.assertLessEqual(d_llm["byte_budget"], 500)


class ReceiverConditionedCompressionTest(unittest.TestCase):
    """workstream E2: assemble_for_receivers budgets and shapes per named receiver in one call."""

    def _corpus_substrate(self):
        s = Substrate()
        ingest_documents(
            s,
            ["cats are mammals that purr", "dogs are mammals that bark", "the moon orbits the earth"],
            source="animal facts",
        )
        return s

    def _shop_substrate(self):
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
        return s

    def test_each_receiver_gets_its_own_budget_and_shape(self):
        s = self._shop_substrate()
        packets = assemble_for_receivers(
            s,
            "refund policy",
            [
                ReceiverProfile("frontier_llm", max_chars=500, shape="passages"),
                ReceiverProfile("local_student", max_chars=60, shape="features", compress=True),
            ],
        )
        self.assertEqual(set(packets), {"frontier_llm", "local_student"})
        llm, student = packets["frontier_llm"], packets["local_student"]
        self.assertLessEqual(llm.used_chars, 500)
        self.assertLess(student.used_chars, llm.used_chars)  # the tight-budget receiver gets far less text
        self.assertGreater(len(llm.render()), len(student.render()))
        self.assertFalse(llm.compressed)
        self.assertTrue(student.compressed)  # only the tight-budget receiver's profile asked for compression

    def test_matches_the_e1_to_knowledge_dict_round_trip_per_receiver(self):
        """Each receiver's packet feeds the E1 transfer contract independently, carrying its own target_kind."""
        s = self._corpus_substrate()
        packets = assemble_for_receivers(
            s,
            "mammals",
            [ReceiverProfile("frontier_llm", max_chars=500), ReceiverProfile("local_student", max_chars=60)],
        )
        d_llm = packets["frontier_llm"].to_knowledge_dict(id="p1", project_id="proj", target_kind="frontier_llm")
        d_student = packets["local_student"].to_knowledge_dict(id="p2", project_id="proj", target_kind="local_student")
        self.assertEqual(d_llm["task"], d_student["task"])
        self.assertNotEqual(d_llm["byte_budget"], d_student["byte_budget"])

    def test_receiver_profile_to_budget_matches_its_fields(self):
        profile = ReceiverProfile("x", max_chars=123, max_items=7, shape="brief")
        budget = profile.to_budget()
        self.assertEqual(budget.max_chars, 123)
        self.assertEqual(budget.max_items, 7)
        self.assertEqual(budget.shape, "brief")


class CanonicalHashTest(unittest.TestCase):
    """MXR-080-0238: the content-hash envelope's JSON encoding rejects anything outside a closed,
    type-aware canonical schema instead of silently stringifying it via ``default=str``."""

    def _item(self, payload):
        return SubstrateItem(id="x", kind="text", text="t", payload=payload)

    def test_a_set_in_the_payload_is_rejected_not_stringified(self):
        """A set has no canonical order -- default=str previously ran it through repr(), which is
        stable only within one process (Python randomizes string hashing per process by default, so
        the SAME logical set's iteration order -- and therefore its hash -- can differ run to run)."""
        with self.assertRaises(TypeError):
            substrate_item_to_knowledge_dict(self._item({"tags": {"a", "b", "c"}}))

    def test_a_path_in_the_payload_is_rejected_not_silently_treated_as_a_string(self):
        """default=str made a Path and the equivalent plain str hash identically, masking a real type
        distinction; the fix requires the caller to convert explicitly (str(path)), as ingest.py
        already does at its own payload-construction boundary. (Not the "ref"/"path" payload KEY,
        which substrate_item_to_knowledge_dict already explicitly str()-converts on its own via
        _as_artifact_ref before this ever runs -- a Path value under any OTHER key.)"""
        with self.assertRaises(TypeError):
            substrate_item_to_knowledge_dict(self._item({"source_file": Path("/data/a.csv")}))

    def test_a_plain_object_in_the_payload_is_rejected_not_stringified(self):
        """default=str on a plain object with no custom __repr__ embeds a process-specific memory
        address -- two semantically-identical (empty) instances would hash differently."""

        class Blob:
            pass

        with self.assertRaises(TypeError):
            substrate_item_to_knowledge_dict(self._item({"v": Blob()}))

    def test_non_finite_float_in_the_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            substrate_item_to_knowledge_dict(self._item({"v": float("nan")}))
        with self.assertRaises(ValueError):
            substrate_item_to_knowledge_dict(self._item({"v": float("inf")}))

    def test_json_native_payloads_still_hash_deterministically(self):
        """Positive control: ordinary JSON-native payloads (the only kind any current call site in
        this codebase actually constructs) are unaffected -- same content, same hash, every time."""
        payload = {"a": [1, 2, "x"], "b": None, "c": True, "d": {"nested": 1.5}}
        d1 = substrate_item_to_knowledge_dict(self._item(dict(payload)))
        d2 = substrate_item_to_knowledge_dict(self._item(dict(payload)))
        self.assertEqual(d1["content_hash"], d2["content_hash"])
        self.assertEqual(len(d1["content_hash"]), 64)  # sha256 hex

    def test_key_order_does_not_affect_the_hash(self):
        """Sorted-keys canonicalization: two dicts with the same content in a different insertion
        order must hash identically."""
        d1 = substrate_item_to_knowledge_dict(self._item({"a": 1, "b": 2}))
        d2 = substrate_item_to_knowledge_dict(self._item({"b": 2, "a": 1}))
        self.assertEqual(d1["content_hash"], d2["content_hash"])

    def test_non_string_dict_key_is_rejected(self):
        """JSON object keys are strings; silently coercing an int/float/bool/None key to its string
        form (as plain json.dumps would) risks two distinct keys colliding after coercion."""
        with self.assertRaises(TypeError):
            substrate_item_to_knowledge_dict(self._item({1: "x"}))


if __name__ == "__main__":
    unittest.main()
