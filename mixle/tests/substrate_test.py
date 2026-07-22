"""The knowledge substrate: typed provenanced items, cross-modal retrieval, scope, persistence."""

import json
import os
import tempfile
import unittest

from mixle.substrate import (
    MODALITIES,
    Substrate,
    SubstrateItem,
    ingest_artifacts,
    ingest_documents,
    ingest_traces,
)

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class SubstrateItemTest(unittest.TestCase):
    def test_rejects_unknown_modality(self):
        with self.assertRaises(ValueError):
            SubstrateItem(kind="hologram")

    def test_json_round_trip(self):
        item = SubstrateItem(kind="text", text="hello", tags=["a"], provenance={"src": "x"})
        back = SubstrateItem.from_json(item.to_json())
        self.assertEqual((back.kind, back.text, back.tags, back.id), (item.kind, item.text, item.tags, item.id))


class SubstrateCrudTest(unittest.TestCase):
    def test_put_get_remove_filter(self):
        s = Substrate()
        a = s.add("text", "a document about cats", tags=["animal"])
        s.add("record", payload={"kind": "refund", "amount": 900}, scope="team-1")
        self.assertEqual(len(s), 2)
        self.assertEqual(s.get(a).text, "a document about cats")
        self.assertEqual(len(s.all(kind="text")), 1)
        self.assertEqual(len(s.all(scope="team-1")), 1)
        self.assertTrue(s.remove(a))
        self.assertEqual(len(s), 1)

    def test_lexical_fallback_under_four_items(self):
        s = Substrate()
        s.add("text", "the quick brown fox")
        s.add("text", "lazy dogs sleep")
        hits = s.search("brown fox", k=1)
        self.assertEqual(hits[0][0].text, "the quick brown fox")  # lexical overlap wins


class IndexDirtyTrackingTest(unittest.TestCase):
    """put()'s dirty-tracking condition must match _text_items()'s real embedding-index inclusion
    rule -- ANY kind with a truthy .text, not a narrower kind whitelist -- on both halves: text
    arriving (a new or newly-text-bearing item) and text leaving (a clear or an overwrite to no
    text). None of these need a real embedder fit, only correct _dirty/_embed_ids bookkeeping."""

    def test_clearing_an_items_text_marks_the_index_dirty(self):
        s = Substrate()
        tid = s.add("text", "original searchable content")
        s.search("original", k=1)  # force a reindex; _dirty is now False
        self.assertFalse(s._dirty)
        self.assertIn(tid, s._embed_ids)

        s.put(SubstrateItem(id=tid, kind="text", text=""))  # clear the text, same id
        self.assertTrue(s._dirty)  # the corpus just shrank -- must be scheduled for reindex

    def test_a_new_text_bearing_record_item_marks_the_index_dirty(self):
        s = Substrate()
        s.add("text", "seed document")
        s.search("seed", k=1)  # force a reindex; _dirty is now False
        self.assertFalse(s._dirty)

        rid = s.add("record", "a record with its own retrievable text surface")
        self.assertIn(rid, [i.id for i in s._text_items(scope=None)])  # _text_items() already covers it
        self.assertTrue(s._dirty)  # put() must now agree and schedule a reindex

    def test_a_new_item_with_no_text_does_not_mark_the_index_dirty(self):
        s = Substrate()
        s.add("text", "seed")
        s.search("seed", k=1)  # force a reindex; _dirty is now False
        self.assertFalse(s._dirty)

        s.add("record", payload={"amount": 42})  # no text at all -- nothing for the index to cover
        self.assertFalse(s._dirty)

    def test_overwriting_a_text_item_with_a_no_text_item_of_the_same_id_marks_it_dirty(self):
        s = Substrate()
        tid = s.add("text", "will be replaced")
        s.search("replaced", k=1)
        self.assertFalse(s._dirty)

        s.put(SubstrateItem(id=tid, kind="record", payload={"amount": 1}))  # same id, no text this time
        self.assertTrue(s._dirty)


@unittest.skipUnless(_HAS_TORCH, "represent embedder needs torch")
class SemanticRetrievalTest(unittest.TestCase):
    def _corpus(self):
        return [
            "the mitochondria produces ATP energy in cellular respiration",
            "photosynthesis converts sunlight into chemical energy",
            "the citric acid cycle oxidizes acetyl-CoA for energy",
            "glycolysis breaks down glucose to release energy",
            "neural networks learn through gradient descent optimization",
            "transformers use self-attention over token sequences",
            "convolutional layers share weights across image positions",
            "backpropagation computes gradients layer by layer",
        ]

    def test_query_retrieves_the_right_topical_cluster(self):
        s = Substrate()
        ingest_documents(s, self._corpus())
        bio_texts = set(self._corpus()[:4])
        hits = s.search("how do cells generate energy", k=3, kind="text")
        bio_in_top3 = sum(1 for item, _ in hits if item.text in bio_texts)
        self.assertGreaterEqual(bio_in_top3, 2)  # the biology cluster dominates the top-3

    def test_persistence_preserves_retrieval(self):
        with tempfile.TemporaryDirectory() as d:
            s = Substrate()
            ingest_documents(s, self._corpus())
            s.save(os.path.join(d, "shard"))
            s2 = Substrate(os.path.join(d, "shard"))
            self.assertEqual(len(s2), len(s))
            self.assertTrue(s2.search("energy in cells", k=1, kind="text"))

    def test_clearing_an_items_text_removes_its_stale_semantic_match(self):
        """End-to-end regression test for the put()/_text_items() dirty-tracking mismatch: search()
        used to keep matching a query against an item's OLD embedding after its text was cleared,
        because put() never marked the index dirty for that change."""
        s = Substrate()
        ids = ingest_documents(s, self._corpus())
        target_id = ids[0]  # "the mitochondria produces ATP energy in cellular respiration"
        query = self._corpus()[0]

        hits = s.search(query, k=1, kind="text")
        self.assertEqual(hits[0][0].id, target_id)  # sanity: a doc matches its own text as the top hit

        s.put(SubstrateItem(id=target_id, kind="text", text=""))  # clear it, same id
        hits2 = s.search(query, k=1, kind="text")
        self.assertNotEqual(hits2[0][0].id, target_id)  # the stale embedding must no longer win
        self.assertEqual(s.get(target_id).text, "")  # the item itself really is empty now


class IngestTest(unittest.TestCase):
    def test_ingest_artifacts_references_not_copies(self):
        s = Substrate()
        with tempfile.TemporaryDirectory() as d:
            adir = os.path.join(d, "router")
            os.makedirs(adir)
            open(os.path.join(adir, "manifest.json"), "w").write(
                json.dumps(
                    {"mixle_artifact": "solve/v1", "meta": {"solve": {"kind": "classifier"}}, "io": {"kind": "record"}}
                )
            )
            ids = ingest_artifacts(s, d)
            self.assertEqual(len(ids), 1)
            art = s.get(ids[0])
            self.assertEqual(art.kind, "artifact")
            self.assertEqual(art.provenance["artifact_kind"], "solve/v1")
            self.assertEqual(art.payload["ref"], adir)  # references the dir, does not copy weights
            self.assertIn("solve", art.text)

    def test_ingest_traces_pairs(self):
        s = Substrate()
        with tempfile.TemporaryDirectory() as d:
            tf = os.path.join(d, "harvested.jsonl")
            open(tf, "w").write(
                '{"input": {"kind": "refund", "amount": 900}, "answer": "finance-escalation"}\n'
                '{"input": {"kind": "bug"}, "label": "support"}\n'
            )
            ids = ingest_traces(s, tf)
            self.assertEqual(len(ids), 2)
            traces = s.all(kind="trace")
            self.assertTrue(any("finance-escalation" in t.text for t in traces))
            self.assertTrue(any("support" in t.text for t in traces))

    def test_ingest_missing_paths_are_empty_not_errors(self):
        s = Substrate()
        self.assertEqual(ingest_artifacts(s, "/no/such/dir"), [])
        self.assertEqual(ingest_traces(s, "/no/such/file.jsonl"), [])


class ModalityTest(unittest.TestCase):
    def test_modalities_cover_the_plan_types(self):
        for m in ("text", "record", "image", "signal", "graph", "field", "artifact", "trace", "context"):
            self.assertIn(m, MODALITIES)


if __name__ == "__main__":
    unittest.main()
