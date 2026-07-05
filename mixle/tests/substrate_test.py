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
