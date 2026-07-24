"""The knowledge substrate: typed provenanced items, cross-modal retrieval, scope, persistence."""

import glob
import json
import os
import tempfile
import unittest
from unittest import mock

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


class ImmutabilityContractTest(unittest.TestCase):
    """MXR-080-0234: put()/get()/all() must not let a caller's live reference reach (or leak from) the
    store, and update() is the only supported way to make a change land."""

    def test_get_returns_a_copy_not_the_stored_object(self):
        s = Substrate()
        item = SubstrateItem(kind="text", text="hi")
        iid = s.put(item)
        self.assertIsNot(s.get(iid), item)  # put() didn't alias the caller's object either
        self.assertIsNot(s.get(iid), s.get(iid))  # two get() calls never share one mutable object

    def test_mutating_a_get_result_does_not_touch_the_store(self):
        s = Substrate()
        iid = s.add("text", "original", scope="team-a", tags=["orig"])

        handle = s.get(iid)
        handle.text = "corrupted"
        handle.scope = "team-b"
        handle.tags.append("mutated-in-place")
        handle.payload["injected"] = True

        stored = s.get(iid)
        self.assertEqual(stored.text, "original")
        self.assertEqual(stored.scope, "team-a")
        self.assertEqual(stored.tags, ["orig"])
        self.assertEqual(stored.payload, {})

    def test_mutating_an_all_result_does_not_touch_the_store(self):
        s = Substrate()
        iid = s.add("text", "original", tags=["orig"])
        s.all()[0].tags.append("mutated-in-place")
        self.assertEqual(s.get(iid).tags, ["orig"])

    def test_put_does_not_alias_the_callers_object(self):
        s = Substrate()
        item = SubstrateItem(kind="text", text="original")
        iid = s.put(item)
        item.text = "mutated after put()"  # the caller keeps mutating their own local object
        self.assertEqual(s.get(iid).text, "original")

    def test_mutation_cannot_walk_an_item_across_the_scope_boundary(self):
        """The concrete MXR-080-0234 repro: get() a team-a item, rewrite .scope on the handle, and
        confirm the store's own scope-filtered view never moves it -- a scope change must go through
        update()/put(), which is the store's only re-validated write path."""
        s = Substrate()
        iid = s.add("text", "secret", scope="team-a")
        s.get(iid).scope = "team-b"
        self.assertEqual([i.id for i in s.all(scope="team-a")], [iid])
        self.assertEqual([i.id for i in s.all(scope="team-b")], [])

    def test_update_changes_the_stored_item(self):
        s = Substrate()
        iid = s.add("text", "original", tags=["v1"])
        updated = s.update(iid, text="revised", tags=["v2"])
        self.assertEqual(updated.text, "revised")
        self.assertEqual(s.get(iid).text, "revised")
        self.assertEqual(s.get(iid).tags, ["v2"])

    def test_update_returns_a_copy_too(self):
        s = Substrate()
        iid = s.add("text", "original")
        updated = s.update(iid, text="revised")
        updated.text = "mutated after update() returned"
        self.assertEqual(s.get(iid).text, "revised")

    def test_update_revalidates_kind(self):
        s = Substrate()
        iid = s.add("text", "original")
        with self.assertRaises(ValueError):
            s.update(iid, kind="hologram")
        self.assertEqual(s.get(iid).kind, "text")  # the invalid update never landed

    def test_update_missing_id_raises_keyerror(self):
        s = Substrate()
        with self.assertRaises(KeyError):
            s.update("no-such-id", text="x")

    def test_update_cannot_change_id(self):
        s = Substrate()
        iid = s.add("text", "original")
        with self.assertRaises(ValueError):
            s.update(iid, id="a-different-id")
        self.assertIsNotNone(s.get(iid))  # the original id is still the one stored

    def test_update_marks_the_index_dirty_like_put_does(self):
        s = Substrate()
        iid = s.add("text", "original searchable content")
        s.search("original", k=1)  # force a reindex; _dirty is now False
        self.assertFalse(s._dirty)

        s.update(iid, text="")  # clearing text shrinks the indexed corpus, same as put() would
        self.assertTrue(s._dirty)


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


_TMP_GLOB = ".tmp-substrate-*"


class PersistenceAtomicityTest(unittest.TestCase):
    """MXR-080-0235: save() must be all-or-nothing (temp file + fsync + os.replace, matching
    mixle.task.artifact._atomic_json_dump / mixle.system.registry.Registry._write_index), and load()
    must not commit a partially-parsed file over whatever this shard held before the call."""

    def test_save_round_trips_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "shard")
            s = Substrate()
            s.add("text", "alpha")
            s.add("record", payload={"n": 1})
            s.save(root)

            s2 = Substrate()
            s2.load(root)
            self.assertEqual(len(s2), 2)
            self.assertEqual(sorted(i.text for i in s2.all() if i.kind == "text"), ["alpha"])
            self.assertEqual(glob.glob(os.path.join(root, _TMP_GLOB)), [])

    def test_mid_write_failure_preserves_the_previous_shard_and_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "shard")
            s = Substrate()
            s.add("text", "keep me")
            s.add("text", "keep me too")
            s.add("text", "and me")
            s.save(root)  # a good, durable v1 shard
            with open(os.path.join(root, "items.jsonl")) as f:
                original = f.read()

            # Simulate a crash partway through a re-save (disk full / killed process / a bad field on
            # a LATER item): json.dumps blows up on its 2nd call, i.e. after the temp file already has
            # some -- but not all -- of the new generation written to it.
            real_dumps = json.dumps
            calls = {"n": 0}

            def flaky_dumps(*a, **kw):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("simulated crash mid-write")
                return real_dumps(*a, **kw)

            with mock.patch("json.dumps", side_effect=flaky_dumps):
                with self.assertRaises(RuntimeError):
                    s.save(root)  # a routine re-save of the same 3 items

            with open(os.path.join(root, "items.jsonl")) as f:
                self.assertEqual(f.read(), original)  # untouched by the failed write
            self.assertEqual(glob.glob(os.path.join(root, _TMP_GLOB)), [])  # temp cleaned up

            s2 = Substrate()
            s2.load(root)
            self.assertEqual(len(s2), 3)  # the durable v1 shard is still fully recoverable

    def test_load_malformed_row_raises_and_preserves_prior_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "shard")
            os.makedirs(root)
            good = [
                SubstrateItem(kind="text", text="alpha").to_json(),
                SubstrateItem(kind="text", text="beta").to_json(),
            ]
            trailing = SubstrateItem(kind="text", text="gamma").to_json()
            with open(os.path.join(root, "items.jsonl"), "w") as f:
                for d_ in good:
                    f.write(json.dumps(d_) + "\n")
                f.write("{not valid json at all\n")  # line 3: malformed
                f.write(json.dumps(trailing) + "\n")  # line 4: valid, but after the break

            s = Substrate()
            prior_id = s.add("text", "pre-existing item, predates this load() call")

            with self.assertRaises(ValueError) as ctx:
                s.load(root)
            self.assertIn("items.jsonl:3", str(ctx.exception))  # names the exact malformed row

            # the failed load must not have touched the shard at all: same single prior item, same id.
            self.assertEqual(len(s), 1)
            self.assertEqual(s.get(prior_id).text, "pre-existing item, predates this load() call")

    def test_load_malformed_row_on_a_fresh_shard_leaves_it_empty_not_partial(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "shard")
            os.makedirs(root)
            with open(os.path.join(root, "items.jsonl"), "w") as f:
                f.write(json.dumps(SubstrateItem(kind="text", text="alpha").to_json()) + "\n")
                f.write(json.dumps(SubstrateItem(kind="text", text="beta").to_json()) + "\n")
                f.write("{not valid json at all\n")

            s = Substrate()  # no prior state
            with self.assertRaises(ValueError):
                s.load(root)
            self.assertEqual(len(s), 0)  # not 2 -- a failed load must not half-apply the new file either


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
