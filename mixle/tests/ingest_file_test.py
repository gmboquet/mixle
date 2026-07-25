"""Data connectors (M1): ingest real files (txt/jsonl/csv) into the substrate.

Also covers MXR-080-0268/0269 (ingestion integrity, identity, and failure visibility): every
``ingest_*`` function's recorded ``content_hash``, its stable/revisioned identity on repeat ingestion,
and its structured per-input :class:`~mixle.substrate.ingest.IngestReceipt` (nothing silently dropped,
nothing partially written with no report of what landed)."""

import csv
import json
import os
import tempfile
import unittest

from mixle.substrate import Substrate, ingest_artifacts, ingest_documents, ingest_file, ingest_records, ingest_traces
from mixle.substrate.freshness import FreshnessState, check_freshness, content_hash
from mixle.substrate.ingest import IngestFailure


class IngestRecordsTest(unittest.TestCase):
    def test_dict_records_keep_payload_and_text_surface(self):
        s = Substrate()
        ids = ingest_records(s, [{"kind": "bug", "note": "crash on save"}], text_fields=["note"])
        self.assertEqual(len(ids), 1)
        item = s.get(ids[0])
        self.assertEqual(item.kind, "record")
        self.assertEqual(item.text, "crash on save")  # text_fields surface
        self.assertEqual(item.payload["kind"], "bug")  # structured payload retained

    def test_tuple_records(self):
        s = Substrate()
        ids = ingest_records(s, [("refund", 900), ("billing", 50)])
        self.assertEqual(len(ids), 2)
        self.assertIn("refund", s.get(ids[0]).text)
        self.assertEqual(s.get(ids[0]).payload["values"], ["refund", 900])


class IngestFileTest(unittest.TestCase):
    def _dir(self):
        return tempfile.TemporaryDirectory()

    def test_txt_one_item_per_line(self):
        with self._dir() as d:
            p = os.path.join(d, "notes.txt")
            open(p, "w").write("refunds within 30 days\n\nsupport open 9 to 5\n")
            s = Substrate()
            ids = ingest_file(s, p)
            self.assertEqual(len(ids), 2)  # blank line skipped
            self.assertTrue(all(i.kind == "text" for i in s.all()))

    def test_jsonl_mixed_strings_texts_and_records(self):
        with self._dir() as d:
            p = os.path.join(d, "kb.jsonl")
            with open(p, "w") as f:
                f.write(json.dumps("a plain string line") + "\n")
                f.write(json.dumps({"text": "a text object", "tags": ["x"]}) + "\n")
                f.write(json.dumps({"ticket": 123, "kind": "refund"}) + "\n")
            s = Substrate()
            ingest_file(s, p)
            self.assertEqual(len(s.all(kind="text")), 2)  # the string + the {text} object
            recs = s.all(kind="record")
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].payload["ticket"], 123)  # structured record preserved

    def test_csv_rows_become_records_keyed_by_header(self):
        with self._dir() as d:
            p = os.path.join(d, "tickets.csv")
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["kind", "amount", "region"])
                w.writeheader()
                w.writerow({"kind": "refund", "amount": "900", "region": "eu"})
                w.writerow({"kind": "billing", "amount": "50", "region": "us"})
            s = Substrate()
            ids = ingest_file(s, p)
            self.assertEqual(len(ids), 2)
            self.assertTrue(any(r.payload.get("region") == "eu" for r in s.all(kind="record")))

    def test_format_forced_by_kind(self):
        with self._dir() as d:
            p = os.path.join(d, "data.dat")  # unknown extension
            open(p, "w").write("line one\nline two\n")
            s = Substrate()
            self.assertEqual(len(ingest_file(s, p, kind="txt")), 2)

    def test_unsupported_format_raises(self):
        with self._dir() as d:
            p = os.path.join(d, "x.parquet")
            open(p, "w").write("binary-ish")
            with self.assertRaises(ValueError):
                ingest_file(Substrate(), p)

    def test_missing_file_is_empty_not_error(self):
        self.assertEqual(ingest_file(Substrate(), "/no/such/file.txt"), [])

    def test_provenance_records_the_source_path(self):
        with self._dir() as d:
            p = os.path.join(d, "notes.txt")
            open(p, "w").write("hello world\n")
            s = Substrate()
            ids = ingest_file(s, p)
            self.assertEqual(s.get(ids[0]).provenance["source"], p)


class IngestReceiptBackwardCompatibilityTest(unittest.TestCase):
    """The returned :class:`~mixle.substrate.ingest.IngestReceipt` must keep behaving exactly like the
    historical bare ``list[str]`` for every existing caller (``mixle.reason.receipt``'s
    ``ids[0] if ids else ""``, ``mixle.scientist.Scientist.learn``'s ``len(ingest_documents(...))``),
    while also exposing the new structured accounting (MXR-080-0268)."""

    def test_receipt_supports_len_indexing_truthiness_and_equality_like_a_plain_list(self):
        s = Substrate()
        receipt = ingest_documents(s, ["doc one", "doc two"])
        self.assertEqual(len(receipt), 2)
        self.assertIsInstance(receipt[0], str)
        self.assertTrue(receipt)
        self.assertTrue(receipt.ok)
        self.assertEqual(receipt.failed, [])

    def test_empty_receipt_equals_a_plain_empty_list(self):
        empty = ingest_documents(Substrate(), [])
        self.assertEqual(empty, [])  # exact historical contract: `ingest_x(...) == []` when nothing lands
        self.assertFalse(empty)
        self.assertTrue(empty.ok)


class IngestContentHashTest(unittest.TestCase):
    """MXR-080-0268: file/artifact ingestion must record the full, algorithm-labelled ``content_hash``
    freshness checks depend on -- byte-for-byte the same digest
    :func:`~mixle.substrate.freshness.content_hash` produces (MXR-080-0267's format), not merely SOME
    hash of SOME shape."""

    def _dir(self):
        return tempfile.TemporaryDirectory()

    def test_ingest_file_records_content_hash_matching_freshness_format(self):
        with self._dir() as d:
            p = os.path.join(d, "notes.txt")
            open(p, "w").write("hello world\n")
            s = Substrate()
            ids = ingest_file(s, p)
            recorded = s.get(ids[0]).provenance["content_hash"]
            self.assertEqual(recorded, content_hash(p))  # byte-identical to freshness.py's own digest
            self.assertRegex(recorded, r"^sha256:[0-9a-f]{64}$")  # full, algorithm-labelled -- never truncated

    def test_ingest_artifacts_records_the_manifest_files_content_hash(self):
        with self._dir() as d:
            adir = os.path.join(d, "router")
            os.makedirs(adir)
            mpath = os.path.join(adir, "manifest.json")
            open(mpath, "w").write(json.dumps({"mixle_artifact": "solve/v1", "meta": {}}))
            s = Substrate()
            receipt = ingest_artifacts(s, d)
            self.assertEqual(len(receipt), 1)
            recorded = s.get(receipt[0]).provenance["content_hash"]
            self.assertEqual(recorded, content_hash(mpath))

    def test_ingest_traces_records_the_source_files_content_hash(self):
        with self._dir() as d:
            tf = os.path.join(d, "harvested.jsonl")
            open(tf, "w").write(json.dumps({"input": "q", "answer": "a"}) + "\n")
            s = Substrate()
            receipt = ingest_traces(s, tf)
            self.assertEqual(len(receipt), 1)
            recorded = s.get(receipt[0]).provenance["content_hash"]
            self.assertEqual(recorded, content_hash(tf))

    def test_no_content_hash_when_source_is_just_a_label_not_a_real_file(self):
        s = Substrate()
        ids = ingest_documents(s, ["a plain in-memory doc"], source="documents")
        self.assertNotIn("content_hash", s.get(ids[0]).provenance)  # nothing on disk to hash -- honest absence


class IngestRevisionTest(unittest.TestCase):
    """MXR-080-0268: repeated ingestion of the SAME source must update/revise the existing record(s),
    never accumulate a fresh random-id duplicate every call."""

    def _dir(self):
        return tempfile.TemporaryDirectory()

    def test_repeat_ingestion_of_same_file_is_a_revision_not_a_duplicate(self):
        with self._dir() as d:
            p = os.path.join(d, "notes.txt")
            open(p, "w").write("first line\nsecond line\n")
            s = Substrate()
            first = ingest_file(s, p)
            second = ingest_file(s, p)
            self.assertEqual(list(first), list(second))  # identical, stable ids -- not fresh random ones
            self.assertEqual(len(s.all(kind="text")), 2)  # two records, not four
            self.assertEqual(s.get(second[0]).provenance["revision"], 2)
            self.assertEqual(s.get(second[1]).provenance["revision"], 2)

    def test_repeat_ingestion_of_same_artifact_directory_is_a_revision(self):
        with self._dir() as d:
            adir = os.path.join(d, "router")
            os.makedirs(adir)
            open(os.path.join(adir, "manifest.json"), "w").write(json.dumps({"mixle_artifact": "solve/v1"}))
            s = Substrate()
            first = ingest_artifacts(s, d)
            second = ingest_artifacts(s, d)
            self.assertEqual(list(first), list(second))
            self.assertEqual(len(s.all(kind="artifact")), 1)
            self.assertEqual(s.get(second[0]).provenance["revision"], 2)

    def test_repeat_ingestion_of_same_traces_file_is_a_revision(self):
        with self._dir() as d:
            tf = os.path.join(d, "harvested.jsonl")
            open(tf, "w").write(json.dumps({"input": "q", "answer": "a"}) + "\n")
            s = Substrate()
            first = ingest_traces(s, tf)
            second = ingest_traces(s, tf)
            self.assertEqual(list(first), list(second))
            self.assertEqual(len(s.all(kind="trace")), 1)
            self.assertEqual(s.get(second[0]).provenance["revision"], 2)

    def test_changed_file_content_on_repeat_ingestion_updates_hash_under_the_same_id(self):
        with self._dir() as d:
            p = os.path.join(d, "notes.txt")
            open(p, "w").write("version one\n")
            s = Substrate()
            first = ingest_file(s, p)
            first_hash = s.get(first[0]).provenance["content_hash"]

            open(p, "w").write("version two, changed\n")
            second = ingest_file(s, p)

            self.assertEqual(first[0], second[0])  # same logical identity
            second_hash = s.get(second[0]).provenance["content_hash"]
            self.assertNotEqual(first_hash, second_hash)  # the revision picked up the new bytes
            self.assertEqual(second_hash, content_hash(p))
            self.assertEqual(s.get(second[0]).provenance["revision"], 2)
            self.assertEqual(len(s.all(kind="text")), 1)  # still one record, not two


class IngestFailureVisibilityTest(unittest.TestCase):
    """MXR-080-0268: a malformed line/manifest among otherwise-valid input must be named on the
    receipt's ``.failed``, not silently ``continue``-d past leaving only a shorter list of ids that
    looks identical to "nothing was wrong"."""

    def _dir(self):
        return tempfile.TemporaryDirectory()

    def test_malformed_jsonl_line_among_valid_ones_is_reported_not_silently_dropped(self):
        with self._dir() as d:
            tf = os.path.join(d, "harvested.jsonl")
            with open(tf, "w") as f:
                f.write(json.dumps({"input": "q1", "answer": "a1"}) + "\n")
                f.write("{not valid json\n")  # malformed
                f.write(json.dumps({"input": "q2", "answer": "a2"}) + "\n")
            s = Substrate()
            receipt = ingest_traces(s, tf)
            self.assertEqual(len(receipt), 2)  # the two valid rows still landed
            self.assertFalse(receipt.ok)
            self.assertEqual(len(receipt.failed), 1)
            self.assertIsInstance(receipt.failed[0], IngestFailure)
            self.assertIn("line 2", receipt.failed[0].error)
            self.assertEqual(len(s.all(kind="trace")), 2)  # substrate state matches the receipt exactly

    def test_malformed_manifest_among_valid_ones_is_reported_not_silently_dropped(self):
        with self._dir() as d:
            good = os.path.join(d, "good")
            os.makedirs(good)
            open(os.path.join(good, "manifest.json"), "w").write(json.dumps({"mixle_artifact": "solve/v1"}))
            bad = os.path.join(d, "bad")
            os.makedirs(bad)
            open(os.path.join(bad, "manifest.json"), "w").write("{not valid json")
            s = Substrate()
            receipt = ingest_artifacts(s, d)
            self.assertEqual(len(receipt), 1)  # the good one still landed
            self.assertFalse(receipt.ok)
            self.assertEqual(len(receipt.failed), 1)
            self.assertIn("bad", receipt.failed[0].source)
            self.assertEqual(len(s.all(kind="artifact")), 1)

    def test_bare_successful_id_list_cannot_be_returned_when_something_failed(self):
        """A caller checking only `len(receipt)` must never mistake a partial batch for a complete
        one: a plain list has no way to carry `.failed`, but `IngestReceipt` always does."""
        with self._dir() as d:
            tf = os.path.join(d, "harvested.jsonl")
            with open(tf, "w") as f:
                f.write(json.dumps({"input": "q1", "answer": "a1"}) + "\n")
                f.write("not json at all\n")
            receipt = ingest_traces(Substrate(), tf)
            self.assertEqual(len(receipt), 1)
            self.assertFalse(receipt.ok)  # distinguishable from a clean single-item batch


class IngestPartialWriteAbortTest(unittest.TestCase):
    """MXR-080-0269: valid JSON of the wrong shape (e.g. a JSON list where an object is required) must
    not abort the rest of the batch, and must not silently lose items ingested earlier in the same
    call. Each scenario below puts a good item BEFORE and a good item AFTER the wrong-shaped one, so a
    regression that aborts partway would show up as a missing id, not just a raised exception."""

    def _dir(self):
        return tempfile.TemporaryDirectory()

    def test_json_list_where_dict_expected_does_not_abort_ingest_traces_or_lose_earlier_writes(self):
        with self._dir() as d:
            tf = os.path.join(d, "harvested.jsonl")
            with open(tf, "w") as f:
                f.write(json.dumps({"input": "q1", "answer": "a1"}) + "\n")  # BEFORE the bad row
                f.write(json.dumps([1, 2, 3]) + "\n")  # valid JSON, wrong shape: a list, not an object
                f.write(json.dumps({"input": "q2", "answer": "a2"}) + "\n")  # AFTER the bad row
            s = Substrate()
            receipt = ingest_traces(s, tf)  # must not raise
            self.assertEqual(len(receipt), 2)  # both valid rows landed -- the one before AND the one after
            self.assertEqual(len(receipt.failed), 1)
            self.assertIn("list", receipt.failed[0].error)
            self.assertEqual(len(s.all(kind="trace")), 2)  # nothing lost, nothing left unreported

    def test_json_list_where_dict_expected_does_not_abort_ingest_artifacts_or_lose_earlier_writes(self):
        with self._dir() as d:
            # names sort a_first < b_bad_shape < c_last, matching ingest_artifacts' own sorted() scan order
            first = os.path.join(d, "a_first")
            os.makedirs(first)
            open(os.path.join(first, "manifest.json"), "w").write(json.dumps({"mixle_artifact": "solve/v1"}))
            bad = os.path.join(d, "b_bad_shape")
            os.makedirs(bad)
            open(os.path.join(bad, "manifest.json"), "w").write(json.dumps([1, 2, 3]))  # valid JSON, wrong shape
            last = os.path.join(d, "c_last")
            os.makedirs(last)
            open(os.path.join(last, "manifest.json"), "w").write(json.dumps({"mixle_artifact": "regress/v1"}))

            s = Substrate()
            receipt = ingest_artifacts(s, d)  # must not raise
            self.assertEqual(len(receipt), 2)  # the one BEFORE and the one AFTER the bad manifest both landed
            self.assertEqual(len(receipt.failed), 1)
            self.assertIn("b_bad_shape", receipt.failed[0].source)
            self.assertEqual(len(s.all(kind="artifact")), 2)


class IngestFreshnessIntegrationTest(unittest.TestCase):
    """End-to-end MXR-080-0268 <-> MXR-080-0267 integration: what ingest.py records must actually be
    what freshness.py needs to detect a real byte change, not just a hash recorded for its own sake."""

    def _dir(self):
        return tempfile.TemporaryDirectory()

    def test_ingest_then_freshness_check_detects_a_subsequent_byte_change(self):
        with self._dir() as d:
            p = os.path.join(d, "policy.txt")
            open(p, "w").write("refunds within 30 days\n")
            s = Substrate()
            ids = ingest_file(s, p)

            verdict = check_freshness(s, ids[0])
            self.assertTrue(verdict.fresh)  # just ingested, untouched -> fresh

            open(p, "w").write("refunds within 60 days, changed policy\n")
            verdict2 = check_freshness(s, ids[0])
            self.assertEqual(verdict2.state, FreshnessState.STALE)
            self.assertTrue(any("changed" in sig for sig in verdict2.signals))


if __name__ == "__main__":
    unittest.main()
