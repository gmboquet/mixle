"""Data connectors (M1): ingest real files (txt/jsonl/csv) into the substrate."""

import csv
import json
import os
import tempfile
import unittest

from mixle.substrate import Substrate, ingest_file, ingest_records


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


if __name__ == "__main__":
    unittest.main()
