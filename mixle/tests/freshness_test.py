"""Knowledge freshness (O4): drift on knowledge items — moved data, changed content, superseded, aged."""

import os
import tempfile
import time
import unittest

from mixle.substrate import Substrate
from mixle.substrate.freshness import check_freshness, content_hash, freshness_report


class FreshnessTest(unittest.TestCase):
    def test_plain_item_is_fresh(self):
        s = Substrate()
        i = s.add(kind="text", text="fresh doc")
        f = check_freshness(s, i)
        self.assertTrue(f.fresh)
        self.assertEqual(f.signals, [])

    def test_missing_item_is_stale(self):
        f = check_freshness(Substrate(), "nope")
        self.assertFalse(f.fresh)
        self.assertIn("missing", f.signals[0])

    def test_moved_referenced_file_is_flagged(self):
        s = Substrate()
        i = s.add(kind="artifact", text="x", payload={"ref": "/no/such/file.bin"})
        f = check_freshness(s, i)
        self.assertFalse(f.fresh)
        self.assertIn("moved", f.signals[0])

    def test_changed_content_hash_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "data.txt")
            open(p, "w").write("v1")
            s = Substrate()
            i = s.add(kind="artifact", text="x", payload={"ref": p}, provenance={"content_hash": content_hash(p)})
            self.assertTrue(check_freshness(s, i).fresh)  # untouched -> fresh
            open(p, "w").write("v2 changed")
            f = check_freshness(s, i)
            self.assertFalse(f.fresh)
            self.assertIn("changed", f.signals[0])

    def test_superseded_by_declaration(self):
        s = Substrate()
        old = s.add(kind="text", text="v1")
        s.add(kind="text", text="v2", provenance={"supersedes": old})
        f = check_freshness(s, old)
        self.assertFalse(f.fresh)
        self.assertIn("superseded", f.signals[0])

    def test_aged_out_is_a_review_trigger_not_proof(self):
        s = Substrate()
        i = s.add(kind="text", text="old")
        s.get(i).created_at = time.time() - 10_000
        f = check_freshness(s, i, max_age_s=3600)
        self.assertFalse(f.fresh)
        self.assertIn("review, not proof", f.signals[0])  # honest wording travels with the signal

    def test_no_age_policy_means_no_age_signal(self):
        s = Substrate()
        i = s.add(kind="text", text="old")
        s.get(i).created_at = time.time() - 10_000
        self.assertTrue(check_freshness(s, i).fresh)  # age alone doesn't fire without a policy

    def test_report_sweeps_the_store(self):
        s = Substrate()
        s.add(kind="text", text="fine")
        s.add(kind="artifact", text="x", payload={"ref": "/gone.bin"})
        rep = freshness_report(s)
        self.assertEqual(rep["n_items"], 2)
        self.assertEqual(rep["n_stale"], 1)
        self.assertEqual(rep["n_fresh"], 1)


if __name__ == "__main__":
    unittest.main()
