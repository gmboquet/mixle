"""Knowledge freshness (O4): drift on knowledge items — moved data, changed content, superseded, aged."""

import os
import stat
import tempfile
import time
import unittest

from mixle.substrate import Substrate
from mixle.substrate.freshness import FreshnessState, check_freshness, content_hash, freshness_report


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
        # get()/all() return defensive copies (MXR-080-0234); update() is how a change actually lands.
        s.update(i, created_at=time.time() - 10_000)
        f = check_freshness(s, i, max_age_s=3600)
        self.assertFalse(f.fresh)
        self.assertIn("review, not proof", f.signals[0])  # honest wording travels with the signal

    def test_no_age_policy_means_no_age_signal(self):
        s = Substrate()
        i = s.add(kind="text", text="old")
        s.update(i, created_at=time.time() - 10_000)
        self.assertTrue(check_freshness(s, i).fresh)  # age alone doesn't fire without a policy

    def test_report_sweeps_the_store(self):
        s = Substrate()
        s.add(kind="text", text="fine")
        s.add(kind="artifact", text="x", payload={"ref": "/gone.bin"})
        rep = freshness_report(s)
        self.assertEqual(rep["n_items"], 2)
        self.assertEqual(rep["n_stale"], 1)
        self.assertEqual(rep["n_fresh"], 1)

    # -- MXR-080-0267: fail-open-on-unreadable-content, hash truncation, unvalidated clocks/policies --

    def test_directory_with_wrong_recorded_hash_is_never_reported_fresh(self):
        """The audit's own adversarial scenario: a referenced path that exists but is a directory (so
        content_hash() cannot read it) with a deliberately WRONG recorded hash. Pre-fix, the None
        digest added no signal at all and this was reported fresh=True -- the fail-open MXR-080-0267
        exists to close. It must now be UNVERIFIABLE, never FRESH."""
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "a_directory")
            os.mkdir(sub)
            self.assertIsNone(content_hash(sub))  # confirms the precondition: unreadable as bytes
            s = Substrate()
            i = s.add(
                kind="artifact",
                text="x",
                payload={"ref": sub},
                provenance={"content_hash": "deliberately-wrong-hash-not-a-real-digest"},
            )
            f = check_freshness(s, i)
            self.assertIs(f.state, FreshnessState.UNVERIFIABLE)
            self.assertFalse(f.fresh)
            self.assertTrue(any("unverifiable" in sig for sig in f.signals), f.signals)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores unreadable permission bits")
    def test_unreadable_file_with_wrong_recorded_hash_is_never_reported_fresh(self):
        """Same fail-open scenario as above, but for a permission-denied file rather than a directory
        -- content_hash() fails with the same OSError family, and the result must again be
        UNVERIFIABLE rather than silently fresh."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "secret.bin")
            with open(p, "w") as fh:
                fh.write("some bytes")
            os.chmod(p, 0o000)
            try:
                self.assertIsNone(content_hash(p))
                s = Substrate()
                i = s.add(
                    kind="artifact",
                    text="x",
                    payload={"ref": p},
                    provenance={"content_hash": "deliberately-wrong-hash-not-a-real-digest"},
                )
                f = check_freshness(s, i)
                self.assertIs(f.state, FreshnessState.UNVERIFIABLE)
                self.assertFalse(f.fresh)
            finally:
                os.chmod(p, stat.S_IWUSR | stat.S_IRUSR)  # restore so tempdir cleanup can delete it

    def test_unverifiable_content_with_no_recorded_hash_is_still_fresh(self):
        """A referenced directory with NO recorded content_hash at all has nothing to verify against
        -- unlike the wrong-recorded-hash case above, this is not a contradicted claim, so it stays
        fresh (the finding is about a silently-skipped COMPARISON, not about requiring every ref to
        carry a hash)."""
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "a_directory")
            os.mkdir(sub)
            s = Substrate()
            i = s.add(kind="artifact", text="x", payload={"ref": sub})
            f = check_freshness(s, i)
            self.assertIs(f.state, FreshnessState.FRESH)
            self.assertTrue(f.fresh)

    def test_content_hash_is_full_length_and_algorithm_labelled(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.txt")
            with open(p, "w") as fh:
                fh.write("hello world")
            h = content_hash(p)
            algo, _, digest = h.partition(":")
            self.assertEqual(algo, "sha256")
            self.assertEqual(len(digest), 64)  # full sha256 hex, not a truncated 128-bit slice

    def test_nan_now_is_rejected(self):
        s = Substrate()
        i = s.add(kind="text", text="x")
        with self.assertRaises(ValueError):
            check_freshness(s, i, now=float("nan"))

    def test_infinite_now_is_rejected(self):
        s = Substrate()
        i = s.add(kind="text", text="x")
        with self.assertRaises(ValueError):
            check_freshness(s, i, now=float("inf"))

    def test_invalid_max_age_s_is_rejected(self):
        s = Substrate()
        i = s.add(kind="text", text="x")
        s.update(i, created_at=time.time() - 100)
        for bad in (float("nan"), float("inf"), float("-inf"), -5.0):
            with self.subTest(max_age_s=repr(bad)):
                with self.assertRaises(ValueError):
                    check_freshness(s, i, max_age_s=bad)

    def test_future_created_at_is_flagged_stale_not_silently_fresh(self):
        """A recorded created_at from the future is definitionally invalid. Pre-fix,
        ``max(0.0, now - created_at)`` silently clamped the negative delta to age_s=0.0 with no
        signal; it must now be a named STALE signal, with the true (negative) age still reported."""
        s = Substrate()
        i = s.add(kind="text", text="x")
        s.update(i, created_at=time.time() + 1_000_000)
        f = check_freshness(s, i)
        self.assertIs(f.state, FreshnessState.STALE)
        self.assertFalse(f.fresh)
        self.assertLess(f.age_s, 0)
        self.assertTrue(any("future-dated" in sig for sig in f.signals), f.signals)

    def test_nan_created_at_is_flagged_stale_without_raising(self):
        """A NaN created_at is corrupted substrate DATA, not a caller argument -- audited and named
        like any other signal (state STALE) rather than raised, so one bad record can't abort a
        freshness_report() sweep over every other item in the store."""
        s = Substrate()
        i = s.add(kind="text", text="x")
        s.update(i, created_at=float("nan"))
        f = check_freshness(s, i)
        self.assertIs(f.state, FreshnessState.STALE)
        self.assertFalse(f.fresh)
        self.assertTrue(any("not a finite timestamp" in sig for sig in f.signals), f.signals)

    def test_freshness_report_separates_stale_from_unverifiable(self):
        s = Substrate()
        s.add(kind="text", text="fine")
        s.add(kind="artifact", text="x", payload={"ref": "/gone.bin"})
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "a_directory")
            os.mkdir(sub)
            s.add(
                kind="artifact",
                text="y",
                payload={"ref": sub},
                provenance={"content_hash": "deliberately-wrong-hash"},
            )
            rep = freshness_report(s)
            self.assertEqual(rep["n_items"], 3)
            self.assertEqual(rep["n_fresh"], 1)
            self.assertEqual(rep["n_stale"], 1)
            self.assertEqual(rep["n_unverifiable"], 1)
            self.assertEqual(len(rep["stale"]), 1)
            self.assertEqual(len(rep["unverifiable"]), 1)


if __name__ == "__main__":
    unittest.main()
