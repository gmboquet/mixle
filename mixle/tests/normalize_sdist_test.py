"""``scripts/normalize_sdist.py`` makes an sdist byte-for-byte reproducible across builds.

``SOURCE_DATE_EPOCH`` alone does not reach every tar-member mtime setuptools' ``sdist`` command writes
(only real wall-clock file mtimes land there), and this environment's ``gzip`` module was observed to
retain platform-sensitive header fields unless every input is explicit -- these tests pin both of
those down using small, synthetic tarballs (not a real ``python -m build``, which is
slow and would only re-test packaging, not this script's own logic) that deliberately carry different
member mtimes/uid/gid, the exact shape two real builds of the same commit, run at different wall-clock
times, produce.
"""

from __future__ import annotations

import gzip
import io
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts.normalize_sdist import normalize


def _make_sdist(path: Path, *, mtime_base: int, content: bytes = b"hello world\n") -> None:
    """Build a small, synthetic .tar.gz whose member mtimes vary (as two real builds' would)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        for i, name in enumerate(("pkg-0.1/PKG-INFO", "pkg-0.1/pkg/__init__.py", "pkg-0.1/pkg/core.py")):
            data = content + f" ({name})".encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = mtime_base + i  # distinct real per-file mtimes, like an actual build tree
            info.uid, info.gid = 501, 20  # a real local user/group, not 0/0
            tf.addfile(info, io.BytesIO(data))
    path.write_bytes(gzip.compress(buf.getvalue(), mtime=int(time.time())))


class NormalizeSdistTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.a = Path(self._tmp.name) / "a.tar.gz"
        self.b = Path(self._tmp.name) / "b.tar.gz"

    def test_two_builds_with_different_wall_clock_mtimes_normalize_identically(self):
        epoch = 1_700_000_000
        _make_sdist(self.a, mtime_base=1_700_000_050)  # a later "build" ...
        _make_sdist(self.b, mtime_base=1_700_000_000)  # ... than this one
        self.assertNotEqual(self.a.read_bytes(), self.b.read_bytes())  # sanity: they start out different

        normalize(self.a, epoch)
        normalize(self.b, epoch)
        self.assertEqual(self.a.read_bytes(), self.b.read_bytes())

    def test_normalized_archive_still_extracts_the_original_content(self):
        epoch = 1_700_000_000
        _make_sdist(self.a, mtime_base=1_700_000_050, content=b"distinctive payload\n")
        normalize(self.a, epoch)

        with tarfile.open(self.a) as tf:
            names = sorted(m.name for m in tf.getmembers())
            self.assertEqual(names, ["pkg-0.1/PKG-INFO", "pkg-0.1/pkg/__init__.py", "pkg-0.1/pkg/core.py"])
            member = tf.getmember("pkg-0.1/PKG-INFO")
            self.assertEqual(tf.extractfile(member).read(), b"distinctive payload\n (pkg-0.1/PKG-INFO)")

    def test_every_member_mtime_uid_gid_is_pinned(self):
        epoch = 1_700_000_000
        _make_sdist(self.a, mtime_base=1_700_000_050)
        normalize(self.a, epoch)

        with tarfile.open(self.a) as tf:
            for member in tf.getmembers():
                self.assertEqual(member.mtime, epoch)
                self.assertEqual(member.uid, 0)
                self.assertEqual(member.gid, 0)
                self.assertEqual(member.uname, "")
                self.assertEqual(member.gname, "")

    def test_a_different_epoch_produces_different_bytes(self):
        # Confirms the epoch is actually used, not silently ignored (a bug that would make the
        # "always identical" property above true for a trivial, useless reason).
        _make_sdist(self.a, mtime_base=1_700_000_050)
        _make_sdist(self.b, mtime_base=1_700_000_050)
        normalize(self.a, 1_700_000_000)
        normalize(self.b, 1_800_000_000)
        self.assertNotEqual(self.a.read_bytes(), self.b.read_bytes())

    def test_repeated_normalization_of_the_same_input_is_deterministic(self):
        # The compressor receives an empty filename and fixed mtime rather than ambient path/time state.
        epoch = 1_700_000_000
        _make_sdist(self.a, mtime_base=1_700_000_050)
        _make_sdist(self.b, mtime_base=1_700_000_050)
        self.assertEqual(self.a.read_bytes(), self.b.read_bytes())  # identical inputs, not just content

        normalize(self.a, epoch)
        normalize(self.b, epoch)
        self.assertEqual(self.a.read_bytes(), self.b.read_bytes())

    def test_failed_atomic_replace_preserves_the_original(self):
        _make_sdist(self.a, mtime_base=1_700_000_050)
        original = self.a.read_bytes()
        with mock.patch("scripts.normalize_sdist.os.replace", side_effect=OSError("injected replace failure")):
            with self.assertRaisesRegex(OSError, "injected replace failure"):
                normalize(self.a, 1_700_000_000)
        self.assertEqual(self.a.read_bytes(), original)
        self.assertEqual(list(self.a.parent.glob(f".{self.a.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
