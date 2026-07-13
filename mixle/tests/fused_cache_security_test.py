"""The fused-kernel disk cache cannot be used for code execution by another user (worklist S13).

``_njit`` writes generated numba source to a cache file and then imports+``exec``s it. The old loader put
that cache in a single shared ``/tmp/mixle_fused_cache`` and imported whatever file was already at the
predicted path -- so on a multi-user host another user could pre-place ``_pysp_fused_<digest>.py`` and get
arbitrary code execution as the victim. These tests pin the containment: a per-user 0700 cache directory,
ownership verification before any import, and a planted file never being executed.
"""

import os
import stat
import tempfile
import unittest

import mixle.stats.compute.fused_codegen as fc


class CacheDirPrivacyTest(unittest.TestCase):
    def test_default_cache_dir_is_per_user(self):
        d = fc._default_cache_dir()
        if hasattr(os, "getuid"):
            self.assertIn(str(os.getuid()), d)  # not a single dir shared across users

    def test_private_cache_dir_is_0700_and_owned(self):
        d = fc._private_cache_dir()
        self.assertIsNotNone(d)
        self.assertEqual(stat.S_IMODE(os.lstat(d).st_mode), 0o700)

    def test_group_or_other_writable_dir_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o777)
            self.assertFalse(fc._owned_privately(tmp, require_dir=True))

    def test_symlinked_cache_file_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "evil.py")
            with open(target, "w") as f:
                f.write("x = 1\n")
            link = os.path.join(tmp, "cache_entry.py")
            os.symlink(target, link)
            # a symlink is not a regular file we privately own -> the loader must not import it
            self.assertFalse(fc._owned_privately(link, require_dir=False))


class PlantedFileNotExecutedTest(unittest.TestCase):
    def test_planted_symlink_at_cache_path_is_overwritten_not_executed(self):
        import pytest

        pytest.importorskip("numba")

        src = "def _fused_sec_probe(x):\n    return x * 2.0\n"
        import hashlib

        # mirrors _njit's digest EXACTLY (v2 salt + parallel flag + source); if this drifts from the
        # module the symlink lands at an unused path and the test can no longer see the overwrite
        digest = hashlib.sha1(f"v2|parallel=False|{src}".encode()).hexdigest()[:16]  # noqa: S324 -- cache key
        path = os.path.join(fc._private_cache_dir(), f"_pysp_fused_{digest}.py")

        with tempfile.TemporaryDirectory() as tmp:
            evil = os.path.join(tmp, "evil.py")
            with open(evil, "w") as f:
                f.write(
                    "import os\nos.environ['MIXLE_FUSED_SEC_BREACH'] = '1'\n"
                    "def _fused_sec_probe(x):\n    return -999.0\n"
                )
            os.environ.pop("MIXLE_FUSED_SEC_BREACH", None)
            if os.path.lexists(path):
                os.remove(path)
            os.symlink(evil, path)  # attacker symlink at the exact predicted cache path

            fn = fc._njit(src, "_fused_sec_probe")

            self.assertEqual(fn(21.0), 42.0)  # our kernel ran, not the planted -999 one
            self.assertIsNone(os.environ.get("MIXLE_FUSED_SEC_BREACH"))  # attacker code never executed
            self.assertFalse(os.path.islink(path))  # the symlink was replaced with our own source
            if os.path.lexists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
