"""The fused-kernel disk cache cannot be used for code execution by another user (worklist S13).

``_njit`` writes generated numba source to a cache file and then imports+``exec``s it. The old loader put
that cache in a single shared ``/tmp/mixle_fused_cache`` and imported whatever file was already at the
predicted path -- so on a multi-user host another user could pre-place ``_pysp_fused_<digest>.py`` and get
arbitrary code execution as the victim. These tests pin the containment: a per-user 0700 cache directory,
ownership verification before any import, and a planted file never being executed.
"""

import ast
import os
import pathlib
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


class CacheKeyHashIsNotASecurityControlTest(unittest.TestCase):
    """The cache key is a content digest, and it has to be spelled as one.

    ``_njit`` names its cache module after a SHA-1 of the generated source. SHA-1 is adequate for
    that -- what makes importing the cached file safe is the ownership check exercised by the tests
    above, never the digest -- but a bare ``hashlib.sha1()`` claims otherwise to every reader and to
    every scanner, and a FIPS-enabled build refuses the call outright, which would take the whole
    fused path down. ``usedforsecurity=False`` states the purpose and keeps the call legal.
    """

    def _weak_hash_calls(self):
        """Yield ``(path, lineno, keywords)`` for md5/sha1 constructor calls in the shipped library."""
        import mixle

        root = pathlib.Path(mixle.__file__).parent
        for source_file in sorted(root.rglob("*.py")):
            if "tests" in source_file.relative_to(root).parts:
                continue  # tests deliberately recompute cache keys the unqualified way
            for node in ast.walk(ast.parse(source_file.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in {"md5", "sha1"}:
                    yield source_file, node.lineno, {kw.arg: kw.value for kw in node.keywords}

    def test_weak_hash_constructors_declare_usedforsecurity_false(self):
        found = list(self._weak_hash_calls())
        self.assertTrue(found, "expected at least the fused cache key to use a weak content digest")
        for source_file, lineno, keywords in found:
            with self.subTest(file=source_file.name, line=lineno):
                flag = keywords.get("usedforsecurity")
                self.assertIsNotNone(
                    flag,
                    f"{source_file}:{lineno} calls a weak hash without usedforsecurity=; either it is a "
                    "non-cryptographic digest and should say so, or it is a security control and must "
                    "not be md5/sha1",
                )
                self.assertIsInstance(flag, ast.Constant)
                self.assertIs(flag.value, False)

    def test_the_flag_does_not_change_the_cache_key(self):
        """Existing on-disk cache entries stay valid: usedforsecurity is policy, not digest input."""
        import hashlib

        payload = b"v2|parallel=False|def _fused_probe(x):\n    return x\n"
        self.assertEqual(
            hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16],
            hashlib.sha1(payload).hexdigest()[:16],  # noqa: S324 -- the point is that both spellings agree
        )


if __name__ == "__main__":
    unittest.main()
