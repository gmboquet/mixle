"""Public-API drift gate (worklist A1.1).

``api_manifest.json`` at the repo root is the reviewed, committed record of everything ``mixle``
exports -- the ``__all__`` of every public package. This test regenerates that manifest from the
current tree and asserts it is unchanged, so that adding, removing, or renaming a public symbol is
never accidental: it forces a manifest diff into the PR, which is exactly the reviewable surface the
0.8.0 feature freeze requires.

When a change intentionally alters the public surface, regenerate and commit::

    python scripts/gen_api_manifest.py

Packages whose ``__all__`` is assembled at runtime are resolved by import; if the current environment
cannot import one (a missing optional dependency), that package is skipped here rather than allowed to
falsely diff against a manifest generated in a fuller environment.
"""

import importlib.util
import json
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "api_manifest.json"
_GEN_PATH = _REPO_ROOT / "scripts" / "gen_api_manifest.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_api_manifest", _GEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicApiManifestTest(unittest.TestCase):
    def test_manifest_files_exist(self):
        self.assertTrue(_MANIFEST_PATH.exists(), "api_manifest.json missing -- run scripts/gen_api_manifest.py")
        self.assertTrue(_GEN_PATH.exists(), "scripts/gen_api_manifest.py missing")

    def test_public_surface_matches_committed_manifest(self):
        committed = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        current = _load_generator().build_manifest(_REPO_ROOT)

        # A runtime-assembled package the current env cannot import is skipped on both sides, so a
        # missing optional dep never turns into a false public-API diff.
        def _resolvable(manifest, key):
            return not (isinstance(manifest.get(key), dict) and "unresolved" in manifest[key])

        keys = sorted(k for k in set(committed) | set(current) if _resolvable(committed, k) and _resolvable(current, k))

        drift = []
        for key in keys:
            want = committed.get(key)
            got = current.get(key)
            if want != got:
                want_set, got_set = set(want or []), set(got or [])
                added = sorted(got_set - want_set)
                removed = sorted(want_set - got_set)
                drift.append(f"  {key}: +{added} -{removed}")

        self.assertFalse(
            drift,
            "public API drifted from api_manifest.json (regenerate with `python scripts/gen_api_manifest.py` "
            "and commit if intended):\n" + "\n".join(drift),
        )


if __name__ == "__main__":
    unittest.main()
