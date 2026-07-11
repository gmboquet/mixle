"""The serialization schema manifest records every serializable type (worklist M11.1).

``serialization_schema_manifest.json`` is the versioned catalog of ``__pysp_type__`` ids -- the serialization
schema surface a saved artifact's loadability depends on. This gate makes a schema change visible: a new
serializable type that is not recorded fails here, prompting a regenerate (and, per M11.2, a cross-version
fixture). The manifest is generated with the optional backends installed, so it is the *full* surface; this
test tolerates a base environment (torch/jax/numba absent) by requiring the live set to be a subset of the
recorded manifest, and only demands exact equality when torch is present.
"""

import json
import unittest
from pathlib import Path

from mixle.utils.serialization import TAG, serializable_class_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "serialization_schema_manifest.json"


def _has_torch():
    import importlib.util

    return importlib.util.find_spec("torch") is not None


class SchemaManifestTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text())
        self.recorded = set(self.manifest["registered_types"])
        self.live = set(serializable_class_ids())

    def test_tag_and_version_recorded(self):
        self.assertEqual(self.manifest["tag"], TAG)
        self.assertTrue(self.manifest["schema_manifest_version"])
        self.assertGreater(len(self.recorded), 100)  # the schema surface is substantial, not empty

    def test_no_unrecorded_serializable_type(self):
        # Every type registered in THIS environment must be in the manifest. Catches a new serializable
        # class added without regenerating -- an unrecorded schema-surface change. Holds in any environment
        # (a base env registers a subset of the full manifest).
        unrecorded = sorted(self.live - self.recorded)
        self.assertEqual(
            unrecorded,
            [],
            "serializable types are not in serialization_schema_manifest.json (run "
            "python scripts/gen_schema_manifest.py):\n" + "\n".join(unrecorded),
        )

    def test_exact_when_full_surface_present(self):
        # With torch installed the live set is the full surface, so the manifest must match exactly --
        # this direction catches a stale entry (a removed/renamed type still recorded).
        if not _has_torch():
            self.skipTest("base environment: manifest is a superset of the live subset (see subset test)")
        stale = sorted(self.recorded - self.live)
        self.assertEqual(stale, [], "manifest records types no longer registered; regenerate:\n" + "\n".join(stale))


if __name__ == "__main__":
    unittest.main()
