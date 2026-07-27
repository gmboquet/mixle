"""Serialization schema profiles are exact, dependency-attested, and per-type versioned."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

from mixle.utils.serialization import OBJECT_SCHEMA_VERSION, TAG, serializable_schema_records

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests" / "serialization_schema_manifest.json"


class SchemaManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_envelope_is_versioned(self):
        self.assertEqual(self.manifest["artifact"], "mixle.serialization_schema_manifest/v2")
        self.assertEqual(self.manifest["tag"], TAG)
        self.assertEqual(self.manifest["object_envelope_version"], OBJECT_SCHEMA_VERSION)
        self.assertEqual(set(self.manifest["profiles"]), {"base", "full"})

    def test_base_profile_is_exact_in_every_environment(self):
        expected = serializable_schema_records("base")
        recorded = self.manifest["profiles"]["base"]
        self.assertEqual(recorded["required_imports"], ["numpy", "scipy"])
        self.assertEqual(recorded["registered_types"], sorted(expected))
        self.assertEqual(recorded["schemas"], expected)

    def test_full_profile_is_exact_when_its_inventory_is_present(self):
        recorded = self.manifest["profiles"]["full"]
        required = recorded["required_imports"]
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            self.skipTest(f"full serialization profile dependencies absent: {missing}")
        expected = serializable_schema_records("full")
        self.assertEqual(recorded["registered_types"], sorted(expected))
        self.assertEqual(recorded["schemas"], expected)

    def test_every_type_has_an_explicit_schema_classification(self):
        for profile_name, profile in self.manifest["profiles"].items():
            with self.subTest(profile=profile_name):
                self.assertEqual(set(profile["registered_types"]), set(profile["schemas"]))
                for type_id, schema in profile["schemas"].items():
                    self.assertEqual(schema["state_version"], OBJECT_SCHEMA_VERSION, type_id)
                    self.assertIn(schema["stability"], {"stable", "provisional"}, type_id)
                    self.assertIn(schema["codec"], {"class-owned", "constructor-validated"}, type_id)
                    if schema["stability"] == "stable":
                        self.assertTrue(schema["state_fields"], type_id)
                        self.assertEqual(schema["migrations"], ["0->1"], type_id)
                    else:
                        self.assertIsNone(schema["state_fields"], type_id)
                        self.assertEqual(schema["migrations"], [], type_id)


if __name__ == "__main__":
    unittest.main()
