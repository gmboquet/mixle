"""Cross-version serialization compatibility (worklist M11.2).

Artifacts serialized by a PREVIOUS released mixle must still load and score under the current code, or
fail with a clear, actionable message -- never a raw ``AttributeError`` / ``ModuleNotFoundError`` or a
silently-wrong result. The fixtures under ``fixtures/v0_7_0/`` were produced by ``mixle==0.7.0``
(``to_json``) together with a probe point and the log-density it produced then; this test loads each via
the current ``from_json`` and asserts it round-trips to the same type and the same log-density.

Regenerate the fixtures only when deliberately changing the compatibility baseline: install the target
released version in a clean env and re-run its ``to_json`` (see the header in ``manifest.json``).
"""

import hashlib
import json
import unittest
from pathlib import Path

from mixle.utils.serialization import ensure_pysp_serialization_registry, from_json

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "v0_7_0"
_SCHEMA_MANIFEST = Path(__file__).resolve().parents[2] / "manifests" / "serialization_schema_manifest.json"


class CrossVersionFixturesTest(unittest.TestCase):
    def setUp(self):
        ensure_pysp_serialization_registry()
        self.manifest = json.loads((_FIXTURES / "manifest.json").read_text(encoding="utf-8"))

    def test_fixture_policy_covers_every_stable_schema_and_family(self):
        schemas = json.loads(_SCHEMA_MANIFEST.read_text(encoding="utf-8"))["profiles"]["full"]["schemas"]
        stable = {type_id for type_id, schema in schemas.items() if schema["stability"] == "stable"}
        cases = self.manifest["cases"]
        covered = {case["schema_id"] for case in cases}
        self.assertEqual(covered, stable)
        required_families = set(self.manifest["coverage_policy"]["required_families"])
        self.assertEqual({case["family"] for case in cases}, required_families)
        nested = [case for case in cases if case["nested_schema_ids"]]
        self.assertTrue(nested, "compatibility policy requires a representative nested composition")
        for case in nested:
            self.assertTrue(set(case["nested_schema_ids"]) <= stable)

    def test_v0_7_0_artifacts_load_and_score_unchanged(self):
        cases = self.manifest["cases"]
        self.assertTrue(cases, "no cross-version fixtures found")
        for case in cases:
            with self.subTest(case=repr(case["name"])):
                fixture_path = _FIXTURES / case["file"]
                fixture_bytes = fixture_path.read_bytes()
                self.assertEqual(hashlib.sha256(fixture_bytes).hexdigest(), case["sha256"])
                payload = fixture_bytes.decode("utf-8")
                try:
                    dist = from_json(payload)
                except Exception as exc:  # noqa: BLE001
                    self.fail(
                        f"{case['name']}: a v0.7.0 artifact no longer deserializes under the current code "
                        f"({type(exc).__name__}: {exc}). If this is an intended compatibility break, it must "
                        f"raise a clear versioned error, not this."
                    )
                self.assertEqual(type(dist).__name__, case["type"], f"{case['name']}: deserialized to the wrong type")
                got = float(dist.log_density(case["probe"]))
                self.assertAlmostEqual(
                    got,
                    case["log_density"],
                    places=9,
                    msg=f"{case['name']}: log_density drifted from the v0.7.0 value",
                )


if __name__ == "__main__":
    unittest.main()
