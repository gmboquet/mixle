"""Every callable on an explicitly stable module has a reviewed exact signature."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "gen_stable_signature_manifest.py"
MANIFEST = ROOT / "manifests" / "stable_api_signatures.json"


def _generator():
    spec = importlib.util.spec_from_file_location("gen_stable_signature_manifest", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StableSignatureManifestTest(unittest.TestCase):
    def test_manifest_is_exact(self):
        current = _generator().build_manifest()
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(recorded, current)
        self.assertGreater(len(recorded["entries"]), 25)

    def test_contract_covers_signature_semantics_without_implicit_exceptions(self):
        policy = json.loads(MANIFEST.read_text(encoding="utf-8"))["compatibility"]
        self.assertEqual(
            policy,
            {
                "parameter_order": "exact",
                "parameter_kinds": "exact",
                "defaults": "exact",
                "annotations": "exact",
                "intentional_exceptions": [],
            },
        )

    def test_standalone_check_uses_the_checkout(self):
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--check"],
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
