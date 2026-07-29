"""The local smoke gate collects only an explicit, bounded manifest."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_smoke

ROOT = Path(__file__).resolve().parents[2]


class SmokeManifestContractTest(unittest.TestCase):
    def test_release_manifest_is_explicit_and_collection_light(self):
        targets = run_smoke.manifest_targets(run_smoke.DEFAULT_MANIFEST)
        self.assertEqual(targets, ("mixle/tests/smoke_test.py",))
        self.assertTrue(all("*" not in target for target in targets))

    def test_manifest_rejects_globs_and_repository_escape(self):
        for target in ("mixle/tests/*_test.py", "../outside_test.py"):
            with self.subTest(target=repr(target)), tempfile.NamedTemporaryFile("w", encoding="utf-8") as stream:
                stream.write(target + "\n")
                stream.flush()
                with self.assertRaises(ValueError):
                    run_smoke.manifest_targets(Path(stream.name))

    def test_runner_enforces_hard_budget_and_explicit_target(self):
        completed = subprocess.CompletedProcess([], 0, "", "")
        with patch("scripts.run_smoke.subprocess.run", return_value=completed) as invoked:
            receipt = run_smoke.run_smoke(("mixle/tests/smoke_test.py",), timeout_seconds=30)
        command = invoked.call_args.args[0]
        self.assertEqual(command[-1], "mixle/tests/smoke_test.py")
        self.assertNotIn("mixle/tests", command)
        self.assertEqual(invoked.call_args.kwargs["timeout"], 30)
        self.assertTrue(receipt["passed"])
        with self.assertRaises(ValueError):
            run_smoke.run_smoke(("mixle/tests/smoke_test.py",), timeout_seconds=31)

    def test_smoke_job_is_a_required_check(self):
        workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        required = (ROOT / ".github" / "release-required-checks.txt").read_text(encoding="utf-8")
        self.assertIn("python scripts/run_smoke.py --out smoke-receipt.json", workflow)
        self.assertIn("timeout-minutes: 1", workflow)
        self.assertIn("smoke / collection-light", required.splitlines())


if __name__ == "__main__":
    unittest.main()
