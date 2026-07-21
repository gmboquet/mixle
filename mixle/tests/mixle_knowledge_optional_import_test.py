"""Regression test: pytest.importorskip only guards the import, not attribute access after it.

`substrate_knowledge_bundle_test.py` and `test_n2_habitat_constraints.py` both do
`pytest.importorskip("mixle_knowledge.contracts")` then immediately read specific attributes off the
returned module. If mixle-knowledge is installed but an older/incompatible version lacking those
attributes, that raised a bare AttributeError at collection time -- which aborted the ENTIRE pytest
session (every file, not just those two), since importorskip only catches the module failing to import.
Both files now guard with an explicit hasattr check + pytest.skip(allow_module_level=True) first.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_FILES = [
    "mixle/tests/substrate_knowledge_bundle_test.py",
    "mixle/tests/test_n2_habitat_constraints.py",
]


def _collect(extra_pythonpath: str | None) -> subprocess.CompletedProcess:
    # Always pin PYTHONPATH to this worktree's own repo root first, matching this repo's editable-install
    # gotcha (the installed editable package can point at a stale sibling worktree) -- otherwise a
    # subprocess run from here could silently exercise different source than the file we just edited.
    env = dict(os.environ)
    path_entries = [str(REPO_ROOT)] + ([extra_pythonpath] if extra_pythonpath else [])
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    return subprocess.run(
        [sys.executable, "-m", "pytest", *TARGET_FILES, "--collect-only", "-q", "-m", ""],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _collect_with_fake_contracts_module(tmp_path, contracts_body: str) -> subprocess.CompletedProcess:
    pkg_dir = tmp_path / "mixle_knowledge"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "contracts.py").write_text(textwrap.dedent(contracts_body))
    return _collect(str(tmp_path))


class MixleKnowledgeOptionalImportTestCase(unittest.TestCase):
    def test_incompatible_installed_package_skips_cleanly_instead_of_aborting_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _collect_with_fake_contracts_module(Path(tmp), "class SomeUnrelatedSymbol:\n    pass\n")

        self.assertNotIn("Interrupted", result.stdout, result.stdout)
        self.assertNotIn("AttributeError", result.stdout, result.stdout)
        self.assertIn("SKIPPED", result.stdout, result.stdout)
        self.assertIn("no tests collected", result.stdout, result.stdout)

    def test_compatible_installed_package_collects_normally(self):
        body = """
            class KnowledgeBundle:
                pass
            class CriticalHabitatDesignation:
                pass
            class ListedSpecies:
                pass
            class SourceRef:
                pass
            class SpatialBounds:
                pass
            """
        with tempfile.TemporaryDirectory() as tmp:
            result = _collect_with_fake_contracts_module(Path(tmp), body)

        self.assertNotIn("Interrupted", result.stdout, result.stdout)
        self.assertNotIn("error", result.stdout.lower(), result.stdout)
        self.assertIn("tests collected", result.stdout, result.stdout)
        self.assertNotIn("no tests collected", result.stdout, result.stdout)

    def test_missing_package_still_skips_as_before(self):
        if importlib.util.find_spec("mixle_knowledge") is not None:
            self.skipTest("mixle-knowledge is installed in this environment; not the base case")

        result = _collect(None)
        self.assertNotIn("Interrupted", result.stdout, result.stdout)
        self.assertIn("SKIPPED", result.stdout, result.stdout)


if __name__ == "__main__":
    unittest.main()
