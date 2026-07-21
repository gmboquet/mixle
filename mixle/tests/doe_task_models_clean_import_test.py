"""Regression test: two more live import-cycle/reverse-dependency risks found while auditing the
epistemic<->task circular import (see epistemic_clean_import_test.py) for other instances of the same
shape, ahead of the mixle 0.8.0 release.

1. mixle.doe (package init) -> mixle.doe.amplify -> mixle.task.collapse (a top-level import) closed a
   live, bidirectional, eager cycle with mixle.task (package init) -> mixle.task.emulate ->
   mixle.doe.active/.bayesopt/.designs, and mixle.task -> mixle.task.propose -> mixle.doe.oracle. It
   "worked" only because each package's __init__.py happens to import its own unaffected submodules
   before reaching the one that reaches into the other -- reordering either import list (e.g. an
   alphabetize-imports pass, or inserting a new eager submodule earlier in the sequence) would
   reproduce the exact ImportError the epistemic<->task fix addressed, just between doe and task
   instead. Fixed by deferring amplify.py's mixle.task.collapse import the same way pilot_ladder.py's
   fix deferred its mixle.epistemic imports: CollapseVerdict (annotation-only, thanks to
   `from __future__ import annotations`) moved to `if TYPE_CHECKING:`, collapse_monitor (an actual
   runtime call) moved to a local import inside amplify_and_capture(). Breaking this one edge is
   sufficient: mixle.task's imports of mixle.doe are a one-way dependency once mixle.doe no longer
   reaches back, not a cycle.

2. mixle.models.qat did a top-level `from mixle.task.quantize import _QMAX, quantize_dequantize_array`,
   and mixle.models.__init__ eagerly imports qat -- so plain `import mixle.models` (core!) already,
   unconditionally, pulled in all of mixle.task and, transitively, all of mixle.doe. Fixed the same
   way: both names are only ever used inside method bodies (_FakeQuantSTE.forward,
   QATWrapper.__init__), so the import moved to be local to each of those two call sites.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_import(*modules: str) -> subprocess.CompletedProcess:
    # Pin PYTHONPATH to this worktree's own repo root (this repo's editable-install gotcha: the
    # installed editable package can point at a stale sibling worktree) so a subprocess run from
    # here exercises the source we just edited, not some other checkout.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    code = "; ".join(f"import {m}" for m in modules)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


class DoeTaskCleanImportTestCase(unittest.TestCase):
    def test_mixle_doe_imports_cleanly(self):
        result = _clean_import("mixle.doe")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixle_task_imports_cleanly(self):
        result = _clean_import("mixle.task")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_doe_then_task_imports_cleanly(self):
        result = _clean_import("mixle.doe", "mixle.task")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_task_then_doe_imports_cleanly(self):
        result = _clean_import("mixle.task", "mixle.doe")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_doe_amplify_imports_cleanly_on_its_own(self):
        result = _clean_import("mixle.doe.amplify")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_doe_no_longer_eagerly_pulls_in_task(self):
        # the specific fix: mixle.doe.amplify's top-level import of mixle.task.collapse is gone, so
        # importing mixle.doe alone must not have mixle.task in sys.modules at all.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, mixle.doe\n"
                "assert 'mixle.task' not in sys.modules, "
                "'mixle.doe still eagerly imports mixle.task'\n"
                "print('ok')",
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


class ModelsTaskCleanImportTestCase(unittest.TestCase):
    def test_mixle_models_imports_cleanly(self):
        result = _clean_import("mixle.models")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_models_no_longer_eagerly_pulls_in_task(self):
        # the specific fix: mixle.models.qat's top-level import of mixle.task.quantize is gone, so
        # importing mixle.models alone must not have mixle.task (or mixle.doe, transitively) in
        # sys.modules at all -- core mixle no longer depends on its own downstream packages just to
        # import.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, mixle.models\n"
                "assert 'mixle.task' not in sys.modules, "
                "'mixle.models still eagerly imports mixle.task'\n"
                "assert 'mixle.doe' not in sys.modules, "
                "'mixle.models still eagerly imports mixle.doe'\n"
                "print('ok')",
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
