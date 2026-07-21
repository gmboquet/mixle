"""Regression test: mixle.epistemic could not be imported in a clean process.

mixle.epistemic.journal -> mixle.epistemic.loop -> mixle.epistemic.likelihood ->
mixle.doe (package __init__) -> mixle.doe.amplify -> mixle.task (package __init__) ->
mixle.task.pilot_ladder -> mixle.epistemic.journal/.loop/.portfolio closed a circular import: by the
time pilot_ladder.py's top-level imports ran, mixle.epistemic.journal (and .loop) were still
mid-import (paused inside their own import statements further up this same chain), so the names
pilot_ladder.py wanted from them did not exist yet.

This only reproduces in a genuinely clean process: pytest's own test collection order happens to
warm sys.modules with mixle.doe/mixle.task/mixle.epistemic via some other, non-cyclic path first,
so `pytest mixle/tests/pilot_ladder_test.py mixle/tests/epistemic_journal_test.py` passed even on
the unfixed code -- only a bare `python -c "import mixle.epistemic"` in a fresh interpreter actually
hit it, matching how the bug was originally reported.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_import(module: str) -> subprocess.CompletedProcess:
    # Pin PYTHONPATH to this worktree's own repo root (this repo's editable-install gotcha: the
    # installed editable package can point at a stale sibling worktree) so a subprocess run from
    # here exercises the source we just edited, not some other checkout.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


class EpistemicCleanImportTestCase(unittest.TestCase):
    def test_mixle_epistemic_imports_cleanly(self):
        result = _clean_import("mixle.epistemic")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixle_doe_imports_cleanly(self):
        result = _clean_import("mixle.doe")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixle_task_imports_cleanly(self):
        result = _clean_import("mixle.task")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixle_task_pilot_ladder_imports_cleanly(self):
        result = _clean_import("mixle.task.pilot_ladder")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reverse_order_still_imports_cleanly(self):
        # import mixle.task before mixle.epistemic -- the cycle is symmetric, so both entry orders
        # must work, not just the one the original report happened to hit first.
        result = subprocess.run(
            [sys.executable, "-c", "import mixle.task; import mixle.epistemic"],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
