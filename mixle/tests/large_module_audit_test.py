"""The large-module audit stays honest (worklist A1.7).

``docs/large-module-audit.rst`` records a risk/test map for every module over 1,500 lines. An audit is only
useful if it cannot silently drift: this pins the audited set to reality. A module that grows past the
threshold without an audit entry -- or an audited path that has been renamed/deleted -- fails here, which is
the prompt to update the page (not to blindly split the module: the audit's own rule is to refactor only to
remove a demonstrated defect).
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "large-module-audit.rst"
LINE_THRESHOLD = 1500


def _large_modules():
    found = set()
    for path in (REPO_ROOT / "mixle").rglob("*.py"):
        if "tests" in path.parts:
            continue
        with path.open() as f:
            if sum(1 for _ in f) > LINE_THRESHOLD:
                found.add(path.relative_to(REPO_ROOT).as_posix())
    return found


def _paths_mentioned_in_doc():
    return set(re.findall(r"mixle/[\w/]+\.py", DOC.read_text()))


class LargeModuleAuditTest(unittest.TestCase):
    def test_every_large_module_is_audited(self):
        missing = sorted(_large_modules() - _paths_mentioned_in_doc())
        self.assertEqual(
            missing,
            [],
            f"modules over {LINE_THRESHOLD} lines with no entry in docs/large-module-audit.rst:\n" + "\n".join(missing),
        )

    def test_no_audited_path_is_stale(self):
        # Every full module path named in the audit must still exist (catch a renamed/deleted module).
        stale = sorted(p for p in _paths_mentioned_in_doc() if not (REPO_ROOT / p).exists())
        self.assertEqual(stale, [], "audit references paths that no longer exist:\n" + "\n".join(stale))


if __name__ == "__main__":
    unittest.main()
