"""Architectural guard: the dependency graph must stay strictly ``mixle.ppl -> core``, never reverse.

The PPL layer (mixle.ppl) is an optional top layer that builds on the core (stats / inference / etc.).
No core module may import from mixle.ppl -- doing so couples the core to the optional torch-backed PPL
and breaks ``import mixle.stats`` standalone. This test scans the source statically so a new upward
import fails CI instead of silently re-entangling the layers.
"""

import ast
import unittest
from pathlib import Path

PYSP_ROOT = Path(__file__).resolve().parent.parent  # .../mixle


def _imports_ppl(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mixle.ppl"):
            return True
        if isinstance(node, ast.Import):
            if any(alias.name == "mixle.ppl" or alias.name.startswith("mixle.ppl.") for alias in node.names):
                return True
    return False


class PplSeparationTest(unittest.TestCase):
    def test_no_core_module_imports_ppl(self):
        offenders = []
        for path in PYSP_ROOT.rglob("*.py"):
            rel = path.relative_to(PYSP_ROOT)
            parts = rel.parts
            if parts[0] in ("ppl", "tests"):  # the PPL layer and the tests may import ppl
                continue
            if _imports_ppl(path):
                offenders.append(str(rel))
        self.assertEqual(
            offenders,
            [],
            f"core modules import upward from mixle.ppl (must stay ppl -> core): {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
