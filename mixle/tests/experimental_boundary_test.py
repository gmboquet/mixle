"""Stable code must not statically import ``mixle.experimental`` (worklist Y4.4 / A1.4).

Experimental mechanisms live under ``mixle.experimental`` and must not enter the stable import graph --
or the stable public surface -- merely by being importable. This gate scans every non-experimental,
non-test module and fails if it contains a static ``import mixle.experimental...`` or
``from mixle.experimental... import ...`` (absolute or relative).

A deliberate deprecation shim (e.g. ``mixle.program``) may still bridge to the experimental module the
API moved to, but only through a dynamic ``importlib.import_module(...)`` call on a string -- which
keeps ``mixle.experimental`` out of the static stable-import graph and makes the bridge an explicit,
scoped act rather than a silent dependency. This runs by AST, so it needs no imports and holds in any
environment; it runs in the standard fast/full CI gates.
"""

import ast
import unittest
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
_TARGET = "mixle.experimental"


def _module_name(path: Path) -> str:
    rel = path.relative_to(_PKG_ROOT.parent)  # e.g. mixle/utils/parallel/foo.py
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve(node: ast.ImportFrom, module_name: str) -> str:
    """Absolute module a ``from ... import`` targets, resolving a relative import against its file."""
    if not node.level:
        return node.module or ""
    base = module_name.split(".")[: -node.level]  # level 1 == the file's own package
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base)


def _experimental_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_name = _module_name(path)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _TARGET or alias.name.startswith(_TARGET + "."):
                    hits.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            target = _resolve(node, module_name)
            if target == _TARGET or target.startswith(_TARGET + "."):
                hits.append((node.lineno, target))
    return hits


class ExperimentalBoundaryTest(unittest.TestCase):
    def test_stable_modules_do_not_statically_import_experimental(self):
        offenders: list[str] = []
        for path in sorted(_PKG_ROOT.rglob("*.py")):
            rel = path.relative_to(_PKG_ROOT)
            if rel.parts[0] == "experimental" or "tests" in rel.parts:
                continue
            for lineno, target in _experimental_imports(path):
                offenders.append(f"  {path.relative_to(_PKG_ROOT.parent)}:{lineno}: imports {target}")
        self.assertFalse(
            offenders,
            "stable modules must not statically import mixle.experimental (bridge via "
            "importlib.import_module on a string in an explicit deprecation shim instead):\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
