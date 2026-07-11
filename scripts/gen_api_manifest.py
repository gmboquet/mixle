#!/usr/bin/env python
"""Generate the public-API manifest (worklist A1.1).

The manifest is the declared public surface of ``mixle``: for every package ``__init__.py`` that
defines ``__all__``, the sorted list of exported names. It is produced by **static AST parsing**, not
by importing anything, so it is deterministic and independent of which optional dependencies happen to
be installed -- the same manifest is produced in a base env, the full env, or CI.

The committed ``api_manifest.json`` is the baseline the drift test
(``mixle/tests/public_api_manifest_test.py``) compares against. When a PR intentionally changes the
public surface, regenerate it::

    python scripts/gen_api_manifest.py

and commit the result -- so that every change to what users can import from ``mixle`` is a reviewed,
recorded diff, per the 0.8.0 feature freeze.
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "api_manifest.json"


def _string_literals(node: ast.expr) -> list[str] | None:
    """The list of string constants in a list/tuple literal, or literal-list concatenations of them.

    Returns ``None`` if ``node`` is not a fully-static literal we can resolve (e.g. a comprehension or a
    reference to another name) -- the caller records that package as dynamically-built rather than
    guessing.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        out: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            else:
                return None
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_literals(node.left)
        right = _string_literals(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def _extract_all(path: Path) -> tuple[list[str], bool]:
    """Return (sorted unique ``__all__`` names, fully_static). ``fully_static`` is False if any
    statement assigning/extending ``__all__`` used a non-literal value we could not resolve."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    fully_static = True
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AugAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        literals = _string_literals(value)
        if literals is None:
            fully_static = False
        else:
            names.extend(literals)
    return sorted(set(names)), fully_static


def build_manifest(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """Map dotted package name -> its public ``__all__``, across every ``mixle`` package
    ``__init__.py`` that declares one.

    A package whose ``__all__`` is a static literal is resolved by AST parse (no import). A package
    that assembles ``__all__`` at runtime (``mixle``, ``mixle.stats``, ``mixle.utils``) is resolved by
    importing it; if that import fails (a missing optional dependency in the current env) it is
    recorded as ``{"unresolved": <ExceptionType>}`` so the drift test can skip it rather than falsely
    diff against a manifest generated in a fuller env."""
    pkg_root = repo_root / "mixle"
    manifest: dict[str, object] = {}
    for init in sorted(pkg_root.rglob("__init__.py")):
        names, fully_static = _extract_all(init)
        if not names and fully_static:
            continue  # no __all__ declared here
        dotted = ".".join(init.relative_to(repo_root).parent.parts)
        if fully_static:
            manifest[dotted] = names
        else:
            try:
                module = importlib.import_module(dotted)
                manifest[dotted] = sorted(getattr(module, "__all__", []))
            except Exception as exc:  # noqa: BLE001 -- a missing optional dep must not break the manifest
                manifest[dotted] = {"unresolved": type(exc).__name__}
    return manifest


def main() -> int:
    manifest = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(len(v if isinstance(v, list) else v["names"]) for v in manifest.values())
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}: {len(manifest)} packages, {total} public names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
