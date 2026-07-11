"""Worklist Y4.5 -- find copy-pasted method/function bodies before they cause sibling bugs.

Duplicated logic is a sibling-bug factory: a fix applied to one copy silently leaves the
other broken. This scanner walks the distribution-family and neural-leaf source trees,
normalizes each function body (docstring stripped, structure compared via ``ast.dump``),
and reports groups where the *same* non-trivial body appears in two or more different
functions.

It is intentionally exact-match (true copy-paste), not fuzzy: exact duplicates are the
high-signal subset worth gating, and the gate must have near-zero false positives to be
trusted. Bodies shorter than ``MIN_STMTS`` statements are ignored so trivial one-liners
(``return self._x``) do not flood the report.

Used two ways:
* ``python scripts/scan_duplicate_bodies.py`` -- print the current duplicate groups;
* ``python scripts/scan_duplicate_bodies.py --write`` -- (re)generate the reviewed
  manifest that the drift-gate test compares against.

The test in ``mixle/tests/duplicate_body_scan_test.py`` fails when a *new* duplicate
appears -- the shared prevention mechanism Y4.5 asks for.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

# Directories the worklist names: distribution families and neural leaves.
TARGET_DIRS = ("mixle/stats", "mixle/models")
MIN_STMTS = 6  # ignore trivial bodies

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
MANIFEST = ROOT / "mixle" / "tests" / "duplicate_body_manifest.json"


def _repo_root() -> Path:
    return ROOT


def _normalized_body(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Structural signature of a function body, or None if it is too short to gate."""
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # drop the docstring
    if len(body) < MIN_STMTS:
        return None
    return "\n".join(ast.dump(stmt) for stmt in body)


def scan(root: Path | None = None) -> list[dict]:
    """Return duplicate-body groups as a sorted, JSON-serializable list.

    Each group: ``{"locations": ["relpath::func", ...], "stmts": <int>}`` where
    ``locations`` is the sorted set of distinct functions sharing one body.
    """
    root = root or _repo_root()
    by_signature: dict[str, set[tuple[str, str]]] = defaultdict(set)
    stmts_of: dict[str, int] = {}

    for target in TARGET_DIRS:
        for path in sorted((root / target).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            rel = path.relative_to(root).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = _normalized_body(node)
                    if sig is None:
                        continue
                    by_signature[sig].add((rel, node.name))
                    stmts_of[sig] = len(node.body)

    groups = []
    for sig, locs in by_signature.items():
        if len(locs) < 2:
            continue
        groups.append(
            {
                "locations": sorted(f"{rel}::{name}" for rel, name in locs),
                "stmts": stmts_of[sig],
            }
        )
    # Stable order: by first location, so the manifest diff is readable.
    groups.sort(key=lambda g: g["locations"])
    return groups


def group_signatures(groups: list[dict]) -> set[tuple[str, ...]]:
    """A hashable identity per group -- its set of locations -- for set comparison."""
    return {tuple(g["locations"]) for g in groups}


def load_manifest() -> list[dict]:
    if not MANIFEST.is_file():
        return []
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["groups"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="(re)write the reviewed manifest")
    args = parser.parse_args()

    groups = scan()
    if args.write:
        MANIFEST.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Worklist Y4.5 baseline of exact copy-pasted bodies in "
                        "mixle/stats and mixle/models. A new entry means new copy-paste: "
                        "dedupe it, or add it here with justification. Regenerate with "
                        "python scripts/scan_duplicate_bodies.py --write."
                    ),
                    "min_stmts": MIN_STMTS,
                    "groups": groups,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {MANIFEST} with {len(groups)} duplicate groups")
        return 0

    print(f"{len(groups)} duplicate-body groups (>= {MIN_STMTS} stmts):")
    for g in groups:
        print(f"  [{g['stmts']} stmts] {g['locations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
