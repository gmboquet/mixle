#!/usr/bin/env python3
"""Reject publication while the candidate's own documents call it unreleased."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def validate(root: Path, version: str) -> list[str]:
    errors: list[str] = []
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if re.search(rf"^## \[{re.escape(version)}\] — \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.MULTILINE) is None:
        errors.append(f"CHANGELOG.md has no dated {version} release heading")
    docs_changelog = (root / "docs" / "changelog.rst").read_text(encoding="utf-8")
    if re.search(rf"(?m)^{re.escape(version)}\n[-=]+$", docs_changelog) is None:
        errors.append(f"docs/changelog.rst has no {version} release section")
    migration = (root / "docs" / "migrations" / f"{version}.md").read_text(encoding="utf-8").lower()
    if "development draft" in migration or "not yet published" in migration:
        errors.append(f"{version} migration guide still declares a development draft")
    charter = (root / "docs" / "charter.md").read_text(encoding="utf-8").lower()
    if f"version {version} is an active development target" in charter:
        errors.append(f"charter still declares {version} an active development target")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        errors = validate(args.root, args.version)
    except OSError as exc:
        errors = [str(exc)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"release documents consistently describe {args.version} as released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
