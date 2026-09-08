#!/usr/bin/env python3
"""Reject publication while the candidate's own documents call it unreleased."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PRE_RELEASE_SUFFIX = re.compile(r"(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$")


def base_release(version: str) -> str:
    """The release a PEP 440 pre-release rehearses: ``0.8.2rc1`` -> ``0.8.2``.

    A release candidate is cut from the release tree with only the version string changed, so its
    changelog heading, docs changelog section and migration guide are the final release's (D-0216).
    """
    return PRE_RELEASE_SUFFIX.sub("", version, count=1)


def validate(root: Path, version: str) -> list[str]:
    errors: list[str] = []
    version = base_release(version)
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
