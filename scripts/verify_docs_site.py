#!/usr/bin/env python3
"""Validate the exact assembled documentation tree before Pages upload."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VERSION = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")


def validate(site: Path) -> list[str]:
    errors: list[str] = []
    for required in ("index.html", "main/index.html", "switcher.json", "versions.json", ".nojekyll"):
        if not (site / required).exists():
            errors.append(f"assembled documentation lacks {required}")
    if errors:
        return errors
    switcher = json.loads((site / "switcher.json").read_text(encoding="utf-8"))
    versions = json.loads((site / "versions.json").read_text(encoding="utf-8"))
    if switcher != versions or not switcher or switcher[0].get("name") != "main":
        errors.append("documentation version inventories differ or omit main")
    release_versions: list[str] = []
    for entry in switcher[1:]:
        name = entry.get("name")
        match = VERSION.fullmatch(name) if isinstance(name, str) else None
        if match is None:
            errors.append(f"invalid documentation version entry {name!r}")
            continue
        version = match.group("version")
        release_versions.append(version)
        release_root = site / name
        try:
            metadata = json.loads((release_root / "docs-build.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{name} has invalid build metadata: {exc}")
            continue
        if metadata.get("version") != version or not (release_root / "index.html").is_file():
            errors.append(f"{name} identity or index differs")
    root = (site / "index.html").read_text(encoding="utf-8")
    if release_versions:
        if f"v{release_versions[0]}/index.html" not in root:
            errors.append("root does not target the newest listed stable release")
    elif "development documentation" not in root or "main/index.html" not in root:
        errors.append("no-release root is not an explicit development landing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    try:
        errors = validate(args.site)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors = [str(exc)]
    if errors:
        print("\n".join(errors))
        return 1
    print("assembled documentation site is internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
