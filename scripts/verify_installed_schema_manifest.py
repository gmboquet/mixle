#!/usr/bin/env python3
"""Verify one committed serialization profile against the imported installation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from mixle.utils.serialization import OBJECT_SCHEMA_VERSION, TAG, serializable_schema_records


def validate(manifest_path: Path, profile: str) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("artifact") != "mixle.serialization_schema_manifest/v2":
        return ["unsupported serialization schema manifest"]
    if manifest.get("tag") != TAG or manifest.get("object_envelope_version") != OBJECT_SCHEMA_VERSION:
        errors.append("serialization envelope identity differs")
    recorded = manifest.get("profiles", {}).get(profile)
    if not isinstance(recorded, dict):
        return [f"serialization profile {profile!r} is absent"]
    missing = [name for name in recorded.get("required_imports", []) if importlib.util.find_spec(name) is None]
    if missing:
        return [f"serialization profile {profile!r} is missing imports: {missing}"]
    expected = serializable_schema_records(profile)
    if recorded.get("registered_types") != sorted(expected):
        errors.append(f"serialization profile {profile!r} added or removed registered types")
    if recorded.get("schemas") != expected:
        errors.append(f"serialization profile {profile!r} schema records differ")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", choices=("base", "full"), required=True)
    args = parser.parse_args()
    try:
        errors = validate(args.manifest, args.profile)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors = [str(exc)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"{args.profile} serialization schema matches the imported installation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
