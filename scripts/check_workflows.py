#!/usr/bin/env python3
"""Parse release workflows and enforce local supply-chain/shell contracts."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    "tests.yml",
    "extras-matrix.yml",
    "docs.yml",
    "security.yml",
    "publish.yml",
    "post-publish-verify.yml",
)
USES = re.compile(r"^\s*-?\s*uses:\s*([^@\s]+)@([^\s#]+)(?:\s+#\s*(.+))?\s*$")
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def validate() -> list[str]:
    errors: list[str] = []
    actions = 0

    def reject_shell_inputs(value: object, name: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "run" and isinstance(child, str) and "${{ inputs." in child:
                    errors.append(f"{name}: workflow input interpolated directly into shell source")
                reject_shell_inputs(child, name)
        elif isinstance(value, list):
            for child in value:
                reject_shell_inputs(child, name)

    for name in WORKFLOWS:
        path = ROOT / ".github" / "workflows" / name
        text = path.read_text(encoding="utf-8")
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            errors.append(f"{name}: invalid YAML: {exc}")
            continue
        if not isinstance(parsed, dict) or "jobs" not in parsed:
            errors.append(f"{name}: workflow has no jobs mapping")
        if not isinstance(parsed, dict) or "permissions" not in parsed:
            errors.append(f"{name}: workflow has no explicit top-level permissions")
        reject_shell_inputs(parsed, name)
        if text.count("actions/checkout@") != text.count("persist-credentials: false"):
            errors.append(f"{name}: every checkout must disable persisted credentials")
        for lineno, line in enumerate(text.splitlines(), 1):
            match = USES.match(line)
            if match is None:
                continue
            action, revision, label = match.groups()
            if action.startswith("./"):
                continue
            actions += 1
            if FULL_COMMIT.fullmatch(revision) is None:
                errors.append(f"{name}:{lineno}: mutable action revision {action}@{revision}")
            if not label:
                errors.append(f"{name}:{lineno}: action pin lacks a version comment")
    if actions < 60:
        errors.append(f"release workflow action inventory unexpectedly small: {actions}")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("\n".join(errors))
        return 1
    print("release workflows parse and satisfy immutable-action/shell-input contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
