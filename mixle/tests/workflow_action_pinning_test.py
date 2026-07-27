"""Release workflows execute third-party actions only at immutable commits."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    "tests.yml",
    "extras-matrix.yml",
    "docs.yml",
    "security.yml",
    "publish.yml",
    "post-publish-verify.yml",
)
ACTION = re.compile(r"^\s*-?\s*uses:\s*([^@\s]+)@([^\s#]+)(?:\s+#\s*(.+))?\s*$")
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def test_release_workflow_actions_are_commit_pinned_and_labeled() -> None:
    checked = 0
    failures: list[str] = []
    for name in WORKFLOWS:
        path = ROOT / ".github" / "workflows" / name
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = ACTION.match(line)
            if match is None:
                continue
            action, revision, label = match.groups()
            if action.startswith("./"):
                continue
            checked += 1
            if FULL_COMMIT.fullmatch(revision) is None:
                failures.append(f"{name}:{lineno}: {action}@{revision} is mutable")
            if not label:
                failures.append(f"{name}:{lineno}: exact action pin has no human-readable version")
    assert checked >= 60
    assert failures == []
