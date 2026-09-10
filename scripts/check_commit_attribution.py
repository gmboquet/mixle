#!/usr/bin/env python3
"""Fail closed when a commit in the release range carries authorship this project does not claim.

The release ledger records author/committer identities, but that check reads only ``%an``/``%cn``:
a ``Co-Authored-By:`` trailer naming a tool sits in the message BODY and passes it untouched. The
0.7.0 line shipped with forty such trailers and needed a history rewrite to remove them, which
changed every commit SHA from mid-2026 -- so this is expensive to fix late and cheap to prevent.

The rules deliberately separate ATTRIBUTION from SUBJECT MATTER. This library legitimately discusses
``openai/clip-vit-base-patch32``, ``OpenAICompatLLM``, ``mixle/reason/llm.py``, ``REQ-AI-OPERABILITY``
and "ten AI adversarial reviews"; a check that flagged those would be turned off within a week. What
is refused is a claim of authorship: an unrecognised ``*-by:`` trailer, a tool's e-mail address, a
generated-by advertisement, or a bare vendor name, which has no reason to appear in this project's
commit messages at all.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Identities permitted in each role. They differ by role on purpose: GitHub's own mailbox is the
# COMMITTER of anything merged through the web UI, which is ordinary, but it authoring or
# co-authoring a commit would not be. dependabot's is GitHub's bot on merged dependency PRs, not a
# co-author this project added.
OWNER = "Grant Boquet <grant.boquet@gmail.com>"
DEPENDABOT = "dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>"
ALLOWED_BY_ROLE = {
    "author": frozenset({OWNER, DEPENDABOT}),
    "committer": frozenset({OWNER, DEPENDABOT, "GitHub <noreply@github.com>"}),
    "co-author": frozenset({OWNER, DEPENDABOT}),
}

TRAILER = re.compile(r"^\s*([A-Za-z][A-Za-z-]*)-by:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# Attribution markers, not topic words: a vendor name, a tool mailbox, or a generated-by banner.
MARKERS = (
    (re.compile(r"\bclaude\b", re.IGNORECASE), "names Claude"),
    (re.compile(r"anthropic", re.IGNORECASE), "names Anthropic"),
    (re.compile(r"\bchat\s?gpt\b", re.IGNORECASE), "names ChatGPT"),
    (re.compile(r"\bco-?pilot\b", re.IGNORECASE), "names Copilot"),
    (re.compile(r"\U0001F916"), "carries the generated-by robot marker"),
    (re.compile(r"claude\.(com|ai)/|copilot\.github\.com|chat\.openai\.com", re.IGNORECASE), "links to a tool"),
    (re.compile(r"noreply@(anthropic|openai)\.com", re.IGNORECASE), "carries a tool mailbox"),
)
# Deliberately NOT a rule: the bare phrase "generated with". This history says "Regenerated with
# scripts/gen_schema_manifest.py", "regenerated with zero drift", and "generated with optional
# backends installed" -- three ordinary sentences that a phrase rule flags and a reviewer then
# learns to ignore. The banner this is meant to catch advertises a vendor and links to it, and both
# of those are caught above on their own.


def _log(base: str, head: str) -> list[tuple[str, str, str, str]]:
    out = subprocess.run(
        ["git", "log", "--format=%H%x1f%an <%ae>%x1f%cn <%ce>%x1f%B%x1e", f"{base}..{head}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    commits = []
    for chunk in out.split("\x1e"):
        if not chunk.strip():
            continue
        fields = chunk.strip("\n").split("\x1f")
        if len(fields) >= 4:
            commits.append((fields[0], fields[1], fields[2], fields[3]))
    return commits


def violations(base: str, head: str = "HEAD") -> list[str]:
    found: list[str] = []
    for sha, author, committer, body in _log(base, head):
        short = sha[:9]
        for role, identity in (("author", author), ("committer", committer)):
            if identity not in ALLOWED_BY_ROLE[role]:
                found.append(f"{short}: {role} {identity!r} is not an allowed identity")
        for match in TRAILER.finditer(body):
            kind, identity = match.group(1), match.group(2)
            if kind.lower() == "signed-off":
                continue
            if identity not in ALLOWED_BY_ROLE["co-author"]:
                found.append(f"{short}: {kind}-by trailer {identity!r} is not an allowed identity")
        for pattern, why in MARKERS:
            for hit in pattern.finditer(body):
                line = body[: hit.start()].count("\n")
                text = body.splitlines()[line].strip() if line < len(body.splitlines()) else ""
                found.append(f"{short}: message {why}: {text[:120]!r}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="exclusive lower bound, e.g. the previous release tag")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args(argv)
    try:
        found = violations(args.base, args.head)
    except subprocess.CalledProcessError as exc:
        print(f"cannot read {args.base}..{args.head}: {exc}", file=sys.stderr)
        return 1
    for line in found:
        print(line, file=sys.stderr)
    if found:
        print(f"{len(found)} attribution violation(s) in {args.base}..{args.head}", file=sys.stderr)
        return 1
    print(f"no attribution violations in {args.base}..{args.head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
