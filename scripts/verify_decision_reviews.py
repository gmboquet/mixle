#!/usr/bin/env python
"""Fail release sign-off while any decision lacks accepted or superseding review evidence."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ENTRY = re.compile(r"^## (D-[0-9]{4})\b", re.MULTILINE)
_REVIEWER = re.compile(r"^- \*\*Reviewer:\*\* (.+)$", re.MULTILINE)
_ACCEPTED = re.compile(r"^accepted — .+; [0-9]{4}-[0-9]{2}-[0-9]{2}; (?:PR #[0-9]+|https://\S+|commit [0-9a-f]{40})$")
_SUPERSEDED = re.compile(r"^superseded — D-[0-9]{4}$")


def unresolved_reviews(text: str) -> list[str]:
    starts = list(_ENTRY.finditer(text))
    unresolved = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        reviewers = _REVIEWER.findall(text[match.start() : end])
        if len(reviewers) != 1 or not (_ACCEPTED.fullmatch(reviewers[0]) or _SUPERSEDED.fullmatch(reviewers[0])):
            unresolved.append(match.group(1))
    if not starts:
        raise ValueError("decision log contains no decision entries")
    return unresolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision_log", type=Path)
    args = parser.parse_args(argv)
    try:
        unresolved = unresolved_reviews(args.decision_log.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if unresolved:
        print("release decisions lack accepted review evidence: " + ", ".join(unresolved), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
