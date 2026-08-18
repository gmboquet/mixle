#!/usr/bin/env python
"""Write ``metadata/release-candidate.json`` -- the record that names WHICH candidate this is.

This is the one producer of that record. ``publish.yml`` calls it in the prepare phase; the receipt
resolver (``scripts/run_repro_entry.py``) and the example-execution manifest builder consume it; the
tests build their fixtures through :func:`release_candidate_record` so producer and consumers cannot
drift apart silently. It used to be an inline one-liner in the workflow that wrote no ``tree``, while
the resolver -- repaired against hand-built review candidates that did carry one -- required it: the
real workflow could not have bound a single receipt. Usage::

    python scripts/release_candidate_record.py \
        --commit "$SHA" --tree "$(git rev-parse 'HEAD^{tree}')" \
        --tag v0.8.0 --version 0.8.0 --workflow-run "$GITHUB_RUN_ID" \
        --out metadata/release-candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARTIFACT = "mixle.release_candidate/v1"


def _hex40(value: object, what: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{what} must be a full lowercase 40-character hexadecimal Git object ID")
    return value


def release_candidate_record(*, commit: str, tree: str, tag: str, version: str, workflow_run: str) -> dict:
    """The candidate-identity record: source commit AND tree, tag, version, and the producing run."""
    for name, value in (("tag", tag), ("version", version), ("workflow_run", workflow_run)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if tag != f"v{version}":
        raise ValueError(f"tag {tag!r} does not name version {version!r}")
    return {
        "artifact": ARTIFACT,
        "commit": _hex40(commit, "commit"),
        "tree": _hex40(tree, "tree"),
        "tag": tag,
        "version": version,
        "workflow_run": workflow_run,
    }


def render(record: dict) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--workflow-run", required=True)
    parser.add_argument("--out", type=Path, default=None, help="write the record here (also printed)")
    args = parser.parse_args(argv)
    try:
        text = render(
            release_candidate_record(
                commit=args.commit,
                tree=args.tree,
                tag=args.tag,
                version=args.version,
                workflow_run=args.workflow_run,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
