#!/usr/bin/env python
"""Render or check CONTRIBUTING.md's summary of the authoritative development policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "manifests" / "development_policy.json"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
BEGIN = "<!-- BEGIN GENERATED DEVELOPMENT POLICY -->"
END = "<!-- END GENERATED DEVELOPMENT POLICY -->"


def render(policy: dict) -> str:
    release = policy["release"]
    validation = policy["validation"]
    compatibility = policy["compatibility"]
    tiers = ", ".join(f"`{tier}`" for tier in validation["test_tiers"])
    blocking = "\n".join(f"- {gate}" for gate in validation["blocking"])
    advisory = "\n".join(f"- {gate}" for gate in validation["advisory"])
    return f"""\
{BEGIN}
## Authoritative development policy

This summary is generated from `manifests/development_policy.json`; edit the manifest and rerun
`python scripts/render_contributing_policy.py` rather than changing this block by hand.

Current work targets `{release["target_branch"]}` and milestone `{release["milestone"]}`. Automated
dependency updates target the same branch. Retarget both the manifest and Dependabot deliberately
when the release line changes.

Local diagnostics must select the affected node and finish within
{validation["local_command_timeout_seconds"]} seconds. Hosted validation owns broader execution.
The execution tiers are {tiers}.

Blocking validation:

{blocking}

Advisory validation:

{advisory}

Public API maturity has three levels:

- **stable** — {compatibility["stable"]}
- **provisional** — {compatibility["provisional"]}
- **experimental** — {compatibility["experimental"]}

Stable deprecations remain functional for at least
{compatibility["stable_deprecation_min_minor_releases"]} minor releases after announcement, emit
`DeprecationWarning`, name their replacement/removal release, and ship migration guidance. Genuine
security or data-corruption repairs may fail closed immediately, but must be documented explicitly.
{END}"""


def replace_block(document: str, generated: str) -> str:
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise ValueError("CONTRIBUTING.md must contain exactly one generated policy block")
    before, remainder = document.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    return before + generated + after


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    current = CONTRIBUTING.read_text(encoding="utf-8")
    expected = replace_block(current, render(policy))
    if args.check:
        if current != expected:
            print("CONTRIBUTING.md policy summary is stale")
            return 1
        return 0
    CONTRIBUTING.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
