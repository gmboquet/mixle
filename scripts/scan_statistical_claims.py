#!/usr/bin/env python
"""Find user-facing statistical claims and gate them on carrying their scope.

Seven consecutive external review passes found the same root defect wearing different clothes: a
free-text claim somewhere in the package -- a docstring, an example header, a report label, a test
comment -- stronger than what the mathematics or the experiment establishes. Claims had no
inventory, so each one drifted independently and was found one audit at a time.

This scanner is the claims analogue of ``scan_duplicate_bodies.py`` (Y4.5): it enumerates every
user-facing surface whose text makes a statistical claim, classifies the claim, and checks that the
same surface carries the claim's SCOPE:

* a COVERAGE claim (conformal/``1 - alpha``/coverage-guarantee language) must be accompanied, in
  the same file, by the scope triad: the statement is MARGINAL, it assumes EXCHANGEABILITY, and
  distribution SHIFT voids it;
* a COMPARATIVE claim (beats/fewer-calls/cheaper language in examples and task tests) must be
  accompanied by uncertainty language (a paired design, an interval, an estimand) or explicitly
  reduced wording.

The manifest (``mixle/tests/statistical_claims_manifest.json``) is the reviewed inventory. The
contract test fails when a NEW claim-bearing file appears outside it (un-audited claim site), when
an inventoried file loses its scope tokens (claim outran its scope again), or when an entry goes
stale. Regenerate with ``python scripts/scan_statistical_claims.py --write`` -- and audit what
changed before committing, exactly as with the duplicate-body manifest.

Used two ways:
* ``python scripts/scan_statistical_claims.py`` -- print the current claim sites and any violations;
* ``python scripts/scan_statistical_claims.py --write`` -- regenerate the reviewed manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
MANIFEST = ROOT / "mixle" / "tests" / "statistical_claims_manifest.json"

# Surfaces whose text is user-facing claim material. API autodoc pages render source docstrings,
# which are audited at the source, so docs/api is excluded.
COVERAGE_SURFACES = ("examples", "docs", "mixle/task", "mixle/reason")
COMPARATIVE_SURFACES = ("examples", "mixle/tests")

# A claim LINE is explicit on its own: coverage coupled with guarantee/probability/contract
# vocabulary in the same sentence fragment. File-level context pairing produced false positives
# (an error message saying "requires at least", prose saying a sample "covers" a claim), so the
# line itself must make the statistical statement.
_COVERAGE_CLAIM = re.compile(
    r"coverage\s+(guarantee|contract)"
    r"|cover(s|ed)\b[^.\n]{0,70}probability"
    r"|guarantee[^.\n]{0,50}(coverage|1\s*-\s*alpha)"
    r"|conformally\s+(covered|bounded)"
    r"|finite-sample[^.\n]{0,40}coverage",
    re.IGNORECASE,
)
_SCOPE_TOKENS = (
    re.compile(r"marginal", re.IGNORECASE),
    re.compile(r"exchangeab", re.IGNORECASE),
    re.compile(r"shift", re.IGNORECASE),
)

_COMPARATIVE_CLAIM = re.compile(
    r"fewer[^.\n]{0,30}(calls|labels|queries)"
    r"|cheaper with use"
    r"|beat(s|ing)\s+random"
    r"|same quality[^.\n]{0,40}(fewer|less)",
    re.IGNORECASE,
)
_UNCERTAINTY_TOKENS = re.compile(
    r"paired|confidence interval|95% (ci|interval)|ci95|standard error|estimand|uncertainty",
    re.IGNORECASE,
)


def _files(surfaces: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for surface in surfaces:
        base = ROOT / surface
        if not base.exists():
            continue
        patterns = ("*.py", "*.rst") if surface == "docs" else ("*.py",)
        for pattern in patterns:
            out.extend(p for p in sorted(base.rglob(pattern)) if "docs/api" not in p.as_posix())
    return out


def scan(root: Path | None = None) -> dict:
    """Return {relative_path: {"classes": [...], "violations": [...]}} for every claim site."""
    del root  # single-tree tool; parameter kept for parity with the duplicate-body scanner
    sites: dict[str, dict] = {}

    for path in _files(COVERAGE_SURFACES):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not _COVERAGE_CLAIM.search(text):
            continue
        rel = path.relative_to(ROOT).as_posix()
        missing = [token.pattern for token in _SCOPE_TOKENS if not token.search(text)]
        entry = sites.setdefault(rel, {"classes": [], "violations": []})
        entry["classes"].append("coverage")
        if missing:
            entry["violations"].append(f"coverage claim without scope tokens: missing {missing}")

    for path in _files(COMPARATIVE_SURFACES):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not _COMPARATIVE_CLAIM.search(text):
            continue
        rel = path.relative_to(ROOT).as_posix()
        entry = sites.setdefault(rel, {"classes": [], "violations": []})
        entry["classes"].append("comparative")
        if not _UNCERTAINTY_TOKENS.search(text):
            entry["violations"].append(
                "comparative claim without uncertainty language (paired/interval/estimand) or reduced wording"
            )

    return dict(sorted(sites.items()))


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the reviewed manifest")
    args = parser.parse_args(argv)

    sites = scan()
    violations = {rel: e["violations"] for rel, e in sites.items() if e["violations"]}
    for rel, entry in sites.items():
        marker = " !! " if entry["violations"] else "    "
        print(f"{marker}{rel}: {sorted(set(entry['classes']))}")
        for violation in entry["violations"]:
            print(f"        {violation}")

    if args.write:
        record = {
            "_comment": (
                "Reviewed inventory of user-facing statistical-claim sites. A new entry means a new "
                "claim surface: audit its scope, then regenerate with "
                "python scripts/scan_statistical_claims.py --write. The contract test refuses "
                "un-inventoried claim sites and inventoried sites whose scope tokens disappear."
            ),
            "sites": {rel: sorted(set(entry["classes"])) for rel, entry in sites.items()},
        }
        if violations:
            print(f"\nrefusing to write a manifest containing {len(violations)} violation(s); fix them first")
            return 1
        MANIFEST.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {MANIFEST} with {len(sites)} claim site(s)")
        return 0

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
