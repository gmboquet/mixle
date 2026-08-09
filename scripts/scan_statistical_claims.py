#!/usr/bin/env python
"""Gate the ENFORCED statistical-claim phrase classes on carrying their scope.

What this is, exactly (STAT-RR16-3 forced the precision): a LEXICAL ratchet over three explicit
phrase classes, not a proof of claim validity and not a complete enumeration of every sentence a
reader might take as a claim. Token presence cannot validate a statistical method -- the exact
paired-rule tests in ``task_active_test.py`` are what validate a conclusion rule; this gate only
guarantees that text matching the enforced classes never ships without its scope language, and
that new files entering those classes get a human audit (the manifest is the reviewed record of
that audit). Its coverage grows by widening the phrase classes; the negative controls in the
contract test pin what each class does and does not match.

The four enforced classes:

* COVERAGE (conformal coverage-guarantee phrasing in examples/docs/task/reason surfaces): the
  same file must state the scope triad -- MARGINAL, EXCHANGEABILITY assumed, SHIFT voids;
* COMPARATIVE (beats/fewer-calls/cheaper/measurably phrasing in examples and task tests): the
  same file must carry uncertainty-or-reduced wording (paired/interval/estimand/exact-test/
  "not an uncertainty-quantified" disclosure);
* COMPETITIVE (named-rival and gap-closing phrasing in examples): the same file must disclose
  its measurement conditions (versions/this-run/if-installed language);
* CERTIFICATION (selective-risk / accepted-error guarantee phrasing in task/reason/docs/example
  surfaces): the same file must state the sampling premise (i.i.d./exchangeability), the failure
  probability over the calibration draw, and the shift voiding.

The manifest (``mixle/tests/statistical_claims_manifest.json``) is the reviewed inventory of
files currently in these classes. The contract test fails when a NEW file enters a class
un-audited, when an inventoried file loses its scope tokens, or when an entry goes stale.
Regenerate with ``python scripts/scan_statistical_claims.py --write`` and audit the change.

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
# which are audited at the source, so docs/api is excluded. mixle/inference joined after a sweep
# found uq.py claiming finite-sample coverage from an unscanned surface.
COVERAGE_SURFACES = ("examples", "docs", "mixle/task", "mixle/reason", "mixle/inference")
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
# A negated mention is a DISCLAIMER, not a claim ("this is a ranking diagnostic, not a conformal
# or bootstrap coverage guarantee") -- requiring the scope triad there would force scope prose
# onto text whose whole point is that no guarantee exists. Suppression is line-local and pinned
# by the contract test's controls.
_NEGATED_COVERAGE = re.compile(
    r"\b(no|not|never|cannot|without( a| any)?|isn'?t|is not)\b[^.\n]{0,80}coverage\s+(guarantee|contract)",
    re.IGNORECASE,
)

_COMPARATIVE_CLAIM = re.compile(
    r"fewer[^.\n]{0,30}(calls|labels|queries)"
    r"|cheaper with use"
    r"|beat(s|ing)\s+random"
    r"|measurably\s+(beats?|better|cheaper|less|worse)"
    r"|(runner-up|winner)[^.\n]{0,60}trails"
    r"|not a close call"
    r"|same quality[^.\n]{0,40}(fewer|less)",
    re.IGNORECASE,
)
_UNCERTAINTY_TOKENS = re.compile(
    r"paired|confidence interval|95% (ci|interval)|ci95|standard error|estimand|uncertainty"
    r"|exact (paired|two-sided|test|p-value)|mcnemar|\bse\b|elpd_se|d_elpd",
    re.IGNORECASE,
)

# CERTIFICATION claims advertise a finite-sample selective-risk / accepted-error guarantee; the
# file must state the sampling premise (i.i.d./exchangeability), the failure probability over
# the calibration draw, and the shift voiding. The verb-object shape (certify + its risk object)
# keeps ledger-provenance narration ("reloaded an uncertified threshold as certified", "which
# rows certified the CURRENT conformal threshold") out of scope -- those describe bookkeeping,
# not a guarantee offered to the caller.
_CERTIFICATION_CLAIM = re.compile(
    r"certif(y|ies|ied|ication)\s+(a |the |an )?[^.\n]{0,30}(risk|selective|accepted-error)"
    r"|selective[- ]risk[^.\n]{0,50}(gate|threshold|control|<=|at most)"
    r"|\(alpha, delta\)-PAC"
    r"|risk control\b",
    re.IGNORECASE,
)
_CERTIFICATION_TOKENS = (
    re.compile(r"i\.i\.d|\biid\b|exchangeab", re.IGNORECASE),
    re.compile(r"\bdelta\b|probability at least|with probability", re.IGNORECASE),
    re.compile(r"shift|drift|same distribution|deployment distribution", re.IGNORECASE),
)
CERTIFICATION_SURFACES = ("mixle/task", "mixle/reason", "docs", "examples")

# COMPETITIVE claims name a rival library or a closed gap; the file must disclose its
# measurement conditions (versions, this-run language, if-installed hedges).
_COMPETITIVE_CLAIM = re.compile(
    r"nearest rival"
    r"|rival (can.?t|cannot|fails)"
    r"|(pomegranate|tensorflow[- ]probability|\btfp\b|scikit-learn|sklearn|raw torch)[^.\n]{0,60}"
    r"(can.?t|cannot|fails|raises|crashes|slower|worse)"
    r"|gap closes"
    r"|(outperforms?|faster than|beats)\s+(pomegranate|tensorflow|tfp|scikit-learn|sklearn|torch)",
    re.IGNORECASE,
)
_CONDITIONS_TOKENS = re.compile(
    r"\d+\.\d+(\.\d+)?"  # a pinned version number
    r"|if installed|this (run|machine|process)|measured (here|in this|on this)|current-run",
    re.IGNORECASE,
)
COMPETITIVE_SURFACES = ("examples",)


def _files(surfaces: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for surface in surfaces:
        base = ROOT / surface
        if not base.exists():
            continue
        patterns = ("*.py", "*.rst") if surface == "docs" else ("*.py",)
        for pattern in patterns:
            out.extend(
                p
                for p in sorted(base.rglob(pattern))
                if "docs/api" not in p.as_posix()
                # the gate's own contract test QUOTES claim phrases as detector fixtures; it
                # makes no claims of its own and would otherwise flag itself
                and p.name != "statistical_claims_contract_test.py"
            )
    return out


def scan(root: Path | None = None) -> dict:
    """Return {relative_path: {"classes": [...], "violations": [...]}} for every claim site."""
    del root  # single-tree tool; parameter kept for parity with the duplicate-body scanner
    sites: dict[str, dict] = {}

    for path in _files(COVERAGE_SURFACES):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        claim_lines = []
        for i, line in enumerate(lines):
            if not _COVERAGE_CLAIM.search(line):
                continue
            # docstring prose wraps mid-sentence, so the negation may sit at the END of the
            # previous line ("... is a ranking diagnostic, not a\nconformal or bootstrap
            # coverage guarantee"); judge negation over the wrapped sentence fragment
            context = (lines[i - 1][-80:] + " " + line) if i else line
            if not _NEGATED_COVERAGE.search(context):
                claim_lines.append(line)
        if not claim_lines:
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

    for path in _files(COMPETITIVE_SURFACES):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not _COMPETITIVE_CLAIM.search(text):
            continue
        rel = path.relative_to(ROOT).as_posix()
        entry = sites.setdefault(rel, {"classes": [], "violations": []})
        entry["classes"].append("competitive")
        if not _CONDITIONS_TOKENS.search(text):
            entry["violations"].append(
                "competitive claim without measurement-conditions disclosure (versions/this-run/if-installed)"
            )

    for path in _files(CERTIFICATION_SURFACES):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not _CERTIFICATION_CLAIM.search(text):
            continue
        rel = path.relative_to(ROOT).as_posix()
        missing = [token.pattern for token in _CERTIFICATION_TOKENS if not token.search(text)]
        entry = sites.setdefault(rel, {"classes": [], "violations": []})
        entry["classes"].append("certification")
        if missing:
            entry["violations"].append(f"certification claim without scope tokens: missing {missing}")

    return dict(sorted(sites.items()))


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the reviewed manifest")
    parser.add_argument(
        "--reviewed-by",
        default="",
        help="attestation name recorded in the manifest (required with --write; STAT-RR17-20)",
    )
    args = parser.parse_args(argv)

    sites = scan()
    violations = {rel: e["violations"] for rel, e in sites.items() if e["violations"]}
    for rel, entry in sites.items():
        marker = " !! " if entry["violations"] else "    "
        print(f"{marker}{rel}: {sorted(set(entry['classes']))}")
        for violation in entry["violations"]:
            print(f"        {violation}")

    if args.write:
        import datetime

        reviewer = args.reviewed_by
        if not reviewer:
            # STAT-RR17-20: the docstring promised a human audit for new sites, but --write
            # recorded only paths and classes -- no reviewer, date, or attestation. The
            # regeneration now refuses to mint a "reviewed inventory" nobody signed.
            print("--write requires --reviewed-by '<name>': the manifest is a REVIEWED inventory")
            return 1
        record = {
            "_comment": (
                "Reviewed inventory of user-facing statistical-claim sites. A new entry means a new "
                "claim surface: audit its scope, then regenerate with "
                "python scripts/scan_statistical_claims.py --write --reviewed-by '<name>'. The "
                "contract test refuses un-inventoried claim sites and inventoried sites whose "
                "scope tokens disappear."
            ),
            "attestation": {
                "reviewed_by": reviewer,
                "reviewed_on": datetime.date.today().isoformat(),
                "statement": "each inventoried site's scope language was read against its claim class",
            },
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
