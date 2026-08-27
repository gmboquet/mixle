"""Build the content-addressed 0.8.0 reproduction-bundle specification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "release-checklists" / "0.8.0-repro-bundle.json"

# The check-evidence record that binds a candidate to APPROVED checks is produced by exactly one
# generator (scripts/verify_required_checks.py, run by publish.yml) over exactly one policy (the
# check-run names below). The bundle embeds that policy so the receipt resolver can require every
# name against the generator's own schema, and closes over both files so a change to either
# without regenerating the bundle fails the canonical-bundle test rather than drifting silently.
REQUIRED_CHECKS_POLICY = ".github/release-required-checks.txt"
CHECK_EVIDENCE_GENERATOR = "scripts/verify_required_checks.py"
CANDIDATE_RECORD_PRODUCER = "scripts/release_candidate_record.py"

# The check-evidence record is APPROVAL evidence only if it was produced by this repository's own
# workflow over the candidate commit's real check runs. A record with the right shape, all 24 names,
# distinct integer ids and plausible URLs is not that (SYS5-01: such a record, authored by hand with
# invented run ids, yielded four verified receipts and a complete manifest). So the record is only
# ever written by one of the workflows below, which attest it through GitHub's OIDC identity
# (actions/attest -> Sigstore); the receipt resolver verifies that attestation with gh against
# Sigstore's public-good root, bound to this repository, the signing workflow, and the candidate
# commit (`--source-digest`), and re-derives the record's selection from the retained check-runs
# payload whose digest the record commits to.
REPOSITORY = "gmboquet/mixle"
CHECK_EVIDENCE_ATTESTATION = {
    "predicate_type": "https://github.com/gmboquet/mixle/release-check-evidence/v1",
    # publish.yml signs the release's record; tests.yml's final dispatch-only job signs a review
    # candidate's (a workflow that only exists on a release branch cannot be dispatched at all --
    # workflow_dispatch needs the file on the default branch -- and tests.yml is there)
    "signer_workflows": [".github/workflows/publish.yml", ".github/workflows/tests.yml"],
    "bundle_record": "metadata/release-check-evidence.sigstore.json",
    "check_runs_record": "metadata/check-runs.json",
    # no retained trusted root: gh's own TUF-fetched Sigstore public-good root is the anchor; a
    # root shipped beside the record would be chosen by whoever ships the record
    "verifier": "gh attestation verify --bundle --repo --cert-identity-regex --cert-oidc-issuer --source-digest --predicate-type --deny-self-hosted-runners",
}

_CLOSURE_PATHS = (
    "pyproject.toml",
    "release-checklists/0.8.0-repro-environment.json",
    "release-checklists/0.8.0-repro-requirements.txt",
    "scripts/build_repro_bundle.py",
    "scripts/run_repro_entry.py",
    REQUIRED_CHECKS_POLICY,
    CHECK_EVIDENCE_GENERATOR,
    CANDIDATE_RECORD_PRODUCER,
)


def _required_check_names() -> list[str]:
    """Parse the publication policy with the generator's own parser, so the two cannot disagree."""
    spec = importlib.util.spec_from_file_location("_verify_required_checks", ROOT / CHECK_EVIDENCE_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CHECK_EVIDENCE_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.required_check_names(ROOT / REQUIRED_CHECKS_POLICY))


_ENTRIES = (
    # No hosted-network entry: the repository carries no direct dataset usage (release owner's
    # decision, 2026-08-04) -- real-data demonstrations live in notebooks outside this repo, so every
    # bundle entry is self-contained and replays offline.
    {
        "id": "gallery-univariate",
        "kind": "self-contained",
        "tier": "local",
        "argv": ["examples/gallery_univariate_example.py"],
        "script": "examples/gallery_univariate_example.py",
        "timeout_seconds": 30,
        "configuration": {"seed": "declared in script", "dataset": "synthetic"},
        "expected": {
            "format": "text",
            # Repinned for the campaign-four shift-sweep wave: StudentTEstimator and
            # LogGaussianEstimator gained the shift-anchored moment track (this example fits both
            # directly), the same repair the Gaussian and Logistic families already carried. Two
            # printed values move in their last two digits -- the Student-t scale
            # 1.9519185926635496 -> 1.9519185926635638 and the log-Gaussian variance
            # 0.2512756175947848 -> 0.25127561759478356, each ~6e-15 relative, because ``estimate()``
            # drives the per-observation ``update`` path, where the anchor activates on the first
            # observation by design (a scalar update carries no chunk to gate on). New digest
            # measured three times, byte-identical.
            "stdout_sha256": "3e54938bf429fccaed6673da4187cb1cde04b695f6dee0e88824288620cb93ef",
            "contains": [
                "fit : GaussianDistribution(1.5485975768849254, 4.020409881516532",
                "fit : PoissonDistribution(4.027",
                "fit : BernoulliDistribution(0.7004",
            ],
        },
    },
    {
        "id": "gallery-structured",
        "kind": "self-contained",
        "tier": "local",
        "argv": ["examples/gallery_structured_example.py"],
        "script": "examples/gallery_structured_example.py",
        "timeout_seconds": 30,
        "configuration": {"seed": "declared in script", "dataset": "synthetic"},
        "expected": {
            "format": "text",
            "stdout_sha256": "4abfe0e6d459a39dfbb457e3495acaec2edf69b4adaaf57ce4dd3cf840a5f98e",
            "contains": [
                "learned parents: [None, 0, 0]",
                "held-out mean log-density: -17.965",
                "held-out mean log-density: -19.791",
            ],
        },
    },
    {
        "id": "production-provenance",
        "kind": "workflow",
        "tier": "local",
        "argv": ["examples/production_example.py"],
        "script": "examples/production_example.py",
        "timeout_seconds": 30,
        "configuration": {"seed": "declared in script", "dataset": "synthetic"},
        "expected": {
            "format": "text",
            # The provenance line names the commit being reproduced from, so it cannot be part of a
            # fixed digest: the recorded one was valid at 6fbb182a and wrong at every commit after.
            "volatile": [
                {
                    "pattern": r"git / mixle  : (?:[0-9a-f]{7,40}|unknown) / ",
                    "placeholder": "git / mixle  : <commit> / ",
                }
            ],
            # Repinned for the c9eb9d2e campaign fix waves: the EM default-initialization repair
            # and FitProvenance.final_objective now describing the RETURNED model change the
            # example's fitted values and its "final loglik" line. New digest measured twice,
            # byte-identical, with every `contains` invariant still holding.
            "stdout_sha256": "d1c0f7da932c277dccf7d24e1c5b02ca976dcc08f73200ecf46f0b11ba0612e1",
            "contains": [
                "# lineage verified: True",
                "drift on shifted batch: True",
                "chain intact: True",
            ],
        },
    },
    {
        "id": "scaling-backend",
        "kind": "backend",
        "tier": "local",
        "argv": ["examples/scaling_example.py"],
        "script": "examples/scaling_example.py",
        "timeout_seconds": 30,
        "configuration": {"workers": 4, "dataset": "synthetic"},
        "expected": {
            "format": "text",
            "stdout_sha256": "5fd8dd73c1c9062439a5a47b0c4266433c1992718b0c6c0a961330319645fe1a",
            "contains": [
                "backend='local'   : mu=2.00  P(a)=0.60  lam=4.01",
                "backend='mp' (x4) : mu=2.00  P(a)=0.60  lam=4.01",
            ],
        },
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(path: str, role: str) -> dict:
    target = ROOT / path
    if not target.is_file():
        raise FileNotFoundError(f"reproduction input does not exist: {path}")
    return {"path": path, "role": role, "sha256": _sha256(target)}


def build() -> dict:
    """Build a deterministic closure over commands, code, data, environment, and expectations."""
    entries = []
    for specification in _ENTRIES:
        entry = dict(specification)
        entry["inputs"] = [_input(entry["script"], "executable")]
        if dataset_record := entry.get("dataset_record"):
            entry["inputs"].append(_input(dataset_record, "dataset-license-and-integrity"))
        entries.append(entry)
    return {
        "artifact": "mixle.reproduction_bundle/v2",
        "release": "0.8.0",
        "candidate_binding": {
            "policy": "exact-publish-workflow-candidate",
            "repository": REPOSITORY,
            "required_records": [
                "metadata/release-candidate.json",
                "metadata/release-check-evidence.json",
                CHECK_EVIDENCE_ATTESTATION["bundle_record"],
                CHECK_EVIDENCE_ATTESTATION["check_runs_record"],
                "metadata/SHA256SUMS",
                "metadata/mixle-0.8.0-py3-none-any.whl.json",
                "metadata/reproduction-*.json",
            ],
            "required_checks": _required_check_names(),
            "required_checks_policy": REQUIRED_CHECKS_POLICY,
            "check_evidence_generator": CHECK_EVIDENCE_GENERATOR,
            "check_evidence_attestation": dict(CHECK_EVIDENCE_ATTESTATION),
            "candidate_record_producer": CANDIDATE_RECORD_PRODUCER,
            "rule": (
                "The final bundle is incomplete unless these retained records bind its source commit, "
                "approved checks, wheel SHA-256, and local entry receipts to the signed v0.8.0 tag."
            ),
        },
        "environment": "release-checklists/0.8.0-repro-environment.json",
        "closure": [_input(path, "bundle-closure") for path in _CLOSURE_PATHS],
        "code_license": {"spdx": "MIT", "files": ["LICENSE", "NOTICE"]},
        "acceptance": (
            "Every entry is local and self-contained: each passes exact output validation offline, "
            "and retained candidate/check/wheel records bind the bundle to publication."
        ),
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the canonical bundle JSON")
    args = parser.parse_args(argv)
    bundle = build()
    text = json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.write:
        BUNDLE.write_text(text, encoding="utf-8")
        print(f"wrote {BUNDLE} with {len(bundle['entries'])} entries")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
