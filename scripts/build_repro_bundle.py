"""Build the content-addressed 0.8.2 reproduction-bundle specification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "release-checklists" / "0.8.2-repro-bundle.json"

# The check-evidence record that binds a candidate to APPROVED checks is produced by exactly one
# generator (scripts/verify_required_checks.py, run by publish.yml) over exactly one policy (the
# check-run names below). The bundle embeds that policy so the receipt resolver can require every
# name against the generator's own schema, and closes over both files so a change to either
# without regenerating the bundle fails the canonical-bundle test rather than drifting silently.
REQUIRED_CHECKS_POLICY = ".github/release-required-checks.txt"
CHECK_EVIDENCE_GENERATOR = "scripts/verify_required_checks.py"
CANDIDATE_RECORD_PRODUCER = "scripts/release_candidate_record.py"

# The check-evidence record is APPROVAL evidence only if it was produced by this repository's own
# workflow over the candidate commit's real check runs. A record with the right shape, every required name,
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
    "release-checklists/0.8.2-repro-environment.json",
    "release-checklists/0.8.2-repro-requirements.txt",
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
            # Repinned for platform-robust output: the example now rounds every printed float to 6
            # significant figures (examples/gallery_univariate_example.py:stable). A one-pass fit of
            # the same seeded sample lands on the same value to ~13 significant figures on every
            # machine, but the last few digits differ by ULPs because BLAS sums in a different order
            # per CPU; full-precision printing exposed that and the raw-float stdout hash reproduced
            # on no platform but the one it was pinned on (macOS arm64, Linux arm64, and the old
            # pinned value were three distinct hashes). Rounded, the output is identical across
            # macOS arm64 and Linux arm64; new digest measured on both.
            "stdout_sha256": "238593732ace807b0735348a5a943e5f41d0c5a49dd8457eddcee4457f1e86b5",  # 0.8.1: the binomial row fixes n (max_val=10)
            "contains": [
                "fit : GaussianDistribution(1.5486, 4.02041",
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
            # Repinned for platform-robust output (same treatment as gallery-univariate): the Markov
            # transition map, the heterogeneous-mixture component reprs, and the record fits printed
            # raw floats that differ by ULPs per CPU/BLAS; the example now rounds them to 6 significant
            # figures. The structural/rounded `contains` lines already matched on x86_64 CI.
            # 0.8.2: repinned because the OUTPUT legitimately changed. P09-F11 raised the
            # SegmentalHiddenMarkov section's iteration budget (it was capped at max_its=10 and
            # printed a held-out -19.79 where -16.36 is reachable, against a truth of -16.35), and
            # both sections now print the true model's value beside the fitted one. Measured on the
            # 0.8.1 tree and on this one: -17.965 -> -15.180 (true -15.071) and -19.791 -> -16.359
            # (true -16.347). The old numbers are the under-budgeted fit the repair removed, so
            # keeping them would pin the defect.
            "stdout_sha256": "691caecdd76bbd23f45043a6b108539b3ab3ed0f55bc3a862807133950178748",
            "contains": [
                "learned parents: [None, 0, 0]",
                "held-out mean log-density: -15.180 (true model -15.071)",
                "held-out mean log-density: -16.359 (true model -16.347)",
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
                },
                # The model fingerprint hashes the fitted parameters' raw float64 bytes
                # (mixle.data.hashing._canonical), and a fit differs by ULPs per CPU/BLAS, so the hex
                # cannot be byte-reproduced across arithmetic; "# lineage verified: True" (pinned in
                # `contains`) is what proves the provenance chain, so the value is normalized away.
                {
                    "pattern": r"model hash  : [0-9a-f]{16} \.\.\.",
                    "placeholder": "model hash  : <fit-dependent> ...",
                },
            ],
            # Repinned for platform-robust output. Two things in the example were machine-dependent:
            # the seeded sample itself (RandomState.normal goes through libm's log, which differs by
            # an ULP between macOS and glibc, so the DATA hash differed per OS) -- the example now
            # rounds every draw to 6 decimals, which pins the data hash everywhere -- and the number
            # of checkpoints (well-separated mixture components reached their EM fixed point in ~3
            # iterations, after which float noise decided whether a later step was "rejected") --
            # the checkpoint demo now fits overlapping components that are still climbing at
            # iteration 9. Digest measured identical on macOS arm64 and emulated x86_64 Linux with
            # the pinned numpy/scipy; the model hash remains the one declared-volatile span.
            # 0.8.2: repinned because the recorded digest was never right for this entry's own
            # volatile rules. The example's output normalizes to the same bytes on the 0.8.1 tree
            # (release/0.8.1) and on this one -- d4f80090... both times, verified by applying the
            # two rules below by hand -- and neither is aa6e4d98..., so the old value could not have
            # been produced by a run through this normalization. Nothing in 0.8.2 changed this
            # example's behaviour; the pin was stale before 0.8.2 opened.
            "stdout_sha256": "d4f800900718c2d3729788103a63ec71b891b6cf40e32c33dd6f0df56e73565c",
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
        "release": "0.8.2",
        "candidate_binding": {
            "policy": "exact-publish-workflow-candidate",
            "repository": REPOSITORY,
            "required_records": [
                "metadata/release-candidate.json",
                "metadata/release-check-evidence.json",
                CHECK_EVIDENCE_ATTESTATION["bundle_record"],
                CHECK_EVIDENCE_ATTESTATION["check_runs_record"],
                "metadata/SHA256SUMS",
                "metadata/mixle-0.8.2-py3-none-any.whl.json",
                "metadata/reproduction-*.json",
            ],
            "required_checks": _required_check_names(),
            "required_checks_policy": REQUIRED_CHECKS_POLICY,
            "check_evidence_generator": CHECK_EVIDENCE_GENERATOR,
            "check_evidence_attestation": dict(CHECK_EVIDENCE_ATTESTATION),
            "candidate_record_producer": CANDIDATE_RECORD_PRODUCER,
            "rule": (
                "The final bundle is incomplete unless these retained records bind its source commit, "
                "approved checks, wheel SHA-256, and local entry receipts to the signed v0.8.2 tag."
            ),
        },
        "environment": "release-checklists/0.8.2-repro-environment.json",
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
