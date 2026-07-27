"""Build the content-addressed 0.8.0 reproduction-bundle specification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "release-checklists" / "0.8.0-repro-bundle.json"

_CLOSURE_PATHS = (
    "pyproject.toml",
    "release-checklists/0.8.0-repro-environment.json",
    "release-checklists/0.8.0-repro-requirements.txt",
    "release-checklists/0.8.0-banking77-dataset.json",
    "scripts/build_repro_bundle.py",
    "scripts/run_repro_entry.py",
)

_ENTRIES = (
    {
        "id": "flagship-banking77-cascade",
        "kind": "flagship",
        "tier": "hosted-network",
        "argv": ["examples/real_receipt_banking77.py", "--smoke", "--json"],
        "script": "examples/real_receipt_banking77.py",
        "timeout_seconds": 30,
        "configuration": {
            "n_seed": 1155,
            "n_round": 40,
            "n_rounds": 1,
            "n_test": 60,
            "student": "generative",
            "seed": 0,
        },
        "dataset_record": "release-checklists/0.8.0-banking77-dataset.json",
        "expected": {
            "format": "json",
            "assertions": [
                {"path": "artifact", "equals": "mixle.banking77_reproduction/v1"},
                {"path": "metrics.task", "equals": "banking77 intents (77 classes)"},
                {"path": "metrics.n_test", "equals": 60},
                {"path": "metrics.end_to_end_accuracy", "minimum": 0.0, "maximum": 1.0},
                {"path": "metrics.local_agreement", "minimum": 0.0, "maximum": 1.0},
                {"path": "metrics.escalation_rate", "minimum": 0.0, "maximum": 1.0},
                {
                    "path": "dataset.source_commit",
                    "equals": "9d081458ff52e53cf7e848f414e6e9344e4e6696",
                },
                {"path": "dataset.splits.train.rows", "equals": 10003},
                {"path": "dataset.splits.test.rows", "equals": 3080},
            ],
        },
    },
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
            "stdout_sha256": "9e7667d4fc942dcbf00f52eea48a0da2b65f1964d216dab35828aec331ca5e7f",
            "contains": [
                "fit : GaussianDistribution(1.5485975768849254, 4.0204098815165565",
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
            "stdout_sha256": "80b8fc6822e95f2d363555037da288e70e552395f86b9e60fd8267c364b5b75f",
            "contains": [
                "learned parents: [None, 0, 0]",
                "held-out mean log-density: -17.965",
                "held-out mean log-density: -19.841",
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
            "stdout_sha256": "c9e3874f3f4dac45ee6f88127c56033821658965a9d4db2ab6055d4d2539b669",
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
            "required_records": [
                "metadata/release-candidate.json",
                "metadata/release-check-evidence.json",
                "metadata/SHA256SUMS",
                "metadata/mixle-0.8.0-py3-none-any.whl.json",
                "metadata/reproduction-*.json",
                "metadata/network/banking77-reproduction-receipt.json",
            ],
            "rule": (
                "The final bundle is incomplete unless these retained records bind its source commit, "
                "approved checks, wheel SHA-256, and local/hosted entry receipts to the signed v0.8.0 tag."
            ),
        },
        "environment": "release-checklists/0.8.0-repro-environment.json",
        "closure": [_input(path, "bundle-closure") for path in _CLOSURE_PATHS],
        "code_license": {"spdx": "MIT", "files": ["LICENSE", "NOTICE"]},
        "acceptance": (
            "Every local entry passes exact output validation; the hosted-network entry passes on the "
            "same candidate; and retained candidate/check/wheel records bind the bundle to publication."
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
