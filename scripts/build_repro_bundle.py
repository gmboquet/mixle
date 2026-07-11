"""Worklist E14.5 -- assemble a reproducibility bundle for the flagship/benchmark reports.

A reproducibility bundle is the set of *commands + configs + results* needed to re-derive
the published reports, with checksums so a reproducer knows they ran the exact artifact,
and licenses so redistribution is clear. E14.5's acceptance is that an external
reproduction (E14 / E4) succeeds from the bundle.

This assembler records, for each reproducible report that ships in this repo, the run
command, the script path, its SHA-256, whether it needs network, and its data license
status. It writes ``release-checklists/0.8.0-repro-bundle.json``. The drift-gate test
(``repro_bundle_test.py``) then verifies every referenced script exists and its checksum
still matches, so the bundle cannot silently point at a moved or altered script.

Usage:
* ``python scripts/build_repro_bundle.py``          -- print the current bundle;
* ``python scripts/build_repro_bundle.py --write``  -- (re)write the bundle JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "release-checklists" / "0.8.0-repro-bundle.json"

# Curated reproducible reports that ship in this repo. Each entry names the exact command
# and the script whose checksum pins the artifact. `data_license` is honest: named where
# known, flagged for confirmation where the dataset is fetched at runtime.
ENTRIES = [
    {
        "id": "flagship-banking77-cascade",
        "kind": "flagship",
        "command": "python examples/real_receipt_banking77.py",
        "script": "examples/real_receipt_banking77.py",
        "needs_network": True,
        "dataset": "Banking77 (PolyAI)",
        "data_license": "CONFIRM-AT-PUBLISH: verify Banking77 redistribution terms before bundling data",
    },
    {
        "id": "gallery-univariate",
        "kind": "self-contained",
        "command": "python examples/gallery_univariate_example.py",
        "script": "examples/gallery_univariate_example.py",
        "needs_network": False,
        "dataset": "none (samples from a known model)",
        "data_license": "n/a (synthetic)",
    },
    {
        "id": "gallery-structured",
        "kind": "self-contained",
        "command": "python examples/gallery_structured_example.py",
        "script": "examples/gallery_structured_example.py",
        "needs_network": False,
        "dataset": "none (samples from a known model)",
        "data_license": "n/a (synthetic)",
    },
    {
        "id": "production-provenance",
        "kind": "workflow",
        "command": "python examples/production_example.py",
        "script": "examples/production_example.py",
        "needs_network": False,
        "dataset": "none (synthetic)",
        "data_license": "n/a (synthetic)",
    },
    {
        "id": "scaling-backend",
        "kind": "backend",
        "command": "python examples/scaling_example.py",
        "script": "examples/scaling_example.py",
        "needs_network": False,
        "dataset": "none (synthetic)",
        "data_license": "n/a (synthetic)",
    },
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    entries = []
    for spec in ENTRIES:
        script = ROOT / spec["script"]
        entry = dict(spec)
        entry["sha256"] = _sha256(script) if script.is_file() else None
        entries.append(entry)
    return {
        "_comment": (
            "Worklist E14.5 reproducibility bundle. Each entry re-derives a shipped report. "
            "Run the command from a clean clone; the sha256 pins the exact script. Regenerate "
            "with `python scripts/build_repro_bundle.py --write`."
        ),
        "code_license": "MIT (see LICENSE, NOTICE)",
        "acceptance": "E14/E4 external reproduction succeeds from this bundle.",
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the bundle JSON")
    args = parser.parse_args()

    bundle = build()
    if args.write:
        BUNDLE.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {BUNDLE} with {len(bundle['entries'])} entries")
        return 0

    print(json.dumps(bundle, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
