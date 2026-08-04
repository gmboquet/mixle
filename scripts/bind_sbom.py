#!/usr/bin/env python
"""Bind a CycloneDX inventory to the exact wheel and source candidate it describes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def bind(
    raw: dict[str, Any],
    wheel: dict[str, Any],
    candidate_sha: str,
    *,
    profile: str = "base",
    waivers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if raw.get("bomFormat") != "CycloneDX" or not isinstance(raw.get("components"), list):
        raise ValueError("input is not a CycloneDX component inventory")
    names = {component.get("name") for component in raw["components"] if isinstance(component, dict)}
    # `pip-audit` builds its CycloneDX inventory from packages it can RESOLVE ON PYPI, and it says so
    # when it cannot: "Dependency not found on PyPI and could not be audited: mixle (0.8.0)". For an
    # unpublished candidate that is structural, not a defect -- and requiring mixle's presence made
    # the SBOM gate unsatisfiable for the first release of any package, which is precisely when an
    # SBOM matters most.
    #
    # The property this check protects is that the bound SBOM describes the wheel actually under
    # audit. That is satisfied without the inventory entry, because the wheel's filename and SHA-256
    # come from --wheel-metadata, which is authoritative and validated just below. So when the entry
    # is absent it is SYNTHESIZED from that metadata rather than waved through, and the bound
    # document records which of the two happened -- an inventory entry pip-audit vouched for is not
    # the same evidence as one derived from the artifact, and a reader must be able to tell.
    mixle_audited = "mixle" in names
    if not _GIT_SHA.fullmatch(candidate_sha):
        raise ValueError("candidate SHA must be a full lowercase Git SHA")
    if not isinstance(wheel.get("filename"), str) or not wheel["filename"].endswith(".whl"):
        raise ValueError("artifact metadata does not describe a wheel")
    if not isinstance(wheel.get("sha256"), str) or not _SHA256.fullmatch(wheel["sha256"]):
        raise ValueError("artifact metadata has no valid SHA-256")
    if profile not in {"base", "all"}:
        raise ValueError("SBOM profile must be base or all")
    if waivers is None:
        waivers = {
            "artifact": "mixle.accepted_vulnerability_waivers/v1",
            "profile": profile,
            "waivers": [],
        }
    if (
        waivers.get("artifact") != "mixle.accepted_vulnerability_waivers/v1"
        or waivers.get("profile") != profile
        or not isinstance(waivers.get("waivers"), list)
    ):
        raise ValueError("accepted vulnerability waivers do not match the SBOM profile")
    inventory = dict(raw)
    if not mixle_audited:
        inventory["components"] = [
            *raw["components"],
            {
                "type": "library",
                "name": "mixle",
                "version": wheel.get("version"),
                "description": (
                    "Synthesized from the audited wheel's own metadata. pip-audit omits a "
                    "distribution it cannot resolve on PyPI, which is every unpublished candidate; "
                    "this entry therefore carries NO vulnerability audit."
                ),
                "hashes": [{"alg": "SHA-256", "content": wheel["sha256"]}],
                "properties": [
                    {"name": "mixle:source", "value": "wheel-metadata"},
                    {"name": "mixle:audited", "value": "false"},
                    {"name": "mixle:filename", "value": wheel["filename"]},
                ],
            },
        ]
    return {
        "artifact": "mixle.bound_sbom/v1",
        "candidate_commit": candidate_sha,
        "wheel": {
            "filename": wheel["filename"],
            "sha256": wheel["sha256"],
            "size_bytes": wheel.get("size_bytes"),
        },
        # Whether the mixle component came from pip-audit's inventory or was synthesized from the
        # artifact. False means the candidate was not resolvable on PyPI at audit time -- expected
        # before first publication, and a signal worth reading afterwards.
        "mixle_component_audited": mixle_audited,
        "inventory_scope": f"isolated-wheel-environment:{profile}",
        "profile": profile,
        "accepted_vulnerability_waivers": waivers,
        "cyclonedx": inventory,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cyclonedx", type=Path, required=True)
    parser.add_argument("--wheel-metadata", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--profile", choices=("base", "all"), default="base")
    parser.add_argument("--waivers", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = bind(
            json.loads(args.cyclonedx.read_text(encoding="utf-8")),
            json.loads(args.wheel_metadata.read_text(encoding="utf-8")),
            args.candidate_sha,
            profile=args.profile,
            waivers=json.loads(args.waivers.read_text(encoding="utf-8")) if args.waivers else None,
        )
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
