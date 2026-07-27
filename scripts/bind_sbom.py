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
    if "mixle" not in names:
        raise ValueError("CycloneDX inventory does not contain the installed Mixle wheel")
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
    return {
        "artifact": "mixle.bound_sbom/v1",
        "candidate_commit": candidate_sha,
        "wheel": {
            "filename": wheel["filename"],
            "sha256": wheel["sha256"],
            "size_bytes": wheel.get("size_bytes"),
        },
        "inventory_scope": f"isolated-wheel-environment:{profile}",
        "profile": profile,
        "accepted_vulnerability_waivers": waivers,
        "cyclonedx": raw,
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
