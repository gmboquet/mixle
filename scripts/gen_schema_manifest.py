"""Generate one attested profile in the versioned serialization-schema manifest.

Usage::

    python scripts/gen_schema_manifest.py --profile base
    python scripts/gen_schema_manifest.py --profile full
    python scripts/gen_schema_manifest.py --profile base --check

Generation never replaces profiles other than the one explicitly selected, and it refuses to build a
profile unless that profile's complete dependency inventory is importable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mixle.utils.serialization import OBJECT_SCHEMA_VERSION, TAG, serializable_schema_records

MANIFEST_PATH = ROOT / "manifests" / "serialization_schema_manifest.json"
PROFILE_REQUIREMENTS = {
    "base": ("numpy", "scipy"),
    "full": ("numpy", "scipy", "torch"),
}


def _require_profile_inventory(profile: str) -> tuple[str, ...]:
    required = PROFILE_REQUIREMENTS[profile]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"cannot generate serialization profile {profile!r}; missing imports: {missing}")
    return required


def build_profile(profile: str) -> dict[str, Any]:
    """Return one exact, dependency-attested serialization profile."""
    required = _require_profile_inventory(profile)
    schemas = serializable_schema_records(profile)
    return {
        "required_imports": list(required),
        "registered_types": sorted(schemas),
        "schemas": schemas,
    }


def _empty_manifest() -> dict[str, Any]:
    return {
        "artifact": "mixle.serialization_schema_manifest/v2",
        "schema_manifest_version": "2",
        "tag": TAG,
        "object_envelope_version": OBJECT_SCHEMA_VERSION,
        "profiles": {},
    }


def load_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_manifest()
    if manifest.get("artifact") != "mixle.serialization_schema_manifest/v2":
        return _empty_manifest()
    return manifest


def render(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_REQUIREMENTS))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        expected = build_profile(args.profile)
        manifest = load_manifest()
        if args.check:
            if manifest.get("profiles", {}).get(args.profile) != expected:
                print(
                    f"serialization schema profile {args.profile!r} is stale; regenerate that profile",
                    file=sys.stderr,
                )
                return 1
            return 0
        manifest["profiles"][args.profile] = expected
        MANIFEST_PATH.write_text(render(manifest), encoding="utf-8")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {args.profile} profile ({len(expected['registered_types'])} registered types)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
