#!/usr/bin/env python3
"""Import every direct dependency module in an installed optional profile."""

from __future__ import annotations

import argparse
import importlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def applicable_modules(profile: str) -> list[str]:
    """Modules a profile's requirements contribute *in this environment*.

    Needs ``packaging`` to evaluate environment markers, so it is separated from :func:`verify`,
    which needs only the standard library and must run inside the environment under test. The
    workflow's floor environment is built from an exact constraint set and its full ``pip freeze``
    is recorded as release evidence, so installing a tooling dependency into it would put
    ``packaging`` in the recorded floor graph of a profile that does not declare it.

    The import is FUNCTION-LOCAL for exactly that reason. At module scope it defeated the split
    entirely: ``--modules``, the standard-library-only path, still failed on
    ``ModuleNotFoundError: No module named 'packaging'`` before reaching ``main``. That went
    unnoticed locally because pytest depends on ``packaging``, so every environment used for testing
    happened to have it; the workflow's floor environment installs neither.
    """
    from packaging.requirements import Requirement

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    if profile not in extras:
        raise ValueError(f"unknown optional profile {profile!r}")
    mapping = json.loads((ROOT / "manifests" / "optional_dependency_imports.json").read_text(encoding="utf-8"))[
        "distribution_to_imports"
    ]
    modules = []
    for requirement in extras[profile]:
        parsed = Requirement(requirement)
        if parsed.name not in mapping:
            raise ValueError(f"no import mapping for {parsed.name}")
        # Honour the environment marker. The requirement was previously matched with a bare name
        # regex, which DISCARDS the marker, so this tried to import a distribution pip had correctly
        # declined to install: `tbb; platform_machine == "x86_64"` failed the whole numba profile on
        # every non-x86_64 machine. A requirement that does not apply here is not a verification
        # failure -- there is nothing installed to verify.
        if parsed.marker is not None and not parsed.marker.evaluate():
            continue
        modules.extend(mapping[parsed.name])
    return sorted(set(modules))


def verify(modules: list[str]) -> list[str]:
    """Import each module, in the interpreter being verified. Standard library only, by design."""
    for module in modules:
        importlib.import_module(module)
    return sorted(set(modules))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument(
        "--print-modules",
        action="store_true",
        help="resolve the profile's applicable modules and print them, without importing",
    )
    parser.add_argument(
        "--modules",
        help="comma-separated modules to import, as produced by --print-modules in a matching environment",
    )
    args = parser.parse_args()
    try:
        if args.print_modules:
            if not args.profile:
                raise ValueError("--print-modules requires --profile")
            print(",".join(applicable_modules(args.profile)))
            return 0
        if args.modules is not None:
            names = [name for name in args.modules.split(",") if name]
            imported = verify(names)
            profile = args.profile or "(supplied)"
        else:
            if not args.profile:
                raise ValueError("supply --profile or --modules")
            profile = args.profile
            imported = verify(applicable_modules(profile))
    except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
        print(str(exc))
        return 1
    print(json.dumps({"artifact": "mixle.extra_profile_imports/v1", "profile": profile, "imports": imported}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
