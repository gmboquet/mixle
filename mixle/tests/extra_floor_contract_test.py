"""Every optional profile must have exact lower-bound and feature-import evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_every_release_profile_generates_exact_floors() -> None:
    generator = _load("generate_extra_floor_constraints")
    profiles = json.loads((ROOT / "release-checklists" / "0.8.0-extra-profiles.json").read_text(encoding="utf-8"))[
        "profiles"
    ]
    for profile in profiles:
        lines = generator.constraints(profile)
        assert lines
        assert all("==" in line and ">=" not in line for line in lines)
        assert any(line.lower().startswith("numpy==") for line in lines)
        assert any(line.lower().startswith("scipy==") for line in lines)


def test_hosted_matrix_installs_and_records_floor_profiles() -> None:
    workflow = (ROOT / ".github" / "workflows" / "extras-matrix.yml").read_text(encoding="utf-8")
    assert "generate_extra_floor_constraints.py" in workflow
    # Asserted as a CONTRACT rather than as one literal command line. This previously pinned the
    # exact string `scripts/verify_extra_profile.py --profile "$PROFILE"`, which broke the moment the
    # invocation was legitimately reformatted: the script's marker resolution needs `packaging` and
    # its import check must not, so the two now run under different interpreters and the call spans
    # two lines. The property that matters is that the floor environment is verified against the
    # profile -- not how the command happens to be wrapped.
    assert "verify_extra_profile.py" in workflow
    assert '--profile "$PROFILE"' in workflow
    assert "--print-modules" in workflow  # markers resolved by the tooling interpreter
    assert '--modules "$FLOOR_MODULES"' in workflow  # imports checked inside the floor environment
    assert "$RUNNER_TEMP/floor-env/bin/python" in workflow
    assert "extra-floor-receipt-" in workflow
    assert "extra-floor-profile-set-receipt.json" in workflow
