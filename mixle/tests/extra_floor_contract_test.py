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
    assert 'scripts/verify_extra_profile.py --profile "$PROFILE"' in workflow
    assert "extra-floor-receipt-" in workflow
    assert "extra-floor-profile-set-receipt.json" in workflow
