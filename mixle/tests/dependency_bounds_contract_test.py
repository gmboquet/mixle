"""Runtime dependency drift must stop at reviewed upper bounds."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _checker():
    path = ROOT / "scripts" / "check_dependency_bounds.py"
    spec = importlib.util.spec_from_file_location("_check_dependency_bounds", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_bounds_match_machine_drift_decisions() -> None:
    assert _checker().validate() == []


def test_resolved_latest_and_exact_floor_graphs_are_both_required() -> None:
    workflow = (ROOT / ".github" / "workflows" / "extras-matrix.yml").read_text(encoding="utf-8")
    assert "extra-profile-set-receipt.json" in workflow
    assert "extra-floor-profile-set-receipt.json" in workflow
