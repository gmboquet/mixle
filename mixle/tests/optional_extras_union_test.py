"""The advertised aggregate and example dependency profiles must be complete."""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _checker():
    path = ROOT / "scripts" / "check_optional_extras.py"
    spec = importlib.util.spec_from_file_location("_check_optional_extras", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_is_exact_runtime_feature_union() -> None:
    assert _checker().validate() == []


def test_external_example_dependencies_are_declared_and_hosted() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    examples = " ".join(project["project"]["optional-dependencies"]["examples"]).lower()
    for package in ("peft", "scikit-learn", "pomegranate", "hmmlearn"):
        assert package in examples
    workflow = (ROOT / ".github" / "workflows" / "extras-matrix.yml").read_text(encoding="utf-8")
    assert "          - examples\n" in workflow
    profiles = json.loads(
        (ROOT / "release-checklists" / "0.8.0-extra-profiles.json").read_text(encoding="utf-8")
    )["profiles"]
    assert set(profiles) >= {"all", "examples", "gmpy2", "kernels"}
    assert len(profiles) == len(set(profiles)) == 24
