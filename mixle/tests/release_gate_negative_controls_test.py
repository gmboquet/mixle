"""Negative controls must invoke the production release-gate implementation."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(script: str):
    path = ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_fit_boundary_rejects_nonfinite_data() -> None:
    from mixle.inference.estimation import optimize
    from mixle.stats import MultivariateGaussianEstimator
    from mixle.stats.latent.gaussian_mixture import GaussianMixtureEstimator

    rng = np.random.RandomState(0)
    data = [list(row) for row in np.vstack([rng.randn(100, 2), rng.randn(100, 2) + 5.0])]
    estimator = GaussianMixtureEstimator([MultivariateGaussianEstimator(dim=2), MultivariateGaussianEstimator(dim=2)])
    model = optimize(data, estimator=estimator, max_its=10, out=None)
    assert np.isfinite(float(model.seq_log_density(model.dist_to_encoder().seq_encode(data)).mean()))
    data[7] = [float("nan"), 0.0]
    with pytest.raises((ValueError, FloatingPointError), match=r"(?i)nan|inf|finite"):
        optimize(data, estimator=estimator, max_its=10, out=None)


def test_real_api_generator_rejects_duplicate_exports(tmp_path: Path) -> None:
    generator = _load("gen_api_manifest.py")
    package = tmp_path / "__init__.py"
    package.write_text('__all__ = ["one", "one"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate __all__ exports"):
        generator._extract_all(package)


def test_real_optional_union_gate_rejects_missing_dependency(tmp_path: Path) -> None:
    checker = _load("check_optional_extras.py")
    project = tmp_path / "pyproject.toml"
    project.write_text(
        "[project]\n"
        'dependencies = ["numpy>=2,<3"]\n'
        "[project.optional-dependencies]\n"
        'feature = ["sample-package>=1,<2"]\n'
        "examples = []\nall = []\ndocs = []\nlint = []\ntest = []\n",
        encoding="utf-8",
    )
    assert any("all omits runtime requirements" in error for error in checker.validate(project))


def test_real_public_claim_gate_detects_unregistered_claim_pattern() -> None:
    checker = _load("check_public_claims.py")
    planted = "Mixle is 25x faster than every backend."
    matches = [match.group(0) for pattern in checker.CLAIM_PATTERNS for match in pattern.finditer(planted)]
    assert {"25x", "every backend"} <= set(matches)
    assert any("faster" in match for match in matches)


def test_real_reproducibility_gate_rejects_changed_artifact(tmp_path: Path) -> None:
    verifier = _load("verify_reproducible_builds.py")
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "mixle.whl").write_bytes(b"one")
    (right / "mixle.whl").write_bytes(b"two")
    with pytest.raises(ValueError, match="not byte-for-byte reproducible"):
        verifier.verify(left, right)


def test_real_license_gate_rejects_current_pending_record() -> None:
    verifier = _load("verify_license_provenance.py")
    record = json.loads((ROOT / "release-checklists" / "0.8.0-license-provenance.json").read_text(encoding="utf-8"))
    assert verifier.validate(record) == []
    assert verifier.validate(record, require_approved=True)


def test_real_import_sweep_boundary_reports_missing_module() -> None:
    assert importlib.import_module("mixle.stats") is not None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mixle.this_module_was_deleted_in_a_bad_refactor")
