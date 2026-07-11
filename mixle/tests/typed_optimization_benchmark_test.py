"""Smoke test for the reproducible typed-optimization benchmark harness."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.experimental, pytest.mark.benchmark, pytest.mark.slow]


def test_quick_reuse_panel_writes_provenance_stamped_finite_json(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "typed-optimization.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "typed_optimization.py"),
            "--quick",
            "--reps",
            "1",
            "--panels",
            "reuse",
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["benchmark"] == "typed_optimization"
    assert result["mixle_minor"] == "0.8"
    assert set(result["panels"]) == {"estep_reuse"}
    assert all(case["median_speedup"] > 0.0 for case in result["panels"]["estep_reuse"]["cases"])
