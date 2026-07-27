"""The fused fuzz soak must reject zero work and emit a machine receipt."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    path = ROOT / "scripts" / "fuzz_fused_soak.py"
    spec = importlib.util.spec_from_file_location("_fuzz_fused_soak", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nonpositive_samples_are_rejected_before_fuzzer_import() -> None:
    module = _load()
    with pytest.raises(SystemExit) as exc:
        module.main(["--samples", "0"])
    assert exc.value.code == 2


def test_success_emits_completed_property_count(monkeypatch, capsys) -> None:
    module = _load()
    fake = types.ModuleType("fused_fuzz_test")
    fake.SIGNATURE_POOL = ["one"]
    fake.check_sample = lambda test_case, rng, sig=None: None
    monkeypatch.setitem(sys.modules, "fused_fuzz_test", fake)
    assert module.main(["--samples", "2", "--seed", "7"]) == 0
    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt == {
        "artifact": "mixle.fused_fuzz_soak/v1",
        "failures": 0,
        "property_checks_completed": 10,
        "samples_completed": 2,
        "samples_requested": 2,
        "seed": 7,
    }
