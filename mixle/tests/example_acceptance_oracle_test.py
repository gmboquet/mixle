"""Examples that advertise recovered structure must fail when their oracle is false."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(filename: str):
    path = ROOT / "examples" / filename
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cross_modal_example_emits_and_enforces_acceptance(capsys) -> None:
    module = _load("cross_modal_fit_receipt.py")
    module.main()
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("ACCEPTANCE "))
    receipt = json.loads(line.removeprefix("ACCEPTANCE "))
    assert receipt["artifact"] == "mixle.multi_vector_fit_acceptance/v1"
    assert receipt["accepted"] is True
    assert receipt["price_parents"] == receipt["required_price_parents"] == [1, 2]
    assert receipt["held_out_correlation"] >= 0.95
    assert receipt["held_out_rmse"] <= 6.0


def test_heterogeneous_example_has_a_failing_recovery_oracle(monkeypatch) -> None:
    module = _load("heterogeneous_correctness_example.py")
    bad = [{"mu": 0.0, "probs": [1 / 3, 1 / 3, 1 / 3]}] * 2
    monkeypatch.setattr(module, "fit_mixle", lambda data: bad)
    import pytest

    with pytest.raises(RuntimeError, match="acceptance failed"):
        module.main()
