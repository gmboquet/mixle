"""Engine benchmark timing must synchronize, consume, and parity-check device work."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "engine_benchmark_example.py"


def _load():
    spec = importlib.util.spec_from_file_location("_engine_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_timer_brackets_every_region_with_synchronization(monkeypatch) -> None:
    module = _load()
    events = []
    monkeypatch.setattr(module, "_synchronize", lambda engine: events.append("sync"))
    monkeypatch.setattr(module, "_consume", lambda value: events.append(("consume", value)))
    elapsed, value = module._time(lambda: events.append("call") or 7, object(), repeat=2)
    assert elapsed >= 0.0 and value == 7
    assert events == [
        "call",
        ("consume", 7),
        "sync",
        "sync",
        "call",
        ("consume", 7),
        "sync",
        "sync",
        "call",
        ("consume", 7),
        "sync",
    ]


def test_measured_workloads_have_numerical_parity_gates() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count("np.testing.assert_allclose") >= 3
    assert "torch.cuda.synchronize" in text
    assert "torch.mps.synchronize" in text
