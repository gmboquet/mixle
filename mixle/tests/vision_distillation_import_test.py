"""Importing the vision trainer must not load assets, train, or write artifacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "vision_edge_distillation" / "distill_clip_features.py"


def test_import_defines_student_without_execution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("_distill_clip_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.Student is not None
    assert not (tmp_path / "student.pt").exists()
    assert not (tmp_path / "student_head.pt").exists()
    assert not (tmp_path / "metrics.json").exists()


def test_network_and_training_calls_are_under_main_guard() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    main_start = text.index("def main(")
    for fragment in ("from datasets import load_dataset", "CLIPModel.from_pretrained", "torch.save("):
        assert text.index(fragment) > main_start
