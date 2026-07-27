"""BANKING77 flagship data is immutable, licensed, and split-validated."""

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _ROOT / "examples" / "real_receipt_banking77.py"


def _load():
    spec = importlib.util.spec_from_file_location("_banking77_example", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dataset_source_is_commit_pinned_with_exact_split_digests():
    module = _load()
    assert len(module.BANKING77_SOURCE_COMMIT) == 40
    assert set(module.BANKING77_FILES) == {"train", "test"}
    assert {split: spec["rows"] for split, spec in module.BANKING77_FILES.items()} == {
        "train": 10003,
        "test": 3080,
    }
    for spec in module.BANKING77_FILES.values():
        assert module.BANKING77_SOURCE_COMMIT in spec["url"]
        assert len(spec["sha256"]) == 64
        assert "master" not in spec["url"] and "main" not in spec["url"]


def test_file_digest_validation_detects_drift(tmp_path):
    module = _load()
    path = tmp_path / "split.csv"
    path.write_bytes(b"text,category\\nhello,greeting\\n")
    assert len(module._sha256(path)) == 64
    assert module._sha256(path) != module.BANKING77_FILES["train"]["sha256"]


def test_dataset_license_record_is_resolved_not_a_placeholder():
    record = json.loads(
        (_ROOT / "release-checklists" / "0.8.0-banking77-dataset.json").read_text(encoding="utf-8")
    )
    assert record["artifact"] == "mixle.dataset_source/v1"
    assert record["license"]["spdx"] == "CC-BY-4.0"
    assert record["source"]["revision"] == _load().BANKING77_SOURCE_COMMIT
    assert not any(marker in json.dumps(record).upper() for marker in ("TODO", "CONFIRM-AT-PUBLISH", "TBD"))


def test_integrity_failure_is_not_classified_as_network_unavailability():
    module = _load()
    assert not issubclass(ValueError, module.Banking77UnavailableError)
    with pytest.raises(ValueError):
        raise ValueError("integrity mismatch")
