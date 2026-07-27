"""Public claim surfaces must resolve to retained evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _checker():
    path = ROOT / "scripts" / "check_public_claims.py"
    spec = importlib.util.spec_from_file_location("_check_public_claims", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_claim_inventory_resolves_all_scanner_hits() -> None:
    checker = _checker()
    data = json.loads((ROOT / "manifests" / "public_claims.json").read_text(encoding="utf-8"))
    assert checker.validate(data) == []


def test_unregistered_quantitative_claim_is_detected() -> None:
    checker = _checker()
    planted = "Mixle is 25x faster than every backend."
    matches = [match.group(0) for pattern in checker.CLAIM_PATTERNS for match in pattern.finditer(planted)]
    assert "25x" in matches
    assert any("faster" in match for match in matches)
    assert "every backend" in matches
