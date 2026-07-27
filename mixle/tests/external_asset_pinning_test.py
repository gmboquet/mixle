"""Evidence-oriented examples must content-address external model and dataset assets."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from mixle.scientist import scientist_asset_manifest

ROOT = Path(__file__).resolve().parents[2]
PINNED_SOURCES = (
    ROOT / "mixle" / "scientist.py",
    ROOT / "mixle" / "tests" / "scientist_test.py",
    ROOT / "examples" / "foundation_to_edge.py",
    ROOT / "examples" / "laptop_scientist.py",
    ROOT / "examples" / "peft_lora_grad_leaf.py",
    ROOT / "examples" / "vision_edge_distillation" / "distill_clip_features.py",
    ROOT / "examples" / "vision_edge_distillation" / "verify_on_laptop.py",
)
FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _constants(tree: ast.Module) -> dict[str, str]:
    values: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
    return values


def _resolved_string(node: ast.expr, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def test_all_external_hugging_face_calls_use_full_commit_revisions() -> None:
    calls_checked = 0
    for path in PINNED_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name not in {"from_pretrained", "SentenceTransformer", "load_dataset"}:
                continue
            revision = next((item.value for item in node.keywords if item.arg == "revision"), None)
            assert revision is not None, f"{path.relative_to(ROOT)}:{node.lineno} has no revision="
            value = _resolved_string(revision, constants)
            assert value is not None and FULL_REVISION.fullmatch(value), (
                f"{path.relative_to(ROOT)}:{node.lineno} revision must resolve to a full commit SHA"
            )
            calls_checked += 1
    assert calls_checked >= 18


def test_scientist_asset_manifest_is_content_addressed() -> None:
    assets = scientist_asset_manifest()
    assert set(assets) == {"clip", "language_model", "sentence_encoder"}
    assert all(FULL_REVISION.fullmatch(asset["revision"]) for asset in assets.values())


def test_distillation_receipt_binds_assets_and_verifier_checks_them() -> None:
    producer = PINNED_SOURCES[-2].read_text(encoding="utf-8")
    verifier = PINNED_SOURCES[-1].read_text(encoding="utf-8")
    assert '"assets": {' in producer
    assert '"train_fingerprint": train._fingerprint' in producer
    assert '"test_fingerprint": test._fingerprint' in producer
    assert 'if metrics.get("assets") != expected_assets:' in verifier


def test_peft_receipt_measures_adapter_movement_and_held_out_likelihood() -> None:
    source = (ROOT / "examples" / "peft_lora_grad_leaf.py").read_text(encoding="utf-8")
    assert "initial = -torch.log" in source
    assert "adapters_before" in source
    assert "adapter_deltas" in source
    assert "held_out = toy_token_sequences" in source
    assert "rng=np.random.RandomState(1)" in source
    assert "after_ll > before_ll" in source
