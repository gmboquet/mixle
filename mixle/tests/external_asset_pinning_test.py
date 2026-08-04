"""Evidence-oriented examples must content-address external model and dataset assets."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from mixle.scientist import scientist_asset_manifest

ROOT = Path(__file__).resolve().parents[2]
# Discovered rather than enumerated: an allowlist silently stops covering the next example that
# reaches for a Hugging Face repository, which is exactly how unpinned assets get in.
SEARCH_ROOTS = ("mixle", "examples", "scripts", "benchmarks", "docs")
LOADERS = ("from_pretrained", "SentenceTransformer", "load_dataset")
# ``datasets`` also ships packaged builders whose first argument names a file format, not a repository.
# Those loads read paths the caller already supplies, so an upstream revision is neither available
# nor meaningful for them.
LOCAL_BUILDERS = frozenset(
    {
        "arrow",
        "audiofolder",
        "csv",
        "generator",
        "imagefolder",
        "json",
        "pandas",
        "parquet",
        "sql",
        "text",
        "videofolder",
        "webdataset",
    }
)
# Dataset-driven sources were removed when direct dataset usage moved out of the repository; what
# remains here are the sources that pin MODEL assets (weights), which stay content-addressed.
REQUIRED_SOURCES = (
    "examples/peft_lora_grad_leaf.py",
    "mixle/scientist.py",
    "mixle/tests/quotient_leaf_test.py",
    "mixle/tests/scientist_test.py",
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


def _module_path(module: str) -> Path | None:
    """Locate a repository module named by an absolute import, if this repository defines it."""
    dotted = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    if dotted.is_file():
        return dotted
    # Smoke gates put ``examples/`` on ``sys.path`` and import an example's pins by bare module name.
    example = ROOT / "examples" / f"{module.rsplit('.', 1)[-1]}.py"
    return example if example.is_file() else None


def _imported_constants(tree: ast.Module) -> dict[str, str]:
    """Resolve names imported from a sibling repository module to their string constants.

    A pin shared between an example and its smoke gate legitimately arrives as an imported name
    rather than a local literal; that is still content addressed and must not be reported.
    """
    values: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None or node.level:
            continue
        source = _module_path(node.module)
        if source is None:
            continue
        constants = _constants(ast.parse(source.read_text(encoding="utf-8")))
        for alias in node.names:
            if alias.name in constants:
                values[alias.asname or alias.name] = constants[alias.name]
    return values


def _resolved_string(node: ast.expr, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _audit(source: str, label: str) -> tuple[int, list[str]]:
    """Return how many upstream asset loads ``source`` makes and which are not content addressed."""
    tree = ast.parse(source)
    constants = _imported_constants(tree) | _constants(tree)
    checked = 0
    unpinned: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name not in LOADERS:
            continue
        if node.args and _resolved_string(node.args[0], constants) in LOCAL_BUILDERS:
            continue
        checked += 1
        revision = next((item.value for item in node.keywords if item.arg == "revision"), None)
        if revision is None:
            unpinned.append(f"{label}:{node.lineno} has no revision=")
            continue
        value = _resolved_string(revision, constants)
        if value is None or not FULL_REVISION.fullmatch(value):
            unpinned.append(f"{label}:{node.lineno} revision must resolve to a full commit SHA")
    return checked, unpinned


def _external_sources() -> list[Path]:
    """Every repository source that names a Hugging Face model or dataset loader."""
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        for path in (ROOT / root).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if any(loader in path.read_text(encoding="utf-8") for loader in LOADERS):
                found.append(path)
    return sorted(found)


def test_discovery_covers_every_source_that_loads_an_external_asset() -> None:
    discovered = {path.relative_to(ROOT).as_posix() for path in _external_sources()}
    missing = sorted(set(REQUIRED_SOURCES) - discovered)
    assert missing == [], f"external-asset sources are outside the pinning gate: {missing}"


def test_all_external_hugging_face_calls_use_full_commit_revisions() -> None:
    calls_checked = 0
    unpinned: list[str] = []
    for path in _external_sources():
        checked, findings = _audit(path.read_text(encoding="utf-8"), path.relative_to(ROOT).as_posix())
        calls_checked += checked
        unpinned.extend(findings)
    assert unpinned == [], "external assets are not content addressed:\n" + "\n".join(unpinned)
    # A floor on how many loader calls the audit actually saw, so a scanner regression that silently
    # matches nothing cannot pass as "no findings". Re-calibrated 2026-08-04 when the dataset-driven
    # examples were removed with all direct dataset usage: the remaining pinned calls are the MODEL
    # loads (mixle/scientist.py CLIP + language model + sentence encoder, the PEFT example, the
    # quotient-leaf and scientist test fixtures) -- nine today. Eight guards the scanner without
    # pinning the exact inventory.
    assert calls_checked >= 8


def test_audit_rejects_mutable_and_abbreviated_asset_identities() -> None:
    assert _audit('load_dataset("uoft-cs/cifar10", split="train")\n', "planted")[1]
    assert _audit('load_dataset("uoft-cs/cifar10", revision="0b27149")\n', "planted")[1]
    assert _audit('CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\n', "planted")[1]
    pinned = (
        'REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"\nload_dataset("uoft-cs/cifar10", revision=REVISION)\n'
    )
    assert _audit(pinned, "planted") == (1, [])


def test_packaged_builder_loads_need_no_upstream_revision() -> None:
    assert _audit('load_dataset("csv", data_files={"train": "a.csv"})\n', "planted") == (0, [])


def test_scientist_asset_manifest_is_content_addressed() -> None:
    assets = scientist_asset_manifest()
    assert set(assets) == {"clip", "language_model", "sentence_encoder"}
    assert all(FULL_REVISION.fullmatch(asset["revision"]) for asset in assets.values())


def test_peft_receipt_measures_adapter_movement_and_held_out_likelihood() -> None:
    source = (ROOT / "examples" / "peft_lora_grad_leaf.py").read_text(encoding="utf-8")
    assert "initial = -torch.log" in source
    assert "adapters_before" in source
    assert "adapter_deltas" in source
    assert "held_out = toy_token_sequences" in source
    assert "rng=np.random.RandomState(1)" in source
    assert "after_ll > before_ll" in source
