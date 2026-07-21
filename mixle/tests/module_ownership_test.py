import json
from pathlib import Path


def test_every_top_level_public_module_has_an_explicit_ownership_decision():
    root = Path(__file__).parents[2]
    artifact = json.loads((root / "manifests" / "module_ownership.json").read_text())
    observed = set()
    for path in (root / "mixle").iterdir():
        if path.name.startswith("_") or path.name in {"tests", "fixtures"}:
            continue
        if path.is_dir() or path.suffix == ".py":
            observed.add(f"mixle.{path.stem}")
    assert set(artifact["modules"]) == observed
    assert {entry["decision"] for entry in artifact["modules"].values()} <= {
        "retain",
        "narrow",
        "migrate",
        "deprecate",
        "experimental",
    }
    for entry in artifact["modules"].values():
        assert entry["owner"] == "PRJ-CORE"
        assert (entry["decision"] == "migrate") == (entry["destination"] is not None)
