"""Worklist M11.5 -- keep deployment claims bounded to what the core package is.

The core mixle package provides deployment building blocks (artifacts, scoring
wrappers, drift/provenance helpers, local registries), NOT a complete serving platform;
serving is the companion `mixle-mlops` project's job. M11.5's acceptance is that public
wording agrees with the maturity guide and companion-package responsibilities.

This test pins that agreement so the boundary cannot quietly erode:
  * ``docs/production.rst`` must carry the explicit Deployment Scope statement -- the
    four things the core provides, the explicit "does not provide ... serving platform"
    disclaimer, and a pointer to mixle-mlops;
  * the README must not claim the core itself is a serving/production platform, and must
    route serving to mixle-mlops.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTION_RST = ROOT / "docs" / "production.rst"
README = ROOT / "README.md"


def _read(path: Path) -> str:
    if not path.is_file():
        pytest.skip(f"{path} not found")
    return path.read_text(encoding="utf-8")


def test_production_doc_states_the_scope_boundary() -> None:
    text = _read(PRODUCTION_RST).lower()
    assert "deployment scope" in text, "production.rst is missing the Deployment Scope section"

    # The four building blocks the core *does* provide must be named.
    for provided in ("model artifacts", "scoring wrapper", "drift", "provenance", "local registr"):
        assert provided in text, f"Deployment Scope does not name the core capability: {provided!r}"

    # The explicit disclaimer that the core is not a serving platform.
    assert "does not" in text and "serving platform" in text, (
        "production.rst must explicitly disclaim being a complete serving platform"
    )
    # ...and point the serving responsibility at the companion project.
    assert "mixle-mlops" in text, "Deployment Scope must route serving to mixle-mlops"


def test_production_doc_ties_to_maturity() -> None:
    """The boundary must reference the maturity framing, per the acceptance criterion."""
    text = _read(PRODUCTION_RST).lower()
    assert "maturity" in text, "Deployment Scope must tie the boundary to the maturity guide"


def test_readme_does_not_claim_core_is_a_serving_platform() -> None:
    text = _read(README).lower()
    # These would overclaim the *core* as a full platform.
    for banned in ("serving platform", "production platform", "complete mlops", "full mlops platform"):
        assert banned not in text, (
            f"README describes the core package as a {banned!r}; that belongs to mixle-mlops (M11.5)"
        )


def test_readme_routes_serving_to_companion() -> None:
    text = _read(README)
    assert "mixle-mlops" in text, "README must route serving/gateway responsibility to mixle-mlops"
