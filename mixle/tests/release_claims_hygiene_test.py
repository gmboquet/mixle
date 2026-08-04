"""Release-facing examples and notes must not revive unaudited headline claims."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_documentation_does_not_suppress_all_docutils_warnings() -> None:
    config = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    assert config.count("suppress_warnings =") == 1
    assert '"docutils"' not in config


def test_vision_verifier_authenticates_restricted_tensor_loads() -> None:
    verifier = (ROOT / "examples" / "vision_edge_distillation" / "verify_on_laptop.py").read_text(encoding="utf-8")
    # Asserted as a property of the CALL SITES, not as a raw occurrence count. The count form read
    # `== 2` and broke the moment the module docstring mentioned the flag it documents -- a prose edit
    # that strengthened the file failed a security test, while the actual protection was untouched.
    # What matters is that no torch.load in this file omits weights_only, however many times the
    # string appears elsewhere.
    loads = [line for line in verifier.splitlines() if "torch.load(" in line and "def " not in line]
    assert len(loads) == 2
    assert all("weights_only=True" in line for line in loads)
    assert verifier.count("_authenticate(") >= 4
    assert verifier.index("_authenticate(student_path") < verifier.index("torch.load(student_path")
    assert verifier.index("_authenticate(head_path") < verifier.index("torch.load(head_path")


def test_release_surfaces_exclude_retracted_headline_numbers() -> None:
    paths = [
        ROOT / "CHANGELOG.md",
        ROOT / "examples" / "foundation_to_edge.py",
        ROOT / "examples" / "laptop_scientist.py",
        ROOT / "examples" / "project_neural_to_structured.py",
        ROOT / "examples" / "real_receipt_banking77.py",
        ROOT / "examples" / "skeptic_challenge_example.py",
        ROOT / "examples" / "vision_edge_distillation" / "README.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for retracted in (
        "1.3-2.3x",
        "1.59x",
        "~76x",
        "0.983",
        "0.679 -> 0.584",
        "14760x",
        "14,760x",
        "118×",
        "0.8165",
        "~35x",
        "~7x",
    ):
        assert retracted not in combined


def test_changelog_has_current_comparison_boundaries() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "[Unreleased]: https://github.com/gmboquet/mixle/compare/v0.7.0...HEAD" in changelog
    assert "[0.8.0]: https://github.com/gmboquet/mixle/compare/v0.7.0...v0.8.0" in changelog
    assert "[0.7.0]: https://github.com/gmboquet/mixle/compare/v0.6.2...v0.7.0" in changelog
