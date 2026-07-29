"""Release-facing examples and notes must not revive unaudited headline claims."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_documentation_does_not_suppress_all_docutils_warnings() -> None:
    config = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    assert config.count("suppress_warnings =") == 1
    assert '"docutils"' not in config


def test_vision_verifier_authenticates_restricted_tensor_loads() -> None:
    verifier = (ROOT / "examples" / "vision_edge_distillation" / "verify_on_laptop.py").read_text(encoding="utf-8")
    assert verifier.count("weights_only=True") == 2
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
