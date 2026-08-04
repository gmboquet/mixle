"""Release-facing examples and notes must not revive unaudited headline claims."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_documentation_does_not_suppress_all_docutils_warnings() -> None:
    config = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    assert config.count("suppress_warnings =") == 1
    assert '"docutils"' not in config


def test_release_surfaces_exclude_retracted_headline_numbers() -> None:
    # The dataset-driven examples this list once covered (foundation_to_edge, laptop_scientist,
    # real_receipt_banking77, vision_edge_distillation) were removed when direct dataset usage moved
    # out of the repository; the retracted numbers must stay absent from what remains.
    paths = [
        ROOT / "CHANGELOG.md",
        ROOT / "examples" / "project_neural_to_structured.py",
        ROOT / "examples" / "skeptic_challenge_example.py",
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
