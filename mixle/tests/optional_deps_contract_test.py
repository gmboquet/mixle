"""Optional dependency probes distinguish absence from broken installations."""

import pytest

import mixle.utils.optional_deps as optional_deps


def test_exact_target_absence_is_the_only_suppressed_failure(monkeypatch):
    def missing_target(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(optional_deps, "import_module", missing_target)
    assert optional_deps._import_optional("optional_target") is None


def test_missing_internal_dependency_propagates(monkeypatch):
    failure = ModuleNotFoundError(
        "No module named 'internal_binary_dependency'",
        name="internal_binary_dependency",
    )

    def broken_target(_name):
        raise failure

    monkeypatch.setattr(optional_deps, "import_module", broken_target)
    with pytest.raises(ModuleNotFoundError) as caught:
        optional_deps._import_optional("optional_target")
    assert caught.value is failure


def test_plain_import_error_from_present_package_propagates(monkeypatch):
    failure = ImportError("binary extension has an undefined symbol")

    def broken_target(_name):
        raise failure

    monkeypatch.setattr(optional_deps, "import_module", broken_target)
    with pytest.raises(ImportError) as caught:
        optional_deps._import_optional("optional_target")
    assert caught.value is failure
