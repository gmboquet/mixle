"""Released documentation is immutable, authenticated, and preferred at the site root."""

from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_assembled_site_validator_accepts_development_landing(tmp_path):
    development = tmp_path / "development"
    archives = tmp_path / "archives"
    output = tmp_path / "site"
    development.mkdir()
    archives.mkdir()
    (development / "index.html").write_text("development", encoding="utf-8")
    _load("assemble_docs_site").assemble(development, archives, output)
    assert _load("verify_docs_site").validate(output) == []
    (output / "main" / "index.html").unlink()
    assert "assembled documentation lacks main/index.html" in _load("verify_docs_site").validate(output)


def _docs(path: Path, label: str) -> Path:
    path.mkdir()
    (path / "index.html").write_text(f"<h1>{label}</h1>", encoding="utf-8")
    (path / "asset.css").write_text("body {}\n", encoding="utf-8")
    return path


def test_deterministic_archive_and_stable_root(tmp_path: Path) -> None:
    package = _load("package_docs_archive")
    assemble = _load("assemble_docs_site")
    source = _docs(tmp_path / "release", "release")
    archives = tmp_path / "archives"
    first = archives / "mixle-docs-v0.8.0.tar.gz"
    second = tmp_path / "copy" / first.name
    digest = package.package(source, first, version="0.8.0", commit="a" * 40, epoch=123)
    assert package.package(source, second, version="0.8.0", commit="a" * 40, epoch=123) == digest
    assert first.read_bytes() == second.read_bytes()

    site = tmp_path / "site"
    switcher = assemble.assemble(_docs(tmp_path / "development", "development"), archives, site)
    assert [entry["name"] for entry in switcher] == ["main", "v0.8.0"]
    assert "v0.8.0/index.html" in (site / "index.html").read_text(encoding="utf-8")
    assert "release" in (site / "v0.8.0" / "index.html").read_text(encoding="utf-8")
    metadata = json.loads((site / "v0.8.0" / "docs-build.json").read_text(encoding="utf-8"))
    assert metadata["commit"] == "a" * 40


def test_no_release_archive_labels_main_as_development(tmp_path: Path) -> None:
    assemble = _load("assemble_docs_site")
    site = tmp_path / "site"
    assemble.assemble(_docs(tmp_path / "development", "development"), tmp_path / "empty", site)
    root = (site / "index.html").read_text(encoding="utf-8")
    assert "development documentation" in root
    assert 'http-equiv="refresh"' not in root


def test_changed_bytes_and_archive_traversal_fail_closed(tmp_path: Path) -> None:
    package = _load("package_docs_archive")
    assemble = _load("assemble_docs_site")
    source = _docs(tmp_path / "release", "release")
    archives = tmp_path / "archives"
    archive = archives / "mixle-docs-v0.8.0.tar.gz"
    package.package(source, archive, version="0.8.0", commit="a" * 40, epoch=123)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        assemble.assemble(_docs(tmp_path / "development", "development"), archives, tmp_path / "site")

    unsafe = archives / "mixle-docs-v0.9.0.tar.gz"
    with tarfile.open(unsafe, "w:gz") as bundle:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        import io

        bundle.addfile(info, io.BytesIO(b"x"))
    import hashlib

    unsafe.with_suffix(unsafe.suffix + ".sha256").write_text(
        f"{hashlib.sha256(unsafe.read_bytes()).hexdigest()}  {unsafe.name}\n",
        encoding="ascii",
    )
    archive.unlink()
    archive.with_suffix(archive.suffix + ".sha256").unlink()
    with pytest.raises(ValueError, match="unsafe archive member"):
        assemble.assemble(_docs(tmp_path / "development-2", "development"), archives, tmp_path / "site-2")


def test_workflows_archive_once_and_never_rebuild_releases() -> None:
    publish = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    docs = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    poly = (ROOT / "docs" / "poly.py").read_text(encoding="utf-8")
    assert "package_docs_archive.py" in publish
    assert "candidate/docs-dist/" in publish
    assert "assemble_docs_site.py" in docs
    assert "gh release download" in docs
    assert "sphinx-polyversion" not in docs
    assert 'TAG_REGEX = r"(?!)"' in poly
