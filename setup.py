"""Build hook that embeds immutable source provenance in every wheel."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


def _git_value(root: Path, expression: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", expression],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = result.stdout.strip().lower()
    return value if result.returncode == 0 and len(value) == 40 else "unknown"


def _source_dirty(root: Path) -> bool | None:
    declared = os.environ.get("MIXLE_SOURCE_DIRTY", "").strip().lower()
    if declared in {"true", "false"}:
        return declared == "true"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "mixle", "pyproject.toml", "setup.py"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout) if result.returncode == 0 else None


def _source_content_digest(root: Path) -> str:
    digest = hashlib.sha256()
    paths = [root / "pyproject.toml", root / "setup.py"]
    paths.extend(
        path
        for path in (root / "mixle").rglob("*")
        if path.is_file() and path.suffix in {".json", ".py", ".pyx"}
    )
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class ProvenanceBuildPy(build_py):
    """Write build provenance into build_lib without mutating the source checkout."""

    def run(self) -> None:
        super().run()
        root = Path(__file__).resolve().parent
        source_commit = os.environ.get("MIXLE_SOURCE_COMMIT", "").strip().lower()
        if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
            source_commit = _git_value(root, "HEAD")
        source_tree = os.environ.get("MIXLE_SOURCE_TREE", "").strip().lower()
        if len(source_tree) != 40 or any(character not in "0123456789abcdef" for character in source_tree):
            source_tree = _git_value(root, "HEAD^{tree}")
        payload = {
            "artifact": "mixle.build_provenance/v1",
            "source_commit": source_commit,
            "source_tree": source_tree,
            "source_dirty": _source_dirty(root),
            "source_content_sha256": _source_content_digest(root),
        }
        destination = Path(self.build_lib) / "mixle" / "_build_provenance.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


setup(cmdclass={"build_py": ProvenanceBuildPy})
