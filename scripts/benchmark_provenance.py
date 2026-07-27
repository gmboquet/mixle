"""Worklist B7.3 -- stamp benchmark results with the version that produced them.

B7.3's deliverable is that no headline performance number comes from a stale (0.5.x /
0.6.x) artifact. The mechanism: every published benchmark result carries the mixle
version and commit it was produced on, and a gate rejects results whose version does not
match the release being prepared. Re-running the panels on 0.8.0 then means: produce
results whose stamp says 0.8.

This module is the stamping helper and the staleness check. The benchmark harness (see
``benchmarks/``) calls ``stamp_result`` when it writes a result; the gate test
(``benchmark_provenance_test.py``) calls ``is_current`` over any results files present.
"""

from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _package_version() -> str:
    """Read the version from this checkout, falling back to installed metadata."""
    try:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        import mixle

        return getattr(mixle, "__version__", "0.0.0")


def minor_of(version: str) -> str:
    """The major.minor prefix -- the granularity a release's headline numbers are tied to."""
    return ".".join(version.split(".")[:2])


def _full_sha(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    sha = value.strip().lower()
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        return None
    return sha


def _artifact_digest(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        return None
    return digest


def _git_commit() -> str:
    for env_var in ("MIXLE_BENCH_COMMIT", "GITHUB_SHA"):
        if sha := _full_sha(os.environ.get(env_var)):
            return sha
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and (sha := _full_sha(out.stdout)):
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def stamp_result(
    result: dict[str, Any],
    *,
    commit: str | None = None,
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Attach exact candidate provenance; unverified local runs remain visibly ineligible."""
    version = _package_version()
    resolved_commit = _full_sha(commit) or _full_sha(_git_commit())
    resolved_artifact = _artifact_digest(artifact_sha256) or _artifact_digest(
        os.environ.get("MIXLE_BENCH_ARTIFACT_SHA256")
    )
    return {
        **result,
        "mixle_version": version,
        "mixle_minor": minor_of(version),
        "mixle_commit": resolved_commit or "unknown",
        "mixle_artifact_sha256": resolved_artifact or "unverified",
    }


def is_current(
    result: dict[str, Any],
    *,
    current_version: str | None = None,
    current_commit: str | None = None,
    artifact_sha256: str | None = None,
) -> bool:
    """Whether a result is bound to the exact approved version, commit, and artifact."""
    expected_commit = _full_sha(current_commit) or _full_sha(_git_commit())
    expected_artifact = _artifact_digest(artifact_sha256) or _artifact_digest(
        os.environ.get("MIXLE_BENCH_ARTIFACT_SHA256")
    )
    if expected_commit is None or expected_artifact is None:
        return False
    return (
        result.get("mixle_version") == (current_version or _package_version())
        and _full_sha(result.get("mixle_commit")) == expected_commit
        and _artifact_digest(result.get("mixle_artifact_sha256")) == expected_artifact
    )


def stale_results(
    results: list[dict[str, Any]],
    *,
    current_version: str | None = None,
    current_commit: str | None = None,
    artifact_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Results not bound to the exact approved release candidate."""
    return [
        result
        for result in results
        if not is_current(
            result,
            current_version=current_version,
            current_commit=current_commit,
            artifact_sha256=artifact_sha256,
        )
    ]


if __name__ == "__main__":
    import json

    demo = stamp_result(
        {"benchmark": "gmm_fit", "n": 100000, "seconds": 0.098},
        artifact_sha256="0" * 64,
    )
    print(json.dumps(demo, indent=2))
