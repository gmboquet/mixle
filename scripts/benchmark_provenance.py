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
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _package_version() -> str:
    import mixle

    return getattr(mixle, "__version__", "0.0.0")


def minor_of(version: str) -> str:
    """The major.minor prefix -- the granularity a release's headline numbers are tied to."""
    return ".".join(version.split(".")[:2])


def _git_commit() -> str:
    for env_var in ("MIXLE_BENCH_COMMIT", "GITHUB_SHA"):
        sha = os.environ.get(env_var)
        if sha:
            return sha[:12]
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def stamp_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return ``result`` with mixle version/minor/commit provenance attached."""
    version = _package_version()
    return {
        **result,
        "mixle_version": version,
        "mixle_minor": minor_of(version),
        "mixle_commit": _git_commit(),
    }


def is_current(result: dict[str, Any], *, current_version: str | None = None) -> bool:
    """Whether ``result`` was produced by the current release line (major.minor match)."""
    current = minor_of(current_version or _package_version())
    stamped = result.get("mixle_minor")
    if stamped is None:
        version = result.get("mixle_version")
        stamped = minor_of(version) if version else None
    return stamped == current


def stale_results(results: list[dict[str, Any]], *, current_version: str | None = None) -> list[dict[str, Any]]:
    """Results that are unstamped or stamped with a different release line."""
    return [r for r in results if not is_current(r, current_version=current_version)]


if __name__ == "__main__":
    import json

    demo = stamp_result({"benchmark": "gmm_fit", "n": 100000, "seconds": 0.098})
    print(json.dumps(demo, indent=2))
