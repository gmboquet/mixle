#!/usr/bin/env python
"""Run the explicit collection-light smoke manifest under a hard 30-second budget."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "mixle" / "tests" / "smoke-manifest.txt"


def manifest_targets(path: Path) -> tuple[str, ...]:
    targets = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("smoke manifest must contain unique explicit targets")
    for target in targets:
        if any(character in target for character in "*?[]"):
            raise ValueError(f"smoke manifest target must not contain a glob: {target}")
        file_part = target.split("::", 1)[0]
        candidate = Path(file_part)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"smoke manifest target escapes the repository: {target}")
        resolved = ROOT / candidate
        if not resolved.is_file() or not resolved.is_relative_to(ROOT / "mixle" / "tests"):
            raise ValueError(f"smoke manifest target is not a test file: {target}")
    return targets


def run_smoke(targets: tuple[str, ...], *, timeout_seconds: int = 30) -> dict:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 0 < timeout_seconds <= 30:
        raise ValueError("smoke timeout must be an integer from 1 through 30 seconds")
    command = [sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "smoke", *targets]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"smoke suite exceeded its {timeout_seconds}-second budget") from exc
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(f"smoke suite failed with exit {result.returncode}:\n{result.stdout}\n{result.stderr}")
    return {
        "artifact": "mixle.smoke_receipt/v1",
        "targets": list(targets),
        "elapsed_seconds": elapsed,
        "budget_seconds": timeout_seconds,
        "passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = run_smoke(manifest_targets(args.manifest))
        text = json.dumps(receipt, sort_keys=True, allow_nan=False)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
    except (OSError, UnicodeError, TypeError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
