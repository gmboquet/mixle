#!/usr/bin/env python
"""Run a required hosted pytest target and fail if any selected case skips."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from importlib.metadata import distributions
from pathlib import Path

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _count(root: ET.Element, name: str) -> int:
    if root.tag == "testsuites":
        return sum(int(suite.attrib.get(name, "0")) for suite in root.findall("testsuite"))
    return int(root.attrib.get(name, "0"))


def run(pytest_args: list[str], *, candidate_sha: str, timeout_seconds: int = 900) -> dict:
    if not _GIT_SHA.fullmatch(candidate_sha):
        raise ValueError("candidate SHA must be a full lowercase Git SHA")
    if not pytest_args:
        raise ValueError("at least one explicit pytest target is required")
    if any("*" in argument or argument in {".", "mixle/tests"} for argument in pytest_args):
        raise ValueError("required hosted tests must use explicit targets, not globs or broad directories")
    with tempfile.TemporaryDirectory() as directory:
        junit = Path(directory) / "required.xml"
        command = [sys.executable, "-m", "pytest", *pytest_args, f"--junitxml={junit}"]
        started = time.monotonic()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        duration = time.monotonic() - started
        if result.returncode != 0:
            raise RuntimeError(f"required hosted pytest failed with exit {result.returncode}: {result.stdout[-2000:]}")
        root = ET.parse(junit).getroot()  # nosec B314 # not untrusted XML: junit is a path inside the TemporaryDirectory this function created, and its only writer is the pytest subprocess this function spawned two statements earlier
    tests = _count(root, "tests")
    skipped = _count(root, "skipped")
    failures = _count(root, "failures")
    errors = _count(root, "errors")
    if tests < 1 or skipped or failures or errors:
        raise RuntimeError(
            f"required hosted pytest did not execute cleanly: tests={tests}, skipped={skipped}, "
            f"failures={failures}, errors={errors}"
        )
    dependencies = sorted(
        {distribution.metadata["Name"].lower(): distribution.version for distribution in distributions()}.items()
    )
    return {
        "artifact": "mixle.required_hosted_pytest/v1",
        "candidate_commit": candidate_sha,
        "command": command[:-1],
        "duration_seconds": duration,
        "tests": tests,
        "skipped": skipped,
        "failures": failures,
        "errors": errors,
        "resolved_dependencies": dict(dependencies),
        "passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    try:
        receipt = run(pytest_args, candidate_sha=args.candidate_sha)
        args.receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ET.ParseError, subprocess.TimeoutExpired, TypeError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
