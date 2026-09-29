"""Verify that every family member's release branch is in lockstep with core (manifests/family_release.json).

The manifest names the release, the shared branch and the version core carries; each member must declare
that version in its own format (PEP 440 for Python packages, semver for the Node workspace), carry the
changelog heading for the cut, and pin core and the other members at the family branch where it depends
on them. Core itself must declare the manifest's version in ``pyproject.toml``.

Two sources: by default each member's files are read from GitHub at the manifest's branch through ``gh``;
``--local ROOT`` reads them from ``ROOT/<member name>/`` (checkouts or worktrees) instead, which is how a
receipt is measured offline. Exit status 0 means every check passed; 1 means at least one did not; 2
means a member could not be read at all.

    python scripts/check_family_sync.py
    python scripts/check_family_sync.py --local /path/to/family-worktrees
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "family_release.json"

_PEP440 = re.compile(
    r"^(?P<base>\d+\.\d+\.\d+)(?:(?P<kind>a|b|rc)(?P<n>\d+))?(?:\.post(?P<post>\d+))?(?:\.dev(?P<dev>\d+))?$"
)


def semver_form(pep440: str) -> str:
    """The semver spelling of a PEP 440 version: ``0.8.3rc1`` -> ``0.8.3-rc.1``, ``0.8.3.dev0`` -> ``0.8.3-dev.0``."""
    match = _PEP440.match(pep440)
    if match is None:
        raise ValueError(f"not a PEP 440 version this family uses: {pep440!r}")
    out = match.group("base")
    parts = []
    if match.group("kind"):
        parts.append(f"{match.group('kind')}.{match.group('n')}")
    if match.group("post"):
        parts.append(f"post.{match.group('post')}")
    if match.group("dev"):
        parts.append(f"dev.{match.group('dev')}")
    return out + ("-" + ".".join(parts) if parts else "")


def expected_version(manifest_version: str, version_format: str) -> str | None:
    if version_format == "pep440":
        return manifest_version
    if version_format == "semver":
        return semver_form(manifest_version)
    if version_format == "none":
        return None
    raise ValueError(f"unknown version format {version_format!r}")


def declared_version(path: str, text: str) -> str:
    if path.endswith("pyproject.toml"):
        return str(tomllib.loads(text)["project"]["version"])
    if path.endswith("package.json"):
        return str(json.loads(text)["version"])
    raise ValueError(f"no rule for reading a version out of {path}")


class Source:
    """Where a member's files come from."""

    def __init__(self, local_root: Path | None, branch: str) -> None:
        self.local_root = local_root
        self.branch = branch

    def read(self, member: dict, path: str) -> str | None:
        if self.local_root is not None:
            file = self.local_root / member["name"] / path
            return file.read_text(encoding="utf-8") if file.is_file() else None
        result = subprocess.run(
            ["gh", "api", f"repos/{member['repository']}/contents/{path}?ref={self.branch}", "--jq", ".content"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return base64.b64decode(result.stdout.strip().replace("\n", "")).decode("utf-8")

    def tip(self, member: dict) -> str:
        if self.local_root is not None:
            result = subprocess.run(
                ["git", "-C", str(self.local_root / member["name"]), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or "?"
        result = subprocess.run(
            ["gh", "api", f"repos/{member['repository']}/branches/{self.branch}", "--jq", ".commit.sha"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()[:8] or "?"


def check_member(member: dict, manifest: dict, source: Source) -> tuple[list[str], list[str]]:
    """Return (passed, failed) descriptions for one member."""
    passed: list[str] = []
    failed: list[str] = []
    want = expected_version(manifest["version"], member["version_format"])
    version_files = ([member["version_file"]] if member.get("version_file") else []) + list(
        member.get("additional_version_files", [])
    )
    for path in version_files:
        text = source.read(member, path)
        if text is None:
            failed.append(f"{path}: unreadable at {manifest['branch']}")
            continue
        try:
            have = declared_version(path, text)
        except (KeyError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
            failed.append(f"{path}: no version could be read ({exc})")
            continue
        (passed if have == want else failed).append(
            f"{path}: version {have}" + ("" if have == want else f" (want {want})")
        )
    changelog = source.read(member, member["changelog_file"])
    heading = member["changelog_heading"]
    if changelog is None:
        failed.append(f"{member['changelog_file']}: unreadable")
    elif any(line.startswith(heading) for line in changelog.splitlines()):
        passed.append(f"{member['changelog_file']}: has heading {heading!r}")
    else:
        failed.append(f"{member['changelog_file']}: no heading starting {heading!r}")
    for pin in member.get("core_pins", []):
        text = source.read(member, pin["file"])
        if text is None:
            failed.append(f"{pin['file']}: unreadable")
        elif pin["must_contain"] in text:
            passed.append(f"{pin['file']}: pins {pin['must_contain']}")
        else:
            failed.append(f"{pin['file']}: missing pin {pin['must_contain']}")
    return passed, failed


def check_core(manifest: dict, core_root: Path) -> tuple[list[str], list[str]]:
    core = manifest["core"]
    text = (core_root / core["version_file"]).read_text(encoding="utf-8")
    have = declared_version(core["version_file"], text)
    want = expected_version(manifest["version"], core["version_format"])
    line = f"{core['name']} {core['version_file']}: version {have}"
    return ([line], []) if have == want else ([], [line + f" (manifest says {want})"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--local", type=Path, default=None, help="read members from ROOT/<name>/ instead of GitHub")
    parser.add_argument("--core-root", type=Path, default=ROOT, help="the core checkout whose pyproject is checked")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("artifact") != "mixle.family_release/v1":
        print("unsupported family manifest artifact", file=sys.stderr)
        return 2
    source = Source(args.local, manifest["branch"])
    report: dict[str, dict] = {}
    passed, failed = check_core(manifest, args.core_root)
    report["mixle"] = {"tip": "-", "passed": passed, "failed": failed}
    unreadable = 0
    for member in manifest["members"]:
        passed, failed = check_member(member, manifest, source)
        if all("unreadable" in item for item in failed) and failed and not passed:
            unreadable += 1
        report[member["name"]] = {"tip": source.tip(member), "passed": passed, "failed": failed}
    if args.json:
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "branch": manifest["branch"],
                    "version": manifest["version"],
                    "members": report,
                },
                indent=2,
            )
        )
    else:
        print(
            f"family manifest {args.manifest.name}: branch {manifest['branch']}, version {manifest['version']} ({manifest['policy']['version_policy']})"
        )
        for name, entry in report.items():
            state = "OK " if not entry["failed"] else "DRIFT"
            print(f"  {state} {name} @ {entry['tip']}")
            for item in entry["passed"]:
                print(f"        ok   {item}")
            for item in entry["failed"]:
                print(f"        FAIL {item}")
    drifted = [name for name, entry in report.items() if entry["failed"]]
    if unreadable:
        print(f"{unreadable} member(s) could not be read", file=sys.stderr)
        return 2
    if drifted:
        print(f"out of sync: {', '.join(drifted)}", file=sys.stderr)
        return 1
    print("every member is in lockstep with core")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
