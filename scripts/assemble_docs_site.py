#!/usr/bin/env python3
"""Assemble development docs with authenticated immutable release archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ARCHIVE = re.compile(r"^mixle-docs-v(?P<version>\d+\.\d+\.\d+)\.tar\.gz$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))  # type: ignore[return-value]


def _expected_digest(archive: Path) -> str:
    digest_file = archive.with_suffix(archive.suffix + ".sha256")
    if not digest_file.is_file():
        raise ValueError(f"missing digest for {archive.name}")
    parts = digest_file.read_text(encoding="ascii").strip().split()
    if len(parts) != 2 or parts[1] != archive.name or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
        raise ValueError(f"malformed digest file {digest_file.name}")
    return parts[0]


def _extract(archive_path: Path, destination: Path, version: str) -> dict:
    expected = _expected_digest(archive_path)
    actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {archive_path.name}")
    destination.mkdir(parents=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported archive member {member.name!r}")
            target = destination.joinpath(*name.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member {member.name!r}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    metadata_path = destination / "docs-build.json"
    if not metadata_path.is_file() or not (destination / "index.html").is_file():
        raise ValueError(f"{archive_path.name} lacks required documentation files")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("artifact") != "mixle.release_docs/v1"
        or metadata.get("version") != version
        or not COMMIT.fullmatch(str(metadata.get("commit", "")))
    ):
        raise ValueError(f"{archive_path.name} metadata does not match its release identity")
    return metadata


def _root_page(latest: str | None) -> str:
    if latest is None:
        return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>mixle development documentation</title></head>
<body><h1>mixle development documentation</h1>
<p>No stable release documentation archive is available.</p>
<p><a href="main/index.html">Open main-branch documentation (development)</a></p></body></html>
"""
    target = f"v{latest}/index.html"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>mixle {latest} documentation</title>
<meta http-equiv="refresh" content="0; url={target}"><link rel="canonical" href="{target}"></head>
<body><p>Redirecting to <a href="{target}">stable mixle {latest} documentation</a>...</p>
<script>location.replace("{target}");</script></body></html>
"""


def assemble(development: Path, archives: Path, output: Path) -> list[dict[str, str]]:
    if not (development / "index.html").is_file():
        raise ValueError("development documentation has no index.html")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(development, output / "main")

    archive_files = sorted(archives.glob("mixle-docs-v*.tar.gz")) if archives.exists() else []
    digest_files = sorted(archives.glob("mixle-docs-v*.tar.gz.sha256")) if archives.exists() else []
    if {path.name + ".sha256" for path in archive_files} != {path.name for path in digest_files}:
        raise ValueError("documentation archives and digest files are not one-to-one")

    releases: list[tuple[str, dict]] = []
    for archive in archive_files:
        match = ARCHIVE.fullmatch(archive.name)
        if match is None:
            raise ValueError(f"invalid documentation archive name {archive.name}")
        version = match.group("version")
        metadata = _extract(archive, output / f"v{version}", version)
        releases.append((version, metadata))
    releases.sort(key=lambda item: _version_key(item[0]), reverse=True)

    switcher = [{"name": "main", "type": "BRANCH", "date": ""}]
    switcher.extend({"name": f"v{version}", "type": "TAG", "date": ""} for version, _ in releases)
    rendered = json.dumps(switcher, indent=2) + "\n"
    (output / "switcher.json").write_text(rendered, encoding="utf-8")
    (output / "versions.json").write_text(rendered, encoding="utf-8")
    (output / "index.html").write_text(_root_page(releases[0][0] if releases else None), encoding="utf-8")
    (output / ".nojekyll").touch()
    return switcher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.development, args.archives, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
