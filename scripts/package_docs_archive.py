#!/usr/bin/env python3
"""Create a deterministic, content-addressed release documentation archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

VERSION = re.compile(r"^\d+\.\d+\.\d+$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _tar_info(name: str, *, size: int, epoch: int, directory: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name + ("/" if directory and not name.endswith("/") else ""))
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.size = 0 if directory else size
    info.mtime = epoch
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if directory else 0o644
    return info


def package(source: Path, output: Path, *, version: str, commit: str, epoch: int) -> str:
    if not VERSION.fullmatch(version):
        raise ValueError(f"invalid release version {version!r}")
    if not COMMIT.fullmatch(commit):
        raise ValueError("commit must be a full lowercase 40-character SHA")
    if epoch < 0:
        raise ValueError("source-date-epoch must be non-negative")
    source = source.resolve()
    if not (source / "index.html").is_file():
        raise ValueError(f"documentation source has no index.html: {source}")
    expected_name = f"mixle-docs-v{version}.tar.gz"
    if output.name != expected_name:
        raise ValueError(f"archive must be named {expected_name}")

    entries = sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix())
    for path in entries:
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"unsupported documentation entry: {path}")

    metadata = (
        json.dumps(
            {
                "artifact": "mixle.release_docs/v1",
                "commit": commit,
                "source_date_epoch": epoch,
                "version": version,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                archive.addfile(_tar_info("docs-build.json", size=len(metadata), epoch=epoch), io.BytesIO(metadata))
                for path in entries:
                    name = path.relative_to(source).as_posix()
                    if path.is_dir():
                        archive.addfile(_tar_info(name, size=0, epoch=epoch, directory=True))
                    else:
                        info = _tar_info(name, size=path.stat().st_size, epoch=epoch)
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()
    print(
        package(
            args.source,
            args.output,
            version=args.version,
            commit=args.commit,
            epoch=args.source_date_epoch,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
