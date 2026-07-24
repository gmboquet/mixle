"""Repack an sdist ``.tar.gz`` so it is byte-for-byte reproducible across builds of the same commit.

``SOURCE_DATE_EPOCH`` alone is not enough for setuptools' ``sdist`` command: it does not apply the epoch
to each tar member's own ``mtime`` (only real wall-clock file mtimes land there), and this environment's
Python ``gzip`` module was observed to emit non-deterministic compressed bytes for byte-identical input --
even within a single process, across two calls with identical arguments -- while the system ``gzip``
binary did not. Wheel builds do not have either problem (verified separately; see
``release-checklists/0.8.0-decisions.md`` D- entry on reproducible builds), so only the sdist needs this.

This script: decompresses the sdist, rewrites every tar member's ``mtime``/``uid``/``gid``/``uname``/
``gname`` to a fixed epoch (0/0 for the ownership fields, since the *real* uid/gid of whoever ran the
build is never meaningful to preserve), and recompresses with the system ``gzip -n`` (``-n`` suppresses
gzip's own header-level name/mtime fields). Run after ``python -m build --sdist``, before ``twine check``
or ``upload-artifact``, on the exact ``dist/*.tar.gz`` that build produced.
"""

from __future__ import annotations

import gzip
import io
import subprocess
import sys
import tarfile
from pathlib import Path


def normalize(path: Path, epoch: int) -> None:
    """Rewrite the sdist at ``path`` in place so its bytes are a pure function of tree content + ``epoch``."""
    raw = gzip.decompress(path.read_bytes())
    out = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as src,
        tarfile.open(fileobj=out, mode="w:", format=tarfile.GNU_FORMAT) as dst,
    ):
        for member in src.getmembers():
            member.mtime = epoch
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            dst.addfile(member, src.extractfile(member) if member.isfile() else None)
    compressed = subprocess.run(["gzip", "-n", "-9"], input=out.getvalue(), capture_output=True, check=True).stdout
    path.write_bytes(compressed)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: normalize_sdist.py <path/to/sdist.tar.gz> <source_date_epoch>", file=sys.stderr)
        return 2
    normalize(Path(argv[0]), int(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
