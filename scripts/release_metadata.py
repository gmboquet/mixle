#!/usr/bin/env python
"""Record reproducibility metadata for a release artifact (worklist P2.5).

Given a built wheel or sdist, emit a JSON record with the filename, size, and full SHA-256, plus the
Python version and platform it was fingerprinted on -- the immutable fingerprint a release must retain
so the exact artifact can be verified byte for byte later (e.g. that the file on PyPI is the one that
passed the release gates). Usage::

    python scripts/release_metadata.py dist/mixle-0.8.0-py3-none-any.whl [--out artifact-metadata.json]

The resolved dependency set (``pip freeze`` of the clean install) is captured separately by the release
checklist; this script fingerprints the artifact file itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


def artifact_metadata(path: Path) -> dict:
    """Filename, size, and SHA-256 of a release artifact, plus the current Python/platform."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {
        "filename": path.name,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fingerprint a release artifact (worklist P2.5).")
    parser.add_argument("artifact", type=Path, help="path to a built wheel or sdist")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON record here (also printed)")
    args = parser.parse_args()
    if not args.artifact.is_file():
        print(f"not a file: {args.artifact}", file=sys.stderr)
        return 2
    text = json.dumps(artifact_metadata(args.artifact), indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
