"""Containment for caller-supplied path segments under a declared root.

A deployment or registry name that comes from a caller is joined onto a root directory, and
``os.path.join`` / ``pathlib`` are both happy to leave that root: ``root / "tasks" / "../../escaped"``
traverses out of it, and an ABSOLUTE segment discards the root entirely
(``Path("registry") / "/tmp/x"`` is ``/tmp/x``). Nothing about that is exotic -- a name arrives from an
API request, a config file, or a CLI argument -- and the consequence is a write outside the artifact
root the caller believes they scoped it to (MXR-080-1910).

:func:`safe_segment` refuses a segment that is not a single path component. :func:`contained_path`
additionally resolves the result and re-checks it against the resolved root, which is what catches a
pre-existing symlink inside the root pointing elsewhere -- a check on the string alone cannot see
that. ``mixle.inference.production.registry`` has enforced the same two rules on its own store since
MXR-080-0264; this is that logic made shared rather than re-derived.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["contained_path", "safe_segment"]


def safe_segment(segment: str, *, kind: str = "name") -> str:
    """Return ``segment`` unchanged, or raise unless it is a single safe path component."""
    if not isinstance(segment, str) or not segment.strip():
        raise ValueError(f"{kind} must be a non-empty string, got {segment!r}")
    if (
        segment in (os.curdir, os.pardir)
        or os.sep in segment
        or (os.altsep and os.altsep in segment)
        or "\x00" in segment
        or os.path.isabs(segment)
        or os.path.basename(segment) != segment
    ):
        raise ValueError(
            f"unsafe {kind} {segment!r}: must be a single path component -- no separators, no '..', "
            "and not absolute. Joining it onto a root would write outside that root."
        )
    return segment


def contained_path(root: str | os.PathLike[str], *segments: str, kind: str = "name") -> Path:
    """Join ``segments`` under ``root``, refusing anything that escapes it.

    Each segment is checked as a single component, and the resolved result is then re-checked against
    the resolved root so a symlink already inside the root cannot redirect the write outside it.
    """
    base = Path(root)
    for segment in segments:
        safe_segment(segment, kind=kind)
    candidate = base.joinpath(*segments)
    root_real = os.path.realpath(base)
    candidate_real = os.path.realpath(candidate)
    if candidate_real != root_real and not candidate_real.startswith(root_real + os.sep):
        raise ValueError(
            f"unsafe {kind}: {'/'.join(segments)!r} resolves to {candidate_real!r}, which is outside "
            f"the declared root {root_real!r}. A symlink inside the root can redirect a write that "
            "looks contained by its string alone."
        )
    return candidate
