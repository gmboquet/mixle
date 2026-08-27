"""Verify a release-candidate handout's human-authored text against its own artifacts.

The tester brief states the candidate's identity -- source commit, source tree, and both artifact
digests -- and testers are told to treat any mismatch as blocking. Those four values were
hand-transcribed twice and were wrong twice: once carrying the PREVIOUS candidate's tree, once
carrying a 40-hex tree hash whose tail was invented from a printed 8-character prefix. Both were
caught by testers, which is the expensive way to catch them.

Nothing here should be typed by a human. This recomputes every identity value from the artifacts
themselves and fails if the brief, the attestation, or the checksum file disagrees with them.

Usage:
    python scripts/verify_candidate_handout.py <candidate-directory>
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

PROVENANCE_MEMBER = "mixle/_build_provenance.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_provenance(wheel: Path) -> dict:
    with zipfile.ZipFile(wheel) as archive:
        return json.loads(archive.read(PROVENANCE_MEMBER))


def _sdist_provenance(sdist: Path) -> dict:
    with tarfile.open(sdist) as archive:
        member = next(n for n in archive.getnames() if n.endswith(PROVENANCE_MEMBER))
        extracted = archive.extractfile(member)
        if extracted is None:  # pragma: no cover - a directory entry cannot match the suffix
            raise ValueError("sdist provenance member is not a regular file")
        return json.loads(extracted.read())


def verify(candidate: Path) -> list[str]:
    """Return the disagreements between the handout's stated identity and its artifacts."""
    problems: list[str] = []
    dist = candidate / "dist"
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        return [f"expected exactly one wheel and one sdist in {dist}, found {len(wheels)} and {len(sdists)}"]
    wheel, sdist = wheels[0], sdists[0]

    provenance = _wheel_provenance(wheel)
    sdist_provenance = _sdist_provenance(sdist)
    truth = {
        "source commit": provenance["source_commit"],
        "source tree": provenance["source_tree"],
        "wheel sha256": _sha256(wheel),
        "sdist sha256": _sha256(sdist),
    }

    # The two records must agree on the identity they share; source_content_sha256 legitimately
    # differs between them (different file populations, each named by its own universe field).
    for field in ("source_commit", "source_tree", "source_dirty"):
        if provenance.get(field) != sdist_provenance.get(field):
            problems.append(
                f"wheel and sdist provenance disagree on {field}: "
                f"{provenance.get(field)!r} vs {sdist_provenance.get(field)!r}"
            )
    if provenance.get("source_dirty"):
        problems.append("the artifacts were built from a dirty source tree")

    brief = candidate / "TESTER-BRIEF.md"
    if not brief.exists():
        problems.append(f"{brief.name} is missing")
    else:
        text = brief.read_text(encoding="utf-8")
        for label, value in truth.items():
            if value not in text:
                problems.append(f"{brief.name} does not state the true {label} {value}")
        # A hash in the identity block that names nothing real is the failure mode that shipped
        # twice: a stale value from a previous candidate, or a real prefix with an invented tail.
        head = text.split("## Role A")[0]
        for stray in re.findall(r"\b[0-9a-f]{40,64}\b", head):
            if stray not in truth.values():
                problems.append(f"{brief.name} identity block names {stray}, which is not this candidate")

    sums = candidate / "SHA256SUMS"
    if not sums.exists():
        problems.append("SHA256SUMS is missing")
    else:
        recorded = dict(
            (line.split()[1].lstrip("*"), line.split()[0]) for line in sums.read_text().splitlines() if line.strip()
        )
        for path, digest in ((wheel, truth["wheel sha256"]), (sdist, truth["sdist sha256"])):
            if recorded.get(path.name) != digest:
                problems.append(f"SHA256SUMS records {recorded.get(path.name)} for {path.name}, artifact is {digest}")

    attestation = candidate / "release-candidate-attestation.json"
    if not attestation.exists():
        problems.append("release-candidate-attestation.json is missing")
    else:
        record = json.loads(attestation.read_text())
        for key, value in (
            ("candidate_commit", truth["source commit"]),
            ("candidate_tree", truth["source tree"]),
            ("wheel_sha256", truth["wheel sha256"]),
            ("sdist_sha256", truth["sdist sha256"]),
        ):
            if record.get(key) != value:
                problems.append(f"attestation {key} is {record.get(key)}, artifacts say {value}")

    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    candidate = Path(argv[1]).resolve()
    problems = verify(candidate)
    if problems:
        print(f"candidate handout {candidate.name} disagrees with its own artifacts:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"candidate handout {candidate.name}: identity consistent across brief, sums, attestation, and artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
