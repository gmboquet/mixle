"""Verify a release-candidate handout's human-authored text against its own artifacts.

The tester brief states the candidate's identity -- source commit, source tree, and both artifact
digests -- and testers are told to treat any mismatch as blocking. Those four values were
hand-transcribed and wrong four times: once carrying the PREVIOUS candidate's tree, once carrying
a 40-hex tree hash whose tail was invented from a printed 8-character prefix, once -- found only
after the first two fixes -- as the same stale commit sitting unnoticed in the Role B example
commands (a git checkout target, a --source-digest flag), well past the identity block the first
fix checked, and a fourth time -- found by an independent replayer, not by hand -- as a stale
8-character ABBREVIATED commit left inside Role B's own `git checkout --detach <sha>` command
block by a brief-generation script that substituted the full 40-hex commit everywhere but never
carried the same substitution to the short form it had itself written into that one line. The
first two were caught by testers; the third was caught by hand; the fourth was caught by a Role B
replayer who followed the brief literally and noticed the checked-out commit didn't match the
identity block above it.

Nothing here should be typed by a human. This recomputes every identity value from the artifacts
themselves and fails if the brief -- ANYWHERE in it, not only its identity block -- the attestation,
or the checksum file disagrees with them.

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
        # A full-length hash naming nothing real is the failure mode that has shipped THREE times:
        # a stale value from a previous candidate in the identity block, a real prefix with an
        # invented tail, and -- found only by hand, after the first two fixes -- the SAME stale
        # commit sitting unnoticed in the Role B example commands (a git checkout target, a
        # --source-digest flag), well past the identity block those first two fixes checked. So
        # this scans the WHOLE document: every full commit (40 hex) or digest (64 hex) anywhere
        # must be either a true value or absent, full stop. A short prefix (e.g. an 8-char
        # abbreviation naming a superseded candidate in prose, "## What changed since dcec5e29")
        # is below this length floor and is not flagged here -- narrative mentions of history are
        # fine. The fourth defect variant proved a short prefix CAN be wrong in a way that matters,
        # when it sits inside a fenced code block instead of prose; that case is handled by the
        # separate, narrower scan below, scoped to code fences only.
        for stray in re.findall(r"\b[0-9a-f]{40,64}\b", text):
            if stray not in truth.values():
                problems.append(f"{brief.name} names {stray}, which is not this candidate")
        # A short (6-8 hex char) abbreviation is legitimate and NOT flagged above -- but only in
        # prose ("## What changed since dcec5e29"), where a reader can tell it names history, not
        # an instruction. Inside a fenced code block it is copy-pasted and run verbatim (this is
        # exactly the fourth defect variant this docstring describes), so within code fences ONLY,
        # a short hex token must be a genuine prefix of this candidate's own commit or tree -- any
        # other short hex token there names a stale or invented value just as surely as a full one.
        short_prefixes = {truth["source commit"][:n] for n in range(6, 9)} | {
            truth["source tree"][:n] for n in range(6, 9)
        }
        for block in re.findall(r"```.*?```", text, flags=re.DOTALL):
            for stray in re.findall(r"\b[0-9a-f]{6,8}\b", block):
                if stray not in short_prefixes:
                    problems.append(
                        f"{brief.name} runs a command naming {stray}, which is not a prefix of this candidate's identity"
                    )

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
