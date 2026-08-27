"""Build hook that embeds immutable source provenance in every wheel."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def _git_value(root: Path, expression: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", expression],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = result.stdout.strip().lower()
    return value if result.returncode == 0 and len(value) == 40 else "unknown"


def _source_dirty(root: Path) -> bool | None:
    declared = os.environ.get("MIXLE_SOURCE_DIRTY", "").strip().lower()
    if declared in {"true", "false"}:
        return declared == "true"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "mixle", "pyproject.toml", "setup.py"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout) if result.returncode == 0 else None


def _source_content_files(root: Path) -> list[Path]:
    provenance = root / "mixle" / "_build_provenance.json"
    paths = [root / "pyproject.toml", root / "setup.py"]
    paths.extend(
        path for path in (root / "mixle").rglob("*") if path.is_file() and path.suffix in {".json", ".py", ".pyx"}
    )
    # The provenance file is excluded from its own digest. It has to be: the sdist digest is
    # computed and then written INTO this file, so including it would make the recorded value
    # unverifiable by construction.
    return sorted(path for path in paths if path.is_file() and path != provenance)


def _source_content_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _source_content_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _provenance_payload(root: Path) -> dict:
    """Build the provenance record for the tree being built from.

    ``source_content_sha256`` digests whatever the BUILD TREE contains, which is not the same set
    of files in every build: a checkout carries the full working tree, while a tree unpacked from
    an sdist carries only what the sdist ships. The two therefore produce different digests from
    the same algorithm, and the bare field name suggested a single canonical "source" digest that
    does not exist (SYS-08). The universe is now described alongside the value -- its file count
    and its selection rule -- so a reader can tell which population was hashed instead of assuming.
    """
    source_commit = os.environ.get("MIXLE_SOURCE_COMMIT", "").strip().lower()
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        source_commit = _git_value(root, "HEAD")
    source_tree = os.environ.get("MIXLE_SOURCE_TREE", "").strip().lower()
    if len(source_tree) != 40 or any(character not in "0123456789abcdef" for character in source_tree):
        source_tree = _git_value(root, "HEAD^{tree}")
    files = _source_content_files(root)
    # One key, two possible populations: a git checkout digests ~2,000 files, an unpacked sdist
    # digests the ~800 it ships. The digests legitimately differ, and a reader diffing the wheel's
    # record against the sdist's saw a "mismatch" that was only a population difference (campaign
    # T3-05). The universe string now names WHICH population was hashed instead of describing both
    # identically, so the records disambiguate themselves.
    population = "the git checkout" if (root / ".git").exists() else "the sdist-packaged tree"
    return {
        "artifact": "mixle.build_provenance/v1",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_dirty": _source_dirty(root),
        "source_content_sha256": _source_content_digest(root),
        "source_content_file_count": len(files),
        "source_content_universe": "pyproject.toml, setup.py, and mixle/**/*.{json,py,pyx} present in %s" % population,
    }


def _carried_provenance(root: Path) -> dict | None:
    """Return a shipped provenance record ONLY if it still describes this tree's contents.

    An sdist carries no ``.git``, so a wheel built from an unpacked sdist could not name the commit
    it came from and recorded ``unknown`` (SYS-01). The sdist ships its attestation so the identity
    survives the sdist -> wheel hop.

    Carrying it forward unconditionally was worse than the bug it fixed: anyone could unpack the
    sdist, edit the code, rebuild, and the wheel would assert the ORIGINAL commit over modified
    bytes -- a label that lies rather than a label that is missing. So the record is only honoured
    when ``sdist_content_sha256`` still matches a digest recomputed over the tree right now. Any
    edit to a shipped ``.py``/``.json``/``.pyx``, ``pyproject.toml`` or ``setup.py`` breaks that
    match and the identity is dropped back to ``unknown``, which is the honest answer for bytes
    nobody can vouch for.

    The digest compared here is over the SDIST's own file population, not the checkout's. The
    checkout digest describes ~2,000 files and an unpacked sdist ships far fewer, so the original
    ``source_content_sha256`` can never be recomputed from an sdist tree -- which is exactly why
    the first version of this function skipped verification instead of doing it against the wrong
    population.
    """
    shipped = root / "mixle" / "_build_provenance.json"
    try:
        payload = json.loads(shipped.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("artifact") != "mixle.build_provenance/v1":
        return None
    commit = payload.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        return None
    recorded = payload.get("sdist_content_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        return None  # pre-verification record, or none written: not adoptable
    if recorded != _source_content_digest(root):
        return None  # the tree was modified after it was packed; the record no longer describes it
    return payload


class ProvenanceBuildPy(build_py):
    """Write build provenance into build_lib without mutating the source checkout."""

    def run(self) -> None:
        super().run()
        root = Path(__file__).resolve().parent
        payload = _provenance_payload(root)
        if payload["source_commit"] != "unknown":
            # The release path: env vars named the commit, so the fresh payload wins -- but a wheel
            # built from an unpacked sdist then DROPPED the sdist digest, leaving no cross-artifact
            # check between the two records (campaign T3-05: sdist_content_sha256 is the one field
            # whose value the sdist and its wheels share). When the shipped record still verifies
            # against this exact tree, carry the sdist digest fields forward alongside the fresh
            # identity; a tree that no longer matches gets nothing, same rule as adoption below.
            shipped = _carried_provenance(root)
            if shipped is not None:
                for key in ("sdist_content_sha256", "sdist_content_file_count", "sdist_content_universe"):
                    if key in shipped:
                        payload[key] = shipped[key]
        if payload["source_commit"] == "unknown":
            # No git here. Either this is a tree unpacked from an sdist -- in which case the sdist
            # shipped the attestation from when it was built, and carrying it forward is what keeps
            # the sdist-installed wheel bound to the candidate (SYS-01) -- or there is genuinely no
            # provenance to state, and "unknown" stands. Only a well-formed record is adopted, and
            # adoption is recorded so a reader can tell a carried identity from a freshly derived
            # one rather than having to infer it.
            carried = _carried_provenance(root)
            if carried is not None:
                payload = dict(carried)
                payload["source_content_carried_through_sdist"] = True
        destination = Path(self.build_lib) / "mixle" / "_build_provenance.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


class ProvenanceSdist(sdist):
    """Ship the build provenance inside the sdist so it survives the sdist -> wheel hop."""

    def make_release_tree(self, base_dir, files):  # noqa: ANN001, ANN201 - distutils signature
        super().make_release_tree(base_dir, files)
        root = Path(__file__).resolve().parent
        destination = Path(base_dir) / "mixle" / "_build_provenance.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        # make_release_tree HARD LINKS files from the checkout when it can, so writing through an
        # existing path here would edit the developer's working tree. Unlink first: this command
        # must leave the checkout byte-identical, which is the same rule ProvenanceBuildPy follows.
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        payload = _provenance_payload(root)
        # Digest of what the SDIST actually ships, taken over the release tree rather than the
        # checkout. A wheel built from this sdist recomputes it and refuses to carry the identity
        # forward if it no longer matches, which is what stops an edited unpacked sdist from
        # producing a wheel that asserts this commit over different bytes. The provenance file
        # itself is excluded from the digest -- it is where the digest is about to be written.
        payload["sdist_content_sha256"] = _source_content_digest(Path(base_dir))
        payload["sdist_content_file_count"] = len(_source_content_files(Path(base_dir)))
        payload["sdist_content_universe"] = (
            "pyproject.toml, setup.py, and mixle/**/*.{json,py,pyx} shipped in this sdist"
        )
        destination.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


setup(cmdclass={"build_py": ProvenanceBuildPy, "sdist": ProvenanceSdist})
