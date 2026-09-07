#!/usr/bin/env python3
"""Produce the manual TestPyPI rehearsal record for a candidate whose version is already on TestPyPI.

Runs, from the draft release's own assets, everything publish.yml's ``testpypi`` phase would have
run after its upload: verifies the wheel and sdist against the candidate's SHA256SUMS, installs the
wheel into a fresh virtual environment, sweeps every public module, confirms the reproducible-build
record, and records what TestPyPI currently serves for the version (which must be an EARLIER
candidate's bytes -- the one situation this record is for; see ``verify_rehearsal_record.py``).
The record is then uploaded to the draft release (``--upload``), where the promote phase fetches
it by name and re-verifies it.

    python scripts/record_manual_rehearsal.py --tag v0.8.1 --prepare-run <prepare-run-id> --upload
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from verify_rehearsal_record import MANUAL, MANUAL_REASON, fetch_testpypi_files, parse_sums  # noqa: E402


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prepare-run", type=int, required=True)
    parser.add_argument("--repository", default="gmboquet/mixle")
    parser.add_argument("--out", type=Path, help="default: testpypi-rehearsal-manual-<sha8>.json")
    parser.add_argument("--upload", action="store_true", help="upload the record to the draft release")
    args = parser.parse_args(argv)

    release = json.loads(_gh("release", "view", args.tag, "--repo", args.repository, "--json", "isDraft,assets"))
    if not release["isDraft"]:
        print("the release is not a draft; the manual rehearsal is for an unpublished candidate", file=sys.stderr)
        return 1
    owner = _gh("api", "user", "--jq", ".login").strip()

    work = Path(tempfile.mkdtemp(prefix="manual-rehearsal-"))
    try:
        assets = work / "assets"
        assets.mkdir()
        for pattern in (
            "mixle-*.whl",
            "mixle-*.tar.gz",
            "SHA256SUMS",
            "release-candidate.json",
            "reproducible-builds.json",
        ):
            _gh("release", "download", args.tag, "--repo", args.repository, "--dir", str(assets), "--pattern", pattern)
        for docs in assets.glob("mixle-docs-*"):
            docs.unlink()  # the docs archive is a release asset but not a distribution
        sums_text = (assets / "SHA256SUMS").read_text(encoding="utf-8")
        sums = parse_sums(sums_text)
        candidate = json.loads((assets / "release-candidate.json").read_text(encoding="utf-8"))
        reproducible = json.loads((assets / "reproducible-builds.json").read_text(encoding="utf-8"))
        if candidate.get("workflow_run") != args.prepare_run:
            print(
                f"draft release was prepared by run {candidate.get('workflow_run')}, not {args.prepare_run}",
                file=sys.stderr,
            )
            return 1
        version = re.match(r"^v(.+)$", args.tag).group(1)

        draft_files = {name: _sha256(assets / name) for name in sums}
        if draft_files != sums:
            print("draft release assets do not equal SHA256SUMS", file=sys.stderr)
            return 1

        env = work / "env"
        venv.create(env, with_pip=True, clear=True)
        python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        wheel = next(assets.glob("mixle-*.whl"))
        subprocess.run([str(python), "-m", "pip", "install", "-q", str(wheel)], check=True)
        sweep = subprocess.run(
            [str(python), str(ROOT / "scripts" / "import_sweep.py")],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(work),
        )
        swept = re.search(r"swept (\d+) public modules", sweep.stdout)
        if not swept or "import cleanly" not in sweep.stdout:
            print("import sweep did not report a clean sweep:\n" + sweep.stdout[-800:], file=sys.stderr)
            return 1

        testpypi_files = fetch_testpypi_files(version)
        if not testpypi_files:
            print(f"TestPyPI does not hold {version}; run the automated testpypi phase instead", file=sys.stderr)
            return 1
        if all(testpypi_files.get(name) == digest for name, digest in sums.items()):
            print(
                "TestPyPI already holds this candidate's exact bytes; the automated testpypi phase applies",
                file=sys.stderr,
            )
            return 1

        record = {
            "artifact": MANUAL,
            "reason": MANUAL_REASON,
            "candidate_commit": candidate["commit"],
            "tag": args.tag,
            "prepare_run": args.prepare_run,
            "sums_sha256": hashlib.sha256(sums_text.encode("utf-8")).hexdigest(),
            "draft_release_files": draft_files,
            "testpypi_files": testpypi_files,
            "reproducible_builds_passed": reproducible.get("passed") is True,
            "clean_install": True,
            "import_sweep_modules": int(swept.group(1)),
            "verified_by": owner,
            "verified_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "passed": True,
        }
        out = args.out or ROOT / f"testpypi-rehearsal-manual-{candidate['commit'][:8]}.json"
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"wrote {out} ({record['import_sweep_modules']} modules swept; TestPyPI holds an earlier candidate's {version})"
        )
        if args.upload:
            _gh("release", "upload", args.tag, str(out), "--repo", args.repository, "--clobber")
            print(f"uploaded {out.name} to the draft release {args.tag}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
