"""Regression coverage for scripts/verify_candidate_handout.py.

This tool exists because the same hand-transcription defect shipped three times in one release:
a stale tree from the previous candidate, a real hash prefix with an invented tail, and -- found
only after both of those were fixed -- the same stale commit sitting unnoticed in the Role B
example commands, past the identity block the first fix checked. Each of these three cases gets
its own test, built by actually corrupting a real, freshly-built candidate directory rather than
hand-authoring fixtures -- the corruption a human introduces looks like real prose, not like a
crafted test string, and only a real wheel/sdist pair exercises the actual provenance-reading code.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tarfile
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_candidate_handout.py"
_SPEC = importlib.util.spec_from_file_location("verify_candidate_handout", _MODULE_PATH)
verify_candidate_handout = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verify_candidate_handout
_SPEC.loader.exec_module(verify_candidate_handout)


def _make_candidate(root: Path) -> tuple[str, str, str, str]:
    """Build a minimal but structurally real candidate directory; return (commit, tree, whl_sha, sdist_sha)."""
    commit = "a" * 40
    tree = "b" * 40
    provenance = (
        '{"artifact":"mixle.build_provenance/v1","source_commit":"%s","source_tree":"%s",'
        '"source_dirty":false,"source_content_sha256":"c" * 64,"source_content_file_count":1,'
        '"source_content_universe":"test"}' % (commit, tree)
    ).replace('"c" * 64', '"' + "c" * 64 + '"')

    dist = root / "dist"
    dist.mkdir(parents=True)

    whl = dist / "mixle-0.8.0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as archive:
        archive.writestr("mixle/_build_provenance.json", provenance)

    sdist = dist / "mixle-0.8.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = provenance.encode("utf-8")
        info = tarfile.TarInfo("mixle-0.8.0/mixle/_build_provenance.json")
        info.size = len(payload)
        import io

        archive.addfile(info, io.BytesIO(payload))

    whl_sha = verify_candidate_handout._sha256(whl)
    sdist_sha = verify_candidate_handout._sha256(sdist)

    (root / "SHA256SUMS").write_text(f"{whl_sha}  {whl.name}\n{sdist_sha}  {sdist.name}\n")

    brief = (
        "# mixle candidate\n\n"
        f"- Source commit: `{commit}` (branch release/0.8.0, exact tip)\n"
        f"- Source tree: `{tree}`\n"
        f"- `{whl.name}` — SHA-256 `{whl_sha}`\n"
        f"- `{sdist.name}` — SHA-256 `{sdist_sha}`\n\n"
        "## Role A — independent release tester\n\n"
        "## Role B — independent reproduction replay\n\n"
        "```\n"
        f"git clone git@github.com:gmboquet/mixle.git src && cd src && git checkout --detach {commit}\n"
        "```\n\n"
        "```\n"
        f"  --source-digest {commit} \\\n"
        "```\n"
    )
    (root / "TESTER-BRIEF.md").write_text(brief)

    (root / "release-candidate-attestation.json").write_text(
        '{"artifact":"mixle.release_candidate_attestation/v1","candidate_commit":"%s",'
        '"candidate_tree":"%s","version":"0.8.0","wheel_sha256":"%s","sdist_sha256":"%s",'
        '"owner_declaration":"PENDING","declared_by":null,"declared_at":null}' % (commit, tree, whl_sha, sdist_sha)
    )
    return commit, tree, whl_sha, sdist_sha


class VerifyCandidateHandoutTest(unittest.TestCase):
    def test_a_freshly_built_consistent_candidate_verifies_clean(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            _make_candidate(root)
            self.assertEqual(verify_candidate_handout.verify(root), [])

    def test_a_previous_candidates_tree_in_the_identity_block_is_caught(self):
        """The first defect that shipped: a stale value from the PREVIOUS candidate."""
        with TemporaryDirectory() as d:
            root = Path(d)
            commit, tree, whl_sha, sdist_sha = _make_candidate(root)
            brief = root / "TESTER-BRIEF.md"
            stale_tree = "d" * 40
            brief.write_text(brief.read_text().replace(f"`{tree}`", f"`{stale_tree}`", 1))
            problems = verify_candidate_handout.verify(root)
            self.assertTrue(any(tree in p for p in problems), problems)
            self.assertTrue(any(stale_tree in p for p in problems), problems)

    def test_a_hash_with_an_invented_tail_is_caught(self):
        """The second defect: a real prefix, a fabricated tail -- names nothing real."""
        with TemporaryDirectory() as d:
            root = Path(d)
            commit, tree, whl_sha, sdist_sha = _make_candidate(root)
            brief = root / "TESTER-BRIEF.md"
            invented = tree[:8] + "9" * 32
            brief.write_text(brief.read_text().replace(f"`{tree}`", f"`{invented}`", 1))
            problems = verify_candidate_handout.verify(root)
            self.assertTrue(any(invented in p for p in problems), problems)

    def test_a_stale_commit_buried_in_the_role_b_example_commands_is_caught(self):
        """The third defect, found only by hand after the first two fixes: past the identity
        block, in the functional command examples a replayer would actually run."""
        with TemporaryDirectory() as d:
            root = Path(d)
            commit, tree, whl_sha, sdist_sha = _make_candidate(root)
            brief = root / "TESTER-BRIEF.md"
            stale_commit = "e" * 40
            text = brief.read_text()
            text = text.replace(f"git checkout --detach {commit}", f"git checkout --detach {stale_commit}", 1)
            brief.write_text(text)
            problems = verify_candidate_handout.verify(root)
            self.assertTrue(any(stale_commit in p for p in problems), problems)

    def test_a_short_prefix_naming_a_superseded_candidate_in_prose_is_not_flagged(self):
        """A narrative mention like '## What changed since dcec5e29' (an 8-char abbreviation),
        OUTSIDE any fenced code block, is never actually run and must not false-positive."""
        with TemporaryDirectory() as d:
            root = Path(d)
            _make_candidate(root)
            brief = root / "TESTER-BRIEF.md"
            brief.write_text(brief.read_text() + "\n## What changed since dcec5e29\n\nSome prose.\n")
            self.assertEqual(verify_candidate_handout.verify(root), [])

    def test_a_stale_short_hash_inside_a_role_b_command_is_caught(self):
        """The fourth defect, found by an independent replayer, not by hand: a brief-generation
        script substituted the full 40-hex commit everywhere but never carried the same
        substitution to an ABBREVIATED (8-char) form it had itself written into a previous
        candidate's Role B checkout command -- a short prefix is legitimate in prose (see the test
        above) but not inside a fenced code block a tester is told to run verbatim."""
        with TemporaryDirectory() as d:
            root = Path(d)
            commit, tree, whl_sha, sdist_sha = _make_candidate(root)
            brief = root / "TESTER-BRIEF.md"
            stale_short = "d" * 8
            text = brief.read_text()
            text = text.replace(f"git checkout --detach {commit}", f"git checkout --detach {stale_short}", 1)
            brief.write_text(text)
            problems = verify_candidate_handout.verify(root)
            self.assertTrue(any(stale_short in p for p in problems), problems)

    def test_a_correct_short_prefix_inside_a_role_b_command_is_not_flagged(self):
        """The candidate's OWN short prefix, if a brief legitimately uses one inside a command
        (rather than the full hash this tool otherwise prefers), must not false-positive."""
        with TemporaryDirectory() as d:
            root = Path(d)
            commit, tree, whl_sha, sdist_sha = _make_candidate(root)
            brief = root / "TESTER-BRIEF.md"
            text = brief.read_text()
            text = text.replace(f"git checkout --detach {commit}", f"git checkout --detach {commit[:8]}", 1)
            brief.write_text(text)
            self.assertEqual(verify_candidate_handout.verify(root), [])

    def test_sha256sums_disagreeing_with_the_artifact_is_caught(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            _make_candidate(root)
            sums = root / "SHA256SUMS"
            lines = sums.read_text().splitlines()
            lines[0] = "f" * 64 + "  " + lines[0].split()[1]
            sums.write_text("\n".join(lines) + "\n")
            problems = verify_candidate_handout.verify(root)
            self.assertTrue(any("SHA256SUMS" in p for p in problems), problems)

    def test_attestation_disagreeing_with_the_artifact_is_caught(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            commit, tree, whl_sha, sdist_sha = _make_candidate(root)
            attestation = root / "release-candidate-attestation.json"
            attestation.write_text(attestation.read_text().replace(tree, "f" * 40, 1))
            problems = verify_candidate_handout.verify(root)
            self.assertTrue(any("attestation" in p for p in problems), problems)

    def test_missing_files_are_named_not_crashed_on(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            _make_candidate(root)
            (root / "TESTER-BRIEF.md").unlink()
            (root / "SHA256SUMS").unlink()
            (root / "release-candidate-attestation.json").unlink()
            problems = verify_candidate_handout.verify(root)
            self.assertTrue(any("TESTER-BRIEF.md is missing" in p for p in problems), problems)
            self.assertTrue(any("SHA256SUMS is missing" in p for p in problems), problems)
            self.assertTrue(any("attestation.json is missing" in p for p in problems), problems)

    def test_cli_exit_codes_and_message(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            _make_candidate(root)
            clean = subprocess.run([sys.executable, str(_MODULE_PATH), str(root)], capture_output=True, text=True)
            self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
            self.assertIn("identity consistent", clean.stdout)

            shutil.rmtree(root / "dist")
            broken = subprocess.run([sys.executable, str(_MODULE_PATH), str(root)], capture_output=True, text=True)
            self.assertEqual(broken.returncode, 1)


if __name__ == "__main__":
    unittest.main()
