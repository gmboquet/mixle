"""Commit authorship in the release range is this project's own, and it is checked, not assumed.

The 0.7.0 line shipped with forty tool co-author trailers and needed a history rewrite to remove
them, which changed every commit SHA from mid-2026. The release ledger's identity gate reads only
``%an``/``%cn``, so a trailer in the message body passed it untouched -- the property was verified by
eye each time and enforced nowhere. These tests pin the rules and run them over the real range.
"""

from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_TAG = "v0.8.1"


def _module():
    path = ROOT / "scripts" / "check_commit_attribution.py"
    spec = importlib.util.spec_from_file_location("_check_commit_attribution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _has_range() -> bool:
    """Whether this checkout can see the release range at all (an sdist tree cannot)."""
    try:
        subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", f"{BASE_TAG}^{{commit}}"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


class RuleTest(unittest.TestCase):
    """The rules separate a claim of authorship from ordinary subject matter."""

    def setUp(self):
        self.module = _module()

    def _markers(self, text: str) -> list[str]:
        return [why for pattern, why in self.module.MARKERS if pattern.search(text)]

    def test_tool_attribution_is_refused(self):
        for text in (
            "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
            "\U0001f916 Generated with [Claude Code](https://claude.com/claude-code)",
            "Generated with [Claude Code](https://claude.ai/code)",
            "Assisted-By: ChatGPT <someone@openai.com>",
            "written with GitHub Copilot",
        ):
            self.assertTrue(self._markers(text), text)

    def test_subject_matter_is_not_attribution(self):
        # Every one of these is a real line from this repository's history. A check that flags them
        # is a check that gets ignored, so they are pinned as explicitly allowed.
        for text in (
            "regenerated with zero drift -- no new top-level names, only methods on classes",
            'failed with "api_manifest.json is stale". Regenerated with',
            "surface (generated with optional backends installed); the test tolerates a base environment",
            "peft-wrapped hf-internal-testing/tiny-random-gpt2 checkpoint dropped into GradLeaf",
            "none. openai/clip-vit-base-patch32 ships only pytorch_model.bin at the pinned revision",
            "Requirements: REQ-AI-OPERABILITY, REQ-REPRODUCIBILITY, REQ-TRACEABILITY",
            "mistake what this was: six AI review agents verified each of D-0001 through D-0132",
            "Ten AI adversarial reviews of the notebooks and examples",
            "fix(reason/llm): tie-correct rank-sum AUC for factuality discrimination",
        ):
            self.assertEqual(self._markers(text), [], text)

    def test_roles_have_different_allowlists(self):
        allowed = self.module.ALLOWED_BY_ROLE
        github = "GitHub <noreply@github.com>"
        # GitHub's mailbox is the committer of anything merged through the web UI, which is ordinary;
        # the same mailbox authoring or co-authoring a commit is not.
        self.assertIn(github, allowed["committer"])
        self.assertNotIn(github, allowed["author"])
        self.assertNotIn(github, allowed["co-author"])
        for role in ("author", "committer", "co-author"):
            self.assertIn(self.module.OWNER, allowed[role])


@unittest.skipUnless(_has_range(), f"needs a git checkout with {BASE_TAG}")
class ReleaseRangeTest(unittest.TestCase):
    def test_the_release_range_claims_only_this_project_s_authorship(self):
        found = _module().violations(BASE_TAG, "HEAD")
        self.assertEqual(found, [], "\n".join(found))

    def test_the_check_actually_detects_a_known_violating_history(self):
        """A gate that cannot fail proves nothing; run it over history known to violate.

        The pre-scrub backup ref is kept locally and is not always present (a fresh clone has no
        ``refs/backup/*``), so this asserts only when it is there.
        """
        module = _module()
        backup = "refs/backup/release-080-preclean"
        probe = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", f"{backup}^{{commit}}"],
            capture_output=True,
        )
        if probe.returncode != 0:
            self.skipTest(f"{backup} is not in this checkout")
        found = module.violations("v0.7.0", backup)
        self.assertTrue(found, "the pre-scrub history must still trip the check")
        self.assertTrue(any("names Claude" in line for line in found))
        self.assertTrue(any("trailer" in line for line in found))


if __name__ == "__main__":
    unittest.main()
