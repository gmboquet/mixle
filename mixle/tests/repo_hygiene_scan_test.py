"""Repository hygiene scan: no secrets or machine-specific artifacts in tracked source (worklist S13.3).

A release artifact should carry no developer machine's fingerprint and no credentials. Two failure modes
have actually occurred in this tree and are cheap to keep out for good:

  * **machine-specific absolute paths** -- e.g. a ``pip freeze`` line pinning ``mixle @ file:///Users/<name>/...``
    leaks a home directory, a username, and a local worktree layout into a tracked release checklist;
  * **credential-shaped strings** -- AWS keys, OpenAI keys, GitHub/Slack tokens, private-key PEM headers.

This scans the *tracked* file set (``git grep``) on every run, so a future commit that reintroduces either
fails here rather than in a downstream leak. The redaction feature's deliberately-fake test fixtures
(``sk-abcdefghij...``, ``AKIA...``) are the only credential-shaped strings the repo needs, so they are
allowlisted explicitly; anything else that matches is a finding.
"""

import re
import subprocess
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# This test file names the fixtures to allowlist them; it must never flag itself.
_SCAN_SELF = "repo_hygiene_scan_test.py"


def _git_grep_lines(ere_pattern):
    """Yield (path, lineno, text) for tracked lines matching an ERE ``ere_pattern``; skip if not a checkout.

    Uses ``-E`` (POSIX ERE, universally available) rather than ``-P`` (PCRE, build-dependent), and ``-I`` to
    skip binary files. Capturing groups in the pattern are fine -- precise extraction is done Python-side.
    """
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout (installed wheel) -- repo hygiene scan is a source-tree gate")
    proc = subprocess.run(
        ["git", "grep", "-nI", "-E", "-e", ere_pattern],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):  # 0 = matches, 1 = no matches; anything else is a real error
        raise RuntimeError(f"git grep failed ({proc.returncode}): {proc.stderr.strip()}")
    for line in proc.stdout.splitlines():
        path, lineno, text = line.split(":", 2)
        if Path(path).name != _SCAN_SELF:
            yield path, lineno, text


# Machine-specific home paths. `/home/runner` is GitHub Actions; the rest are generic doc placeholders.
_ALLOWED_PATH_USERS = {"runner", "user", "username", "you", "youruser", "me"}
_HOME_PATH = re.compile(r"/(?:Users|home)/([A-Za-z0-9_.-]+)")

# Deliberately-fake fixtures the secret-redaction feature is tested against (examples/ + tests/).
_ALLOWED_SECRETS = {"sk-abcdefghij1234567890XYZ", "AKIA1234567890ABCDEF"}
# Credential shapes, kept conservative so the gate stays low-noise. ERE for git grep; Python re extracts
# the full match via group(0), so the capturing groups here do not disturb allowlisting.
_SECRET_ERE = (
    r"AKIA[0-9A-Z]{16}"  # AWS access key id
    r"|sk-[A-Za-z0-9]{20,}"  # OpenAI-style secret key
    r"|ghp_[A-Za-z0-9]{36}"  # GitHub personal access token
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"  # Slack token
    r"|-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"  # private key PEM header
)
_SECRET_RE = re.compile(_SECRET_ERE)


class RepoHygieneScanTest(unittest.TestCase):
    def test_no_machine_specific_absolute_paths(self):
        findings = []
        for path, lineno, text in _git_grep_lines(r"/(Users|home)/[A-Za-z0-9_.-]+"):
            for m in _HOME_PATH.finditer(text):
                if m.group(1) not in _ALLOWED_PATH_USERS:
                    findings.append(f"{path}:{lineno}: machine-specific path for user '{m.group(1)}'")
        self.assertEqual(
            findings,
            [],
            "tracked files contain machine-specific home paths (leaks username / local layout):\n"
            + "\n".join(findings),
        )

    def test_no_unallowlisted_credentials(self):
        findings = []
        for path, lineno, text in _git_grep_lines(_SECRET_ERE):
            for m in _SECRET_RE.finditer(text):
                token = m.group(0)
                if token not in _ALLOWED_SECRETS:
                    findings.append(f"{path}:{lineno}: credential-shaped string not on the allowlist")
        self.assertEqual(
            findings,
            [],
            "tracked files contain credential-shaped strings outside the fixture allowlist:\n" + "\n".join(findings),
        )


if __name__ == "__main__":
    unittest.main()
