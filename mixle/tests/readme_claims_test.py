"""Guard the README against retired overclaims (worklist D8.2 / N9.2).

0.8.0 is a credibility release: public claims are qualified and evidenced, not universal absolutes.
This gate fails if a retired overclaim reappears in the README, so the honest wording cannot silently
regress.
"""

import unittest
from pathlib import Path

_README = Path(__file__).resolve().parents[2] / "README.md"

# Retired because they overclaim: "any engine" / "any backend" / "no rewrite" assert universal,
# unverified portability across engines/backends (several distributed backends are not CI-exercised);
# "safe to put in front of users" is an unqualified safety claim (a safety claim needs E5 evidence,
# not expected in 0.8.0). Keep this list in sync with the claim-evidence discipline in the worklist.
_BANNED = (
    "any engine",
    "any backend",
    "no rewrite",
    "safe to put in front of users",
)


class ReadmeClaimsTest(unittest.TestCase):
    def test_readme_exists(self):
        self.assertTrue(_README.exists(), f"README not found at {_README}")

    def test_no_retired_overclaims(self):
        text = _README.read_text(encoding="utf-8").lower()
        found = [phrase for phrase in _BANNED if phrase in text]
        self.assertFalse(found, f"README contains retired overclaim(s) — qualify or remove: {found}")


if __name__ == "__main__":
    unittest.main()
