"""The glossary defines the terminology whose consistency 0.8.0 depends on (worklist X12.6).

X12.6 calls out term pairs that must be used consistently across README, API docs, and examples --
fit/train/optimize, model/distribution/estimator, engine/backend, artifact/checkpoint,
calibration/confidence, exact/certified/approximate, stable/experimental. Consistent usage starts with a
single authoritative definition, so this pins that the glossary actually *defines* the distinguishing terms.
A term added to a pair (a new backend name, a new maturity tier) without a glossary entry fails here.
"""

import re
import unittest
from pathlib import Path

_GLOSSARY = Path(__file__).resolve().parents[2] / "docs" / "glossary.rst"

# The distinguishing terms of X12.6's pairs that must each have an authoritative glossary definition.
_REQUIRED_TERMS = {
    "distribution",
    "estimator",
    "engine",
    "backend",
    "calibration",
    "certified",  # exact / certified / approximate
    "checkpoint",  # artifact / checkpoint
    "experimental",  # stable / experimental
    "train",  # fit / train / optimize
    "composite",
    "record",
    "task model",
}


def _defined_terms():
    """Glossary entries: a non-indented, non-blank line immediately followed by an indented definition."""
    lines = _GLOSSARY.read_text().splitlines()
    terms = set()
    for i, line in enumerate(lines[:-1]):
        nxt = lines[i + 1]
        if line and not line[0].isspace() and not line.startswith(("=", "-", ".", "*")) and nxt.startswith("    "):
            terms.add(re.sub(r"\s+", " ", line.strip()).lower())
    return terms


class GlossaryCoverageTest(unittest.TestCase):
    def test_required_terms_are_defined(self):
        defined = _defined_terms()
        missing = sorted(t for t in _REQUIRED_TERMS if t not in defined)
        self.assertEqual(
            missing,
            [],
            "these terminology terms have no glossary definition (X12.6 -- consistent usage needs one "
            "authoritative definition each):\n" + "\n".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
