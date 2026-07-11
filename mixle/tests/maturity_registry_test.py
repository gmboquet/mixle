"""The machine-readable maturity registry mirrors the docs and A1.5 tiers (worklist A1.2).

``mixle.maturity`` is the machine-readable form of ``docs/maturity.rst``. These tests keep the two in sync
(a surface documented as stable must resolve as stable, etc.), pin the longest-prefix resolution and the
conservative default, and enforce consistency with the deprecation policy's tiers (worklist A1.5): only
``mixle.experimental`` is EXPERIMENTAL.
"""

import re
import unittest
from pathlib import Path

from mixle.maturity import DEFAULT_MATURITY, MATURITY_REGISTRY, Maturity, maturity_of, status_of

REPO_ROOT = Path(__file__).resolve().parents[2]
MATURITY_DOC = REPO_ROOT / "docs" / "maturity.rst"


def _documented_surfaces():
    """Parse (surface, status) pairs from the Maturity Map list-table in docs/maturity.rst."""
    lines = MATURITY_DOC.read_text().splitlines()
    pairs = []
    surfaces, status = None, None
    for ln in lines:
        m_row = re.match(r"\s*\*\s*-\s*(.+)", ln)
        m_cont = re.match(r"\s*-\s*(.+)", ln)
        if m_row:
            if surfaces:
                for s in surfaces:
                    pairs.append((s, status))
            surfaces = re.findall(r"``(mixle[\w.]*)``", m_row.group(1))
            status = None
        elif m_cont and surfaces and status is None:
            status = m_cont.group(1).strip()
    if surfaces:
        for s in surfaces:
            pairs.append((s, status))
    return [(s, st) for s, st in pairs if s and st]


class MaturityDocSyncTest(unittest.TestCase):
    def test_registry_matches_documented_maturity(self):
        surfaces = _documented_surfaces()
        self.assertGreaterEqual(len(surfaces), 10, "failed to parse the maturity map from docs/maturity.rst")
        for surface, status in surfaces:
            expected = Maturity.STABLE if status.startswith("Stable core") else Maturity.PROVISIONAL
            self.assertEqual(
                maturity_of(surface),
                expected,
                f"{surface!r} is documented as {status!r} but the registry resolves it to "
                f"{maturity_of(surface).value!r}",
            )


class MaturityResolutionTest(unittest.TestCase):
    def test_longest_prefix_inheritance(self):
        self.assertEqual(maturity_of("mixle.stats.latent.hidden_markov"), Maturity.STABLE)
        self.assertEqual(maturity_of("mixle.inference.optimize"), Maturity.STABLE)

    def test_more_specific_prefix_overrides(self):
        # mixle.inference is stable, but mixle.inference.production is only practical-helper provisional.
        self.assertEqual(maturity_of("mixle.inference"), Maturity.STABLE)
        self.assertEqual(maturity_of("mixle.inference.production.registry"), Maturity.PROVISIONAL)

    def test_experimental_namespace(self):
        self.assertEqual(maturity_of("mixle.experimental"), Maturity.EXPERIMENTAL)
        self.assertEqual(maturity_of("mixle.experimental.program"), Maturity.EXPERIMENTAL)

    def test_unclassified_defaults_to_provisional(self):
        self.assertEqual(maturity_of("mixle.some_unlisted_surface"), DEFAULT_MATURITY)
        self.assertEqual(DEFAULT_MATURITY, Maturity.PROVISIONAL)
        self.assertIn("provisional", status_of("mixle.some_unlisted_surface").lower())


class MaturityPolicyConsistencyTest(unittest.TestCase):
    def test_only_experimental_namespace_is_experimental(self):
        experimental = {k for k, (tier, _) in MATURITY_REGISTRY.items() if tier is Maturity.EXPERIMENTAL}
        self.assertEqual(experimental, {"mixle.experimental"})

    def test_stable_surfaces_are_the_documented_core(self):
        stable = {k for k, (tier, _) in MATURITY_REGISTRY.items() if tier is Maturity.STABLE}
        self.assertEqual(stable, {"mixle.stats", "mixle.inference"})


if __name__ == "__main__":
    unittest.main()
