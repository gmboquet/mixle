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
        # mixle.inference itself is only provisional (docs/maturity.rst documents just the narrow
        # "optimize and direct estimation helpers" slice as stable, not the whole package); the
        # direct-estimation core carves itself out as stable, while mixle.inference.production stays at
        # the provisional default under its own, more specific label. Both are more-specific prefixes
        # overriding the general one -- in opposite directions.
        self.assertEqual(maturity_of("mixle.inference"), Maturity.PROVISIONAL)
        self.assertEqual(maturity_of("mixle.inference.estimation"), Maturity.STABLE)
        self.assertEqual(maturity_of("mixle.inference.production.registry"), Maturity.PROVISIONAL)

    def test_direct_estimation_core_is_stable(self):
        # The narrow set of names docs/maturity.rst documents as the stable core: the literal symbolic
        # name the doc's table row uses ("mixle.inference.optimize"), the real module that name refers to
        # (estimation.py: optimize()/fit()/best_of()), and the EM strategy machinery estimation.py itself
        # builds on (em.py: EMStrategy/MonteCarloEM/OnlineEM/CompiledEM, imported back into estimation.py).
        for name in ("mixle.inference.optimize", "mixle.inference.estimation", "mixle.inference.em"):
            self.assertEqual(maturity_of(name), Maturity.STABLE, name)

    def test_inference_submodules_default_to_provisional(self):
        # Regression test for an over-broad blanket "mixle.inference" STABLE entry (fixed in this change):
        # only the direct-estimation core is documented stable in docs/maturity.rst. Every other
        # mixle.inference submodule -- applied/evolving surfaces such as causal inference, scoring rules,
        # resampling, uncertainty decomposition, multiple-testing correction, and model comparison -- was
        # never claimed stable and must not inherit it just by sharing the mixle.inference prefix.
        for name in (
            "mixle.inference.causal",
            "mixle.inference.scoring",
            "mixle.inference.resampling",
            "mixle.inference.uncertainty",
            "mixle.inference.multiple_testing",
            "mixle.inference.model_comparison",
        ):
            self.assertEqual(maturity_of(name), Maturity.PROVISIONAL, name)

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
        self.assertEqual(
            stable,
            {
                "mixle.stats",
                "mixle.semantics",
                "mixle.inference.optimize",
                "mixle.inference.estimation",
                "mixle.inference.em",
            },
        )


if __name__ == "__main__":
    unittest.main()
