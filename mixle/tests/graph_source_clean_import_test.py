"""Regression test: mixle.data.sources.graph_source could not be imported in a clean process.

mixle.data.sources.graph_source's own top-level `from mixle.stats.compute.pdist import
DataSequenceEncoder` (needed as GraphDataEncoder's base class) forces mixle.stats's package
__init__ to run first. mixle/stats/__init__.py eagerly imports three graph-family modules that each
did a top-level `from mixle.data.sources.graph_source import ...` of their own: erdos_renyi_graph.py,
random_dot_product_graph.py, and stochastic_block_graph.py. Whichever of those loads first re-enters
graph_source while it is still mid-import (paused inside its own DataSequenceEncoder import, above
this module in the same chain), so the names it wants (GraphDataEncoder, GraphObservation, the private
coercion/validation helpers) do not exist as attributes yet -- ImportError, e.g.:

    ImportError: cannot import name 'GraphDataEncoder' from partially initialized module
    'mixle.data.sources.graph_source' (most likely due to a circular import)

Fixing only the first module hit (erdos_renyi_graph.py, imported first in stats/__init__.py) is not
enough: random_dot_product_graph.py and stochastic_block_graph.py have the exact same shape of
top-level import and are reached later in the same package-init chain, so the failure just moves to
whichever of them mixle.stats reaches next. All three needed the fix.

This only reproduces in a genuinely clean process: mixle/tests/graph_source_audit_test.py already
carries a `import mixle.stats  # noqa: F401 -- fully initialize the package to avoid a circular
import` workaround at its top for exactly this reason -- warming mixle.stats via its normal top-down
order (which happens to load mixle.stats.compute.pdist before reaching the graphs.* imports, see the
erdos_renyi_graph.py fix comment) sidesteps the bug without fixing it. Only a bare
`python -c "import mixle.data.sources.graph_source"` in a fresh interpreter, before anything else has
warmed mixle.stats, actually hits it -- matching how the bug was originally reported (a direct import
from the installed wheel, not through mixle.stats or mixle.data first).
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_import(*modules: str) -> subprocess.CompletedProcess:
    # Pin PYTHONPATH to this worktree's own repo root (this repo's editable-install gotcha: the
    # installed editable package can point at a stale sibling worktree) so a subprocess run from
    # here exercises the source we just edited, not some other checkout.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    code = "; ".join(f"import {m}" for m in modules)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


class GraphSourceCleanImportTestCase(unittest.TestCase):
    def test_graph_source_imports_cleanly_as_the_entry_point(self):
        # The exact reported entry point: mixle.data.sources.graph_source imported directly, before
        # anything else has warmed mixle.stats.
        result = _clean_import("mixle.data.sources.graph_source")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_erdos_renyi_graph_imports_cleanly_on_its_own(self):
        # Reverse order: the cycle is symmetric, so the module that used to close the loop must also
        # import standalone.
        result = _clean_import("mixle.stats.graphs.erdos_renyi_graph")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_random_dot_product_graph_imports_cleanly_on_its_own(self):
        result = _clean_import("mixle.stats.graphs.random_dot_product_graph")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stochastic_block_graph_imports_cleanly_on_its_own(self):
        result = _clean_import("mixle.stats.graphs.stochastic_block_graph")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixle_stats_imports_cleanly(self):
        result = _clean_import("mixle.stats")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixle_data_imports_cleanly(self):
        result = _clean_import("mixle.data")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_graph_source_then_stats_imports_cleanly(self):
        result = _clean_import("mixle.data.sources.graph_source", "mixle.stats")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stats_then_graph_source_imports_cleanly(self):
        result = _clean_import("mixle.stats", "mixle.data.sources.graph_source")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
