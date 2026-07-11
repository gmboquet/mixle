"""Absolute latency budgets for import and a smoke fit (worklist B7.4).

B7.4 asks for absolute budgets on import and smoke latency as a regression tripwire, distinct from the
throughput benchmarks (which run on a dedicated runner). These are deliberately loose -- ~100x the measured
values -- so they never flake on a loaded or cold CI runner, but still catch a catastrophic regression: a
heavy dependency accidentally imported at ``import mixle`` (turning a 0.05 s import into seconds), or an
``O(n^2)`` / non-converging blowup in a basic fit. They are a floor of correctness for latency, not a
performance measurement, so they never pressure anyone to weaken correctness to pass.
"""

import subprocess
import sys
import time
import unittest

import numpy as np

from mixle.inference.estimation import optimize
from mixle.stats import GaussianEstimator, MixtureEstimator

# Measured on a laptop: import ~0.05 s, each fit ~0.005 s. Budgets are ~100x that -- catastrophic-regression
# tripwires, not performance targets.
_IMPORT_BUDGET_S = 5.0
_FIT_BUDGET_S = 3.0


class LatencyBudgetTest(unittest.TestCase):
    def test_base_import_is_fast(self):
        # Fresh interpreter so this measures a real cold ``import mixle``, not an already-imported module.
        # A regression that pulls a heavy optional backend to the top level would blow this budget.
        t = time.perf_counter()
        proc = subprocess.run([sys.executable, "-c", "import mixle"], capture_output=True, text=True, timeout=60)
        elapsed = time.perf_counter() - t
        self.assertEqual(proc.returncode, 0, f"import mixle failed: {proc.stderr[-400:]}")
        self.assertLess(elapsed, _IMPORT_BUDGET_S, f"import mixle took {elapsed:.2f}s (budget {_IMPORT_BUDGET_S}s)")

    def test_smoke_fits_are_fast(self):
        data = np.random.RandomState(0).normal(0.0, 1.0, 2000).tolist()
        t = time.perf_counter()
        optimize(data, GaussianEstimator(), max_its=20, out=None)
        gaussian = time.perf_counter() - t
        self.assertLess(gaussian, _FIT_BUDGET_S, f"Gaussian smoke fit took {gaussian:.2f}s (budget {_FIT_BUDGET_S}s)")

        blobs = np.concatenate(
            [np.random.RandomState(0).randn(1000), np.random.RandomState(1).randn(1000) + 5.0]
        ).tolist()
        t = time.perf_counter()
        optimize(blobs, MixtureEstimator([GaussianEstimator(), GaussianEstimator()]), max_its=30, out=None)
        mixture = time.perf_counter() - t
        self.assertLess(mixture, _FIT_BUDGET_S, f"mixture smoke fit took {mixture:.2f}s (budget {_FIT_BUDGET_S}s)")


if __name__ == "__main__":
    unittest.main()
