"""Reconcile mixle's two MPI EM routes (worklist D8.5).

mixle has two MPI distributed-EM transports that coexisted without a stated relationship:

  * ``mixle.inference.mpi_executor.mpi_fit`` -- a standalone ``comm.reduce`` **tree-fold** (``O(log W)``);
  * ``mixle.utils.parallel.mpi.MPIEncodedData`` -- the integrated ``optimize`` backend, **gather-to-root**.

Both fold sufficient statistics over ALL the data every EM step -- route A over contiguous shards, route B
over round-robin shards -- from the same initial model. Summation is partition-invariant, so the two routes
compute the same total statistics, the same M-step, and the same fit up to floating-point summation order.
This test runs BOTH under ``mpirun`` and confirms they agree to numerical precision (measured ~1e-13 on the
means, ~1e-16 on the weights), which is the strong form of the reconciliation: the two routes are the same
distributed EM via different reductions, not merely similar. See each module's docstring for canonical use.
"""

import json
import os
import shutil
import subprocess
import sys
import unittest

import numpy as np
import pytest

pytest.importorskip("mpi4py")

pytestmark = [pytest.mark.optional]

_MPIRUN = shutil.which("mpirun") or shutil.which("mpiexec")
_RUNNER = os.path.join(os.path.dirname(__file__), "_mpi_route_equivalence_runner.py")


@unittest.skipUnless(_MPIRUN and os.path.exists(_RUNNER), "no MPI runtime")
class MPIRouteEquivalenceTest(unittest.TestCase):
    def test_both_routes_reach_the_same_fit(self):
        proc = subprocess.run(
            [_MPIRUN, "-n", "3", "--oversubscribe", sys.executable, _RUNNER],
            capture_output=True,
            text=True,
            timeout=600,
        )
        results = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
        self.assertTrue(results, "no RESULT from mpirun; rc=%s stderr=%s" % (proc.returncode, proc.stderr[-800:]))
        d = json.loads(results[0][len("RESULT ") :])
        self.assertEqual(d["size"], 3)  # really ran 3 ranks

        w_a, w_b = np.array(d["w_a"]), np.array(d["w_b"])
        mu_a, mu_b = np.array(d["mu_a"]), np.array(d["mu_b"])
        # Same total sufficient statistics -> same fit, up to floating-point summation order only.
        self.assertEqual(len(w_a), len(w_b))
        self.assertTrue(np.allclose(w_a, w_b, atol=1e-9), f"weights disagree: A={w_a} B={w_b}")
        self.assertTrue(np.allclose(mu_a, mu_b, atol=1e-8), f"means disagree: A={mu_a} B={mu_b}")


if __name__ == "__main__":
    unittest.main()
