"""MPI distributed EM (mixle.inference.mpi_executor): an mpirun fit must match the serial fit.

Launches the runner under ``mpirun -n 3`` as a subprocess and compares rank-0's fitted parameters to the
in-process serial baseline. Skips cleanly when mpi4py or an MPI runtime is unavailable.
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

_MPIRUN = shutil.which("mpirun") or "/opt/homebrew/bin/mpirun"
_RUNNER = os.path.join(os.path.dirname(__file__), "_mpi_fit_runner.py")


def _serial_baseline():
    import mixle.stats as st
    from mixle.inference.heterogeneous_executor import heterogeneous_fit

    rng = np.random.RandomState(0)
    comps = [st.GaussianDistribution(float(6 * rng.randn()), float(0.5 + rng.rand())) for _ in range(3)]
    m = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(3))))
    data = m.sampler(1).sample(4000)
    fit = heterogeneous_fit(m, data, max_its=12, n_shards=1)
    return sorted(float(x) for x in fit.w), sorted(float(c.mu) for c in fit.components)


@unittest.skipUnless(os.path.exists(_MPIRUN), "no mpirun executable")
class MPIDistributedEMTest(unittest.TestCase):
    def test_mpi_fit_matches_serial(self):
        proc = subprocess.run(
            [_MPIRUN, "-n", "3", "--oversubscribe", sys.executable, _RUNNER],
            capture_output=True,
            text=True,
            timeout=600,
        )
        results = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
        self.assertTrue(results, "no RESULT from mpirun; rc=%s stderr=%s" % (proc.returncode, proc.stderr[-800:]))
        mpi = json.loads(results[0][len("RESULT ") :])
        self.assertEqual(mpi["size"], 3)  # really ran 3 ranks

        w, mu = _serial_baseline()
        self.assertTrue(np.allclose(mpi["w"], w, atol=1e-7), "weights: mpi=%r serial=%r" % (mpi["w"], w))
        self.assertTrue(np.allclose(mpi["mu"], mu, atol=1e-6), "means: mpi=%r serial=%r" % (mpi["mu"], mu))


if __name__ == "__main__":
    unittest.main()
