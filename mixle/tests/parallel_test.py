"""Tests for the multiprocessing and MPI estimation backends."""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from mixle.inference import seq_estimate, seq_initialize
from mixle.inference.estimation import constant, optimize
from mixle.inference.streaming import StreamingEstimator
from mixle.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    CompositeDistribution,
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    seq_encode,
    seq_log_density_sum,
)
from mixle.utils.parallel.multiprocessing import MPEncodedData
from mixle.utils.parallel.planner import EncodedDataHandle, encoded_data, is_encoded_data_handle

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_data(n=400, seed=1):
    dist = MixtureDistribution(
        [
            CompositeDistribution((GaussianDistribution(-3.0, 1.0), CategoricalDistribution({"a": 0.8, "b": 0.2}))),
            CompositeDistribution((GaussianDistribution(3.0, 1.0), CategoricalDistribution({"a": 0.1, "b": 0.9}))),
        ],
        [0.6, 0.4],
    )
    return dist.sampler(seed=seed).sample(n)


def make_estimator():
    return MixtureEstimator([CompositeEstimator((GaussianEstimator(), CategoricalEstimator(pseudo_count=0.5)))] * 2)


def make_start_model():
    # deliberately perturbed starting point: EM separates quickly from here,
    # whereas a symmetric initialization stalls near the saddle for many
    # iterations (not what these tests are about)
    return MixtureDistribution(
        [
            CompositeDistribution((GaussianDistribution(-1.0, 4.0), CategoricalDistribution({"a": 0.6, "b": 0.4}))),
            CompositeDistribution((GaussianDistribution(1.0, 4.0), CategoricalDistribution({"a": 0.4, "b": 0.6}))),
        ],
        [0.5, 0.5],
    )


class MultiprocessingBackendTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.est = make_estimator()
        cls.enc_local = seq_encode(cls.data, estimator=cls.est)
        cls.m0 = seq_initialize(cls.enc_local, cls.est, np.random.RandomState(7), p=1.0)

    def test_seq_estimate_matches_serial(self):
        with MPEncodedData(self.data, estimator=self.est, num_workers=3) as enc:
            m_par = enc.pysp_seq_estimate(self.est, self.m0)
        m_ser = seq_estimate(self.enc_local, self.est, self.m0)
        self.assertTrue(np.allclose(sorted(m_par.w), sorted(m_ser.w), atol=1.0e-9))
        ll_par = sum(m_par.log_density(x) for x in self.data[:50])
        ll_ser = sum(m_ser.log_density(x) for x in self.data[:50])
        self.assertAlmostEqual(ll_par, ll_ser, places=8)

    def test_shared_protocol_factory_can_build_multiprocessing_handle(self):
        with encoded_data(self.data, estimator=self.est, backend="mp", num_workers=2) as enc:
            self.assertIsInstance(enc, EncodedDataHandle)
            self.assertIsInstance(enc, MPEncodedData)
            self.assertTrue(is_encoded_data_handle(enc))
            cnt, ll = seq_log_density_sum(enc, self.m0)
        self.assertEqual(cnt, len(self.data))
        self.assertTrue(np.isfinite(ll))

    def test_log_density_sum_matches_serial(self):
        with MPEncodedData(self.data, estimator=self.est, num_workers=4) as enc:
            cnt_p, ll_p = enc.pysp_seq_log_density_sum(self.m0)
        cnt_s, ll_s = seq_log_density_sum(self.enc_local, self.m0)
        self.assertEqual(cnt_p, cnt_s)
        self.assertAlmostEqual(ll_p, ll_s, places=7)

    def test_dispatch_through_module_functions(self):
        # mixle.stats seq_* entry points must recognize the handle
        with MPEncodedData(self.data, estimator=self.est, num_workers=2) as enc:
            cnt, ll = seq_log_density_sum(enc, self.m0)
            m1 = seq_estimate(enc, self.est, self.m0)
            m_init = seq_initialize(enc, self.est, np.random.RandomState(3), p=0.5)
        self.assertEqual(cnt, len(self.data))
        self.assertTrue(np.isfinite(ll))
        self.assertEqual(len(m1.components), 2)
        self.assertEqual(len(m_init.components), 2)

    def test_optimize_with_handle(self):
        m_start = make_start_model()
        with MPEncodedData(self.data, estimator=self.est, num_workers=2) as enc:
            model = optimize(None, self.est, enc_data=enc, max_its=30, prev_estimate=m_start, out=io.StringIO())
            _, ll_fit = enc.pysp_seq_log_density_sum(model)
        _, ll_init = seq_log_density_sum(self.enc_local, m_start)
        self.assertGreater(ll_fit, ll_init)
        mus = sorted(c.dists[0].mu for c in model.components)
        self.assertAlmostEqual(mus[0], -3.0, delta=0.5)
        self.assertAlmostEqual(mus[1], 3.0, delta=0.5)

    def test_optimize_can_build_multiprocessing_backend(self):
        m_start = make_start_model()
        model = optimize(
            self.data, self.est, max_its=30, prev_estimate=m_start, backend="mp", num_workers=2, out=io.StringIO()
        )
        _, ll_fit = seq_log_density_sum(self.enc_local, model)
        _, ll_init = seq_log_density_sum(self.enc_local, m_start)
        self.assertGreater(ll_fit, ll_init)
        mus = sorted(c.dists[0].mu for c in model.components)
        self.assertAlmostEqual(mus[0], -3.0, delta=0.5)
        self.assertAlmostEqual(mus[1], 3.0, delta=0.5)

    def test_streaming_estimator_can_use_multiprocessing_handle(self):
        estimator = GaussianEstimator()
        start = GaussianDistribution(0.0, 1.0)
        batch1 = [-2.0, -1.0, 0.0, 1.0]
        batch2 = [1.5, 2.0, 2.5]

        serial = StreamingEstimator(estimator, schedule=constant(0.4), model=start)
        serial.update(batch1)
        serial.update(batch2)

        parallel = StreamingEstimator(estimator, schedule=constant(0.4), model=start)
        # num_workers=1 is enough to exercise the real subprocess/pickling path for
        # this handle; each `with` block still pays the fixed spawn+import cost of a
        # fresh worker process (the dominant cost here, not data volume or worker
        # count), so keep worker count minimal rather than paying for parallelism
        # this tiny batch doesn't need.
        with encoded_data(batch1, estimator=estimator, model=start, backend="mp", num_workers=1) as enc:
            model1 = parallel.update(enc_data=enc)
        with encoded_data(batch2, estimator=estimator, model=model1, backend="mp", num_workers=1) as enc:
            parallel.update(enc_data=enc)

        np.testing.assert_allclose(
            np.asarray(parallel.value(), dtype=float),
            np.asarray(serial.value(), dtype=float),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(parallel.model.mu, serial.model.mu, places=12)
        self.assertAlmostEqual(parallel.model.sigma2, serial.model.sigma2, places=12)

    def test_close_idempotent_and_len(self):
        enc = MPEncodedData(self.data, estimator=self.est, num_workers=2)
        self.assertEqual(len(enc), len(self.data))
        enc.close()
        enc.close()


MPI_SCRIPT = r"""
import io, sys
import numpy as np
sys.path.insert(0, %(repo)r)
from mixle.tests.parallel_test import make_data, make_estimator, make_start_model
from mixle.stats import seq_encode, seq_log_density_sum
from mixle.inference.estimation import optimize
from mixle.utils.parallel.mpi import MPIEncodedData, mpi_out
from mpi4py import MPI

data = make_data()
est = make_estimator()
enc_local = seq_encode(data, estimator=est)
m0 = make_start_model()

enc = MPIEncodedData(data, estimator=est)
assert len(enc) == len(data)

cnt, ll = enc.pysp_seq_log_density_sum(m0)
_, ll_ser = seq_log_density_sum(enc_local, m0)
assert abs(ll - ll_ser) < 1.0e-6, (ll, ll_ser)

model = optimize(None, est, enc_data=enc, max_its=30, prev_estimate=m0, out=mpi_out())
_, ll_fit = enc.pysp_seq_log_density_sum(model)
assert ll_fit > ll, (ll_fit, ll)

mus = sorted(c.dists[0].mu for c in model.components)
assert abs(mus[0] + 3.0) < 0.5 and abs(mus[1] - 3.0) < 0.5, mus

# every rank must hold the identical model
all_w = MPI.COMM_WORLD.gather(tuple(model.w), root=0)
if MPI.COMM_WORLD.Get_rank() == 0:
    assert all(w == all_w[0] for w in all_w), all_w
    print('MPI-BACKEND-OK')
"""


class MPIBackendTestCase(unittest.TestCase):
    def test_mpi_backend_two_ranks(self):
        try:
            import mpi4py  # noqa: F401
        except ImportError:
            self.skipTest("mpi4py not installed")
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpiexec is None:
            self.skipTest("no MPI launcher on PATH")

        with tempfile.TemporaryDirectory() as td:
            script = os.path.join(td, "mpi_check.py")
            with open(script, "w") as f:
                f.write(MPI_SCRIPT % {"repo": REPO})
            env = dict(os.environ, PYTHONPATH=REPO)
            res = subprocess.run(
                [mpiexec, "-n", "2", sys.executable, script], capture_output=True, text=True, timeout=600, env=env
            )
        self.assertEqual(res.returncode, 0, "mpiexec failed:\n%s\n%s" % (res.stdout, res.stderr))
        self.assertIn("MPI-BACKEND-OK", res.stdout)


if __name__ == "__main__":
    unittest.main()
