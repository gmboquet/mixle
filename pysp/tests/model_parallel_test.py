"""Model-parallel estimation (C3): the model axis is distributed; bit-identical to the replicated path,
and composes with every data backend (local / mp / Spark / MPI) for the data x model decomposition."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import pysp.stats as stats
from pysp.inference import optimize
from pysp.utils.parallel.model_parallel import ModelParallelEncodedData, ModelParallelEstimator
from pysp.utils.parallel.planner import available_encoded_data_backends, encoded_data

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _composite():
    est = stats.CompositeEstimator((stats.GaussianEstimator(), stats.PoissonEstimator(), stats.CategoricalEstimator()))
    init = stats.CompositeDistribution(
        (
            stats.GaussianDistribution(0.0, 1.0),
            stats.PoissonDistribution(1.0),
            stats.CategoricalDistribution({"a": 0.5, "b": 0.5}),
        )
    )
    rng = np.random.RandomState(0)
    data = [(float(rng.randn()), int(rng.poisson(3)), "a" if rng.rand() < 0.5 else "b") for _ in range(400)]
    return est, init, data


class RegistrationTest(unittest.TestCase):
    def test_backend_is_registered(self):
        self.assertIn("model_parallel", available_encoded_data_backends())
        self.assertIsInstance(
            encoded_data(_composite()[2], model=_composite()[1], backend="model_parallel"), ModelParallelEncodedData
        )


class FactorParallelExactnessTest(unittest.TestCase):
    def test_estep_value_is_bit_identical(self):
        # the factor-parallel fold runs each child's seq_update with the IDENTICAL call as the serial
        # path, so the M-step output is exactly equal (not merely close).
        est, init, data = _composite()
        enc = init.dist_to_encoder().seq_encode(data)
        local = est.accumulator_factory().make()
        local.seq_update(enc, np.ones(len(data)), init)
        d = {}
        local.key_merge(d)
        local.key_replace(d)
        m_local = est.estimate(float(len(data)), local.value())

        mp = ModelParallelEncodedData(data, estimator=est, model=init, num_workers=3)
        m_mp = mp.pysp_seq_estimate(est, init)
        self.assertEqual(str(m_local), str(m_mp))

    def test_optimize_end_to_end_bit_identical(self):
        est, init, data = _composite()
        local = optimize(data, est, prev_estimate=init, max_its=10, out=None, backend="local")
        mp = optimize(data, est, prev_estimate=init, max_its=10, out=None, backend="model_parallel")
        self.assertEqual(str(local), str(mp))  # same init + bit-identical folds => identical EM trajectory

    def test_log_density_sum_matches(self):
        est, init, data = _composite()
        mp = ModelParallelEncodedData(data, estimator=est, model=init)
        n, ll = mp.pysp_seq_log_density_sum(init)
        self.assertEqual(n, float(len(data)))
        self.assertAlmostEqual(
            ll, float(np.sum(init.seq_log_density(init.dist_to_encoder().seq_encode(data)))), places=9
        )


class ComponentParallelTest(unittest.TestCase):
    """Mixtures are component-parallel: scoring + accumulation distributed, normalization central, exact."""

    def _mixture(self):
        est = stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(4)])
        init = stats.MixtureDistribution(
            [stats.GaussianDistribution(float(i) - 1.5, 1.0) for i in range(4)], [0.25] * 4
        )
        rng = np.random.RandomState(1)
        data = [float(rng.randn() + 3 * (rng.randint(4) - 1.5)) for _ in range(400)]
        return est, init, data

    def test_component_estep_bit_identical(self):
        est, init, data = self._mixture()
        enc = init.dist_to_encoder().seq_encode(data)
        local = est.accumulator_factory().make()
        local.seq_update(enc, np.ones(len(data)), init)
        d = {}
        local.key_merge(d)
        local.key_replace(d)
        m_local = est.estimate(float(len(data)), local.value())
        mp = ModelParallelEncodedData(data, estimator=est, model=init, num_workers=3)
        m_mp = mp.pysp_seq_estimate(est, init)
        self.assertEqual(str(m_local), str(m_mp))

    def test_optimize_end_to_end_bit_identical(self):
        est, init, data = self._mixture()
        local = optimize(data, est, prev_estimate=init, max_its=10, out=None, backend="local")
        mp = optimize(data, est, prev_estimate=init, max_its=10, out=None, backend="model_parallel")
        self.assertEqual(str(local), str(mp))


class NestedRecursiveTest(unittest.TestCase):
    """Recursive fold: nested shardable models are model-parallel at the widest axis, still bit-identical."""

    def _check(self, est, init, data, its=8):
        local = optimize(data, est, prev_estimate=init, max_its=its, out=None, backend="local")
        mp = optimize(data, est, prev_estimate=init, max_its=its, out=None, backend="model_parallel")
        self.assertEqual(str(local), str(mp))

    def test_composite_of_mixture_and_leaf(self):
        # widest axis is the inner mixture's components, nested inside factor 0 of the composite
        est = stats.CompositeEstimator(
            (stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(5)]), stats.PoissonEstimator())
        )
        init = stats.CompositeDistribution(
            (
                stats.MixtureDistribution([stats.GaussianDistribution(float(i) - 2, 1.0) for i in range(5)], [0.2] * 5),
                stats.PoissonDistribution(2.0),
            )
        )
        rng = np.random.RandomState(3)
        data = [(float(rng.randn() + 2 * (rng.randint(5) - 2)), int(rng.poisson(2))) for _ in range(400)]
        self._check(est, init, data)

    def test_mixture_of_composites(self):
        def comp_est():
            return stats.CompositeEstimator((stats.GaussianEstimator(), stats.PoissonEstimator()))

        def comp(mu, lam):
            return stats.CompositeDistribution((stats.GaussianDistribution(mu, 1.0), stats.PoissonDistribution(lam)))

        est = stats.MixtureEstimator([comp_est(), comp_est(), comp_est()])
        init = stats.MixtureDistribution([comp(-2.0, 1.0), comp(0.0, 3.0), comp(2.0, 6.0)], [1 / 3] * 3)
        rng = np.random.RandomState(4)
        data = [(float(rng.randn() + 2 * (rng.randint(3) - 1)), int(rng.poisson(3))) for _ in range(400)]
        self._check(est, init, data)


class FallbackTest(unittest.TestCase):
    def test_leaf_model_falls_back_and_is_identical(self):
        # a plain Gaussian is atomic -> replicated accumulation, still exact via the same handle.
        est = stats.GaussianEstimator()
        init = stats.GaussianDistribution(0.0, 1.0)
        rng = np.random.RandomState(2)
        data = [float(rng.randn() * 2 + 1) for _ in range(200)]
        local = optimize(data, est, prev_estimate=init, max_its=5, out=None, backend="local")
        mp = optimize(data, est, prev_estimate=init, max_its=5, out=None, backend="model_parallel")
        self.assertEqual(str(local), str(mp))


def _mixture_data(seed=7, k=6, n=800):
    est = stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(k)])
    init = stats.MixtureDistribution([stats.GaussianDistribution(float(i) - 2.5, 1.0) for i in range(k)], [1 / k] * k)
    rng = np.random.RandomState(seed)
    data = [float(rng.randn() + 3 * (rng.randint(k) - 2.5)) for _ in range(n)]
    return est, init, data


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


class DataModelCompositionTest(unittest.TestCase):
    """ModelParallelEstimator distributes the model axis and composes with any data backend."""

    def test_local_backend_is_bit_identical(self):
        # model-parallel estimator + the single-partition local backend == plain estimator, exactly
        est, init, data = _mixture_data()
        base = optimize(data, est, prev_estimate=init, max_its=8, out=None, backend="local")
        dm = optimize(
            data, ModelParallelEstimator(est, num_workers=3), prev_estimate=init, max_its=8, out=None, backend="local"
        )
        self.assertEqual(str(base), str(dm))

    def test_mp_backend_data_and_model_parallel(self):
        # data sharded across worker processes + model axis distributed in each -> matches base LL
        est, init, data = _mixture_data()
        base = optimize(data, est, prev_estimate=init, max_its=8, out=None, backend="local")
        dm = optimize(
            data,
            ModelParallelEstimator(est, num_workers=2),
            prev_estimate=init,
            max_its=8,
            out=None,
            backend="mp",
            num_workers=2,
        )
        self.assertTrue(np.isclose(_ll(base, data), _ll(dm, data), rtol=1e-6), (_ll(base, data), _ll(dm, data)))


class SparkDataModelTest(unittest.TestCase):
    def test_spark_data_and_model_parallel(self):
        try:
            from pyspark import SparkConf, SparkContext
        except ImportError:
            self.skipTest("pyspark not installed")
        java_home = os.environ.get("JAVA_HOME")
        java_bin = (os.path.join(java_home, "bin", "java") if java_home else None) or shutil.which("java")
        try:  # the macOS /usr/bin/java stub exists but is non-functional -> probe it, skip if it fails
            if java_bin is None or subprocess.run([java_bin, "-version"], capture_output=True).returncode != 0:
                self.skipTest("no functional Java runtime for Spark")
        except OSError:
            self.skipTest("no functional Java runtime for Spark")

        est, init, data = _mixture_data()
        base = optimize(data, est, prev_estimate=init, max_its=8, out=None, backend="local")
        sc = SparkContext(conf=SparkConf().setMaster("local[3]").setAppName("mp-dm").set("spark.ui.enabled", "false"))
        try:
            sc.setLogLevel("ERROR")
            rdd = sc.parallelize(data, 4)
            dm = optimize(
                rdd,
                ModelParallelEstimator(est, num_workers=2),
                prev_estimate=init,
                max_its=8,
                out=None,
                backend="spark",
            )
        finally:
            sc.stop()
        self.assertTrue(np.isclose(_ll(base, data), _ll(dm, data), rtol=1e-6), (_ll(base, data), _ll(dm, data)))


_MPI_SCRIPT = r"""
import sys
import numpy as np
sys.path.insert(0, %(repo)r)
import pysp.stats as stats
from pysp.inference import optimize
from pysp.utils.parallel.mpi import MPIEncodedData, mpi_out
from pysp.utils.parallel import ModelParallelEstimator
from mpi4py import MPI

est = stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(6)])
init = stats.MixtureDistribution([stats.GaussianDistribution(float(i) - 2.5, 1.0) for i in range(6)], [1 / 6] * 6)
rng = np.random.RandomState(7)
data = [float(rng.randn() + 3 * (rng.randint(6) - 2.5)) for _ in range(800)]  # identical on every rank

mp_est = ModelParallelEstimator(est, num_workers=2)
enc = MPIEncodedData(data, estimator=mp_est)  # data sharded round-robin across ranks
dm = optimize(None, mp_est, enc_data=enc, prev_estimate=init, max_its=8, out=mpi_out())

if MPI.COMM_WORLD.Get_rank() == 0:
    base = optimize(data, est, prev_estimate=init, max_its=8, out=None, backend="local")
    def ll(m):
        return float(np.sum(m.seq_log_density(m.dist_to_encoder().seq_encode(data))))
    assert np.isclose(ll(base), ll(dm), rtol=1e-6), (ll(base), ll(dm))
    print("MPI-DM-OK")
"""


class MPIDataModelTest(unittest.TestCase):
    def test_mpi_data_and_model_parallel(self):
        try:
            import mpi4py  # noqa: F401
        except ImportError:
            self.skipTest("mpi4py not installed")
        launcher = shutil.which("mpiexec") or shutil.which("mpirun")
        if launcher is None:
            self.skipTest("no MPI launcher on PATH")

        with tempfile.TemporaryDirectory() as td:
            script = os.path.join(td, "mpi_dm.py")
            with open(script, "w") as f:
                f.write(_MPI_SCRIPT % {"repo": REPO})
            env = dict(os.environ, PYTHONPATH=REPO)
            res = subprocess.run(
                [launcher, "-n", "3", sys.executable, script], capture_output=True, text=True, timeout=600, env=env
            )
        self.assertEqual(res.returncode, 0, "mpi launch failed:\n%s\n%s" % (res.stdout, res.stderr))
        self.assertIn("MPI-DM-OK", res.stdout)


class AutoWiringTest(unittest.TestCase):
    """auto_parallel_estimator consults the C2 planner to pick the axis and size the model split."""

    def test_wide_mixture_picks_model_parallel(self):
        from pysp.utils.parallel import auto_parallel_estimator
        from pysp.utils.parallel.planner import Resources

        est = stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(12)])
        model = stats.MixtureDistribution([stats.GaussianDistribution(float(i), 1.0) for i in range(12)], [1 / 12] * 12)
        chosen, dec = auto_parallel_estimator(est, model, Resources.local(4), n_data=40)
        self.assertTrue(dec.is_model_parallel)
        self.assertIsInstance(chosen, ModelParallelEstimator)
        self.assertEqual(chosen.num_workers, len(dec.cuts))

    def test_small_model_large_n_picks_data_parallel(self):
        from pysp.utils.parallel import auto_parallel_estimator
        from pysp.utils.parallel.planner import Resources

        est = stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(5)])
        model = stats.MixtureDistribution([stats.GaussianDistribution(float(i), 1.0) for i in range(5)], [0.2] * 5)
        chosen, dec = auto_parallel_estimator(est, model, Resources.local(4), n_data=100_000)
        self.assertFalse(dec.is_model_parallel)
        self.assertIs(chosen, est)  # plain estimator -> replicate model, shard data

    def test_single_device_picks_data_parallel(self):
        from pysp.utils.parallel import auto_parallel_estimator
        from pysp.utils.parallel.planner import Resources

        est = stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(8)])
        model = stats.MixtureDistribution([stats.GaussianDistribution(float(i), 1.0) for i in range(8)], [1 / 8] * 8)
        chosen, dec = auto_parallel_estimator(est, model, Resources.local(1), n_data=40)
        self.assertFalse(dec.is_model_parallel)
        self.assertIs(chosen, est)

    def test_auto_chosen_estimator_matches_base(self):
        from pysp.utils.parallel import auto_parallel_estimator
        from pysp.utils.parallel.planner import Resources

        est, init, data = _mixture_data(k=8, n=400)
        base = optimize(data, est, prev_estimate=init, max_its=8, out=None, backend="local")
        chosen, dec = auto_parallel_estimator(est, init, Resources.local(4), n_data=len(data))
        self.assertTrue(dec.is_model_parallel)  # 8 components over 4 cpus, small N
        auto = optimize(data, chosen, prev_estimate=init, max_its=8, out=None, backend="local")
        self.assertEqual(str(base), str(auto))  # single-partition local -> bit-identical


if __name__ == "__main__":
    unittest.main()
