"""Spark distributed EM (mixle.inference.spark_executor): RDD.treeReduce must match the serial fit.

Skips cleanly when pyspark or a Java runtime is unavailable. Locates a brew keg-only openjdk if JAVA_HOME
is unset. Marked 'spark'/'optional' so the default fast gate does not pay the JVM startup.
"""

import os
import sys
import unittest

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")

pytestmark = [pytest.mark.spark, pytest.mark.optional]

_SC = None


def _ensure_java() -> bool:
    if os.environ.get("JAVA_HOME") and os.path.exists(os.environ["JAVA_HOME"]):
        return True
    for p in ("/opt/homebrew/opt/openjdk@17", "/opt/homebrew/opt/openjdk", "/usr/local/opt/openjdk@17"):
        if os.path.exists(p):
            os.environ["JAVA_HOME"] = p
            os.environ["PATH"] = p + "/bin:" + os.environ.get("PATH", "")
            return True
    return False


def setUpModule() -> None:
    global _SC
    if not _ensure_java():
        raise unittest.SkipTest("no Java runtime for Spark")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark import SparkConf, SparkContext

    try:
        _SC = SparkContext(
            conf=SparkConf().setMaster("local[2]").setAppName("mixle-test").set("spark.ui.enabled", "false")
        )
        _SC.setLogLevel("ERROR")
    except Exception as e:  # pragma: no cover - environment dependent  # noqa: BLE001
        raise unittest.SkipTest("could not start Spark: %s" % e)


def tearDownModule() -> None:
    if _SC is not None:
        _SC.stop()


def _gmm():
    rng = np.random.RandomState(0)
    comps = [st.GaussianDistribution(float(6 * rng.randn()), float(0.5 + rng.rand())) for _ in range(3)]
    return st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(3))))


import mixle.stats as st  # noqa: E402
from mixle.inference.heterogeneous_executor import heterogeneous_em_step, heterogeneous_fit  # noqa: E402
from mixle.inference.spark_executor import spark_em_step, spark_fit  # noqa: E402


class SparkDistributedEMTest(unittest.TestCase):
    def test_spark_em_step_matches_serial(self):
        # 1000 obs / 4 shards is plenty to exercise the depth=2 tree-reduce (still a genuine
        # 4 -> 2 -> 1 combine tree) while avoiding redundant Spark task-scheduling overhead;
        # data size beyond this does not change the numeric result (sufficient stats are
        # shard-invariant) so it was only adding runtime, not coverage.
        m = _gmm()
        data = m.sampler(1).sample(1000)
        est = m.estimator()
        serial = heterogeneous_em_step(est, m, data, n_shards=8)
        spark = spark_em_step(_SC, est, m, data, n_shards=4, depth=2)
        self.assertTrue(np.allclose(sorted(serial.w), sorted(spark.w), atol=1e-9))
        sm = sorted(c.mu for c in serial.components)
        km = sorted(c.mu for c in spark.components)
        self.assertTrue(np.allclose(sm, km, atol=1e-9))

    def test_spark_fit_matches_serial(self):
        m = _gmm()
        data = m.sampler(1).sample(1000)
        serial = heterogeneous_fit(m, data, max_its=12, n_shards=1)
        spark = spark_fit(_SC, m, data, max_its=12, n_shards=4, depth=2)
        self.assertTrue(np.allclose(sorted(serial.w), sorted(spark.w), atol=1e-7))
        sm = sorted(c.mu for c in serial.components)
        km = sorted(c.mu for c in spark.components)
        self.assertTrue(np.allclose(sm, km, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
