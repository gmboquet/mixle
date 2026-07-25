"""Contracts for public PPL maturity labels and lossless production/distributed adapters."""

import unittest
from unittest.mock import patch

import numpy as np

import mixle.ppl as ppl
from mixle.inference.production.monitor import Monitor
from mixle.inference.spark_executor import spark_em_step, spark_fit


class _Accumulator:
    def __init__(self):
        self.total = 0.0

    def from_value(self, value):
        self.total = float(value)
        return self

    def combine(self, value):
        self.total += float(value)

    def value(self):
        return self.total


class _Factory:
    @staticmethod
    def make():
        return _Accumulator()


class _Estimator:
    @staticmethod
    def accumulator_factory():
        return _Factory()

    @staticmethod
    def estimate(count, value):
        return int(count), float(value)


class _RDD:
    def __init__(self, values):
        self.values = list(values)

    def map(self, function):
        return _RDD([function(value) for value in self.values])

    def treeReduce(self, function, depth=2):
        values = list(self.values)
        while len(values) > 1:
            values.append(function(values.pop(0), values.pop(0)))
        return values[0]

    def cache(self):
        return self

    def unpersist(self):
        return None


class _SparkContext:
    @staticmethod
    def parallelize(values, partitions):
        if partitions != len(values):
            raise AssertionError("partition receipt differs from the shard count")
        return _RDD(values)


class _ScoreModel:
    @staticmethod
    def log_density(value):
        return -float(value) ** 2


class PublicPPLBoundaryContractTest(unittest.TestCase):
    def test_diagnostics_are_stable_and_scaling_laws_are_explicitly_experimental(self):
        for name in ("split_rhat", "bulk_ess", "tail_ess", "convergence_diagnostics", "psis_loo"):
            self.assertIn(name, ppl.STABLE_EXPORTS)
            self.assertIn(name, ppl.__all__)
        for name in ("fit_scaling_law", "allocate_compute", "ScalingLawFit"):
            self.assertIn(name, ppl.EXPERIMENTAL_EXPORTS)
            self.assertIn(name, ppl.__all__)
        self.assertTrue(ppl.STABLE_EXPORTS.isdisjoint(ppl.EXPERIMENTAL_EXPORTS))


class SparkObservationContractTest(unittest.TestCase):
    def test_generator_is_materialized_once_and_every_observation_is_counted(self):
        data = (value for value in [1.0, 2.0, 3.0, 4.0])
        with patch(
            "mixle.inference.spark_executor._shard_estep",
            side_effect=lambda estimator, model, shard: (len(shard), sum(shard)),
        ):
            result = spark_em_step(_SparkContext(), _Estimator(), object(), data, n_shards=2)
        self.assertEqual(result, (4, 10.0))

    def test_empty_partitions_and_invalid_controls_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            spark_em_step(_SparkContext(), _Estimator(), object(), [], n_shards=1)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            spark_em_step(_SparkContext(), _Estimator(), object(), [1.0], n_shards=2)
        with self.assertRaises((TypeError, ValueError)):
            spark_fit(_SparkContext(), object(), [1.0], max_its=0, n_shards=1)


class ProductionMonitorContractTest(unittest.TestCase):
    def test_one_shot_current_batch_is_processed_once_and_counted_exactly(self):
        monitor = Monitor(_ScoreModel(), object(), [0.0, 0.1, -0.1])
        consumed = []

        def current():
            for value in [0.0, 0.2, -0.2, 0.1]:
                consumed.append(value)
                yield value

        result = monitor.update(current(), retrain=False)
        self.assertEqual(consumed, [0.0, 0.2, -0.2, 0.1])
        self.assertEqual(result["processed_count"], 4)
        self.assertEqual(result["report"].processed_count, 4)
        self.assertEqual(monitor.history[-1]["n_current"], 4)
        self.assertTrue(np.isfinite(result["report"].score["mean_loglik_current"]))

    def test_empty_batches_and_invalid_thresholds_fail_closed(self):
        with self.assertRaises(ValueError):
            Monitor(_ScoreModel(), object(), [])
        with self.assertRaises(ValueError):
            Monitor(_ScoreModel(), object(), [0.0], ks_threshold=1.1)
        monitor = Monitor(_ScoreModel(), object(), [0.0])
        with self.assertRaises(ValueError):
            monitor.check(iter(()))


if __name__ == "__main__":
    unittest.main()
