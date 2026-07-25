"""Focused regressions for the 0.8.0 data-contract audit repairs."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from mixle.data.core import LazySource, MaterializedSource
from mixle.data.exchangeability import exchangeability_check
from mixle.data.hashing import dataset_hash
from mixle.data.partition import encode_partitions, partition_records
from mixle.data.schema import Boolean, Count, Field, Real, Schema, Timestamp, Vector
from mixle.data.sources.pandas_source import dataframe_records
from mixle.data.sources.text_source import read_json
from mixle.data.structure import partially_exchangeable
from mixle.data.validate import check_dataset
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class _LengthAwareIterator:
    def __init__(self, values):
        self._values = iter(values)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._values)

    def __len__(self):
        return 3


class _RecordingEncoder:
    def __init__(self):
        self.calls = []

    def seq_encode(self, rows):
        self.calls.append(list(rows))
        return tuple(rows)


class SourceContractTest(unittest.TestCase):
    def test_length_aware_iterator_is_snapshotted_once(self):
        source = MaterializedSource(_LengthAwareIterator([1, 2, 3]))
        self.assertEqual(list(source.records()), [1, 2, 3])
        self.assertEqual(source.materialize(), [1, 2, 3])

    def test_records_and_materialize_share_conformed_identity(self):
        source = MaterializedSource(["1", "2"], schema=Schema((Field("x", Count()),)))
        self.assertEqual(list(source.records()), [1, 2])
        self.assertEqual(source.materialize(), [1, 2])

    def test_lazy_cache_is_not_mutable_and_length_hint_is_verified(self):
        source = LazySource(lambda: iter([1, 2, 3]), length=3)
        copy = source.materialize()
        copy.clear()
        self.assertEqual(list(source.records()), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "declared length"):
            LazySource(lambda: iter([1, 2, 3]), length=99).materialize()


class PartitionContractTest(unittest.TestCase):
    def test_empty_partitions_do_not_invoke_encoder(self):
        encoder = _RecordingEncoder()
        encoded = encode_partitions([1], encoder, num_chunks=3)
        self.assertEqual(encoded, [(1, (1,))])
        self.assertEqual(encoder.calls, [[1]])

    def test_group_placement_is_size_aware_and_deterministic(self):
        rows = [{"g": "a", "x": i} for i in range(100)]
        rows += [{"g": "b", "x": 0}, {"g": "c", "x": 0}]
        parts = partition_records(rows, partially_exchangeable("g"), 2)
        self.assertEqual(sorted(map(len, parts)), [2, 100])
        self.assertEqual(sum(any(row["g"] == "a" for row in part) for part in parts), 1)


class LogicalTypeContractTest(unittest.TestCase):
    def test_invalid_scalar_and_vector_observations_fail_closed(self):
        for value in (True, np.nan, np.inf):
            with self.subTest(real=value), self.assertRaises(ValueError):
                Real().coerce(value)
        for value in (None, 2, -1, np.nan):
            with self.subTest(boolean=value), self.assertRaises(ValueError):
                Boolean().coerce(value)
        with self.assertRaises(ValueError):
            Timestamp().coerce(np.datetime64("NaT"))
        for value in ([[1.0]], [], [1.0, np.nan]):
            with self.subTest(vector=value), self.assertRaises(ValueError):
                Vector().coerce(value)

    def test_schema_rejects_empty_and_duplicate_names(self):
        with self.assertRaises(ValueError):
            Schema(())
        with self.assertRaises(ValueError):
            Schema((Field("x", Real()), Field("x", Real())))


class HashAndValidationContractTest(unittest.TestCase):
    def test_hash_limit_is_exact_and_does_not_consume_an_unhashed_record(self):
        for bad in (-1, 1.5, True, np.nan):
            with self.subTest(max_records=bad), self.assertRaises(ValueError):
                dataset_hash(iter([1, 2, 3]), max_records=bad)
        stream = iter([1, 2, 3])
        dataset_hash(stream, max_records=1)
        self.assertEqual(next(stream), 2)

    def test_empty_dataset_is_not_certified(self):
        report = check_dataset(GaussianDistribution(0.0, 1.0), [])
        self.assertFalse(report.ok)
        self.assertEqual(report.n_checked, 0)


class ConnectorAndDiagnosticContractTest(unittest.TestCase):
    def test_json_requires_an_array_of_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "records.json")
            with open(path, "w") as handle:
                json.dump({"x": 1}, handle)
            with self.assertRaisesRegex(ValueError, "top-level array"):
                list(read_json(path).records())

    def test_dataframe_aliases_use_logical_names(self):
        import pandas as pd

        frame = pd.DataFrame({"physical": [7]})
        self.assertEqual(
            dataframe_records(frame, fields=[("logical", "physical")], as_dict=True),
            [{"logical": 7}],
        )

    def test_exchangeability_reads_only_its_declared_bound(self):
        consumed = []

        def records():
            for value in range(100):
                consumed.append(value)
                yield float(value)

        report = exchangeability_check(records(), n_perm=1, max_records=20)
        self.assertEqual(report.n_examined, 20)
        self.assertTrue(report.bounded)
        self.assertEqual(consumed, list(range(20)))


if __name__ == "__main__":
    unittest.main()
