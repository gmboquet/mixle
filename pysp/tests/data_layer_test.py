"""Data layer: DataSource / Schema / SampleStructure -- additive, structure-aware encoding."""

import unittest

import numpy as np

import pysp.stats as st
from pysp.data import (
    EXCHANGEABLE,
    IID,
    SEQUENTIAL,
    Categorical,
    Field,
    MaterializedSource,
    Real,
    Schema,
    Vector,
    partially_exchangeable,
)
from pysp.data.partition import partition_records
from pysp.stats import seq_encode


class DataLayerTest(unittest.TestCase):
    def test_datasource_is_bit_identical_to_list_striding(self):
        g = st.GaussianDistribution(0.0, 1.0)
        data = list(np.random.RandomState(0).randn(37))
        for nc in (1, 3, 8):
            a = seq_encode(data, model=g, num_chunks=nc)
            b = seq_encode(MaterializedSource(data, EXCHANGEABLE), model=g, num_chunks=nc)
            with self.subTest(num_chunks=nc):
                self.assertEqual(len(a), len(b))
                self.assertTrue(all(ca == cb and np.array_equal(pa, pb) for (ca, pa), (cb, pb) in zip(a, b)))

    def test_fit_through_datasource_matches_list(self):
        g = st.GaussianDistribution(2.0, 3.0)
        data = list(g.sampler(seed=1).sample(500))
        est = g.estimator()
        from pysp.inference.estimation import fit

        m_list = fit(data, est, max_its=5)
        m_src = fit(MaterializedSource(data, IID), est, max_its=5)
        self.assertAlmostEqual(m_list.mu, m_src.mu, places=10)

    def test_strideable_structures_partition_by_stride(self):
        recs = list(range(20))
        for s in (IID, EXCHANGEABLE, SEQUENTIAL):
            with self.subTest(structure=str(s)):
                self.assertEqual(partition_records(recs, s, 4), [recs[k::4] for k in range(4)])

    def test_partially_exchangeable_keeps_groups_intact(self):
        recs = [{"g": i % 4, "x": float(i)} for i in range(24)]
        parts = partition_records(recs, partially_exchangeable("g"), 4)
        for part in parts:
            self.assertLessEqual(len({r["g"] for r in part}), 1)  # at most one group per partition here
        self.assertEqual(sum(len(p) for p in parts), 24)  # nothing lost

    def test_schema_coercion_and_validation(self):
        sch = Schema((Field("c", Categorical(("a", "b"))), Field("x", Real()), Field("v", Vector(2))))
        out = sch.conform_record(("a", 3, [1, 2]))
        self.assertEqual(out[0], "a")
        self.assertEqual(out[1], 3.0)
        self.assertTrue(np.array_equal(out[2], np.array([1.0, 2.0])))
        with self.assertRaises(ValueError):
            sch.conform_record(("z", 1, [0, 0]))  # bad category

    def test_schema_for_model(self):
        self.assertEqual(repr(Schema.for_model(st.GaussianDistribution(0, 1)).fields[0].type), "Real")
        self.assertEqual(repr(Schema.for_model(st.PoissonDistribution(2.0)).fields[0].type), "Count")


if __name__ == "__main__":
    unittest.main()


class StructureCapabilityTest(unittest.TestCase):
    def test_supported_structures_inference(self):
        from pysp.data.structure import supported_structures

        self.assertEqual(
            supported_structures(st.GaussianDistribution(0, 1).estimator()), frozenset({"iid", "exchangeable"})
        )
        hmm = st.HiddenMarkovModelDistribution([st.GaussianDistribution(0, 1)], [1.0], [[1.0]])
        self.assertEqual(supported_structures(hmm.estimator()), frozenset({"sequential"}))

    def test_fit_warns_on_structure_mismatch_but_not_on_default(self):
        import warnings

        data = list(np.random.RandomState(0).randn(40))
        from pysp.inference.estimation import fit

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit(MaterializedSource(data, SEQUENTIAL), st.GaussianDistribution(0, 1).estimator(), max_its=2, out=None)
            self.assertTrue(any("structure" in str(x.message) for x in w))  # footgun caught
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit(data, st.GaussianDistribution(0, 1).estimator(), max_its=2, out=None)  # bare list
            self.assertFalse(any("structure" in str(x.message) for x in w))  # never on existing calls

    def test_strict_mode_raises(self):
        from pysp.data.structure import check_model_structure

        with self.assertRaises(ValueError):
            check_model_structure(st.GaussianDistribution(0, 1).estimator(), SEQUENTIAL, strict=True)


class ConnectorTest(unittest.TestCase):
    def test_open_csv_roundtrip_and_fit(self):
        import os
        import tempfile

        from pysp.data import Field, Real, Schema, open_source
        from pysp.inference.estimation import fit

        path = tempfile.mktemp(suffix=".csv")
        xs = np.random.RandomState(0).randn(400) * 2.0 + 5.0
        with open(path, "w") as fh:
            fh.write("x\n" + "\n".join(str(v) for v in xs))
        try:
            src = open_source("csv", path, columns=["x"], schema=Schema((Field("x", Real()),)))
            m = fit(src, st.GaussianDistribution(0, 1).estimator(), max_its=10, out=None)
            self.assertAlmostEqual(m.mu, 5.0, delta=0.3)
            self.assertAlmostEqual(m.sigma2, 4.0, delta=0.6)
        finally:
            os.unlink(path)

    def test_jsonl_records(self):
        import json
        import os
        import tempfile

        from pysp.data import open_source

        path = tempfile.mktemp(suffix=".jsonl")
        with open(path, "w") as fh:
            for i in range(5):
                fh.write(json.dumps({"a": i, "b": i * i}) + "\n")
        try:
            recs = list(open_source("jsonl", path).records())
            self.assertEqual(recs[2], {"a": 2, "b": 4})
        finally:
            os.unlink(path)

    def test_unknown_kind_and_missing_driver(self):
        from pysp.data import open_source
        from pysp.data.sources import sql_source

        with self.assertRaises(ValueError):
            open_source("does_not_exist", "x")
        if getattr(sql_source, "_sa", None) is None:  # only when sqlalchemy is absent
            with self.assertRaises(ImportError):
                sql_source.read_sql("sqlite://", "select 1").materialize()


class ColdImportTest(unittest.TestCase):
    """``import pysp.data`` must work standalone. The graph adapter (GraphDataEncoder) subclasses a
    contract under pysp.stats whose graph distributions import the adapter back, so eager re-export
    used to deadlock when pysp.data was imported before pysp.stats. Each case runs in a fresh
    interpreter because the test process has already imported everything."""

    def _run(self, snippet):
        import subprocess
        import sys

        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_import_pysp_data_cold(self):
        self._run("import pysp.data")

    def test_cold_graph_adapter_access_without_stats_first(self):
        self._run("from pysp.data import GraphDataEncoder; assert GraphDataEncoder().directed is False")

    def test_cold_star_import(self):
        self._run("from pysp.data import *; assert GraphObservation is not None and Schema is not None")
