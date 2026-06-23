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
