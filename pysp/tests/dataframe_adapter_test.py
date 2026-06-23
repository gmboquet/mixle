import io
import unittest

import numpy as np
import pytest

pd = pytest.importorskip("pandas")  # pandas is an optional extra; skip this suite if it is not installed

from pysp.data import dataframe_records, seq_encode_dataframe
from pysp.inference.estimation import optimize
from pysp.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    GaussianEstimator,
    RecordDistribution,
    RecordEstimator,
    field,
    seq_encode,
    seq_log_density_sum,
)


class DataFrameAdapterTestCase(unittest.TestCase):
    def test_single_field_dataframe_records_are_scalars(self):
        df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "unused": ["a", "b", "c"]})
        self.assertEqual(dataframe_records(df, fields="x"), [0.0, 1.0, 2.0])

    def test_composite_dataframe_records_preserve_field_order(self):
        df = pd.DataFrame({"label": ["b", "a"], "x": [2.0, 1.0]})
        self.assertEqual(dataframe_records(df, fields=["x", "label"]), [(2.0, "b"), (1.0, "a")])

    def test_seq_encode_dataframe_matches_list_encoding_for_scalar_model(self):
        df = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0]})
        model = GaussianDistribution(0.0, 1.0)
        records = dataframe_records(df, fields="x")

        enc_df = seq_encode_dataframe(df, fields="x", model=model)
        enc_list = seq_encode(records, model=model)
        np.testing.assert_allclose(
            seq_log_density_sum(enc_df, model), seq_log_density_sum(enc_list, model), rtol=0, atol=0
        )

    def test_seq_encode_dataframe_matches_list_encoding_for_composite_model(self):
        df = pd.DataFrame(
            {
                "x": [-1.0, 0.5, 1.5, 2.0],
                "label": ["a", "b", "a", "c"],
            }
        )
        model = CompositeDistribution(
            (
                GaussianDistribution(0.5, 1.7),
                CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}),
            )
        )
        fields = ["x", "label"]
        records = dataframe_records(df, fields=fields)

        enc_df = seq_encode_dataframe(df, fields=fields, model=model)
        enc_list = seq_encode(records, model=model)
        np.testing.assert_allclose(
            seq_log_density_sum(enc_df, model), seq_log_density_sum(enc_list, model), rtol=0, atol=0
        )

    def test_missing_dataframe_field_is_error(self):
        df = pd.DataFrame({"x": [1.0]})
        with self.assertRaises(KeyError):
            dataframe_records(df, fields=["x", "missing"])

    def test_dataframe_records_can_return_dict_rows(self):
        df = pd.DataFrame({"x": [0.0, 1.0], "label": ["a", "b"]})
        self.assertEqual(
            dataframe_records(df, fields=["x", "label"], as_dict=True),
            [{"x": 0.0, "label": "a"}, {"x": 1.0, "label": "b"}],
        )

    def test_record_distribution_reads_dataframe_by_name(self):
        df = pd.DataFrame(
            {
                "x": [-1.0, 0.5, 1.5, 2.0],
                "label": ["a", "b", "a", "c"],
            }
        )
        model = RecordDistribution(
            {
                "x": GaussianDistribution(0.5, 1.7),
                "label": CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}),
            }
        )
        records = dataframe_records(df, fields=["x", "label"], as_dict=True)

        enc_df = seq_encode_dataframe(df, model=model)
        enc_list = seq_encode(records, model=model)
        np.testing.assert_allclose(
            seq_log_density_sum(enc_df, model), seq_log_density_sum(enc_list, model), rtol=0, atol=0
        )

    def test_record_distribution_can_reuse_dataframe_source_column(self):
        df = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0]})
        model = RecordDistribution(
            {
                field("x_left", source="x"): GaussianDistribution(0.0, 1.0),
                field("x_right", source="x"): GaussianDistribution(1.0, 2.0),
            }
        )

        enc = seq_encode_dataframe(df, model=model)[0][1]
        records = dataframe_records(df, fields=["x"], as_dict=True)
        expected = np.asarray([model.log_density(row) for row in records])
        np.testing.assert_allclose(model.seq_log_density(enc), expected, rtol=1.0e-12, atol=1.0e-12)

    def test_record_estimator_accepts_dataframe_fields(self):
        df = pd.DataFrame({"x": [-1.0, 0.0, 1.0], "y": [0.5, 1.5, 2.5]})
        est = RecordEstimator(
            {
                field("x_view", source="x"): GaussianDistribution(0.0, 1.0).estimator(),
                field("y_view", source="y"): GaussianDistribution(0.0, 1.0).estimator(),
            }
        )

        enc = seq_encode_dataframe(df, estimator=est)
        acc = est.accumulator_factory().make()
        acc.seq_update(enc[0][1], np.ones(len(df)), None)
        model = est.estimate(len(df), acc.value())

        self.assertIsInstance(model, RecordDistribution)
        self.assertTrue(np.isfinite(model.log_density({"x": 0.1, "y": 1.2})))

    def test_optimize_accepts_dataframe_fields(self):
        df = pd.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0], "unused": list("abcde")})
        data = dataframe_records(df, fields="x")
        start = GaussianDistribution(3.0, 5.0)
        est = GaussianEstimator()

        from_df = optimize(
            df, est, fields="x", prev_estimate=start, max_its=1, delta=None, out=io.StringIO(), print_iter=100
        )
        from_list = optimize(data, est, prev_estimate=start, max_its=1, delta=None, out=io.StringIO(), print_iter=100)

        self.assertAlmostEqual(from_df.mu, from_list.mu, places=12)
        self.assertAlmostEqual(from_df.sigma2, from_list.sigma2, places=12)

    def test_optimize_accepts_dataframe_fields_with_resources(self):
        from pysp.utils.parallel.planner import Resources

        df = pd.DataFrame({"x": np.linspace(-2.0, 2.0, 20), "unused": np.arange(20)})
        data = dataframe_records(df, fields="x")
        start = GaussianDistribution(1.0, 3.0)
        est = GaussianEstimator()

        placed = optimize(
            df,
            est,
            fields="x",
            prev_estimate=start,
            max_its=1,
            delta=None,
            resources=Resources.local(num_cpus=2),
            sub_chunks=2,
            out=io.StringIO(),
            print_iter=100,
        )
        serial = optimize(data, est, prev_estimate=start, max_its=1, delta=None, out=io.StringIO(), print_iter=100)

        self.assertAlmostEqual(placed.mu, serial.mu, places=12)
        self.assertAlmostEqual(placed.sigma2, serial.sigma2, places=12)


if __name__ == "__main__":
    unittest.main()
