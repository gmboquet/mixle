"""Campaign-three regressions: pandas missing sentinels, and the ignored-encoder row-count contract.

Two defects filed together (T2-1 blocking, T2-2 major) share one story.

* ``pd.NA``/``pd.NaT`` are pandas' own missing markers, and every nullable extension dtype
  (``Float64``/``Int64``/``boolean``/``string``, i.e. what ``convert_dtypes()`` and a parquet round
  trip produce) uses them. They are neither ``None`` nor a float, so they used to reach the profiler
  as ordinary *values*: a nullable numeric column was silently fitted as a uniform categorical
  memorizer with ``<NA>`` as one more level, and every unseen number then scored ``-inf`` -- while
  the same column spelled with ``NaN`` fitted a continuous family with a missing-value probability.
* Those frozen columns then encoded to a categorical ``(indices, levels)`` payload whose two members
  have different lengths, which the ``IgnoredDataEncoder`` could not report a row count for. That
  crashed ``optimize``/``Model.fit`` with an internal ``NotImplementedError`` -- for pandas datetime
  columns and identifier string columns too, which are the frozen-column path's whole purpose.

The pandas-free half of this file must keep running on a core install (numpy + scipy only), so
``pytest.importorskip("pandas")`` is called inside the tests that need pandas rather than at module
scope.
"""

import datetime as dt
import unittest

import numpy as np
import pytest

from mixle.inference.estimation import optimize
from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    IgnoredDistribution,
    OptionalDistribution,
    seq_encode,
)

_TIMESTAMPS = [dt.datetime(2020, 1, 1) + dt.timedelta(days=d) for d in (0, 1, 0, 2, 1, 0, 3, 1)]


class IgnoredEncoderRowCountTestCase(unittest.TestCase):
    """T2-2: the ignored wrapper delegates encoding, so it must delegate ``row_count`` too."""

    def test_row_count_delegates_when_levels_and_rows_differ(self):
        # Three levels, eight rows: the payload is (indices[8], levels[3]), which the abstract
        # default could not read a leading count from -- it raised NotImplementedError naming
        # IgnoredDataEncoder. Only the child encoder knows this layout.
        rows = ["a", "b", "a", "c", "b", "a", "c", "b"]
        model = IgnoredDistribution(CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}))
        encoder = model.dist_to_encoder()

        payload = encoder.seq_encode(rows)

        self.assertEqual(encoder.row_count(payload), len(rows))
        self.assertEqual(encoder.row_count(payload), model.dist.dist_to_encoder().row_count(payload))

    def test_seq_encode_conserves_rows_for_an_ignored_child(self):
        rows = ["a", "b", "a", "c", "b", "a", "c", "b"]
        model = IgnoredDistribution(CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}))

        enc = seq_encode(rows, model=model)

        self.assertEqual([count for count, _ in enc], [len(rows)])

    def test_optimize_fits_a_frozen_datetime_column(self):
        # A datetime column is modeled by the frozen "identifier" stand-in on purpose; the missing
        # row_count made that documented path unfittable outright.
        model = optimize(_TIMESTAMPS, out=None)

        self.assertIsInstance(model, IgnoredDistribution)
        self.assertTrue(np.isfinite(model.log_density(_TIMESTAMPS[0])))

    def test_optimize_fits_a_record_carrying_a_frozen_column(self):
        rows = list(zip(_TIMESTAMPS, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))

        model = optimize(rows, out=None)

        self.assertIsInstance(model, CompositeDistribution)
        self.assertTrue(np.isfinite(model.log_density(rows[0])))


class PandasMissingSentinelTestCase(unittest.TestCase):
    """T2-1: ``pd.NA``/``pd.NaT`` mean missing everywhere ``NaN``/``None`` do."""

    @staticmethod
    def _pandas():
        return pytest.importorskip("pandas")  # pandas is an optional extra

    def test_records_replace_pandas_na_with_the_column_marker(self):
        # UPDATED for campaign four, T2-02. This test used to assert ``records[1] is None``, i.e.
        # that pd.NA always becomes None. That rule made the fitted missing_value a function of the
        # caller's dtype backend (Float64 -> None, float64 -> NaN), and the two models then rejected
        # each other's frames. pd.NA now becomes the marker the COLUMN's kind determines -- NaN for
        # a numeric column, which is what the numpy-backed spelling of this same column produces.
        pd = self._pandas()
        from mixle.data import dataframe_records

        df = pd.DataFrame({"x": pd.Series([1.5, None, 2.5], dtype="Float64")})

        records = dataframe_records(df, fields="x")

        self.assertTrue(np.isnan(records[1]))
        self.assertEqual([records[0], records[2]], [1.5, 2.5])
        # The point of the change: the numpy-backed spelling of the same column agrees cell for cell.
        nan_frame = pd.DataFrame({"x": np.array([1.5, np.nan, 2.5])})
        self.assertEqual(dataframe_records(nan_frame, fields="x")[0::2], [1.5, 2.5])
        self.assertTrue(np.isnan(dataframe_records(nan_frame, fields="x")[1]))

    def test_records_replace_pandas_nat_with_none(self):
        pd = self._pandas()
        from mixle.data import dataframe_records

        df = pd.DataFrame({"t": pd.to_datetime(["2020-01-01", None, "2020-01-03"])})

        records = dataframe_records(df, fields="t")

        self.assertIsNone(records[1])

    def test_dict_and_tuple_rows_replace_pandas_na_with_the_column_marker(self):
        # UPDATED for campaign four, T2-02: the expected marker is now per column (NaN for the
        # numeric field, None for the string field) rather than None for both. See
        # ``test_records_replace_pandas_na_with_the_column_marker`` for why.
        pd = self._pandas()
        from mixle.data import dataframe_records

        df = pd.DataFrame(
            {
                "x": pd.Series([1.5, None], dtype="Float64"),
                "s": pd.Series(["a", None], dtype="string"),
            }
        )

        tuples = dataframe_records(df, fields=["x", "s"])
        self.assertEqual(tuples[0], (1.5, "a"))
        self.assertTrue(np.isnan(tuples[1][0]))
        self.assertIsNone(tuples[1][1])

        dicts = dataframe_records(df, fields=["x", "s"], as_dict=True)
        self.assertEqual(dicts[0], {"x": 1.5, "s": "a"})
        self.assertTrue(np.isnan(dicts[1]["x"]))
        self.assertIsNone(dicts[1]["s"])

    def test_ordinary_records_are_untouched(self):
        pd = self._pandas()
        from mixle.data import dataframe_records

        df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "label": ["a", "b", "c"]})

        self.assertEqual(dataframe_records(df, fields="x"), [0.0, 1.0, 2.0])
        self.assertEqual(dataframe_records(df, fields=["x", "label"]), [(0.0, "a"), (1.0, "b"), (2.0, "c")])

    def test_nan_is_not_rewritten_to_none(self):
        # NaN is already the float spelling of missing; rewriting it would silently change the
        # fitted OptionalDistribution's missing_value sentinel out from under float data.
        from mixle.data.sources.pandas_source import normalize_pandas_missing

        self._pandas()  # the sentinel table is only populated once pandas is importable
        value = float("nan")

        self.assertIs(normalize_pandas_missing(value), value)
        self.assertIs(normalize_pandas_missing(3.5), 3.5)

    def test_nested_pandas_na_inside_an_object_cell_is_normalized(self):
        # An object column's cells can be containers, so the vectorized ``isna()`` shortcut (which
        # is a per-cell test) must not be trusted to clear one.
        pd = self._pandas()
        from mixle.data import dataframe_records

        df = pd.DataFrame({"pair": pd.Series([(1.0, 2.0), (3.0, pd.NA)], dtype=object)})

        self.assertEqual(dataframe_records(df, fields="pair"), [(1.0, 2.0), (3.0, None)])

    def test_missing_scan_is_skipped_for_columns_that_cannot_carry_a_sentinel(self):
        # The scan gate: a numpy float/int/bool column cannot hold pd.NA, so it is never walked.
        # UPDATED for campaign four, T2-02: the gate now takes the column's gap PLAN as well, since
        # a numpy numeric column may also need skipping because its only possible gap (NaN) is
        # already the marker that column canonicalizes to. Renamed with it.
        from mixle.data.sources.pandas_source import _column_gap_plan, _column_may_hold_gap

        def gate(column):
            return _column_may_hold_gap(column, _column_gap_plan(column))

        self.assertFalse(gate(np.array([1.0, np.nan])))
        self.assertFalse(gate(np.array([1, 2, 3])))
        self.assertTrue(gate(np.array(["a", None], dtype=object)))
        self.assertTrue(gate(object()))  # a duck-typed column fails open

    def test_nullable_column_without_missing_values_round_trips(self):
        pd = self._pandas()
        from mixle.data import dataframe_records

        df = pd.DataFrame({"x": pd.Series([1.5, 2.5, 3.5], dtype="Float64")})

        self.assertEqual(dataframe_records(df, fields="x"), [1.5, 2.5, 3.5])

    def test_nullable_float_column_fits_like_the_nan_spelling(self):
        pd = self._pandas()
        observed = [12.5, 40.1, 33.0, 55.2, 41.7, 38.4, 29.9]
        na_frame = pd.DataFrame({"x": pd.Series(observed[:3] + [None] + observed[3:], dtype="Float64")})
        nan_frame = pd.DataFrame({"x": np.array(observed[:3] + [float("nan")] + observed[3:], dtype=float)})

        from_na = optimize(na_frame, fields="x", out=None)
        from_nan = optimize(nan_frame, fields="x", out=None)

        # The defect: <NA> became an eighth category and every unseen number scored -inf.
        self.assertIsInstance(from_na, OptionalDistribution)
        self.assertTrue(np.isfinite(from_na.log_density(37.0)))
        # UPDATED for campaign four, T2-02: this line used to read ``log_density(None)``, because
        # the nullable spelling used to fit ``missing_value=None`` while the NaN spelling fitted
        # ``missing_value=nan``. That divergence WAS the defect; a numeric column's marker is now
        # NaN whichever way pandas spelled the gap, so the two fits are interchangeable.
        self.assertTrue(np.isfinite(from_na.log_density(float("nan"))))
        self.assertIs(type(from_na.dist), type(from_nan.dist))
        self.assertEqual(repr(from_na.missing_value), repr(from_nan.missing_value))
        self.assertAlmostEqual(from_na.log_density(37.0), from_nan.log_density(37.0), places=12)

    def test_nullable_string_column_stays_categorical(self):
        # The boundary the numeric fix must not cross: a genuine categorical column keeps its
        # categorical fit, with missingness carried by the Optional wrapper instead of becoming a
        # category of its own, exactly as the None spelling behaves.
        pd = self._pandas()
        df = pd.DataFrame({"s": pd.Series(["a", "b", "a", None, "b", "a", "c", "b"], dtype="string")})

        model = optimize(df, fields="s", out=None)

        self.assertIsInstance(model, OptionalDistribution)
        self.assertIsInstance(model.dist, CategoricalDistribution)
        self.assertNotIn(pd.NA, model.dist.pmap)
        self.assertTrue(np.isfinite(model.log_density("a")))
        self.assertTrue(np.isfinite(model.log_density(None)))
        self.assertEqual(model.log_density("unseen-level"), -np.inf)

    def test_nullable_integer_column_fits_a_count_model(self):
        pd = self._pandas()
        df = pd.DataFrame({"k": pd.Series([1, 2, 3, None, 5, 6, 7, 8], dtype="Int64")})

        model = optimize(df, fields="k", out=None)

        self.assertIsInstance(model, OptionalDistribution)
        self.assertNotIsInstance(model.dist, IgnoredDistribution)
        self.assertTrue(np.isfinite(model.log_density(4)))
        # UPDATED for campaign four, T2-02: an Int64 column is a column of NUMBERS, so its gap now
        # canonicalizes to NaN -- the marker the same column stored as float64 (pandas upcasts an
        # int column with a gap) has always produced. This line used to read ``log_density(None)``.
        self.assertTrue(np.isfinite(model.log_density(float("nan"))))

    def test_nullable_boolean_column_fits(self):
        pd = self._pandas()
        df = pd.DataFrame({"b": pd.Series([True, False, True, None, False, True, False, True], dtype="boolean")})

        model = optimize(df, fields="b", out=None)

        self.assertIsInstance(model, OptionalDistribution)
        self.assertTrue(np.isfinite(model.log_density(True)))
        self.assertTrue(np.isfinite(model.log_density(None)))

    def test_datetime_column_with_nat_models_missingness(self):
        pd = self._pandas()
        stamps = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-01", None, "2020-01-02", "2020-01-01", "2020-01-03", "2020-01-02"]
        )
        df = pd.DataFrame({"t": stamps})

        model = optimize(df, fields="t", out=None)

        self.assertIsInstance(model, OptionalDistribution)
        self.assertTrue(np.isfinite(model.log_density(stamps[0])))
        self.assertTrue(np.isfinite(model.log_density(None)))

    def test_convert_dtypes_frame_fits_every_column(self):
        # The filed T2-2 reproduction: one frame carrying all four nullable dtypes with a pd.NA.
        pd = self._pandas()
        df = pd.DataFrame(
            {
                "v": [12.5, 40.1, 33.0, None, 55.2, 41.7, 38.4, 29.9],
                "k": [1, 2, 3, 4, 5, 6, 7, 8],
                "b": [True, False, True, True, False, True, False, True],
                "s": ["a", "b", "a", "c", "b", "a", "c", "b"],
            }
        ).convert_dtypes()
        df.loc[4, "b"] = pd.NA

        model = optimize(df, out=None)

        self.assertIsInstance(model, CompositeDistribution)
        self.assertIsInstance(model.dists[0], OptionalDistribution)
        self.assertTrue(np.isfinite(model.log_density((37.0, 4, True, "a"))))

    def test_propose_then_fit_accepts_a_nullable_frame(self):
        pd = self._pandas()
        from mixle import propose

        df = pd.DataFrame({"x": pd.Series([12.5, 40.1, 33.0, None, 55.2, 41.7, 38.4, 29.9], dtype="Float64")})

        model = propose(df, fit=True)

        self.assertIsInstance(model.fitted, OptionalDistribution)
        self.assertTrue(np.isfinite(model.fitted.log_density(37.0)))


if __name__ == "__main__":
    unittest.main()
