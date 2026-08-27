"""Campaign-four T2-02: the fitted missing-value sentinel must not depend on the pandas dtype.

The defect. ``OptionalDistribution`` matches its ``missing_value`` by identity, and the pandas
adapter used to decide that marker from the dtype BACKEND: a gap arrived as ``NaN`` from a numpy
column and as ``pd.NA`` from a nullable one, and only pandas' own sentinels were rewritten (always
to ``None``). So the same 300 values fitted ``missing_value=nan`` as ``float64`` and
``missing_value=None`` as ``Float64``, and the two models rejected each other's frames -- the whole
batch, in both directions, at ``Model.evaluate(DataFrame)``. A ``convert_dtypes()`` call or a
parquet round trip between fit and serve was enough to trigger it.

The mirror image on the string side is the reason this file tests every column kind at once rather
than the numeric one that was filed. Under pandas 3 a plain ``str`` column spells its gap ``NaN``
while its ``convert_dtypes()`` twin spells it ``pd.NA``; rewriting only pandas' sentinels made one
frame fit ``missing_value=nan`` and the other ``missing_value=None``, and cross-scoring then
returned ``mean_log_density = -inf`` for every row with NO error at all. Fixing the numeric
direction alone would have left a silent wrong answer behind a loud one.

The fix canonicalizes a gap onto the marker the column's KIND determines -- ``NaN`` for a column of
numbers, ``None`` for everything else -- which is the marker that column's numpy-backed spelling
has always produced. ``pandas`` is an optional extra, so every test that needs it calls
``pytest.importorskip`` rather than importing at module scope (a bare import fails collection for
the whole core CI job).
"""

import math
import unittest

import numpy as np
import pytest

from mixle.data.sources.pandas_source import column_records, dataframe_records, seq_encode_dataframe
from mixle.inference.estimation import optimize
from mixle.stats import (
    CategoricalEstimator,
    CompositeEstimator,
    GaussianEstimator,
    OptionalEstimator,
)
from mixle.stats.compute.pdist import ContractError

_N = 300
_MISSING_EVERY = 10


def _pandas():
    return pytest.importorskip("pandas")  # pandas is an optional extra


def _numeric_column():
    rng = np.random.default_rng(11)
    values = rng.normal(50.0, 7.4, size=_N)
    values[::_MISSING_EVERY] = np.nan
    return values


def _label_column():
    labels = np.array(["basic", "plus", "pro"] * (_N // 3), dtype=object)
    labels[::_MISSING_EVERY] = None
    return labels


def _is_nan(value):
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


class ColumnKindMarkerTestCase(unittest.TestCase):
    """A gap takes the marker its column's KIND determines, never the dtype backend's spelling."""

    def test_numeric_column_gap_is_nan_in_every_dtype_backend(self):
        # Before: Float64/pd.NA -> None while float64/np.nan -> nan, from the same values.
        pd = _pandas()
        values = [1.5, None, 2.5, 3.0]
        spellings = {
            "float64": pd.Series([1.5, np.nan, 2.5, 3.0]),
            "Float64": pd.Series(values, dtype="Float64"),
            "object": pd.Series(values, dtype=object),
        }
        try:
            spellings["pyarrow"] = pd.Series(values, dtype="float64[pyarrow]")
        except Exception:  # noqa: BLE001 - pyarrow is an extra of an extra
            pass

        for name, column in spellings.items():
            with self.subTest(dtype=name):
                records = dataframe_records(pd.DataFrame({"x": column}))
                self.assertTrue(_is_nan(records[1]), f"{name} gap was {records[1]!r}")
                self.assertEqual([records[0], records[2], records[3]], [1.5, 2.5, 3.0])

    def test_nullable_integer_column_is_numeric_too(self):
        # An int column with a gap is UPCAST to float64/NaN by numpy-backed pandas, so Int64's
        # pd.NA has to land on the same marker or the two spellings of one column diverge again.
        pd = _pandas()

        records = dataframe_records(pd.DataFrame({"k": pd.Series([1, None, 3], dtype="Int64")}))

        self.assertTrue(_is_nan(records[1]))
        self.assertEqual([records[0], records[2]], [1, 3])

    def test_non_numeric_column_gap_is_none_in_every_dtype_backend(self):
        # The SILENT half. pandas 3 gives a plain str column a NaN gap and its nullable twin a
        # pd.NA gap; the mismatched markers cross-scored to -inf per row with no error.
        pd = _pandas()
        labels = ["a", None, "c"]
        spellings = {
            "object": pd.Series(labels, dtype=object),
            "str": pd.Series(labels).astype("str"),
            "string": pd.Series(labels, dtype="string"),
            "boolean": pd.Series([True, None, False], dtype="boolean"),
            "datetime": pd.to_datetime(["2020-01-01", None, "2020-01-03"]).to_series().reset_index(drop=True),
        }

        for name, column in spellings.items():
            with self.subTest(dtype=name):
                records = dataframe_records(pd.DataFrame({"s": column}))
                self.assertIsNone(records[1], f"{name} gap was {records[1]!r}")

    def test_frame_and_convert_dtypes_twin_mark_every_gap_alike(self):
        pd = _pandas()
        frame = pd.DataFrame({"v": _numeric_column(), "s": _label_column()})

        plain = dataframe_records(frame)
        nullable = dataframe_records(frame.convert_dtypes())

        self.assertEqual(len(plain), len(nullable))
        for index, (left, right) in enumerate(zip(plain, nullable, strict=True)):
            with self.subTest(row=index):
                self.assertEqual(_is_nan(left[0]), _is_nan(right[0]))
                self.assertIs(left[1] is None, right[1] is None)

    def test_tuple_and_dict_rows_take_a_marker_per_column(self):
        pd = _pandas()
        frame = pd.DataFrame(
            {"v": pd.Series([1.5, None], dtype="Float64"), "s": pd.Series(["a", None], dtype="string")}
        ).convert_dtypes()

        tuples = dataframe_records(frame)
        dicts = dataframe_records(frame, as_dict=True)

        self.assertTrue(_is_nan(tuples[1][0]))
        self.assertIsNone(tuples[1][1])
        self.assertTrue(_is_nan(dicts[1]["v"]))
        self.assertIsNone(dicts[1]["s"])

    def test_single_column_helper_agrees_with_the_frame(self):
        # column_records is the one-column half of the same rule: a Series must not canonicalize
        # differently from the identical column inside a frame.
        pd = _pandas()
        frame = pd.DataFrame({"v": _numeric_column(), "s": _label_column()})

        for name in ("v", "s"):
            with self.subTest(column=name):
                from_frame = dataframe_records(frame, fields=name)
                from_column = column_records(frame[name])
                self.assertEqual(len(from_frame), len(from_column))
                for left, right in zip(from_frame, from_column, strict=True):
                    self.assertEqual(_is_nan(left), _is_nan(right))
                    self.assertIs(left is None, right is None)

    def test_column_records_is_the_documented_route_for_a_bare_series(self):
        # A bare Series does NOT yet route through the canonicalizer (the profiler's and the
        # encoder's Series branches have to move together -- see column_records' docstring), so its
        # docstring points callers at this one-call route instead. Pin that the route works, both
        # spellings landing on one sentinel, so the stated workaround cannot rot.
        pd = _pandas()
        values = _numeric_column()
        plain = pd.Series(values)

        from_plain = optimize(column_records(plain), out=None)
        from_nullable = optimize(column_records(plain.convert_dtypes()), out=None)

        self.assertEqual(repr(from_plain.missing_value), repr(from_nullable.missing_value))
        self.assertTrue(_is_nan(from_plain.missing_value))
        self.assertTrue(np.isfinite(from_plain.log_density(column_records(plain.convert_dtypes())[1])))


class CrossDtypeScoringTestCase(unittest.TestCase):
    """The filed reproduction: fit on one dtype family, serve the other."""

    @staticmethod
    def _frames():
        pd = _pandas()
        frame = pd.DataFrame({"v": _numeric_column()})
        return frame, frame.convert_dtypes()

    def test_fitted_missing_value_does_not_depend_on_the_dtype_backend(self):
        plain, nullable = self._frames()

        from_plain = optimize(plain, out=None)
        from_nullable = optimize(nullable, out=None)

        self.assertEqual(repr(from_plain.missing_value), repr(from_nullable.missing_value))
        self.assertTrue(_is_nan(from_plain.missing_value))

    def test_each_models_evaluate_accepts_both_dtype_spellings(self):
        # Before: the two off-diagonal cells raised ContractError on the WHOLE batch.
        from mixle import propose

        plain, nullable = self._frames()
        models = {"plain": propose(plain, fit=True), "nullable": propose(nullable, fit=True)}

        scores = {}
        for model_name, model in models.items():
            for frame_name, frame in (("plain", plain), ("nullable", nullable)):
                with self.subTest(model=model_name, frame=frame_name):
                    result = model.evaluate(frame)
                    self.assertEqual(result["n"], _N)
                    self.assertTrue(np.isfinite(result["total_log_density"]))
                    scores[(model_name, frame_name)] = result["total_log_density"]

        self.assertEqual(len(set(scores.values())), 1, scores)

    def test_string_column_cross_scoring_is_not_silently_minus_inf(self):
        # The mirror-image defect, which produced a finite-looking API call and an -inf answer.
        from mixle import propose

        pd = _pandas()
        plain = pd.DataFrame({"s": pd.Series(_label_column()).astype("str")})
        nullable = plain.convert_dtypes()
        models = {"plain": propose(plain, fit=True), "nullable": propose(nullable, fit=True)}

        for model_name, model in models.items():
            for frame_name, frame in (("plain", plain), ("nullable", nullable)):
                with self.subTest(model=model_name, frame=frame_name):
                    result = model.evaluate(frame)
                    self.assertTrue(
                        np.isfinite(result["mean_log_density"]),
                        f"{model_name} model scored the {frame_name} frame {result['mean_log_density']}",
                    )

    def test_seq_encode_dataframe_round_trips_both_spellings_through_one_model(self):
        from mixle.stats import seq_log_density

        plain, nullable = self._frames()
        model = optimize(plain, out=None)

        totals = []
        for frame in (plain, nullable):
            encoded = seq_encode_dataframe(frame, model=model)
            totals.append(float(np.sum(seq_log_density(encoded, model))))

        self.assertTrue(np.isfinite(totals[0]))
        self.assertEqual(totals[0], totals[1])


class CanonicalizationBoundaryTestCase(unittest.TestCase):
    """What the rewrite must NOT touch, and what it must not walk."""

    def test_a_nan_inside_a_container_cell_is_the_cells_own_data(self):
        # A cell that is a vector observation carries its own NaNs; those are the wrapped field's
        # data (a missing coordinate), not this column's gap, and rewriting them on the strength of
        # the OUTER column's dtype would change values the adapter was never asked about.
        pd = _pandas()
        frame = pd.DataFrame({"pair": pd.Series([(1.0, 2.0), (3.0, float("nan")), None], dtype=object)})

        records = dataframe_records(frame, fields="pair")

        self.assertEqual(records[0], (1.0, 2.0))
        self.assertEqual(records[1][0], 3.0)
        self.assertTrue(_is_nan(records[1][1]))
        self.assertIsNone(records[2])

    def test_a_nested_pandas_sentinel_is_still_normalized(self):
        pd = _pandas()
        frame = pd.DataFrame({"pair": pd.Series([(1.0, 2.0), (3.0, pd.NA)], dtype=object)})

        self.assertEqual(dataframe_records(frame, fields="pair"), [(1.0, 2.0), (3.0, None)])

    def test_a_column_with_no_seen_values_keeps_the_conservative_marker(self):
        # An empty or all-missing object column has nothing to characterize, so it is NOT declared
        # numeric: guessing NaN there would rewrite a column nobody has looked at.
        pd = _pandas()
        frame = pd.DataFrame({"blank": pd.Series([None, None, None], dtype=object)})

        self.assertEqual(dataframe_records(frame, fields="blank"), [None, None, None])

    def test_a_mixed_object_column_is_not_declared_numeric(self):
        pd = _pandas()
        frame = pd.DataFrame({"mixed": pd.Series([1.5, None, "n/a"], dtype=object)})

        self.assertIsNone(dataframe_records(frame, fields="mixed")[1])

    def test_a_column_that_cannot_answer_dtype_kind_gets_the_legacy_plan(self):
        # Fail OPEN, not closed: a duck-typed frame's column is not characterizable, so pandas'
        # own sentinels are mapped to None (the pre-0.8.0 behaviour) and nothing else is touched.
        from mixle.data.sources.pandas_source import _column_gap_plan, _column_may_hold_gap

        class _Opaque:
            pass

        self.assertEqual(_column_gap_plan(_Opaque()), (None, False, False))
        self.assertTrue(_column_may_hold_gap(_Opaque(), (None, False, False)))

    def test_a_plain_numeric_frame_skips_the_rewrite_pass_entirely(self):
        # The perf gate: a numpy float/int column's only possible gap is already its marker, so no
        # per-row Python pass is planned for it at all.
        from mixle.data.sources.pandas_source import (
            _column_gap_plan,
            _column_may_hold_gap,
            _selection_gap_plans,
        )

        pd = _pandas()
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1, 2, 3]})

        self.assertIsNone(_selection_gap_plans(frame, ["a", "b"]))
        self.assertFalse(_column_may_hold_gap(np.array([1.0, np.nan]), _column_gap_plan(np.array([1.0, np.nan]))))


class MissingValueDocumentationTestCase(unittest.TestCase):
    """The remedies ``optimize`` now names have to keep working, and keep being named."""

    def test_optimize_docstring_names_the_remedy_that_works(self):
        # The NaN rejection message names three remedies and omits this one; a tester found it only
        # by reading a constructor signature (campaign four, T1/T2 documentation findings).
        doc = optimize.__doc__

        self.assertIn("OptionalEstimator", doc)
        self.assertIn('missing_value=float("nan")', doc)
        self.assertIn("missing_value=float('nan')", doc)

    def test_optimize_docstring_says_seq_encode_needs_records(self):
        doc = optimize.__doc__

        self.assertIn("seq_encode", doc)
        self.assertIn("dataframe_records", doc)
        self.assertIn("seq_encode_dataframe", doc)

    def test_seq_encode_dataframe_docstring_names_its_record_counterpart(self):
        self.assertIn("seq_encode", seq_encode_dataframe.__doc__)
        self.assertIn("dataframe_records", seq_encode_dataframe.__doc__)

    def test_the_documented_nan_remedy_actually_fits_composite_rows(self):
        # Pins the docstring's factual claim so it cannot rot: the default None sentinel does NOT
        # accept NaN, and naming the sentinel explicitly is what makes the tabular case fit.
        rng = np.random.default_rng(5)
        values = list(rng.normal(0.0, 1.0, 60))
        for index in range(3, 60, 10):
            values[index] = float("nan")
        rows = [(value, "a" if position % 2 else "b") for position, value in enumerate(values)]

        with self.assertRaises(ContractError):
            optimize(
                rows, CompositeEstimator([OptionalEstimator(GaussianEstimator()), CategoricalEstimator()]), out=None
            )

        model = optimize(
            rows,
            CompositeEstimator(
                [OptionalEstimator(GaussianEstimator(), missing_value=float("nan")), CategoricalEstimator()]
            ),
            out=None,
        )

        self.assertTrue(np.isfinite(model.log_density((0.5, "a"))))

    def test_the_documented_dataframe_conversion_actually_encodes(self):
        pd = _pandas()
        from mixle.stats import seq_encode

        frame = pd.DataFrame({"v": _numeric_column(), "s": _label_column()})
        model = optimize(frame, out=None)

        with self.assertRaises(ContractError):
            seq_encode(frame, model=model)

        self.assertEqual(len(seq_encode(dataframe_records(frame), model=model)), 1)
        self.assertEqual(len(seq_encode_dataframe(frame, model=model)), 1)


if __name__ == "__main__":
    unittest.main()
