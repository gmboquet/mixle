"""Data adapters: the container a fit is handed decides nothing about the answer.

Pass 06 of the ten adversarial reviews of the 0.8.1 candidate (P06-F01..F10) found the default
``structure='auto'`` front door refusing a mixle ``DataSource``, crashing on a structured array and
on a masked one, reading a MultiIndex column label as a field alias, fitting a mapping of columns as
a categorical over its FIELD NAMES, and answering several malformed inputs with numpy's or Python's
own message instead of a named one. A closed parallel handle answered every request with a default
model and a zero score.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

import mixle
import mixle.stats as S
from mixle.data import as_source
from mixle.inference import optimize
from mixle.utils.optional_deps import HAS_PANDAS


def _quiet(callable_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return callable_()


SCALARS = [float(value) for value in np.random.RandomState(0).normal(size=200)]


class DataSourceTest(unittest.TestCase):
    """P06-F05: the automatic path refused what the explicit-estimator path accepted."""

    def test_every_route_fits_a_data_source_to_the_same_model(self):
        explicit = _quiet(lambda: optimize(as_source(SCALARS), S.GaussianEstimator(), max_its=3))
        without_structure = _quiet(lambda: optimize(as_source(SCALARS), max_its=3, structure="off"))
        automatic = _quiet(lambda: optimize(as_source(SCALARS), max_its=3))
        for other in (without_structure, automatic):
            self.assertAlmostEqual(other.mu, explicit.mu, places=12)
            self.assertAlmostEqual(other.sigma2, explicit.sigma2, places=12)

    def test_a_tagged_source_is_still_checked_against_the_model_it_feeds(self):
        from mixle.data.core import MaterializedSource
        from mixle.data.structure import SEQUENTIAL
        from mixle.inference.estimation import fit

        with self.assertRaises(ValueError):
            fit(
                MaterializedSource(SCALARS, SEQUENTIAL),
                S.GaussianDistribution(0.0, 1.0).estimator(),
                max_its=2,
                out=None,
            )


class ContainerShapeTest(unittest.TestCase):
    """P06-F06, F07, F09, F10: containers that iterate as something other than their records."""

    def test_a_masked_array_is_named_on_both_routes(self):
        masked = np.ma.masked_array(np.asarray(SCALARS), mask=(np.arange(200) % 10 == 0))
        for label, estimator in (("explicit", S.GaussianEstimator()), ("automatic", None)):
            with self.subTest(route=label):
                with self.assertRaises(ValueError) as caught:
                    optimize(masked, estimator, max_its=2)
                self.assertIn("masked array", str(caught.exception))

    def test_a_structured_array_fits_as_its_rows(self):
        structured = np.array([(1.0, "a"), (2.0, "b")] * 50, dtype=[("x", "f8"), ("k", "U1")])
        from_array = _quiet(lambda: optimize(structured, max_its=2))
        from_rows = _quiet(lambda: optimize(structured.tolist(), max_its=2))
        self.assertEqual(type(from_array).__name__, type(from_rows).__name__)

    def test_a_mapping_of_columns_fits_the_columns_not_their_names(self):
        table = {"x": SCALARS, "k": ["a", "b"] * 100}
        model = _quiet(lambda: optimize(table, max_its=2))
        self.assertNotIsInstance(model, S.CategoricalDistribution)
        rows = [(value, kind) for value, kind in zip(table["x"], table["k"])]
        as_rows = _quiet(lambda: optimize(rows, max_its=2))
        self.assertEqual(type(model).__name__, type(as_rows).__name__)

    def test_a_single_column_mapping_and_a_ragged_one(self):
        self.assertIsInstance(_quiet(lambda: optimize({"x": SCALARS}, max_its=2)), S.GaussianDistribution)
        with self.assertRaises(ValueError) as caught:
            optimize({"x": SCALARS, "k": ["a"]}, max_its=2)
        self.assertIn("lengths differ", str(caught.exception))

    def test_describe_routes_a_table_shaped_mapping_to_propose(self):
        self.assertIn("propose", mixle.describe({"x": SCALARS, "k": ["a", "b"] * 100}))
        self.assertIn("no catalogued capability", mixle.describe({"a": 1, "b": 2}))

    def test_the_malformed_containers_are_named(self):
        cases = (
            ("string", "hello world", "iterates as its individual characters"),
            ("bytes", b"hello world", "iterates as its individual characters"),
            ("numpy.matrix", np.matrix(np.asarray(SCALARS).reshape(-1, 1)), "numpy.matrix"),
            ("0-d array", np.array(3.0), "0-dimensional"),
            ("timedelta64", np.array([np.timedelta64(i, "s") for i in range(50)]), "timedelta64"),
        )
        for label, data, expected in cases:
            with self.subTest(container=label):
                with self.assertRaises(ValueError) as caught:
                    optimize(data, max_its=2)
                self.assertIn(expected, str(caught.exception))

    def test_a_timedelta_column_still_fits_in_its_numeric_form(self):
        raw = np.array([np.timedelta64(i, "s") for i in range(50)])
        self.assertIsNotNone(_quiet(lambda: optimize(raw.astype("int64").astype(float), max_its=2)))


@unittest.skipUnless(HAS_PANDAS, "pandas not installed; pip install mixle[pandas]")
class DataFrameLabelTest(unittest.TestCase):
    """P06-F02, F03, F09: labels read as alias pairs, and a duplicated label with no name."""

    @staticmethod
    def _frame(**columns):
        import pandas as pd

        return pd.DataFrame(columns)

    def test_multiindex_columns_are_not_read_as_name_source_pairs(self):
        import pandas as pd

        frame = pd.DataFrame(
            np.random.RandomState(0).normal(size=(100, 2)),
            columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")]),
        )
        model = _quiet(lambda: optimize(frame, max_its=2))
        self.assertIsInstance(model, S.CompositeDistribution)

    def test_a_repeated_column_label_is_named(self):
        import pandas as pd

        frame = pd.DataFrame(np.random.RandomState(0).normal(size=(50, 2)), columns=["x", "x"])
        for fields in (None, ["x"]):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError) as caught:
                    optimize(frame, max_its=2, fields=fields)
                self.assertIn("repeated column label", str(caught.exception))

    def test_an_aliased_record_estimator_fits_from_a_frame(self):
        frame = self._frame(x=[1.0, 2.0, 3.0] * 40, k=["a", "b", "c"] * 40)
        estimator = S.RecordEstimator(
            [S.field("mean", "x"), S.field("kind", "k")], [S.GaussianEstimator(), S.CategoricalEstimator()]
        )
        model = _quiet(lambda: optimize(frame, estimator, max_its=2))
        self.assertEqual(sorted(model.sources), ["k", "x"])
        self.assertTrue(np.isfinite(model.log_density({"x": 1.0, "k": "a"})))


class RankingInputTest(unittest.TestCase):
    """P06-F09 (f), (g): a None comparison row and a ragged ordering had no named message."""

    COMPARISONS = [[0, 1], [1, 2], [2, 0]] * 30

    def test_a_non_sequence_comparison_row_is_named_by_both_families(self):
        for label, estimator in (
            ("Thurstone-Mosteller", S.ThurstoneMostellerEstimator(3)),
            ("Bradley-Terry", S.BradleyTerryEstimator(3)),
        ):
            with self.subTest(family=label):
                with self.assertRaises(ValueError) as caught:
                    optimize(self.COMPARISONS + [None], estimator, max_its=2)
                message = str(caught.exception)
                self.assertIn("row 90", message)
                self.assertIn("NoneType", message)

    def test_a_ragged_ordering_is_named_by_both_ranking_families(self):
        orderings = [[0, 1, 2, 3]] * 40
        for label, estimator in (("Mallows", S.MallowsEstimator(4)), ("Plackett-Luce", S.PlackettLuceEstimator(4))):
            with self.subTest(family=label):
                with self.assertRaises(ValueError) as caught:
                    optimize(orderings + [[0, 1]], estimator, max_its=2)
                self.assertIn("4", str(caught.exception))
                self.assertNotIn("inhomogeneous", str(caught.exception))

    def test_clean_ranking_data_still_fits(self):
        self.assertIsNotNone(_quiet(lambda: optimize(self.COMPARISONS, S.BradleyTerryEstimator(3), max_its=2)))
        self.assertIsNotNone(_quiet(lambda: optimize([[0, 1, 2, 3]] * 40, S.MallowsEstimator(4), max_its=2)))


class ColumnArrayFitTest(unittest.TestCase):
    """P06-F08: an (n, 1) array scored one way and could not be fitted at all."""

    def test_the_encoder_refuses_the_shape_it_cannot_fit(self):
        column = np.asarray(SCALARS).reshape(-1, 1)
        law = S.GaussianDistribution(0.0, 1.0)
        with self.assertRaises(ValueError) as caught:
            law.dist_to_encoder().seq_encode(column)
        self.assertIn("one-dimensional sequence", str(caught.exception))
        with self.assertRaises(ValueError):
            optimize(column, S.GaussianEstimator(), max_its=2)
        self.assertIsNotNone(_quiet(lambda: optimize(np.ravel(column), S.GaussianEstimator(), max_its=2)))


if __name__ == "__main__":
    unittest.main()
