"""Identifier columns stop poisoning fits, and both spellings of "missing" mean one model.

T2-09a: a string column in which nearly every value is distinct (an ID or timestamp column -- column 1
of most real CSVs) resolved to ``IgnoredEstimator()``, whose default child is a point mass at ``None``.
That child scored every actual identifier at ``-inf``, so ``get_dpm_mixture`` over a table containing
such a column died with "EM did not produce a finite objective from its non-finite initial model" --
loud, but naming neither the column nor the remedy -- and ``get_prototype`` returned a model whose
log-density was ``-inf`` on the very rows it was inferred from. The trigger is the ratio k/n
(ID_DISTINCT_FRACTION), not a fixed cardinality. The repair freezes the empirical categorical of the
observed values inside the Ignored wrapper (finite per-row constant factor, samplable, never
re-estimated), and ``get_dpm_mixture`` refuses data with NO modelable field at all with a per-column
error naming the field, its cardinality, and the remedy.

T2-02: ``None`` and ``np.nan`` -- interchanged behind the caller's back by pandas float Series -- meant
different models: ``None`` got a fitted missingness rate and a generative wrapper, ``nan`` a
marginalized wrapper whose ``.p`` read 0.0 despite 20% of the data being missing, and whose sampler
raised. Both spellings now take the fitted-rate generative wrapper; the infinity sentinels stay
representational (they are values, not absences).

T2-09a addendum: the same -inf point-mass poison fired for every OTHER leaf the profiler freezes --
a scalar type it does not recognize (datetime64/Timestamp, i.e. any ``read_csv(parse_dates=...)``
column) and an ambiguous bool/numeric mix -- because those branches still built the bare ignored
estimator. They now take the same frozen empirical-categorical stand-in, and the per-column
``get_dpm_mixture`` refusal names the concrete type ("Timestamp field") instead of "unmodelable".
"""

import io
import math
import unittest

import numpy as np
import pytest

from mixle.stats.combinator.ignored import IgnoredEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalEstimator
from mixle.utils.automatic import get_dpm_mixture, get_estimator, get_prototype


def _id_table(n=200):
    """A two-field table whose first column is an all-distinct identifier."""
    return [("id_%05d" % i, float((i * 37) % 100) / 7.0) for i in range(n)]


def _fit(data):
    """Fit the automatically selected estimator by direct accumulation (no EM loop in the way)."""
    est = get_estimator(data)
    acc = est.accumulator_factory().make()
    for x in data:
        acc.update(x, 1.0, None)
    return est.estimate(None, acc.value())


class IdentifierColumnFitsTest(unittest.TestCase):
    """T2-09a: a high-cardinality string column yields a usable model when other fields carry one."""

    def test_dpm_fits_table_with_identifier_column(self):
        # Before: ValueError("EM did not produce a finite objective from its non-finite initial model.")
        model = get_dpm_mixture(_id_table(), max_components=3, max_its=5, out=io.StringIO())
        self.assertTrue(np.isfinite(model.log_density(_id_table()[0])))

    def test_prototype_scores_its_own_rows_finitely(self):
        rows = _id_table()
        proto = get_prototype(rows)
        # Before: -inf on every row the model was inferred from.
        self.assertTrue(np.isfinite(proto.log_density(rows[0])))

    def test_prototype_with_identifier_column_is_still_samplable(self):
        # Sampling worked before the repair (the ID field sampled as None); it must keep working.
        draws = get_prototype(_id_table()).sampler(seed=0).sample(2)
        self.assertEqual(len(draws), 2)
        for row in draws:
            self.assertIsInstance(row[0], str)

    def test_identifier_leaf_is_frozen_and_scores_observed_values(self):
        ids = ["id_%05d" % i for i in range(200)]
        est = get_estimator(ids)
        self.assertIsInstance(est, IgnoredEstimator)
        model = _fit(ids)
        # Every observed identifier scores its finite empirical mass -- the -inf point mass is gone.
        self.assertAlmostEqual(model.log_density("id_00000"), math.log(1.0 / 200.0))
        # Unseen labels keep the same finite-support -inf every auto-fitted categorical has.
        self.assertEqual(model.log_density("never_seen"), -np.inf)

    def test_integer_identifier_column_takes_the_same_repair(self):
        model = _fit([i * 1000 + 7 for i in range(200)])
        self.assertTrue(np.isfinite(model.log_density(7)))

    def test_lone_identifier_data_gets_a_per_column_error(self):
        ids = ["id_%05d" % i for i in range(200)]
        with self.assertRaises(ValueError) as caught:
            get_dpm_mixture(ids, max_components=3, max_its=5, out=io.StringIO())
        message = str(caught.exception)
        self.assertIn("no modelable field", message)
        self.assertIn("200 distinct value(s) in 200 observation(s)", message)
        self.assertIn("identifier-like", message)
        self.assertIn("Drop identifier-like fields", message)

    def test_all_identifier_table_names_each_column(self):
        rows = [("id_%05d" % i, "sess_%05d" % (i * 3)) for i in range(150)]
        with self.assertRaises(ValueError) as caught:
            get_dpm_mixture(rows, max_components=2, max_its=5, out=io.StringIO())
        message = str(caught.exception)
        self.assertIn("field 0", message)
        self.assertIn("field 1", message)
        self.assertIn("150 distinct value(s) in 150 observation(s)", message)


class UnrecognizedScalarLeafFitsTest(unittest.TestCase):
    """T2-09a addendum: datetime and bool/numeric-mix leaves take the same finite-scoring freeze.

    The end-to-end (composite/DPM) cases here use all-distinct frozen values on purpose: a frozen
    column carrying a DUPLICATE value still dies in ``seq_encode`` because
    ``IgnoredDataEncoder`` (mixle/stats/combinator/ignored.py) inherits the base ``row_count``
    guesser instead of delegating to its wrapped encoder, and the categorical payload
    ``(idx (n,), uniques (k,))`` defeats the guess whenever k != n. That is outside this package --
    handed off to the stats owner -- so the duplicate-carrying cases below stay on the direct
    accumulation path, which pins the estimator half of the repair.
    """

    def _timestamp_table(self, n=200):
        pd = pytest.importorskip("pandas")  # optional dep: base CI envs ship only numpy+scipy

        start = pd.Timestamp("2026-01-01")
        return [(start + pd.Timedelta(hours=i), float((i * 37) % 100) / 7.0) for i in range(n)]

    def test_datetime_column_in_a_table_fits_and_scores_finitely(self):
        rows = self._timestamp_table()
        # Before: ValueError("EM did not produce a finite objective from its non-finite initial model.")
        model = get_dpm_mixture(rows, max_components=3, max_its=5, out=io.StringIO())
        self.assertTrue(np.isfinite(model.log_density(rows[0])))

    def test_datetime_prototype_scores_its_own_rows_and_samples_timestamps(self):
        pd = pytest.importorskip("pandas")  # optional dep: base CI envs ship only numpy+scipy

        rows = self._timestamp_table()
        proto = get_prototype(rows)
        self.assertTrue(np.isfinite(proto.log_density(rows[0])))  # before: -inf on every row
        draws = proto.sampler(seed=0).sample(2)
        for row in draws:
            self.assertIsInstance(row[0], pd.Timestamp)  # before: the frozen field sampled None

    def test_datetime_only_data_error_names_the_concrete_type(self):
        rows = [r[0] for r in self._timestamp_table()]
        with self.assertRaises(ValueError) as caught:
            get_dpm_mixture(rows, max_components=2, max_its=5, out=io.StringIO())
        message = str(caught.exception)
        # Before: "a unmodelable field ... (not modelable by automatic profiling)" -- type unnamed.
        self.assertIn("Timestamp field", message)
        self.assertIn("identifier-like", message)
        self.assertIn("200 distinct value(s) in 200 observation(s)", message)

    def test_bool_numeric_mixed_leaf_is_frozen_and_scores_observed_values(self):
        col = [bool(i % 2) if i % 3 else i % 5 for i in range(120)]
        est = get_estimator(col)
        self.assertIsInstance(est, IgnoredEstimator)
        model = _fit(col)
        # Before: point mass at None -- every actual value scored -inf, poisoning any table fit.
        self.assertTrue(np.isfinite(model.log_density(True)))
        self.assertTrue(np.isfinite(model.log_density(3)))

    def test_duplicate_carrying_identifier_leaf_scores_duplicates_finitely(self):
        # 95% distinct: the k/n trigger regime with actual duplicates present.
        col = [("id_%05d" % i) if i < 190 else "dup" for i in range(200)]
        model = _fit(col)
        self.assertTrue(np.isfinite(model.log_density("dup")))
        self.assertTrue(np.isfinite(model.log_density("id_00000")))


class IdentifierTriggerIsARatioTest(unittest.TestCase):
    """The trigger is k/n, so the same label set flips reading with the row count -- by design."""

    def test_ninety_five_percent_distinct_reads_as_identifier(self):
        vals = ["v_%03d" % i for i in range(190)] + ["dup"] * 10  # k/n = 0.95 at n=200
        self.assertIsInstance(get_estimator(vals), IgnoredEstimator)

    def test_same_labels_at_lower_ratio_stay_categorical(self):
        vals = (["v_%03d" % i for i in range(190)] * 2) + ["dup"] * 20  # k/n = 0.475 at n=400
        self.assertIsInstance(get_estimator(vals), CategoricalEstimator)

    def test_moderate_cardinality_categorical_column_is_untouched(self):
        # The overreach hazard named in the finding: ordinary categorical columns keep fitting.
        vals = ["cat_%d" % (i % 10) for i in range(200)]
        self.assertIsInstance(get_estimator(vals), CategoricalEstimator)
        table = [("cat_%d" % (i % 10), float(i % 7)) for i in range(200)]
        model = get_dpm_mixture(table, max_components=3, max_its=5, out=io.StringIO())
        self.assertTrue(np.isfinite(model.log_density(table[0])))

    def test_below_the_count_floor_small_distinct_data_still_fits(self):
        ids = ["id_%05d" % i for i in range(50)]  # n < ID_MIN_COUNT: not identifier territory
        self.assertIsInstance(get_estimator(ids), CategoricalEstimator)
        model = get_dpm_mixture(ids, max_components=2, max_its=5, out=io.StringIO())
        self.assertTrue(np.isfinite(model.log_density(ids[0])))


class MissingSpellingsAgreeTest(unittest.TestCase):
    """T2-02: None and nan produce the same model shape, rate, and generativity."""

    BASE = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    def test_nan_and_none_fit_the_same_missingness_rate(self):
        with_none = _fit(self.BASE + [None, None])
        with_nan = _fit(self.BASE + [float("nan"), float("nan")])
        # Before: with_nan.p read 0.0 (no rate at all) while with_none.p read 0.2.
        self.assertAlmostEqual(with_none.p, 0.2)
        self.assertAlmostEqual(with_nan.p, 0.2)
        self.assertAlmostEqual(with_nan.log_density(float("nan")), math.log(0.2))
        self.assertAlmostEqual(with_none.log_density(None), math.log(0.2))
        # The observed-data submodel is identical either way.
        self.assertAlmostEqual(with_none.log_density(2.0), with_nan.log_density(2.0))

    def test_nan_containing_array_yields_a_generative_model(self):
        model = _fit(list(np.array(self.BASE + [np.nan, np.nan])))
        # Before: NonGenerativeOptionalError -- the marginalized wrapper cannot sample.
        draws = model.sampler(seed=1).sample(200)
        nan_share = sum(1 for d in draws if isinstance(d, float) and math.isnan(d)) / len(draws)
        self.assertGreater(nan_share, 0.05)  # the fitted 20% rate really generates missing values

    def test_pandas_series_coercion_of_none_does_not_change_the_model(self):
        pd = pytest.importorskip("pandas")  # optional dep: base CI envs ship only numpy+scipy

        as_list = _fit(self.BASE + [None, None])
        as_series = _fit(pd.Series(self.BASE + [None, None]))  # float dtype stores None as nan
        self.assertAlmostEqual(as_series.p, as_list.p)
        self.assertAlmostEqual(as_series.log_density(2.0), as_list.log_density(2.0))

    def test_infinity_stays_a_representational_value_not_a_missing_rate(self):
        model = _fit(self.BASE + [math.inf, math.inf])
        # +/-inf is a value a numeric field can carry, not an absence: no fitted rate.
        self.assertFalse(model.has_p)


if __name__ == "__main__":
    unittest.main()
