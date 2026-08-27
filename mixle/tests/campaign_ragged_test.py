"""Rows of uneven width: a malformed table is named, real variable-length data is left alone.

``optimize(data)`` with no estimator used to answer a rectangular table carrying one bad row by
quietly abandoning the table reading and fitting a SequenceDistribution over a CategoricalDistribution
that memorized every distinct value in the table. Nothing said so -- no warning, ``numerical_repairs()``
empty -- and the resulting model scored every one of the training rows with the SAME constant, so
fit-then-score work (anomaly ranking, per-row scoring) came back constant instead of wrong-looking.

The two readings are genuinely ambiguous, and variable-length sequence data is a supported input to the
same call, so the fix is not "refuse ragged input". It is to decide on the arity evidence and say which
reading was taken: a dominant arity means a table with a malformed row, spread-out arities mean sequence
data, and the band in between is fitted as a sequence and discloses that it was.
"""

import unittest
import warnings

import numpy as np

from mixle.inference import optimize
from mixle.stats.compute.pdist import ContractError
from mixle.utils.automatic import analyze_structure, get_estimator, get_prototype
from mixle.utils.automatic.profiling import diagnose_ragged_rows


def _table(n=400, seed=99):
    """A rectangular two-field table: a measurement and an arm label."""
    rng = np.random.default_rng(seed)
    return [(float(rng.normal(10.0, 2.0)), "ctrl" if i % 2 else "treat") for i in range(n)]


def _variable_length_sequences(n=300, seed=7):
    """Genuine variable-length numeric sequences -- arities spread over a wide range."""
    rng = np.random.default_rng(seed)
    return [tuple(float(x) for x in rng.normal(0.0, 1.0, size=int(rng.integers(2, 12)))) for _ in range(n)]


def _fitted(rows, **kw):
    """A fit whose parameters are comparable across calls: estimators have no value repr, models do."""
    return optimize(rows, get_estimator(rows, **kw), max_its=5, seed=0, out=None)


class MalformedTableIsNamedTest(unittest.TestCase):
    """A dominant arity plus a few odd rows is a broken table, and the odd row gets named."""

    def _assert_names_row(self, rows, row_index, row_width, modal_width=2):
        with self.assertRaises(ContractError) as caught:
            optimize(rows, out=None)
        err = caught.exception
        self.assertIn("row %d" % row_index, err.path)
        self.assertEqual(err.expected, "a row of %d field(s)" % modal_width)
        self.assertEqual(err.actual, "a row of %d field(s)" % row_width)
        self.assertIn("missing or extra field", err.fix)

    def test_short_row_is_named_with_its_index(self):
        rows = _table()
        self._assert_names_row(rows[:300] + [(9.9,)] + rows[300:], row_index=300, row_width=1)

    def test_extra_field_is_named(self):
        # An unquoted comma inside a text field: the row gains a column instead of losing one.
        rows = _table()
        self._assert_names_row(rows[:120] + [(9.9, "ctrl", "trailing")] + rows[120:], row_index=120, row_width=3)

    def test_blank_line_is_named(self):
        # csv.reader emits a zero-length row for a blank line, including one at end of file.
        rows = _table()
        self._assert_names_row(rows[:50] + [()] + rows[50:], row_index=50, row_width=0)

    def test_list_rows_are_named_too(self):
        # csv/JSON pipelines hand over lists, not tuples; the same table is at stake.
        rows = [list(row) for row in _table()]
        self._assert_names_row(rows[:75] + [[9.9]] + rows[75:], row_index=75, row_width=1)

    def test_all_numeric_table_is_named(self):
        # A clean all-float table is read as a vector, not a record, but a lost field is still a
        # lost field: the reading must not silently become a sequence here either.
        rng = np.random.default_rng(3)
        rows = [(float(rng.normal(0, 1)), float(rng.normal(5, 1))) for _ in range(400)]
        self._assert_names_row(rows[:10] + [(1.0,)] + rows[10:], row_index=10, row_width=1)

    def test_the_error_offers_the_sequence_reading_as_the_other_remedy(self):
        rows = _table()
        with self.assertRaises(ContractError) as caught:
            optimize(rows[:300] + [(9.9,)] + rows[300:], out=None)
        self.assertIn("ragged='sequence'", caught.exception.fix)

    def test_get_prototype_and_get_estimator_agree_with_optimize(self):
        rows = _table()
        bad = rows[:200] + [(9.9,)] + rows[200:]
        for call in (lambda: get_estimator(bad), lambda: get_prototype(bad, seed=0)):
            with self.assertRaises(ContractError):
                call()


class SequenceDataIsUntouchedTest(unittest.TestCase):
    """Variable-length sequence data is a supported input and must fit exactly as it always has."""

    def _assert_fits_silently_as_sequence(self, rows):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(rows, out=None)
        self.assertEqual(type(model).__name__, "SequenceDistribution")
        self.assertEqual([str(w.message) for w in caught], [])
        return model

    def test_variable_length_numeric_sequences(self):
        model = self._assert_fits_silently_as_sequence(_variable_length_sequences())
        # Out-of-sample scoring stays finite: this is a real density over sequences, not a lookup
        # table of the training rows.
        self.assertTrue(np.isfinite(model.log_density((0.1, 0.2, 0.3))))

    def test_variable_length_word_lists(self):
        # The shape automatic_test.py's test_variable_length_lists_sequence_with_length_model pins.
        self._assert_fits_silently_as_sequence([["a", "b"], ["a"], ["b", "c", "a"]] * 40)

    def test_near_uniform_length_homogeneous_lists(self):
        # Homogeneous lists are read as sequences at a SINGLE arity too, so raggedness changes no
        # reading here and there is nothing to report -- even though one arity dominates.
        rows = [["a", "b"] for _ in range(200)] + [["a"]]
        self._assert_fits_silently_as_sequence(rows)

    def test_the_default_reading_matches_the_explicit_sequence_reading(self):
        rows = _variable_length_sequences()
        self.assertEqual(repr(_fitted(rows)), repr(_fitted(rows, ragged="sequence")))

    def test_clean_table_still_reads_as_a_table(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(_table(), out=None)
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertNotEqual(type(model).__name__, "SequenceDistribution")


class AmbiguousRaggednessIsDisclosedTest(unittest.TestCase):
    """A majority arity that is not overwhelming: fit it, but say which reading was taken."""

    def _ambiguous_rows(self):
        rows = _table()
        return rows[:280] + [(1.0,)] * 40 + rows[280:]

    def test_ambiguous_input_is_fitted_as_a_sequence_with_a_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(self._ambiguous_rows(), out=None)
        self.assertEqual(type(model).__name__, "SequenceDistribution")
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(messages), 1)
        self.assertIn("read as variable-length sequence data", messages[0])
        self.assertIn("row 280", messages[0])
        self.assertIn("ragged='sequence'", messages[0])

    def test_the_explicit_sequence_reading_silences_it_without_changing_the_model(self):
        rows = self._ambiguous_rows()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            quiet = _fitted(rows, ragged="sequence")
        self.assertEqual([str(w.message) for w in caught], [])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            noisy = _fitted(rows)
        self.assertEqual(repr(quiet), repr(noisy))


class DiagnosisSurfacesTest(unittest.TestCase):
    """The verdict is inspectable, and analyze_structure reports rather than refuses."""

    def test_analyze_structure_reports_the_malformed_row_instead_of_raising(self):
        rows = _table()
        profile = analyze_structure(rows[:150] + [(9.9,)] + rows[150:], pairwise=False)
        ragged = [line for line in profile.warnings if line.startswith("ragged rows:")]
        self.assertEqual(len(ragged), 1)
        self.assertIn("row 150", ragged[0])
        self.assertIn("malformed row", ragged[0])

    def test_diagnosis_reports_exact_counts_not_a_rounded_percentage(self):
        rows = _table(n=2000)
        diagnosis = diagnose_ragged_rows(rows[:1500] + [(9.9,)] + rows[1500:])
        self.assertTrue(diagnosis.malformed)
        self.assertEqual(diagnosis.modal_width, 2)
        self.assertEqual(diagnosis.row_index, 1500)
        self.assertEqual((diagnosis.modal_count, diagnosis.row_count), (2000, 2001))
        self.assertIn("2000 of 2001 rows", diagnosis.note())

    def test_shapes_the_arity_question_does_not_apply_to(self):
        rng = np.random.default_rng(11)
        for rows in (
            list(rng.normal(0.0, 1.0, 200)),  # scalars
            [{"kind": "a", "score": 1.0}, {"kind": "b"}] * 60,  # mappings: keyed, not positional
            [{"x", "y"}, {"y"}, {"x", "z"}] * 40,  # sets
            _table(),  # a table with no ragged row at all
        ):
            self.assertIsNone(diagnose_ragged_rows(rows))

    def test_ragged_argument_is_validated(self):
        with self.assertRaises(ValueError):
            get_estimator(_table(), ragged="strict")


class ScoringIsNotConstantTest(unittest.TestCase):
    """The user-visible damage the silent reading did, pinned so it cannot come back quietly."""

    def test_a_fitted_table_scores_its_own_rows_distinctly(self):
        rows = _table(n=600)
        model = optimize(rows, out=None)
        scores = {model.log_density(row) for row in rows}
        self.assertGreater(len(scores), 1)

    def test_the_sequence_reading_of_a_broken_table_is_reachable_only_on_request(self):
        # It is still a legal model -- the opt-out returns it -- but it is the degenerate one, so
        # nobody may land on it without asking: every training row scores identically.
        rows = _table(n=600)
        bad = rows[:300] + [(9.9,)] + rows[300:]
        model = optimize(bad, get_estimator(bad, ragged="sequence"), out=None)
        self.assertEqual(len({model.log_density(row) for row in rows}), 1)


if __name__ == "__main__":
    unittest.main()
