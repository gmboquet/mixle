"""mixle.data.schema: two real bugs found in an audit and fixed here.

1. Boolean.coerce used bare bool(value), which is True for ANY non-empty string -- including the
   string "False" itself. Since this module's whole reason for existing is coercing string-typed
   values from CSV/SQL/Mongo connectors, this silently inverted the primary intended use case.
2. Schema.conform_record built the positional (non-dict) case via zip(self.fields, record), which
   silently stops at the shorter side instead of raising on a record/field-count mismatch -- exactly
   the "connectors silently get wrong" failure mode this module's own docstring says it exists to
   prevent.
"""

import unittest

import numpy as np

from mixle.data.schema import Boolean, Field, Real, Schema, Vector


class BooleanCoerceTest(unittest.TestCase):
    def test_the_string_false_coerces_to_false(self):
        self.assertIs(Boolean().coerce("False"), False)
        self.assertIs(Boolean().coerce("false"), False)
        self.assertIs(Boolean().coerce("FALSE"), False)
        self.assertIs(Boolean().coerce("0"), False)
        self.assertIs(Boolean().coerce("no"), False)

    def test_the_string_true_coerces_to_true(self):
        self.assertIs(Boolean().coerce("True"), True)
        self.assertIs(Boolean().coerce("true"), True)
        self.assertIs(Boolean().coerce("1"), True)
        self.assertIs(Boolean().coerce("yes"), True)

    def test_leading_trailing_whitespace_is_tolerated(self):
        self.assertIs(Boolean().coerce("  false  "), False)

    def test_an_unrecognized_string_raises_rather_than_guesses(self):
        with self.assertRaises(ValueError):
            Boolean().coerce("maybe")

    def test_real_python_bools_and_ints_still_work(self):
        self.assertIs(Boolean().coerce(True), True)
        self.assertIs(Boolean().coerce(False), False)
        self.assertIs(Boolean().coerce(1), True)
        self.assertIs(Boolean().coerce(0), False)


class ConformRecordLengthTest(unittest.TestCase):
    def _schema(self):
        return Schema((Field("a", Real()), Field("b", Real()), Field("c", Real())))

    def test_matching_length_still_works(self):
        s = self._schema()
        self.assertEqual(s.conform_record((1.0, 2.0, 3.0)), (1.0, 2.0, 3.0))

    def test_a_short_record_raises_instead_of_silently_dropping_fields(self):
        s = self._schema()
        with self.assertRaises(ValueError):
            s.conform_record((1.0, 2.0))

    def test_a_long_record_raises_instead_of_silently_dropping_values(self):
        s = self._schema()
        with self.assertRaises(ValueError):
            s.conform_record((1.0, 2.0, 3.0, 4.0))

    def test_a_generator_record_is_still_length_checked(self):
        s = self._schema()
        with self.assertRaises(ValueError):
            s.conform_record(x for x in (1.0, 2.0))

    def test_single_field_free_length_vector_schema_raises_instead_of_silently_truncating(self):
        # the specific real-world trap: a single-field Vector(dim=None) schema fed the raw vector as
        # the record used to pair only the FIRST element with the field and silently drop the rest
        # (components 2.0, 3.0 vanished with no error). This fix does not attempt to guess that the
        # whole list was meant as the one field's vector value -- that's a real ambiguity a caller
        # should resolve explicitly (e.g. wrap it: conform_record(([1.0, 2.0, 3.0],))) -- but it DOES
        # convert the silent data loss into a clear, loud error instead.
        s = Schema((Field("v", Vector(dim=None)),))
        with self.assertRaises(ValueError):
            s.conform_record([1.0, 2.0, 3.0])
        # the unambiguous, correctly-wrapped form still works.
        result = s.conform_record(([1.0, 2.0, 3.0],))
        np.testing.assert_array_equal(result[0], np.array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
