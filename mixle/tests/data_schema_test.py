"""mixle.data.schema: real bugs found in audits and fixed here.

1. Boolean.coerce used bare bool(value), which is True for ANY non-empty string -- including the
   string "False" itself. Since this module's whole reason for existing is coercing string-typed
   values from CSV/SQL/Mongo connectors, this silently inverted the primary intended use case.
2. Schema.conform_record built the positional (non-dict) case via zip(self.fields, record), which
   silently stops at the shorter side instead of raising on a record/field-count mismatch -- exactly
   the "connectors silently get wrong" failure mode this module's own docstring says it exists to
   prevent.
3. Schema.for_model derives a single-field Vector/Nested schema for a multivariate-shaped model (e.g.
   a DiagonalGaussianDistribution), but conform_record's single-field disambiguation only bypassed the
   "N values for N fields" interpretation for non-(tuple/list/dict) records -- true for e.g. a raw
   np.ndarray, FALSE for a plain Python list. A dataset of raw lists (the natural shape for a
   multivariate observation, e.g. [[1.0, 2.0], [0.5, -0.5]]) was misread as "one record with too many
   top-level values" and every record raised, while the IDENTICAL data as a list of np.ndarray passed
   fine. Schema now carries a container_value flag (set by for_model when the derived schema has
   exactly one Vector/Nested field) that makes conform_record treat a raw list/tuple record as that
   one field's whole value -- see ForModelContainerValueTest below. A schema built by hand (as
   ConformRecordLengthTest does) still defaults container_value=False and keeps the stricter
   raise-on-ambiguity behavior.
4. _type_for checked the broad Discrete capability (finite-or-enumerable support) before ever
   considering a categorical class fallback, and Discrete is true for a CategoricalDistribution too
   (it has finite support) -- so EVERY discrete model, categorical or not, got Count. A
   CategoricalDistribution({"red": .5, "blue": .5}) therefore got a Count schema field, and schema
   conformance tried int("red"), rejecting the model's own valid support. _type_for now inspects a
   Discrete distribution's own declared support values (via its enumerator) and classifies them by
   what they actually are -- see CategoricalSchemaTypeTest below.
5. Count.coerce was bare int(value): it silently truncated fractional values (1.9 -> 1), silently
   accepted True/False as 1/0 (bool is an int subclass), and silently accepted negative values --
   turning a data problem into a wrong-but-plausible observation instead of an error. Count.coerce now
   validates exact integrality, sign, and finiteness before converting -- see CountCoerceTest below.
"""

import unittest

import numpy as np

from mixle.data.schema import Boolean, Categorical, Count, Field, Nested, Real, Schema, Vector
from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution
from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution


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


class ForModelContainerValueTest(unittest.TestCase):
    """Bug-2 regression: a for_model-derived single-Vector-field schema must accept a raw list/tuple
    record directly, not just an np.ndarray, and not raise the multi-field ambiguity error."""

    def test_for_model_on_a_multivariate_model_sets_container_value(self):
        dist = DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0])
        schema = Schema.for_model(dist)
        self.assertEqual(len(schema.fields), 1)
        self.assertIsInstance(schema.fields[0].type, Vector)
        self.assertTrue(schema.container_value)

    def test_raw_list_record_conforms_the_same_as_an_ndarray_record(self):
        dist = DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0])
        schema = Schema.for_model(dist)
        from_list = schema.conform_record([1.0, 2.0])
        from_array = schema.conform_record(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(from_list, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(from_list, from_array)

    def test_raw_tuple_record_also_conforms_directly(self):
        dist = DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0])
        schema = Schema.for_model(dist)
        result = schema.conform_record((1.0, 2.0))
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    def test_hand_built_schema_keeps_container_value_false_by_default(self):
        # Schema.for_model opts in; a schema a caller assembles directly (e.g. Schema((Field(...),)))
        # is unaffected -- container_value defaults to False, so ConformRecordLengthTest's stricter
        # raise-on-ambiguity behavior for a hand-built single Vector/Nested field is unchanged.
        s = Schema((Field("v", Vector(dim=None)),))
        self.assertFalse(s.container_value)
        with self.assertRaises(ValueError):
            s.conform_record([1.0, 2.0, 3.0])

    def test_nested_field_type_also_qualifies_for_container_value(self):
        inner = Schema((Field("x", Real()), Field("y", Real())))
        s = Schema((Field("n", Nested(inner)),), container_value=True)
        # a raw list record is handed whole to Nested.coerce (-> inner.conform_record), never split as
        # "N values for N fields" at the OUTER level -- there is only one outer field. Without
        # container_value=True this raises the exact same ambiguity error the Vector case did.
        result = s.conform_record([1.0, 2.0])
        self.assertEqual(result, inner.conform_record([1.0, 2.0]))
        self.assertEqual(result, (1.0, 2.0))
        s_default = Schema((Field("n", Nested(inner)),))  # container_value defaults False
        with self.assertRaises(ValueError):
            s_default.conform_record([1.0, 2.0])


class CategoricalSchemaTypeTest(unittest.TestCase):
    """Bug-4 (MXR-080-0054) regression: a categorical model's derived schema field must be Categorical,
    never Count, and its own valid support must round-trip through conform_record."""

    def test_categorical_distribution_gets_a_categorical_field_not_count(self):
        dist = CategoricalDistribution({"red": 0.5, "blue": 0.5})
        schema = Schema.for_model(dist)
        self.assertIsInstance(schema.fields[0].type, Categorical)
        self.assertNotIsInstance(schema.fields[0].type, Count)

    def test_categorical_labels_round_trip_through_conform_record(self):
        dist = CategoricalDistribution({"red": 0.5, "blue": 0.5})
        schema = Schema.for_model(dist)
        self.assertEqual(schema.conform_record("red"), "red")
        self.assertEqual(schema.conform_record("blue"), "blue")

    def test_poisson_a_genuine_count_model_still_gets_count(self):
        # the fix must not turn EVERY discrete model into Categorical -- a distribution whose declared
        # support values are actually integers must still resolve to Count.
        schema = Schema.for_model(PoissonDistribution(2.0))
        self.assertIsInstance(schema.fields[0].type, Count)
        self.assertEqual(schema.conform_record(3), 3)

    def test_bernoulli_a_binary_model_gets_boolean_not_count(self):
        # declared support {False, True} is neither a categorical label set nor an integer count.
        schema = Schema.for_model(BernoulliDistribution(0.5))
        self.assertIsInstance(schema.fields[0].type, Boolean)


class CountCoerceTest(unittest.TestCase):
    """Bug-5 (MXR-080-0055) regression: Count.coerce must validate exact integrality, sign, and
    finiteness instead of silently truncating/accepting via bare int(value)."""

    def test_fractional_values_are_rejected_not_truncated(self):
        with self.assertRaises(ValueError):
            Count().coerce(1.9)

    def test_bool_is_rejected_not_treated_as_zero_or_one(self):
        with self.assertRaises(ValueError):
            Count().coerce(True)
        with self.assertRaises(ValueError):
            Count().coerce(False)

    def test_negative_values_are_rejected(self):
        with self.assertRaises(ValueError):
            Count().coerce(-2)

    def test_nan_and_inf_are_rejected(self):
        with self.assertRaises(ValueError):
            Count().coerce(float("nan"))
        with self.assertRaises(ValueError):
            Count().coerce(float("inf"))
        with self.assertRaises(ValueError):
            Count().coerce(float("-inf"))

    def test_legitimate_counts_still_work(self):
        self.assertEqual(Count().coerce(42), 42)
        self.assertEqual(Count().coerce(0), 0)
        self.assertEqual(Count().coerce(5.0), 5)  # a whole-number float is still an exact count

    def test_numeric_strings_are_still_accepted(self):
        # connector input (CSV/SQL text columns) legitimately arrives as numeric strings.
        self.assertEqual(Count().coerce("42"), 42)

    def test_non_integral_numeric_strings_are_rejected(self):
        with self.assertRaises(ValueError):
            Count().coerce("1.9")


if __name__ == "__main__":
    unittest.main()
