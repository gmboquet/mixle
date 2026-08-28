"""T1-03: T4-01's dirty-cell -inf memorization failure, reopened for object/str-typed numeric columns.

T4-01 fixed "one non-numeric cell in an otherwise-numeric column silently demotes the column to a
memorization table scoring -inf on unseen values" for the case where pandas keeps the column
float64-typed with NaN for the bad cell -- i.e. the column arrives as a mix of NATIVE Python floats
and a native string. But pandas coerces the WHOLE column to object/str dtype whenever a non-numeric
string that is not one of its own NA markers appears in a numeric column read from CSV/text, and in
that shape every value -- the genuinely numeric ones included -- arrives as a Python ``str``. Before
this fix, DatumNode's type counting only recognized number/string mixes by native Python type, so a
column that was uniformly ``str``-typed (however numeric its text) was indistinguishable from an
identifier or categorical column: it fit a frozen lookup table over the observed strings and scored
every unseen numeric-as-string value ``-inf``, with ``fit_provenance()`` still reporting
``converged=True`` and ``repairs=()``. This was not an edge case -- swept over cardinality, an
all-numeric-as-string column with NO dirty cell at all failed 100% of held-out queries at every n,
whether auto-typing picked ``CategoricalDistribution`` (low n) or ``IgnoredDistribution`` (high n).

The fix teaches DatumNode's counting -- and the typed-dispatch-mixture router built from it -- to
read a decimal-point/exponent-bearing string ("46.8037") as the float it names, exactly as it
already reads a native Python float. A bare digit string ("02139") is deliberately left alone: see
``parse_numeric_text`` in ``mixle.utils.automatic.factories`` for why.
"""

import math
import unittest

import numpy as np
import pytest

from mixle.inference import optimize
from mixle.stats.combinator.select import SelectDistribution
from mixle.utils.automatic import analyze_structure


def _numeric_text_column(seed=11, n=300, loc=50.0, scale=10.0):
    """The exact real-world trigger: values that are numeric, but arrive typed as ``str``.

    This is what ``list(df['x'])`` hands back once pandas has coerced a column to object dtype --
    not a synthetic string built to look like a float, but the same round-tripped text a user's own
    ``str(value)`` or a CSV round-trip would produce.
    """
    return [str(round(float(x), 4)) for x in np.random.RandomState(seed).normal(loc, scale, n)]


class PureNumericStringColumnTest(unittest.TestCase):
    """No dirty cell at all: every value is numeric text, and the old code still broke on all of it."""

    def setUp(self):
        self.train = _numeric_text_column()
        self.held_out = [str(round(float(x), 4)) for x in np.random.RandomState(11).normal(50, 10, 350)[300:]]

    def test_the_column_fits_a_continuous_family_not_a_lookup_table(self):
        model = optimize(self.train, out=None)
        self.assertEqual(type(model).__name__, "GaussianDistribution")

    def test_held_out_numeric_strings_score_finitely(self):
        # Before: every one of these 50 held-out numeric-as-string values scored -inf.
        model = optimize(self.train, out=None)
        n_neg_inf = sum(1 for x in self.held_out if model.log_density(x) == -np.inf)
        self.assertEqual(n_neg_inf, 0)

    def test_the_fit_matches_the_native_float_column(self):
        """The text and the numbers it names must fit the identical model -- str(x) is not a
        different measurement from x."""
        native = optimize([float(v) for v in self.train], out=None)
        text = optimize(self.train, out=None)
        self.assertEqual(text.mu, native.mu)
        self.assertEqual(text.sigma2, native.sigma2)
        self.assertEqual(text.log_density("50.0"), native.log_density(50.0))

    def test_cardinality_sweep_all_finite(self):
        """The failure was the DEFAULT outcome at every n, not an identifier-cardinality edge case --
        pin it at a handful of column sizes spanning the auto-typing thresholds."""
        for n in (20, 50, 120, 300):
            with self.subTest(n=n):
                train = _numeric_text_column(seed=3, n=n)
                held_out = [str(round(float(x), 4)) for x in np.random.RandomState(4).normal(50, 10, 30)]
                model = optimize(train, out=None)
                n_neg_inf = sum(1 for x in held_out if model.log_density(x) == -np.inf)
                self.assertEqual(n_neg_inf, 0, msg="n=%d" % n)

    def test_fit_provenance_is_no_longer_the_only_thing_that_looked_healthy(self):
        """Before, converged=True/repairs=() described a model that was silently wrong on unseen
        data; now the same healthy provenance describes a model that actually is."""
        model = optimize(self.train, out=None)
        provenance = model.fit_provenance()
        self.assertTrue(provenance.converged)
        self.assertEqual(provenance.repairs, ())
        self.assertTrue(math.isfinite(model.log_density(self.held_out[0])))


class PandasObjectDtypeColumnTest(unittest.TestCase):
    """The literal trigger: a CSV read where one straggler forces the whole column to object dtype."""

    def test_a_csv_dirty_cell_that_pandas_stringifies_the_whole_column_over(self):
        pd = pytest.importorskip("pandas")  # optional dep: base CI envs ship only numpy+scipy
        import io

        vals = [round(float(x), 4) for x in np.random.RandomState(11).normal(50, 10, 300)]
        csv_text = "x\n" + "\n".join(str(v) for v in vals) + "\nunknown\n"
        df = pd.read_csv(io.StringIO(csv_text))
        # pandas' default dtype for this coerced column varies by version (plain "object" on older
        # pandas, a dedicated "str"/StringDtype on newer ones); what actually matters is that every
        # element -- including the 300 numeric rows -- comes back as a Python str, which is the
        # shape that broke auto-inference.
        column = list(df["x"])
        self.assertTrue(all(isinstance(v, str) for v in column))  # including the 300 numeric rows

        model = optimize(column, out=None)
        self.assertIsInstance(model, SelectDistribution)

        held_out = [str(round(float(x), 4)) for x in np.random.RandomState(11).normal(50, 10, 350)[300:]]
        n_neg_inf = sum(1 for x in held_out if model.log_density(x) == -np.inf)
        self.assertEqual(n_neg_inf, 0)

        # The genuinely non-numeric straggler still scores through its own branch, not silently
        # dropped or merged into the numeric one.
        self.assertTrue(math.isfinite(model.log_density("unknown")))
        self.assertEqual(model.log_density("never-observed-string"), -np.inf)

    def test_profile_names_the_column_correctly(self):
        pd = pytest.importorskip("pandas")
        import io

        vals = [round(float(x), 4) for x in np.random.RandomState(11).normal(50, 10, 300)]
        csv_text = "x\n" + "\n".join(str(v) for v in vals) + "\nunknown\n"
        df = pd.read_csv(io.StringIO(csv_text))
        column = list(df["x"])
        profile = analyze_structure(column, pairwise=False).fields[0]
        self.assertEqual(profile.recommendation, "typed_mixture")


class OverreachIsScopedToParseableNumericTextTest(unittest.TestCase):
    """The verifier's demolition attempts: things that must NOT change behavior."""

    def test_ordinary_native_float_column_is_bit_identical(self):
        """A column with no strings at all never touches the new text-parsing path."""
        vals = [round(float(x), 4) for x in np.random.RandomState(11).normal(50, 10, 300)]
        model = optimize(vals, out=None)
        self.assertEqual(type(model).__name__, "GaussianDistribution")
        self.assertEqual(model.mu, 49.997748666666666)
        self.assertEqual(model.sigma2, 93.32376314029874)

    def test_genuinely_categorical_string_and_int_mix_is_untouched(self):
        data = ["low", 2, "high", 3] * 40
        model = optimize(data, out=None)
        self.assertIn("Categorical", type(model).__name__)
        profile = analyze_structure(data, pairwise=False).fields[0]
        self.assertEqual(profile.recommendation, "categorical")

    def test_bare_digit_identifier_text_is_not_folded_into_the_numeric_branch(self):
        """ "02139" is left alone: parsing it as an int would drop the leading zero, and the
        identifier/categorical path it would otherwise take is not proven to accept a raw string
        query the way every continuous family does. The column stays exactly what it was before
        this fix -- an identifier, frozen and finite-scoring, not a claimed Gaussian over ids."""
        codes = ["%05d" % i for i in range(300)]
        model = optimize(codes + ["N/A"], out=None)
        self.assertEqual(type(model).__name__, "IgnoredDistribution")
        self.assertTrue(math.isfinite(model.log_density(codes[0])))

    def test_t4_01_native_dirty_cell_case_is_unaffected(self):
        """The original T4-01 shape -- native floats plus a string marker -- must still resolve to
        the identical model this fix must not touch."""
        train = [round(float(x), 4) for x in np.random.RandomState(11).normal(50, 10, 300)]
        dirty = optimize(train + ["N/A"], out=None)
        spelled_missing = optimize(train + [float("nan")], out=None)
        self.assertEqual(dirty.log_density(50.0), spelled_missing.log_density(50.0))
        self.assertEqual(
            dirty.fit_provenance().final_objective,
            spelled_missing.fit_provenance().final_objective,
        )


if __name__ == "__main__":
    unittest.main()
