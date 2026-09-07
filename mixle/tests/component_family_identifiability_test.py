"""A component of a mixture must not be flexible enough to be the mixture.

Two repairs of one shape, both surfaced by shipped notebooks (P08-F09, P07-F03/P08-F10):

* The automatic detector adds a per-field 2-component Gaussian mixture whenever a column looks
  multimodal. That is right for a standalone fit and wrong for the COMPONENTS of an outer mixture:
  the outer mixture is what models multimodality, and a mixture inside a component lets one
  component absorb several regimes. A Dirichlet-process mixture built that way stopped shrinking its
  truncation and clustered a four-segment corpus worse than its numeric fields alone.
* Writing ``free`` in a slot the family's estimator does not move is not an error -- the estimator
  is documented as fixed there -- but it looked like a fit, and a notebook reported a Student-t's
  default degrees of freedom as an estimate.
"""

from __future__ import annotations

import io
import unittest
import warnings

import numpy as np

from mixle.utils.automatic import get_dpm_mixture
from mixle.utils.automatic.profiling import get_estimator


def _bimodal(seed=0, n=400):
    rng = np.random.RandomState(seed)
    return [float(v) for v in np.concatenate([rng.normal(-6.0, 1.0, n // 2), rng.normal(6.0, 1.0, n // 2)])]


class UnimodalLeavesTest(unittest.TestCase):
    def test_a_multimodal_column_gets_a_mixture_by_default(self):
        estimator = get_estimator(_bimodal())
        self.assertIn("Mixture", type(estimator).__name__)

    def test_and_a_unimodal_family_when_the_caller_says_it_is_a_component(self):
        estimator = get_estimator(_bimodal(), unimodal_leaves=True)
        self.assertNotIn("Mixture", type(estimator).__name__)

    def test_the_flag_reaches_the_fields_of_a_record(self):
        rows = [(value, "a" if index % 2 else "b") for index, value in enumerate(_bimodal())]
        default = get_estimator(rows)
        component = get_estimator(rows, unimodal_leaves=True)
        self.assertIn("Mixture", type(default.estimators[0]).__name__)
        self.assertNotIn("Mixture", type(component.estimators[0]).__name__)

    def test_a_unimodal_column_is_unaffected(self):
        rows = [float(v) for v in np.random.RandomState(0).normal(0.0, 1.0, 400)]
        self.assertEqual(type(get_estimator(rows)).__name__, type(get_estimator(rows, unimodal_leaves=True)).__name__)

    def test_the_dp_factory_builds_components_not_little_mixtures(self):
        model = get_dpm_mixture(_bimodal(), rng=np.random.RandomState(1), max_components=4, out=io.StringIO())
        for component in model.components:
            self.assertNotIn("Mixture", type(component).__name__)

    def test_the_dp_still_shrinks_its_truncation_on_separated_groups(self):
        rng = np.random.RandomState(0)
        rows = [float(v) for k in range(4) for v in rng.normal(k * 20.0, 1.0, 125)]
        rng.shuffle(rows)
        model = get_dpm_mixture(rows, rng=np.random.RandomState(1), max_components=12, max_its=200, out=io.StringIO())
        weights = np.exp(np.asarray(model.log_w))
        self.assertLessEqual(int((weights > 0.01).sum()), 6)
        self.assertGreaterEqual(float(np.sort(weights)[::-1][:4].sum()), 0.9)


class HeldParameterDisclosureTest(unittest.TestCase):
    """P07-F03: a `free` the estimator will not move must not read back as an estimate."""

    @staticmethod
    def _fit(model, rows):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = model.fit(rows)
        return fitted, [entry for entry in caught if "does not fit" in str(entry.message)]

    def test_a_family_that_holds_a_parameter_fixed_says_so(self):
        from mixle.ppl import StudentT, free

        rows = list(np.random.RandomState(0).standard_t(2.5, 800) * 0.3)
        fitted, notes = self._fit(StudentT(free, free, free), rows)
        self.assertEqual(len(notes), 1)
        self.assertIn("df", str(notes[0].message))
        self.assertIn("gradient", str(notes[0].message))
        self.assertEqual(float(fitted.dist.df), 5.0)  # the family default, exactly as the note says
        self.assertNotAlmostEqual(float(fitted.dist.scale), 1.0, places=2)  # loc/scale ARE fitted

    def test_a_family_that_fits_everything_is_silent(self):
        from mixle.ppl import Normal, free

        rows = list(np.random.RandomState(0).normal(3.0, 2.0, 400))
        fitted, notes = self._fit(Normal(free, free), rows)
        self.assertEqual(notes, [])
        self.assertAlmostEqual(float(fitted.dist.mu), 3.0, delta=0.3)

    def test_the_declaration_is_part_of_the_family_record(self):
        from mixle.ppl.core import _FAMILIES

        self.assertEqual(_FAMILIES["StudentT"].holds_fixed, ("df",))
        self.assertEqual(_FAMILIES["Normal"].holds_fixed, ())


if __name__ == "__main__":
    unittest.main()
