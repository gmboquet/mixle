"""What a fit cannot tell you from its likelihood, it must tell you some other way.

Two residual items from the 0.8.1 adversarial reviews, both of the same shape: a fit converges, its
receipt is clean, and the answer is still not the one the data supports.

* A-02 -- a mixture can converge onto a component holding a handful of rows (an initialization that
  latched onto an outlier group on a heavy-tailed panel). The share alone is not the test, because a
  genuinely rare component is a legitimate answer; a component with fewer effective rows than it has
  free parameters is not identified by anything, and now says so.
* A-03 -- on a two-regime corpus two optima sit within a nat of each other: the regime split, whose
  components pass ``mixture_structure_health``, and a category split whose components each absorb
  both regimes. Which one a restart search returned depended on the seed. Health now breaks that
  tie; likelihood still decides outside it.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from mixle.inference import optimize
from mixle.inference.structure import learn_mixture_structure, mixture_structure_health
from mixle.stats import GaussianEstimator, MixtureEstimator


def _notes(callable_):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = callable_()
    return result, [entry for entry in caught if "less data than they have parameters" in str(entry.message)]


class UnidentifiedComponentTest(unittest.TestCase):
    ESTIMATOR = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])

    def test_a_component_holding_fewer_rows_than_parameters_is_named(self):
        rows = list(np.random.RandomState(0).normal(0.0, 1.0, 400)) + [40.0]
        model, notes = _notes(
            lambda: optimize(rows, self.ESTIMATOR, max_its=100, rng=np.random.RandomState(0), out=None)
        )
        self.assertTrue(notes)
        self.assertIn("component 1", str(notes[0].message))
        self.assertIn("component_row_mass", str(notes[0].message))
        self.assertEqual(notes[0].filename, __file__)
        # ``estimate`` runs once per EM iteration, so the note is raised once per iteration -- which
        # is why its text carries no per-iteration number: under the ordinary "once per location"
        # filter every raise collapses to the one line a reader sees.
        self.assertEqual(len({str(entry.message) for entry in notes}), 1)
        self.assertAlmostEqual(min(model.component_row_mass), 1.0, delta=0.5)
        self.assertAlmostEqual(sum(model.component_row_mass), float(len(rows)), delta=1.0)

    def test_the_default_warning_filter_shows_it_once(self):
        rows = list(np.random.RandomState(0).normal(0.0, 1.0, 400)) + [40.0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            optimize(rows, self.ESTIMATOR, max_its=100, rng=np.random.RandomState(0), out=None)
        named = [e for e in caught if "less data than they have parameters" in str(e.message)]
        self.assertEqual(len(named), 1)

    def test_a_fit_whose_components_are_all_supported_says_nothing(self):
        rng = np.random.RandomState(0)
        rows = list(rng.normal(-4.0, 1.0, 300)) + list(rng.normal(4.0, 1.0, 300))
        model, notes = _notes(
            lambda: optimize(rows, self.ESTIMATOR, max_its=60, rng=np.random.RandomState(0), out=None)
        )
        self.assertEqual(notes, [])
        self.assertEqual(len(model.component_row_mass), 2)
        for mass in model.component_row_mass:
            self.assertGreater(mass, 200.0)

    def test_the_recorded_mass_is_not_part_of_the_models_value(self):
        """It records how the fit went, not what the model IS, so it must not enter serialization."""
        from mixle.stats import dump_models, load_models

        rng = np.random.RandomState(0)
        rows = list(rng.normal(-4.0, 1.0, 300)) + list(rng.normal(4.0, 1.0, 300))
        model = optimize(rows, self.ESTIMATOR, max_its=40, rng=np.random.RandomState(0), out=None)
        restored = load_models(dump_models(model, verify=True))
        self.assertAlmostEqual(float(model.log_density(0.5)), float(restored.log_density(0.5)), places=12)
        self.assertFalse(hasattr(restored, "component_row_mass"))


class HealthTieBreakTest(unittest.TestCase):
    @staticmethod
    def _two_regime(seed, n=800):
        rng = np.random.RandomState(seed)
        category = rng.choice(["x", "y"], size=n)
        regime = rng.rand(n) < 0.5
        level = np.where(regime, 6.0, -6.0) + rng.normal(0.0, 1.0, n)
        slope = np.where(regime, 1.0, -1.0) * level + rng.normal(0.0, 1.0, n)
        return [(str(c), float(a), float(b)) for c, a, b in zip(category, level, slope)]

    def test_health_breaks_a_likelihood_tie_and_does_not_overrule_a_better_fit(self):
        rows = self._two_regime(8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tie_broken = learn_mixture_structure(rows, 2, restarts=4, seed=0, tie_tolerance=1.0)
            likelihood_only = learn_mixture_structure(rows, 2, restarts=4, seed=0, tie_tolerance=0.0)
        encoder = tie_broken.dist_to_encoder()
        broken_ll = float(np.sum(tie_broken.seq_log_density(encoder.seq_encode(rows))))
        plain_ll = float(np.sum(likelihood_only.seq_log_density(likelihood_only.dist_to_encoder().seq_encode(rows))))
        # The tie-break may only give up likelihood WITHIN the tolerance it was given.
        self.assertGreaterEqual(broken_ll, plain_ll - 1.0)

    def test_the_tolerance_is_validated(self):
        rows = self._two_regime(1, n=120)
        for bad in (-1.0, float("nan"), float("inf")):
            with self.subTest(tie_tolerance=repr(bad)):
                with self.assertRaises(ValueError):
                    learn_mixture_structure(rows, 2, restarts=1, seed=0, tie_tolerance=bad)
        with self.assertRaises(TypeError):
            learn_mixture_structure(rows, 2, restarts=1, seed=0, tie_tolerance="1")

    def test_the_receipt_still_arbitrates_whatever_is_returned(self):
        rows = self._two_regime(3, n=600)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = learn_mixture_structure(rows, 2, restarts=3, seed=0)
        health = mixture_structure_health(model, rows)
        self.assertIn("diagnosis", health)
        self.assertIn("components", health)


if __name__ == "__main__":
    unittest.main()
