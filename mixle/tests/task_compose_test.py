"""compose(): chaining two models/teachers into one ledger-carrying callable (workstream D10)."""

import unittest

from mixle.task.compose import ComposedAnswer, compose

# Synthetic 2-stage world: neither stage alone maps a species name to its habitat.
_SPECIES_TO_TAXON = {
    "cat": "mammal",
    "dog": "mammal",
    "shark": "fish",
    "tuna": "fish",
    "eagle": "bird",
    "sparrow": "bird",
}
_TAXON_TO_HABITAT = {"mammal": "land", "fish": "water", "bird": "air"}
_TAXON_CONFIDENCE = {"mammal": 0.9, "fish": 0.8, "bird": 0.7}
_HABITAT_CONFIDENCE = {"land": -0.1, "water": -0.2, "bird": -0.3, "air": -0.05}


class _Classifier:
    """A tiny scored callable: ``__call__`` predicts, ``score`` returns a log-confidence for its input."""

    def __init__(self, table: dict, confidence: dict) -> None:
        self.table = table
        self.confidence = confidence

    def __call__(self, x):
        return self.table[x]

    def score(self, x):
        return float(self.confidence.get(x, self.confidence.get(self.table[x], 0.0)))


def _taxon_classifier() -> _Classifier:
    return _Classifier(_SPECIES_TO_TAXON, _TAXON_CONFIDENCE)


def _habitat_classifier() -> _Classifier:
    return _Classifier(_TAXON_TO_HABITAT, _HABITAT_CONFIDENCE)


class ComposeTest(unittest.TestCase):
    def test_composed_answers_what_neither_stage_answers_alone(self):
        species_to_taxon = _taxon_classifier()
        taxon_to_habitat = _habitat_classifier()
        composed = compose(species_to_taxon, taxon_to_habitat, name_a="species_to_taxon", name_b="taxon_to_habitat")

        for species, expected_taxon in _SPECIES_TO_TAXON.items():
            expected_habitat = _TAXON_TO_HABITAT[expected_taxon]
            self.assertEqual(composed(species), expected_habitat)
            # neither stage alone answers species -> habitat: stage a stops at the taxon,
            # and stage b cannot even accept a species name as input.
            self.assertEqual(species_to_taxon(species), expected_taxon)
            self.assertNotEqual(species_to_taxon(species), expected_habitat)
            with self.assertRaises(KeyError):
                taxon_to_habitat(species)

    def test_receipt_attributes_the_answer_to_both_stages(self):
        composed = compose(
            _taxon_classifier(), _habitat_classifier(), name_a="species_to_taxon", name_b="taxon_to_habitat"
        )
        result = composed.answer("shark")

        self.assertIsInstance(result, ComposedAnswer)
        self.assertEqual(result.answer, "water")
        self.assertEqual(result.intermediate, "fish")
        self.assertEqual([name for name, _, _ in result.stages], ["species_to_taxon", "taxon_to_habitat"])
        self.assertEqual([out for _, out, _ in result.stages], ["fish", "water"])
        # both stages contributed a nonzero, distinct piece of evidence to the final answer
        contributions = [c for _, _, c in result.stages]
        self.assertTrue(all(c != 0.0 for c in contributions))
        self.assertTrue(result.check())
        self.assertAlmostEqual(sum(contributions), result.total_contribution)

    def test_unscored_stage_contributes_zero_honestly(self):
        # a plain function has no .score/.confidence -- the ledger must not fabricate a number for it
        composed = compose(lambda x: x.upper(), lambda x: f"{x}!")
        result = composed.answer("hi")
        self.assertEqual(result.answer, "HI!")
        self.assertEqual([c for _, _, c in result.stages], [0.0, 0.0])
        self.assertTrue(result.check())

    def test_chained_composition_of_composed_models(self):
        stage1 = compose(_taxon_classifier(), _habitat_classifier())
        habitat_to_upper = compose(stage1, lambda h: h.upper())
        self.assertEqual(habitat_to_upper("eagle"), "AIR")


if __name__ == "__main__":
    unittest.main()
