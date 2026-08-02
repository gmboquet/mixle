"""A declared endpoint probability is an exact law, not a clipped one (MXR-080-1889).

``_bernoulli_log_likelihood`` clipped ``p`` into ``[1e-12, 1 - 1e-12]``, so every declared endpoint
became a finite score in both directions:

* a ``p = 0`` model scored a PRESENT edge at ``log(1e-12) = -27.63``. Evidence the model declares
  impossible entered likelihood ratios, BIC, and EM responsibilities as merely unlikely.
* the same model scored a graph with no edges at ``-1e-12`` rather than the exact ``0`` a certain
  event has.

The clip was presumably guarding ``0 * log(0)``, which computes as ``0 * -inf = nan``. The convention
is that an outcome observed zero times contributes nothing, so the endpoints are counted rather than
clipped.
"""

import unittest

import numpy as np

from mixle.models.random_graph import ErdosRenyiGraphModel, StochasticBlockGraphModel

PRESENT = np.array([[0.0, 1.0], [1.0, 0.0]])
ABSENT = np.zeros((2, 2))


class EndpointLawTest(unittest.TestCase):
    def test_impossible_evidence_scores_negative_infinity(self):
        self.assertEqual(ErdosRenyiGraphModel(0.0).log_likelihood(PRESENT), float("-inf"))
        self.assertEqual(ErdosRenyiGraphModel(1.0).log_likelihood(ABSENT), float("-inf"))

    def test_certain_evidence_scores_exactly_zero(self):
        self.assertEqual(ErdosRenyiGraphModel(0.0).log_likelihood(ABSENT), 0.0)
        self.assertEqual(ErdosRenyiGraphModel(1.0).log_likelihood(PRESENT), 0.0)

    def test_the_endpoints_do_not_produce_nan(self):
        # 0 * log(0) is 0 by convention but nan when computed; that is what the clip was avoiding.
        for p, adjacency in ((0.0, ABSENT), (1.0, PRESENT)):
            with self.subTest(p=p):
                self.assertFalse(np.isnan(ErdosRenyiGraphModel(p).log_likelihood(adjacency)))

    def test_interior_probabilities_are_unchanged(self):
        # One present and no absent edge among the single undirected pair.
        model = ErdosRenyiGraphModel(0.25)
        self.assertAlmostEqual(model.log_likelihood(PRESENT), float(np.log(0.25)))
        self.assertAlmostEqual(model.log_likelihood(ABSENT), float(np.log1p(-0.25)))

    def test_a_block_model_endpoint_is_exact_too(self):
        probs = np.array([[0.0, 0.0], [0.0, 0.0]])
        model = StochasticBlockGraphModel(probs, [0, 1])
        self.assertEqual(model.log_likelihood(PRESENT), float("-inf"))
        self.assertEqual(model.log_likelihood(ABSENT), 0.0)


class OwnedParametersTest(unittest.TestCase):
    def test_the_block_probability_matrix_is_not_aliased(self):
        # np.asarray does not copy an array that is already float64, so every validation passed and
        # the caller could then rewrite what the model scores.
        probs = np.array([[0.0, 0.5], [0.5, 0.0]])
        model = StochasticBlockGraphModel(probs, [0, 1])
        before = model.log_likelihood(PRESENT)
        probs[0, 1] = 0.9
        probs[1, 0] = 0.9
        self.assertEqual(model.log_likelihood(PRESENT), before)

    def test_a_boolean_seed_is_refused_by_both_samplers(self):
        # RandomState(True) silently means seed 1. ErdosRenyi already refused it; the block model
        # bypassed the module's own validator.
        with self.assertRaises((TypeError, ValueError)):
            StochasticBlockGraphModel(np.array([[0.5, 0.5], [0.5, 0.5]]), [0, 1]).sample(seed=True)
        with self.assertRaises((TypeError, ValueError)):
            ErdosRenyiGraphModel(0.5).sample(4, seed=True)

    def test_an_ordinary_seed_still_draws_reproducibly(self):
        model = StochasticBlockGraphModel(np.array([[0.5, 0.5], [0.5, 0.5]]), [0, 1])
        np.testing.assert_array_equal(model.sample(seed=7), model.sample(seed=7))


if __name__ == "__main__":
    unittest.main()
