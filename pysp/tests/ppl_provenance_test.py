"""Reproducible artifacts for the PPL surface: provenance headers for fitted RandomVariables."""

import unittest

import numpy as np

from pysp.inference.provenance import ModelHeader
from pysp.ppl import Normal, fit_with_provenance, free


class PPLProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.data = np.random.RandomState(0).normal(3.0, 2.0, 400).tolist()

    def test_em_path_full_header(self):
        fitted, header = fit_with_provenance(Normal(free, free), self.data, how="em", max_its=30)
        self.assertEqual(header.training["method"], "em")
        self.assertEqual(header.training["surface"], "ppl")
        self.assertEqual(header.schema, [("value", "Real")])  # built from the lowered distribution
        self.assertIsNotNone(header.final_loglik)
        self.assertIn("duration_s", header.timing)
        # the fitted RV recovers the truth, so the header reflects a real fit
        self.assertAlmostEqual(fitted.dist.mu, 3.0, delta=0.4)

    def test_map_path_full_header(self):
        _, header = fit_with_provenance(Normal(free, free), self.data, how="map", max_its=50)
        self.assertEqual(header.training["method"], "map")
        self.assertIsNotNone(header.final_loglik)
        self.assertEqual(len(header.dataset_hash), 64)

    def test_header_round_trips(self):
        _, header = fit_with_provenance(Normal(free, free), self.data, how="em", max_its=20)
        back = ModelHeader.from_dict(header.to_dict())
        self.assertEqual(back.dataset_hash, header.dataset_hash)
        self.assertEqual(back.training["surface"], "ppl")


if __name__ == "__main__":
    unittest.main()
