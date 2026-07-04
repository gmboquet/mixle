"""C4: the flagship cross-modal fit receipt — one graph over categorical + image + signal + target."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from cross_modal_fit_receipt import _clg_mean, make_records  # noqa: E402

from mixle.inference import certify, learn_bayesian_network  # noqa: E402


class CrossModalFitTest(unittest.TestCase):
    def setUp(self):
        self.net = learn_bayesian_network(make_records(300, 0), max_parents=2)

    def test_modality_fields_become_vector_nodes(self):
        kinds = {f.child: type(f).__name__ for f in self.net.factors}
        self.assertEqual(kinds[1], "_VectorMarginalFactor")  # image latent is a vector node
        self.assertEqual(kinds[2], "_VectorMarginalFactor")  # signal latent is a vector node

    def test_cross_modal_edges_are_recovered(self):
        price = next(f for f in self.net.factors if f.child == 3)
        self.assertEqual(set(price.parents), {1, 2})  # price <- image AND signal

    def test_fit_certifies_global_no_gradient(self):
        cert = certify(self.net)
        self.assertGreaterEqual(int(cert.guarantee), 4)  # GLOBAL or better
        self.assertEqual(len(cert.gradient_blocks), 0)  # nothing needed ADAM
        self.assertIn("No gradient descent", cert.why_not_adam())

    def test_held_out_price_prediction_is_accurate(self):
        pf = next(f for f in self.net.factors if f.child == 3)
        test = make_records(300, 1)
        truth = [r[3] for r in test]
        pred = [_clg_mean(pf, r) for r in test]
        corr = float(np.corrcoef(truth, pred)[0, 1])
        self.assertGreater(corr, 0.95)  # the closed-form CLG readout tracks held-out price


if __name__ == "__main__":
    unittest.main()
