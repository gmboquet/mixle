"""Neural-net ops in the log number system (mixle.engines.lns_nn): softmax/cross-entropy + sum-product.

Validated against float64 within the LNS step bound. The wins are the log-space parts (the softmax/CE
normalizer, the whole sum-product forward) -- the integer logsumexp replaces exp/log.
"""

import math
import unittest

import numpy as np
import pytest

from mixle.engines.lns import LogNumberSystem
from mixle.engines.lns_nn import SumProductCircuit, cross_entropy, log_softmax, softmax

sp = pytest.importorskip("scipy.special")


class SoftmaxCrossEntropyTest(unittest.TestCase):
    def test_log_softmax_matches_float64(self):
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(0)
        logits = rng.randn(256, 1000) * 5  # (tokens, vocab)
        ref = sp.log_softmax(logits, axis=1)
        got = log_softmax(logits, lns, axis=1)
        # 1000-way vocab log-partition: depth-10 pairwise tree (MXR-080-0139)
        self.assertLessEqual(float(np.max(np.abs(got - ref))), lns.max_logsumexp_error(1000))

    def test_softmax_is_a_distribution(self):
        lns = LogNumberSystem(step=0.005)
        p = softmax(np.random.RandomState(1).randn(64, 500) * 4, lns, axis=1)
        self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1e-2))
        self.assertTrue(np.all(p >= 0))

    def test_cross_entropy_matches_float64(self):
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(2)
        logits = rng.randn(2000, 800) * 5
        targets = rng.randint(0, 800, size=2000)
        ref = float(np.mean(sp.logsumexp(logits, axis=1) - logits[np.arange(2000), targets]))
        got = cross_entropy(logits, targets, lns, axis=1)
        self.assertLessEqual(abs(got - ref), lns.max_logsumexp_error(800))  # 800-class log-partition


class SumProductCircuitTest(unittest.TestCase):
    def _circuit(self):
        ln = math.log
        return SumProductCircuit(
            [
                ("leaf", 0),
                ("leaf", 1),
                ("leaf", 2),
                ("leaf", 3),
                ("sum", [0, 1], [ln(0.6), ln(0.4)]),  # node 4: mixture of leaves 0,1
                ("sum", [2, 3], [ln(0.3), ln(0.7)]),  # node 5: mixture of leaves 2,3
                ("product", [4, 5]),  # node 6: independent product of the two sub-mixtures
                ("sum", [6, 0], [ln(0.8), ln(0.2)]),  # node 7 (root): mix the product with leaf 0
            ]
        )

    def test_lns_forward_matches_float64(self):
        lns = LogNumberSystem(step=0.002)
        rng = np.random.RandomState(3)
        leaves = {i: rng.randn(5000) * 6 for i in range(4)}  # batched leaf log-values
        circuit = self._circuit()
        ref = circuit.evaluate_float(leaves)
        got = circuit.evaluate_lns(lns, leaves)
        # Each individual "sum" node here is a 2-way logadd (max_logsumexp_error(2)), but the root's
        # error compounds through THREE dependent sum nodes plus a product (MXR-080-0139's bound
        # covers a single flat n-way reduction, not a general sum/product DAG -- lns_nn.py's circuit
        # evaluator has its own, broader error-composition question, tracked separately), so keep a
        # multiplier on top of the single-node certificate rather than treating it as one.
        self.assertLessEqual(float(np.max(np.abs(got - ref))), 4 * lns.max_logsumexp_error(2))

    def test_product_node_is_exact_integer_add(self):
        # a pure product of leaves is exact integer addition of the quantized leaf log-values
        lns = LogNumberSystem(step=0.01)
        circuit = SumProductCircuit([("leaf", 0), ("leaf", 1), ("product", [0, 1])])
        leaves = {0: np.array([-2.0, -3.0]), 1: np.array([-1.0, -0.5])}
        got = circuit.evaluate_lns(lns, leaves)
        expect = lns.dequantize(lns.quantize(leaves[0]) + lns.quantize(leaves[1]))
        self.assertTrue(np.array_equal(got, expect))


if __name__ == "__main__":
    unittest.main()
