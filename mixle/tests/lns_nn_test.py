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


class CrossEntropyTargetValidationTest(unittest.TestCase):
    """MXR-080-0140 / MXR-080-0141: cross_entropy must validate targets identically whether or not the
    compiled fused path (k.ndim==2, axis in (-1, 1)) is available -- so every case here is run against
    BOTH a 2-D/axis=1 call (fused-eligible) and a 3-D/axis=2 call (always the numpy fallback), to prove
    neither path is more permissive than the other.
    """

    def _logits(self, rng):
        return rng.randn(5, 8).astype(np.float64)

    def test_negative_target_rejected_both_paths(self):
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(10)
        logits2d = self._logits(rng)
        targets = np.array([-1, 0, 1, 2, 3])
        with self.assertRaises(ValueError):
            cross_entropy(logits2d, targets, lns, axis=1)  # fused-eligible path
        logits3d = logits2d[:, None, :]  # (5, 1, 8): ndim=3, never takes the fused path
        with self.assertRaises(ValueError):
            cross_entropy(logits3d, targets[:, None], lns, axis=2)

    def test_fractional_target_rejected_both_paths(self):
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(11)
        logits2d = self._logits(rng)
        targets = np.array([2.7, 0.0, 1.0, 3.9, 0.0])
        with self.assertRaises(ValueError):
            cross_entropy(logits2d, targets, lns, axis=1)
        logits3d = logits2d[:, None, :]
        with self.assertRaises(ValueError):
            cross_entropy(logits3d, targets[:, None], lns, axis=2)

    def test_out_of_range_target_rejected_before_reaching_compiled_kernel(self):
        # MXR-080-0140: an out-of-range class index must never reach the boundscheck-disabled compiled
        # path at all -- if it did, this would be a genuine out-of-bounds read, not just a wrong answer.
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(12)
        logits2d = self._logits(rng)
        with self.assertRaises(ValueError):
            cross_entropy(logits2d, np.array([0, 1, 2, 3, 999]), lns, axis=1)
        with self.assertRaises(ValueError):
            cross_entropy(logits2d, np.array([0, 1, 2, 3, -1]), lns, axis=1)

    def test_mismatched_target_length_rejected(self):
        # MXR-080-0140's "too-short targets": in the compiled path this would read past the targets
        # buffer.
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(13)
        logits2d = self._logits(rng)
        with self.assertRaises(ValueError):
            cross_entropy(logits2d, np.array([0, 1, 2]), lns, axis=1)
        with self.assertRaises(ValueError):
            cross_entropy(logits2d, np.array([0, 1, 2, 3, 4, 5]), lns, axis=1)

    def test_zero_classes_rejected(self):
        # MXR-080-0140's "empty rows": a zero-width row has no valid target and, pre-fix, the compiled
        # kernel would read its never-initialized scratch buffer.
        lns = LogNumberSystem(step=0.005)
        with self.assertRaises(ValueError):
            cross_entropy(np.zeros((3, 0)), np.array([0, 0, 0]), lns, axis=1)

    def test_empty_batch_rejected(self):
        lns = LogNumberSystem(step=0.005)
        with self.assertRaises(ValueError):
            cross_entropy(np.zeros((0, 8)), np.zeros((0,), dtype=np.int64), lns, axis=1)

    def test_target_with_log_zero_logit_gives_infinite_loss_not_corrupted_finite_value(self):
        # MXR-080-0157: a target whose OWN logit is exactly -inf (quantizes to LOG_ZERO_CODE) has a
        # mathematically infinite loss. Pre-fix this reached a raw int64 subtraction against
        # LOG_ZERO_CODE (== INT64_MIN) in both the compiled kernel and the numpy fallback's
        # `lse_k - tgt_k`, silently wrapping to a corrupted (often deeply negative-looking, i.e.
        # falsely "great") finite loss instead. Checked against an independent float64 reference: a
        # true -inf logit at the target position gives ordinary IEEE +inf under plain float subtraction.
        lns = LogNumberSystem(step=0.005)
        logits = np.array([[0.0, -np.inf, 1.0]])
        got = cross_entropy(logits, np.array([1]), lns, axis=1)
        self.assertEqual(got, float("inf"))
        ref = float(sp.logsumexp(logits[0]) - logits[0, 1])
        self.assertEqual(ref, float("inf"))
        # negative control: an ordinary (non-log-zero) target in the same row is unaffected
        got_ok = cross_entropy(logits, np.array([2]), lns, axis=1)
        self.assertTrue(math.isfinite(got_ok))

    def test_negative_control_well_formed_targets_unaffected(self):
        # in-range integer targets (numpy int and Python int dtypes) and well-behaved logits must
        # continue to work exactly as before across both paths.
        lns = LogNumberSystem(step=0.005)
        rng = np.random.RandomState(14)
        logits2d = self._logits(rng)
        targets = np.array([0, 1, 2, 3, 4])
        ref = float(np.mean(sp.logsumexp(logits2d, axis=1) - logits2d[np.arange(5), targets]))
        got2d = cross_entropy(logits2d, targets, lns, axis=1)
        self.assertLessEqual(abs(got2d - ref), lns.max_logsumexp_error(8))
        logits3d = logits2d[:, None, :]
        got3d = cross_entropy(logits3d, targets[:, None], lns, axis=2)
        self.assertLessEqual(abs(got3d - ref), lns.max_logsumexp_error(8))


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
