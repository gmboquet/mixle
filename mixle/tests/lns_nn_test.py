"""Neural-net ops in the log number system (mixle.engines.lns_nn): softmax/cross-entropy + sum-product.

Validated against float64 within the LNS step bound. The wins are the log-space parts (the softmax/CE
normalizer, the whole sum-product forward) -- the integer logsumexp replaces exp/log.
"""

import math
import unittest

import numpy as np
import pytest

from mixle.engines.lns import LOG_ZERO_CODE, LogNumberSystem
from mixle.engines.lns_nn import SumProductCircuit, cross_entropy, log_softmax, softmax

sp = pytest.importorskip("scipy.special")


class SoftmaxCrossEntropyTest(unittest.TestCase):
    def test_impossible_logits_preserve_sentinel_semantics(self):
        lns = LogNumberSystem(step=0.005)
        with self.assertRaises(ValueError):
            log_softmax([-np.inf, -np.inf], lns)
        with self.assertRaises(ValueError):
            softmax([-np.inf, -np.inf], lns)
        got = log_softmax([-np.inf, 0.0], lns)
        self.assertEqual(got[0], -np.inf)
        self.assertAlmostEqual(got[1], 0.0, places=12)
        p = softmax([-np.inf, 0.0], lns)
        np.testing.assert_array_equal(p, [0.0, 1.0])

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

    def test_softmax_of_equal_logits_sums_to_exactly_one(self):
        # MXR-080-0141: exponentiating the LNS-approximate log-softmax without renormalizing let 100
        # equal logits sum to 1.00518 pre-fix -- a real violation of the probability-simplex contract a
        # function named "softmax" must satisfy. Post-fix this must be exact to float64 precision, not
        # merely "close" under a loose tolerance (test_softmax_is_a_distribution above already covers
        # the loose-tolerance case; this is the audit's own exact scenario).
        lns = LogNumberSystem(step=0.005)
        p = softmax(np.zeros(100), lns, axis=-1)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=12)
        # and it must still be the (renormalized) uniform distribution, not just "sums to 1 somehow"
        self.assertTrue(np.allclose(p, 1.0 / 100))

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


class SumProductCircuitValidationTest(unittest.TestCase):
    """MXR-080-0142: every structural-validity violation must be rejected at construction, not deferred
    to (possibly silently-wrong, per test_forward_reference_is_rejected below) evaluation.
    """

    def test_empty_circuit_is_rejected(self):
        with self.assertRaises(ValueError):
            SumProductCircuit([])

    def test_empty_product_node_is_rejected(self):
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("product", [])])

    def test_empty_sum_node_is_rejected(self):
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("sum", [], [])])

    def test_forward_reference_is_rejected(self):
        # pre-fix, this was worse than "fails to error": node 0 ("product", [1]) reads vals[1] while
        # it's still None (nodes are evaluated in list order) and silently produces a wrong answer built
        # from the leaf alone, with no exception at all -- exactly the "accepted until ... arithmetic
        # fails" (or, here, doesn't even fail) hazard the finding describes.
        with self.assertRaises(ValueError):
            SumProductCircuit([("product", [1]), ("leaf", 0)])

    def test_self_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("product", [1])])

    def test_out_of_range_child_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("product", [5])])

    def test_negative_child_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("leaf", 1), ("product", [-1, 1])])

    def test_boolean_child_and_nonvector_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("product", [False])])
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("sum", [0], [[0.0]])])

    def test_circuit_owns_an_immutable_canonical_graph(self):
        children = [0]
        weights = np.array([0.0])
        nodes = [("leaf", 0), ("sum", children, weights)]
        circuit = SumProductCircuit(nodes)
        children[0] = 1
        weights[0] = 99.0
        nodes[1] = ("product", [1])
        self.assertEqual(circuit.nodes, (("leaf", 0), ("sum", (0,), (0.0,))))
        self.assertIsInstance(circuit.nodes, tuple)
        self.assertIsInstance(circuit.leaf_ids, frozenset)
        self.assertEqual(float(circuit.evaluate_float({0: 2.0})), 2.0)

    def test_cardinality_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("leaf", 1), ("sum", [0, 1], [0.0])])

    def test_weights_not_summing_to_one_are_rejected(self):
        # pre-fix this did not even fail at evaluation -- it silently returned a wrong (non-probability)
        # answer, since exp(0) + exp(0) = 2 is a perfectly well-defined (if invalid) logsumexp input.
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("leaf", 1), ("sum", [0, 1], [0.0, 0.0])])

    def test_non_finite_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("leaf", 1), ("sum", [0, 1], [0.0, float("nan")])])
        with self.assertRaises(ValueError):
            SumProductCircuit([("leaf", 0), ("leaf", 1), ("sum", [0, 1], [0.0, float("inf")])])

    def test_unrecognized_node_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            SumProductCircuit([("bogus", 0)])

    def test_missing_leaf_value_is_rejected_at_evaluation(self):
        # the set of expected leaf ids is fixed at construction; whether leaf_values actually supplies
        # them can only be checked once leaf_values is provided, at evaluation.
        lns = LogNumberSystem(step=0.01)
        circuit = SumProductCircuit([("leaf", 0), ("leaf", 1), ("product", [0, 1])])
        with self.assertRaises(ValueError):
            circuit.evaluate_lns(lns, {0: np.array([1.0])})  # leaf 1 missing
        with self.assertRaises(ValueError):
            circuit.evaluate_float({0: np.array([1.0])})

    def test_negative_control_well_formed_circuit_still_works(self):
        ln = math.log
        circuit = SumProductCircuit(
            [
                ("leaf", 0),
                ("leaf", 1),
                ("leaf", 2),
                ("leaf", 3),
                ("sum", [0, 1], [ln(0.6), ln(0.4)]),
                ("sum", [2, 3], [ln(0.3), ln(0.7)]),
                ("product", [4, 5]),
                ("sum", [6, 0], [ln(0.8), ln(0.2)]),
            ]
        )
        self.assertEqual(circuit.leaf_ids, {0, 1, 2, 3})
        lns = LogNumberSystem(step=0.002)
        leaves = {i: np.random.RandomState(7).randn(200) * 6 for i in range(4)}
        ref = circuit.evaluate_float(leaves)
        got = circuit.evaluate_lns(lns, leaves)
        self.assertLessEqual(float(np.max(np.abs(got - ref))), 4 * lns.max_logsumexp_error(2))


class SumProductCircuitSafeMultiplyTest(unittest.TestCase):
    """MXR-080-0142's multiply()-not-raw-`+` gap flagged alongside the lns.py LOG_ZERO_CODE fix: a
    product (or a sum node's weighted-child term, which is the same "LNS multiply" operation) that
    combines a LOG_ZERO_CODE-valued child with an ordinary code must not silently overflow int64.
    """

    def test_raw_int64_add_of_log_zero_code_silently_overflows(self):
        # establishes WHY this matters: confirms the hazard the fix avoids is real, not hypothetical.
        # LOG_ZERO_CODE (== INT64_MIN) plus any negative code underflows int64, and numpy performs no
        # overflow check on integer arithmetic by default -- it wraps silently rather than raising.
        raw = np.array([LOG_ZERO_CODE], dtype=np.int64) + np.array([-50], dtype=np.int64)
        self.assertNotEqual(int(raw[0]), LOG_ZERO_CODE - 50)  # the "true" (unrepresentable) value
        self.assertGreater(int(raw[0]), 0)  # wrapped around to a spurious huge positive code

    def test_product_node_with_log_zero_leaf_uses_multiply_not_raw_add(self):
        lns = LogNumberSystem(step=0.01)
        circuit = SumProductCircuit([("leaf", 0), ("leaf", 1), ("product", [0, 1])])
        # lane 0: ordinary product (exact integer add). lane 1: leaf 1 is a true zero (-inf log-value),
        # quantizing to LOG_ZERO_CODE -- raw `+` against that sentinel would silently wrap (as proven
        # above); lns.multiply() must instead propagate the absorbing zero exactly.
        leaves = {0: np.array([-2.0, -3.0]), 1: np.array([-1.0, -np.inf])}
        got = circuit.evaluate_lns(lns, leaves)
        self.assertAlmostEqual(float(got[0]), -3.0, places=6)
        self.assertEqual(float(got[1]), -np.inf)
        # cross-check directly against lns.multiply() as the independent reference for lane 1
        k0 = lns.quantize(leaves[0])
        k1 = lns.quantize(leaves[1])
        self.assertEqual(int(k1[1]), LOG_ZERO_CODE)
        expect = lns.multiply(k0, k1)
        self.assertTrue(np.array_equal(lns.quantize(got), expect))

    def test_sum_node_weighted_child_with_log_zero_uses_multiply_not_raw_add(self):
        # the "sum" node's per-child term is log(child_prob * weight) = child_code (LNS-multiply)
        # weight_code -- the same hazard as the product-node case, for a weight applied to a
        # LOG_ZERO_CODE-valued child.
        lns = LogNumberSystem(step=0.01)
        ln = math.log
        circuit = SumProductCircuit([("leaf", 0), ("leaf", 1), ("sum", [0, 1], [ln(0.5), ln(0.5)])])
        leaves = {0: np.array([-np.inf]), 1: np.array([-1.0])}  # child 0 is a true zero
        got = circuit.evaluate_lns(lns, leaves)
        # a 50/50 mixture of (probability 0) and (probability exp(-1)) is just 0.5 * exp(-1); must be
        # finite and close to the float64 reference, not a spuriously huge/corrupted value.
        ref = circuit.evaluate_float(leaves)
        self.assertTrue(np.isfinite(got[0]))
        self.assertLessEqual(abs(float(got[0]) - float(ref[0])), 4 * lns.max_logsumexp_error(2))


if __name__ == "__main__":
    unittest.main()
