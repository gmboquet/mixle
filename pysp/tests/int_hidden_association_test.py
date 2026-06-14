"""Regression tests for IntegerHiddenAssociation's uniform-background gate."""

import unittest

import numpy as np

from pysp.stats.int_hidden_association import IntegerHiddenAssociationDistribution

COND_WEIGHTS = np.asarray(
    [
        [0.80, 0.20],
        [0.25, 0.75],
    ],
    dtype=float,
)

STATE_PROB = np.asarray(
    [
        [0.90, 0.10],
        [0.20, 0.80],
    ],
    dtype=float,
)

DATA = (
    [(0, 2.0), (1, 1.0)],
    [(0, 3.0), (1, 2.0)],
)


def make_dist(alpha=0.35, use_numba=False):
    return IntegerHiddenAssociationDistribution(
        state_prob_mat=STATE_PROB,
        cond_weights=COND_WEIGHTS,
        alpha=alpha,
        use_numba=use_numba,
    )


def target_probabilities(alpha, datum):
    sources, targets = datum
    vx = np.asarray([u for u, _ in sources], dtype=int)
    cx = np.asarray([c for _, c in sources], dtype=float)
    vy = np.asarray([v for v, _ in targets], dtype=int)

    source_weights = cx / cx.sum()
    structured = np.zeros(len(vy), dtype=float)
    for pos, target in enumerate(vy):
        for src, src_weight in zip(vx, source_weights):
            structured[pos] += src_weight * np.dot(COND_WEIGHTS[src, :], STATE_PROB[:, target])

    return (1.0 - alpha) * structured + alpha / STATE_PROB.shape[1]


def expected_log_density(alpha, datum):
    cy = np.asarray([c for _, c in datum[1]], dtype=float)
    return float(np.dot(np.log(target_probabilities(alpha, datum)), cy))


def expected_structured_total(alpha, datum, weight):
    cy = np.asarray([c for _, c in datum[1]], dtype=float)
    probs = target_probabilities(alpha, datum)
    structured_mass = probs - alpha / STATE_PROB.shape[1]
    gate = np.divide(structured_mass, probs, out=np.zeros_like(probs), where=probs > 0.0)
    return float(weight * np.dot(cy, gate))


class IntegerHiddenAssociationGateTestCase(unittest.TestCase):
    def test_log_density_uses_one_normalized_uniform_component(self):
        alpha = 0.35
        dist = make_dist(alpha=alpha)

        self.assertAlmostEqual(dist.log_density(DATA), expected_log_density(alpha, DATA), places=12)

    def test_seq_log_density_matches_hand_calculation(self):
        alpha = 0.35
        data = [DATA, ([(0, 1.0)], [(1, 4.0)])]

        for use_numba in (False, True):
            dist = make_dist(alpha=alpha, use_numba=use_numba)
            enc = dist.dist_to_encoder().seq_encode(data)
            expected = np.asarray([expected_log_density(alpha, datum) for datum in data])

            np.testing.assert_allclose(dist.seq_log_density(enc), expected, rtol=1.0e-12, atol=1.0e-12)

    def test_structured_counts_receive_only_posterior_gate_mass(self):
        alpha = 0.40
        weight = 1.7
        expected = expected_structured_total(alpha, DATA, weight)

        dist = make_dist(alpha=alpha)
        acc = dist.estimator().accumulator_factory().make()
        acc.update(DATA, weight, dist)
        _, weight_count, state_count, _, _ = acc.value()

        self.assertAlmostEqual(float(weight_count.sum()), expected, places=12)
        self.assertAlmostEqual(float(state_count.sum()), expected, places=12)

    def test_seq_update_uses_same_gate_in_numba_and_python_encodings(self):
        alpha = 0.40
        weights = np.asarray([1.7])
        expected = expected_structured_total(alpha, DATA, weights[0])

        for use_numba in (False, True):
            dist = make_dist(alpha=alpha, use_numba=use_numba)
            acc = dist.estimator().accumulator_factory().make()
            enc = dist.dist_to_encoder().seq_encode([DATA])

            acc.seq_update(enc, weights, dist)
            _, weight_count, state_count, _, _ = acc.value()

            self.assertAlmostEqual(float(weight_count.sum()), expected, places=12)
            self.assertAlmostEqual(float(state_count.sum()), expected, places=12)


if __name__ == "__main__":
    unittest.main()
