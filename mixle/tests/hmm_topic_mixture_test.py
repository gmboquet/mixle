"""HMM topic-mixture emissions (taus): HiddenMarkovModelDistribution(..., taus=...).

Regression for a confirmed audit finding: log_density's has_topics branch recomputed a from-scratch,
non-recursive quantity every timestep instead of a proper forward recursion, discarding every timestep
but the last (returning log_density==0.0, i.e. probability 1.0, for a 3-step sequence that should score
around -4.6 nats). It also indexed log_taus with the wrong axis, crashing outright whenever the topic
pool size differs from the state count. HiddenMarkovModelDistributionSampler already implements the
correct semantics for sampling (state i emits from MixtureDistribution(topics, taus[i, :])); this module
verifies log_density now matches that same semantics via brute-force enumeration over hidden-state paths,
including the n_topics != n_states case the old code couldn't even run.
"""

import itertools
import unittest

import numpy as np

from mixle.stats import CategoricalDistribution, HiddenMarkovModelDistribution


def _brute_force_log_density(w, trans, topics, taus, seq):
    """log P(seq) by summing over every hidden-state path, state i emitting Mixture(topics, taus[i, :])."""
    n_states = len(w)
    n_topics = len(topics)
    total = 0.0
    for path in itertools.product(range(n_states), repeat=len(seq)):
        p = w[path[0]]
        for t in range(1, len(path)):
            p *= trans[path[t - 1]][path[t]]
        for t, s in enumerate(path):
            emit_p = sum(taus[s][j] * topics[j].density(seq[t]) for j in range(n_topics))
            p *= emit_p
        total += p
    return float(np.log(total))


class HmmTopicMixtureLogDensityTest(unittest.TestCase):
    def test_non_identity_taus_with_more_topics_than_states_matches_brute_force(self):
        # n_topics=3 != n_states=2: the old code's log_taus[:, i] indexing crashed outright here.
        topics = [
            CategoricalDistribution({"a": 0.7, "b": 0.2, "c": 0.1}),
            CategoricalDistribution({"a": 0.1, "b": 0.7, "c": 0.2}),
            CategoricalDistribution({"a": 0.2, "b": 0.1, "c": 0.7}),
        ]
        w = [0.6, 0.4]
        trans = [[0.7, 0.3], [0.4, 0.6]]
        taus = [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
        hmm = HiddenMarkovModelDistribution(topics, w=w, transitions=trans, taus=taus)

        for seq in (["a", "b", "c"], ["a"], ["c", "c", "c", "a"], ["b", "a"]):
            with self.subTest(seq=seq):
                expected = _brute_force_log_density(w, trans, topics, taus, seq)
                self.assertAlmostEqual(hmm.log_density(seq), expected, places=8)

    def test_identity_taus_reduces_to_the_plain_per_state_hmm(self):
        # taus=identity means state i's mixture collapses onto topics[i] alone -- log_density must then
        # agree exactly with the taus-free HMM built from the same topics/w/transitions.
        topic0 = CategoricalDistribution({"a": 1.0, "b": 0.0})
        topic1 = CategoricalDistribution({"a": 0.0, "b": 1.0})
        w = [0.5, 0.5]
        trans = [[0.9, 0.1], [0.2, 0.8]]
        taus = [[1.0, 0.0], [0.0, 1.0]]

        hmm_taus = HiddenMarkovModelDistribution([topic0, topic1], w=w, transitions=trans, taus=taus)
        hmm_plain = HiddenMarkovModelDistribution([topic0, topic1], w=w, transitions=trans)
        seq = ["a", "b", "a"]
        self.assertAlmostEqual(hmm_taus.log_density(seq), hmm_plain.log_density(seq), places=10)
        self.assertAlmostEqual(hmm_taus.log_density(seq), -4.605170185988091, places=8)

    def test_log_density_of_a_zero_probability_sequence_is_negative_infinity(self):
        topic0 = CategoricalDistribution({"a": 1.0, "b": 0.0})
        topic1 = CategoricalDistribution({"a": 0.0, "b": 1.0})
        w = [0.5, 0.5]
        trans = [[0.9, 0.1], [0.2, 0.8]]
        taus = [[1.0, 0.0], [0.0, 1.0]]
        hmm = HiddenMarkovModelDistribution([topic0, topic1], w=w, transitions=trans, taus=taus)
        # "c" is outside every topic's support -> the whole sequence has zero probability
        self.assertEqual(hmm.log_density(["a", "c"]), float("-inf"))

    def test_single_observation_matches_the_marginal_mixture(self):
        topics = [
            CategoricalDistribution({"a": 0.9, "b": 0.1}),
            CategoricalDistribution({"a": 0.2, "b": 0.8}),
        ]
        w = [0.3, 0.7]
        trans = [[0.5, 0.5], [0.5, 0.5]]
        taus = [[0.6, 0.4], [0.1, 0.9]]
        hmm = HiddenMarkovModelDistribution(topics, w=w, transitions=trans, taus=taus)
        # log p(x[0]) = log sum_i w[i] * (sum_j taus[i,j] * topics[j].density(x[0]))
        expected = np.log(
            w[0] * (taus[0][0] * topics[0].density("a") + taus[0][1] * topics[1].density("a"))
            + w[1] * (taus[1][0] * topics[0].density("a") + taus[1][1] * topics[1].density("a"))
        )
        self.assertAlmostEqual(hmm.log_density(["a"]), float(expected), places=10)


if __name__ == "__main__":
    unittest.main()
