import unittest

import numpy as np
from scipy.special import logsumexp

from pysp.stats import (
    CategoricalDistribution,
    GaussianDistribution,
    HeterogeneousPCFGDistribution,
    HeterogeneousPCFGEstimator,
    InducedHeterogeneousPCFGEstimator,
    SequenceDistribution,
    SequenceEstimator,
    StudentTDistribution,
    seq_encode,
    seq_estimate,
    seq_initialize,
)
from pysp.stats.compute.pdist import EnumerationError


class HeterogeneousPCFGTestCase(unittest.TestCase):
    def make_numeric_model(self):
        return HeterogeneousPCFGDistribution(
            binary_rules={
                "S": [("A", "B", 0.4), ("B", "A", 0.6)],
            },
            terminal_rules={
                "A": [(GaussianDistribution(-1.0, 0.8), 1.0)],
                "B": [(StudentTDistribution(5.0, loc=1.0, scale=1.2), 1.0)],
            },
            start="S",
            name="pcfg",
        )

    def test_log_density_matches_brute_force_for_ambiguous_binary_rules(self):
        model = self.make_numeric_model()
        x = [-0.8, 1.1]
        a = model.emissions[0]
        b = model.emissions[1]
        expected = logsumexp(
            [
                np.log(0.4) + a.log_density(x[0]) + b.log_density(x[1]),
                np.log(0.6) + b.log_density(x[0]) + a.log_density(x[1]),
            ]
        )
        self.assertAlmostEqual(model.log_density(x), expected, places=10)

    def test_seq_log_density_matches_scalar(self):
        model = self.make_numeric_model()
        data = [[-1.0, 1.0], [1.25, -0.75], [-0.5, 1.5]]
        enc = model.dist_to_encoder().seq_encode(data)
        seq_ll = model.seq_log_density(enc)
        scalar_ll = np.asarray([model.log_density(x) for x in data])
        self.assertTrue(np.allclose(seq_ll, scalar_ll, rtol=1.0e-12, atol=1.0e-12))

    def test_sample_and_em_step_with_heterogeneous_terminal_classes(self):
        model = self.make_numeric_model()
        data = model.sampler(11).sample(80)
        self.assertTrue(all(len(x) == 2 for x in data))
        self.assertTrue(np.all(np.isfinite([model.log_density(x) for x in data])))

        est = HeterogeneousPCFGEstimator(
            binary_rules={"S": [("A", "B", 0.5), ("B", "A", 0.5)]},
            terminal_rules={
                "A": [(GaussianDistribution(0.0, 4.0).estimator(), 1.0)],
                "B": [(StudentTDistribution(5.0).estimator(), 1.0)],
            },
            start="S",
            pseudo_count=1.0,
        )
        enc = seq_encode(data, model=model)
        fitted = seq_estimate(enc, est, model)
        self.assertIsInstance(fitted, HeterogeneousPCFGDistribution)
        self.assertAlmostEqual(float(fitted.binary_probs.sum()), 1.0)
        self.assertTrue(np.all(np.isfinite(fitted.seq_log_density(enc[0][1]))))

    def test_sequence_valued_terminal_emissions(self):
        model = HeterogeneousPCFGDistribution(
            binary_rules={"S": [("A", "B", 1.0)]},
            terminal_rules={
                "A": [
                    (
                        SequenceDistribution(
                            GaussianDistribution(-2.0, 0.5), len_dist=CategoricalDistribution({1: 1.0})
                        ),
                        1.0,
                    )
                ],
                "B": [
                    (
                        SequenceDistribution(
                            GaussianDistribution(2.0, 0.5), len_dist=CategoricalDistribution({2: 1.0})
                        ),
                        1.0,
                    )
                ],
            },
            start="S",
        )
        data = model.sampler(7).sample(20)
        for obs in data:
            self.assertEqual(len(obs), 2)
            self.assertTrue(all(isinstance(token, list) for token in obs))
            self.assertTrue(np.isfinite(model.log_density(obs)))

        est = HeterogeneousPCFGEstimator(
            binary_rules={"S": [("A", "B", 1.0)]},
            terminal_rules={
                "A": [
                    (
                        SequenceEstimator(
                            GaussianDistribution(-1.0, 2.0).estimator(),
                            len_estimator=CategoricalDistribution({1: 1.0}).estimator(),
                        ),
                        1.0,
                    )
                ],
                "B": [
                    (
                        SequenceEstimator(
                            GaussianDistribution(1.0, 2.0).estimator(),
                            len_estimator=CategoricalDistribution({2: 1.0}).estimator(),
                        ),
                        1.0,
                    )
                ],
            },
            start="S",
            pseudo_count=1.0,
        )
        enc = seq_encode(data, model=model)
        fitted = seq_estimate(enc, est, model)
        self.assertTrue(np.all(np.isfinite(fitted.seq_log_density(enc[0][1]))))

    def test_discrete_enumerator_and_quantized_index(self):
        model = HeterogeneousPCFGDistribution(
            binary_rules={"S": [("A", "B", 1.0)]},
            terminal_rules={
                "A": [(CategoricalDistribution({"a": 0.75, "b": 0.25}), 1.0)],
                "B": [(CategoricalDistribution({"x": 0.5, "y": 0.5}), 1.0)],
            },
            start="S",
        )

        top = model.enumerator().top_k(5)
        self.assertEqual(
            [u[0] for u in top],
            [
                ["a", "x"],
                ["a", "y"],
                ["b", "x"],
                ["b", "y"],
            ],
        )
        for value, log_prob in top:
            self.assertAlmostEqual(log_prob, model.log_density(value), places=12)

        index = model.quantized_index(max_bits=4)
        self.assertEqual(index.counts, {2: 2, 3: 2})
        self.assertEqual(index.total_count, 4)
        indexed = list(index.iter_from())
        self.assertEqual([u[0] for u in indexed], [u[0] for u in top])
        self.assertEqual(index.slice(1, 2), indexed[1:3])
        for value, log_prob in indexed:
            self.assertAlmostEqual(log_prob, model.log_density(value), places=12)

    def test_recursive_pcfg_quantized_index_uses_bounded_dp(self):
        model = HeterogeneousPCFGDistribution(
            binary_rules={"S": [("S", "A", 0.5)]},
            terminal_rules={
                "S": [(CategoricalDistribution({"z": 1.0}), 0.5)],
                "A": [(CategoricalDistribution({"a": 1.0}), 1.0)],
            },
            start="S",
        )

        with self.assertRaises(EnumerationError):
            model.enumerator()

        index = model.quantized_index(max_bits=4)
        self.assertEqual(index.counts, {1: 1, 2: 1, 3: 1, 4: 1})
        self.assertEqual(
            [u[0] for u in index.iter_from()],
            [
                ["z"],
                ["z", "a"],
                ["z", "a", "a"],
                ["z", "a", "a", "a"],
            ],
        )
        for value, log_prob in index.iter_from():
            self.assertAlmostEqual(log_prob, model.log_density(value), places=12)

    def test_induced_estimator_learns_overcomplete_sparse_grammar(self):
        a_dist = CategoricalDistribution({"a": 1.0})
        b_dist = CategoricalDistribution({"b": 1.0})
        est = InducedHeterogeneousPCFGEstimator(
            max_nonterminals=3,
            terminal_estimators=[a_dist.estimator(pseudo_count=1.0), b_dist.estimator(pseudo_count=1.0)],
            terminal_rule_mass=0.45,
            rule_pseudo_count=1.0e-3,
            prune_threshold=1.0e-9,
            min_rule_prob=1.0e-8,
        )
        model0 = est.initial_model([a_dist, b_dist], rng=np.random.RandomState(1), jitter=0.1)
        data = [list("ab") for _ in range(40)]
        enc = seq_encode(data, model=model0)
        model1 = seq_estimate(enc, est, model0)

        self.assertIsInstance(model1, HeterogeneousPCFGDistribution)
        self.assertEqual(model1.num_binary_rules, 27)
        self.assertEqual(model1.num_terminal_rules, 6)
        self.assertGreater(model1.log_density(list("ab")), -np.inf)
        self.assertLess(np.count_nonzero(model1.binary_probs > 0.0), model1.num_binary_rules)

        index = model1.quantized_index(max_bits=20)
        self.assertGreater(index.total_count, 0)
        self.assertTrue(any(value == list("ab") for value, _ in index.iter_from()))

    def test_induced_estimator_can_initialize_without_initial_grammar(self):
        est = InducedHeterogeneousPCFGEstimator(
            max_nonterminals=2,
            terminal_estimators=[CategoricalDistribution({"a": 0.5, "b": 0.5}).estimator(pseudo_count=1.0)],
            terminal_rule_mass=0.6,
            rule_pseudo_count=1.0e-3,
            prune_threshold=0.0,
        )
        data = [list("a"), list("b"), list("ab"), list("ba")]
        enc = seq_encode(data, estimator=est)
        model = seq_initialize(enc, est, np.random.RandomState(3), p=1.0)

        self.assertIsInstance(model, HeterogeneousPCFGDistribution)
        self.assertEqual(model.num_binary_rules, 8)
        self.assertEqual(model.num_terminal_rules, 2)
        self.assertTrue(np.all(np.isfinite(model.seq_log_density(enc[0][1]))))


if __name__ == "__main__":
    unittest.main()
