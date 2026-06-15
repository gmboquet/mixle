import unittest

import numpy as np

from pysp.models import (
    POMDPModel,
    TransEKnowledgeGraphModel,
    TruncatedDPMModel,
    baum_welch_pomdp,
    discrete_conditional_mutual_information,
    fit_induced_pcfg,
    fit_truncated_dpm,
    gaussian_conditional_independence,
    grammar_rule_table,
    learn_pc_skeleton,
    mean_stick_weights,
    orient_v_structures,
    pcfg_log_likelihood,
    sample_crp_assignments,
    stick_breaking_weights,
    viterbi_parse,
)
from pysp.stats import CategoricalDistribution, HeterogeneousPCFGDistribution
from pysp.stats.leaf.gaussian import GaussianDistribution, GaussianEstimator


class DPMModelHelpersTestCase(unittest.TestCase):
    def test_stick_breaking_and_crp_utilities(self):
        weights = stick_breaking_weights([0.5, 0.25])
        np.testing.assert_allclose(weights, [0.5, 0.125, 0.375])
        self.assertAlmostEqual(weights.sum(), 1.0)

        mean_weights = mean_stick_weights([[2.0, 3.0], [4.0, 2.0], [1.0, 1.0]])
        self.assertEqual(mean_weights.shape, (3,))
        self.assertAlmostEqual(mean_weights.sum(), 1.0)

        assignments, counts = sample_crp_assignments(25, alpha=0.75, seed=3)
        self.assertEqual(assignments.shape, (25,))
        self.assertEqual(counts.sum(), 25)
        self.assertGreaterEqual(counts.size, 1)

    def test_truncated_dpm_fit_improves_mixture_likelihood(self):
        rng = np.random.RandomState(4)
        data = list(rng.normal(-2.0, 0.25, size=35)) + list(rng.normal(2.0, 0.25, size=35))
        initial = [
            GaussianDistribution(-3.0, 1.0),
            GaussianDistribution(0.0, 1.0),
            GaussianDistribution(3.0, 1.0),
        ]
        initial_ll = sum(TruncatedDPMModel(initial, alpha=0.5).log_density(x) for x in data)

        result = fit_truncated_dpm(data, initial, GaussianEstimator(), alpha=0.5, max_its=20)
        final_ll = sum(result.model.log_density(x) for x in data)
        means = sorted(c.mu for c in result.model.components)

        self.assertGreater(final_ll, initial_ll)
        self.assertEqual(result.responsibilities.shape, (len(data), 3))
        np.testing.assert_allclose(result.responsibilities.sum(axis=1), np.ones(len(data)))
        self.assertLess(means[0], -1.5)
        self.assertGreater(means[-1], 1.5)


class POMDPModelHelpersTestCase(unittest.TestCase):
    def test_filtering_matches_first_step_by_hand(self):
        model = POMDPModel(
            transition=[[[0.9, 0.1], [0.2, 0.8]]],
            observation=[[[0.85, 0.15], [0.1, 0.9]]],
            initial_belief=[0.5, 0.5],
        )
        result = model.filter([0, 0, 0], [0, 1, 1])

        self.assertTrue(np.isfinite(result.log_likelihood))
        np.testing.assert_allclose(result.beliefs.sum(axis=1), np.ones(3))
        np.testing.assert_allclose(result.predictive_observation_probs[0], 0.5125)
        np.testing.assert_allclose(result.beliefs[0], [0.4675 / 0.5125, 0.045 / 0.5125])

    def test_baum_welch_pomdp_improves_likelihood(self):
        truth = POMDPModel(
            transition=[[[0.92, 0.08], [0.15, 0.85]]],
            observation=[[[0.9, 0.1], [0.2, 0.8]]],
            initial_belief=[0.6, 0.4],
        )
        actions = [0] * 40
        sequences = [(actions, truth.sample(actions, seed=i)[1]) for i in range(6)]
        initial = POMDPModel(
            transition=[[[0.55, 0.45], [0.45, 0.55]]],
            observation=[[[0.55, 0.45], [0.45, 0.55]]],
            initial_belief=[0.5, 0.5],
        )
        initial_ll = sum(initial.sequence_log_likelihood(a, o) for a, o in sequences)
        result = baum_welch_pomdp(sequences, 2, 1, 2, initial_model=initial, max_its=8, pseudo_count=0.1)
        final_ll = sum(result.model.sequence_log_likelihood(a, o) for a, o in sequences)

        self.assertGreater(final_ll, initial_ll)
        self.assertGreaterEqual(result.history[-1], result.history[0] - 1.0e-8)


class KnowledgeGraphHelpersTestCase(unittest.TestCase):
    def test_transe_margin_training_reduces_fixed_negative_loss(self):
        model = TransEKnowledgeGraphModel.random(
            3, 1, embedding_dim=4, seed=2, entity_names=["alice", "bob", "carol"], relation_names=["likes"]
        )
        positives = [("alice", "likes", "bob"), ("bob", "likes", "carol")]
        negatives = [("alice", "likes", "carol"), ("carol", "likes", "alice")]
        before = model.margin_loss(positives, negatives, margin=0.5)
        result = model.fit_margin(positives, negatives, margin=0.5, lr=0.03, max_its=80, seed=3)
        after = model.margin_loss(positives, negatives, margin=0.5)

        self.assertLess(after, before)
        self.assertLessEqual(result.history[-1], result.history[0])
        self.assertGreater(np.mean(model.score_triples(positives)), np.mean(model.score_triples(negatives)))


class GrammarLearningHelpersTestCase(unittest.TestCase):
    def test_viterbi_parse_matches_unambiguous_pcfg_log_density(self):
        model = HeterogeneousPCFGDistribution(
            binary_rules={"S": [("A", "B", 1.0)]},
            terminal_rules={
                "A": [(CategoricalDistribution({"a": 1.0}), 1.0)],
                "B": [(CategoricalDistribution({"b": 1.0}), 1.0)],
            },
            start="S",
        )
        parse = viterbi_parse(model, list("ab"))

        self.assertEqual(parse.label, "S")
        self.assertEqual(parse.span, (0, 2))
        self.assertEqual(parse.leaves(), list("ab"))
        self.assertAlmostEqual(parse.log_prob, model.log_density(list("ab")))
        self.assertEqual(len(grammar_rule_table(model)), 3)

    def test_fit_induced_pcfg_returns_finite_learned_grammar(self):
        data = [list("ab") for _ in range(25)] + [list("ba") for _ in range(5)]
        terminal_estimator = CategoricalDistribution({"a": 0.5, "b": 0.5}).estimator(pseudo_count=1.0)
        result = fit_induced_pcfg(
            data,
            [terminal_estimator],
            max_nonterminals=2,
            max_its=2,
            terminal_rule_mass=0.6,
            rule_pseudo_count=1.0e-3,
            prune_threshold=0.0,
            seed=7,
        )

        self.assertIsInstance(result.model, HeterogeneousPCFGDistribution)
        self.assertEqual(len(result.history), 3)
        self.assertTrue(np.all(np.isfinite(result.history)))
        self.assertGreater(pcfg_log_likelihood(result.model, [list("ab")]), -np.inf)


class DependenceAndCausalityHelpersTestCase(unittest.TestCase):
    def test_discrete_conditional_mutual_information_detects_dependence(self):
        data = np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]] * 20)
        self.assertGreater(discrete_conditional_mutual_information(data, 0, 1), 0.6)

    def test_gaussian_pc_skeleton_removes_conditionally_independent_chain_edge(self):
        rng = np.random.RandomState(5)
        x = rng.normal(size=1500)
        y = 0.9 * x + rng.normal(scale=0.2, size=1500)
        z = 0.9 * y + rng.normal(scale=0.2, size=1500)
        data = np.column_stack([x, y, z])

        ci = gaussian_conditional_independence(data, 0, 2, given=[1], alpha=0.05)
        skeleton = learn_pc_skeleton(data, alpha=0.05, max_cond_set=1)

        self.assertTrue(ci.independent)
        self.assertTrue(skeleton.has_edge(0, 1))
        self.assertTrue(skeleton.has_edge(1, 2))
        self.assertFalse(skeleton.has_edge(0, 2))

    def test_orient_v_structures_finds_gaussian_collider(self):
        rng = np.random.RandomState(6)
        x = rng.normal(size=1500)
        z = rng.normal(size=1500)
        xc = x - x.mean()
        z = z - z.mean()
        z = z - xc * np.dot(xc, z) / np.dot(xc, xc)
        y = x + z + rng.normal(scale=0.15, size=1500)
        data = np.column_stack([x, y, z])

        skeleton = learn_pc_skeleton(data, alpha=0.05, max_cond_set=1)
        graph = orient_v_structures(skeleton)

        self.assertFalse(skeleton.has_edge(0, 2))
        self.assertIn((0, 1), graph.directed_edges)
        self.assertIn((2, 1), graph.directed_edges)


if __name__ == "__main__":
    unittest.main()
