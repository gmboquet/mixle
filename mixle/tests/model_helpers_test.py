import unittest

import numpy as np

from mixle.models import (
    PartiallyObservableMarkovDecisionProcessModel,
    TransEKnowledgeGraphModel,
    TruncatedDirichletProcessMixtureModel,
    baum_welch_pomdp,
    discrete_conditional_independence,
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
from mixle.stats import CategoricalDistribution, HeterogeneousPCFGDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution, GaussianEstimator


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
        initial_ll = sum(TruncatedDirichletProcessMixtureModel(initial, alpha=0.5).log_density(x) for x in data)

        result = fit_truncated_dpm(data, initial, GaussianEstimator(), alpha=0.5, max_its=20)
        final_ll = sum(result.model.log_density(x) for x in data)
        means = sorted(c.mu for c in result.model.components)

        self.assertGreater(final_ll, initial_ll)
        self.assertEqual(result.responsibilities.shape, (len(data), 3))
        np.testing.assert_allclose(result.responsibilities.sum(axis=1), np.ones(len(data)))
        self.assertLess(means[0], -1.5)
        self.assertGreater(means[-1], 1.5)


class PartiallyObservableMarkovDecisionProcessModelHelpersTestCase(unittest.TestCase):
    def test_filtering_matches_first_step_by_hand(self):
        model = PartiallyObservableMarkovDecisionProcessModel(
            transition=[[[0.9, 0.1], [0.2, 0.8]]],
            observation=[[[0.85, 0.15], [0.1, 0.9]]],
            initial_belief=[0.5, 0.5],
        )
        result = model.filter([0, 0, 0], [0, 1, 1])

        self.assertTrue(np.isfinite(result.log_likelihood))
        np.testing.assert_allclose(result.beliefs.sum(axis=1), np.ones(3))
        np.testing.assert_allclose(result.predictive_observation_probs[0], 0.5125)
        np.testing.assert_allclose(result.beliefs[0], [0.4675 / 0.5125, 0.045 / 0.5125])

    def test_forward_backward_matches_brute_force_controlled_paths(self):
        model = PartiallyObservableMarkovDecisionProcessModel(
            transition=[
                [[0.8, 0.2], [0.3, 0.7]],
                [[0.55, 0.45], [0.15, 0.85]],
            ],
            observation=[
                [[0.9, 0.1], [0.25, 0.75]],
                [[0.35, 0.65], [0.8, 0.2]],
            ],
            initial_belief=[0.6, 0.4],
        )
        actions = [0, 1]
        observations = [1, 0]
        gamma, xi, ll = model.forward_backward(actions, observations)

        weights = np.zeros((2, 2, 2))
        for s0 in range(2):
            for s1 in range(2):
                for s2 in range(2):
                    weights[s0, s1, s2] = (
                        model.initial_belief[s0]
                        * model.transition[0, s0, s1]
                        * model.observation[0, s1, 1]
                        * model.transition[1, s1, s2]
                        * model.observation[1, s2, 0]
                    )
        z = weights.sum()
        np.testing.assert_allclose(ll, np.log(z))
        np.testing.assert_allclose(xi[0], weights.sum(axis=2) / z)
        np.testing.assert_allclose(xi[1], weights.sum(axis=0) / z)
        np.testing.assert_allclose(gamma[0], xi[0].sum(axis=0))
        np.testing.assert_allclose(gamma[1], xi[1].sum(axis=0))
        np.testing.assert_allclose(xi[0].sum(axis=1), weights.sum(axis=(1, 2)) / z)

    def test_baum_welch_pomdp_improves_likelihood(self):
        truth = PartiallyObservableMarkovDecisionProcessModel(
            transition=[[[0.92, 0.08], [0.15, 0.85]]],
            observation=[[[0.9, 0.1], [0.2, 0.8]]],
            initial_belief=[0.6, 0.4],
        )
        actions = [0] * 40
        sequences = [(actions, truth.sample(actions, seed=i)[1]) for i in range(6)]
        initial = PartiallyObservableMarkovDecisionProcessModel(
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

    def test_fit_rejects_invalid_budget_data_and_schema(self):
        estimator = CategoricalDistribution({"a": 0.5, "b": 0.5}).estimator(pseudo_count=1.0)
        for max_its in (0, -1, 1.5, True):
            with self.subTest(max_its=max_its), self.assertRaises(ValueError):
                fit_induced_pcfg([["a"]], [estimator], 2, max_its=max_its)
        for data in ([], [[]], ["ab"]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                fit_induced_pcfg(data, [estimator], 2)
        with self.assertRaises(TypeError):
            fit_induced_pcfg([["a"]], [object()], 2)
        with self.assertRaises(ValueError):
            fit_induced_pcfg([["a"]], [estimator], 2, init_p=0)
        with self.assertRaises(ValueError):
            fit_induced_pcfg([["a", "b"]], [estimator], 2, terminal_rule_mass=1)

    def test_fit_rejects_incompatible_initial_grammar(self):
        estimator = CategoricalDistribution({"a": 0.5, "b": 0.5}).estimator(pseudo_count=1.0)
        incompatible = HeterogeneousPCFGDistribution(
            binary_rules={"S": [("A", "A", 0.5)]},
            terminal_rules={"S": [(CategoricalDistribution({"a": 1.0}), 0.5)]},
            start="S",
        )
        with self.assertRaisesRegex(ValueError, "induced grammar skeleton"):
            fit_induced_pcfg([["a"]], [estimator], 2, initial_model=incompatible)


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

    def test_insufficient_gaussian_evidence_is_inconclusive(self):
        result = gaussian_conditional_independence([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]], 0, 1)
        self.assertEqual(result.status, "inconclusive")
        self.assertIsNone(result.independent)
        self.assertIsNone(result.p_value)

    def test_discrete_test_uses_a_calibrated_p_value(self):
        rng = np.random.RandomState(9)
        independent = np.column_stack([rng.randint(0, 2, 2000), rng.randint(0, 2, 2000)])
        dependent_x = rng.randint(0, 2, 2000)
        dependent = np.column_stack([dependent_x, dependent_x])
        null_result = discrete_conditional_independence(independent, 0, 1, alpha=0.05)
        alternative_result = discrete_conditional_independence(dependent, 0, 1, alpha=0.05)
        self.assertEqual(null_result.status, "independent")
        self.assertGreater(null_result.p_value, 0.05)
        self.assertEqual(alternative_result.status, "dependent")
        self.assertLess(alternative_result.p_value, 0.05)

    def test_discrete_pc_uses_significance_not_mutual_information_scale(self):
        rng = np.random.RandomState(11)
        x = rng.randint(0, 2, 12000)
        y = np.where(rng.random_sample(len(x)) < 0.45, 1 - x, x)
        data = np.column_stack([x, y])
        self.assertLess(discrete_conditional_mutual_information(data, 0, 1), 0.05)
        self.assertEqual(discrete_conditional_independence(data, 0, 1).status, "dependent")
        skeleton = learn_pc_skeleton(data, alpha=0.05, max_cond_set=0, method="discrete")
        self.assertTrue(skeleton.has_edge(0, 1))

    def test_ci_problem_schema_is_validated(self):
        data = np.arange(30.0).reshape(10, 3)
        invalid = (
            (0, 0, ()),
            (0, 1, (0,)),
            (0, 1, (2, 2)),
            (0, 3, ()),
        )
        for x, y, given in invalid:
            with self.subTest(x=x, y=y, given=given), self.assertRaises(ValueError):
                gaussian_conditional_independence(data, x, y, given=given)
        with self.assertRaises(ValueError):
            gaussian_conditional_independence(data, 0, 1, ridge=0)
        with self.assertRaises(ValueError):
            gaussian_conditional_independence(data, 0, 1, alpha=1)

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

    def test_orientation_rejects_conflicting_collider_evidence(self):
        from mixle.models import CausalSkeleton

        skeleton = CausalSkeleton(
            edges={(0, 1), (0, 2), (1, 3)},
            separating_sets={(1, 2): frozenset(), (0, 3): frozenset()},
            variable_names=["a", "b", "c", "d"],
            all_separating_sets={
                (1, 2): (frozenset(),),
                (0, 3): (frozenset(),),
            },
        )
        with self.assertRaisesRegex(ValueError, "conflicting orientations"):
            orient_v_structures(skeleton)


if __name__ == "__main__":
    unittest.main()
