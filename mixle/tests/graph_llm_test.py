"""KG-producing LLM: UQ on the information by marginalizing over graphs (mixle.reason.graph_llm)."""

import unittest

import numpy as np

from mixle.reason.graph_llm import GraphDistribution, GraphLLM, canonical_graph


def _dist(pairs):
    """Build a GraphDistribution from (triples, prob) pairs."""
    graphs = [canonical_graph(t) for t, _ in pairs]
    probs = np.array([p for _, p in pairs], dtype=float)
    return GraphDistribution(graphs, probs / probs.sum())


# a knowledge graph as a set of (subject, relation, object) triples
GA = [("eiffel", "height", 330), ("eiffel", "city", "paris")]
GB = [("eiffel", "height", 330), ("eiffel", "city", "lyon")]  # hallucinated city


class MarginalizationTest(unittest.TestCase):
    def setUp(self):
        self.d = _dist([(GA, 0.7), (GB, 0.3)])

    def test_edge_marginals_are_per_fact_reliability(self):
        m = self.d.edge_marginals()
        # the height fact is in every graph -> reliable; the city splits 0.7 / 0.3
        self.assertAlmostEqual(m[("eiffel", "height", 330)], 1.0, places=10)
        self.assertAlmostEqual(m[("eiffel", "city", "paris")], 0.7, places=10)
        self.assertAlmostEqual(m[("eiffel", "city", "lyon")], 0.3, places=10)

    def test_marginalize_over_subgraphs_producing_an_outcome(self):
        # outcome = "which city does the graph assert?" -> marginalize P(G) over graphs giving each
        city = lambda g: next((o for (s, r, o) in g if r == "city"), None)  # noqa: E731
        out = dict(self.d.marginalize(city))
        self.assertAlmostEqual(out["paris"], 0.7, places=10)
        self.assertAlmostEqual(out["lyon"], 0.3, places=10)
        # entropy of the city query
        self.assertAlmostEqual(self.d.entropy(city), -(0.7 * np.log(0.7) + 0.3 * np.log(0.3)), places=10)

    def test_query_completion_posterior(self):
        post = dict(self.d.query("eiffel", "city"))
        self.assertAlmostEqual(post["paris"], 0.7, places=10)
        self.assertAlmostEqual(post["lyon"], 0.3, places=10)

    def test_fact_probability(self):
        self.assertAlmostEqual(self.d.fact_probability(("eiffel", "height", 330)), 1.0, places=10)
        self.assertAlmostEqual(self.d.fact_probability(("eiffel", "city", "lyon")), 0.3, places=10)
        self.assertEqual(self.d.fact_probability(("eiffel", "city", "berlin")), 0.0)

    def test_most_likely_graph(self):
        g, p = self.d.most_likely_graph()
        self.assertEqual(g, canonical_graph(GA))
        self.assertAlmostEqual(p, 0.7, places=10)


class GraphLLMTest(unittest.TestCase):
    def _kg_llm(self, city_prob_paris, seed=0):
        rng = np.random.RandomState(seed)

        def generate(prompt):
            city = "paris" if rng.random() < city_prob_paris else "lyon"
            return f"height:330;city:{city}"  # a structured generation

        def parse(text):
            for kv in text.split(";"):
                k, v = kv.split(":")
                yield ("eiffel", k, int(v) if v.isdigit() else v)

        return GraphLLM(generate, parse, n=400)

    def test_distribution_recovers_graph_frequencies(self):
        d = self._kg_llm(0.75).distribution("describe the eiffel tower")
        # exact equivalence (graph identity) -> exactly two distinct graphs
        self.assertEqual(len(d.graphs), 2)
        m = d.edge_marginals()
        self.assertAlmostEqual(m[("eiffel", "height", 330)], 1.0, places=10)
        # MC estimate of the city marginal ~ 0.75
        self.assertAlmostEqual(m[("eiffel", "city", "paris")], 0.75, delta=0.06)

    def test_confident_graph_low_query_entropy(self):
        sure = self._kg_llm(0.98).distribution("q")
        unsure = self._kg_llm(0.5).distribution("q", n=400)
        city = lambda g: next((o for (s, r, o) in g if r == "city"), None)  # noqa: E731
        self.assertLess(sure.entropy(city), unsure.entropy(city))


class LogProbMarginalizationTest(unittest.TestCase):
    def test_summing_sequence_likelihoods_differs_from_counting(self):
        # Two samples parse to graph A, one to graph B. Counting says P(A)=2/3. But if the single B
        # string carries almost all the probability MASS, marginalizing strings->graphs must flip it.
        gs = [canonical_graph(GA), canonical_graph(GA), canonical_graph(GB)]
        d_count = GraphLLM(lambda p: "", lambda t: []).distribution("q", graphs=gs)
        pA_count = d_count.fact_probability(("eiffel", "city", "paris"))
        self.assertAlmostEqual(pA_count, 2 / 3, places=6)

        log_probs = [np.log(0.01), np.log(0.01), np.log(0.5)]  # B string is far more likely
        d_lp = GraphLLM(lambda p: "", lambda t: []).distribution("q", graphs=gs, log_probs=log_probs)
        pA_lp = d_lp.fact_probability(("eiffel", "city", "paris"))
        pB_lp = d_lp.fact_probability(("eiffel", "city", "lyon"))
        # marginalizing over the actual string probabilities makes B dominant, unlike counting
        self.assertLess(pA_lp, 0.1)
        self.assertGreater(pB_lp, 0.9)
        self.assertAlmostEqual(pA_lp + pB_lp, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
