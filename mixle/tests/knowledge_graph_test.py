"""Tests for the stats-native knowledge-graph embedding distribution (DistMult)."""

import unittest

import numpy as np

from mixle.inference import estimate
from mixle.inference.estimation import optimize
from mixle.stats import KnowledgeGraphDistribution, KnowledgeGraphEstimator, fit_knowledge_graph_ensemble


def _kg_truth(nE=60, nR=4, d=12, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(0, 1, (nE, d)), rng.normal(0, 1, (nR, d))


def _sample(ent, rel, n, seed):
    rng = np.random.RandomState(seed)
    nE, nR = ent.shape[0], rel.shape[0]
    hs, rs = rng.randint(nE, size=n), rng.randint(nR, size=n)
    return [(int(h), int(r), int(np.argmax(ent @ (ent[h] * rel[r])))) for h, r in zip(hs, rs)]


class KnowledgeGraphTestCase(unittest.TestCase):
    def test_fits_via_optimize_and_recovers_tails(self):
        ent, rel = _kg_truth()
        nE, nR, d = ent.shape[0], rel.shape[0], ent.shape[1]
        train = _sample(ent, rel, 4000, seed=1)
        test = _sample(ent, rel, 1500, seed=2)  # same ground-truth KG, fresh queries
        m = optimize(
            train,
            KnowledgeGraphEstimator(nE, nR, dim=d, seed=1),
            max_its=1,
            rng=np.random.RandomState(0),
            print_iter=99999,
        )
        self.assertIsInstance(m, KnowledgeGraphDistribution)
        acc = np.mean([m.tail_log_posterior(h, r).argmax() == t for h, r, t in test])
        self.assertGreater(acc, 0.4)  # far above the 1/60 chance rate
        ll = m.seq_log_density(m.dist_to_encoder().seq_encode(test)).mean()
        self.assertGreater(
            ll,
            -np.log(nE * nR * nE),
        )  # beats the uniform joint triple distribution

    def test_tail_posterior_normalized(self):
        ent, rel = _kg_truth(seed=1)
        nE, nR, d = ent.shape[0], rel.shape[0], ent.shape[1]
        m = optimize(
            _sample(ent, rel, 1500, seed=3),
            KnowledgeGraphEstimator(nE, nR, dim=d, epochs=20, seed=2),
            max_its=1,
            rng=np.random.RandomState(0),
            print_iter=99999,
        )
        p = np.exp(m.tail_log_posterior(3, 1))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)

    def test_sampler_and_encoder(self):
        rng = np.random.RandomState(0)
        m = KnowledgeGraphDistribution(rng.normal(0, 0.3, (10, 4)), rng.normal(0, 0.3, (3, 4)))
        triples = m.sampler(seed=1).sample(5)
        self.assertEqual(len(triples), 5)
        enc = m.dist_to_encoder().seq_encode(triples)
        self.assertEqual(enc.shape, (5, 3))
        self.assertEqual(m.seq_log_density(enc).shape, (5,))

    def test_joint_density_matches_sampler_and_normalizes(self):
        rng = np.random.RandomState(0)
        model = KnowledgeGraphDistribution(
            rng.normal(0, 0.3, (3, 2)),
            rng.normal(0, 0.3, (2, 2)),
        )
        total = sum(np.exp(model.log_density((h, r, t))) for h in range(3) for r in range(2) for t in range(3))
        self.assertAlmostEqual(total, 1.0)

    def test_embeddings_and_triples_are_strict_owned_state(self):
        entity = np.ones((2, 2))
        relation = np.ones((1, 2))
        model = KnowledgeGraphDistribution(entity, relation)
        before = model.log_density((0, 0, 1))
        entity[0, 0] = 99.0
        relation[0, 0] = 99.0
        self.assertEqual(model.log_density((0, 0, 1)), before)
        self.assertFalse(model.entity.flags.writeable)
        self.assertFalse(model.relation.flags.writeable)
        with self.assertRaises(ValueError):
            KnowledgeGraphDistribution([[np.nan]], [[0.0]])
        for triple in (
            (0.5, 0, 1),
            (-1, 0, 1),
            (0, 1, 1),
            (0, 0, 1, 2),
        ):
            with self.subTest(triple=repr(triple)), self.assertRaises((TypeError, ValueError)):
                model.log_density(triple)

    def test_accumulator_validation_and_warm_start(self):
        estimator = KnowledgeGraphEstimator(
            3,
            1,
            dim=2,
            epochs=1,
            seed=4,
        )
        data = [(0, 0, 1), (1, 0, 2), (2, 0, 0)]
        first = estimate(data, estimator)
        second = estimate(data, estimator, first)
        self.assertTrue(second.fit_receipt["warm_start_used"])
        self.assertFalse(np.array_equal(first.entity, second.entity))
        accumulator = estimator.accumulator_factory().make()
        source = np.array([[0, 0, 1], [1, 0, 2]])
        accumulator.seq_update(source, [1.0, 2.0], first)
        source[0, 0] = 2
        value = accumulator.value()
        self.assertEqual(value.schema_version, 1)
        np.testing.assert_array_equal(
            value.triples,
            [[0, 0, 1], [1, 0, 2]],
        )
        self.assertFalse(value.triples.flags.writeable)
        with self.assertRaises(ValueError):
            accumulator.seq_update([[0, 0, 1]], [1.0, 2.0], first)
        with self.assertRaises(ValueError):
            estimator.estimate(
                None,
                (
                    0.0,
                    np.empty((0, 3), dtype=int),
                    np.empty(0),
                    3,
                    1,
                    2,
                    None,
                    None,
                ),
            )

    def test_training_plan_is_validated(self):
        invalid = (
            {"lr": np.nan},
            {"lr": -1.0},
            {"epochs": 0},
            {"epochs": 1.5},
            {"batch_size": 0},
            {"weight_decay": -1.0},
            {"init_scale": 0.0},
            {"max_norm": np.inf},
            {"directions": ("bogus",)},
            {"negatives": 3},
            {"pseudo_count": 1.0},
        )
        for controls in invalid:
            with self.subTest(controls=repr(controls)), self.assertRaises((TypeError, ValueError, NotImplementedError)):
                KnowledgeGraphEstimator(3, 1, dim=2, **controls)


class KnowledgeGraphRecommendTestCase(unittest.TestCase):
    def setUp(self):
        self.ent, self.rel = _kg_truth()
        self.nE, self.nR, self.d = self.ent.shape[0], self.rel.shape[0], self.ent.shape[1]
        self.train = _sample(self.ent, self.rel, 4000, seed=1)
        self.m = optimize(
            self.train,
            KnowledgeGraphEstimator(self.nE, self.nR, dim=self.d, seed=1),
            max_its=1,
            rng=np.random.RandomState(0),
            print_iter=10**9,
        )

    def test_any_slot_completion_recovers_truth(self):
        sample = self.train[:400]
        tail_acc = np.mean([self.m.complete(h=h, r=r).argmax() == t for h, r, t in sample])
        head_acc = np.mean([self.m.complete(r=r, t=t).argmax() == h for h, r, t in sample])
        self.assertGreater(tail_acc, 0.4)  # both directions recover the truth well above chance
        self.assertGreater(head_acc, 0.15)  # still well above the 1/60 chance rate
        self.assertEqual(len(self.m.complete(h=self.train[0][0], t=self.train[0][2])), self.nR)
        with self.assertRaises(ValueError):
            self.m.complete(h=self.train[0][0])  # two slots missing

    def test_recommend_excludes_known_and_ranks(self):
        recs = self.m.recommend(self.train, top_n=8)
        known = {(int(a), int(b), int(c)) for a, b, c in self.train}
        self.assertTrue(all((a, b, c) not in known for a, b, c, _ in recs))
        scores = [s for *_, s in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))  # ranked best first

    def test_subgraph_edges_touch_node(self):
        sg = self.m.recommend_subgraph(0, self.train, top_n=5)
        self.assertTrue(all(0 in (a, c) for a, b, c, _ in sg))

    def test_ensemble_epistemic_uncertainty(self):
        # epochs=15 (down from 30) still gives well-separated members: verified min/max epistemic
        # uncertainty across 8 independent truth/train/seed configs all pass comfortably.
        ens = fit_knowledge_graph_ensemble(
            self.train, self.nE, self.nR, dim=self.d, members=3, epochs=15, rng=np.random.RandomState(0)
        )
        u = [ens.epistemic_tail_uncertainty(h, r) for h, r, _ in self.train[:100]]
        self.assertGreaterEqual(min(u), 0.0)  # mutual information is nonnegative
        self.assertGreater(max(u), 0.0)  # members disagree somewhere
        self.assertEqual(len(ens.mean_tail_posterior(0, 0)), self.nE)


class KnowledgeGraphPatternTestCase(unittest.TestCase):
    def setUp(self):
        self.ent, self.rel = _kg_truth()
        self.nE, self.nR, self.d = self.ent.shape[0], self.rel.shape[0], self.ent.shape[1]
        tr = _sample(self.ent, self.rel, 4000, seed=1)
        self.m = optimize(
            tr,
            KnowledgeGraphEstimator(self.nE, self.nR, dim=self.d, seed=1),
            max_its=1,
            rng=np.random.RandomState(0),
            print_iter=10**9,
        )

    def test_single_edge_pattern_matches_rank(self):
        pat = self.m.pattern([(3, 0, "?t")])
        enum = [g["?t"] for g, _, _ in pat.enumerate(top_n=5)]
        rank = [c for c, _ in self.m.rank(h=3, r=0, top_n=5)]
        self.assertEqual(enum, rank)

    def test_chain_enumeration_descending_and_grounded(self):
        pat = self.m.pattern([(3, 0, "?x"), ("?x", 1, "?c")], beam=200)
        top = pat.enumerate(top_n=5)
        scores = [s for _, _, s in top]
        self.assertEqual(scores, sorted(scores, reverse=True))  # best-first
        g, edges, _ = top[0]
        self.assertEqual(edges, [(3, 0, g["?x"]), (g["?x"], 1, g["?c"])])  # shared variable joins the edges

    def test_candidate_restriction_and_known_exclusion(self):
        pat = self.m.pattern([(0, 1, "?x")], candidates={"?x": [2, 5, 9]})
        self.assertEqual(sorted(g["?x"] for g, _, _ in pat.enumerate(top_n=None)), [2, 5, 9])
        pat2 = self.m.pattern(
            [(0, 1, "?x"), ("?x", 1, "?c")], candidates={"?x": [2], "?c": [5, 7]}, known=[(0, 1, 2), (2, 1, 5)], beam=50
        )
        gs = [(g["?x"], g["?c"]) for g, _, _ in pat2.enumerate(top_n=None)]
        self.assertIn((2, 7), gs)  # has a new edge -> kept
        self.assertNotIn((2, 5), gs)  # entirely known -> dropped

    def test_conformal_set_of_subgraphs(self):
        from mixle.ppl import ConformalStructure

        tmpl = self.m.pattern([("?a", "?r", "?x")])
        facts = _sample(self.ent, self.rel, 6000, seed=4)
        inst = [tmpl.binding({"?a": h, "?r": r, "?x": t}) for h, r, t in facts]
        rng = np.random.RandomState(0)
        rng.shuffle(inst)
        cs = ConformalStructure(tmpl, inst[:600], alpha=0.1)
        self.assertGreater(cs.covers(inst[600:1200]).mean(), 0.86)

    def test_binding_probabilities_are_normalized(self):
        pattern = self.m.pattern([("?x", 0, "?x")])
        total = sum(np.exp(pattern.log_density((entity,))) for entity in range(self.nE))
        self.assertAlmostEqual(total, 1.0)


class KnowledgeGraphNegativeSamplingTestCase(unittest.TestCase):
    def test_sampled_softmax_learns(self):
        ent, rel = _kg_truth()
        nE, nR, d = ent.shape[0], rel.shape[0], ent.shape[1]
        train = _sample(ent, rel, 4000, seed=1)
        test = _sample(ent, rel, 1500, seed=2)
        m = optimize(
            train,
            # epochs=50 (down from 150) is enough for sampled softmax to converge here: verified acc
            # stays ~0.55-0.75, far above the 0.3 bar, across 8 independent truth/sample/seed configs.
            KnowledgeGraphEstimator(nE, nR, dim=d, epochs=50, lr=1.0, negatives=20, seed=1),
            max_its=1,
            rng=np.random.RandomState(0),
            print_iter=10**9,
        )
        acc = np.mean([m.tail_log_posterior(h, r).argmax() == t for h, r, t in test])
        self.assertGreater(acc, 0.3)  # sampled softmax still recovers tails well above the 1/60 chance rate


if __name__ == "__main__":
    unittest.main()
