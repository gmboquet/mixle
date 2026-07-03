"""Heterogeneous Bayesian network learning (mixle.inference.bayesian_network): regression edges + multi-parent DAG.

The deepening of the dependency-structure moat: continuous dependence is parametric (linear-Gaussian), not
quantile-binned, and a field may have several parents. Must (a) capture continuous dependence with a parametric
edge (far beating independence), (b) recover a node's multiple parents when orientation is forced, and (c)
score/sample coherently. (The single-parent forest ``learn_structure`` now also uses parametric edges, so the
DAG's distinct advantage is multi-parent structure -- see ``MultiParentTest`` -- not parametric-vs-binned.)
"""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import fit
from mixle.inference.bayesian_network import (
    HeterogeneousBayesianNetwork,
    MixtureOfBayesianNetworks,
    learn_bayesian_network,
    learn_mixture_bayesian_network,
)


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


def _linear_pair(seed, n=800):
    r = np.random.RandomState(seed)
    x = r.randn(n)
    y = 2.0 * x - 1.0 + 0.3 * r.randn(n)
    return list(zip(x.tolist(), y.tolist()))


class RegressionEdgeTest(unittest.TestCase):
    def test_regression_edge_captures_linear_dependence(self):
        train, test = _linear_pair(1), _linear_pair(2)
        bn = learn_bayesian_network(train, max_parents=1)
        self.assertIsInstance(bn, HeterogeneousBayesianNetwork)
        self.assertEqual(len(bn.edges()), 1)  # the two fields are linked by a linear-Gaussian edge
        # the parametric edge crushes modeling the two fields independently (the dependence is worth many nats)
        indep = fit(
            train, st.CompositeEstimator((st.GaussianEstimator(), st.GaussianEstimator())), max_its=20, out=None
        )
        indep_ll = float(np.sum([indep.log_density(d) for d in test]))
        self.assertGreater(_ll(bn, test) - indep_ll, 300.0)

    def test_categorical_parent_via_one_hot(self):
        # cat -> real: the CLG factor one-hot-encodes the categorical parent (a per-category mean)
        r = np.random.RandomState(0)
        data = [("hi" if r.rand() < 0.5 else "lo", 0.0) for _ in range(600)]
        data = [(c, (5.0 if c == "hi" else -5.0) + 0.5 * r.randn()) for c, _ in data]
        bn = learn_bayesian_network(data, max_parents=1)
        self.assertIn((0, 1), bn.edges())


class MultiParentTest(unittest.TestCase):
    def test_discrete_child_recovers_both_parents(self):
        # two independent categoricals drive a count via their INTERACTION -> the count needs BOTH as parents
        r = np.random.RandomState(0)
        data = []
        for _ in range(2000):
            a, b = int(r.rand() < 0.5), int(r.rand() < 0.5)
            rate = 2.0 + 3.0 * a + 5.0 * b + 4.0 * a * b
            data.append((str(a), str(b), int(r.poisson(rate))))
        bn = learn_bayesian_network(data, max_parents=2)
        count_parents = {p for (p, c) in bn.edges() if c == 2}
        self.assertEqual(count_parents, {0, 1})  # rate[a,b] interaction -> both parents recovered

    def test_continuous_node_uses_multiple_parents(self):
        # y = f(x1, x2); a linear-Gaussian orientation is non-identifiable, so assert the orientation-robust facts:
        # some node has two parents, the graph is one connected component, and it beats independence.
        r = np.random.RandomState(1)
        n = 1200
        x1, x2 = r.randn(n), r.randn(n)
        y = 1.5 * x1 - 2.0 * x2 + 0.3 * r.randn(n)
        train = list(zip(x1.tolist(), x2.tolist(), y.tolist()))
        r2 = np.random.RandomState(2)
        z1, z2 = r2.randn(n), r2.randn(n)
        test = list(zip(z1.tolist(), z2.tolist(), (1.5 * z1 - 2.0 * z2 + 0.3 * r2.randn(n)).tolist()))

        bn = learn_bayesian_network(train, max_parents=2)
        self.assertTrue(any(len(f.parents) >= 2 for f in bn.factors))  # multi-parent used
        ind = fit(
            train,
            st.CompositeEstimator((st.GaussianEstimator(), st.GaussianEstimator(), st.GaussianEstimator())),
            max_its=30,
            out=None,
            rng=np.random.RandomState(0),
        )
        self.assertGreater(_ll(bn, test) - _ll(ind, test), 300.0)  # models the (x1,x2)->y dependence


class ScoreSampleTest(unittest.TestCase):
    def test_log_density_matches_seq(self):
        bn = learn_bayesian_network(_linear_pair(3), max_parents=1)
        rows = _linear_pair(4)[:40]
        seq = bn.seq_log_density(bn.dist_to_encoder().seq_encode(rows))
        for i, row in enumerate(rows):
            self.assertAlmostEqual(bn.log_density(row), float(seq[i]), places=6)

    def test_sampling_respects_dependence(self):
        bn = learn_bayesian_network(_linear_pair(5, n=1500), max_parents=1)
        rows = bn.sampler(0).sample(600)
        a = np.array([r[0] for r in rows])
        b = np.array([r[1] for r in rows])
        self.assertGreater(abs(np.corrcoef(a, b)[0, 1]), 0.8)  # the linear dependence is reproduced

    def test_independent_fields_stay_near_independence(self):
        r = np.random.RandomState(7)
        data = [(float(r.randn()), float(r.randn()), int(r.poisson(3))) for _ in range(800)]
        bn = learn_bayesian_network(data, max_parents=2)
        ind = fit(
            data,
            st.CompositeEstimator((st.GaussianEstimator(), st.GaussianEstimator(), st.PoissonEstimator())),
            max_its=30,
            out=None,
            rng=np.random.RandomState(0),
        )
        # no real structure -> the network does not meaningfully beat the independent composite in-sample
        self.assertLess(_ll(bn, data) - _ll(ind, data), 40.0)


def _slope_regimes(seed, n=1600):
    """Two clusters with OPPOSITE y-on-x slopes (separable by level) -- a per-cluster regression only a mixture gets."""
    r = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        z = r.randint(0, 2)
        x = r.randn()
        y = (2.0 * x + 6.0 if z == 0 else -2.0 * x - 6.0) + 0.3 * r.randn()
        out.append((float(x), float(y)))
    return out


class MixtureOfBayesianNetworksTest(unittest.TestCase):
    def test_captures_per_cluster_regression(self):
        train, test = _slope_regimes(1), _slope_regimes(2)
        mix = learn_mixture_bayesian_network(train, 2, restarts=3, seed=0)
        self.assertIsInstance(mix, MixtureOfBayesianNetworks)
        single = learn_bayesian_network(train, max_parents=1)
        self.assertGreater(_ll(mix, test) - _ll(single, test), 1000.0)  # a single DAG can't hold two slopes
        self.assertTrue(all(len(c.edges()) >= 1 for c in mix.components))  # each cluster learned its regression

    def test_responsibilities_recover_clusters(self):
        r = np.random.RandomState(3)
        rows, z = [], []
        for _ in range(1200):
            zi = r.randint(0, 2)
            x = r.randn()
            rows.append((float(x), float((2.0 * x + 6.0 if zi == 0 else -2.0 * x - 6.0) + 0.3 * r.randn())))
            z.append(zi)
        mix = learn_mixture_bayesian_network(rows, 2, restarts=3, seed=0)
        assign = mix.responsibilities(rows).argmax(axis=1)
        z = np.array(z)
        self.assertGreater(max((assign == z).mean(), (assign != z).mean()), 0.9)

    def test_samples_and_scores(self):
        mix = learn_mixture_bayesian_network(_slope_regimes(4), 2, restarts=2, seed=0)
        s = mix.sampler(0).sample(20)
        self.assertEqual(len(s), 20)
        self.assertEqual(len(s[0]), 2)
        self.assertTrue(np.isfinite(mix.log_density(s[0])))


if __name__ == "__main__":
    unittest.main()
