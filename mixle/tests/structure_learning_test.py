"""Automatic dependency-structure learning (mixle.inference.structure): the tagline, actually delivered.

A CompositeDistribution models heterogeneous fields as independent; on data where they depend, that is badly
wrong. learn_structure must discover the dependency and fit a joint model that beats the independent composite
by a wide margin on held-out data -- and must NOT invent edges where the fields are truly independent.
"""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import fit
from mixle.inference.structure import (
    DependencyTreeDistribution,
    LinearGaussianEdge,
    MixtureOfDependencyTrees,
    dependency_gain,
    learn_mixture_structure,
    learn_structure,
    regression_gain,
)


def _dependent(seed, n=600):
    """(category -> shifts a real's mean) + an independent count field."""
    r = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        c = "hi" if r.rand() < 0.5 else "lo"
        x = (5.0 if c == "hi" else -5.0) + r.randn()
        k = int(r.poisson(3))
        out.append((c, float(x), k))
    return out


def _independent(seed, n=600):
    r = np.random.RandomState(seed)
    return [(str(r.randint(0, 3)), float(r.randn()), int(r.poisson(3))) for _ in range(n)]


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


def _composite(data):
    est = st.CompositeEstimator((st.CategoricalEstimator(), st.GaussianEstimator(), st.PoissonEstimator()))
    return fit(data, est, max_its=30, out=None)


class DependencyGainTest(unittest.TestCase):
    def test_positive_when_dependent(self):
        data = _dependent(0)
        cat = [d[0] for d in data]
        real = [d[1] for d in data]
        gain = dependency_gain(cat, real, st.GaussianEstimator())
        self.assertGreater(gain, 100.0)  # a strong category->real link is worth many nats

    def test_near_zero_when_independent(self):
        data = _independent(1)
        cat = [d[0] for d in data]
        real = [d[1] for d in data]
        gain = dependency_gain(cat, real, st.GaussianEstimator())
        self.assertLess(gain, 20.0)  # BIC penalty keeps a spurious edge from paying off


def _continuous_parent(seed, n=800):
    """A continuous field drives a count: rate = exp(field/3), plus an independent category."""
    r = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        x = float(r.uniform(-6, 6))
        k = int(r.poisson(np.exp(x / 3.0)))
        c = str(r.randint(0, 3))
        out.append((x, k, c))
    return out


class ContinuousParentTest(unittest.TestCase):
    def _composite_xc(self, data):
        est = st.CompositeEstimator((st.GaussianEstimator(), st.PoissonEstimator(), st.CategoricalEstimator()))
        return fit(data, est, max_its=30, out=None)

    def test_continuous_field_drives_count(self):
        train, test = _continuous_parent(1), _continuous_parent(2)
        model = learn_structure(train)
        self.assertIn((0, 1), model.edges())  # real(0) -> count(1) via quantile binning
        gain = _ll(model, test) - _ll(self._composite_xc(train), test)
        self.assertGreater(gain, 50.0)  # modeling the real->count link beats the independent composite

    def test_sampling_respects_continuous_dependence(self):
        model = learn_structure(_continuous_parent(3))
        rows = model.sampler(0).sample(600)
        xs = np.array([r[0] for r in rows])
        ks = np.array([r[1] for r in rows])
        # high-x rows have larger counts than low-x rows
        self.assertGreater(ks[xs > 2].mean(), ks[xs < -2].mean() + 1.0)


class LearnStructureTest(unittest.TestCase):
    def test_finds_the_edge_and_beats_composite(self):
        train, test = _dependent(1), _dependent(2)
        model = learn_structure(train)
        self.assertIsInstance(model, DependencyTreeDistribution)
        self.assertIn((0, 1), model.edges())  # category(0) -> real(1)
        # the independent count field is not spuriously attached to anything
        self.assertNotIn(2, [c for _p, c in model.edges()])
        gain = _ll(model, test) - _ll(_composite(train), test)
        self.assertGreater(gain, 300.0)  # dramatically better held-out likelihood

    def test_no_edges_when_independent(self):
        train = _independent(3)
        model = learn_structure(train)
        self.assertEqual(model.edges(), [])  # nothing to model -> falls back to independent marginals

    def test_scores_and_samples(self):
        model = learn_structure(_dependent(4))
        s = model.sampler(0).sample(20)
        self.assertEqual(len(s), 20)
        self.assertEqual(len(s[0]), 3)
        # a drawn record scores finite under the model
        self.assertTrue(np.isfinite(model.log_density(s[0])))
        # the dependency is respected: 'hi' rows sample high reals, 'lo' rows low
        big = model.sampler(1).sample(400)
        hi = np.mean([r[1] for r in big if r[0] == "hi"])
        lo = np.mean([r[1] for r in big if r[0] == "lo"])
        self.assertGreater(hi, lo + 3.0)

    def test_log_density_matches_seq(self):
        model = learn_structure(_dependent(5))
        rows = _dependent(6)[:50]
        seq = model.seq_log_density(model.dist_to_encoder().seq_encode(rows))
        for i, r in enumerate(rows):
            self.assertAlmostEqual(model.log_density(r), float(seq[i]), places=6)


def _two_regime(seed, n=1600):
    """Two clusters that differ in level (findable) AND both have a category->real dependence (structure)."""
    r = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        z = r.randint(0, 2)
        c = "hi" if r.rand() < 0.5 else "lo"
        base = 5.0 if z == 0 else -5.0
        out.append((c, float(base + (3.0 if c == "hi" else -3.0) + r.randn())))
    return out


class MixtureOfTreesTest(unittest.TestCase):
    def test_beats_single_tree_and_independent_mixture(self):
        train, test = _two_regime(1), _two_regime(2)
        # more restarts so the mixture EM reliably reaches the good optimum: the fit is stochastic and
        # its convergence differs across numpy/BLAS versions (Linux-x64 CI vs local), so a low restart
        # count let one cluster miss the dependency edge on CI while passing locally.
        mot = learn_mixture_structure(train, 2, restarts=12, seed=0)
        self.assertIsInstance(mot, MixtureOfDependencyTrees)

        tree_ll = _ll(learn_structure(train), test)
        ind = fit(
            train,
            st.MixtureEstimator(
                [st.CompositeEstimator((st.CategoricalEstimator(), st.GaussianEstimator())) for _ in range(2)]
            ),
            max_its=80,
            out=None,
            rng=np.random.RandomState(0),  # seed the baseline's EM init so the margin is order-independent
        )
        ind_ll = float(np.sum([ind.log_density(d) for d in test]))
        mot_ll = _ll(mot, test)

        self.assertGreater(mot_ll - tree_ll, 200.0)  # a single tree can't capture per-cluster structure
        self.assertGreater(mot_ll - ind_ll, 500.0)  # an independent mixture misses within-cluster dependence
        # both clusters recovered the category->real edge
        self.assertTrue(all((0, 1) in c.edges() for c in mot.components))

    def test_responsibilities_recover_clusters(self):
        # label each row by its regime and check the mixture's hard assignment separates them
        r = np.random.RandomState(7)
        rows, z = [], []
        for _ in range(1200):
            zi = r.randint(0, 2)
            c = "hi" if r.rand() < 0.5 else "lo"
            rows.append((c, float((5.0 if zi == 0 else -5.0) + (3.0 if c == "hi" else -3.0) + r.randn())))
            z.append(zi)
        mot = learn_mixture_structure(rows, 2, restarts=4, seed=0)
        assign = mot.responsibilities(rows).argmax(axis=1)
        z = np.array(z)
        purity = max((assign == z).mean(), (assign != z).mean())
        self.assertGreater(purity, 0.9)

    def test_samples_and_scores(self):
        mot = learn_mixture_structure(_two_regime(3), 2, restarts=3, seed=0)
        s = mot.sampler(0).sample(50)
        self.assertEqual(len(s), 50)
        self.assertEqual(len(s[0]), 2)
        self.assertTrue(np.isfinite(mot.log_density(s[0])))

    def test_log_density_matches_seq(self):
        mot = learn_mixture_structure(_two_regime(4), 2, restarts=2, seed=0)
        rows = _two_regime(5)[:40]
        seq = mot.seq_log_density(mot.dist_to_encoder().seq_encode(rows))
        for i, row in enumerate(rows):
            self.assertAlmostEqual(mot.log_density(row), float(seq[i]), places=6)


def _linear(seed, n=400):
    """A linear continuous dependence y ~ 2x + noise, plus an independent categorical field."""
    r = np.random.RandomState(seed)
    x = r.normal(0.0, 2.0, n)
    y = 2.0 * x + r.normal(0.0, 0.5, n)
    z = r.choice(["a", "b", "c"], n)
    return list(zip(x.tolist(), y.tolist(), z.tolist()))


class RegressionEdgeTest(unittest.TestCase):
    """A continuous->continuous dependence should be modeled by a linear-Gaussian REGRESSION edge (1 slope param),
    not a coarse per-bin conditional -- more accurate AND more parsimonious."""

    def test_regression_edge_is_chosen_and_wins(self):
        data = _linear(0)
        train, test = data[:300], data[300:]
        model = learn_structure(train)

        # the continuous fields are linked, and the child edge is a regression (not a binned conditional)
        child_edges = [(p, i) for i, p in enumerate(model.parents) if p is not None]
        self.assertEqual(len(child_edges), 1)
        child = child_edges[0][1]
        self.assertIsInstance(model.factors[child], LinearGaussianEdge)

        # decisively beats the independent composite on held-out data
        marg = [
            fit(col, est, max_its=30, out=None)
            for col, est in zip(
                zip(*train), (st.GaussianEstimator(), st.GaussianEstimator(), st.CategoricalEstimator())
            )
        ]
        independent = DependencyTreeDistribution([None, None, None], marg)
        self.assertGreater(_ll(model, test), _ll(independent, test) + 50.0)

    def test_regression_gain_beats_binned_on_linear_data(self):
        data = _linear(1)
        x = [r[0] for r in data]
        y = [r[1] for r in data]
        binner_keys = [round(v, 1) for v in x]  # a crude discretization of the parent
        binned = dependency_gain(binner_keys, y, st.GaussianEstimator())
        regression = regression_gain(x, y, st.GaussianEstimator())
        self.assertGreater(regression, binned)  # 1 slope param beats a per-bin conditional

    def test_regression_edge_samples_and_scores(self):
        edge = LinearGaussianEdge(1.0, 2.0, 0.25)
        self.assertTrue(np.isfinite(edge.log_density((3.0, 7.0))))
        s = edge.sampler(0).sample_given(3.0)  # E[child | parent=3] = 1 + 2*3 = 7
        self.assertAlmostEqual(s, 7.0, delta=2.0)
        seq = edge.seq_log_density((np.array([0.0, 1.0]), np.array([1.0, 3.0])))
        self.assertEqual(len(seq), 2)


if __name__ == "__main__":
    unittest.main()
