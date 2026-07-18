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
from mixle.utils.serialization import from_json, to_json


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
        # two independent categoricals drive a count via their INTERACTION -> the count needs BOTH as parents.
        # n=2000 leaves this orientation (a,b->count) tied against an equally-fitting alternative DAG in the
        # same equivalence class (a->count, a->b, count->b) for this seed -- a fixed parameter-count bug
        # (_num_free_params misdetecting NegativeBinomialDistribution/CategoricalDistribution params) used to
        # happen to break the tie the "expected" way here by coincidence. 6000 rows makes the true a,b->count
        # dependence dominate reliably (confirmed stable at n>=6000; n=4000 still flips for this seed).
        r = np.random.RandomState(0)
        data = []
        for _ in range(6000):
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


class SoftEMTest(unittest.TestCase):
    """em='soft': responsibility-weighted structure + factor fits; BIC selects the cluster count."""

    def test_soft_em_matches_hard_on_separated_regimes(self):
        rows = _slope_regimes(4, n=1200)
        soft = learn_mixture_bayesian_network(rows, 2, em="soft", restarts=2, seed=0)
        hard = learn_mixture_bayesian_network(rows, 2, em="hard", restarts=2, seed=0)
        self.assertGreater(_ll(soft, rows), _ll(hard, rows) - 5.0)  # no floor-contamination regression

    def test_soft_em_recovers_regimes(self):
        r = np.random.RandomState(7)
        rows, z = [], []
        for _ in range(1000):
            zi = r.randint(0, 2)
            x = r.randn()
            rows.append((float(x), float((2.0 * x + 6.0 if zi == 0 else -2.0 * x - 6.0) + 0.3 * r.randn())))
            z.append(zi)
        mix = learn_mixture_bayesian_network(rows, 2, em="soft", restarts=2, seed=0)
        assign = mix.responsibilities(rows).argmax(axis=1)
        z = np.array(z)
        self.assertGreater(max((assign == z).mean(), (assign != z).mean()), 0.95)

    def test_bic_selects_the_true_cluster_count(self):
        from mixle.inference.bayesian_network import select_mixture_components

        rows = _slope_regimes(4, n=1200)
        model, rep = select_mixture_components(rows, (1, 2, 3), em="soft", restarts=2, seed=0)
        self.assertEqual(rep["k"], 2)
        self.assertLess(rep["bic"][2], rep["bic"][1])
        self.assertLess(rep["bic"][2], rep["bic"][3])

    def test_invalid_em_raises(self):
        with self.assertRaises(ValueError):
            learn_mixture_bayesian_network(_slope_regimes(1, n=100), 2, em="fuzzy")


class GLMFactorTest(unittest.TestCase):
    """Continuous -> discrete edges (the direction the greedy search used to refuse outright)."""

    def test_continuous_drives_binary_edge_found_and_pays_held_out(self):
        rng = np.random.RandomState(0)
        x = rng.randn(600) * 1.5
        data = [(float(v), "hi" if rng.rand() < 1.0 / (1.0 + np.exp(-3.0 * (v - 0.5))) else "lo") for v in x]
        net = learn_bayesian_network(data)
        self.assertEqual(len(net.edges()), 1)  # the dependence is found (either orientation is valid:
        # logistic y|x with Gaussian x == shared-variance class-Gaussians x|y describe the same joint)
        fresh_x = rng.randn(300) * 1.5
        fresh = [(float(v), "hi" if rng.rand() < 1.0 / (1.0 + np.exp(-3.0 * (v - 0.5))) else "lo") for v in fresh_x]
        indep = learn_bayesian_network(data, max_parents=0)
        self.assertGreater(_ll(net, fresh), _ll(indep, fresh) + 50.0)

    def test_continuous_drives_count_poisson_link(self):
        rng = np.random.RandomState(1)
        x = rng.uniform(0.0, 2.0, 700)
        data = [(float(v), int(c)) for v, c in zip(x, rng.poisson(np.exp(0.8 * x + 0.2)))]
        net = learn_bayesian_network(data)
        self.assertEqual(len(net.edges()), 1)
        fresh_x = rng.uniform(0.0, 2.0, 300)
        fresh = [(float(v), int(c)) for v, c in zip(fresh_x, rng.poisson(np.exp(0.8 * fresh_x + 0.2)))]
        indep = learn_bayesian_network(data, max_parents=0)
        self.assertGreater(_ll(net, fresh), _ll(indep, fresh) + 30.0)
        # the fitted factor is the GLM node, and off-support child values score -inf, not nan
        glm_f = [f for f in net.factors if type(f).__name__ == "_GLMFactor"]
        if glm_f:
            self.assertEqual(glm_f[0].kind, "poisson")
            self.assertEqual(glm_f[0].log_density((1.0, -3)), -np.inf)

    def test_continuous_drives_three_way_categorical(self):
        rng = np.random.RandomState(2)
        x = rng.randn(900) * 2.0
        data = [(float(v), "low" if v < -0.8 else ("mid" if v < 0.8 else "high")) for v in x]
        net = learn_bayesian_network(data)
        self.assertEqual(len(net.edges()), 1)
        # separable bands stay finite (multinomial ridge) and sampling respects the bands
        rows = net.sampler(seed=3).sample(400)
        agree = sum(lab == ("low" if v < -0.8 else ("mid" if v < 0.8 else "high")) for v, lab in rows)
        self.assertGreater(agree / len(rows), 0.8)
        for v, lab in rows[:50]:
            self.assertTrue(np.isfinite(net.log_density((v, lab))))

    def test_mixed_parents_glm_uses_onehot_and_raw(self):
        # n=800 leaves the x->y vs y->x BIC gain a near-exact tie for this seed (129.49 vs 129.68 nats --
        # noise-level, not signal), which the greedy search can legitimately break either way; a fixed
        # parameter-count bug (_num_free_params misdetecting CategoricalDistribution.pmap params) used to
        # happen to break the tie the "expected" way here by pure coincidence. 1600 rows makes the true
        # x->y dependence dominate the tie reliably (confirmed stable at n>=1600 across the same seed).
        rng = np.random.RandomState(3)
        g = [["a", "b"][i] for i in rng.randint(0, 2, 1600)]
        x = rng.randn(1600)
        logit = 2.5 * x + np.where(np.asarray(g) == "b", 2.0, -2.0)
        y = ["t" if rng.rand() < 1.0 / (1.0 + np.exp(-z)) else "f" for z in logit]
        data = list(zip(g, x.tolist(), y))
        net = learn_bayesian_network(data)
        self.assertIn((1, 2), net.edges())  # the continuous driver reaches the discrete child


class VectorNodeTest(unittest.TestCase):
    """Vector-valued fields (embeddings) as first-class nodes: multivariate marginal / CLG, both directions."""

    def _records(self, n, seed):
        r = np.random.RandomState(seed)
        out = []
        for _ in range(n):
            cat = ["a", "b", "c"][r.randint(0, 3)]
            center = {"a": [2, 0, 0, 0], "b": [0, 2, 0, 0], "c": [0, 0, 2, 0]}[cat]
            vec = np.asarray(center, dtype=float) + 0.3 * r.randn(4)
            price = float(2.0 * vec[0] - 1.0 * vec[1] + 0.4 * r.randn())
            out.append((cat, vec, price))
        return out

    def test_vector_is_both_a_clg_child_and_a_continuous_parent(self):
        net = learn_bayesian_network(self._records(500, 0), max_parents=2)
        kinds = {f.child: type(f).__name__ for f in net.factors}
        self.assertEqual(kinds[1], "_VectorCLGFactor")  # the vector is driven by the category (multivariate CLG)
        self.assertIn((0, 1), net.edges())  # cat -> vector
        self.assertTrue(any(1 in f.parents for f in net.factors if f.child == 2))  # vector -> price

    def test_scores_and_samples_coherently(self):
        net = learn_bayesian_network(self._records(400, 0), max_parents=2)
        test = self._records(200, 1)
        ll = net.seq_log_density(net.dist_to_encoder().seq_encode(test))
        self.assertTrue(np.isfinite(ll).all())
        rows = net.sampler(seed=3).sample(5)
        self.assertEqual(np.asarray(rows[0][1]).shape, (4,))  # sampled vector has the right dim
        self.assertTrue(np.isfinite(net.log_density(rows[0])))

    def test_vector_marginal_when_independent(self):
        r = np.random.RandomState(2)
        data = [(float(r.randn()), (r.randn(3)).astype(float)) for _ in range(300)]  # scalar and vector, independent
        net = learn_bayesian_network(data, max_parents=1)
        vfac = [f for f in net.factors if f.child == 1]
        self.assertEqual(type(vfac[0]).__name__, "_VectorMarginalFactor")  # no spurious edge -> a bare MVN marginal
        self.assertEqual(net.edges(), [])


class NumFreeParamsTest(unittest.TestCase):
    """Regression: _num_free_params used to fall through to a flat constant 2 for any distribution whose
    parameters aren't named mu/p/lam/beta/alpha -- silently true for CategoricalDistribution (params live
    in .pmap) and CompositeDistribution (a structural wrapper with no scalar param attrs at all). This
    undercounted the BIC complexity penalty for categorical fields, letting learn_bayesian_network accept
    spurious edges between independent categorical fields once cardinality grew past a couple of levels."""

    def test_categorical_counts_k_minus_1_not_a_flat_constant(self):
        from mixle.inference.structure import _num_free_params

        for k in (2, 5, 20):
            col = [str(i % k) for i in range(400)]
            dist = fit(col, st.CategoricalEstimator(), max_its=5, out=None)
            with self.subTest(k=k):
                self.assertEqual(_num_free_params(dist), k - 1)

    def test_composite_sums_its_fields_not_a_flat_constant(self):
        from mixle.inference.structure import _num_free_params

        comp = st.CompositeDistribution((st.GaussianDistribution(0.0, 1.0), st.PoissonDistribution(2.0)))
        self.assertEqual(_num_free_params(comp), 3)  # 2 (Gaussian: mean+var) + 1 (Poisson: rate)

    def test_single_scalar_families_are_not_doubled(self):
        from mixle.inference.structure import _num_free_params

        self.assertEqual(_num_free_params(st.PoissonDistribution(3.0)), 1)
        self.assertEqual(_num_free_params(st.BernoulliDistribution(0.3)), 1)
        self.assertEqual(_num_free_params(st.GaussianDistribution(0.0, 1.0)), 2)

    def test_negative_binomial_counts_both_r_and_p(self):
        # NegativeBinomialEstimator fits both r and p by default (estimate_r=True) -- unlike Poisson/
        # Bernoulli/Binomial's single free scalar, this family genuinely has 2, not 1.
        from mixle.inference.structure import _num_free_params

        self.assertEqual(_num_free_params(st.NegativeBinomialDistribution(3.0, 0.4)), 2)

    def test_independent_categorical_fields_no_longer_produce_a_spurious_edge(self):
        # Two independently-permuted categorical columns (20 and 4 levels) have no real dependence; the
        # old flat-constant-2 penalty under-charged the extra per-parent-config categorical table enough
        # for the greedy search to accept a spurious edge here (confirmed against the pre-fix code).
        rng = np.random.RandomState(0)
        n = 400
        c1 = [f"L{i % 20}" for i in rng.permutation(n) % 20]
        c2 = [f"K{i % 4}" for i in rng.permutation(n) % 4]
        data = list(zip(c1, c2))
        net = learn_bayesian_network(data, max_parents=2, min_gain=0.0)
        self.assertEqual(net.edges(), [])


class SerializationTest(unittest.TestCase):
    """HeterogeneousBayesianNetwork + its factor classes round-trip through the safe JSON artifact path
    (mixle.utils.serialization), not just pickle -- worklist F10.1 needs a fresh-process save/reload with
    identical scores, and every factor kind the structure search can produce must actually support it."""

    def _records_with_missing_categorical(self, n=500, seed=0):
        # workclass-like field: a categorical parent with a missing sentinel (None), driving a continuous
        # child -- the exact shape real heterogeneous tabular data (e.g. UCI Adult) exercises.
        r = np.random.RandomState(seed)
        cats = ["Private", "Self-emp", "Gov"]
        out = []
        for _ in range(n):
            workclass = None if r.rand() < 0.1 else cats[r.randint(0, 3)]
            base = 45.0 if workclass == "Self-emp" else 38.0
            hours = float(r.randn() * 3 + base)
            income = ">50K" if hours > 42 else "<=50K"
            out.append((workclass, hours, income))
        return out

    def _assert_roundtrip_identical(self, net, probes):
        text = to_json(net)
        reloaded = from_json(text)
        self.assertIsInstance(reloaded, type(net))
        self.assertIsNot(reloaded, net)
        self.assertEqual(reloaded.edges(), net.edges())
        ll = np.array([net.log_density(x) for x in probes])
        ll2 = np.array([reloaded.log_density(x) for x in probes])
        np.testing.assert_array_equal(ll2, ll)  # bit-identical, not just close
        enc = net.dist_to_encoder().seq_encode(probes)
        enc2 = reloaded.dist_to_encoder().seq_encode(probes)
        np.testing.assert_array_equal(reloaded.seq_log_density(enc2), net.seq_log_density(enc))
        return reloaded

    def test_roundtrip_with_missing_categorical_and_mixed_factors(self):
        data = self._records_with_missing_categorical()
        net = learn_bayesian_network(data, max_parents=2)
        self.assertGreater(len(net.edges()), 0)  # the planted workclass -> hours dependence is found
        missing_row = next(x for x in data if x[0] is None)
        reloaded = self._assert_roundtrip_identical(net, data[:50] + [missing_row])
        # the reloaded sampler works and produces the same field shape
        sample = reloaded.sampler(seed=1).sample(3)
        self.assertEqual(len(sample[0]), 3)

    def test_roundtrip_glm_factor(self):
        rng = np.random.RandomState(1)
        x = rng.randn(600) * 1.5
        data = [(float(v), "hi" if rng.rand() < 1.0 / (1.0 + np.exp(-3.0 * (v - 0.5))) else "lo") for v in x]
        net = learn_bayesian_network(data)
        kinds = {type(f).__name__ for f in net.factors}
        self.assertIn("_GLMFactor", kinds)
        self._assert_roundtrip_identical(net, data[:60])

    def test_roundtrip_vector_nodes(self):
        r = np.random.RandomState(2)
        data = []
        for _ in range(400):
            cat = ["a", "b", "c"][r.randint(0, 3)]
            center = {"a": [2, 0, 0, 0], "b": [0, 2, 0, 0], "c": [0, 0, 2, 0]}[cat]
            vec = np.asarray(center, dtype=float) + 0.3 * r.randn(4)
            price = float(2.0 * vec[0] - 1.0 * vec[1] + 0.4 * r.randn())
            data.append((cat, vec, price))
        net = learn_bayesian_network(data, max_parents=2)
        kinds = {type(f).__name__ for f in net.factors}
        self.assertTrue({"_VectorCLGFactor", "_VectorMarginalFactor"} & kinds)
        self._assert_roundtrip_identical(net, data[:40])

    def test_roundtrip_independent_composite_baseline(self):
        # the transparent baseline (structure="off") is a plain mixle.stats CompositeDistribution --
        # already registered, but confirm it round-trips through the same call our flagship example uses.
        from mixle.inference import optimize

        data = self._records_with_missing_categorical()
        baseline = optimize(data, out=None, structure="off")
        self.assertEqual(type(baseline).__name__, "CompositeDistribution")
        text = to_json(baseline)
        reloaded = from_json(text)
        missing_row = next(x for x in data if x[0] is None)
        for row in data[:20] + [missing_row]:
            self.assertEqual(reloaded.log_density(row), baseline.log_density(row))


class DescribeTest(unittest.TestCase):
    """``HeterogeneousBayesianNetwork.describe()`` (worklist F10.1's explain_fit building block): every
    factor kind must report REAL fitted numbers that match the planted ground truth, not just avoid
    crashing -- a describe() that silently mislabeled a coefficient would be worse than none."""

    def test_linear_gaussian_coefficient_matches_planted_effect(self):
        # workclass shifts hours.per.week's mean by a known, exact amount -- describe() must recover it.
        r = np.random.RandomState(0)
        cats = ["Private", "Self-emp"]
        data = []
        for _ in range(2000):
            w = cats[r.randint(0, 2)]
            hours = (45.0 if w == "Self-emp" else 38.0) + 0.5 * r.randn()
            data.append((w, hours))
        net = learn_bayesian_network(data, max_parents=1)
        report = net.describe(("workclass", "hours"))
        edge = next(f for f in report["edges"] if f["field"] == "hours")
        self.assertEqual(edge["kind"], "linear-gaussian")
        self.assertIn("workclass='Self-emp'", edge["coefficients"])
        # Private is the dropped-first level (folded into the intercept); Self-emp's coefficient is the
        # planted +7 offset relative to it.
        self.assertAlmostEqual(edge["coefficients"]["workclass='Self-emp'"], 7.0, delta=0.3)
        self.assertAlmostEqual(edge["sigma"], 0.5, delta=0.1)

    def test_glm_binomial_reports_correct_direction_and_levels(self):
        # levels sort alphabetically ("down" < "up"); _GLMFactor's own convention is levels[0]=reference,
        # levels[1]=positive -- describe() must report exactly that pairing, whichever wins the sort, and
        # the coefficient sign must match the planted direction (higher x -> more "up").
        rng = np.random.RandomState(1)
        x = rng.randn(1500) * 1.5
        data = [(float(v), "up" if rng.rand() < 1.0 / (1.0 + np.exp(-3.0 * v)) else "down") for v in x]
        net = learn_bayesian_network(data, max_parents=1)
        report = net.describe(("x", "label"))
        edge = next(f for f in report["edges"] if f["field"] == "label")
        self.assertEqual(edge["kind"], "glm-binomial")
        self.assertEqual(edge["reference_level"], "down")
        self.assertEqual(edge["positive_level"], "up")
        self.assertGreater(edge["coefficients"]["x"], 0.0)  # higher x -> more "up", the planted direction

    def test_root_marginal_reports_the_fitted_distribution(self):
        r = np.random.RandomState(2)
        data = [(float(r.randn()), float(r.randn())) for _ in range(300)]  # independent -- both stay roots
        net = learn_bayesian_network(data, max_parents=1)
        report = net.describe()
        self.assertEqual(len(report["edges"]), 0)
        self.assertEqual(len(report["roots"]), 2)
        self.assertTrue(all("GaussianDistribution" in f["fitted"] for f in report["roots"]))

    def test_root_marginal_surfaces_missingness(self):
        r = np.random.RandomState(3)
        col = [None if r.rand() < 0.2 else "a" for _ in range(300)]
        net = learn_bayesian_network([(v,) for v in col], max_parents=0)
        report = net.describe(("cat",))
        self.assertEqual(report["roots"][0]["field"], "cat")
        self.assertIn("OptionalDistribution", report["roots"][0]["fitted"])

    def test_discrete_conditional_reports_configurations(self):
        r = np.random.RandomState(4)
        data = []
        for _ in range(1500):
            a = "x" if r.rand() < 0.5 else "y"
            b = int(r.poisson(8 if a == "x" else 2))
            data.append((a, b))
        net = learn_bayesian_network(data, max_parents=1)
        report = net.describe(("a", "b"))
        edge = next(f for f in report["edges"] if f["field"] == "b")
        self.assertEqual(edge["kind"], "discrete-conditional")
        self.assertEqual(edge["n_configurations"], 2)
        self.assertEqual({c["given"]["a"] for c in edge["configurations"]}, {"x", "y"})

    def test_wrong_field_names_length_raises(self):
        net = learn_bayesian_network([(1.0, 2.0) for _ in range(50)], max_parents=1)
        with self.assertRaises(ValueError):
            net.describe(("only_one_name",))

    def test_describe_is_json_serializable(self):
        # explain_fit's whole point is a report a caller can print/save -- it must not carry back numpy
        # scalars or other non-plain-json types.
        import json

        r = np.random.RandomState(5)
        data = [("a" if r.rand() < 0.5 else "b", float(r.randn())) for _ in range(300)]
        net = learn_bayesian_network(data, max_parents=1)
        json.dumps(net.describe(("cat", "value")))  # raises TypeError if anything isn't plain-JSON


if __name__ == "__main__":
    unittest.main()
