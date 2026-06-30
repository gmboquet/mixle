"""Probabilistic circuit / sum-product network (mixle.stats.latent.probabilistic_circuit).

Forward (float + LNS), construction validity (decomposability + smoothness), ancestral sampling, and EM
fitting (circuit-flow soft counts) -- the deep model that scores its whole forward pass in integer log-space.
"""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import optimize
from mixle.stats.latent.probabilistic_circuit import (
    ProbabilisticCircuitDistribution as PC,
)
from mixle.stats.latent.probabilistic_circuit import (
    leaf,
    prod,
    summ,
)


def _diag_gmm_circuit(m0, m1, w, s=1.0):
    def comp(m):
        return prod([leaf(0, st.GaussianDistribution(m, s)), leaf(1, st.GaussianDistribution(m, s))])

    return PC(summ([comp(m0), comp(m1)], w), num_vars=2)


def _equiv_mixture(m0, m1, w, s=1.0):
    return st.MixtureDistribution(
        [
            st.CompositeDistribution((st.GaussianDistribution(m0, s), st.GaussianDistribution(m0, s))),
            st.CompositeDistribution((st.GaussianDistribution(m1, s), st.GaussianDistribution(m1, s))),
        ],
        w,
    )


class ForwardTest(unittest.TestCase):
    def test_log_density_matches_equivalent_mixture(self):
        pc = _diag_gmm_circuit(0.0, 5.0, [0.6, 0.4])
        mix = _equiv_mixture(0.0, 5.0, [0.6, 0.4])
        for x in ([0.1, -0.2], [5.2, 4.8], [2.5, 2.5]):
            self.assertAlmostEqual(pc.log_density(x), mix.log_density(tuple(x)), places=9)

    def test_seq_log_density_matches_mixture(self):
        pc = _diag_gmm_circuit(-1.0, 3.0, [0.5, 0.5])
        mix = _equiv_mixture(-1.0, 3.0, [0.5, 0.5])
        rows = pc.sampler(0).sample(500)
        enc = pc.dist_to_encoder().seq_encode(rows)
        menc = mix.dist_to_encoder().seq_encode([tuple(r) for r in rows])
        self.assertTrue(np.allclose(pc.seq_log_density(enc), mix.seq_log_density(menc), atol=1e-9))

    def test_lns_scoring_within_bound(self):
        pc = _diag_gmm_circuit(0.0, 5.0, [0.6, 0.4])
        pc_lns = _diag_gmm_circuit(0.0, 5.0, [0.6, 0.4])  # same structure, but scored in the LNS
        pc_lns.lns_step = 0.005
        rows = pc.sampler(1).sample(400)
        enc = pc.dist_to_encoder().seq_encode(rows)
        self.assertLess(float(np.max(np.abs(pc_lns.seq_log_density(enc) - pc.seq_log_density(enc)))), 0.05)


class ValidityTest(unittest.TestCase):
    def test_decomposability_enforced(self):
        with self.assertRaises(ValueError):
            PC(prod([leaf(0, st.GaussianDistribution(0, 1)), leaf(0, st.GaussianDistribution(1, 1))]), num_vars=1)

    def test_smoothness_enforced(self):
        with self.assertRaises(ValueError):
            PC(summ([leaf(0, st.GaussianDistribution(0, 1)), leaf(1, st.GaussianDistribution(0, 1))]), num_vars=2)

    def test_root_must_cover_all_variables(self):
        with self.assertRaises(ValueError):
            PC(leaf(0, st.GaussianDistribution(0, 1)), num_vars=3)


class SamplingAndDagTest(unittest.TestCase):
    def test_sampler_shape(self):
        s = _diag_gmm_circuit(0.0, 5.0, [0.5, 0.5]).sampler(0).sample(7)
        self.assertEqual(len(s), 7)
        self.assertEqual(len(s[0]), 2)

    def test_shared_subcircuit_dag(self):
        # a child shared by two parents (true DAG): leaf over var 1 reused under both products
        shared = leaf(1, st.GaussianDistribution(0.0, 1.0))
        root = summ(
            [
                prod([leaf(0, st.GaussianDistribution(-2.0, 1.0)), shared]),
                prod([leaf(0, st.GaussianDistribution(2.0, 1.0)), shared]),
            ],
            [0.5, 0.5],
        )
        pc = PC(root, num_vars=2)
        # leaf side table has 3 leaves (two var-0 leaves + one shared var-1 leaf), not 4
        self.assertEqual(len(pc.leaf_dists), 3)
        self.assertTrue(np.isfinite(pc.log_density([0.0, 0.0])))


class EMTest(unittest.TestCase):
    def test_em_recovers_known_circuit(self):
        true = _diag_gmm_circuit(-3.0, 4.0, [0.7, 0.3])
        data = true.sampler(1).sample(8000)
        init = _diag_gmm_circuit(-1.0, 1.0, [0.5, 0.5], s=2.0)
        fit = optimize(data, init.estimator(), prev_estimate=init, max_its=40, out=None)
        mus = sorted(float(fit.leaf_dists[lid].mu) for lid in fit.leaf_dists)
        self.assertTrue(np.allclose(mus, [-3.0, -3.0, 4.0, 4.0], atol=0.2))
        w = sorted(float(np.exp(lw)) for lw in fit.nodes[-1][2])
        self.assertTrue(np.allclose(w, [0.3, 0.7], atol=0.05))

    def test_em_increases_log_likelihood(self):
        true = _diag_gmm_circuit(-3.0, 4.0, [0.6, 0.4])
        data = true.sampler(2).sample(4000)

        def ll(model):
            return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))

        init = _diag_gmm_circuit(0.0, 1.0, [0.5, 0.5], s=3.0)
        fit = optimize(data, init.estimator(), prev_estimate=init, max_its=25, out=None)
        self.assertGreater(ll(fit), ll(init))

    def test_heterogeneous_leaves(self):
        # a continuous coord (Gaussian) and a discrete coord (Categorical) under each product
        def comp(m, p):
            return prod([leaf(0, st.GaussianDistribution(m, 1.0)), leaf(1, st.CategoricalDistribution(p))])

        true = PC(summ([comp(-2.0, {"a": 0.8, "b": 0.2}), comp(3.0, {"a": 0.2, "b": 0.8})], [0.5, 0.5]), num_vars=2)
        data = true.sampler(1).sample(5000)
        init = PC(summ([comp(-0.5, {"a": 0.5, "b": 0.5}), comp(0.5, {"a": 0.5, "b": 0.5})], [0.5, 0.5]), num_vars=2)
        fit = optimize(data, init.estimator(), prev_estimate=init, max_its=30, out=None)
        mus = sorted(
            float(fit.leaf_dists[lid].mu)
            for lid in fit.leaf_dists
            if type(fit.leaf_dists[lid]).__name__ == "GaussianDistribution"
        )
        self.assertTrue(np.allclose(mus, [-2.0, 3.0], atol=0.3))


if __name__ == "__main__":
    unittest.main()
