"""Probabilistic circuit / sum-product network (mixle.stats.latent.probabilistic_circuit).

Forward (float + LNS), construction validity (decomposability + smoothness), ancestral sampling, and EM
fitting (circuit-flow soft counts) -- the deep model that scores its whole forward pass in integer log-space.
"""

import unittest

import numpy as np

import mixle.stats as st
from mixle.engines.lns import LogNumberSystem
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


class LnsProductZeroTest(unittest.TestCase):
    """MXR-080-0138 follow-up: ``_seq_log_density_lns``'s "product" AND "sum" branches must combine LNS
    codes via ``LogNumberSystem.multiply()``, not raw int64 ``+`` -- a categorical leaf scoring an unseen
    label (the default, unsmoothed ``CategoricalDistribution`` configuration -- not an adversarial setup)
    is a realistic ``log(0) = -inf`` leaf that quantizes to ``LOG_ZERO_CODE``, and an exactly-zero mixture
    weight (``log(0) = -inf``) is an equally realistic ``LOG_ZERO_CODE`` on the "sum" branch's weight side.
    Raw-adding that sentinel to an ordinary sibling code (virtually always negative, since real
    log-densities are <= 0) silently overflows int64 instead of propagating "impossible observation" (or,
    for a mixture, the correct finite absorbing result); confirmed pre-fix the product branch returned a
    spurious ~9.2e16 log-density (i.e. the single MOST likely row) for what is actually a zero-probability
    observation -- see the negative-control run in this task's investigation, not just a lost-precision bug.
    """

    def _gauss_cat_circuit(self, num_vars: int = 2) -> PC:
        return PC(
            prod(
                [
                    leaf(0, st.GaussianDistribution(0.0, 1.0)),
                    leaf(1, st.CategoricalDistribution({"a": 0.5, "b": 0.5})),
                ]
            ),
            num_vars=num_vars,
            lns_step=0.01,
        )

    def test_product_with_unseen_categorical_leaf_is_log_zero_not_overflow(self):
        pc = self._gauss_cat_circuit()
        # row 0's category "z" is out-of-vocab -> leaf log-density -inf -> LOG_ZERO_CODE at the product.
        rows = [[0.0, "z"], [0.2, "a"], [-0.3, "b"]]
        enc = pc.dist_to_encoder().seq_encode(rows)
        ref = pc._node_values(enc)[-1]  # float64 reference forward pass (unaffected by this bug)
        got = pc._seq_log_density_lns(enc)

        self.assertTrue(np.isneginf(ref[0]))  # sanity: the reference agrees row 0 is impossible
        self.assertTrue(np.isneginf(got[0]), f"expected exactly -inf for the LOG_ZERO leaf row, got {got[0]!r}")
        for i in (1, 2):  # ordinary rows must still score within the usual certified LNS bound
            self.assertLess(abs(float(got[i]) - float(ref[i])), 0.05)

    def test_product_lns_codes_match_logsystem_multiply_directly(self):
        # cross-check against LogNumberSystem.multiply() itself, not just the dequantized outcome, so
        # this fails if the product branch ever goes back to raw `+` even in a way that happens not to
        # move the final dequantized value for this particular fixture.
        lns = LogNumberSystem(step=0.01)
        pc = self._gauss_cat_circuit()
        rows = [[0.0, "z"], [0.2, "a"]]
        enc = pc.dist_to_encoder().seq_encode(rows)
        got = pc._seq_log_density_lns(enc)

        g_logp = pc.leaf_dists[0].seq_log_density(enc[0])
        c_logp = pc.leaf_dists[1].seq_log_density(enc[1])
        expect = lns.dequantize(lns.multiply(lns.quantize(g_logp), lns.quantize(c_logp)))
        self.assertTrue(np.array_equal(got, expect))

    def test_three_way_product_chain_with_log_zero_leaf(self):
        # a longer chain (3 children -> 2 chained multiply() calls), zero in the middle position, so
        # both the seed accumulator and a later step touch LOG_ZERO_CODE.
        pc = PC(
            prod(
                [
                    leaf(0, st.GaussianDistribution(0.0, 1.0)),
                    leaf(1, st.CategoricalDistribution({"a": 0.5, "b": 0.5})),
                    leaf(2, st.GaussianDistribution(1.0, 1.0)),
                ]
            ),
            num_vars=3,
            lns_step=0.01,
        )
        rows = [[0.0, "z", 1.0], [0.2, "a", 0.5]]
        enc = pc.dist_to_encoder().seq_encode(rows)
        ref = pc._node_values(enc)[-1]
        got = pc._seq_log_density_lns(enc)

        self.assertTrue(np.isneginf(ref[0]))
        self.assertTrue(np.isneginf(got[0]))
        self.assertLess(abs(float(got[1]) - float(ref[1])), 0.05)

    def _mixed_categorical_circuit(self) -> PC:
        # branch 0 has NO mass on "b" (default_value=0.0) -> log-density -inf -> LOG_ZERO_CODE child.
        return PC(
            summ(
                [
                    leaf(0, st.CategoricalDistribution({"a": 1.0})),
                    leaf(0, st.CategoricalDistribution({"a": 0.3, "b": 0.7})),
                ],
                [0.5, 0.5],
            ),
            num_vars=1,
            lns_step=0.01,
        )

    def test_sum_node_with_log_zero_child_is_finite_not_overflow(self):
        # a 50/50 mixture of (probability 0) and (probability 0.7) is just 0.5*0.7 -- finite and close
        # to the float64 reference, not a spuriously huge/corrupted value from raw-add overflow.
        pc = self._mixed_categorical_circuit()
        rows = [["b"]]
        enc = pc.dist_to_encoder().seq_encode(rows)
        ref = pc._node_values(enc)[-1]
        got = pc._seq_log_density_lns(enc)

        self.assertTrue(np.isfinite(ref[0]))  # sanity: the reference agrees this is an ordinary density
        self.assertTrue(np.isfinite(got[0]), f"expected a finite mixture density, got {got[0]!r}")
        self.assertLess(abs(float(got[0]) - float(ref[0])), 0.05)

    def test_sum_node_with_zero_weight_component_is_finite_not_overflow(self):
        # component 1's weight is exactly 0.0 -> log(0) = -inf -> LOG_ZERO_CODE weight code. The mixture
        # value must equal component 0's (weight-1.0) density at every row, not a raw-add overflow artifact.
        pc = PC(
            summ(
                [leaf(0, st.GaussianDistribution(0.0, 1.0)), leaf(0, st.GaussianDistribution(5.0, 1.0))],
                [1.0, 0.0],
            ),
            num_vars=1,
            lns_step=0.01,
        )
        rows = [[0.0], [5.0]]
        enc = pc.dist_to_encoder().seq_encode(rows)
        ref = pc._node_values(enc)[-1]
        got = pc._seq_log_density_lns(enc)

        self.assertTrue(np.all(np.isfinite(got)), f"expected finite densities, got {got!r}")
        self.assertLess(float(np.max(np.abs(got - ref))), 0.05)

    def test_sum_node_lns_codes_match_logsystem_multiply_directly(self):
        # cross-check against LogNumberSystem.multiply() itself (the child-code/weight-code product term
        # feeding logsumexp), not just the dequantized outcome -- fails if the sum branch ever goes back
        # to raw `+` even in a way that happens not to move the final value for this particular fixture.
        lns = LogNumberSystem(step=0.01)
        pc = self._mixed_categorical_circuit()
        rows = [["b"]]
        enc = pc.dist_to_encoder().seq_encode(rows)
        got = pc._seq_log_density_lns(enc)

        c0 = pc.leaf_dists[0].seq_log_density(enc[0])
        c1 = pc.leaf_dists[1].seq_log_density(enc[1])
        wk = lns.quantize(np.asarray(pc.nodes[-1][2]))
        terms = np.stack([lns.multiply(lns.quantize(c0), wk[0]), lns.multiply(lns.quantize(c1), wk[1])], axis=0)
        expect = lns.dequantize(lns.logsumexp(terms, axis=0))
        self.assertTrue(np.array_equal(got, expect))


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
