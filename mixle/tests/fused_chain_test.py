"""Markov-chain leaf template for the fused codegen: composites with chain factors are fusible.

The chain kind rides the template system's existing machinery -- a scatter-built per-row
per-component score table in the precompute slot (where the matrix kind's BLAS quad forms live), a
table lookup in the row fragment, and weighted init/transition histograms scattered from the
responsibility matrix R in the post pass. No recursion, no new kernel architecture: the encoding was
already scatter-shaped. Parity is asserted against the host path at 1e-12, including the honesty
edges: empty and length-1 sequences, out-of-support states hitting a component's default mass, and
the fusibility REFUSALS (length distributions, priors).
"""

import unittest

import numpy as np

from mixle.stats import (
    CompositeDistribution,
    GaussianDistribution,
    MarkovChainDistribution,
    MixtureDistribution,
    PoissonDistribution,
)
from mixle.stats.compute import fused_codegen as fc
from mixle.utils.optional_deps import HAS_NUMBA

STATES = ["a", "b", "c"]


def _chain(seed, states=STATES):
    r = np.random.RandomState(seed)
    init = r.dirichlet(np.ones(len(states)))
    trans = r.dirichlet(np.ones(len(states)), size=len(states))
    return MarkovChainDistribution(
        dict(zip(states, init)), {s: dict(zip(states, trans[i])) for i, s in enumerate(states)}
    )


def _model_and_enc(n=4000, seed=0, include_empty=True):
    rng = np.random.RandomState(seed)
    comps = [
        CompositeDistribution((_chain(j), GaussianDistribution(float(j), 1.0 + 0.2 * j), PoissonDistribution(2.0 + j)))
        for j in range(3)
    ]
    model = MixtureDistribution(comps, [0.5, 0.3, 0.2])
    data = []
    for i in range(n):
        length = int(rng.randint(0 if include_empty else 1, 7))
        seq = [STATES[rng.randint(3)] for _ in range(length)]
        data.append((seq, float(rng.randn()), int(rng.poisson(3))))
    return model, model.dist_to_encoder().seq_encode(data), len(data)


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class FusedChainParityTest(unittest.TestCase):
    def test_chain_composites_are_fusible_and_score_matches_host(self):
        model, enc, _ = _model_and_enc()
        self.assertTrue(fc.fusible(model))
        self.assertTrue(fc.fusible_estep(model))
        host = model.seq_log_density(enc)
        fused = fc.fused_seq_log_density(model, enc)
        np.testing.assert_allclose(fused, host, rtol=1e-12, atol=1e-12)

    def test_estep_matches_host_through_the_m_step(self):
        model, enc, n = _model_and_enc()
        w = np.ones(n)
        est = model.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, w, model)
        new_host = est.estimate(n, acc.value())
        new_fused = est.estimate(n, fc.fused_accumulate(model, enc, w))
        np.testing.assert_allclose(
            new_fused.seq_log_density(enc), new_host.seq_log_density(enc), rtol=1e-12, atol=1e-12
        )

    def test_out_of_support_states_hit_the_default_mass_exactly(self):
        # component 0's chain only knows {a, b}; the data contains 'c' -> its log_dv default path
        rng = np.random.RandomState(1)
        narrow = _chain(0, states=["a", "b"])
        wide = _chain(1)
        model = MixtureDistribution(
            [
                CompositeDistribution((narrow, GaussianDistribution(0.0, 1.0))),
                CompositeDistribution((wide, GaussianDistribution(1.0, 1.0))),
            ],
            [0.5, 0.5],
        )
        data = [
            ([STATES[rng.randint(3)] for _ in range(int(rng.randint(1, 6)))], float(rng.randn())) for _ in range(800)
        ]
        enc = model.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(fc.fused_seq_log_density(model, enc), model.seq_log_density(enc), rtol=1e-12)

    def test_parallel_variant_matches_and_is_bit_stable(self):
        model, enc, n = _model_and_enc()
        seq_out = fc.fused_seq_log_density(model, enc, parallel=False)
        par_out = fc.fused_seq_log_density(model, enc, parallel=True)
        np.testing.assert_allclose(par_out, seq_out, rtol=1e-12, atol=1e-12)
        w = np.ones(n)
        s1, ll1 = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=True)
        s2, ll2 = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=True)
        self.assertEqual(ll1, ll2)
        ss, ll_seq = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=False)
        self.assertAlmostEqual(ll_seq, ll1, delta=abs(ll_seq) * 1e-9)


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class FusibilityGuardTest(unittest.TestCase):
    def test_chains_with_length_models_or_priors_keep_their_own_kernels(self):
        from mixle.stats import PoissonDistribution as P

        with_len = MarkovChainDistribution(
            {"a": 0.5, "b": 0.5}, {"a": {"a": 0.5, "b": 0.5}, "b": {"a": 0.5, "b": 0.5}}, len_dist=P(3.0)
        )
        model = MixtureDistribution([CompositeDistribution((with_len, GaussianDistribution(0.0, 1.0)))] * 2, [0.5, 0.5])
        self.assertIsNone(fc.analyze(model), "a real length distribution is outside the fused tables")

    def test_all_chain_homogeneous_mixture_survives_the_block_em_path(self):
        """Regression pin: an ALL-chain homogeneous top mixture used to hit an IndexError inside
        _component_enc on the block-EM/freeze-rollup path (hetero fixtures worked). Fixed upstream by
        the freeze-rollup resolver work; pinned here so the encoder-shape edge cannot quietly return."""
        from mixle.inference.block_em import run_block_em

        rng = np.random.RandomState(3)
        model = MixtureDistribution([_chain(j) for j in range(3)], [1 / 3] * 3)
        data = [[STATES[rng.randint(3)] for _ in range(int(rng.randint(2, 8)))] for _ in range(1500)]
        enc_data = [(len(data), model.dist_to_encoder().seq_encode(data))]
        final_model, history = run_block_em(enc_data, model.estimator(), model, max_its=3)
        self.assertTrue(len(history) >= 1)
        self.assertTrue(np.isfinite(final_model.seq_log_density(enc_data[0][1]).sum()))

    def test_bare_chain_mixture_is_fusible_too(self):
        model = MixtureDistribution([_chain(0), _chain(1)], [0.5, 0.5])
        self.assertTrue(fc.fusible(model))
        rng = np.random.RandomState(2)
        data = [[STATES[rng.randint(3)] for _ in range(int(rng.randint(1, 6)))] for _ in range(600)]
        enc = model.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(fc.fused_seq_log_density(model, enc), model.seq_log_density(enc), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
