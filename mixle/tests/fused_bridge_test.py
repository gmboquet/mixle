"""Bridge factor kind: combinators inside composite factors fuse through their own native machinery.

A bridged factor (nested Mixture, HMM, nested Composite) is scored from a precomputed per-component
table (each column that factor's OWN ``seq_log_density``) and its sufficient statistics come from its
OWN accumulator driven by the responsibility matrix the fused kernel hands back -- byte-for-byte the
host mixture E-step's semantics for that factor, priors included, while everything else (leaf scoring,
the softmax, leaf statistics) stays in one nopython pass.

Honesty receipts asserted here: parity vs the host path at 1e-12 through score AND the M-step;
the bare Mixture-of-Mixtures guard (fused_nested's faster in-kernel path is NOT stolen by the
bridge); heterogeneous per-component factor configurations; and the parallel variant's bit-stable
reruns. Measured wall-clock (recorded in the introducing PR, cold-compile excluded): 1.2x over host
on nested-mixture composites, parity on HMM-heavy composites (the HMM's native numba kernels carry
the bulk either way), with the chain template's 5.1x untouched.
"""

import unittest

import numpy as np

from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    HiddenMarkovModelDistribution,
    MixtureDistribution,
    PoissonDistribution,
)
from mixle.stats.compute import fused_codegen as fc
from mixle.stats.compute.fused_nested import fusible_nested
from mixle.utils.optional_deps import HAS_NUMBA


def _inner_mix(seed):
    return MixtureDistribution(
        [GaussianDistribution(-3.0 + seed, 1.0), GaussianDistribution(3.0 + seed, 1.0 + 0.1 * seed)], [0.6, 0.4]
    )


def _hmm(seed):
    r = np.random.RandomState(seed)
    return HiddenMarkovModelDistribution(
        [GaussianDistribution(-2.0 + seed, 1.0), GaussianDistribution(2.0 + seed, 1.2)],
        w=list(r.dirichlet(np.ones(2))),
        transitions=r.dirichlet(np.ones(2), size=2).tolist(),
    )


def _parity(self, model, data):
    enc = model.dist_to_encoder().seq_encode(data)
    self.assertTrue(fc.fusible(model))
    self.assertTrue(fc.fusible_estep(model))
    np.testing.assert_allclose(fc.fused_seq_log_density(model, enc), model.seq_log_density(enc), rtol=1e-12)
    w = np.ones(len(data))
    est = model.estimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(enc, w, model)
    new_host = est.estimate(len(data), acc.value())
    new_fused = est.estimate(len(data), fc.fused_accumulate(model, enc, w))
    np.testing.assert_allclose(new_fused.seq_log_density(enc), new_host.seq_log_density(enc), rtol=1e-12, atol=1e-12)


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class BridgeParityTest(unittest.TestCase):
    def test_nested_mixture_factor_fuses_and_matches_host_through_the_m_step(self):
        rng = np.random.RandomState(0)
        comps = [
            CompositeDistribution(
                (_inner_mix(j), GaussianDistribution(float(j), 1.5), CategoricalDistribution({"x": 0.5, "y": 0.5}))
            )
            for j in range(3)
        ]
        model = MixtureDistribution(comps, [0.5, 0.3, 0.2])
        data = [(float(rng.randn() * 3), float(rng.randn() + (i % 3)), ("x", "y")[rng.randint(2)]) for i in range(3000)]
        _parity(self, model, data)

    def test_hmm_factor_fuses_and_matches_host_through_the_m_step(self):
        rng = np.random.RandomState(1)
        comps = [CompositeDistribution((_hmm(j), PoissonDistribution(2.0 + j))) for j in range(2)]
        model = MixtureDistribution(comps, [0.6, 0.4])
        data = [([float(rng.randn()) for _ in range(int(rng.randint(2, 7)))], int(rng.poisson(3))) for _ in range(1500)]
        _parity(self, model, data)

    def test_nested_composite_factor_fuses(self):
        rng = np.random.RandomState(2)
        nested = lambda j: CompositeDistribution((GaussianDistribution(float(j), 1.0), PoissonDistribution(1.0 + j)))  # noqa: E731
        comps = [CompositeDistribution((nested(j), GaussianDistribution(-float(j), 2.0))) for j in range(2)]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = [((float(rng.randn()), int(rng.poisson(2))), float(rng.randn())) for _ in range(2000)]
        _parity(self, model, data)

    def test_parallel_variant_matches_and_is_bit_stable(self):
        rng = np.random.RandomState(3)
        comps = [CompositeDistribution((_inner_mix(j), GaussianDistribution(float(j), 1.5))) for j in range(2)]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = [(float(rng.randn() * 3), float(rng.randn())) for _ in range(4000)]
        enc = model.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(
            fc.fused_seq_log_density(model, enc, parallel=True),
            fc.fused_seq_log_density(model, enc, parallel=False),
            rtol=1e-12,
        )
        w = np.ones(len(data))
        _, ll1 = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=True)
        _, ll2 = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=True)
        self.assertEqual(ll1, ll2)


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class BridgeBoundaryTest(unittest.TestCase):
    def test_bare_nested_mixtures_keep_the_faster_fused_nested_path(self):
        model = MixtureDistribution([_inner_mix(0), _inner_mix(1)], [0.5, 0.5])
        self.assertIsNone(fc.analyze(model), "the bridge must not steal fused_nested's in-kernel path")
        self.assertTrue(fusible_nested(model))

    def test_specific_templates_outrank_the_bridge(self):
        comps = [
            CompositeDistribution((GaussianDistribution(float(j), 1.0), CategoricalDistribution({"x": 0.5, "y": 0.5})))
            for j in range(2)
        ]
        plan = fc.analyze(MixtureDistribution(comps, [0.5, 0.5]))
        self.assertIsNotNone(plan)
        self.assertEqual([t.name for t in plan.leaf_templates], ["gaussian", "categorical"])


if __name__ == "__main__":
    unittest.main()
