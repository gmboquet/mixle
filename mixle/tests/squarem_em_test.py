"""SQUAREM acceleration (mixle.inference.em.SquaremEM): the convergence win, the monotone gate, and
the packer's honesty guards.

The falsification receipt this strategy ships on (probe, 2026-07-12): overlapping 6-component GMM at
n=100k, plain EM needed 200 sweeps to a target log-likelihood SQUAREM reached in 33 sweeps with every
cycle accepted and the sequence monotone. The test below re-runs a scaled-down version of exactly that
comparison as a regression gate, counting SWEEPS (SQUAREM cycles cost up to 3) rather than iterations,
so the acceleration claim can never quietly degrade into iteration-count bookkeeping.
"""

import unittest

import numpy as np

from mixle.inference import CompiledEM, SquaremEM, squarem_packer
from mixle.inference.em import StandardEM, observed_log_likelihood
from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    ExponentialDistribution,
    GaussianDistribution,
    LaplaceDistribution,
    MixtureDistribution,
    PoissonDistribution,
)


def _overlapping_gmm(n=20_000, k=6, seed=0):
    rng = np.random.RandomState(seed)
    means = np.linspace(0.0, 4.0, k)  # 0.8 apart at sd 1.0: the slow-EM regime
    data = np.concatenate([rng.normal(m, 1.0, n // k) for m in means])
    rng.shuffle(data)
    model = MixtureDistribution(
        [GaussianDistribution(float(m), 1.0) for m in means + rng.normal(0, 0.35, k)], [1.0 / k] * k
    )
    enc = model.dist_to_encoder().seq_encode(list(map(float, data)))
    return model, model.estimator(), [(len(data), enc)]


class SquaremConvergenceTest(unittest.TestCase):
    def test_reaches_plain_em_target_in_at_most_half_the_sweeps_and_stays_monotone(self):
        # sized for the fast gate (~1s); the full-scale receipt (n=100k, 200-sweep target, 5.6x
        # fewer sweeps, zero gated cycles) lives in the introducing PR
        model, est, enc_data = _overlapping_gmm(n=8_000)
        objective = observed_log_likelihood(enc_data)
        plain = StandardEM()

        ref = model
        for _ in range(80):
            ref = plain.step(enc_data, est, ref).model
        target = objective(ref)

        sq = SquaremEM()
        cur = model
        sweeps = 0
        values = [objective(cur)]
        while sweeps < 80 and values[-1] < target:
            result = sq.step(enc_data, est, cur, objective=objective)
            cur = result.model
            sweeps += result.metadata["sweeps"]
            values.append(result.objective)
        self.assertLessEqual(sweeps, 40, f"SQUAREM needed {sweeps} sweeps to plain EM's 80-sweep target")
        self.assertGreaterEqual(values[-1], target)
        self.assertTrue(
            all(b - a >= -1e-9 for a, b in zip(values, values[1:])),
            "the gated sequence must be monotone",
        )

    def test_step_reports_receipts(self):
        model, est, enc_data = _overlapping_gmm(n=6_000)
        result = SquaremEM().step(enc_data, est, model)
        meta = result.metadata
        self.assertIn(meta["sweeps"], (2, 3))
        self.assertIn("squarem_alpha", meta)
        if meta["accelerated"]:
            self.assertLessEqual(meta["squarem_alpha"], -1.0)
            self.assertIsNone(meta["fallback"])


class SquaremGateTest(unittest.TestCase):
    def test_invalid_proposal_falls_back_to_the_plain_two_sweep_result(self):
        model, est, enc_data = _overlapping_gmm(n=6_000)
        pack, _ = squarem_packer(model)

        def exploding_unpack(theta):
            raise ValueError("constraint violation on purpose")

        result = SquaremEM(packer=(pack, exploding_unpack)).step(enc_data, est, model)
        self.assertTrue(result.metadata["fallback"].startswith("invalid_proposal:ValueError"))
        self.assertFalse(result.metadata["accelerated"])
        # the fallback is exactly two plain EM sweeps: strictly improves the start point
        self.assertGreater(result.objective, observed_log_likelihood(enc_data)(model))


class CompiledEMTest(unittest.TestCase):
    def test_compiled_full_sweep_matches_standard_em(self):
        model, est, enc_data = _overlapping_gmm(n=6_000)
        expected = StandardEM().step(enc_data, est, model).model
        result = CompiledEM().step(enc_data, est, model)
        probe = model.dist_to_encoder().seq_encode([-1.0, 0.0, 1.0, 3.0, 5.0])
        np.testing.assert_allclose(
            result.model.seq_log_density(probe), expected.seq_log_density(probe), rtol=1e-11, atol=1e-11
        )
        self.assertTrue(result.metadata["compiled"])
        self.assertGreater(result.metadata["mstep_seconds"], 0.0)
        self.assertAlmostEqual(
            result.metadata["input_data_objective"],
            observed_log_likelihood(enc_data)(model),
            places=9,
        )

    def test_squarem_accepts_compiled_base_strategy(self):
        model, est, enc_data = _overlapping_gmm(n=6_000)
        result = SquaremEM(base_strategy=CompiledEM()).step(enc_data, est, model)
        self.assertIn(result.metadata["sweeps"], (2, 3))
        self.assertTrue(np.isfinite(result.objective))


class SquaremPackerTest(unittest.TestCase):
    def test_pack_unpack_round_trips_the_supported_families(self):
        model = MixtureDistribution(
            [
                CompositeDistribution(
                    (
                        GaussianDistribution(float(j), 1.0 + 0.5 * j),
                        ExponentialDistribution(1.0 + j),
                        PoissonDistribution(2.0 + j),
                        CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}),
                    )
                )
                for j in range(2)
            ],
            [0.6, 0.4],
        )
        pack, unpack = squarem_packer(model)
        rebuilt = unpack(pack(model))
        data = [(0.3, 1.2, 2, "b"), (1.5, 0.4, 4, "a")]
        enc = model.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(rebuilt.seq_log_density(enc), model.seq_log_density(enc), rtol=1e-12)

    def test_pack_unpack_round_trips_nested_mixtures_and_laplace(self):
        inner = MixtureDistribution(
            [LaplaceDistribution(-1.0, 0.7), GaussianDistribution(1.5, 1.2)],
            [0.35, 0.65],
            name="inner",
        )
        model = MixtureDistribution(
            [
                CompositeDistribution((inner, PoissonDistribution(3.0))),
                CompositeDistribution((LaplaceDistribution(2.0, 1.1), PoissonDistribution(5.0))),
            ],
            [0.4, 0.6],
            name="outer",
        )
        pack, unpack = squarem_packer(model)
        rebuilt = unpack(pack(model))
        data = [(-0.5, 2), (1.2, 4), (2.5, 7)]
        enc = model.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(rebuilt.seq_log_density(enc), model.seq_log_density(enc), rtol=1e-12)
        self.assertEqual(rebuilt.name, "outer")
        self.assertEqual(rebuilt.components[0].dists[0].name, "inner")

    def test_unsupported_and_map_models_are_refused_with_the_escape_hatch_named(self):
        with self.assertRaises(NotImplementedError) as ctx:
            squarem_packer(GaussianDistribution(0.0, 1.0))
        self.assertIn("packer=(pack, unpack)", str(ctx.exception))

        from mixle.stats import GaussianDistribution as G

        prior_model = MixtureDistribution([G(0.0, 1.0, prior=G(0.0, 10.0)), G(1.0, 1.0)], [0.5, 0.5])
        with self.assertRaises(NotImplementedError) as ctx:
            squarem_packer(prior_model)  # packer construction itself must refuse
        self.assertIn("prior", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
