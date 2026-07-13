"""Numerical edge panel for the fused paths: the regimes that break naive kernels.

Q5.2's stress panel covers the distributions; this covers the COMPILED paths those distributions
take -- extreme scales, degenerate mixtures, out-of-support rows scoring -inf, tiny and empty data,
and weight edge cases -- each asserted against the host oracle, through both the sequential and
parallel kernels. Every case here is a way a fused kernel could silently diverge from the host
while ordinary fixtures stay green.
"""

import unittest

import numpy as np

from mixle.utils.optional_deps import HAS_NUMBA

if HAS_NUMBA:
    from mixle.stats import (
        CategoricalDistribution,
        CompositeDistribution,
        ExponentialDistribution,
        GaussianDistribution,
        MixtureDistribution,
        PoissonDistribution,
    )
    from mixle.stats.compute import fused_codegen as fc


def _parity(tc, model, data, rtol=1e-9, check_estep=True):
    enc = model.dist_to_encoder().seq_encode(data)
    host = model.seq_log_density(enc)
    for parallel in (False, True):
        fused = fc.fused_seq_log_density(model, enc, parallel=parallel)
        np.testing.assert_allclose(fused, host, rtol=rtol, atol=1e-300, err_msg=f"parallel={parallel}")
    if check_estep and len(data):
        w = np.ones(len(data))
        est = model.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, w, model)
        new_host = est.estimate(len(data), acc.value())
        new_fused = est.estimate(len(data), fc.fused_accumulate(model, enc, w))
        np.testing.assert_allclose(new_fused.seq_log_density(enc), new_host.seq_log_density(enc), rtol=rtol)
    return host


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class ExtremeScaleTest(unittest.TestCase):
    def test_huge_and_tiny_location_scale(self):
        # locations at 1e150 with scales at 1e140: the log-densities are finite (~1e19-magnitude
        # exponents cancel in the standardized residual); any premature exp in a kernel overflows
        model = MixtureDistribution(
            [GaussianDistribution(1.0e150, 1.0e280), GaussianDistribution(-1.0e150, 1.0e280)], [0.5, 0.5]
        )
        rng = np.random.RandomState(0)
        data = [float(v) for v in rng.randn(500) * 1.0e150]
        host = _parity(self, model, data)
        self.assertTrue(np.all(np.isfinite(host)))

    def test_tiny_variances_do_not_overflow_the_lse(self):
        model = MixtureDistribution(
            [GaussianDistribution(0.0, 1.0e-12), GaussianDistribution(1.0, 1.0e-12)], [0.5, 0.5]
        )
        data = [0.0, 1.0, 0.5, 1.0e-6, 1.0 - 1.0e-6]
        _parity(self, model, data, check_estep=False)  # densities are huge but finite; parity is the claim


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class DegenerateMixtureTest(unittest.TestCase):
    def test_near_zero_component_weights(self):
        model = MixtureDistribution(
            [GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0), GaussianDistribution(0.0, 1.0)],
            [1.0 - 2e-15, 1e-15, 1e-15],
        )
        rng = np.random.RandomState(1)
        _parity(self, model, [float(v) for v in rng.randn(400)])

    def test_identical_components(self):
        model = MixtureDistribution([GaussianDistribution(0.0, 1.0)] * 3, [1 / 3] * 3)
        rng = np.random.RandomState(2)
        _parity(self, model, [float(v) for v in rng.randn(300)], check_estep=False)  # host path handles
        # shared-object components through its own machinery; scoring parity is the fused claim here

    def test_out_of_support_data_is_refused_at_the_shared_encoder_boundary(self):
        """Truly-invalid data (negative Exponential draws) never REACHES scoring: the encoder both
        paths share raises a ContractError. The fused path cannot diverge on data it never sees --
        the boundary, not the kernel, owns this case (found by this panel's first draft, which
        wrongly expected a -inf row)."""
        model = MixtureDistribution([ExponentialDistribution(1.0), ExponentialDistribution(2.0)], [0.5, 0.5])
        with self.assertRaises(Exception) as ctx:
            model.dist_to_encoder().seq_encode([1.0, -1.0])
        self.assertIn("x >= 0", str(ctx.exception))

    def test_per_component_and_all_component_minus_inf_rows_match_host(self):
        """The -inf paths that genuinely reach the scorer: a category one component lacks (its
        table entry is -inf, the mixture stays finite) and a category EVERY component lacks (the
        all--inf log-sum-exp guard must yield -inf, not NaN) -- fused == host on both."""
        model = MixtureDistribution(
            [CategoricalDistribution({"x": 0.7, "y": 0.3}), CategoricalDistribution({"x": 1.0})], [0.5, 0.5]
        )
        data = ["x", "y", "z"]  # "y": component 2 scores -inf; "z": every component scores -inf
        enc = model.dist_to_encoder().seq_encode(data)
        host = model.seq_log_density(enc)
        for parallel in (False, True):
            fused = fc.fused_seq_log_density(model, enc, parallel=parallel)
            self.assertTrue(np.isfinite(fused[0]) and np.isfinite(fused[1]))
            self.assertTrue(np.isneginf(fused[2]), f"all--inf row must score -inf, got {fused[2]}")
            np.testing.assert_allclose(fused[:2], host[:2], rtol=1e-9)
            self.assertTrue(np.isneginf(host[2]))


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class TinyDataTest(unittest.TestCase):
    def _model(self):
        return MixtureDistribution(
            [
                CompositeDistribution((GaussianDistribution(float(j), 1.0), PoissonDistribution(2.0 + j)))
                for j in range(2)
            ],
            [0.5, 0.5],
        )

    def test_single_row(self):
        _parity(self, self._model(), [(0.3, 2)])

    def test_empty_data_scores_empty(self):
        model = self._model()
        enc = model.dist_to_encoder().seq_encode([])
        self.assertEqual(len(fc.fused_seq_log_density(model, enc)), 0)
        self.assertEqual(len(fc.fused_seq_log_density(model, enc, parallel=True)), 0)

    def test_zero_and_fractional_weights_match_host(self):
        model = self._model()
        rng = np.random.RandomState(3)
        data = [(float(rng.randn()), int(rng.poisson(2))) for _ in range(200)]
        enc = model.dist_to_encoder().seq_encode(data)
        w = rng.rand(200)
        w[::7] = 0.0  # zero-weight rows must contribute nothing, exactly
        est = model.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, w, model)
        new_host = est.estimate(200, acc.value())
        new_fused = est.estimate(200, fc.fused_accumulate(model, enc, w))
        np.testing.assert_allclose(new_fused.seq_log_density(enc), new_host.seq_log_density(enc), rtol=1e-9)


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class SquaremSoakTest(unittest.TestCase):
    def test_squarem_never_loses_to_plain_em_across_random_slow_fixtures(self):
        """The acceleration property, soaked: on every random overlapped mixture, SQUAREM's gated
        sequence is monotone AND reaches at least plain EM's objective in at most the same sweeps."""
        from mixle.inference import SquaremEM
        from mixle.inference.em import StandardEM, observed_log_likelihood

        for seed in range(6):
            rng = np.random.RandomState(seed)
            K = int(rng.randint(3, 6))
            means = np.linspace(0.0, 0.9 * K, K)
            data = [float(v) for v in np.concatenate([rng.normal(m, 1.0, 3000) for m in means])]
            model = MixtureDistribution(
                [GaussianDistribution(float(m + rng.normal(0, 0.3)), 1.0) for m in means], [1 / K] * K
            )
            est = model.estimator()
            enc_data = [(len(data), model.dist_to_encoder().seq_encode(data))]
            objective = observed_log_likelihood(enc_data)

            plain = model
            for _ in range(30):
                plain = StandardEM().step(enc_data, est, plain).model
            target = objective(plain)

            sq = SquaremEM()
            cur, sweeps, vals = model, 0, [objective(model)]
            while sweeps < 30 and vals[-1] < target:
                r = sq.step(enc_data, est, cur, objective=objective)
                cur = r.model
                sweeps += r.metadata["sweeps"]
                vals.append(r.objective)
            self.assertTrue(all(b - a >= -1e-9 for a, b in zip(vals, vals[1:])), f"seed {seed}: not monotone")
            self.assertGreaterEqual(vals[-1], target - 1e-9 * abs(target), f"seed {seed}: fell short in 30 sweeps")


if __name__ == "__main__":
    unittest.main()
