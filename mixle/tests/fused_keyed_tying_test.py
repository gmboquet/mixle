"""Keyed (tied) parameters survive the fused engine path (the #432 bug class, fused edition).

merge_accumulator_keys is the pass every EM driver must run exactly once after accumulation.
_engine_fused_step -- the path optimize()'s auto-fusion gate routes large fits onto -- skipped it,
so shared-key estimators silently untied: a shared-key (whole-state-tied) Gaussian mixture estimated
per-component variances (1.36 vs 3.18) where the host pools both. Found by a compiler review's live
probe; fixed by
running the same pass the local step and seq_estimate run. These tests pin both the engine step and
the end-to-end optimize route.
"""

import unittest

import numpy as np

from mixle.utils.optional_deps import HAS_NUMBA

if HAS_NUMBA:
    from mixle.engines import FUSED_NUMPY_ENGINE
    from mixle.inference import optimize
    from mixle.inference.estimation import _engine_fused_step
    from mixle.stats import GaussianDistribution, MixtureDistribution
    from mixle.stats.compute.sequence import seq_estimate
    from mixle.stats.latent.mixture import MixtureEstimator
    from mixle.stats.univariate.continuous.gaussian import GaussianEstimator


def _tied_fixture(n_per=4000):
    rng = np.random.RandomState(0)
    data = [float(v) for v in np.concatenate([rng.normal(-3, 1.0, n_per), rng.normal(3, 2.0, n_per)])]
    model = MixtureDistribution([GaussianDistribution(-2.0, 1.5), GaussianDistribution(2.0, 1.5)], [0.5, 0.5])
    est = MixtureEstimator([GaussianEstimator(keys="shared"), GaussianEstimator(keys="shared")])
    enc_data = [(len(data), model.dist_to_encoder().seq_encode(data))]
    return model, est, enc_data, data


@unittest.skipUnless(HAS_NUMBA, "the fused engine path requires numba")
class FusedKeyedTyingTest(unittest.TestCase):
    def test_engine_fused_step_pools_shared_keys_like_the_host(self):
        model, est, enc_data, _ = _tied_fixture()
        host = seq_estimate(enc_data, est, model)
        fused, _ll = _engine_fused_step(enc_data, est, model, FUSED_NUMPY_ENGINE)
        hv = [c.sigma2 for c in host.components]
        fv = [c.sigma2 for c in fused.components]
        self.assertAlmostEqual(hv[0], hv[1], places=12, msg="host must tie the shared-key variances")
        self.assertAlmostEqual(fv[0], fv[1], places=12, msg="the fused engine path must tie them too")
        np.testing.assert_allclose(fv, hv, rtol=1e-9)

    def test_engine_seq_estimate_pools_shared_keys_too(self):
        from mixle.inference.estimation import _engine_seq_estimate

        model, est, enc_data, _ = _tied_fixture()
        host = seq_estimate(enc_data, est, model)
        fused = _engine_seq_estimate(enc_data, est, model, FUSED_NUMPY_ENGINE)
        np.testing.assert_allclose([c.sigma2 for c in fused.components], [c.sigma2 for c in host.components], rtol=1e-9)

    def test_optimize_with_the_fused_engine_keeps_tying_end_to_end(self):
        model, est, enc_data, data = _tied_fixture(n_per=2500)
        fit = optimize(
            data, estimator=est, prev_estimate=model, max_its=5, delta=None, engine=FUSED_NUMPY_ENGINE, out=None
        )
        sig = [c.sigma2 for c in fit.components]
        self.assertAlmostEqual(sig[0], sig[1], places=12, msg=f"tied variances diverged: {sig}")


if __name__ == "__main__":
    unittest.main()
