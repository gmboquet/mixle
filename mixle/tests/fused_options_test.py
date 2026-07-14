"""optimize(fused_options=...) reaches the fused kernels: parallel override + quantized-LSE scoring.

The chunk-parallel kernels and quantized LSE existed but had no user-facing switch -- parallel was
auto-gated on observation count only, and lse_bits was reachable only by calling the codegen wrappers
directly. These tests pin the plumbing: unknown keys refuse loudly, ``parallel=True`` engages the
parallel E-step below the auto-gate floor with allclose-identical fits, and ``lse_bits`` bounds the
scorer against the exact path while E-steps stay exact (the knob never reaches accumulate()).
"""

import unittest

import numpy as np

from mixle.utils.optional_deps import HAS_NUMBA

if HAS_NUMBA:
    from mixle.engines import FUSED_NUMPY_ENGINE
    from mixle.inference import optimize
    from mixle.stats import GaussianDistribution, MixtureDistribution
    from mixle.stats.compute import fused_codegen
    from mixle.stats.latent.mixture import MixtureEstimator
    from mixle.stats.univariate.continuous.gaussian import GaussianEstimator


class FusedCacheGcTest(unittest.TestCase):
    """clean_fused_cache removes only stale cache-owned files (age-gated, pattern-gated, dry-runnable)."""

    def test_gc_removes_stale_cache_files_and_nothing_else(self):
        import os
        import tempfile
        from unittest import mock

        from mixle.stats.compute import fused_codegen as fc

        with tempfile.TemporaryDirectory() as td:
            os.chmod(td, 0o700)
            pyc = os.path.join(td, "__pycache__")
            os.makedirs(pyc)
            old = os.path.join(td, "_pysp_fused_deadbeef00000000.py")
            old_nb = os.path.join(pyc, "_pysp_fused_deadbeef00000000.cpython-314.nbc")
            fresh = os.path.join(td, "_pysp_fused_cafe000000000000.py")
            alien = os.path.join(td, "keep_me.py")
            tmp_orphan = os.path.join(td, "_pysp_fused_deadbeef00000000.py.999.tmp")
            for p in (old, old_nb, fresh, alien, tmp_orphan):
                with open(p, "w") as fh:
                    fh.write("# cache test\n")
            stale_t = 1.0  # epoch -- ancient
            os.utime(old, (stale_t, stale_t))
            os.utime(old_nb, (stale_t, stale_t))
            os.utime(alien, (stale_t, stale_t))
            os.utime(tmp_orphan, (stale_t, stale_t))
            with mock.patch.object(fc, "_CACHE_DIR", td):
                dry = fc.clean_fused_cache(max_age_days=30.0, dry_run=True)
                self.assertEqual(sorted(dry["removed"]), sorted([old, old_nb, tmp_orphan]))
                self.assertTrue(all(os.path.exists(p) for p in (old, old_nb, fresh, alien, tmp_orphan)))
                wet = fc.clean_fused_cache(max_age_days=30.0)
                self.assertEqual(sorted(wet["removed"]), sorted([old, old_nb, tmp_orphan]))
            self.assertFalse(os.path.exists(old))
            self.assertFalse(os.path.exists(old_nb))
            self.assertFalse(os.path.exists(tmp_orphan))
            self.assertTrue(os.path.exists(fresh), "fresh cache entries must survive")
            self.assertTrue(os.path.exists(alien), "non-cache-pattern files must never be touched")


def _fixture(n_per=1500):
    rng = np.random.RandomState(7)
    data = [float(v) for v in np.concatenate([rng.normal(-3, 1.0, n_per), rng.normal(3, 2.0, n_per)])]
    model = MixtureDistribution([GaussianDistribution(-2.0, 1.5), GaussianDistribution(2.0, 1.5)], [0.5, 0.5])
    est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
    return data, model, est


@unittest.skipUnless(HAS_NUMBA, "fused kernels require numba")
class FusedOptionsTest(unittest.TestCase):
    def test_unknown_keys_refuse_loudly(self):
        data, model, est = _fixture(n_per=50)
        with self.assertRaisesRegex(ValueError, "unknown keys.*prallel"):
            optimize(data, estimator=est, prev_estimate=model, max_its=1, fused_options={"prallel": True})

    def test_best_of_forwards_fused_options(self):
        from mixle.inference.estimation import best_of

        data, model, est = _fixture(n_per=50)
        # the unknown-key refusal proves the knob reaches optimize through the trials wrapper
        with self.assertRaisesRegex(ValueError, "unknown keys.*prallel"):
            best_of(data, None, est, trials=1, max_its=1, init_p=1.0, delta=1e-9, fused_options={"prallel": True})

    def test_parallel_true_engages_below_the_auto_gate_and_matches_the_sequential_fit(self):
        data, model, est = _fixture()  # 3,000 obs << the 65,536 auto-gate floor
        base = optimize(
            data, estimator=est, prev_estimate=model, max_its=4, delta=None, engine=FUSED_NUMPY_ENGINE, out=None
        )
        # evict this signature's parallel E-step so engagement is provable regardless of which
        # earlier test already compiled it (the cache is process-global; recompiles on demand)
        sig = fused_codegen.analyze(model).signature
        fused_codegen._ESTEP_COMPILED.pop((sig, True), None)
        forced = optimize(
            data,
            estimator=est,
            prev_estimate=model,
            max_its=4,
            delta=None,
            engine=FUSED_NUMPY_ENGINE,
            out=None,
            fused_options={"parallel": True},
        )
        self.assertIn(
            (sig, True),
            fused_codegen._ESTEP_COMPILED,
            "parallel=True below the floor must compile and use the parallel E-step variant",
        )
        for b, f in zip(base.components, forced.components):
            self.assertAlmostEqual(b.mu, f.mu, places=9)
            self.assertAlmostEqual(b.sigma2, f.sigma2, places=9)

    def test_lse_bits_bounds_the_scorer_and_never_touches_the_estep(self):
        data, model, _ = _fixture(n_per=400)
        kernel = fused_codegen.FusedKernel(model, FUSED_NUMPY_ENGINE)
        enc = kernel.encode(data)
        exact = kernel.score(enc)
        kernel.lse_bits = 12
        quantized = kernel.score(enc)
        bound = 2.0**-12 * 4.0  # generous multiple of the per-level relative bound
        self.assertTrue(np.all(np.abs(quantized - exact) <= np.abs(exact) * bound + 1e-12))
        self.assertFalse(np.allclose(quantized, exact, rtol=0, atol=0), "lse_bits must actually change the scorer path")
        # E-step: the knob is scoring-only; accumulate stays exact regardless
        w = np.ones(len(data), dtype=np.float64)
        suff_q = kernel.accumulate(enc, w)
        kernel.lse_bits = None
        suff_e = kernel.accumulate(enc, w)
        np.testing.assert_array_equal(np.asarray(suff_q[0], dtype=np.float64), np.asarray(suff_e[0], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
