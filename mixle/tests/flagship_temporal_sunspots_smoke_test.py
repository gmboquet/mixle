"""Temporal real-data flagship smoke gate (worklist F10.2).

Flagship B (``examples/flagship_temporal_sunspots.py``) fits a discrete-emission HMM on the real monthly
sunspot series and validates its held-out log-likelihood against ``hmmlearn`` on the same split. This is
its fast, bounded gate: it asserts both HMMs produce a finite held-out LL and that they AGREE (mixle's
number is validated by an independent implementation, not just asserted).

Needs network to fetch the series and ``hmmlearn`` for the baseline; skips cleanly when either is
missing, so it never fails a base-install run. It runs for real in the optional CI lane.

The three test classes below exercise the flagship's other worklist F10.2 pieces -- seed stability,
runtime/memory characterization, and impossible-observation handling against the REAL fitted model.
None of these need ``hmmlearn`` (they only exercise mixle's own fit), so they are not gated by
``_HAS_HMMLEARN``; they need only network, handled the same way as the class above. All four classes
are triaged ``("optional", "slow")`` in ``conftest.py`` so they stay out of the default fast gate.
"""

import importlib.util
import math
import sys
import unittest
import urllib.error
from pathlib import Path

_HAS_HMMLEARN = importlib.util.find_spec("hmmlearn") is not None
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))


def _run_or_skip(test_case: unittest.TestCase, **kwargs):
    from flagship_temporal_sunspots import run

    try:
        return run(**kwargs)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        test_case.skipTest(f"sunspots series unavailable (offline?): {type(exc).__name__}: {exc}")


def _seed_stability_or_skip(test_case: unittest.TestCase, **kwargs):
    from flagship_temporal_sunspots import seed_stability

    try:
        return seed_stability(**kwargs)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        test_case.skipTest(f"sunspots series unavailable (offline?): {type(exc).__name__}: {exc}")


@unittest.skipUnless(_HAS_HMMLEARN, "hmmlearn not installed")
class TemporalSunspotsFlagshipSmokeTest(unittest.TestCase):
    def test_mixle_hmm_matches_hmmlearn_on_held_out(self):
        receipt = _run_or_skip(self, n_states=3, n_symbols=8, seq_len=60, verbose=False)

        mixle_ll = receipt["mixle_test_ll_per_obs"]
        hmmlearn_ll = receipt["hmmlearn_test_ll_per_obs"]
        self.assertTrue(math.isfinite(mixle_ll), f"mixle held-out LL not finite: {mixle_ll}")
        self.assertIsNotNone(hmmlearn_ll, "hmmlearn baseline did not run")
        self.assertTrue(math.isfinite(hmmlearn_ll))
        # the receipt is that an independent HMM implementation agrees on the same held-out split
        self.assertLess(
            abs(mixle_ll - hmmlearn_ll),
            0.5,
            f"mixle ({mixle_ll:.4f}) and hmmlearn ({hmmlearn_ll:.4f}) disagree on held-out log-likelihood",
        )


class TemporalSunspotsSeedStabilityTest(unittest.TestCase):
    """Worklist F10.2 seed-stability characterization: does NOT need hmmlearn."""

    def test_seed_stability_all_finite_and_spread_is_bounded(self):
        summary = _seed_stability_or_skip(self, seeds=range(6), verbose=False)

        per_seed = summary["per_seed"]
        self.assertEqual(len(per_seed), 6)
        for r in per_seed:
            ll = r["test_ll_per_obs"]
            self.assertTrue(math.isfinite(ll), f"non-finite held-out LL for seed {r['seed']}: {ll}")
            # a discrete 8-symbol categorical HMM's held-out LL/obs is <= 0 (log of a probability),
            # and should not be catastrophically below the uniform-model floor log(1/8) = -2.079 if
            # EM is behaving.
            self.assertLess(ll, 0.0)
            self.assertGreater(ll, -3.0, f"seed {r['seed']}: held-out LL {ll:.4f} implausibly low")

        # EM is a local method: seeds are NOT expected to be identical (see seed_stability's own
        # docstring), but the spread must stay within an empirically-justified bound. A regression
        # that made fits wildly seed-dependent (e.g. a broken initialization) would blow this open;
        # the real local-optima spread observed during development (~0.83 over 10 seeds at max_its=200)
        # comfortably clears it.
        self.assertLess(
            summary["range"],
            2.0,
            f"seed-to-seed spread {summary['range']:.4f} exceeds the expected bound (per-seed results: {per_seed})",
        )
        print(
            f"F10.2 seed stability (6 seeds): mean={summary['mean']:.4f} std={summary['std']:.4f} "
            f"range=[{summary['min']:.4f}, {summary['max']:.4f}] spread={summary['range']:.4f}"
        )


class TemporalSunspotsRuntimeMemoryTest(unittest.TestCase):
    """Worklist F10.2 runtime/memory characterization: does NOT need hmmlearn."""

    def test_fit_runtime_and_peak_memory_are_reported_and_bounded(self):
        receipt = _run_or_skip(self, compare_hmmlearn=False, verbose=False)

        wall = receipt["fit_wall_time_sec"]
        peak_mb = receipt["fit_peak_memory_mb"]
        self.assertTrue(math.isfinite(wall) and wall > 0.0, f"fit_wall_time_sec not a positive finite value: {wall}")
        self.assertTrue(
            math.isfinite(peak_mb) and peak_mb > 0.0, f"fit_peak_memory_mb not a positive finite value: {peak_mb}"
        )
        # generous ceilings -- catch a true perf/memory regression without being flaky on a slow runner
        self.assertLess(wall, 60.0, f"core fit took {wall:.2f}s -- expected well under a minute at this data scale")
        self.assertLess(peak_mb, 500.0, f"core fit peaked at {peak_mb:.1f} MB -- expected a few MB at this data scale")
        print(
            f"F10.2 runtime/memory (n_states=3, n_symbols=8, seq_len=60): fit_wall_time_sec={wall:.4f} fit_peak_memory_mb={peak_mb:.2f}"
        )


class TemporalSunspotsImpossibleObservationIntegrationTest(unittest.TestCase):
    """Worklist F10.2 impossible-observation check against the REAL fitted sunspots model, does NOT
    need hmmlearn. Complements the synthetic-data unit tests in
    ``flagship_temporal_sunspots_inspection_test.py``, which exercise the same function without
    network."""

    def test_real_fitted_model_rejects_out_of_support_symbol(self):
        receipt = _run_or_skip(self, compare_hmmlearn=False, verbose=False)

        check = receipt["impossible_observation_check"]
        self.assertTrue(check["valid_is_finite"], "a fully in-support sequence must still score finite")
        self.assertTrue(check["correctly_flagged"], "an out-of-support symbol must be scored -inf, not silently finite")
        self.assertEqual(check["impossible_ll"], float("-inf"))


if __name__ == "__main__":
    unittest.main()
