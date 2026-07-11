"""Temporal real-data flagship smoke gate (worklist F10.2).

Flagship B (``examples/flagship_temporal_sunspots.py``) fits a discrete-emission HMM on the real monthly
sunspot series and validates its held-out log-likelihood against ``hmmlearn`` on the same split. This is
its fast, bounded gate: it asserts both HMMs produce a finite held-out LL and that they AGREE (mixle's
number is validated by an independent implementation, not just asserted).

Needs network to fetch the series and ``hmmlearn`` for the baseline; skips cleanly when either is
missing, so it never fails a base-install run. It runs for real in the optional CI lane.
"""

import importlib.util
import math
import sys
import unittest
import urllib.error
from pathlib import Path

_HAS_HMMLEARN = importlib.util.find_spec("hmmlearn") is not None
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))


@unittest.skipUnless(_HAS_HMMLEARN, "hmmlearn not installed")
class TemporalSunspotsFlagshipSmokeTest(unittest.TestCase):
    def test_mixle_hmm_matches_hmmlearn_on_held_out(self):
        from flagship_temporal_sunspots import run

        try:
            receipt = run(n_states=3, n_symbols=8, seq_len=60, verbose=False)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.skipTest(f"sunspots series unavailable (offline?): {type(exc).__name__}: {exc}")

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


if __name__ == "__main__":
    unittest.main()
