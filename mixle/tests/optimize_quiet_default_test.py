"""The fit verbs are quiet by default (worklist Q5.4).

``optimize`` / ``fit`` / ``best_of`` used to default ``out=sys.stdout``, so any ordinary fit sprayed
per-iteration ``Iteration N: ...`` lines onto the caller's stdout -- unsolicited output that a library
should not produce. The default is now ``None`` (silent); progress is strictly opt-in via ``out=``. These
tests pin both halves: no output unless asked, and the requested stream still receives it.
"""

import contextlib
import io
import unittest

import numpy as np

from mixle.inference.estimation import best_of, fit, optimize
from mixle.stats import MultivariateGaussianEstimator
from mixle.stats.latent.gaussian_mixture import GaussianMixtureEstimator


def _data(seed=0):
    rng = np.random.RandomState(seed)
    return [list(x) for x in np.vstack([rng.randn(120, 2), rng.randn(120, 2) + 5.0])]


def _est():
    return GaussianMixtureEstimator([MultivariateGaussianEstimator(dim=2), MultivariateGaussianEstimator(dim=2)])


class OptimizeQuietDefaultTest(unittest.TestCase):
    def test_optimize_is_silent_by_default(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            optimize(_data(), _est(), max_its=8)
        self.assertEqual(buf.getvalue(), "")

    def test_fit_is_silent_by_default(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fit(_data(), _est(), max_its=8, delta=None)
        self.assertEqual(buf.getvalue(), "")

    def test_best_of_is_silent_by_default(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            best_of(_data(), None, _est(), 2, 5, 0.1, 1e-9, np.random.RandomState(0))
        self.assertEqual(buf.getvalue(), "")

    def test_progress_is_opt_in(self):
        # Passing an explicit stream still produces per-iteration progress -- the capability is intact.
        stream = io.StringIO()
        optimize(_data(), _est(), max_its=8, out=stream)
        self.assertIn("Iteration", stream.getvalue())
        # and it did not leak to stdout in the process.
        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            optimize(_data(), _est(), max_its=8, out=io.StringIO())
        self.assertEqual(stdout_buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
