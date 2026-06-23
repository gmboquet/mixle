"""Random-init parity between the numba and numpy HMM E-step paths.

The numba and numpy ``HiddenMarkovAccumulator.seq_initialize`` branches consume different sequence
encodings (sequence-major flat ``xs`` vs. time-major "bands"). Both draw the SAME values from the
same ``RandomState``; the only thing that ever differed was how that draw was mapped onto the
emission observations during initialization, which used to send the two paths to different EM local
optima for the same seed. With the numba init aligned to the numpy ("bands") layout, fitting with
random init under ``use_numba=True`` vs. ``use_numba=False`` and the same ``rng`` must converge to
the same model, so defaulting numba on is a transparent speedup.
"""

import importlib.util
import os
import unittest

import numpy as np

from pysp.inference.estimation import optimize
from pysp.stats.latent.hidden_markov import HiddenMarkovEstimator, HiddenMarkovModelDistribution
from pysp.stats.univariate.continuous.gaussian import GaussianDistribution, GaussianEstimator
from pysp.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator

_HAS_NUMBA = importlib.util.find_spec("numba") is not None


@unittest.skipUnless(_HAS_NUMBA, "numba not installed")
class HmmNumbaParityTest(unittest.TestCase):
    def test_random_init_numba_matches_numpy(self):
        devnull = open(os.devnull, "w")
        try:
            true = HiddenMarkovModelDistribution(
                [GaussianDistribution(-2, 1), GaussianDistribution(2, 1)],
                [0.5, 0.5],
                [[0.9, 0.1], [0.2, 0.8]],
                len_dist=CategoricalDistribution({12: 1.0}),
            )
            seqs = true.sampler(2).sample(3000)

            def fit(use_numba):
                est = HiddenMarkovEstimator(
                    [GaussianEstimator(), GaussianEstimator()],
                    len_estimator=CategoricalEstimator(),
                    use_numba=use_numba,
                )
                return optimize(seqs, est, max_its=25, rng=np.random.RandomState(0), out=devnull)

            m_numba = fit(True)
            m_numpy = fit(False)

            np.testing.assert_allclose(m_numba.w, m_numpy.w, atol=1e-9)
            np.testing.assert_allclose(m_numba.transitions, m_numpy.transitions, atol=1e-9)
            mu_numba = np.array([t.mu for t in m_numba.topics])
            mu_numpy = np.array([t.mu for t in m_numpy.topics])
            s2_numba = np.array([t.sigma2 for t in m_numba.topics])
            s2_numpy = np.array([t.sigma2 for t in m_numpy.topics])
            np.testing.assert_allclose(mu_numba, mu_numpy, atol=1e-9)
            np.testing.assert_allclose(s2_numba, s2_numpy, atol=1e-9)
        finally:
            devnull.close()


if __name__ == "__main__":
    unittest.main()
