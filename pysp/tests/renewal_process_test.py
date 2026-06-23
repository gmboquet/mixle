"""WS-2: renewal process -- i.i.d. inter-arrivals from a base distribution on a censored window."""

import unittest

import numpy as np

from pysp.stats.processes.renewal_process import RenewalProcessDistribution
from pysp.stats.univariate.continuous.gamma import GammaDistribution


class RenewalProcessTest(unittest.TestCase):
    def _truth(self):
        return RenewalProcessDistribution(GammaDistribution(k=3.0, theta=0.5), window=200.0)

    def test_sample_and_score(self):
        truth = self._truth()
        data = truth.sampler(seed=1).sample(20)
        # realizations are sorted event-time arrays within the window
        for d in data:
            self.assertTrue(np.all(np.diff(d) > 0) if len(d) > 1 else True)
            self.assertTrue(len(d) == 0 or d[-1] <= truth.window)
        self.assertTrue(np.isfinite(truth.log_density(data[0])))

    def test_seq_log_density_matches_scalar(self):
        truth = self._truth()
        data = truth.sampler(seed=2).sample(15)
        enc = truth.dist_to_encoder().seq_encode(data)
        seq = truth.seq_log_density(enc)
        scalar = np.array([truth.log_density(d) for d in data])
        self.assertTrue(np.allclose(seq, scalar, atol=1e-8))

    def test_out_of_window_events_score_neg_inf(self):
        truth = self._truth()
        self.assertEqual(truth.log_density(np.array([10.0, 250.0])), -np.inf)  # 250 > window

    def test_recovers_interarrival_parameters(self):
        truth = self._truth()
        data = truth.sampler(seed=3).sample(60)
        # Recover via the estimator's direct M-step (the closed-form full-data MLE) rather than fit():
        # this is fully deterministic and independent of any global init/engine state a parallel test
        # runner might leave behind.
        est = truth.estimator()
        acc = est.accumulator_factory().make()
        enc = truth.dist_to_encoder().seq_encode(data)
        acc.seq_update(enc, np.ones(len(data), dtype=np.float64), None)
        model = est.estimate(None, acc.value())
        # consistent: recover Gamma shape/scale from the ~8000 observed gaps
        self.assertAlmostEqual(model.interarrival.k, 3.0, delta=0.4)
        self.assertAlmostEqual(model.interarrival.theta, 0.5, delta=0.1)
        self.assertEqual(model.window, 200.0)


if __name__ == "__main__":
    unittest.main()
