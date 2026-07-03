"""The quantized-inference certificate: run enumeration forwards on a quantized model, soundly.

Enumeration never needs logits sharper than the fine-bucket width, so int8/int4 inference is admissible
if its effect is bounded. ``logit_error_bucket_slack(eps, steps, quantizer)`` is that bound: with every
step log-probability within ``eps`` nats of the true model's, any sequence's accumulated fine bucket
shifts by at most the returned number of buckets. Verified empirically here: an int8-simulated provider
(logits rounded to 256 levels, renormalized) is measured for its true per-step error, and EVERY sequence's
bucket shift between the quantized-model index and the fp64 index is within the predicted slack; count
queries agree within the induced band.
"""

import itertools
import math
import unittest

import numpy as np

from mixle.enumeration import AutoregressiveEnumerable, SeekIndex, logit_error_bucket_slack
from mixle.enumeration.quantization.core import Quantizer


def _fp64_model(V, L, seed=0):
    W = np.random.RandomState(seed).randn(L, V, V) * 1.5

    def nlp(prefix):
        d = len(prefix)
        last = prefix[-1] if prefix else 0
        lg = W[d, last]
        m = np.max(lg)
        return np.arange(V), lg - (m + math.log(np.sum(np.exp(lg - m))))

    return nlp


def _int8_wrap(nlp):
    """Simulate int8 inference: round the logits to 256 levels over their range, then renormalize."""

    def quantized(prefix):
        tokens, lps = nlp(prefix)
        lo, hi = float(lps.min()), float(lps.max())
        scale = (hi - lo) / 255.0 if hi > lo else 1.0
        q = np.round((lps - lo) / scale) * scale + lo
        m = np.max(q)
        return tokens, q - (m + math.log(np.sum(np.exp(q - m))))

    return quantized


class QuantizedInferenceCertificateTest(unittest.TestCase):
    def setUp(self):
        self.V, self.L = 6, 4
        self.nlp = _fp64_model(self.V, self.L, seed=0)
        self.qnlp = _int8_wrap(self.nlp)
        self.quantizer = Quantizer(oversample=8)
        self.full = AutoregressiveEnumerable(self.nlp, max_len=self.L)
        self.quant = AutoregressiveEnumerable(self.qnlp, max_len=self.L)

    def _measured_eps(self):
        """The int8 provider's true worst per-step log-prob error over every context."""
        eps = 0.0
        for d in range(self.L):
            for prefix in itertools.product(range(self.V), repeat=d):
                _t, lps = self.nlp(prefix)
                _tq, qlps = self.qnlp(prefix)
                eps = max(eps, float(np.max(np.abs(lps - qlps))))
        return eps

    def test_bucket_shifts_within_predicted_slack(self):
        eps = self._measured_eps()
        slack = logit_error_bucket_slack(eps, self.L, self.quantizer)
        self.assertGreater(slack, 0)  # int8 rounding is a real, nonzero perturbation
        worst = 0
        for seq in itertools.product(range(self.V), repeat=self.L):
            b_true = self.quantizer.fine_bucket(self.full.log_density(seq))
            b_quant = self.quantizer.fine_bucket(self.quant.log_density(seq))
            worst = max(worst, abs(b_true - b_quant))
        self.assertLessEqual(worst, slack)

    def test_count_queries_agree_within_the_band(self):
        eps = self._measured_eps()
        slack = logit_error_bucket_slack(eps, self.L, self.quantizer)
        band_nats = (slack + self.L) / self.quantizer.fine_per_bit() * math.log(2.0)  # + structural smear
        si_true = SeekIndex(self.full)
        si_quant = SeekIndex(self.quant)
        thr = self.full.unrank(150)[1]
        n_true = float(si_true.count(thr))
        # sound sandwich: the quantized index at a band-relaxed/tightened threshold brackets the truth
        self.assertGreaterEqual(float(si_quant.count(thr - band_nats)), n_true)
        self.assertLessEqual(float(si_quant.count(thr + band_nats)), n_true)

    def test_slack_formula_and_validation(self):
        q = Quantizer(bin_width_bits=1.0, oversample=8)
        self.assertEqual(logit_error_bucket_slack(0.0, 10, q), 0)
        # 0.01 nats/step, 9 steps, 8 buckets/bit: 9 * 0.01/ln2 * 8 = 1.038... -> 2 buckets
        self.assertEqual(logit_error_bucket_slack(0.01, 9, q), 2)
        with self.assertRaises(ValueError):
            logit_error_bucket_slack(-0.1, 3, q)
        with self.assertRaises(ValueError):
            logit_error_bucket_slack(0.1, -3, q)


if __name__ == "__main__":
    unittest.main()
