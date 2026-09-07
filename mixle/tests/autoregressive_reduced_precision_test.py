"""A next-token table that is normalized to reduced precision is accepted and renormalized.

0.8.1 adversarial review P08-F01: transformers 5 loads a checkpoint stored in bfloat16 at that
dtype, and a bfloat16 ``log_softmax`` over a 49k-token vocabulary exponentiates to 0.9985. The step
parser rejected any table whose probabilities summed more than ``1e-4`` away from one, so the README's
enumeration snippet raised on the candidate venv. Within the widened tolerance the table is
renormalized, so counts and ranks are computed from an exact categorical; a genuine truncation or a
missing normalization is still refused.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from mixle.enumeration import AutoregressiveEnumerable
from mixle.enumeration.autoregressive import _NORM_TOL


def _model(deficit: float):
    """A two-token model whose every step's probabilities sum to ``1 - deficit``."""
    probs = np.array([0.7, 0.3]) * (1.0 - deficit)

    def next_logprobs(prefix):
        if len(prefix) >= 2:
            return (np.array([], dtype=object), np.array([], dtype=float))
        return (np.array(["a", "b"], dtype=object), np.log(probs))

    return next_logprobs


class ReducedPrecisionNormalizationTest(unittest.TestCase):
    def test_bfloat16_sized_deficit_is_accepted_and_renormalized(self):
        ar = AutoregressiveEnumerable(_model(1.0 - 0.998532), max_len=2, oversample=64)
        tokens, logprobs = ar._steps_np(())
        self.assertAlmostEqual(float(np.sum(np.exp(logprobs))), 1.0, places=12)
        self.assertEqual(list(tokens), ["a", "b"])
        self.assertAlmostEqual(float(logprobs[0]), math.log(0.7), places=12)
        self.assertEqual(ar.count(-50.0), 4)

    def test_exactly_normalized_table_is_untouched(self):
        ar = AutoregressiveEnumerable(_model(0.0), max_len=2, oversample=64)
        _tokens, logprobs = ar._steps_np(())
        np.testing.assert_array_equal(logprobs, np.log(np.array([0.7, 0.3])))

    def test_a_real_truncation_is_still_refused(self):
        for deficit in (0.1, -0.5, 1.5 * _NORM_TOL):
            with self.assertRaises(ValueError, msg=deficit):
                AutoregressiveEnumerable(_model(deficit), max_len=2, oversample=64)._steps_np(())


if __name__ == "__main__":
    unittest.main()
