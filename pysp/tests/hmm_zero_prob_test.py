"""Regression: a zero-probability observation must not NaN-poison HMM Baum-Welch / EM.

An observation with zero emission probability under every state (all-``-inf`` log-emissions) made the
max-subtracted emission row NaN, which the linear-space Baum-Welch kernels turned into NaN ``pi``/``xi``/
``alpha`` -> NaN EM sufficient statistics with no error raised. The fix zeroes the impossible row at the
emission level and guards the forward (``alpha_sum``) and backward (``beta_sum``) normalizations in the
numba kernels. The numba and numpy paths must stay bit-identical on ordinary data.
"""

import unittest
import warnings

import numpy as np

import pysp.stats as stats
from pysp.inference import optimize


def _hmm(use_numba, zero_symbol):
    emit = (
        ({"a": 0.6, "b": 0.4, "c": 0.0}, {"a": 0.3, "b": 0.7, "c": 0.0})
        if zero_symbol
        else ({"a": 0.6, "b": 0.4}, {"a": 0.3, "b": 0.7})
    )
    return stats.HiddenMarkovModelDistribution(
        [stats.CategoricalDistribution(emit[0]), stats.CategoricalDistribution(emit[1])],
        [0.5, 0.5],
        [[0.7, 0.3], [0.4, 0.6]],
        len_dist=stats.CategoricalDistribution({5: 1.0}),
        use_numba=use_numba,
    )


def _has_nan(model) -> list[str]:
    bad = [
        a
        for a in ("log_w", "log_transitions", "w", "transitions")
        if np.any(np.isnan(np.asarray(getattr(model, a), float)))
    ]
    bad += [
        f"topic{i}" for i, c in enumerate(model.topics) if np.any(np.isnan(np.asarray(list(c.pmap.values()), float)))
    ]
    return bad


# 'c' has zero emission probability in every state -> impossible observation, mid-sequence.
_IMPOSSIBLE = [
    ["a", "b", "a", "b", "a"],
    ["b", "a", "b", "a", "b"],
    ["a", "b", "c", "b", "a"],
    ["b", "a", "b", "b", "a"],
]
_NORMAL = [["a", "b", "a", "b", "a"], ["b", "a", "b", "a", "b"], ["a", "a", "b", "b", "a"], ["b", "b", "a", "a", "b"]]


class HmmZeroProbTest(unittest.TestCase):
    def test_impossible_observation_does_not_nan_poison_em(self):
        for use_numba in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a NaN/overflow warning fails the test
                res = optimize(
                    _IMPOSSIBLE, _hmm(use_numba, True).estimator(), max_its=5, out=None, rng=np.random.RandomState(1)
                )
            self.assertEqual(_has_nan(res), [], f"use_numba={use_numba}")

    def test_impossible_sequence_log_density_is_neg_inf_not_nan(self):
        for use_numba in (True, False):
            hmm = _hmm(use_numba, True)
            enc = hmm.dist_to_encoder().seq_encode([["a", "b", "c", "b", "a"], ["a", "b", "a", "b", "a"]])
            ll = np.asarray(hmm.seq_log_density(enc), dtype=float)
            self.assertEqual(ll[0], -np.inf)  # impossible sequence
            self.assertTrue(np.isfinite(ll[1]))  # normal sequence unaffected

    def test_numba_numpy_bit_identical_on_normal_data(self):
        # the guards only fire on impossible rows, so ordinary fits must be unchanged across backends
        rn = optimize(_NORMAL, _hmm(True, False).estimator(), max_its=8, out=None, rng=np.random.RandomState(2))
        rp = optimize(_NORMAL, _hmm(False, False).estimator(), max_its=8, out=None, rng=np.random.RandomState(2))

        def flat(m):
            return np.concatenate(
                [
                    np.asarray(m.log_w, float).ravel(),
                    np.asarray(m.log_transitions, float).ravel(),
                    np.asarray(m.transitions, float).ravel(),
                ]
            )

        self.assertTrue(np.array_equal(flat(rn), flat(rp), equal_nan=True))


if __name__ == "__main__":
    unittest.main()
