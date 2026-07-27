"""Impossible HMM evidence is explicit and never becomes synthetic Baum-Welch mass."""

import unittest
import warnings

import numpy as np

import mixle.stats as stats
from mixle.inference import optimize
from mixle.utils.vector import ImpossibleEvidenceError


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
    def test_impossible_observation_rejects_the_estep_transactionally(self):
        for use_numba in (True, False):
            hmm = _hmm(use_numba, True)
            accumulator = hmm.estimator().accumulator_factory().make()
            encoded = hmm.dist_to_encoder().seq_encode(_IMPOSSIBLE)
            with warnings.catch_warnings(), self.assertRaisesRegex(ImpossibleEvidenceError, "zero-probability"):
                warnings.simplefilter("error")
                accumulator.seq_update(encoded, np.ones(len(_IMPOSSIBLE)), hmm)
            np.testing.assert_array_equal(accumulator.init_counts, [0.0, 0.0])
            np.testing.assert_array_equal(accumulator.state_counts, [0.0, 0.0])
            np.testing.assert_array_equal(accumulator.trans_counts, np.zeros((2, 2)))

    def test_impossible_sequence_log_density_is_neg_inf_not_nan(self):
        for use_numba in (True, False):
            hmm = _hmm(use_numba, True)
            enc = hmm.dist_to_encoder().seq_encode([["a", "b", "c", "b", "a"], ["a", "b", "a", "b", "a"]])
            ll = np.asarray(hmm.seq_log_density(enc), dtype=float)
            self.assertEqual(ll[0], -np.inf)  # impossible sequence
            self.assertTrue(np.isfinite(ll[1]))  # normal sequence unaffected

    def test_numba_numpy_bit_identical_on_normal_data(self):
        # the guards only fire on impossible rows, so ordinary fits must be unchanged across backends
        init_n = _hmm(True, False)
        init_p = _hmm(False, False)
        rn = optimize(
            _NORMAL,
            init_n.estimator(),
            max_its=8,
            out=None,
            rng=np.random.RandomState(2),
            prev_estimate=init_n,
        )
        rp = optimize(
            _NORMAL,
            init_p.estimator(),
            max_its=8,
            out=None,
            rng=np.random.RandomState(2),
            prev_estimate=init_p,
        )

        def flat(m):
            return np.concatenate(
                [
                    np.asarray(m.log_w, float).ravel(),
                    np.asarray(m.log_transitions, float).ravel(),
                    np.asarray(m.transitions, float).ravel(),
                ]
            )

        np.testing.assert_allclose(flat(rn), flat(rp), rtol=0.0, atol=1.0e-15)


if __name__ == "__main__":
    unittest.main()
