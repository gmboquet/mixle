"""Weighted determinization of a quantized terminal HMM -> exact, duplicate-free n-best sequences.

Determinization (Mohri 1997; Mohri & Riley 2002) rebuilds an ambiguous HMM over belief states so each
sequence has one path; the result reproduces the exact marginals and ranks sequences (not paths). It
terminates iff the twins property holds (always for acyclic HMMs); a cyclic ergodic chain whose belief
drifts per prefix is not finitely determinizable and is reported via EnumerationError.
"""

import itertools
import unittest

import numpy as np

from pysp.stats import QuantizedHiddenMarkovModelDistribution as Q
from pysp.stats.compute.pdist import EnumerationError


def _brute(d, alphabet, max_len=6):
    out = []
    for length in range(1, max_len + 1):
        for pre in itertools.product(alphabet, repeat=length - 1):
            s = list(pre) + ["."]
            lp = d.log_density(s)
            if np.isfinite(lp):
                out.append((s, lp))
    out.sort(key=lambda kv: -kv[1])
    return out


class DeterminizableTest(unittest.TestCase):
    """Acyclic ambiguous HMM (shared emissions on states 0,1; state 2 emits only the terminal)."""

    def _dist(self):
        return Q.left_to_right(
            0.5,
            ["a", "b", "."],
            [[-1, 0, 1], [-1, -1, 0], [-1, -1, 0]],  # acyclic for reachable sequences
            [[0, 1, 2], [1, 0, 2], [-1, -1, 0]],  # states 0,1 share {a,b,.}; state 2 only '.'
            initial_exponents=[0, 1, 2],
            terminal_values={"."},
        )

    def test_determinization_is_exact_and_unambiguous(self):
        d = self._dist()
        det = d.determinize()
        order = _brute(d, ["a", "b"])
        # (1) the determinized machine reproduces the original marginal exactly
        for s, lp in order:
            self.assertAlmostEqual(det.log_density(s), lp, places=12)
        # (2) enumeration is duplicate-free and in exact marginal order (by probability level; ties allowed)
        enum = det.enumerator().top_k(len(order))
        self.assertEqual(len({tuple(s) for s, _ in enum}), len(enum))
        self.assertEqual([round(lp, 9) for _, lp in enum], [round(lp, 9) for _, lp in order])
        self.assertAlmostEqual(sum(np.exp(lp) for _, lp in enum), 1.0, delta=1e-9)

    def test_seek_returns_distinct_sequences_at_correct_level(self):
        d = self._dist()
        det = d.determinize()
        order = _brute(d, ["a", "b"])
        seen = set()
        for k in range(len(order)):
            v = tuple(det.enumerator().seek(k).value)
            self.assertNotIn(v, seen)  # determinized => one entry per sequence, no path over-count
            seen.add(v)
            self.assertAlmostEqual(round(det.log_density(list(v)), 9), round(order[k][1], 9), places=9)

    def test_sampler_produces_terminated_sequences(self):
        s = self._dist().determinize().sampler(seed=0).sample(20)
        self.assertEqual(len(s), 20)
        for seq in s:
            self.assertEqual(seq[-1], ".")


class NonDeterminizableTest(unittest.TestCase):
    def test_ergodic_self_loop_is_reported(self):
        # full self-loop chain with shared emissions: the belief drifts through a new point per prefix,
        # so determinization does not terminate -- detected via the max_states cap.
        d = Q.left_to_right(
            0.5,
            ["a", "b", "."],
            [[0, 1, 2], [-1, 0, 2], [-1, -1, 0]],  # state 0 self-loops (cyclic) + shared emissions
            [[0, 1, 3], [1, 0, 3], [-1, -1, 0]],
            initial_exponents=[0, 1, 2],
            terminal_values={"."},
        )
        with self.assertRaises(EnumerationError):
            d.determinize(max_states=64)


class GeneralHmmTest(unittest.TestCase):
    """Determinization of a GENERAL (non-quantized) terminal HMM: float probabilities are rationalized,
    then determinized exactly. Plus the deepening sub-linear seek (robust to cost-spectrum gaps)."""

    def _dist(self):
        from pysp.stats import CategoricalDistribution as Cat
        from pysp.stats import HiddenMarkovModelDistribution as H

        return H(
            [Cat({"a": 0.5, "b": 0.3, ".": 0.2}), Cat({"a": 0.3, "b": 0.4, ".": 0.3}), Cat({".": 1.0})],
            w=[0.5, 0.4, 0.1],
            transitions=[[0.0, 0.6, 0.4], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],  # acyclic, shared emissions
            terminal_values={"."},
        )

    def test_general_determinization_exact_and_unambiguous(self):
        d = self._dist()
        det = d.determinize()
        order = _brute(d, ["a", "b"])
        for s, lp in order:
            self.assertAlmostEqual(det.log_density(s), lp, places=10)  # rationalized -> exact here
        enum = det.enumerator().top_k(len(order))
        self.assertEqual(len({tuple(s) for s, _ in enum}), len(enum))
        self.assertEqual([round(lp, 9) for _, lp in enum], [round(lp, 9) for _, lp in order])
        self.assertAlmostEqual(sum(np.exp(lp) for _, lp in enum), 1.0, delta=1e-9)

    def test_sublinear_seek_correct_and_bounded(self):
        d = self._dist()
        det = d.determinize()
        order = _brute(d, ["a", "b"])
        lp_o = [round(lp, 9) for _, lp in order]
        seen = set()
        for k in range(len(order)):
            v = tuple(det.enumerator().seek(k).value)  # deepening structural seek, no prefix enumeration
            self.assertNotIn(v, seen)
            seen.add(v)
            self.assertAlmostEqual(round(det.log_density(list(v)), 9), lp_o[k], places=9)
        with self.assertRaises(IndexError):  # robustly bounded: index past support raises (no false-stop)
            det.enumerator().seek(len(order))


if __name__ == "__main__":
    unittest.main()
