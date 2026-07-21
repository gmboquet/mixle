"""Vectorized Markov-chain encoding: byte-parity with the original dict walk, plus its fixed edges.

The chain encoder was the single largest wall-clock sink on chain-bearing fits (a pure-Python
per-token loop: ~86% of a 300k-sequence optimize call). The vectorized encoder must reproduce the
dict walk EXACTLY -- including inv_key_map's first-seen state order, empty and length-1 sequences --
via the sortable fast path and the dict fallback alike, and it must not let numpy's asarray COERCION
merge states the dict kept distinct ([1, "1"] is two states, not one).
"""

import unittest

import numpy as np

from mixle.stats import CompositeDistribution, GaussianDistribution, MarkovChainDistribution, PoissonDistribution


def _dict_walk_oracle(x):
    """The original encoder's semantics, verbatim, as the parity oracle."""
    init_entries, pair_entries, entries_idx0, entries_idx1 = [], [], [], []
    key_map: dict = {}
    for i, entry in enumerate(x):
        if len(entry) == 0:
            continue
        if entry[0] not in key_map:
            key_map[entry[0]] = len(key_map)
        prev = key_map[entry[0]]
        init_entries.append(prev)
        entries_idx0.append(i)
        for j in range(1, len(entry)):
            if entry[j] not in key_map:
                key_map[entry[j]] = len(key_map)
            nxt = key_map[entry[j]]
            pair_entries.append([prev, nxt])
            entries_idx1.append(i)
            prev = nxt
    inv = [None] * len(key_map)
    for k, v in key_map.items():
        inv[v] = k
    pairs = np.asarray(pair_entries) if pair_entries else np.zeros((0, 2), dtype=np.int64)
    return (
        np.asarray(entries_idx0),
        np.asarray(entries_idx1),
        np.asarray(init_entries),
        pairs[:, 0],
        pairs[:, 1],
        list(inv),
    )


def _chain(states):
    return MarkovChainDistribution(
        {s: 1.0 / len(states) for s in states}, {s: {t: 1.0 / len(states) for t in states} for s in states}
    )


class ChainEncodeParityTest(unittest.TestCase):
    def _assert_parity(self, states, data):
        got = _chain(states).dist_to_encoder().seq_encode(data)
        exp = _dict_walk_oracle(data)
        for g, e, name in zip(got[1:7], exp, ("idx0", "idx1", "init", "prev", "next", "inv_key_map")):
            self.assertEqual(
                list(np.asarray(g).ravel().tolist()),
                list(np.asarray(e, dtype=object).ravel().tolist())
                if name == "inv_key_map"
                else list(np.asarray(e).tolist()),
                name,
            )

    def test_string_states_mixed_lengths_including_empty_and_len1(self):
        rng = np.random.RandomState(0)
        states = ["a", "b", "c", "d"]
        data = [[states[rng.randint(4)] for _ in range(int(rng.randint(0, 6)))] for _ in range(3000)]
        self._assert_parity(states, data)

    def test_integer_states(self):
        rng = np.random.RandomState(1)
        data = [[int(v) for v in rng.randint(0, 9, int(rng.randint(1, 5)))] for _ in range(2000)]
        self._assert_parity(list(range(9)), data)

    def test_first_seen_order_is_preserved_not_sorted(self):
        got = _chain(["z", "m", "a"]).dist_to_encoder().seq_encode([["z", "m"], ["a", "z"]])
        self.assertEqual([str(v) for v in got[6].tolist()], ["z", "m", "a"], "inv_key_map must be first-seen order")

    def test_all_length_one_sequences_have_no_transitions(self):
        got = _chain(["a", "b"]).dist_to_encoder().seq_encode([["a"], ["b"], ["a"]])
        self.assertEqual(len(got[2]), 0)  # idx1 empty
        self.assertEqual(len(got[4]), 0)  # prev empty
        self.assertEqual(list(got[1]), [0, 1, 2])  # every row contributes an init

    def test_mixed_type_states_stay_distinct(self):
        """np.asarray coercion must never merge int 1 with str \"1\" -- the dict walk keeps them
        distinct, and so must both the fast path (which refuses mixed types) and the fallback's
        object-dtype inv_key_map (a pre-existing coercion hazard, fixed alongside the fast path)."""
        got = _chain(["a"]).dist_to_encoder().seq_encode([[1, "1", 1], ["1", 1]])
        inv = got[6].tolist()
        self.assertEqual(len(inv), 2)
        self.assertEqual({type(v) for v in inv}, {int, str})

    def test_scoring_parity_end_to_end(self):
        rng = np.random.RandomState(2)
        states = ["a", "b", "c"]
        d = _chain(states)
        data = [[states[rng.randint(3)] for _ in range(int(rng.randint(1, 7)))] for _ in range(500)]
        enc = d.dist_to_encoder().seq_encode(data)
        per_row = [d.log_density(seq) for seq in data]
        np.testing.assert_allclose(d.seq_log_density(enc), per_row, rtol=1e-12)

    def test_scoring_an_all_length_one_corpus_does_not_crash_or_misscore(self):
        """A corpus where EVERY sequence has length 1 (idx1/prev/next all empty -- e.g. a
        single-domain task decomposition fit) used to crash `seq_log_density` with a numpy
        same-kind casting error: `np.bincount([], weights=[], minlength=n)` silently returns
        int64 zeros (not float64) when both its inputs are empty, and the subsequent
        `rv[idx0] += <float log-density>` then fails to cast. `test_scoring_parity_end_to_end`
        above doesn't catch this because it mixes lengths 1-6, so `idx1` is never globally empty."""
        states = ["a", "b", "c"]
        d = _chain(states)
        data = [["a"], ["b"], ["a"], ["c"]]
        enc = d.dist_to_encoder().seq_encode(data)
        per_row = [d.log_density(seq) for seq in data]
        np.testing.assert_allclose(d.seq_log_density(enc), per_row, rtol=1e-12)


class CompositeValidationFastPathTest(unittest.TestCase):
    def setUp(self):
        self.enc = CompositeDistribution((GaussianDistribution(0.0, 1.0), PoissonDistribution(2.0))).dist_to_encoder()

    def test_well_formed_rows_encode(self):
        self.assertIsNotNone(self.enc.seq_encode([(1.0, 2), (0.5, 3)]))

    def test_error_paths_keep_the_per_row_contract_messages(self):
        # the C-speed validation pass must hand EVERY malformed shape to the original per-row loop,
        # so the error still names the offending row
        for bad in ([(1.0, 2), "ab"], [(1.0, 2), (1.0,)], [(1.0, 2), 5], [(1.0, 2), (1.0, 2, 3)]):
            with self.assertRaises(Exception) as ctx:
                self.enc.seq_encode(bad)
            self.assertIn("row 1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
