"""HiddenMarkovModelDistribution Viterbi decoding: viterbi() and seq_viterbi().

Regression for a confirmed audit finding: neither the scalar viterbi() nor the vectorized
seq_viterbi() (non-numba branch) ever built or used a backpointer array. Decode instead took an
independent argmax(v[t, :]) at every timestep, which is NOT equivalent to a real Viterbi backtrack and
does not, in general, return the joint most-probable state path -- confirmed against brute-force
enumeration and against the library's own correct latent_posterior(x).mode() (forward-backward-based).
seq_viterbi's numba-encoded branch (x1 is not None) had no code path at all and silently returned None
under the class's own default (use_numba=HAS_NUMBA), which is True whenever numba is installed.
"""

import itertools
import unittest

import numpy as np

from mixle.stats import CategoricalDistribution, HiddenMarkovModelDistribution


def _brute_force_map_path(w, trans, topics, seq):
    n_states = len(w)
    best_p, best_path = -1.0, None
    for path in itertools.product(range(n_states), repeat=len(seq)):
        p = w[path[0]]
        for t in range(1, len(path)):
            p *= trans[path[t - 1]][path[t]]
        for t, s in enumerate(path):
            p *= topics[s].density(seq[t])
        if p > best_p:
            best_p, best_path = p, path
    return best_path, float(np.log(best_p))


def _path_log_prob(w, trans, topics, seq, path):
    p = w[path[0]]
    for t in range(1, len(path)):
        p *= trans[path[t - 1]][path[t]]
    for t, s in enumerate(path):
        p *= topics[s].density(seq[t])
    return float(np.log(p))


class ViterbiScalarTest(unittest.TestCase):
    def test_matches_brute_force_map_path_across_random_models(self):
        rng = np.random.RandomState(0)
        for trial in range(40):
            n_states = rng.randint(2, 4)
            topics = []
            for _ in range(n_states):
                p = rng.dirichlet(np.ones(3))
                topics.append(CategoricalDistribution({"x": p[0], "y": p[1], "z": p[2]}))
            w = rng.dirichlet(np.ones(n_states))
            trans = np.array([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)])
            hmm = HiddenMarkovModelDistribution(topics, w=list(w), transitions=trans.tolist())
            seq = [["x", "y", "z"][rng.randint(3)] for _ in range(rng.randint(2, 7))]
            path = hmm.viterbi(seq)
            _, bf_logp = _brute_force_map_path(w, trans, topics, seq)
            my_logp = _path_log_prob(w, trans, topics, seq, path)
            with self.subTest(trial=trial):
                self.assertAlmostEqual(my_logp, bf_logp, places=8)

    def test_agrees_with_latent_posterior_mode(self):
        rng = np.random.RandomState(0)
        topics = []
        for _ in range(3):
            p = rng.dirichlet(np.ones(3))
            topics.append(CategoricalDistribution({"x": p[0], "y": p[1], "z": p[2]}))
        w = rng.dirichlet(np.ones(3))
        trans = np.array([rng.dirichlet(np.ones(3)) for _ in range(3)])
        hmm = HiddenMarkovModelDistribution(topics, w=list(w), transitions=trans.tolist())
        seq = ["x", "y", "y", "z", "x", "y"]
        path = list(hmm.viterbi(seq))
        mode_path = list(hmm.latent_posterior(seq).mode())
        self.assertEqual(path, mode_path)


class SeqViterbiTest(unittest.TestCase):
    def test_non_numba_batch_matches_scalar_viterbi(self):
        rng = np.random.RandomState(1)
        topics = []
        for _ in range(3):
            p = rng.dirichlet(np.ones(3))
            topics.append(CategoricalDistribution({"x": p[0], "y": p[1], "z": p[2]}))
        w = rng.dirichlet(np.ones(3))
        trans = np.array([rng.dirichlet(np.ones(3)) for _ in range(3)])
        hmm = HiddenMarkovModelDistribution(topics, w=list(w), transitions=trans.tolist(), use_numba=False)
        seqs = [[["x", "y", "z"][rng.randint(3)] for _ in range(rng.randint(2, 8))] for _ in range(15)]
        enc = hmm.dist_to_encoder().seq_encode(seqs)
        batch_ptr = hmm.seq_viterbi(enc)
        (_, _idx_bands, _has_next, len_vec, idx_mat, _idx_vec, _enc_data), _, _len_enc = enc[0]
        for s, seq in enumerate(seqs):
            with self.subTest(s=s):
                row = idx_mat[s, : int(len_vec[s])]
                self.assertTrue(np.array_equal(batch_ptr[row], hmm.viterbi(seq)))

    def test_numba_batch_returns_paths_not_none_and_matches_scalar_viterbi(self):
        # Regression: this branch (x1 is not None) used to have no code at all and silently returned
        # None under the class's own default (use_numba=HAS_NUMBA is True whenever numba is installed).
        rng = np.random.RandomState(2)
        topics = []
        for _ in range(3):
            p = rng.dirichlet(np.ones(3))
            topics.append(CategoricalDistribution({"x": p[0], "y": p[1], "z": p[2]}))
        w = rng.dirichlet(np.ones(3))
        trans = np.array([rng.dirichlet(np.ones(3)) for _ in range(3)])
        hmm = HiddenMarkovModelDistribution(topics, w=list(w), transitions=trans.tolist(), use_numba=True)
        self.assertTrue(hmm.use_numba)
        seqs = [[["x", "y", "z"][rng.randint(3)] for _ in range(rng.randint(2, 8))] for _ in range(15)]
        enc = hmm.dist_to_encoder().seq_encode(seqs)
        self.assertIsNotNone(enc[1])  # confirm this test actually exercises the numba-encoded branch
        batch_ptr = hmm.seq_viterbi(enc)
        self.assertIsNotNone(batch_ptr)
        (idx, sz, _enc_data), _len_enc = enc[1]
        tz = np.concatenate([[0], sz]).cumsum()
        for n, seq in enumerate(seqs):
            with self.subTest(n=n):
                path = batch_ptr[tz[n] : tz[n + 1]]
                self.assertTrue(np.array_equal(path, hmm.viterbi(seq)))

    def test_numba_batch_matches_brute_force_map_path(self):
        rng = np.random.RandomState(3)
        topics = []
        for _ in range(3):
            p = rng.dirichlet(np.ones(3))
            topics.append(CategoricalDistribution({"x": p[0], "y": p[1], "z": p[2]}))
        w = rng.dirichlet(np.ones(3))
        trans = np.array([rng.dirichlet(np.ones(3)) for _ in range(3)])
        hmm = HiddenMarkovModelDistribution(topics, w=list(w), transitions=trans.tolist(), use_numba=True)
        seqs = [[["x", "y", "z"][rng.randint(3)] for _ in range(rng.randint(2, 6))] for _ in range(10)]
        enc = hmm.dist_to_encoder().seq_encode(seqs)
        batch_ptr = hmm.seq_viterbi(enc)
        (idx, sz, _enc_data), _len_enc = enc[1]
        tz = np.concatenate([[0], sz]).cumsum()
        for n, seq in enumerate(seqs):
            with self.subTest(n=n):
                path = batch_ptr[tz[n] : tz[n + 1]]
                _, bf_logp = _brute_force_map_path(w, trans, topics, seq)
                my_logp = _path_log_prob(w, trans, topics, seq, path)
                self.assertAlmostEqual(my_logp, bf_logp, places=8)


if __name__ == "__main__":
    unittest.main()
