"""Tests for cross-modal RAG with a raw-data fallback (mixle.reason.store)."""

import unittest

import numpy as np

from mixle.reason import CrossModalStore, Evidence, Latent


def _corpus(rng, n, d):
    """Each item is a noisy linear readout of the shared latent; its embedding key is a lossy 1-d
    projection of the readout. coarse = high-variance (embedding); fine = low-variance (raw)."""
    z_true = rng.normal(size=d)
    payloads, keys = [], []
    for _ in range(n):
        H = rng.normal(size=(1, d))
        y = float((H @ z_true)[0] + rng.normal(0, 0.1))
        payloads.append({"H": H, "y": np.array([y])})
        keys.append([y])  # lossy 1-d embedding: just the scalar readout
    coarse = lambda p: Evidence(p["H"], p["y"], [[4.0]], "emb")  # noqa: E731  (lossy)
    fine = lambda p: Evidence(p["H"], p["y"], [[0.01]], "raw")  # noqa: E731  (precise)
    return z_true, np.array(keys), payloads, coarse, fine


class RetrieveTest(unittest.TestCase):
    def test_retrieve_nearest_by_key(self):
        keys = np.array([[0.0], [1.0], [2.0], [3.0]])
        store = CrossModalStore(keys, list(range(4)), coarse=lambda p: None, fine=lambda p: None)
        self.assertEqual(store.retrieve([2.1], k=2), [2, 3])
        self.assertEqual(store.retrieve([-0.4], k=1), [0])

    def test_len_and_validation(self):
        with self.assertRaises(ValueError):
            CrossModalStore(np.zeros((3, 2)), [1, 2], coarse=lambda p: None, fine=lambda p: None)


class AssimilateTest(unittest.TestCase):
    def test_raw_fallback_recovers_accuracy_embedding_loses(self):
        rng = np.random.RandomState(0)
        d = 4
        z_true, keys, payloads, coarse, fine = _corpus(rng, 40, d)
        store = CrossModalStore(keys, payloads, coarse=coarse, fine=fine)
        prior = Latent.vector(d, var=100.0)

        # embedding-only: force epsilon huge so the cheap (lossy) evidence is always used.
        emb_only, steps_emb = store.assimilate(prior, [z_true @ rng.normal(size=d)], k=20, epsilon=1e9)
        # raw-fallback: epsilon 0, so raw is fetched whenever it helps the query more.
        raw_fb, steps_raw = store.assimilate(prior, [z_true @ rng.normal(size=d)], k=20, epsilon=0.0)

        err_emb = np.linalg.norm(emb_only.mean() - z_true)
        err_raw = np.linalg.norm(raw_fb.mean() - z_true)
        self.assertLess(err_raw, err_emb)  # raw fallback is closer to the truth
        self.assertTrue(all(s.fidelity == "embedding" for s in steps_emb))
        self.assertTrue(any(s.fidelity == "raw" for s in steps_raw))  # raw actually fetched

    def test_epsilon_gates_raw_fetch(self):
        rng = np.random.RandomState(1)
        _, keys, payloads, coarse, fine = _corpus(rng, 15, 3)
        store = CrossModalStore(keys, payloads, coarse=coarse, fine=fine)
        prior = Latent.vector(3, var=50.0)
        _, steps_lo = store.assimilate(prior, [0.0], k=10, epsilon=0.0)
        _, steps_hi = store.assimilate(prior, [0.0], k=10, epsilon=1e9)
        n_raw_lo = sum(s.fidelity == "raw" for s in steps_lo)
        n_raw_hi = sum(s.fidelity == "raw" for s in steps_hi)
        self.assertGreater(n_raw_lo, n_raw_hi)  # lower epsilon fetches raw more often
        self.assertEqual(n_raw_hi, 0)

    def test_query_restricts_sufficiency_to_a_coordinate(self):
        # Sufficiency measured on a single query coordinate still runs and returns provenance.
        rng = np.random.RandomState(2)
        _, keys, payloads, coarse, fine = _corpus(rng, 10, 3)
        store = CrossModalStore(keys, payloads, coarse=coarse, fine=fine)
        belief, steps = store.assimilate(Latent.vector(3, var=20.0), [0.0], k=5, query=[0], epsilon=0.0)
        self.assertEqual(len(steps), 5)
        self.assertLess(belief.marginal([0]).entropy(), Latent.vector(3, var=20.0).marginal([0]).entropy())


class ActiveRetrievalTest(unittest.TestCase):
    def test_greedy_active_beats_random_order(self):
        # Greedily fetching the highest-EIG item each step reduces entropy at least as fast as a
        # fixed arbitrary order.
        rng = np.random.RandomState(3)
        d = 4
        _, keys, payloads, coarse, fine = _corpus(rng, 30, d)
        store = CrossModalStore(keys, payloads, coarse=coarse, fine=fine)

        # greedy: 5 active picks
        greedy = Latent.vector(d, var=100.0)
        used = set()
        for _ in range(5):
            cands = [i for i in range(len(store)) if i not in used]
            idx, gain = store.next_evidence(greedy, candidates=cands, fidelity="fine")
            used.add(idx)
            greedy = greedy.update(*(lambda e: (e.H, e.y, e.R))(store.fine(store.payloads[idx])))
            self.assertGreaterEqual(gain, 0.0)

        # random: first 5 items in arbitrary order
        randb = Latent.vector(d, var=100.0)
        for idx in range(5):
            e = store.fine(store.payloads[idx])
            randb = randb.update(e.H, e.y, e.R)

        self.assertLessEqual(greedy.entropy(), randb.entropy() + 1e-9)

    def test_next_evidence_picks_max_gain(self):
        rng = np.random.RandomState(4)
        _, keys, payloads, coarse, fine = _corpus(rng, 12, 3)
        store = CrossModalStore(keys, payloads, coarse=coarse, fine=fine)
        b = Latent.vector(3, var=100.0)
        idx, gain = store.next_evidence(b, fidelity="fine")
        # brute-force check it is the argmax
        gains = []
        base = b.entropy()
        for i in range(len(store)):
            e = store.fine(store.payloads[i])
            gains.append(base - b.update(e.H, e.y, e.R).entropy())
        self.assertEqual(idx, int(np.argmax(gains)))
        self.assertAlmostEqual(gain, max(gains), places=10)


if __name__ == "__main__":
    unittest.main()
