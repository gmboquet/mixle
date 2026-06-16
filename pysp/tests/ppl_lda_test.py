"""Test LDA / topic model through the pysp.ppl surface."""

import unittest

import numpy as np

from pysp.ppl import LDA


class LDATestCase(unittest.TestCase):
    def test_lda_recovers_topics(self):
        rng = np.random.RandomState(0)
        V, K = 6, 2
        T = [np.array([0.5, 0.3, 0.15, 0.03, 0.01, 0.01]), np.array([0.01, 0.01, 0.03, 0.15, 0.3, 0.5])]

        def gen_doc():
            theta = rng.dirichlet([1.0, 1.0])
            L = rng.randint(30, 60)
            words = [rng.choice(V, p=T[rng.choice(2, p=theta)]) for _ in range(L)]
            v, c = np.unique(words, return_counts=True)
            return list(zip(v.tolist(), c.astype(float).tolist()))

        docs = [gen_doc() for _ in range(600)]
        m = LDA(num_topics=2, vocab_size=6).fit(docs, max_its=60, rng=np.random.RandomState(3))
        topics = m.params["topics"]
        # match recovered topics to truth (order may swap)
        order = sorted(range(K), key=lambda i: np.argmax(topics[i]))
        self.assertGreater(topics[order[0]][0], 0.4)  # first topic favors low word ids
        self.assertGreater(topics[order[1]][5], 0.4)  # second favors high word ids


if __name__ == "__main__":
    unittest.main()
