"""Classification conformal sets (mixle.inference.conformal): distribution-free label coverage from any scorer.

The guarantee is the point: even when the class scores are *not* a describable probability (a softmax over a
ReLU net), the prediction sets cover the true label at >= 1 - alpha on exchangeable held-out data.
"""

import unittest

import numpy as np

from mixle.inference.conformal import conformal_label_sets, conformal_label_threshold


def _scores_and_labels(n, k, seed, sharpness=2.0):
    # synthetic K-class scorer: scores are deliberately NOT calibrated probabilities (unnormalized logits->softmax)
    rng = np.random.RandomState(seed)
    y = rng.randint(0, k, n)
    logits = rng.randn(n, k)
    logits[np.arange(n), y] += sharpness  # the true class tends to score higher, but noisily
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = ex / ex.sum(axis=1, keepdims=True)
    return prob, y


class ConformalClassificationTest(unittest.TestCase):
    def test_marginal_coverage_holds(self):
        prob, y = _scores_and_labels(4000, 5, seed=0)
        cal, test = slice(0, 2000), slice(2000, 4000)
        alpha = 0.1
        sets, qhat = conformal_label_sets(prob[cal][np.arange(2000), y[cal]], prob[test], alpha=alpha)
        covered = sets[np.arange(2000), y[test]].mean()
        self.assertGreaterEqual(covered, 1.0 - alpha - 0.03)  # finite-sample slack
        self.assertTrue(np.isfinite(qhat))

    def test_smaller_alpha_gives_larger_sets(self):
        prob, y = _scores_and_labels(3000, 6, seed=1)
        cal_true = prob[:1500][np.arange(1500), y[:1500]]
        sizes = {}
        for alpha in (0.2, 0.05):
            sets, _ = conformal_label_sets(cal_true, prob[1500:], alpha=alpha)
            sizes[alpha] = sets.sum(axis=1).mean()
        self.assertGreaterEqual(sizes[0.05], sizes[0.2])  # higher coverage -> bigger sets

    def test_threshold_matches_sets(self):
        prob, y = _scores_and_labels(1000, 4, seed=2)
        cal_true = prob[:500][np.arange(500), y[:500]]
        qhat = conformal_label_threshold(cal_true, alpha=0.1)
        sets, qhat2 = conformal_label_sets(cal_true, prob[500:], alpha=0.1)
        self.assertAlmostEqual(qhat, qhat2, places=12)
        expected = (1.0 - prob[500:]) <= qhat
        self.assertTrue(np.array_equal(sets, expected))


if __name__ == "__main__":
    unittest.main()
