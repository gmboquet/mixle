"""Conformal calibration must survive a held-out label the student never learned.

``CalibratedTaskModel.calibrate`` builds an index over the student's own labels and looks up each
teacher label's true-class score. A real teacher (or a real dataset split) can hand back a label the
student's class set does not contain -- a rare intent absent from the seed, a new category. That label
has no column in the student's probability vector, so its true-class score is 0: the student is
guaranteed to miss it, and the conformal threshold must *see* that miss (it makes the set-valued
predictor correctly more conservative), not crash the whole calibration pass with a ``KeyError``.

These tests use a tiny fixed-probability fake adapter (no torch) so they exercise exactly the
``calibrate`` lookup/scoring path and nothing else.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from mixle.task.calibrate import CalibratedTaskModel


def _fake_task(labels, row_prob):
    """A TaskModel-shaped stub: an adapter with ``labels`` + a constant ``proba_batch``."""
    row = np.asarray(row_prob, dtype=float)
    adapter = SimpleNamespace(
        labels=list(labels),
        proba_batch=lambda model, inputs: np.tile(row, (len(inputs), 1)),
    )
    return SimpleNamespace(adapter=adapter, model=None)


class UnseenCalibrationLabelTest(unittest.TestCase):
    def test_unseen_label_does_not_crash(self):
        # 'phishing' is not in the student's label set -> used to KeyError on index[str(y)].
        task = _fake_task(["ham", "spam"], [0.7, 0.3])
        model = CalibratedTaskModel(task, alpha=0.1).calibrate(["a", "b", "c", "d"], ["ham", "spam", "phishing", "ham"])
        self.assertTrue(np.isfinite(model.qhat) or model.qhat == float("inf"))

    def test_unseen_label_scored_as_a_miss(self):
        # An unseen true label gets true-class score 0 (nonconformity 1.0), so a calibration set
        # salted with unseen labels yields a threshold at least as conservative as the all-seen one.
        task = _fake_task(["ham", "spam"], [0.7, 0.3])
        n = 50
        seen = CalibratedTaskModel(task, alpha=0.1).calibrate(["x"] * n, ["ham"] * n)
        salted = CalibratedTaskModel(task, alpha=0.1).calibrate(["x"] * n, ["ham"] * (n - 10) + ["phishing"] * 10)
        self.assertGreaterEqual(salted.qhat, seen.qhat)

    def test_all_unseen_labels_still_produces_a_threshold(self):
        # Degenerate but must not crash: every calibration label is unseen -> every score 0.
        task = _fake_task(["ham", "spam"], [0.7, 0.3])
        model = CalibratedTaskModel(task, alpha=0.1).calibrate(["x"] * 20, ["mystery"] * 20)
        # Every point is a max-nonconformity miss, so the threshold saturates at 1.0.
        self.assertAlmostEqual(model.qhat, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
