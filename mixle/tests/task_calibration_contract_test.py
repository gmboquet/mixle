"""Fail-closed classification-calibration contracts."""

import unittest
from types import SimpleNamespace

import numpy as np

from mixle.task.calibrate import (
    CalibratedTaskModel,
    _qhat_from_json,
    _qhat_to_json,
)


class _Adapter:
    labels = ["a", "b"]

    def __init__(self, probabilities=None):
        self.probabilities = probabilities

    def proba_batch(self, _model, rows):
        if self.probabilities is not None:
            return np.asarray(self.probabilities)
        return np.repeat([[0.75, 0.25]], len(rows), axis=0)


def _task(probabilities=None):
    return SimpleNamespace(adapter=_Adapter(probabilities), model=object())


class CalibrationContractTest(unittest.TestCase):
    def test_a_label_outside_the_model_space_is_scored_as_a_miss_not_rejected(self):
        # A teacher or a real split can return a label the student never learned. Refusing to
        # calibrate makes the method unusable on exactly that data; scoring the label as a
        # guaranteed miss is both usable and conservative, since the zero true-class score lowers
        # qhat and widens the prediction sets. mixle/tests/calibrate_unseen_label_test.py owns the
        # full contract -- this asserts only that calibration completes and stays sound.
        calibrated = CalibratedTaskModel(_task()).calibrate(["x"], ["unseen"])
        self.assertIsNotNone(calibrated.qhat)
        seen = CalibratedTaskModel(_task()).calibrate(["x"] * 20, ["a"] * 20)
        salted = CalibratedTaskModel(_task()).calibrate(["x"] * 20, ["a"] * 10 + ["unseen"] * 10)
        self.assertGreaterEqual(salted.qhat, seen.qhat)

    def test_calibration_rows_are_nonempty_aligned_and_row_stochastic(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            CalibratedTaskModel(_task()).calibrate([], [])
        with self.assertRaisesRegex(ValueError, "identical"):
            CalibratedTaskModel(_task()).calibrate(["x", "y"], ["a"])
        with self.assertRaisesRegex(ValueError, "row-stochastic"):
            CalibratedTaskModel(_task([[0.8, 0.8]])).calibrate(["x"], ["a"])
        with self.assertRaisesRegex(ValueError, "shape"):
            CalibratedTaskModel(_task([[1.0]])).calibrate(["x"], ["a"])

    def test_alpha_and_qhat_reject_bool_nan_negative_infinity_and_out_of_range(self):
        for alpha in (True, np.nan, -0.1, 1.1):
            with self.subTest(alpha=repr(alpha)), self.assertRaises(ValueError):
                CalibratedTaskModel(_task(), alpha=alpha)
        for qhat in (True, np.nan, float("-inf"), -0.1, 1.1):
            with self.subTest(qhat=repr(qhat)), self.assertRaises(ValueError):
                CalibratedTaskModel(_task(), qhat=qhat)

    def test_qhat_json_has_one_canonical_infinity_and_rejects_corruption(self):
        self.assertEqual(_qhat_to_json(float("inf")), "inf")
        self.assertEqual(_qhat_from_json("inf"), float("inf"))
        for value in (np.nan, float("-inf"), "nan", "-inf", "0.5", True):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                _qhat_from_json(value)
        for value in (np.nan, float("-inf")):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                _qhat_to_json(value)


if __name__ == "__main__":
    unittest.main()
