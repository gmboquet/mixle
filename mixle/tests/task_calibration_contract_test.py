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
    def test_calibration_support_must_equal_the_model_label_space(self):
        calibrated = CalibratedTaskModel(_task())
        with self.assertRaisesRegex(ValueError, "outside the model label space"):
            calibrated.calibrate(["x"], ["unseen"])
        self.assertIsNone(calibrated.qhat)

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
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                CalibratedTaskModel(_task(), alpha=alpha)
        for qhat in (True, np.nan, float("-inf"), -0.1, 1.1):
            with self.subTest(qhat=qhat), self.assertRaises(ValueError):
                CalibratedTaskModel(_task(), qhat=qhat)

    def test_qhat_json_has_one_canonical_infinity_and_rejects_corruption(self):
        self.assertEqual(_qhat_to_json(float("inf")), "inf")
        self.assertEqual(_qhat_from_json("inf"), float("inf"))
        for value in (np.nan, float("-inf"), "nan", "-inf", "0.5", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _qhat_from_json(value)
        for value in (np.nan, float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _qhat_to_json(value)


if __name__ == "__main__":
    unittest.main()
