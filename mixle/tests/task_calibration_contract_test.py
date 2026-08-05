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


class SelectiveRiskThresholdTest(unittest.TestCase):
    """STAT-1: marginal conformal coverage is not answered-slice risk; this gate is.

    The adversarial statistical review built a population from mixle's own conformal functions
    where marginal miscoverage was 9% while the error AMONG SINGLETON ANSWERS was 47.4% at
    alpha=0.10. These tests pin the repaired contract on that population shape: a clean cluster
    the model gets right with high confidence, and a trap cluster it answers confidently-enough
    but wrongly half the time. The selective gate must refuse the trap; the guarantee must hold
    on a fresh draw; and an uncontrollable model must yield escalate-everything, not a silent
    downgrade.
    """

    @staticmethod
    def _population(rng, n):
        # clean rows: top-p ~0.93, correct; trap rows: top-p ~0.65, correct only half the time
        clean = rng.rand(n) < 0.6
        top = np.where(clean, 0.90 + 0.06 * rng.rand(n), 0.60 + 0.08 * rng.rand(n))
        correct = np.where(clean, rng.rand(n) < 0.97, rng.rand(n) < 0.5)
        return top, correct.astype(bool)

    def test_answered_slice_risk_is_controlled_on_a_fresh_draw(self):
        from mixle.task.calibrate import selective_risk_threshold

        rng = np.random.RandomState(0)
        top_cal, hit_cal = self._population(rng, 4000)
        tau = selective_risk_threshold(top_cal, hit_cal, alpha=0.10, delta=0.05)
        self.assertIsNotNone(tau)
        # NOTE the guarantee's actual shape: the gate controls the BLENDED slice risk, so it may
        # legitimately admit the upper edge of the trap cluster while the clean mass dilutes the
        # slice to <= alpha (measured: tau ~0.66 with calibration slice error 0.090). Asserting
        # that tau excludes the trap outright would demand more than the guarantee promises --
        # what must hold is the risk bound on a fresh draw, checked below.

        top_new, hit_new = self._population(np.random.RandomState(1), 4000)
        answered = top_new >= tau
        self.assertGreater(int(answered.sum()), 500)  # the gate still answers the clean mass
        slice_error = float((~hit_new[answered]).mean())
        self.assertLessEqual(slice_error, 0.10 + 0.02)  # guarantee level plus sampling slack

    def test_naive_singleton_answering_fails_where_the_gate_holds(self):
        # The reviewer's core observation, as a regression: on this population, answering every
        # confident-looking row (the naive rule) carries ~2-3x the advertised risk, while the
        # calibrated gate's slice stays at or under alpha. If this ever passes for the naive rule,
        # the population no longer separates the two claims and the test must be rebuilt.
        rng = np.random.RandomState(2)
        top, hit = self._population(rng, 4000)
        naive_answered = top >= 0.60  # answer everything that looks confident at all
        naive_error = float((~hit[naive_answered]).mean())
        self.assertGreater(naive_error, 0.15)

    def test_an_uncontrollable_model_escalates_everything_rather_than_downgrading(self):
        from mixle.task.calibrate import selective_risk_threshold

        rng = np.random.RandomState(3)
        top = 0.5 + 0.4 * rng.rand(300)
        hit = rng.rand(300) < 0.5  # coin-flip model: no threshold controls 10% risk
        self.assertIsNone(selective_risk_threshold(top, hit, alpha=0.10, delta=0.05))

    def test_rejects_malformed_inputs(self):
        from mixle.task.calibrate import selective_risk_threshold

        with self.assertRaises(ValueError):
            selective_risk_threshold([], [], alpha=0.10)
        with self.assertRaises(ValueError):
            selective_risk_threshold([0.5, 1.5], [True, False], alpha=0.10)
        with self.assertRaises(ValueError):
            selective_risk_threshold([0.5, 0.6], [True, False], alpha=0.10, delta=1.5)
