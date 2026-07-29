"""Focused regressions for the 0.8.0 lifecycle audit."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import numpy as np

from mixle.lifecycle import Model, propose, saddle_suspect


class _PosteriorModel:
    components = (object(), object())

    def __init__(self, posterior):
        self._posterior = posterior

    def posterior(self, value):
        return self._posterior


class _IdentityEncoder:
    def seq_encode(self, rows):
        return rows


class _Scorer:
    def __init__(self, scores):
        self.scores = scores

    def dist_to_encoder(self):
        return _IdentityEncoder()

    def seq_log_density(self, encoded):
        return self.scores


class SaddleAndEvaluationTest(unittest.TestCase):
    def test_saddle_validation_rejects_malformed_posteriors(self):
        for posterior in ([np.nan], [0.5], [0.6, 0.6], [-0.1, 1.1]):
            with self.subTest(posterior=repr(posterior)), self.assertRaises(ValueError):
                saddle_suspect(_PosteriorModel(posterior), [1])

    def test_saddle_sampling_does_not_consume_beyond_bound(self):
        consumed = []

        def rows():
            for value in range(10):
                consumed.append(value)
                yield value

        self.assertTrue(saddle_suspect(_PosteriorModel([0.5, 0.5]), rows(), sample=2))
        self.assertEqual(consumed, [0, 1])

    def test_evaluate_requires_one_valid_score_per_input_row(self):
        for scores in (3.0, [1.0], [1.0, 2.0, 3.0, 4.0], [1.0, np.nan, 2.0]):
            model = Model()
            model.fitted = _Scorer(scores)
            with self.subTest(scores=repr(scores)), self.assertRaises(ValueError):
                model.evaluate([1, 2, 3])
        model = Model()
        model.fitted = _Scorer([1.0, 2.0, 3.0])
        self.assertEqual(model.evaluate([1, 2, 3])["n"], 3)
        with self.assertRaises(ValueError):
            model.evaluate([])


class ReplayAndProposalTest(unittest.TestCase):
    def test_fit_uses_one_replayable_snapshot(self):
        observed = []
        fitted = object()

        def optimize(data, spec, **kwargs):
            observed.append(("optimize", list(data)))
            return fitted

        def inspect_saddle(model, data):
            observed.append(("saddle", list(data)))
            return False

        with (
            patch("mixle.inference.optimize", side_effect=optimize),
            patch("mixle.inference.certify", return_value=None),
            patch("mixle.lifecycle.saddle_suspect", side_effect=inspect_saddle),
        ):
            Model(object()).fit(value for value in [1, 2, 3])
        self.assertEqual(observed, [("optimize", [1, 2, 3]), ("saddle", [1, 2, 3])])

    def test_invalid_candidate_budget_is_rejected_before_proposer_work(self):
        with patch("mixle.task.recommend_model") as recommend:
            for value in (True, 0.5, np.nan, -1):
                with self.subTest(max_candidates=repr(value)), self.assertRaises(ValueError):
                    propose([1, 2, 3], max_candidates=value)
            recommend.assert_not_called()

    def test_nan_candidate_is_quarantined_instead_of_winning(self):
        class Estimator:
            def to_dict(self):
                return {"estimator": "recommended"}

        recommendation = types.SimpleNamespace(
            estimator=Estimator(),
            fields=[],
            dependencies=[],
            warnings=[],
        )

        class Candidate:
            def __init__(self, scores):
                self.scores = scores

            def dist_to_encoder(self):
                return _IdentityEncoder()

            def seq_log_density(self, encoded):
                return np.asarray(self.scores[: len(encoded)])

        fitted = [Candidate([np.nan, np.nan]), Candidate([1.0, 1.0])]

        with (
            patch("mixle.task.recommend_model", return_value=recommendation),
            patch("mixle.utils.automatic.get_estimator", side_effect=RuntimeError("no baseline")),
            patch("mixle.inference.optimize", side_effect=fitted),
        ):
            model = propose([1, 2, 3, 4], holdout=0.5)
        self.assertEqual(model.frontier[0]["name"], "structured")
        self.assertIn("error", next(row for row in model.frontier if row["name"] == "recommended"))


if __name__ == "__main__":
    unittest.main()
