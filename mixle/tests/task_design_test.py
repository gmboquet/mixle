"""LLM-designed models (mixle.task.design): the LLM proposes a spec, mixle builds + fits it, fallback grounds it.

A deterministic local LLM stand-in returns specs, good and bad, so the build/fit/validate gate and the heuristic
fallback are tested without a network service.
"""

import unittest
from unittest import mock

import numpy as np

from mixle.task.design import (
    ALLOWED_FAMILIES,
    design_model,
    spec_to_estimator,
)
from mixle.task.llm import CallableLLM


def _hetero(n=400, seed=0):
    rng = np.random.RandomState(seed)
    return [("a" if rng.rand() < 0.5 else "b", float(rng.randn()), int(rng.poisson(4))) for _ in range(n)]


def _reals(n=400, seed=0):
    rng = np.random.RandomState(seed)
    return [float(x) for x in rng.randn(n)]


class SpecToEstimatorTest(unittest.TestCase):
    def test_scalar_family(self):
        est = spec_to_estimator({"family": "gaussian"})
        self.assertEqual(type(est).__name__, "GaussianEstimator")

    def test_composite(self):
        est = spec_to_estimator(
            {"type": "composite", "fields": [{"family": "categorical"}, {"family": "gaussian"}, {"family": "poisson"}]}
        )
        self.assertEqual(type(est).__name__, "CompositeEstimator")

    def test_mixture(self):
        est = spec_to_estimator({"type": "mixture", "k": 3, "component": {"family": "gaussian"}})
        self.assertEqual(type(est).__name__, "MixtureEstimator")

    def test_off_allowlist_rejected(self):
        with self.assertRaises(ValueError):
            spec_to_estimator({"family": "definitely_not_a_real_family"})
        self.assertIn("gaussian", ALLOWED_FAMILIES)

    def test_spec_schema_rejects_ambiguous_keys_and_inexact_sizes(self):
        with self.assertRaises(ValueError):
            spec_to_estimator({"family": "gaussian", "type": "mixture"})
        with self.assertRaises(TypeError):
            spec_to_estimator({"type": "mixture", "k": 2.5, "component": {"family": "gaussian"}})
        with self.assertRaises(ValueError):
            spec_to_estimator({"type": "mixture", "k": 1000, "component": {"family": "gaussian"}})
        with self.assertRaises(TypeError):
            spec_to_estimator({"type": "composite", "fields": ["gaussian"]})


class DesignModelTest(unittest.TestCase):
    def test_llm_design_is_built_and_fit(self):
        spec = '{"type":"composite","fields":[{"family":"categorical"},{"family":"gaussian"},{"family":"poisson"}]}'
        llm = CallableLLM(lambda prompt, system=None: f"Here is the model:\n```json\n{spec}\n```")
        data = _hetero()
        designed = design_model(data, llm)
        self.assertEqual(designed.source, "llm")
        self.assertEqual(type(designed.estimator).__name__, "CompositeEstimator")
        self.assertTrue(designed.acceptance.accepted)
        self.assertGreater(designed.acceptance.n_train, 0)
        self.assertGreater(designed.acceptance.n_holdout, 0)
        self.assertTrue(np.isfinite(designed.acceptance.holdout_mean_log_density))
        self.assertIsNone(designed.failure)
        model = designed.fit(data, out=None)
        self.assertTrue(np.isfinite(model.log_density(data[0])))

    def test_llm_can_propose_a_mixture(self):
        llm = CallableLLM(lambda prompt, system=None: '{"type":"mixture","k":2,"component":{"family":"gaussian"}}')
        designed = design_model(_reals(), llm)
        self.assertEqual(designed.source, "llm")
        self.assertEqual(type(designed.estimator).__name__, "MixtureEstimator")

    def test_garbage_falls_back_to_heuristic(self):
        llm = CallableLLM(lambda prompt, system=None: "I cannot help with that.")
        designed = design_model(_hetero(), llm, fallback=True)
        self.assertEqual(designed.source, "fallback")
        self.assertIn("fallback", designed.note)
        self.assertIsNotNone(designed.estimator)
        self.assertTrue(designed.acceptance.accepted)
        self.assertEqual(designed.failure.stage, "json_parse")
        self.assertIn("cannot help", designed.failure.reply_excerpt)

    def test_no_fallback_raises(self):
        llm = CallableLLM(lambda prompt, system=None: "nope")  # no JSON object in the reply
        with self.assertRaises(ValueError):
            design_model(_hetero(), llm, fallback=False)

    def test_bad_family_in_spec_falls_back(self):
        llm = CallableLLM(lambda prompt, system=None: '{"family":"not_real"}')
        designed = design_model(_reals(), llm, fallback=True)
        self.assertEqual(designed.source, "fallback")
        self.assertEqual(designed.failure.proposed_spec, {"family": "not_real"})
        self.assertEqual(designed.failure.stage, "spec_construction")

    def test_schema_incompatible_proposal_is_retained_as_rejection_evidence(self):
        llm = CallableLLM(lambda prompt, system=None: '{"family":"gaussian"}')
        designed = design_model(_hetero(), llm)
        self.assertEqual(designed.source, "fallback")
        self.assertEqual(designed.failure.proposed_spec, {"family": "gaussian"})
        self.assertEqual(designed.failure.stage, "independent_acceptance")
        self.assertIn("structured observations", designed.failure.message)

    def test_held_out_regression_rejects_candidate_and_preserves_receipt(self):
        llm = CallableLLM(lambda prompt, system=None: '{"family":"gaussian"}')
        with mock.patch(
            "mixle.task.design._fit_scores",
            side_effect=[(0.0, -10.0), (0.0, 0.0), (0.0, 0.0)],
        ):
            designed = design_model(_reals(), llm, max_holdout_regret=0.5)
        self.assertEqual(designed.source, "fallback")
        self.assertIsNotNone(designed.failure.acceptance)
        self.assertFalse(designed.failure.acceptance.accepted)
        self.assertEqual(designed.failure.acceptance.holdout_regret, 10.0)

    def test_empty_or_too_small_data_cannot_be_certified(self):
        llm = CallableLLM(lambda prompt, system=None: '{"family":"gaussian"}')
        for data in ([], [1.0, 2.0, 3.0]):
            with self.assertRaises(ValueError):
                design_model(data, llm)

    def test_broken_fallback_is_not_returned_unvalidated(self):
        llm = CallableLLM(lambda prompt, system=None: "no model")

        class _Recommendation:
            estimator = object()

        with mock.patch("mixle.task.recommend.recommend_model", return_value=_Recommendation()):
            with self.assertRaises(AttributeError):
                design_model(_reals(), llm)


if __name__ == "__main__":
    unittest.main()
