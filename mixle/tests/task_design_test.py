"""LLM-designed models (mixle.task.design): the LLM proposes a spec, mixle builds + fits it, fallback grounds it.

A deterministic local LLM stand-in returns specs, good and bad, so the build/fit/validate gate and the heuristic
fallback are tested without a network service.
"""

import unittest

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


class DesignModelTest(unittest.TestCase):
    def test_llm_design_is_built_and_fit(self):
        spec = '{"type":"composite","fields":[{"family":"categorical"},{"family":"gaussian"},{"family":"poisson"}]}'
        llm = CallableLLM(lambda prompt, system=None: f"Here is the model:\n```json\n{spec}\n```")
        data = _hetero()
        designed = design_model(data, llm)
        self.assertEqual(designed.source, "llm")
        self.assertEqual(type(designed.estimator).__name__, "CompositeEstimator")
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

    def test_no_fallback_raises(self):
        llm = CallableLLM(lambda prompt, system=None: "nope")  # no JSON object in the reply
        with self.assertRaises(ValueError):
            design_model(_hetero(), llm, fallback=False)

    def test_bad_family_in_spec_falls_back(self):
        llm = CallableLLM(lambda prompt, system=None: '{"family":"not_real"}')
        designed = design_model(_reals(), llm, fallback=True)
        self.assertEqual(designed.source, "fallback")


if __name__ == "__main__":
    unittest.main()
