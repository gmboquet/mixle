"""Calibrated generator (mixle.task.calibrated_generator): conformal accept-or-abstain for generation.

Coverage of the conformal accept decision must hold on held-out data the same way it does for
CalibratedTaskModel's label sets; abstention (ABSTAIN) must compose with Cascade as an escalation
signal; and serving must be reproducible given a seed.
"""

import hashlib
import unittest

import numpy as np

from mixle.task.calibrated_generator import ABSTAIN, CalibratedGenerator
from mixle.task.cascade import Cascade

# --- synthetic "double the prompt" generation task -------------------------------------------------
#
# A candidate is (n, guess). The correct answer for prompt n is 2 * n. `generate` draws k candidates:
# the correct answer is included with probability `hit_prob`, plus decoys at a random offset. `score`
# is a pure function of the candidate alone (no access to ground truth beyond the same "double it" prior
# a real verifier might encode) with a small deterministic jitter, so it ranks well but not perfectly.


def _stable_unit(obj) -> float:
    """Deterministic pseudo-noise in [0, 1) from a hash of `obj` -- keeps `score` a pure function."""
    digest = hashlib.sha256(repr(obj).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def make_generate(hit_prob: float, decoy_spread: int = 6):
    def generate(prompt, k, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        n = prompt
        correct = 2 * n
        cands = []
        if rng.random() < hit_prob:
            cands.append((n, correct))
        while len(cands) < k:
            offset = int(rng.integers(-decoy_spread, decoy_spread + 1))
            if offset == 0:
                continue
            cands.append((n, correct + offset))
        rng.shuffle(cands)
        return cands

    return generate


def score(candidate) -> float:
    n, guess = candidate
    deviation = abs(guess - 2 * n)
    jitter = _stable_unit(candidate) * 0.5
    return -float(deviation) + jitter


def is_correct(prompt, candidate) -> bool:
    n, guess = candidate
    return guess == 2 * n and n == prompt


class CoverageTest(unittest.TestCase):
    def test_accepted_error_rate_and_abstention_track_alpha(self):
        alpha = 0.1
        cal_prompts = list(range(0, 700))
        test_prompts = list(range(10_000, 10_700))

        # hit_prob's miss rate (5%) must stay below alpha (10%) for a selective threshold to exist at all --
        # otherwise no candidate ever clears 1 - alpha coverage and the calibration saturates to "abstain always"
        # (qhat pinned at its max), the same way CalibratedTaskModel's qhat saturates to +inf when the true class
        # is missing from the candidate set more often than alpha tolerates.
        gen = make_generate(hit_prob=0.95)
        model = CalibratedGenerator(gen, score, alpha=alpha, k=5, seed=1).calibrate(cal_prompts, is_correct)

        served = [model.serve(p) for p in test_prompts]
        accepted = [(p, c) for p, c in zip(test_prompts, served) if c is not ABSTAIN]
        abstain_rate = 1.0 - len(accepted) / len(test_prompts)

        self.assertGreater(len(accepted), 0)
        error_rate = np.mean([not is_correct(p, c) for p, c in accepted])
        # finite-sample slack, mirroring task_calibrate_test.py's coverage tolerance
        self.assertLessEqual(error_rate, alpha + 0.08)
        # abstention is doing real, non-degenerate work: it neither accepts nor rejects everything
        self.assertGreater(abstain_rate, 0.0)
        self.assertLess(abstain_rate, 1.0)

    def test_abstention_rate_decreases_as_alpha_relaxes(self):
        cal_prompts = list(range(0, 700))
        test_prompts = list(range(20_000, 20_700))
        gen = make_generate(hit_prob=0.7)

        tight = CalibratedGenerator(gen, score, alpha=0.05, k=5, seed=2).calibrate(cal_prompts, is_correct)
        loose = CalibratedGenerator(gen, score, alpha=0.4, k=5, seed=2).calibrate(cal_prompts, is_correct)

        tight_rate = tight.abstention_rate(test_prompts)
        loose_rate = loose.abstention_rate(test_prompts)
        # a larger alpha tolerates more risk, so it should abstain no more than a stricter alpha
        self.assertLessEqual(loose_rate, tight_rate + 1e-9)


class CascadeIntegrationTest(unittest.TestCase):
    def test_abstention_escalates_through_cascade(self):
        alpha = 0.1
        cal_prompts = list(range(0, 500))
        gen = make_generate(hit_prob=0.5, decoy_spread=8)  # frequent misses -> frequent abstention
        model = CalibratedGenerator(gen, score, alpha=alpha, k=5, seed=3).calibrate(cal_prompts, is_correct)

        def teacher(prompts):
            return [(n, 2 * n) for n in prompts]

        casc = Cascade(model, teacher)

        probe_prompts = list(range(30_000, 30_300))
        abstained_prompt = next((p for p in probe_prompts if model.decide(p) is ABSTAIN), None)
        self.assertIsNotNone(abstained_prompt, "test setup should produce at least one abstention")

        result = casc(abstained_prompt)  # must escalate, not raise
        self.assertEqual(result, (abstained_prompt, 2 * abstained_prompt))
        self.assertEqual(casc.stats.n_escalated, 1)

        # serving a batch mixes accepted and escalated outcomes without error, and every escalation
        # is answered correctly by the (perfect) teacher
        results = casc.serve(probe_prompts)
        self.assertEqual(len(results), len(probe_prompts))
        self.assertGreater(casc.stats.n_escalated, 0)
        self.assertTrue(all(is_correct(p, r) for p, r in zip(probe_prompts, results)))


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_outcome(self):
        cal_prompts = list(range(0, 400))
        gen = make_generate(hit_prob=0.75)

        def build():
            return CalibratedGenerator(gen, score, alpha=0.1, k=5, seed=42).calibrate(cal_prompts, is_correct)

        model_a = build()
        model_b = build()

        self.assertAlmostEqual(model_a.qhat, model_b.qhat, places=12)

        for prompt in range(40_000, 40_050):
            out_a = model_a.serve(prompt)
            out_b = model_b.serve(prompt)
            self.assertEqual(out_a, out_b)

        # repeated calls on the same instance/prompt are also stable
        prompt = 40_100
        first = model_a.serve(prompt)
        second = model_a.serve(prompt)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
