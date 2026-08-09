"""Calibrated generator: held-out selective-risk certification for generation.

The accepted-error certificate must be based on a fixed outcome and independent
certification split; abstention (ABSTAIN) must compose with Cascade as an
escalation signal; and serving must be reproducible given a seed.
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


class SelectiveRiskTest(unittest.TestCase):
    def test_accepted_error_rate_and_abstention_track_alpha(self):
        alpha = 0.1
        cal_prompts = list(range(0, 700))
        test_prompts = list(range(10_000, 10_700))

        # hit_prob's miss rate (5%) is below the target accepted error (10%), so
        # either a selective threshold or even accept-all may certify.
        gen = make_generate(hit_prob=0.95)
        model = CalibratedGenerator(gen, score, alpha=alpha, k=5, seed=1).calibrate(cal_prompts, is_correct)

        served = [model.serve(p) for p in test_prompts]
        accepted = [(p, c) for p, c in zip(test_prompts, served) if c is not ABSTAIN]
        abstain_rate = 1.0 - len(accepted) / len(test_prompts)

        self.assertGreater(len(accepted), 0)
        error_rate = np.mean([not is_correct(p, c) for p, c in accepted])
        # Realized future risk is a diagnostic; the actual release claim is the
        # independent exact-binomial certificate stored in risk_receipt.
        self.assertLessEqual(error_rate, alpha + 0.08)
        self.assertGreaterEqual(abstain_rate, 0.0)
        self.assertLessEqual(abstain_rate, 1.0)
        self.assertEqual(model.risk_receipt["method"], "split-selective-risk/clopper-pearson-bonferroni/v1")
        self.assertLessEqual(model.risk_receipt["error_upper"], alpha)
        self.assertEqual(model.risk_receipt["proposal_count"] + model.risk_receipt["certification_count"], 700)

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

    def test_uncertifiable_generator_abstains_instead_of_claiming_candidate_coverage(self):
        def always_wrong(prompt, k, rng=None):
            return [(prompt, prompt + i + 1) for i in range(k)]

        model = CalibratedGenerator(always_wrong, lambda candidate: -candidate[1], alpha=0.1, k=3).calibrate(
            list(range(100)),
            lambda prompt, candidate: candidate[1] == prompt,
        )
        self.assertTrue(np.isposinf(model.qhat))
        self.assertIsNone(model.risk_receipt["error_upper"])
        self.assertIs(model.serve(1000), ABSTAIN)


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

        escalated = [(p, r) for p, r in zip(probe_prompts, results) if model.decide(p) is ABSTAIN]
        accepted = [(p, r) for p, r in zip(probe_prompts, results) if model.decide(p) is not ABSTAIN]
        self.assertTrue(escalated and accepted, "the probe should exercise both cascade branches")
        # The teacher is perfect, so an escalated row is never wrong.
        self.assertTrue(all(is_correct(p, r) for p, r in escalated))
        # An accepted row is bounded by the certified risk, NOT guaranteed correct. Asserting zero
        # errors here held only by luck of which half certified: alpha=0.1 permits errors among
        # accepted rows by construction, and demanding perfection would make this test fail whenever
        # a legitimate change moved the threshold at all.
        errors = sum(1 for p, r in accepted if not is_correct(p, r))
        self.assertLessEqual(errors / len(accepted), alpha)


class DrawDispatchTest(unittest.TestCase):
    """_draw's generate(prompt, k, rng=...) / generate(prompt, k) dispatch."""

    def test_a_bug_inside_generate_is_not_masked_by_a_retry_without_rng(self):
        # generate(prompt, k, rng=None) accepts rng, so it must be called with it exactly once; a
        # TypeError from inside its own body must propagate, not be swallowed and silently retried
        # as generate(prompt, k) -- which would draw an entirely separate, uncontrolled-seed batch
        # (e.g. duplicating a real generator/LLM call).
        calls = []

        def buggy_generate(prompt, k, rng=None):
            calls.append((prompt, k, rng is not None))
            return None + k  # an internal bug unrelated to whether rng is accepted

        model = CalibratedGenerator(buggy_generate, score, k=3, seed=0)
        with self.assertRaises(TypeError):
            model._draw(5, seed=1)
        self.assertEqual(calls, [(5, 3, True)])  # called once, with rng -- never retried without it

    def test_legacy_generate_without_rng_support_falls_back_correctly(self):
        def legacy_generate(prompt, k):
            return [(prompt, prompt + i) for i in range(k)]

        model = CalibratedGenerator(legacy_generate, score, k=3, seed=0)
        cands = model._draw(5, seed=1)
        self.assertEqual(cands, [(5, 5), (5, 6), (5, 7)])


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


class CertificateCoversServedPolicyTest(unittest.TestCase):
    """STAT-RR17-07: the certificate must cover the policy that serves.

    Calibration used to seed candidate draws by (row_index, prompt) while serving seeds by
    prompt alone: on a repeated-prompt population, per-row randomness measured 1/150 errors and
    certified an 0.0366 upper bound while the served (prompt-only, deterministic-per-prompt)
    policy answered 1000/1000 wrong. With one schedule, certification can only record what
    serving would do, so that split is structurally impossible.
    """

    @staticmethod
    def _seed_flipped_generator():
        # candidate correctness depends ONLY on the rng the policy supplies: a per-row schedule
        # makes calibration rows disagree with serving on the same prompt; prompt-only cannot
        import numpy as np

        def generate(prompt, k, rng=None):
            if rng is None:
                rng = np.random.default_rng()
            flip = float(rng.random())
            return [(prompt, "right" if flip < 0.5 else "wrong", i) for i in range(k)]

        def score(candidate):
            return 1.0

        return generate, score

    def test_served_answers_replay_certification_on_repeated_prompts(self):
        from mixle.task.calibrated_generator import ABSTAIN, CalibratedGenerator

        generate, score = self._seed_flipped_generator()
        gate = CalibratedGenerator(generate, score, alpha=0.1, k=3, seed=7)
        prompts = ["the same prompt"] * 150
        calibration_verdicts: list[bool] = []

        def is_correct(prompt, candidate):
            ok = candidate[1] == "right"
            calibration_verdicts.append(ok)
            return ok

        gate.calibrate(prompts, is_correct)
        served = gate.serve("the same prompt")
        if gate.risk_receipt["error_upper"] is not None and served is not ABSTAIN:
            # certified AND answering: the served answer must be the exact decision
            # certification measured -- certify-low-then-serve-high is impossible
            self.assertTrue(all(calibration_verdicts))
            self.assertEqual(served[1], "right")
        else:
            # not certified (or abstaining): serving must abstain rather than answer uncovered
            self.assertIs(served, ABSTAIN)
        # the receipt discloses the schedule and the duplicate-collapsed sample size
        self.assertEqual(gate.risk_receipt["seed_schedule"], "prompt-only (identical to serving)")
        self.assertEqual(gate.risk_receipt["unique_prompt_count"], 1)

    def test_mismatched_calibrate_seed_is_refused(self):
        from mixle.task.calibrated_generator import CalibratedGenerator

        generate, score = self._seed_flipped_generator()
        gate = CalibratedGenerator(generate, score, alpha=0.1, k=3, seed=7)
        with self.assertRaisesRegex(ValueError, "served policy"):
            gate.calibrate(["a", "b", "c", "d"], lambda p, c: True, seed=8)

    def test_serve_seed_override_is_refused_once_certified(self):
        from mixle.task.calibrated_generator import CalibratedGenerator

        generate, score = self._seed_flipped_generator()
        gate = CalibratedGenerator(generate, score, alpha=0.5, k=3, seed=7)
        gate.calibrate([f"p{i}" for i in range(40)], lambda p, c: c[1] == "right")
        with self.assertRaisesRegex(ValueError, "certificate covers only the prompt-derived schedule"):
            gate.serve("p0", seed=123)


class DuplicatePromptBoundTest(unittest.TestCase):
    """STAT-RR18-04: the risk bound must agree with the receipt's effective sample size."""

    def test_bound_uses_unique_certification_prompts(self):
        from mixle.task.calibrated_generator import CalibratedGenerator

        def generate(prompt, k, rng=None):
            return [(prompt, "right", i) for i in range(k)]

        def score(candidate):
            return 1.0

        gate = CalibratedGenerator(generate, score, alpha=0.1, k=3, seed=5)
        gate.calibrate(["the one prompt"] * 300, lambda p, c: True)
        receipt = gate.risk_receipt
        # 150 certification rows collapse to ONE unique prompt: the exact binomial bound at
        # 0/1 is 0.975 (the receipt's own stated effective sample), so nothing certifies at
        # alpha=0.1 -- the reviewer's construction certified 0.024 by counting 150 copies
        self.assertEqual(receipt["certification_effective_count"], 1)
        self.assertEqual(receipt["outcome_declaration"], "per-prompt")
        self.assertEqual(receipt["unique_prompt_count"], 1)
        self.assertTrue(receipt["error_upper"] is None or receipt["error_upper"] > 0.9)
        self.assertEqual(receipt["threshold"], "inf")

    def test_traffic_weighted_certification_requires_the_sampling_declaration(self):
        # Pass-19 blocker: on 40%-heavy i.i.d. traffic whose heavy prompt always errs, collapsing
        # duplicates certified error_upper ~0.08 while the SERVED traffic risk measured 0.37-0.41
        # (every trial) -- the collapse silently swapped the estimand to uniform-over-distinct-
        # prompts. sampling='iid-traffic' keeps every row (a duplicate's multiplicity is its
        # traffic weight), so this stream refuses to certify alpha=0.15; the constructed default
        # still certifies its uniform estimand but the receipt now NAMES it as not traffic-
        # weighted.
        import hashlib

        from mixle.task.calibrated_generator import CalibratedGenerator

        def generate(prompt, k, rng=None):
            if rng is None:
                rng = np.random.default_rng()
            cands = [(prompt, 2 * prompt)]
            while len(cands) < k:
                offset = int(rng.integers(-6, 7))
                if offset:
                    cands.append((prompt, 2 * prompt + offset))
            rng.shuffle(cands)
            return cands

        def score_local(candidate):
            n, guess = candidate
            jitter = int.from_bytes(hashlib.sha256(repr(candidate).encode()).digest()[:8], "big") / 2**64
            return -abs(guess - 2 * n) + 0.5 * jitter

        heavy = 7

        def oracle(prompt, candidate):
            n, guess = candidate
            return False if prompt == heavy else guess == 2 * n

        rng = np.random.RandomState(0)
        cal = [heavy if rng.rand() < 0.4 else int(rng.randint(100_000, 104_000)) for _ in range(400)]

        honest = CalibratedGenerator(generate, score_local, alpha=0.15, k=5, seed=0)
        honest.calibrate(cal, oracle, sampling="iid-traffic")
        self.assertEqual(honest.risk_receipt["threshold"], "inf")  # 40% traffic risk: no certificate
        self.assertEqual(honest.risk_receipt["sampling_declaration"], "iid-traffic")
        self.assertIn("traffic-weighted", honest.risk_receipt["certified_estimand"])
        self.assertEqual(honest.risk_receipt["certification_effective_count"], 200)

        disclosed = CalibratedGenerator(generate, score_local, alpha=0.15, k=5, seed=0)
        disclosed.calibrate(cal, oracle)
        self.assertIn("NOT traffic-weighted", disclosed.risk_receipt["certified_estimand"])
        self.assertEqual(disclosed.risk_receipt["sampling_declaration"], "constructed")
        with self.assertRaisesRegex(ValueError, "sampling"):
            disclosed.calibrate(cal, oracle, sampling="whatever")

    def test_disagreeing_duplicate_verdicts_are_refused(self):
        from mixle.task.calibrated_generator import CalibratedGenerator

        def generate(prompt, k, rng=None):
            return [(prompt, "right", i) for i in range(k)]

        def score(candidate):
            return 1.0

        gate = CalibratedGenerator(generate, score, alpha=0.5, k=3, seed=5)
        flip = {"n": 0}

        def inconsistent(prompt, candidate):
            flip["n"] += 1
            return flip["n"] % 2 == 0

        with self.assertRaisesRegex(ValueError, "duplicate certification"):
            gate.calibrate(["p"] * 40, inconsistent)
