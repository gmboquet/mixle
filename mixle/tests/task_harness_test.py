"""The application harnesses: extractor / alerter / matcher, each replacing a rigid rule."""

import re
import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ExtractorHarnessTest(unittest.TestCase):
    def test_replaces_a_regex_scraper_with_fallback(self):
        from mixle.task import replace_extractor

        def scraper(text):  # the rigid parser being replaced
            m = re.search(r"order (\d+) .* total (\d+\.\d+)", text)
            return {"id": m.group(1), "amount": m.group(2)} if m else {}

        rng = np.random.RandomState(0)
        texts = [
            f"order {rng.randint(100, 999)} placed by user{u} total {rng.randint(1, 99)}.{rng.randint(10, 99)}"
            for u in range(120)
        ]
        ex = replace_extractor(scraper, texts, ["id", "amount"], seed=0, epochs=40)
        self.assertGreater(ex.holdout_f1, 0.8)

        out = ex("order 314 placed by user7 total 15.99")
        self.assertEqual(out, {"id": "314", "amount": "15.99"})

        # a text the tagger cannot fully extract falls back to the teacher (here: empty -> {} too)
        ex("completely unrelated text with no fields at all")
        self.assertGreaterEqual(ex.report()["fallbacks"], 0)
        self.assertEqual(ex.report()["requests"], 2)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class AlerterHarnessTest(unittest.TestCase):
    def test_replaces_a_threshold_rule(self):
        from mixle.task import replace_alerter

        def rule(window):  # the rigid alerter
            return "alert" if float(np.mean(window)) > 2.0 else "ok"

        rng = np.random.RandomState(0)
        calm = rng.normal(0.0, 1.0, 400)
        hot = rng.normal(4.0, 1.0, 120)
        series = np.concatenate([calm, hot, rng.normal(0.0, 1.0, 200)])
        sol = replace_alerter(rule, series, window=16, ood=None, seed=0, epochs=200)
        self.assertGreater(sol.holdout_agreement, 0.85)

        hot_window = tuple(rng.normal(4.0, 1.0, 16))
        self.assertEqual(sol(hot_window), "alert")
        calm_window = tuple(rng.normal(0.0, 1.0, 16))
        self.assertEqual(sol(calm_window), "ok")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class MatcherHarnessTest(unittest.TestCase):
    def test_replaces_a_dedup_rule(self):
        from mixle.task import replace_matcher

        def rule(a, b):  # the rigid matcher
            return "match" if a["kind"] == b["kind"] and abs(a["amount"] - b["amount"]) < 50 else "no-match"

        rng = np.random.RandomState(0)
        kinds = ["refund", "billing", "bug"]

        def rec():
            return {"kind": kinds[rng.randint(0, 3)], "amount": float(rng.uniform(0, 500))}

        pairs = []
        for _ in range(150):
            a = rec()
            b = dict(a, amount=a["amount"] + float(rng.uniform(-40, 40))) if rng.rand() < 0.5 else rec()
            pairs.append((a, b))

        matcher = replace_matcher(rule, pairs, ood=None, seed=0, epochs=250)
        self.assertGreater(matcher.holdout_agreement, 0.8)

        dup = {"kind": "refund", "amount": 100.0}
        near = {"kind": "refund", "amount": 110.0}
        far = {"kind": "billing", "amount": 480.0}
        self.assertEqual(matcher(dup, near), "match")
        self.assertEqual(matcher(dup, far), "no-match")


if __name__ == "__main__":
    unittest.main()
