"""solve(): the closed loop — teacher labels the dataset, student trains, calibrated cascade deploys."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _route(ticket):
    """The 'rigid code': a rule-based ticket router (record -> queue)."""
    if ticket["amount"] > 500 and ticket["kind"] == "refund":
        return "finance-escalation"
    if ticket["kind"] in ("refund", "billing"):
        return "billing"
    return "support"


def _tickets(n, seed=0):
    rng = np.random.RandomState(seed)
    kinds = ["refund", "billing", "question", "bug"]
    return [
        {
            "kind": kinds[rng.randint(0, 4)],
            "amount": float(rng.gamma(2.0, 150.0)),
            "region": ["us", "eu"][rng.randint(0, 2)],
        }
        for _ in range(n)
    ]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class SolveTest(unittest.TestCase):
    def test_closed_loop_replaces_rigid_code(self):
        from mixle.task import solve

        sol = solve(_route, _tickets(400), alpha=0.1, seed=0, epochs=300)

        # verification happened on held-out data the student never trained on
        self.assertGreater(sol.holdout_agreement, 0.8)
        self.assertTrue(sol.promoted)

        # the deployed callable is a drop-in for the original function and NEVER disagrees
        # on confident answers... escalations go to the real router, so every answer is safe.
        fresh = _tickets(200, seed=1)
        for t in fresh:
            got = sol(t)
            local = sol.cascade.model.decide(t)
            if local is not None:  # answered locally
                self.assertEqual(got, local)
            else:  # escalated -> exact teacher answer
                self.assertEqual(got, _route(t))
        rep = sol.report()
        self.assertEqual(rep["requests"], len(fresh))
        self.assertGreaterEqual(rep["live_escalated"], 0)

    def test_improve_folds_harvested_labels_with_anti_regression(self):
        from mixle.task import solve

        sol = solve(_route, _tickets(300), alpha=0.15, seed=0, epochs=200)
        base_agree = sol.holdout_agreement
        for t in _tickets(150, seed=2):
            sol(t)
        if sol.cascade.stats.escalated_labels:  # improve() only acts when something was harvested
            sol.improve()
        # anti-regression invariant: agreement never got worse, whatever improve() decided
        self.assertGreaterEqual(sol.holdout_agreement + 1e-12, base_agree)

    def test_target_agreement_gate_falls_back_to_teacher(self):
        from mixle.task import solve

        # an impossible target -> not promoted -> the callable IS the teacher (honest failure)
        sol = solve(_route, _tickets(80), target_agreement=1.01, seed=0, epochs=50)
        self.assertFalse(sol.promoted)
        t = {"kind": "refund", "amount": 900.0, "region": "us"}
        self.assertEqual(sol(t), _route(t))

    def test_ood_gate_escalates_novel_inputs(self):
        from mixle.task import ESCALATE, solve

        sol = solve(_route, _tickets(300), alpha=0.15, ood=0.05, seed=0, epochs=200)
        self.assertIsNotNone(sol.cascade.model.density_gate)
        # a wildly out-of-distribution record must escalate — and hence get the TEACHER's exact answer —
        # regardless of how confident the softmax looks.
        alien = {"kind": "zzz-never-seen", "amount": 1.0e9, "region": "??", "extra": "fields" * 50}
        self.assertIs(sol.cascade.model.decide(alien), ESCALATE)
        self.assertEqual(sol(alien), _route(alien))

    def test_propose_auto_tunes_the_recipe(self):
        from mixle.task import solve

        sol = solve(_route, _tickets(240), propose="auto", propose_budget=4, seed=0)
        # the tuned recipe was recorded (so improve() re-distills with it) and the solution verifies
        self.assertIn("dim", sol.distill_kw)
        self.assertIn("epochs", sol.distill_kw)
        self.assertGreater(sol.holdout_agreement, 0.7)

    def test_synthesize_creates_teacher_labeled_training_data(self):
        from mixle.task import solve

        real = _tickets(60)  # scarce
        sol = solve(_route, real, synthesize=150, ood=None, seed=0, epochs=200)
        rep = sol.report()
        self.assertGreater(rep["synthesized_inputs"], 100)  # the training set materially grew
        self.assertEqual(len(sol.train_inputs), len(sol.train_labels))
        # every synthetic label is the TEACHER's answer on that exact synthetic input (labels stay real)
        n_real_train = len(sol.train_inputs) - sol.synthesized
        for x, y in zip(sol.train_inputs[n_real_train:], sol.train_labels[n_real_train:]):
            self.assertEqual(y, _route(x))
        self.assertGreater(sol.holdout_agreement, 0.7)

    def test_synthesize_rejects_text_inputs(self):
        from mixle.task import solve

        texts = [f"hello {i}" for i in range(20)]
        with self.assertRaises(ValueError):
            solve(lambda s: "x", texts, synthesize=10, seed=0, epochs=10)

    def test_save_load_reconstitutes_the_serving_cascade(self):
        import tempfile

        from mixle.task import Solution, solve

        sol = solve(_route, _tickets(300), alpha=0.15, seed=0, epochs=200)
        fresh = _tickets(60, seed=3)
        want = [sol(t) for t in fresh]
        with tempfile.TemporaryDirectory() as d:
            path = sol.save(d + "/router")
            served = Solution.load(path, _route)
            self.assertEqual(served.kind, "record")
            got = [served(t) for t in fresh]
        self.assertEqual(got, want)  # identical serving behavior in a fresh process
        with self.assertRaises(RuntimeError):
            served.improve()  # loaded artifacts serve + harvest; improving needs the original data

    def test_text_path_and_input_sniffing(self):
        from mixle.task import solve

        def lang(s):  # rigid text rule
            return "greeting" if any(w in s for w in ("hi", "hello", "hey")) else "other"

        texts = [f"hi there {i}" for i in range(30)] + [f"invoice number {i}" for i in range(30)]
        sol = solve(lang, texts, seed=0, epochs=200)
        self.assertEqual(sol.kind, "text")
        self.assertIn(sol("hello friend"), ("greeting", "other"))


if __name__ == "__main__":
    unittest.main()
