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
            # the artifact answers "is this trustworthy" by itself: the verification record rides along
            ver = (served.cascade.model.task.meta or {})["solve"]["verification"]
            self.assertAlmostEqual(ver["holdout_agreement"], sol.holdout_agreement, places=6)
            self.assertTrue(ver["promoted"])
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


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ServingLoopRoundTripTest(unittest.TestCase):
    """harvested.jsonl (the serving feedback format) -> load_harvested -> solve(prelabeled=) -> deploy."""

    def test_harvested_pairs_feed_the_next_solve(self):
        import json
        import tempfile

        from mixle.task import load_harvested, solve

        with tempfile.TemporaryDirectory() as d:
            # what the mlops /v1/tasks/{name}/feedback endpoint accumulates (dict + list inputs)
            harvested = [
                {"input": {"kind": "refund", "amount": 900.0, "region": "us"}, "label": "finance-escalation"},
                {"input": ["refund", 30.0], "label": "billing"},
            ]
            p = d + "/harvested.jsonl"
            with open(p, "w") as f:
                for row in harvested:
                    f.write(json.dumps(row) + "\n")
            ins, labs = load_harvested(p)
            self.assertEqual(labs, ["finance-escalation", "billing"])
            self.assertIsInstance(ins[1], tuple)  # JSON lists coerce back to the tuple record shape

            pre_in = [{"kind": "refund", "amount": 800.0 + i, "region": "us"} for i in range(20)]
            pre = (pre_in, [_route(x) for x in pre_in])
            sol = solve(_route, _tickets(200), prelabeled=pre, ood=0.05, seed=0, epochs=150)
            self.assertGreaterEqual(len(sol.train_inputs), 150 + 20)  # prelabeled joined training
            for x, y in zip(sol.train_inputs[-20:], sol.train_labels[-20:]):
                self.assertEqual(y, _route(x))  # exact teacher labels, in order
            self.assertGreater(sol.holdout_agreement, 0.7)

            path = sol.deploy("router", root=d)  # the serving layout the mlops routes read
            self.assertTrue((__import__("pathlib").Path(d) / "tasks" / "router" / "manifest.json").exists())
            self.assertIn("tasks/router", path.replace("\\", "/"))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class SolveOnDeviceTest(unittest.TestCase):
    """solve(device=DeviceSpec(...)): 'give me this capability on that device' as one call."""

    def _space(self):
        from mixle.task import EdgeSpace

        return EdgeSpace(
            families=("mlp", "structured"),
            dim_choices=(64, 128),
            hidden_range=(4, 24),
            epochs_range=(30, 90),
            components_range=(1, 2),
            max_its_range=(8, 20),
        )

    def test_solve_under_device_budget(self):
        from mixle.task import DeviceSpec, solve

        sol = solve(
            _route,
            _tickets(300),
            device=DeviceSpec(max_bytes=200_000),
            device_space=self._space(),
            propose_budget=5,
            seed=0,
        )
        self.assertTrue(sol.promoted)
        self.assertIsNotNone(sol.edge)
        self.assertTrue(sol.edge.feasible)
        self.assertLessEqual(sol.edge.footprint.bytes, 200_000)
        rep = sol.report()
        self.assertIn("device", rep)
        self.assertTrue(rep["device"]["feasible"])
        # still a drop-in: confident answers are the local model's, escalations exact teacher answers
        for t in _tickets(60, seed=2):
            got = sol(t)
            local = sol.cascade.model.decide(t)
            self.assertEqual(got, local if local is not None else _route(t))

    def test_torch_free_device_gives_deployable_torch_free_artifact(self):
        import tempfile

        from mixle.task import DeviceSpec, Solution, solve

        sol = solve(
            _route,
            _tickets(300),
            device=DeviceSpec(torch_free=True),
            device_space=self._space(),
            propose_budget=4,
            seed=0,
        )
        self.assertTrue(sol.edge.footprint.torch_free)
        self.assertNotEqual(sol.cascade.model.task.payload, "torch")
        # the artifact round-trips and serves in a fresh Solution
        with tempfile.TemporaryDirectory() as d:
            path = sol.save(d)
            served = Solution.load(path, _route)
        for t in _tickets(30, seed=3):
            self.assertEqual(served(t), sol(t))

    def test_infeasible_budget_demotes_to_teacher(self):
        from mixle.task import DeviceSpec, solve

        sol = solve(
            _route,
            _tickets(200),
            device=DeviceSpec(max_bytes=50),  # 50 bytes: nothing fits
            device_space=self._space(),
            propose_budget=3,
            seed=0,
        )
        self.assertFalse(sol.promoted)
        self.assertFalse(sol.edge.feasible)
        for t in _tickets(20, seed=4):
            self.assertEqual(sol(t), _route(t))  # honest failure: everything routes to the teacher

    def test_device_string_keeps_torch_device_meaning(self):
        from mixle.task import solve

        sol = solve(_route, _tickets(200), device="cpu", seed=0, epochs=120)
        self.assertIsNone(sol.edge)  # the old kwarg path, no edge search
        self.assertGreater(sol.holdout_agreement, 0.7)

    def test_device_conflicts_with_propose_auto(self):
        from mixle.task import DeviceSpec, solve

        with self.assertRaises(ValueError):
            solve(_route, _tickets(100), device=DeviceSpec(max_bytes=1000), propose="auto", seed=0)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class HealthTest(unittest.TestCase):
    def test_in_distribution_traffic_is_healthy_and_shift_alarms(self):
        from mixle.task import solve

        sol = solve(_route, _tickets(400), alpha=0.15, ood=0.05, seed=0, epochs=200)

        for t in _tickets(200, seed=21):  # same world as training
            sol(t)
        ok = sol.health(recent_inputs=_tickets(200, seed=22))
        self.assertGreaterEqual(ok["requests"], 200)

        shifted = [
            {"kind": "zzz-" + str(i), "amount": 1.0e8 + i, "region": "??"} for i in range(200)
        ]  # a different world: the gate + ambiguity must push escalation far off baseline
        for t in shifted:
            sol(t)
        bad = sol.health(recent_inputs=shifted)
        self.assertTrue(bad["drifted"])
        self.assertGreater(bad["live_ood_rate"], bad["design_ood_rate"])


if __name__ == "__main__":
    unittest.main()
