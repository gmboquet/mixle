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
        self.assertEqual(sol.calibration_evidence, "solve-split")
        for t in _tickets(150, seed=2):
            sol(t)
        promoted = False
        if sol.cascade.stats.escalated_labels:  # improve() only acts when something was harvested
            promoted = sol.improve()
        # anti-regression invariant: agreement never got worse, whatever improve() decided
        self.assertGreaterEqual(sol.holdout_agreement + 1e-12, base_agree)
        # STAT-RR12-1: a no-argument promotion recalibrated on rows that gated the harvest the
        # candidate trained on; the artifact must SAY its threshold is in the reused regime
        expected = "reused-after-adaptive-harvest" if promoted else "solve-split"
        self.assertEqual(sol.calibration_evidence, expected)
        self.assertEqual(sol.report()["calibration_evidence"], expected)

    def test_improve_with_fresh_evidence_keeps_the_certified_regime(self):
        from mixle.task import solve

        sol = solve(_route, _tickets(300), alpha=0.15, seed=0, epochs=200)
        for t in _tickets(150, seed=2):
            sol(t)
        # deterministic given the seeds: this fixture escalates (the sibling test relies on the
        # same behavior), and the required zero-skip CI receipt forbids runtime skips here
        self.assertTrue(sol.cascade.stats.escalated_labels)
        promoted = sol.improve(evidence_inputs=_tickets(60, seed=9))
        if promoted:
            self.assertEqual(sol.calibration_evidence, "fresh-evidence")
            self.assertEqual(sol.report()["calibration_evidence"], "fresh-evidence")

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

    def test_save_load_round_trips_the_honesty_ledgers(self):
        # STAT-RR13-1/2: an artifact whose threshold was recalibrated on reused rows reloaded as
        # "solve-split" (falsely certified), and selection_uses reset to 0, flipping
        # selection_evidence_is_single_use from False back to True -- a spent receipt reloaded
        # as fresh evidence.
        import tempfile

        from mixle.task import Solution, solve

        sol = solve(_route, _tickets(300), alpha=0.15, seed=0, epochs=200)
        sol.selection_uses = 4
        sol.calibration_evidence = "reused-after-adaptive-harvest"
        self.assertFalse(sol.selection_evidence_is_single_use)
        with tempfile.TemporaryDirectory() as d:
            back = Solution.load(sol.save(d + "/router"), _route)
        self.assertEqual(back.calibration_evidence, "reused-after-adaptive-harvest")
        self.assertEqual(back.selection_uses, 4)
        self.assertFalse(back.selection_evidence_is_single_use)
        self.assertEqual(back.report()["calibration_evidence"], "reused-after-adaptive-harvest")

    def test_save_load_preserves_rejected_promotion(self):
        import tempfile

        from mixle.task import Solution, solve

        sol = solve(_route, _tickets(80), target_agreement=1.01, ood=None, seed=0, epochs=30)
        self.assertFalse(sol.promoted)
        with tempfile.TemporaryDirectory() as d:
            served = Solution.load(sol.save(d), _route)
        self.assertFalse(served.promoted)
        ticket = {"kind": "refund", "amount": 900.0, "region": "us"}
        self.assertEqual(served(ticket), _route(ticket))

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

    def test_answer_rows_keep_their_shape(self):
        import json
        import tempfile

        from mixle.task import load_harvested

        with tempfile.TemporaryDirectory() as d:
            # what the mlops /v1/solutions/{name}/feedback endpoint accumulates: shape-typed answers
            rows = [
                {"input": {"kind": "bug", "amount": 5.0}, "answer": 2520.0},
                {"input": "flag this one", "answer": ["high-value", "eu-rules"]},
                {"input": ["refund", 30.0], "answer": {"queue": "billing", "priority": 12.5}},
            ]
            p = d + "/harvested.jsonl"
            with open(p, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            ins, answers = load_harvested(p)
            self.assertEqual(answers[0], 2520.0)  # regression: a number, NOT str-coerced
            self.assertEqual(answers[1], ["high-value", "eu-rules"])  # multilabel: the set
            self.assertEqual(answers[2], {"queue": "billing", "priority": 12.5})  # structured: the dict
            self.assertIsInstance(ins[2], tuple)  # record inputs still coerce to tuples


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

    def test_improve_respects_device_budget(self):
        """Regression: improve() used to refit with the generic, unconstrained default student
        (a full-size torch MLP) no matter what device budget solve() originally searched under,
        so a promoted post-improve() artifact could blow way past the budget while report() kept
        reporting the stale pre-improve() footprint as if nothing had changed."""
        from mixle.task import DeviceSpec, footprint, solve
        from mixle.task.edge import EdgeSpace

        space = EdgeSpace(families=("mlp",), dim_choices=(32,), hidden_range=(2, 3), epochs_range=(30, 60))
        sol = solve(
            _route,
            _tickets(300),
            alpha=0.15,
            device=DeviceSpec(max_bytes=6000),
            device_space=space,
            propose_budget=4,
            seed=0,
        )
        self.assertTrue(sol.edge.feasible)
        self.assertLessEqual(sol.edge.footprint.bytes, 6000)

        # Fuel improve() with a DETERMINISTIC harvest instead of serving-and-hoping: whether live
        # traffic escalates depends on the platform's float arithmetic (ubuntu CPU torch trains a
        # student confident on all 500 tickets; arm64 escalates dozens), and this test's subject
        # is the device-budget gate, not escalation propensity. The stats lists are the documented
        # harvest channel Solution.improve() reads via cascade.harvested().
        harvest = _tickets(60, seed=7)
        sol.cascade.stats.escalated_texts.extend(harvest)
        sol.cascade.stats.escalated_labels.extend(_route(t) for t in harvest)

        promoted = sol.improve()
        # the regression under test is the BUDGET, not promotion propensity: whether this exact
        # candidate beats the incumbent on the selection rows varies with platform arithmetic,
        # but the deployed artifact must respect the original device budget either way -- and
        # when a promotion did happen, it must have gone through the budget-constrained search
        fp = footprint(sol.cascade.model.task)
        self.assertLessEqual(fp.bytes, 6000)  # the DEPLOYED student must still respect the original budget
        if promoted:
            self.assertTrue(sol.edge.feasible)
            self.assertLessEqual(sol.edge.footprint.bytes, 6000)
        # report() must describe what's actually deployed, not a stale pre-improve() snapshot
        self.assertEqual(sol.report()["device"]["bytes"], fp.bytes)
        self.assertTrue(sol.report()["device"]["feasible"])

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
        # exchangeable traffic must NOT alarm: the two-sample exact test holds its nominal size
        # where the old one-sample test against the plug-in baseline false-alarmed at a measured
        # 57-71% under no drift (any baseline estimation error dominates as live traffic grows)
        self.assertFalse(ok["drifted"])
        self.assertIn("escalation_p_value", ok)

        shifted = [
            {"kind": "zzz-" + str(i), "amount": 1.0e8 + i, "region": "??"} for i in range(200)
        ]  # a different world: the gate + ambiguity must push escalation far off baseline
        for t in shifted:
            sol(t)
        bad = sol.health(recent_inputs=shifted)
        self.assertTrue(bad["drifted"])
        self.assertGreater(bad["live_ood_rate"], bad["design_ood_rate"])


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class AnsweredSliceMeasurementTest(unittest.TestCase):
    def test_measurement_matches_the_real_gate_and_survives_the_round_trip(self):
        # STAT-RR16-2 symmetry: classification's answered-slice numbers must be the REAL
        # answer-or-escalate rule run over the selection rows -- not holdout_agreement relabeled.
        import tempfile

        from mixle.task import ESCALATE, Solution, solve

        sol = solve(_route, _tickets(400), alpha=0.1, seed=0, epochs=300)
        self.assertEqual(sol.sel_rows, len(sol.sel_inputs))
        self.assertLessEqual(sol.answered_sel_correct, sol.answered_sel_n)
        self.assertLessEqual(sol.answered_sel_n, sol.sel_rows)

        decisions = sol.cascade.model.batch_decide(list(sol.sel_inputs))
        answered = [(d, y) for d, y in zip(decisions, sol.sel_labels) if d is not ESCALATE]
        self.assertEqual(len(answered), sol.answered_sel_n)
        self.assertEqual(sum(1 for d, y in answered if str(d) == str(y)), sol.answered_sel_correct)

        block = sol.report()["answered_slice"]
        if sol.answered_sel_n == 0:
            self.assertIsNone(block)
        else:
            self.assertEqual(block["n_answered"], sol.answered_sel_n)
            self.assertEqual(block["n_evaluated"], sol.sel_rows)
            self.assertAlmostEqual(block["agreement"], sol.answered_sel_correct / sol.answered_sel_n, places=4)
            low, high = block["ci95"]
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)
            self.assertLessEqual(low, block["agreement"] + 1e-4)
            self.assertGreaterEqual(high, block["agreement"] - 1e-4)

        with tempfile.TemporaryDirectory() as d:
            path = sol.save(d + "/router")
            back = Solution.load(path, _route)
            self.assertEqual(back.sel_rows, sol.sel_rows)
            self.assertEqual(back.answered_sel_n, sol.answered_sel_n)
            self.assertEqual(back.answered_sel_correct, sol.answered_sel_correct)
            # missing-member refusal for these fields is covered registry-wide by
            # task_ledger_lifecycle_test (they ride CLASSIFICATION_LEDGER)


if __name__ == "__main__":
    unittest.main()
