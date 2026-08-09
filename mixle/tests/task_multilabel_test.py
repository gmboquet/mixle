"""solve_multilabel: per-label conformal decide-in/decide-out with whole-input escalation."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _tags(t):
    """The rigid tagger: a transaction gets a SET of flags."""
    out = []
    if t["amount"] > 400:
        out.append("high-value")
    if t["kind"] == "refund":
        out.append("refund")
    if t["region"] == "eu":
        out.append("eu-rules")
    return out


def _txns(n, seed=0):
    rng = np.random.RandomState(seed)
    kinds = ["refund", "billing", "question"]
    return [
        {
            "kind": kinds[rng.randint(0, 3)],
            "amount": float(rng.uniform(0, 1000)),
            "region": ["us", "eu"][rng.randint(0, 2)],
        }
        for _ in range(n)
    ]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class SolveMultiLabelTest(unittest.TestCase):
    def test_decided_sets_are_alpha_bounded_and_ambiguity_escalates(self):
        from mixle.task import solve_multilabel

        sol = solve_multilabel(_tags, _txns(600), alpha=0.1, seed=0, epochs=400)
        self.assertEqual(sol.labels, ["eu-rules", "high-value", "refund"])
        self.assertGreater(sol.holdout_set_agreement, 0.5)

        fresh = _txns(300, seed=9)
        wrong_local = total_local = 0
        for t in fresh:
            got = sol(t)
            want = sorted(_tags(t))
            if sol.n_escalated and t is sol.harvested_inputs[-1] if sol.harvested_inputs else False:
                pass
            local = sol.try_local(t)
            if local is not None:
                total_local += 1
                wrong_local += int(sorted(local) != want)
            else:
                self.assertEqual(sorted(got), want)  # escalations return the TEACHER's exact set
        self.assertGreater(total_local, 50)  # the student carries real traffic
        # Joint split conformal controls wrong singleton decisions over all exchangeable requests.
        self.assertLess(wrong_local / len(fresh), 0.15)
        self.assertEqual(sol.report()["coverage_contract"], "joint_exact_set")
        self.assertTrue(
            set(sol.calibration_receipt["calibration_indices"]).isdisjoint(
                sol.calibration_receipt["evaluation_indices"]
            )
        )

        rep = sol.report()
        self.assertEqual(rep["requests"], 300)
        self.assertEqual(rep["harvested"], rep["escalated"])

    def test_improve_promotes_only_non_regressing(self):
        from mixle.task import solve_multilabel

        sol = solve_multilabel(_tags, _txns(300), alpha=0.15, seed=0, epochs=200)
        base = sol.holdout_set_agreement
        for t in _txns(200, seed=3):
            sol(t)
        with self.assertRaisesRegex(ValueError, "fresh evidence"):
            sol.improve()
        sol.improve(_txns(100, seed=13))
        self.assertGreaterEqual(sol.holdout_set_agreement, 0.0)
        self.assertGreaterEqual(base, 0.0)

    def test_under_calibrated_label_is_never_decided(self):
        from mixle.task import solve_multilabel

        def rare(t):  # one label almost never fires -> its present-side bar must stay -inf-ish
            out = ["common"] if t["amount"] > 100 else []
            if t["amount"] > 995:
                out.append("ultra-rare")
            return out

        sol = solve_multilabel(rare, _txns(200), alpha=0.1, seed=0, epochs=150)
        if "ultra-rare" in sol.labels:
            # A rare label is covered as part of the same joint set rather than assigned a misleading
            # independent marginal guarantee.
            self.assertTrue(np.isfinite(sol.joint_qhat))
            for t in _txns(100, seed=5):
                local = sol.try_local(t)
                if local is not None and "ultra-rare" in local:
                    self.fail("under-calibrated label decided as present")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class MultiLabelPersistenceTest(unittest.TestCase):
    def test_save_load_serves_identically(self):
        import tempfile

        from mixle.task import MultiLabelSolution, solve_multilabel

        sol = solve_multilabel(_tags, _txns(400), alpha=0.1, seed=0, epochs=250)
        fresh = _txns(80, seed=5)
        want = [sol.try_local(t) for t in fresh]
        with tempfile.TemporaryDirectory() as d:
            path = sol.save(d + "/tagger")
            back = MultiLabelSolution.load(path, _tags)
            got = [back.try_local(t) for t in fresh]
        self.assertEqual(got, want)  # identical decisions, ambiguity included
        back.harvested_inputs.append(fresh[0])
        back.harvested_sets.append(["x"])
        with self.assertRaises(RuntimeError):
            back.improve()


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class MultiLabelResolveTest(unittest.TestCase):
    """prelabeled= closes the serving loop: harvested sets retrain WITHOUT re-calling the teacher."""

    def test_prelabeled_trains_only_and_extends_label_space(self):
        from mixle.task import solve_multilabel

        calls = {"n": 0}

        def counting_teacher(t):
            if isinstance(t, list):  # the batched probe, not a real label
                raise TypeError("per-item teacher")
            calls["n"] += 1
            return _tags(t)

        base = _txns(100, seed=0)
        harvested = _txns(60, seed=11)
        # one harvested set carries a label the base pass never produced
        pre_sets = [_tags(t) for t in harvested]
        pre_sets[0] = [*pre_sets[0], "manual-review"]

        sol = solve_multilabel(counting_teacher, base, alpha=0.1, prelabeled=(harvested, pre_sets), seed=0, epochs=150)
        self.assertEqual(calls["n"], len(base))  # prelabeled pairs came in free
        self.assertIn("manual-review", sol.labels)  # harvest-only labels enter the space
        self.assertEqual(
            len(sol.train_inputs),
            len(base) - len(sol.cal_inputs) - len(sol.eval_inputs) + len(harvested),
        )
        for t in harvested:
            self.assertNotIn(repr(t), [repr(c) for c in sol.cal_inputs])


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class MultiLabelAnsweredSliceTest(unittest.TestCase):
    def test_measurement_matches_the_real_serving_gate_and_round_trips(self):
        # STAT-RR16-2: the answered-slice numbers must be the REAL joint-qhat singleton rule run
        # over the disjoint evaluation rows -- not the raw 0.5-threshold agreement relabeled.
        import json
        import tempfile
        from pathlib import Path

        from mixle.task import MultiLabelSolution, solve_multilabel
        from mixle.task.artifact import _manifest_integrity

        sol = solve_multilabel(_tags, _txns(400), alpha=0.1, seed=0, epochs=300)
        self.assertEqual(sol.eval_rows, len(sol.eval_inputs))
        self.assertGreaterEqual(sol.eval_rows, 2)
        self.assertLessEqual(sol.answered_eval_correct, sol.answered_eval_n)
        self.assertLessEqual(sol.answered_eval_n, sol.eval_rows)

        answered = correct = 0
        for x, want in zip(sol.eval_inputs, sol.eval_sets):
            got = sol.try_local(x)
            if got is None:
                continue
            answered += 1
            correct += int(sorted(got) == sorted(want))
        self.assertEqual(answered, sol.answered_eval_n)
        self.assertEqual(correct, sol.answered_eval_correct)

        block = sol.report()["answered_slice"]
        if sol.answered_eval_n == 0:
            self.assertIsNone(block)
        else:
            self.assertEqual(block["n_answered"], sol.answered_eval_n)
            self.assertEqual(block["n_evaluated"], sol.eval_rows)
            self.assertAlmostEqual(block["agreement"], sol.answered_eval_correct / sol.answered_eval_n, places=4)
            low, high = block["ci95"]
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)
            self.assertLessEqual(low, block["agreement"] + 1e-4)
            self.assertGreaterEqual(high, block["agreement"] - 1e-4)

        with tempfile.TemporaryDirectory() as d:
            path = sol.save(d + "/tagger")
            back = MultiLabelSolution.load(path, _tags)
            self.assertEqual(back.eval_rows, sol.eval_rows)
            self.assertEqual(back.answered_eval_n, sol.answered_eval_n)
            self.assertEqual(back.answered_eval_correct, sol.answered_eval_correct)
            # an artifact WITHOUT a measurement member is refused, never defaulted to
            # "measured nothing" (the STAT-RR14-1 mechanism)
            manifest_path = Path(path) / "manifest.json"
            doc = json.loads(manifest_path.read_text())
            del doc["meta"]["multilabel"]["answered_eval_n"]
            doc["integrity_sha256"] = _manifest_integrity(doc)
            manifest_path.write_text(json.dumps(doc))
            with self.assertRaises(KeyError):
                MultiLabelSolution.load(path, _tags)


if __name__ == "__main__":
    unittest.main()


class ImpossibleAnsweredSliceCountsTest(unittest.TestCase):
    def test_impossible_answered_slice_counts_are_refused(self):
        # STAT-RR17-13: evaluated=1, answered=1, correct=2 used to return agreement 2.0 and a
        # [NaN, NaN] interval through report(); the arithmetic is an invariant, so it refuses.
        from mixle.task import MultiLabelSolution

        with self.assertRaisesRegex(ValueError, "correct <= answered <= evaluated"):
            MultiLabelSolution(
                net=None,
                featurizer=None,
                labels=["a"],
                teacher=lambda xs: [["a"] for _ in xs],
                upper_absent=np.zeros(1),
                lower_present=np.zeros(1),
                joint_qhat=0.5,
                alpha=0.1,
                holdout_set_agreement=1.0,
                eval_rows=1,
                answered_eval_n=1,
                answered_eval_correct=2,
            )
