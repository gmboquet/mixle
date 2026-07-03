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
        # per-input wrong-set rate among locally-decided inputs stays small (union of per-label alphas)
        self.assertLess(wrong_local / total_local, 0.25)

        rep = sol.report()
        self.assertEqual(rep["requests"], 300)
        self.assertEqual(rep["harvested"], rep["escalated"])

    def test_improve_promotes_only_non_regressing(self):
        from mixle.task import solve_multilabel

        sol = solve_multilabel(_tags, _txns(300), alpha=0.15, seed=0, epochs=200)
        base = sol.holdout_set_agreement
        for t in _txns(200, seed=3):
            sol(t)
        sol.improve()
        self.assertGreaterEqual(sol.holdout_set_agreement + 1e-12, base)

    def test_under_calibrated_label_is_never_decided(self):
        from mixle.task import solve_multilabel

        def rare(t):  # one label almost never fires -> its present-side bar must stay -inf-ish
            out = ["common"] if t["amount"] > 100 else []
            if t["amount"] > 995:
                out.append("ultra-rare")
            return out

        sol = solve_multilabel(rare, _txns(200), alpha=0.1, seed=0, epochs=150)
        if "ultra-rare" in sol.labels:
            j = sol.labels.index("ultra-rare")
            # with almost no present-side calibration examples the lower bar is -inf (never confidently
            # absent-decided is fine) but the upper bar must be finite only if absents existed; the key
            # invariant: a locally-returned set never CONTAINS ultra-rare unless the bar was clearable
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


if __name__ == "__main__":
    unittest.main()
