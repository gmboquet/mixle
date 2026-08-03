"""select_alpha_for_cost: cost-aware threshold selection (workstream B2) -- recommend_route actually
connected to the calibration step, choosing alpha from a CostModel target instead of a fixed default.
"""

import unittest

from mixle.task.economics import CostModel, RoutePlan, recommend_route, select_alpha_for_cost


class _FakeCalibratedModel:
    """A minimal stand-in for CalibratedTaskModel: a mutable alpha, a no-op calibrate(), and an
    escalation_rate() that reads from a known alpha -> p_escalate curve (tighter alpha = more escalation,
    the real conformal relationship) -- so the sweep has a genuine, checkable optimum."""

    def __init__(self, escalation_by_alpha: dict) -> None:
        self.alpha = None
        self.qhat = None
        self._curve = escalation_by_alpha
        self.calibration_batches = []

    def calibrate(self, texts, labels):
        self.calibration_batches.append((list(texts), list(labels)))
        self.qhat = f"qhat@alpha={self.alpha}"  # stands in for a real recalibration
        return self

    def escalation_rate(self, texts):
        return self._curve[self.alpha]


class SelectAlphaForCostTest(unittest.TestCase):
    def test_picks_the_alpha_with_the_lowest_recommend_route_cost(self):
        # tight alpha (0.01) escalates almost everything (expensive frontier calls dominate);
        # loose alpha (0.3) escalates almost nothing but risks quality -- 0.1 is the sweet spot here.
        curve = {0.01: 0.9, 0.05: 0.5, 0.1: 0.15, 0.15: 0.2, 0.2: 0.4, 0.3: 0.7}
        model = _FakeCalibratedModel(curve)
        cost = CostModel(c_frontier=1.0, c_local=0.01, c_label=0.05, train_cost=1.0)

        best_alpha, best_plan, plans = select_alpha_for_cost(
            model,
            ["cal text"],
            ["cal label"],
            ["probe text"],
            cost,
            certification_texts=["cert text"],
            certification_labels=["cert label"],
            volume=10_000,
            n_label=200,
        )

        self.assertEqual(best_alpha, 0.1)
        self.assertIsInstance(best_plan, RoutePlan)
        self.assertEqual(set(plans), {0.01, 0.05, 0.1, 0.15, 0.2, 0.3})
        # every candidate's plan is independently reproducible via recommend_route directly
        for alpha, plan in plans.items():
            expected = recommend_route(cost, volume=10_000, n_label=200, p_escalate=curve[alpha])
            self.assertEqual(plan.total, expected.total)
        # the model is left calibrated at the winning alpha, not the last swept one
        self.assertEqual(model.alpha, 0.1)
        self.assertEqual(model.qhat, "qhat@alpha=0.1")
        self.assertEqual(model.calibration_batches[-1], (["cert text"], ["cert label"]))

    def test_all_plans_beat_frontier_only_when_escalation_is_ever_cheap_enough(self):
        curve = {0.05: 0.5, 0.1: 0.1, 0.2: 0.05}
        model = _FakeCalibratedModel(curve)
        cost = CostModel(c_frontier=1.0, c_local=0.0, c_label=0.01, train_cost=0.0)

        best_alpha, best_plan, _ = select_alpha_for_cost(
            model,
            ["selection"],
            ["label"],
            ["probe"],
            cost,
            certification_texts=["certification"],
            certification_labels=["label"],
            volume=5_000,
            n_label=100,
            alphas=(0.05, 0.1, 0.2),
        )
        self.assertEqual(best_alpha, 0.2)
        self.assertGreater(best_plan.savings_vs_frontier, 0)

    def test_custom_alpha_grid_is_honored(self):
        curve = {0.02: 0.6, 0.4: 0.05}
        model = _FakeCalibratedModel(curve)
        cost = CostModel(c_frontier=1.0, c_local=0.0, c_label=0.0, train_cost=0.0)

        _, _, plans = select_alpha_for_cost(
            model,
            ["selection"],
            ["label"],
            ["probe"],
            cost,
            certification_texts=["certification"],
            certification_labels=["label"],
            volume=1_000,
            n_label=0,
            alphas=(0.02, 0.4),
        )
        self.assertEqual(set(plans), {0.02, 0.4})

    def test_selection_and_certification_rows_cannot_overlap(self):
        model = _FakeCalibratedModel({0.1: 0.2})
        with self.assertRaisesRegex(ValueError, "disjoint"):
            select_alpha_for_cost(
                model,
                ["same"],
                ["a"],
                ["probe"],
                CostModel(c_frontier=1.0),
                certification_texts=["same"],
                certification_labels=["a"],
                volume=100,
                n_label=1,
                alphas=(0.1,),
            )


class _FailingModel(_FakeCalibratedModel):
    """Fails partway through the sweep, the way an unavailable scoring backend would."""

    def __init__(self, escalation_by_alpha, fail_at_alpha):
        super().__init__(escalation_by_alpha)
        self.alpha = 0.05
        self.qhat = "certified@0.05"
        self._fail_at = fail_at_alpha

    def escalation_rate(self, texts):
        if self.alpha == self._fail_at:
            raise RuntimeError("scoring backend unavailable")
        return self._curve[self.alpha]


class FailedSweepLeavesNoPartialMutationTest(unittest.TestCase):
    """MXR-080-1895: a failed alpha sweep left the model partially mutated.

    Reproduced: with the sweep raising on its third candidate, the model was abandoned at
    ``alpha=0.1`` carrying a threshold fitted on the POLICY-SELECTION rows -- the rows this function's
    docstring promises never bear coverage -- instead of its entry state or the certification result.
    A caller that caught the error and kept serving was serving an uncertified threshold.
    """

    CURVE = {0.01: 0.9, 0.05: 0.5, 0.1: 0.15, 0.15: 0.2}
    ARGS = dict(volume=100, n_label=1, alphas=(0.01, 0.05, 0.1, 0.15))

    def _call(self, model):
        return select_alpha_for_cost(
            model,
            ["cal a", "cal b"],
            ["x", "y"],
            ["probe a"],
            CostModel(c_frontier=1.0, c_local=0.01),
            certification_texts=["cert a", "cert b"],
            certification_labels=["x", "y"],
            **self.ARGS,
        )

    def test_state_is_unchanged_after_a_failed_sweep(self):
        model = _FailingModel(self.CURVE, fail_at_alpha=0.1)
        before = (model.alpha, model.qhat)
        with self.assertRaises(RuntimeError):
            self._call(model)
        self.assertEqual((model.alpha, model.qhat), before)

    def test_the_shallow_snapshots_stated_limit_holds_in_practice(self):
        # Pinning the documented boundary of the fix rather than leaving it to the docstring: the
        # rollback rebinds attributes, so state MUTATED IN PLACE through an attribute (here an audit
        # list the fake model appends to) survives. That is why _attribute_snapshot says so plainly --
        # a caller whose calibrate() mutates nested state in place still has cleanup of its own to do.
        # Undoing it generically would mean deep-copying an arbitrary model, which for a real
        # torch-backed student is both expensive and not always possible.
        model = _FailingModel(self.CURVE, fail_at_alpha=0.1)
        with self.assertRaises(RuntimeError):
            self._call(model)
        self.assertEqual(len(model.calibration_batches), 3)  # in-place appends are NOT rolled back
        self.assertEqual(model.alpha, 0.05)  # the load-bearing calibration state IS

    def test_state_is_unchanged_when_the_final_certification_fails(self):
        # The certification calibrate() is the last mutation; failing there used to leave the model on
        # the winning alpha with the LAST SWEEP's threshold -- the most dangerous shape of all, since
        # alpha reads as certified.
        class _FailingCertification(_FakeCalibratedModel):
            def __init__(self, curve):
                super().__init__(curve)
                self.alpha = 0.05
                self.qhat = "certified@0.05"

            def calibrate(self, texts, labels):
                if "cert a" in list(texts):
                    raise RuntimeError("certification rows rejected")
                return super().calibrate(texts, labels)

        model = _FailingCertification(self.CURVE)
        before = (model.alpha, model.qhat)
        with self.assertRaises(RuntimeError):
            self._call(model)
        self.assertEqual((model.alpha, model.qhat), before)

    def test_a_successful_sweep_still_recalibrates_the_model(self):
        # Negative control: rollback must not undo the documented in-place effect of a sweep that
        # SUCCEEDS. The model must end up on the winning alpha with the certification threshold.
        model = _FakeCalibratedModel(self.CURVE)
        best_alpha, _, _ = self._call(model)
        self.assertEqual(model.alpha, best_alpha)
        self.assertEqual(model.qhat, f"qhat@alpha={best_alpha}")
        self.assertEqual(model.calibration_batches[-1][0], ["cert a", "cert b"])

    def test_a_slotted_model_restores_the_attribute_the_contract_names(self):
        # A model without __dict__ cannot be snapshotted wholesale; alpha is the one attribute the
        # duck-typed contract names, and it is restored rather than left on an abandoned candidate.
        class _Slotted:
            __slots__ = ("alpha", "qhat")

            def __init__(self):
                self.alpha = 0.05
                self.qhat = "certified@0.05"

            def calibrate(self, texts, labels):
                self.qhat = f"qhat@alpha={self.alpha}"
                return self

            def escalation_rate(self, texts):
                raise RuntimeError("scoring backend unavailable")

        model = _Slotted()
        with self.assertRaises(RuntimeError):
            self._call(model)
        self.assertEqual(model.alpha, 0.05)


if __name__ == "__main__":
    unittest.main()
