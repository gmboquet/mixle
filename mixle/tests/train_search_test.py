"""DoE for LLM training (mixle.models.train_search): multi-fidelity BO of the recipe + learning-curve extrapolation.

The search must find a good recipe spending mostly cheap low-budget runs, wire to a REAL mixle LM training loop,
and extrapolate a partial learning curve to a full-budget loss.
"""

import importlib.util
import unittest

import numpy as np

from mixle.models.train_search import (
    TrainingSearchResult,
    TrainingSpace,
    extrapolate_learning_curve,
    lm_train_fn,
    tune_training,
)

# `tune_training` fits a Gaussian-process surrogate over the recipe space, and that surrogate is
# torch-backed (`mixle.models.gaussian_process._torch_engine`). Every test that calls it therefore
# needs torch, and the hard-budget core CI tier installs no optional extras -- these failed there as
# bare ModuleNotFoundError rather than skipping, which reads as a broken candidate instead of an
# absent optional dependency.
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "multi-fidelity search fits a torch-backed GP surrogate")
class MultiFidelitySearchTest(unittest.TestCase):
    def test_space_rejects_invalid_schema_and_points(self):
        invalid_spaces = (
            {"d_model_choices": []},
            {"d_model_choices": [64, 64]},
            {"batch_choices": [0]},
            {"n_layer_range": (4, 2)},
            {"n_layer_range": (2.5, 4)},
            {"log10_lr_range": (np.nan, -2.0)},
        )
        for kwargs in invalid_spaces:
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises(ValueError):
                TrainingSpace(**kwargs)

        space = TrainingSpace()
        for point in (
            [0.5, 0.5, 0.5],
            [0.5] * 5,
            [0.5, np.nan, 0.5, 0.5],
            [-0.1, 0.5, 0.5, 0.5],
            [1.1, 0.5, 0.5, 0.5],
        ):
            with self.subTest(point=repr(point)), self.assertRaises(ValueError):
                space.decode(point)

    def test_finds_good_recipe_using_both_fidelities(self):
        # surrogate: optimum near d_model=256, n_layer=6, lr=1e-3; cheap fidelity is noisier
        def train(recipe, budget):
            d, nl, lr = recipe["d_model"], recipe["n_layer"], recipe["lr"]
            true = (np.log2(d) - 8) ** 2 + 0.1 * (nl - 6) ** 2 + (np.log10(lr) + 3) ** 2
            noise = (1.0 - budget) * np.random.RandomState(int(d + nl + 1000 * lr)).randn() * 0.3
            return float(true + noise)

        res = tune_training(train, TrainingSpace(), fidelities=(0.25, 1.0), max_cost=15, seed=0)
        self.assertEqual(res.recipe["d_model"], 256)  # the search located the optimum's scale
        # generous loss bound: the BO search explores slightly different recipes across numpy
        # versions/platforms (GP/acquisition float math), so pin the meaningful result (the d_model
        # scale) and only assert the loss is far better than a mis-scaled recipe -- not a tight value.
        self.assertLess(res.loss, 2.0)
        fids = sorted(set(np.round(np.asarray(res.history["X"])[:, -1], 2).tolist()))
        self.assertIn(0.25, fids)  # it actually spent cheap low-budget evaluations
        self.assertIn(1.0, fids)

    def test_raises_clear_error_when_budget_never_reaches_target_fidelity(self):
        # max_cost=0.3 clears multi_fidelity_minimize's upfront "cheapest fidelity affordable" check
        # (the cheapest fidelity, 0.25, is <= 0.3) but can never afford a single evaluation at the
        # target fidelity (cost 1.0, the max of `fidelities`), so target_evaluated stays False and
        # `result["x"]`/`result["y"]` are None (MXR-080-0181). Before the tune_training fix, that None
        # reached `np.asarray(None, dtype=np.float64)` -- which silently becomes `array(nan)` rather
        # than raising -- and only failed later, inside TrainingSpace.decode, with an unrelated-looking
        # IndexError instead of a message naming the actual problem (the budget was too tight).
        def train(recipe, budget):
            return 1.0  # value is irrelevant; the target fidelity is never reached regardless

        with self.assertRaises(ValueError) as cm:
            tune_training(train, TrainingSpace(), fidelities=(0.25, 1.0), max_cost=0.3, n_init=4, seed=0)
        message = str(cm.exception)
        self.assertIn("tune_training", message)  # names the failing function
        self.assertIn("target fidelity", message)  # explains why
        self.assertIn("0.3", message)  # the actual max_cost, for actionability


class LearningCurveTest(unittest.TestCase):
    def test_power_law_extrapolation(self):
        t = np.array([1, 2, 4, 8, 16.0])
        y = 0.5 + 4.0 * t**-0.6  # a clean learning curve
        pred = extrapolate_learning_curve(t, y, at=64)
        self.assertAlmostEqual(pred, 0.5 + 4.0 * 64**-0.6, places=2)

    def test_invalid_or_inconclusive_curves_fail_explicitly(self):
        invalid = (
            ([], [], 10),
            ([1, 2], [3.0], 10),
            ([1, 2], [3.0, 2.0], 10),
            ([1, 1, 2], [3.0, 2.5, 2.0], 10),
            ([0, 1, 2], [3.0, 2.5, 2.0], 10),
            ([1, 2, 3], [3.0, np.nan, 2.0], 10),
            ([1, 2, 3], [3.0, 2.5, 2.0], 3),
        )
        for steps, losses, at in invalid:
            with self.subTest(steps=repr(steps), losses=repr(losses), at=repr(at)), self.assertRaises(ValueError):
                extrapolate_learning_curve(steps, losses, at=at)


class RealLMCouplingTest(unittest.TestCase):
    def test_lm_train_fn_rejects_invalid_budgets_before_training(self):
        import pytest

        pytest.importorskip("torch")
        from mixle.models.train_search import lm_train_fn

        train = lm_train_fn([0, 1, 0, 1], [0, 1], vocab=2, block=2, max_epochs=2)
        for budget in (0, -0.5, 1.1, np.nan, np.inf, True):
            with self.subTest(budget=repr(budget)), self.assertRaises(ValueError):
                train({}, budget)

    def test_lm_train_fn_trains_a_real_lm(self):
        import pytest

        pytest.importorskip("torch")
        from mixle.models.train_search import lm_train_fn

        vocab = 8
        tokens = list(range(vocab)) * 200  # a learnable repeating cycle
        val = list(range(vocab)) * 40
        train = lm_train_fn(tokens, val, vocab=vocab, block=16, max_epochs=3)

        recipe = {"d_model": 32, "n_layer": 2, "lr": 3e-3, "batch_size": 32}
        loss_full = train(recipe, 1.0)
        self.assertTrue(np.isfinite(loss_full) and loss_full > 0)
        # a real LM trained on a predictable cycle drives held-out nll well below the uniform ln(vocab)
        self.assertLess(loss_full, np.log(vocab) - 0.5)


class FidelityIsRealWorkTest(unittest.TestCase):
    """MXR-080-1890: a declared budget must buy work proportional to it, not a rounded epoch count."""

    CORPUS = list(range(8)) * 200  # 1600 tokens
    VAL = list(range(8)) * 40

    def _fn(self, **kwargs):
        return lm_train_fn(self.CORPUS, self.VAL, vocab=8, block=16, **kwargs)

    def test_distinct_budgets_no_longer_execute_identical_work(self):
        # REPRODUCED before the fix: fidelity was `max(1, round(max_epochs * budget))` whole epochs,
        # so at the library's own default max_epochs=3 the budgets below ALL executed exactly one
        # epoch -- a "cheap screening" run and a 4x more expensive one were byte-identical work, and
        # the multi-fidelity search was comparing recipes at budgets that were secretly the same.
        budgets = (0.05, 0.1, 0.2, 0.25, 0.3, 0.4)
        old_epochs = {max(1, int(round(3 * b))) for b in budgets}
        self.assertEqual(old_epochs, {1}, "the pre-fix mapping collapsed all of these onto one epoch")

        plan = self._fn(max_epochs=3).plan
        work = [plan(b).total_tokens for b in budgets]
        self.assertEqual(len(set(work)), len(budgets), f"budgets still collapse onto shared work: {work}")
        self.assertEqual(work, sorted(work))  # and more budget still means more work

    def test_max_epochs_one_is_no_longer_a_single_fidelity(self):
        # The worst case of the same defect: with max_epochs=1 EVERY budget rounded to one epoch, so
        # the low fidelity and the target fidelity ran the identical job and low-fidelity screening
        # -- the entire premise of the search -- was not happening at all.
        fn = self._fn(max_epochs=1)
        work = [fn.plan(b).total_tokens for b in (0.05, 0.25, 0.5, 1.0)]
        self.assertEqual(len(set(work)), 4, f"max_epochs=1 still yields one fidelity: {work}")

    def test_full_budget_is_unchanged_by_the_fix(self):
        # The fix must not quietly redefine what a full-budget run is: budget=1.0 still means
        # max_epochs whole epochs over the whole corpus, exactly as before.
        full = self._fn(max_epochs=3).plan(1.0)
        self.assertEqual(full.epochs, 3)
        self.assertEqual(full.tokens_per_epoch, len(self.CORPUS))
        self.assertEqual(full.total_tokens, 3 * len(self.CORPUS))

    def test_plan_never_falls_below_one_trainable_window(self):
        # Guarding against the fix's own failure mode: a tiny budget must not produce a token prefix
        # too short to form a single training window (LM.fit raises "yielded no training batches").
        fn = self._fn(max_epochs=3)
        self.assertGreaterEqual(fn.plan(1e-9).tokens_per_epoch, 16 + 1)  # block + 1

    def test_declared_fidelity_ladder_collapse_is_reported_not_silent(self):
        # Opt-in (`fidelities=`), so existing callers are unaffected. Token resolution genuinely
        # runs out on a short corpus; the point is that it is reported rather than assumed away.
        with self.assertRaises(ValueError) as cm:
            lm_train_fn(list(range(4)) * 3, [0, 1], vocab=4, block=2, max_epochs=1, fidelities=(0.5, 0.51))
        self.assertIn("identical work", str(cm.exception))
        # a well-spaced ladder over the same corpus is accepted
        lm_train_fn(list(range(4)) * 3, [0, 1], vocab=4, block=2, max_epochs=1, fidelities=(0.25, 1.0))

    def test_seed_is_exact_and_recorded_on_every_trial(self):
        for bad in (2.9, True, -1, "0"):
            with self.subTest(seed=repr(bad)), self.assertRaises(ValueError):
                self._fn(max_epochs=3, seed=bad)
        fn = self._fn(max_epochs=3, seed=7)
        self.assertEqual(fn.seed, 7)
        self.assertEqual(fn.plan(0.5).seed, 7)
        self.assertEqual(fn.trials, ())  # nothing executed yet


class CommonRandomNumbersTest(unittest.TestCase):
    def test_identical_recipe_and_budget_now_returns_an_identical_loss(self):
        # REPRODUCED before the fix: LM.__init__ draws its weights from the ambient torch RNG and
        # lm_train_fn seeded nothing, so two back-to-back calls with the SAME recipe and budget
        # returned 6.46 then 7.65 nats/token. That run-to-run spread is far wider than the
        # differences the search is meant to rank, so the search was ranking noise.
        import pytest

        pytest.importorskip("torch")

        tokens = list(range(8)) * 120
        fn = lm_train_fn(tokens, list(range(8)) * 20, vocab=8, block=16, max_epochs=2, seed=3)
        recipe = {"d_model": 16, "n_layer": 1, "batch_size": 8, "lr": 1e-3}
        first = fn(dict(recipe), 0.5)
        second = fn(dict(recipe), 0.5)
        self.assertEqual(first, second, "same recipe + same budget must be the same run")
        # and the executed work/seed of both runs is recorded, so a finished search is auditable
        self.assertEqual(len(fn.trials), 2)
        self.assertEqual(fn.trials[0], fn.trials[1])
        self.assertEqual(fn.trials[0].seed, 3)
        self.assertGreater(fn.trials[0].total_tokens, 0)

    def test_seeding_the_search_does_not_reseed_the_callers_torch_rng(self):
        import pytest

        pytest.importorskip("torch")
        import torch

        torch.manual_seed(1234)
        expected = torch.randn(3)
        torch.manual_seed(1234)
        fn = lm_train_fn(list(range(8)) * 120, list(range(8)) * 20, vocab=8, block=16, max_epochs=1, seed=99)
        fn({"d_model": 16, "n_layer": 1, "batch_size": 8, "lr": 1e-3}, 0.5)
        self.assertTrue(torch.equal(torch.randn(3), expected), "the caller's torch stream was consumed/reseeded")


class TrainingSearchResultOwnershipTest(unittest.TestCase):
    """MXR-080-1890: the recorded outcome must not be editable through the caller's own references."""

    def test_recipe_and_history_are_detached_from_the_caller(self):
        # REPRODUCED before the fix: both were stored by reference, so mutating the caller's dict
        # rewrote the recorded winning recipe after the fact.
        recipe = {"d_model": 64}
        history = ["step0"]
        result = TrainingSearchResult(recipe=recipe, loss=1.5, history=history, seed=0)
        recipe["d_model"] = 999
        history.append("MUTATED")
        self.assertEqual(result.recipe, {"d_model": 64})
        self.assertEqual(result.history, ["step0"])
        self.assertIsInstance(result.recipe, dict)  # type stays load-bearing for a training callback
        self.assertIsInstance(result.history, list)

    def test_inconsistent_results_are_refused(self):
        # REPRODUCED before the fix: loss=nan and loss="not a number" were both accepted, as was a
        # recipe that was not a mapping at all.
        for bad in ({"loss": np.nan}, {"loss": np.inf}, {"loss": "not a number"}, {"seed": -1}, {"seed": True}):
            kwargs = {"recipe": {"lr": 1e-3}, "loss": 1.0, **bad}
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                TrainingSearchResult(**kwargs)
        with self.assertRaises(TypeError):
            TrainingSearchResult(recipe="not a mapping", loss=1.0)

    # Only this method drives a real search; the rest construct the record directly and need no
    # optional dependency, so the guard is per-method rather than on the class.
    @unittest.skipUnless(HAS_TORCH, "tune_training fits a torch-backed GP surrogate")
    def test_tune_training_records_the_seed_it_searched_under(self):
        def train(recipe, budget):
            return float((np.log2(recipe["d_model"]) - 8) ** 2)

        result = tune_training(train, TrainingSpace(), fidelities=(0.25, 1.0), max_cost=15, seed=5)
        self.assertEqual(result.seed, 5)


if __name__ == "__main__":
    unittest.main()
