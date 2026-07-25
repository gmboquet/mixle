"""DoE for LLM training (mixle.models.train_search): multi-fidelity BO of the recipe + learning-curve extrapolation.

The search must find a good recipe spending mostly cheap low-budget runs, wire to a REAL mixle LM training loop,
and extrapolate a partial learning curve to a full-budget loss.
"""

import unittest

import numpy as np

from mixle.models.train_search import (
    TrainingSpace,
    extrapolate_learning_curve,
    tune_training,
)


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
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingSpace(**kwargs)

        space = TrainingSpace()
        for point in (
            [0.5, 0.5, 0.5],
            [0.5] * 5,
            [0.5, np.nan, 0.5, 0.5],
            [-0.1, 0.5, 0.5, 0.5],
            [1.1, 0.5, 0.5, 0.5],
        ):
            with self.subTest(point=point), self.assertRaises(ValueError):
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
            with self.subTest(steps=steps, losses=losses, at=at), self.assertRaises(ValueError):
                extrapolate_learning_curve(steps, losses, at=at)


class RealLMCouplingTest(unittest.TestCase):
    def test_lm_train_fn_rejects_invalid_budgets_before_training(self):
        import pytest

        pytest.importorskip("torch")
        from mixle.models.train_search import lm_train_fn

        train = lm_train_fn([0, 1, 0, 1], [0, 1], vocab=2, block=2, max_epochs=2)
        for budget in (0, -0.5, 1.1, np.nan, np.inf, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
