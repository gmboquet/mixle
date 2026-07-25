"""Random forests as a conditional leaf in the estimation framework.

Exercises the full contract: seq_encode -> accumulator seq_update/value/combine -> estimate, the optimize()
driver at max_its=1, vectorized seq_log_density returning log p(y | x), and the conditional sampler. Confirms the
fitted forest recovers a held-out signal for both classification and regression.
"""

import unittest

import numpy as np

from mixle.inference.estimation import optimize
from mixle.models._forest import NativeRandomForest
from mixle.models.random_forest import (
    RandomForestAccumulatorFactory,
    RandomForestConditional,
    RandomForestEncoder,
    RandomForestEstimator,
)
from mixle.stats import log_density, seq_encode


def _classification_data(seed=0, n=400):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.randn(n)) > 0).astype(int)
    data = list(zip(X.tolist(), y.tolist()))
    return data[:300], data[300:]


def _regression_data(seed=1, n=400):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = 2.0 * X[:, 0] - X[:, 2] + 0.5 * rng.randn(n)
    data = list(zip(X.tolist(), y.tolist()))
    return data[:300], data[300:]


class RandomForestLeafTestCase(unittest.TestCase):
    def test_manual_accumulate_estimate_path(self):
        tr, te = _classification_data()
        est = RandomForestEstimator(task="classification", n_estimators=50, random_state=0)
        # framework path by hand: encode -> accumulate -> value -> estimate
        enc = seq_encode(tr, model=None, estimator=est)
        acc = est.accumulator_factory().make()
        for sz, e in enc:
            acc.seq_update(e, np.ones(sz), None)
        model = est.estimate(None, acc.value())
        self.assertIsInstance(model, RandomForestConditional)
        ld = log_density(te, model)
        self.assertEqual(ld.shape, (len(te),))
        self.assertTrue(np.all(ld <= 1e-9))  # log-probabilities

    def test_optimize_driver(self):
        tr, te = _classification_data()
        model = optimize(
            tr,
            RandomForestEstimator(task="classification", n_estimators=60, random_state=0),
            max_its=1,
            out=None,
        )
        Xte = np.asarray([f for f, _ in te])
        yte = np.asarray([t for _, t in te])
        acc = (model.forest.predict(Xte) == yte).mean()
        self.assertGreater(acc, 0.8)

    def test_combine_value_roundtrip(self):
        tr, _ = _classification_data()
        est = RandomForestEstimator(task="classification", n_estimators=10, random_state=0)
        enc = seq_encode(tr, estimator=est)[0][1]
        a = est.accumulator_factory().make()
        b = est.accumulator_factory().make()
        a.seq_update((enc[0][:150], enc[1][:150]), np.ones(150), None)
        b.seq_update((enc[0][150:], enc[1][150:]), np.ones(len(enc[1]) - 150), None)
        merged = est.accumulator_factory().make()
        merged.combine(a.value())
        merged.combine(b.value())
        X, y, w = merged.value()
        self.assertEqual(len(y), len(tr))
        self.assertEqual(X.shape, (len(tr), 5))

    def test_task_must_be_explicit_even_for_float_targets(self):
        tr, _ = _regression_data()
        with self.assertRaisesRegex(ValueError, "explicitly"):
            RandomForestEstimator()
        with self.assertRaisesRegex(ValueError, "automatic dtype inference"):
            RandomForestEstimator(task="auto")
        model = optimize(
            tr,
            RandomForestEstimator(task="regression", n_estimators=40, random_state=0),
            max_its=1,
            out=None,
        )
        self.assertEqual(model.task, "regression")
        self.assertIsNotNone(model.sigma)

    def test_regression_recovers_signal(self):
        tr, te = _regression_data()
        model = optimize(
            tr, RandomForestEstimator(task="regression", n_estimators=80, random_state=0), max_its=1, out=None
        )
        Xte = np.asarray([f for f, _ in te])
        yte = np.asarray([t for _, t in te], dtype=float)
        pred = model.forest.predict(Xte)
        r2 = 1.0 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
        self.assertGreater(r2, 0.8)
        ld = log_density(te, model)
        self.assertTrue(np.isfinite(ld).all())
        oob = np.asarray(model.forest.oob_prediction_)
        covered = np.isfinite(oob)
        expected_sigma = np.sqrt(np.mean((np.asarray([y for _, y in tr])[covered] - oob[covered]) ** 2))
        self.assertAlmostEqual(model.sigma, max(expected_sigma, 1.0e-3))

    def test_reestimation_preserves_full_model_specification(self):
        tr, _ = _regression_data()
        expected = {
            "n_estimators": 17,
            "max_depth": 7,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 31,
            "min_sigma": 0.05,
            "n_features": 5,
        }
        model = optimize(
            tr,
            RandomForestEstimator(task="regression", **expected),
            max_its=1,
            out=None,
        )
        estimator = model.estimator()
        self.assertEqual(
            {
                "n_estimators": estimator.n_estimators,
                "max_depth": estimator.max_depth,
                "min_samples_split": estimator.min_samples_split,
                "min_samples_leaf": estimator.min_samples_leaf,
                "max_features": estimator.max_features,
                "random_state": estimator.random_state,
                "min_sigma": estimator.min_sigma,
                "n_features": estimator.n_features,
            },
            expected,
        )

    def test_invalid_forest_contracts_fail_before_training_or_prediction(self):
        X = np.ones((4, 2))
        y = np.arange(4.0)
        invalid_estimators = [
            {"n_estimators": 0},
            {"max_depth": 0},
            {"min_samples_split": 1},
            {"min_samples_leaf": 0},
            {"max_features": 0},
            {"min_sigma": 0.0},
        ]
        for controls in invalid_estimators:
            with self.subTest(controls=controls), self.assertRaises((TypeError, ValueError)):
                RandomForestEstimator(task="regression", **controls)
        forest = NativeRandomForest(task="regression", n_estimators=2, random_state=0)
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            forest.predict(X)
        for bad_X, bad_y, bad_w in [
            (np.ones((0, 2)), np.zeros(0), np.zeros(0)),
            (np.array([[1.0, np.nan]]), np.array([1.0]), np.array([1.0])),
            (X, y[:-1], np.ones(4)),
            (X, y, np.array([1.0, -1.0, 1.0, 1.0])),
            (X, y, np.array([1.0, np.inf, 1.0, 1.0])),
            (X, y, np.zeros(4)),
        ]:
            with self.subTest(X=bad_X, y=bad_y, w=bad_w), self.assertRaises((TypeError, ValueError)):
                forest.fit(bad_X, bad_y, bad_w)

    def test_conditional_sampler(self):
        tr, te = _classification_data()
        model = optimize(
            tr, RandomForestEstimator(task="classification", n_estimators=30, random_state=0), max_its=1, out=None
        )
        s = model.sampler(seed=3)
        with self.assertRaises(NotImplementedError):
            s.sample(5)
        Xte = np.asarray([f for f, _ in te[:10]])
        drawn = s.sample_y(Xte)
        self.assertEqual(len(drawn), 10)
        self.assertTrue(set(np.unique(drawn)).issubset({0, 1}))

    def test_encoder_roundtrip(self):
        tr, _ = _classification_data()
        enc = RandomForestEncoder().seq_encode(tr)
        X, y = enc
        self.assertEqual(X.shape, (len(tr), 5))
        self.assertEqual(len(y), len(tr))
        self.assertIsInstance(RandomForestAccumulatorFactory().make().acc_to_encoder(), RandomForestEncoder)


if __name__ == "__main__":
    unittest.main()
