"""Random forests as a conditional leaf in the estimation framework.

Exercises the full contract: seq_encode -> accumulator seq_update/value/combine -> estimate, the optimize()
driver at max_its=1, vectorized seq_log_density returning log p(y | x), and the conditional sampler. Confirms the
fitted forest recovers a held-out signal for both classification and regression.
"""

import unittest

import numpy as np

from mixle.inference.estimation import optimize
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

    def test_auto_task_inference(self):
        tr, _ = _regression_data()
        model = optimize(tr, RandomForestEstimator(task="auto", n_estimators=40, random_state=0), max_its=1, out=None)
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
