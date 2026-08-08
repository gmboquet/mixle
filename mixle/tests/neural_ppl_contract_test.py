"""Fast boundary tests for declarative neural PPL fitting."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

try:
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.torch
@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class NeuralPPLContractTest(unittest.TestCase):
    @staticmethod
    def _rv(family, predictor):
        return SimpleNamespace(_args=(predictor,), _family=SimpleNamespace(name=family))

    def test_categorical_rejects_invalid_labels_and_weights(self):
        from mixle.ppl.core import Net
        from mixle.ppl.neural import neural_fit

        rv = self._rv("Categorical", Net(out=2))
        x = np.ones((3, 2), dtype=np.float32)
        invalid = [
            ([0, 1], None),
            ([0.0, 1.0, 0.0], None),
            ([0, 1, 2], None),
            ([0, 1, 0], [1.0, 1.0]),
            ([0, 1, 0], [1.0, -1.0, 1.0]),
            ([0, 1, 0], [0.0, 0.0, 0.0]),
            ([0, 1, 0], [1.0, np.nan, 1.0]),
        ]
        for labels, weights in invalid:
            with self.subTest(labels=repr(labels), weights=repr(weights)), self.assertRaises(ValueError):
                neural_fit(rv, labels, given={"x": x}, weights=weights, epochs=1)

    def test_neural_fit_rejects_invalid_geometry_and_controls(self):
        from mixle.ppl.core import Net
        from mixle.ppl.neural import neural_fit

        rv = self._rv("Categorical", Net(out=2))
        with self.assertRaises(ValueError):
            neural_fit(rv, [0], given={"x": [[np.nan, 0.0]]}, epochs=1)
        for kwargs, error in [
            ({"epochs": 0}, ValueError),
            ({"epochs": 1.5}, TypeError),
            ({"lr": np.inf}, ValueError),
            ({"batch_size": 0}, ValueError),
            ({"device": "meta"}, ValueError),
        ]:
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises(error):
                neural_fit(rv, [0], given={"x": [[1.0, 0.0]]}, **kwargs)

    def test_normal_rejects_controls_it_cannot_honor(self):
        from mixle.ppl.core import Net
        from mixle.ppl.neural import neural_fit

        rv = self._rv("Normal", Net(out=1))
        with self.assertRaises(NotImplementedError):
            neural_fit(rv, [1.0], given={"x": [[0.0]]}, epochs=1, batch_size=1)
        with self.assertRaises(NotImplementedError):
            neural_fit(rv, [1.0], given={"x": [[0.0]]}, epochs=1, ewc=([], [], 1.0))

    def test_normal_routes_validated_weights_and_device_to_estimator(self):
        from mixle.ppl.core import Net
        from mixle.ppl.neural import neural_fit

        observed = {}

        class Accumulator:
            def acc_to_encoder(self):
                return self

            def seq_encode(self, data):
                return list(data)

            def seq_update(self, encoded, weights, estimate):
                observed["encoded"] = encoded
                observed["weights"] = np.asarray(weights)

            def value(self):
                return "sufficient-statistics"

        class Estimator:
            def accumulator_factory(self):
                return SimpleNamespace(make=Accumulator)

            def estimate(self, nobs, value):
                observed["value"] = value
                return SimpleNamespace(module=observed["module"])

        class Leaf:
            def __init__(self, module, *, m_steps, lr, device):
                observed.update(module=module, m_steps=m_steps, lr=lr, device=device)

            def estimator(self):
                return Estimator()

        rv = self._rv("Normal", Net(out=1))
        with patch("mixle.models.neural_leaf.NeuralLeaf", Leaf):
            result = neural_fit(
                rv,
                [1.0, 2.0],
                given={"x": [[0.0], [1.0]]},
                epochs=3,
                lr=0.2,
                device="cpu",
                weights=[2.0, 0.0],
            )

        np.testing.assert_array_equal(observed["weights"], [2.0, 0.0])
        self.assertEqual(observed["device"], "cpu")
        self.assertEqual(observed["m_steps"], 3)
        self.assertEqual(observed["lr"], 0.2)
        self.assertEqual(observed["value"], "sufficient-statistics")
        self.assertEqual(result.kind, "normal")

    def test_incompatible_initialization_and_modules_fail_at_boundary(self):
        from mixle.ppl.core import Net, _NeuralPredictor
        from mixle.ppl.neural import NeuralResult, neural_fit

        categorical = self._rv("Categorical", Net(out=2))
        with self.assertRaises(TypeError):
            neural_fit(categorical, [0], given={"x": [[0.0]]}, epochs=1, init=object())
        wrong_fit = NeuralResult(SimpleNamespace(module=nn.Linear(1, 2)), "other", "normal")
        with self.assertRaises(ValueError):
            neural_fit(categorical, [0], given={"x": [[0.0]]}, epochs=1, init=wrong_fit)

        class WrongWidth(_NeuralPredictor):
            field = "x"
            out = 2

            def build(self, in_shape):
                return nn.Linear(int(in_shape[0]), 1)

        with self.assertRaisesRegex(ValueError, "output must have shape"):
            neural_fit(
                self._rv("Categorical", WrongWidth()),
                [0, 1],
                given={"x": [[0.0], [1.0]]},
                epochs=1,
            )


if __name__ == "__main__":
    unittest.main()
