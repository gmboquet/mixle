"""Fast boundary tests for declarative neural-density fitting."""

import unittest
from unittest.mock import patch

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.torch
@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class DensityContractTest(unittest.TestCase):
    def test_constructor_dimensions_and_training_controls_are_strict(self):
        from mixle.ppl import EBM, MDN, VAE, CondFlow, DiscreteAR, Flow

        invalid = [
            lambda: Flow(0),
            lambda: Flow(2, hidden=0),
            lambda: Flow(2, layers=0),
            lambda: Flow(2, m_steps=0),
            lambda: Flow(2, lr=np.nan),
            lambda: VAE(2, latent=0),
            lambda: VAE(2, eval_samples=0),
            lambda: DiscreteAR(2, 1),
            lambda: EBM(2, noise_ratio=0),
            lambda: MDN(1, 1, k=0),
            lambda: CondFlow(1, 1),
        ]
        for build in invalid:
            with self.subTest(build=repr(build)), self.assertRaises((TypeError, ValueError)):
                build()

    def test_unconditional_data_shape_finiteness_and_discrete_support(self):
        from mixle.ppl import DiscreteAR, Flow

        invalid_continuous = [
            [],
            [1.0, 2.0],
            [[1.0]],
            [[1.0, np.nan]],
            [[1.0, 2.0, 3.0]],
        ]
        for data in invalid_continuous:
            with self.subTest(data=repr(data)), self.assertRaises(ValueError):
                Flow(2).fit(data, max_its=1)

        for data in [
            [[0.0, 1.0]],
            [[0, 3]],
            [[-1, 1]],
            [[0]],
        ]:
            with self.subTest(data=repr(data)), self.assertRaises(ValueError):
                DiscreteAR(2, 3).fit(data, max_its=1)

    def test_conditional_axes_and_support_are_aligned(self):
        from mixle.ppl import MDN, CondDiscreteAR

        with self.assertRaises(ValueError):
            MDN(2, 1).fit([[0.0]], given={"x": [[1.0]]}, max_its=1)
        with self.assertRaises(ValueError):
            MDN(1, 1).fit([[0.0], [1.0]], given={"x": [[1.0]]}, max_its=1)
        with self.assertRaises(ValueError):
            MDN(1, 1).fit([[0.0]], given={"x": [[1.0]], "unused": [[2.0]]}, max_its=1)
        with self.assertRaises(ValueError):
            CondDiscreteAR(1, 2, 3).fit([[0.0, 1.0]], given={"x": [[1.0]]}, max_its=1)

    def test_supported_controls_are_forwarded_and_unknown_ones_rejected(self):
        from mixle.ppl import Flow

        observed = {}

        def fake_optimize(data, estimator, **kwargs):
            observed["data"] = data
            observed.update(kwargs)
            return kwargs["prev_estimate"]

        with patch("mixle.inference.optimize", side_effect=fake_optimize):
            fitted = Flow(2).fit(
                [[0.0, 1.0], [1.0, 2.0]],
                max_its=3,
                delta=0.0,
                seed=7,
                backend="local",
            )
        self.assertEqual(fitted._kind, "bound")
        self.assertEqual(observed["max_its"], 3)
        self.assertEqual(observed["delta"], 0.0)
        self.assertEqual(observed["seed"], 7)
        self.assertEqual(observed["backend"], "local")

        with self.assertRaisesRegex(TypeError, "weights"):
            Flow(2).fit([[0.0, 1.0]], max_its=1, weights=[1.0])
        with self.assertRaises(NotImplementedError):
            Flow(2).fit([[0.0, 1.0]], max_its=1, missing="marginalize")


if __name__ == "__main__":
    unittest.main()
