"""Fast state-isolation and manifest checks for continual-learning helpers."""

import copy
import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.continual import ParameterBundle, ewc, fisher_diagonal, snapshot  # noqa: E402


class _Leaf:
    def __init__(self, module):
        self.module = module


class FisherIsolationTest(unittest.TestCase):
    def setUp(self):
        self.module = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2, 2),
        )
        self.module.train()
        self.module[0].eval()
        for parameter in self.module.parameters():
            parameter.grad = torch.ones_like(parameter)
        self.x = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0], [-1.0, 1.0]], dtype=np.float32)
        self.y = np.array([0, 1, 0, 1])

    def test_fisher_runs_on_an_isolated_copy_and_restores_nothing_because_source_is_untouched(self):
        state_before = copy.deepcopy(self.module.state_dict())
        modes_before = [submodule.training for submodule in self.module.modules()]
        gradients_before = [parameter.grad.clone() for parameter in self.module.parameters()]

        fisher = fisher_diagonal(_Leaf(self.module), self.x, self.y, samples=3, seed=4)

        self.assertIsInstance(fisher, ParameterBundle)
        self.assertEqual(fisher.names, tuple(name for name, _ in self.module.named_parameters()))
        self.assertEqual(modes_before, [submodule.training for submodule in self.module.modules()])
        for name, value in self.module.state_dict().items():
            torch.testing.assert_close(value, state_before[name])
        for parameter, gradient in zip(self.module.parameters(), gradients_before):
            torch.testing.assert_close(parameter.grad, gradient)
        self.assertTrue(all(torch.isfinite(value).all() for value in fisher))

    def test_invalid_or_empty_data_and_sample_counts_are_rejected(self):
        for samples in (0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                fisher_diagonal(_Leaf(self.module), self.x, self.y, samples=samples)
        with self.assertRaisesRegex(ValueError, "non-empty aligned"):
            fisher_diagonal(_Leaf(self.module), np.empty((0, 2)), np.empty(0, dtype=int))


class EWCManifestTest(unittest.TestCase):
    def setUp(self):
        self.module = torch.nn.Linear(2, 2)
        self.anchor = snapshot(self.module)
        self.fisher = ParameterBundle([torch.ones_like(value) for value in self.anchor], self.anchor.names)

    def test_valid_bundle_preserves_identity_manifest(self):
        anchor, fisher, penalty = ewc(self.anchor, self.fisher, lam=3.0)
        self.assertEqual(anchor.names, fisher.names)
        self.assertEqual(penalty, 3.0)

    def test_invalid_penalty_or_manifest_fails_before_training(self):
        for penalty in (-1.0, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                ewc(self.anchor, self.fisher, lam=penalty)
        wrong_names = ParameterBundle(list(self.fisher), tuple(f"other.{i}" for i in range(len(self.fisher))))
        with self.assertRaisesRegex(ValueError, "identity manifests"):
            ewc(self.anchor, wrong_names)

    def test_shape_device_dtype_and_finiteness_are_validated(self):
        wrong_shape_values = list(self.fisher)
        wrong_shape_values[0] = torch.ones(1)
        wrong_shape = ParameterBundle(wrong_shape_values, self.fisher.names)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            ewc(self.anchor, wrong_shape)

        nonfinite_values = [value.clone() for value in self.fisher]
        nonfinite_values[0].reshape(-1)[0] = np.nan
        nonfinite = ParameterBundle(nonfinite_values, self.fisher.names)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ewc(self.anchor, nonfinite)


if __name__ == "__main__":
    unittest.main()
