"""Bring-your-own-model interop contract (worklist N9.6).

The interop promise is: take an external torch model that scores a batch to per-row log-densities, wrap it
through the documented ``GradLeaf`` adapter, and use it like any mixle distribution -- score through the
distribution contract and compose it with a classical probabilistic component in a larger model, fit by the
same ``optimize``. This pins that contract with a minimal local torch density module (no network); the
real-Hugging-Face-checkpoint instantiation of the same adapter is exercised, network permitting, by
``peft_lora_grad_leaf_smoke_test.py``.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mixle.inference.estimation import optimize  # noqa: E402
from mixle.models.grad_leaf import GradLeaf, looks_like_torch_module  # noqa: E402
from mixle.stats import GaussianEstimator, MixtureEstimator  # noqa: E402


class _TinyDensity(nn.Module):
    """A stand-in 'bring-your-own' torch model: scores a batch to per-row Gaussian log-densities."""

    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1))
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def log_density(self, batch):
        x = batch.float().reshape(-1)
        sig = torch.exp(self.log_sigma)
        return -0.5 * ((x - self.mu) / sig) ** 2 - self.log_sigma - 0.9189385332


class ByoModelContractTest(unittest.TestCase):
    def test_torch_module_is_detected_and_wraps_as_a_leaf(self):
        m = _TinyDensity()
        self.assertTrue(looks_like_torch_module(m))
        leaf = GradLeaf(m)
        # scores through the mixle distribution contract (finite log-density).
        self.assertTrue(np.isfinite(float(leaf.log_density(1.0))))
        self.assertTrue(np.isfinite(float(leaf.log_density(-2.0))))

    def test_composes_with_a_classical_component_and_fits(self):
        # a bring-your-own leaf and a classical Gaussian, together in one mixture, fit by the same optimize.
        data = np.concatenate([np.random.RandomState(0).randn(200), np.random.RandomState(1).randn(200) + 5.0]).tolist()
        est = MixtureEstimator([GradLeaf(_TinyDensity()).estimator(), GaussianEstimator()])
        fitted = optimize(data, est, max_its=8, out=None)
        lls = np.array([float(fitted.log_density(x)) for x in data])
        self.assertTrue(np.all(np.isfinite(lls)))
        self.assertEqual(len(fitted.components), 2)  # the external leaf and the classical component coexist


if __name__ == "__main__":
    unittest.main()
