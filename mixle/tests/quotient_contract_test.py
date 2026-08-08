"""Serialization, invariance, and matched-capacity contracts for quotient modules."""

from __future__ import annotations

import pickle
import unittest

import numpy as np
import pytest

try:
    import torch

    from mixle.models.quotient import (
        CyclicTranslationGroup,
        TranslationQuotientLeaf,
        build_translation_quotient_module,
        build_unpooled_conv_module,
        shift_image_batch,
    )

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.torch
@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class QuotientContractTest(unittest.TestCase):
    def test_module_is_importably_pickleable(self):
        torch.manual_seed(0)
        module = build_translation_quotient_module(3, in_channels=2, hidden_channels=4, out_channels=5)
        restored = pickle.loads(pickle.dumps(module))
        x = torch.randn(2, 2, 7, 7)
        self.assertTrue(torch.equal(module(x), restored(x)))

    def test_declared_periodic_action_is_invariant(self):
        torch.manual_seed(1)
        module = build_translation_quotient_module(3, in_channels=2, hidden_channels=4, out_channels=5).eval()
        x = torch.randn(2, 2, 7, 9)
        shifted = torch.roll(x, shifts=(2, -3), dims=(-2, -1))
        self.assertTrue(torch.allclose(module(x), module(shifted), atol=1e-6, rtol=1e-6))
        group = TranslationQuotientLeaf(module).declared_group()
        self.assertIsInstance(group, CyclicTranslationGroup)
        self.assertEqual(group.order(7, 9), 63)

        x_numpy = x.numpy()
        self.assertTrue(np.array_equal(shift_image_batch(x_numpy, 2, -3), shifted.numpy()))

    def test_baseline_matches_parameters_but_retains_position(self):
        torch.manual_seed(2)
        quotient = build_translation_quotient_module(3, in_channels=2, hidden_channels=4, out_channels=5).eval()
        torch.manual_seed(2)
        baseline = build_unpooled_conv_module(
            3,
            spatial_size=7,
            in_channels=2,
            hidden_channels=4,
            out_channels=5,
        ).eval()
        baseline.load_state_dict(quotient.state_dict())
        self.assertEqual(sum(p.numel() for p in quotient.parameters()), sum(p.numel() for p in baseline.parameters()))
        x = torch.randn(2, 2, 7, 7)
        shifted = torch.roll(x, shifts=(1, 2), dims=(-2, -1))
        self.assertFalse(torch.allclose(baseline(x), baseline(shifted)))


if __name__ == "__main__":
    unittest.main()
