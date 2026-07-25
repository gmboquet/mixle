"""Construction and forward-boundary contracts for the causal Transformer."""

from __future__ import annotations

import unittest

try:
    import torch

    from mixle.models.transformer import build_causal_lm

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TransformerContractTest(unittest.TestCase):
    def test_architecture_is_validated_before_allocation(self):
        invalid = (
            {"vocab": 0},
            {"vocab": 10, "d_model": 0},
            {"vocab": 10, "n_layer": 1.5},
            {"vocab": 10, "n_head": True},
            {"vocab": 10, "block": -1},
            {"vocab": 10, "d_model": 10, "n_head": 3},
            {"vocab": 10, "gradient_checkpointing": [True]},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_causal_lm(**kwargs)

    def test_token_batch_contract_is_validated(self):
        model = build_causal_lm(vocab=7, d_model=8, n_layer=2, n_head=2, block=4)
        invalid = (
            torch.tensor([]),
            torch.ones(2, 2, 1),
            torch.empty(0, 2),
            torch.empty(2, 0),
            torch.ones(2, 5),
            torch.tensor([[0.0, float("nan")]]),
            torch.tensor([[0.0, 1.5]]),
            torch.tensor([[0, 7]]),
            torch.tensor([[0, -1]]),
        )
        for value in invalid:
            with self.subTest(shape=tuple(value.shape), dtype=value.dtype), self.assertRaises(ValueError):
                model(value)
        with self.assertRaises(TypeError):
            model([[0, 1]])

    def test_position_and_checkpoint_policy_contracts_are_validated(self):
        model = build_causal_lm(vocab=7, d_model=8, n_layer=2, n_head=2, block=4)
        tokens = torch.tensor([[0, 1], [2, 3]])
        invalid_positions = (
            torch.tensor([0]),
            torch.tensor([[0, 1]]),
            torch.tensor([0.0, 1.5]),
            torch.tensor([0, 4]),
        )
        for positions in invalid_positions:
            with self.subTest(shape=tuple(positions.shape)), self.assertRaises(ValueError):
                model(tokens, position_ids=positions)
        with self.assertRaises(TypeError):
            model(tokens, position_ids=[0, 1])

        for policy in ([True], [True, 1], "all"):
            model.gradient_checkpointing = policy
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                model(tokens)
        model.gradient_checkpointing = [True, False]
        self.assertEqual(model(tokens).shape, (2, 7))

    def test_valid_float_tokens_and_batched_positions_work(self):
        model = build_causal_lm(vocab=7, d_model=8, n_layer=1, n_head=2, block=4)
        tokens = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        positions = torch.tensor([[0, 1], [1, 2]])
        self.assertEqual(model(tokens, position_ids=positions).shape, (2, 7))
        self.assertEqual(model(tokens, position_ids=positions, return_all_logits=True).shape, (2, 2, 7))


if __name__ == "__main__":
    unittest.main()
