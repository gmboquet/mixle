"""Acceptance tests for mixle.experimental.growth_operators (roadmap H1: growth operators -- net2net
widening + progressive depth stacking, G3's coarsening run backwards).

Each test below IS an acceptance criterion from the roadmap item, not just a smoke test:
    1. Net2Net widening on a raw Linear pair reproduces the exact textbook duplication rule and preserves
       the function exactly (measured max abs/rel forward-pass difference, close to float precision).
    2. `widen_block` grows an entire transformer Block's width and stays function-preserving, verified by
       a real before/after forward-pass comparison (the required output-parity receipt).
    3. `insert_block` grows a CausalLM's depth and stays function-preserving (bitwise-exact, since the
       inserted block is a literal zero-residual identity, not a Taylor approximation), verified the same
       way, on real token-id batches through the whole model.
    4. Grown-then-trained beats from-scratch at matched additional compute on a small ladder: a small
       model is pretrained, grown one layer deeper via `insert_block`, and continued-trained for K steps;
       a fresh model at the grown (larger) size is trained from scratch for the SAME K steps; the grown
       arm reaches a lower held-out loss, since it inherits useful structure the from-scratch arm lacks.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mixle.experimental.growth_operators import (  # noqa: E402
    insert_block,
    net2net_widen,
    verify_output_parity,
    widen_block,
)
from mixle.models.transformer import Block, build_causal_lm


class Net2NetWideningExactnessTest(unittest.TestCase):
    """Acceptance criterion: the real Net2Net duplication rule preserves the composed function exactly,
    to within a stated, tight numerical tolerance -- not an approximation.
    """

    def test_net2net_widen_preserves_function_exactly(self):
        torch.manual_seed(0)
        d_in, old_h, d_out = 6, 8, 5
        lin1 = torch.nn.Linear(d_in, old_h)
        lin2 = torch.nn.Linear(old_h, d_out)

        new_lin1, new_lin2, receipt = net2net_widen(lin1, lin2, new_width=20, seed=0)
        self.assertEqual(new_lin1.out_features, 20)
        self.assertEqual(new_lin2.in_features, 20)

        x = torch.randn(32, d_in)
        with torch.no_grad():
            y_old = lin2(F.gelu(lin1(x)))
            y_new = new_lin2(F.gelu(new_lin1(x)))
        diff = (y_old - y_new).abs()
        max_abs = float(diff.max())
        max_rel = float((diff / y_old.abs().clamp_min(1e-8)).max())

        print(f"\n[H1 acceptance] net2net_widen(8->20): max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}")
        self.assertLess(max_abs, 1e-5)

    def test_net2net_widen_systematic_duplication_mapping(self):
        """The duplication rule is the real textbook one: new hidden rows are EXACT copies of an existing
        row's incoming weights (not fresh random init), and the corresponding outgoing weight columns are
        divided by the replication count -- checked directly against the weight tensors, not just the
        forward-pass outcome.
        """
        torch.manual_seed(1)
        lin1 = torch.nn.Linear(3, 4)
        lin2 = torch.nn.Linear(4, 2)
        new_lin1, new_lin2, _receipt = net2net_widen(lin1, lin2, new_width=8, seed=0, systematic=True)

        # systematic mapping cycles 0,1,2,3,0,1,2,3 -> new rows [4:8] duplicate source rows [0:4].
        with torch.no_grad():
            for new_idx, src_idx in zip(range(4, 8), range(0, 4)):
                self.assertTrue(torch.allclose(new_lin1.weight[new_idx], lin1.weight[src_idx]))
                # every original hidden unit is duplicated exactly once more -> replication count 2 ->
                # every outgoing column (original AND duplicate) is halved.
                self.assertTrue(torch.allclose(new_lin2.weight[:, src_idx], lin2.weight[:, src_idx] / 2.0))
                self.assertTrue(torch.allclose(new_lin2.weight[:, new_idx], lin2.weight[:, src_idx] / 2.0))


class WidenBlockParityTest(unittest.TestCase):
    """Acceptance criterion: widening a WHOLE transformer Block (LayerNorms, attention qkv/proj, MLP)
    coherently stays function-preserving, verified by a real before/after forward pass -- the required
    output-parity receipt "at the moment of the edit".
    """

    def test_widen_block_output_parity_2x(self):
        torch.manual_seed(0)
        d_model, n_head = 16, 2
        block = Block(d_model, n_head)
        new_block, receipt = widen_block(block, new_d_model=32, seed=0)

        self.assertIsNotNone(receipt.parity)
        print(f"\n[H1 acceptance] widen_block(16->32): {receipt.parity}")
        self.assertLess(receipt.parity.max_abs_diff, 1e-4)
        self.assertTrue(receipt.parity.within_tolerance)

    def test_widen_block_output_parity_3x(self):
        torch.manual_seed(1)
        d_model, n_head = 12, 3
        block = Block(d_model, n_head)
        new_block, receipt = widen_block(block, new_d_model=36, seed=1)

        print(f"\n[H1 acceptance] widen_block(12->36): {receipt.parity}")
        self.assertLess(receipt.parity.max_abs_diff, 1e-4)
        self.assertTrue(receipt.parity.within_tolerance)

    def test_widen_block_rejects_non_multiple_width(self):
        block = Block(8, 2)
        with self.assertRaises(ValueError):
            widen_block(block, new_d_model=13, seed=0)


class InsertBlockParityTest(unittest.TestCase):
    """Acceptance criterion: inserting a new near-identity Block into a real CausalLM stays
    function-preserving, verified by a real before/after forward pass on actual token-id batches -- and,
    because the inserted block is an EXACT zero-residual identity (not a Taylor approximation), the
    measured difference should be at (or extremely close to) bitwise float precision, not merely "small".
    """

    def test_insert_block_output_parity(self):
        torch.manual_seed(0)
        model = build_causal_lm(vocab=23, d_model=16, n_layer=3, n_head=2, block=16)
        new_model, receipt = insert_block(model, position=1, seed=0)

        self.assertEqual(new_model.n_layer, model.n_layer + 1)
        print(f"\n[H1 acceptance] insert_block(pos=1): {receipt.parity}")
        self.assertLess(receipt.parity.max_abs_diff, 1e-5)
        self.assertTrue(receipt.parity.within_tolerance)

    def test_insert_block_at_every_position_stays_exact(self):
        torch.manual_seed(2)
        model = build_causal_lm(vocab=17, d_model=8, n_layer=2, n_head=2, block=10)
        for position in range(model.n_layer + 1):
            new_model, receipt = insert_block(model, position=position, seed=position)
            self.assertTrue(
                receipt.parity.within_tolerance,
                f"position={position}: {receipt.parity}",
            )

    def test_insert_block_rejects_out_of_range_position(self):
        model = build_causal_lm(vocab=17, d_model=8, n_layer=2, n_head=2, block=10)
        with self.assertRaises(ValueError):
            insert_block(model, position=model.n_layer + 1, seed=0)


class VerifyOutputParityTest(unittest.TestCase):
    """Direct unit test of the shared parity-receipt helper: an unchanged model has zero diff, and a
    perturbed model is correctly reported as out of tolerance.
    """

    def test_identical_model_has_zero_parity_diff(self):
        torch.manual_seed(0)
        model = build_causal_lm(vocab=11, d_model=8, n_layer=1, n_head=2, block=8)
        batch = torch.randint(0, 11, size=(4, 8)).float()
        receipt = verify_output_parity(model, model, batch, tolerance=1e-9)
        self.assertEqual(receipt.max_abs_diff, 0.0)
        self.assertTrue(receipt.within_tolerance)

    def test_perturbed_model_fails_tolerance(self):
        torch.manual_seed(0)
        model_a = build_causal_lm(vocab=11, d_model=8, n_layer=1, n_head=2, block=8)
        model_b = build_causal_lm(vocab=11, d_model=8, n_layer=1, n_head=2, block=8)
        batch = torch.randint(0, 11, size=(4, 8)).float()
        receipt = verify_output_parity(model_a, model_b, batch, tolerance=1e-9)
        self.assertFalse(receipt.within_tolerance)
        self.assertGreater(receipt.max_abs_diff, 1e-9)


class GrownBeatsFromScratchTest(unittest.TestCase):
    """Acceptance criterion: "grown-then-trained beats from-scratch at matched additional compute on a
    small ladder". A small model is pretrained for N steps, then grown one layer deeper
    (:func:`insert_block`) and continued-trained for K more steps; a FRESH model at the grown size is
    trained from scratch for the SAME K steps (matched additional compute, post-growth). The grown arm
    should reach a lower held-out loss, since it inherits the pretrained shallow model's useful structure
    while the from-scratch arm starts from nothing.

    Synthetic task: predict a fixed deterministic function of the last two context tokens
    (``(3*last + prev) mod vocab``) -- small vocab/model/step counts so this runs in a few seconds, real
    (not toy/hand-waved) cross-entropy loss numbers reported for both arms.
    """

    VOCAB = 16
    BLOCK = 8
    D_MODEL = 16
    N_HEAD = 2

    @staticmethod
    def _make_batch(batch_size: int, rng: np.random.Generator):
        ctx = rng.integers(0, GrownBeatsFromScratchTest.VOCAB, size=(batch_size, GrownBeatsFromScratchTest.BLOCK))
        target = (ctx[:, -1] * 3 + ctx[:, -2]) % GrownBeatsFromScratchTest.VOCAB
        x = torch.as_tensor(ctx, dtype=torch.float32)
        y = torch.as_tensor(target, dtype=torch.long)
        return x, y

    def _train(self, model, steps: int, lr: float, rng: np.random.Generator, batch_size: int = 64) -> float:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        last_loss = float("nan")
        for _ in range(steps):
            x, y = self._make_batch(batch_size, rng)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        return last_loss

    def _eval(self, model, rng: np.random.Generator, batch_size: int = 512) -> float:
        with torch.no_grad():
            x, y = self._make_batch(batch_size, rng)
            return float(F.cross_entropy(model(x), y).item())

    def test_grown_then_trained_beats_from_scratch_at_matched_compute(self):
        torch.manual_seed(0)
        pretrain_steps = 150
        continue_steps = 150
        lr = 3e-3

        # (a) pretrain a small (1-layer) model.
        rng_pretrain = np.random.default_rng(1)
        small_model = build_causal_lm(
            vocab=self.VOCAB, d_model=self.D_MODEL, n_layer=1, n_head=self.N_HEAD, block=self.BLOCK
        )
        self._train(small_model, pretrain_steps, lr, rng_pretrain)

        # grow: insert a new near-identity block (function-preserving at the moment of growth -- checked
        # by insert_block's own receipt).
        grown_model, growth_receipt = insert_block(small_model, position=0, seed=0)
        self.assertTrue(growth_receipt.parity.within_tolerance)
        self.assertEqual(grown_model.n_layer, 2)

        # continue-train the GROWN model for K more steps.
        rng_continue = np.random.default_rng(2)
        self._train(grown_model, continue_steps, lr, rng_continue)

        # (b) a FRESH model at the same (grown, 2-layer) size, trained from scratch for the SAME K steps
        # (matched additional compute -- same optimizer, lr, batch size, step count, and RNG stream for
        # the training data as the continuation phase).
        rng_scratch = np.random.default_rng(2)
        scratch_model = build_causal_lm(
            vocab=self.VOCAB, d_model=self.D_MODEL, n_layer=2, n_head=self.N_HEAD, block=self.BLOCK
        )
        self._train(scratch_model, continue_steps, lr, rng_scratch)

        # Held-out evaluation on a fresh, independent batch (same batch for both arms for a fair
        # comparison).
        rng_eval = np.random.default_rng(999)
        grown_loss = self._eval(grown_model, rng_eval)
        rng_eval = np.random.default_rng(999)
        scratch_loss = self._eval(scratch_model, rng_eval)

        print(
            f"\n[H1 acceptance] grow-then-train vs. from-scratch @ matched compute "
            f"({continue_steps} steps post-growth):\n"
            f"  grown-then-trained held-out loss = {grown_loss:.4f}\n"
            f"  from-scratch held-out loss       = {scratch_loss:.4f}\n"
            f"  grown beats scratch by {scratch_loss - grown_loss:.4f} nats"
        )
        self.assertLess(grown_loss, scratch_loss)


if __name__ == "__main__":
    unittest.main()
