"""Quantization-aware training (mixle.models.qat): straight-through int4 fake-quant, and the headline
acceptance claim -- QAT beats post-training quantization (PTQ) at matched int4 size on fixtures.
"""

from __future__ import annotations

import copy
import unittest

import numpy as np
import pytest

pytest.importorskip("torch")  # QAT wraps torch Linear layers; skip cleanly where torch is absent

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mixle.inference.estimation import optimize  # noqa: E402
from mixle.models.grad_leaf import GradLeaf  # noqa: E402
from mixle.models.qat import (  # noqa: E402
    QATWrapper,
    apply_qat,
    fake_quantize_int4,
    set_fake_quant_enabled,
)
from mixle.models.transformer import build_causal_lm  # noqa: E402
from mixle.task.quantize import quantize_dequantize_array  # noqa: E402


class STECorrectnessTest(unittest.TestCase):
    """fake_quantize_int4: forward is the real int4 round trip, backward is the identity (STE)."""

    def test_forward_matches_real_int4_quantize_dequantize(self):
        rng = np.random.RandomState(0)
        w = rng.normal(size=(11, 7)).astype(np.float64)
        x = torch.as_tensor(w, dtype=torch.float32).clone().requires_grad_(True)

        got = fake_quantize_int4(x).detach().numpy()

        wq, scale = quantize_dequantize_array(w, bits=4)
        expected = wq.astype(np.float64) * scale
        np.testing.assert_allclose(got, expected, atol=1e-6)
        # and it is really quantized: at most 15 distinct values (int4 symmetric range [-7, 7])
        self.assertLessEqual(len(np.unique(got)), 15)

    def test_backward_passes_gradient_through_unchanged(self):
        # STE's contract is on the LOCAL Jacobian-vector product of the op itself: for ANY incoming
        # gradient g, d(fake_quantize(x))/d(x) applied to g must equal g exactly (identity backward)
        # -- regardless of what the (necessarily different, quantized) forward VALUE was. A downstream
        # loss like (fake_quantize(x) - target)**2 is the wrong probe for this: its gradient scales
        # with (fake_quantize(x) - target), which legitimately differs from (x - target) because the
        # forward values differ -- that isn't a violation of STE, it's just the chain rule.
        rng = np.random.RandomState(1)
        w = rng.normal(size=(9, 5)).astype(np.float32)
        g = rng.normal(size=(9, 5)).astype(np.float32)

        x = torch.as_tensor(w).clone().requires_grad_(True)
        (grad_x,) = torch.autograd.grad(fake_quantize_int4(x), x, grad_outputs=torch.as_tensor(g))

        np.testing.assert_allclose(grad_x.numpy(), g, atol=1e-6)

    def test_gradient_is_nonzero_even_though_forward_is_a_step_function(self):
        # the whole point of STE: without it, d(loss)/d(input) through round() would be zero a.e.
        x = torch.as_tensor(np.linspace(-1, 1, 33).astype(np.float32)).clone().requires_grad_(True)
        loss = fake_quantize_int4(x).sum()
        loss.backward()
        self.assertTrue(torch.all(x.grad == 1.0))  # d(sum)/d(x_i) = 1 for every element, unchanged by STE


class QATWrapperTest(unittest.TestCase):
    """apply_qat / QATWrapper: composition mechanics -- weight shape, real quantization, toggling."""

    def test_apply_qat_replaces_every_linear(self):
        model = build_causal_lm(vocab=13, d_model=16, n_layer=2, n_head=2, block=8)
        n_linear_before = sum(1 for m in model.modules() if type(m).__name__ == "Linear")
        apply_qat(model)
        n_qat = sum(1 for m in model.modules() if isinstance(m, QATWrapper))
        n_linear_after = sum(1 for m in model.modules() if type(m).__name__ == "Linear")
        self.assertEqual(n_qat, n_linear_before)
        self.assertEqual(n_linear_after, n_linear_before)  # the base Linear is still there, just wrapped

    def test_forward_shape_and_gradient_flow(self):
        model = build_causal_lm(vocab=13, d_model=16, n_layer=2, n_head=2, block=8)
        apply_qat(model)
        x = torch.randint(0, 13, (4, 8)).float()
        y = torch.randint(0, 13, (4,))
        out = model(x)
        self.assertEqual(tuple(out.shape), (4, 13))
        loss = F.cross_entropy(out, y)
        loss.backward()
        wrappers = [m for m in model.modules() if isinstance(m, QATWrapper)]
        self.assertTrue(wrappers)
        for w in wrappers:
            self.assertIsNotNone(w.base.weight.grad)  # STE routed a real gradient to the base weight

    def test_set_fake_quant_enabled_toggles_real_quantization(self):
        model = build_causal_lm(vocab=13, d_model=16, n_layer=1, n_head=2, block=8)
        apply_qat(model)
        x = torch.randint(0, 13, (4, 8)).float()
        set_fake_quant_enabled(model, True)
        model.eval()
        with torch.no_grad():
            q_out = model(x)
        set_fake_quant_enabled(model, False)
        with torch.no_grad():
            fp_out = model(x)
        self.assertFalse(torch.allclose(q_out, fp_out))  # quantized vs fp32 forward genuinely differ

    def test_only_wraps_linear_not_embedding(self):
        model = build_causal_lm(vocab=13, d_model=16, n_layer=1, n_head=2, block=8)
        apply_qat(model)
        self.assertEqual(type(model.tok).__name__, "Embedding")  # untouched -- QAT here targets Linear only


def _corpus(block: int):
    text = (
        "the model is the message. a small model with sharp weights beats a big model with soft "
        "weights. a tight model spends computation where it earns its keep. "
    ) * 30
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(len(ids) * 0.8)
    train_ids, val_ids = ids[:split], ids[split:]

    def windows(seq):
        n = len(seq) - block
        xs = np.stack([seq[i : i + block] for i in range(n)]).astype(np.float32)
        ys = np.array([seq[i + block] for i in range(n)], dtype=np.int64)
        return torch.as_tensor(xs), torch.as_tensor(ys)

    return len(chars), windows(train_ids), windows(val_ids)


def _train(model, x, y, *, steps, lr, seed):
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return model


def _eval_ce(model, x, y) -> float:
    model.eval()
    with torch.no_grad():
        return float(F.cross_entropy(model(x), y))


def _quantize_linears_in_place(model, *, bits: int = 4):
    """Real PTQ applied to every Linear-shaped weight (the same math QAT's STE forward uses) --
    the eval-time int4 model for BOTH the QAT-trained and the PTQ-trained module, so the comparison
    isolates the training procedure, not the eval-time quantizer. Returns the (module, original
    weight) pairs so the caller can restore fp32 weights afterward."""
    saved = []
    for m in model.modules():
        base = m.base if isinstance(m, QATWrapper) else (m if type(m).__name__ == "Linear" else None)
        if base is None or any(base is b for b, _ in saved):
            continue
        w = base.weight.detach().cpu().numpy().astype(np.float64)
        wq, scale = quantize_dequantize_array(w, bits=bits)
        dq = torch.as_tensor((wq.astype(np.float64) * scale), dtype=base.weight.dtype)
        saved.append((base, base.weight.data.clone()))
        base.weight.data.copy_(dq)
    return saved


def _restore(saved) -> None:
    for base, w in saved:
        base.weight.data.copy_(w)


class QATBeatsPTQTest(unittest.TestCase):
    """The acceptance criterion: QAT int4 beats PTQ int4 at matched size on fixtures."""

    @classmethod
    def setUpClass(cls):
        block = 16
        vocab, (train_x, train_y), (val_x, val_y) = _corpus(block)
        cls.train_x, cls.train_y, cls.val_x, cls.val_y = train_x, train_y, val_x, val_y

        torch.manual_seed(0)
        base = build_causal_lm(vocab=vocab, d_model=32, n_layer=2, n_head=4, block=block)
        init_state = copy.deepcopy(base.state_dict())

        # same architecture, same param count, same init, same steps -- only the training procedure differs.
        cls.qat_model = build_causal_lm(vocab=vocab, d_model=32, n_layer=2, n_head=4, block=block)
        cls.qat_model.load_state_dict(init_state)
        apply_qat(cls.qat_model)  # every Linear now fake-quantizes to int4 on every forward call
        _train(cls.qat_model, train_x, train_y, steps=250, lr=3e-3, seed=1)

        cls.ptq_model = build_causal_lm(vocab=vocab, d_model=32, n_layer=2, n_head=4, block=block)
        cls.ptq_model.load_state_dict(init_state)
        _train(cls.ptq_model, train_x, train_y, steps=250, lr=3e-3, seed=1)  # plain fp32 training

    def test_qat_int4_beats_ptq_int4_at_matched_size(self):
        # QAT model: full precision weights it actually trained -> real int4 quantize at eval time.
        set_fake_quant_enabled(self.qat_model, False)
        saved_qat = _quantize_linears_in_place(self.qat_model, bits=4)
        qat_int4_ce = _eval_ce(self.qat_model, self.val_x, self.val_y)
        _restore(saved_qat)

        # PTQ model: normal fp32 training, quantized only now, at the end.
        saved_ptq = _quantize_linears_in_place(self.ptq_model, bits=4)
        ptq_int4_ce = _eval_ce(self.ptq_model, self.val_x, self.val_y)
        _restore(saved_ptq)

        print(f"\n[qat_test] held-out int4 cross-entropy -- QAT: {qat_int4_ce:.4f}  PTQ: {ptq_int4_ce:.4f}")
        self.assertLess(qat_int4_ce, ptq_int4_ce)  # QAT is measurably better at real int4
        self.assertLess(qat_int4_ce, 0.97 * ptq_int4_ce)  # "measurably" -- at least a 3% relative margin

    def test_qat_full_precision_eval_is_not_wrecked(self):
        # sanity check: the fake-quant noise during training shouldn't wreck learning at fp32 either.
        set_fake_quant_enabled(self.qat_model, False)  # QAT model's OWN real fp32 weights, no quantization
        qat_fp32_ce = _eval_ce(self.qat_model, self.val_x, self.val_y)
        ptq_fp32_ce = _eval_ce(self.ptq_model, self.val_x, self.val_y)  # normally-trained model, fp32

        print(f"[qat_test] held-out fp32 cross-entropy -- QAT: {qat_fp32_ce:.4f}  PTQ: {ptq_fp32_ce:.4f}")
        # this fixture is small and heavily overfit (both losses are near-zero nats, vs. a log(vocab)
        # ~ 3.3 random baseline), so a *relative* ratio is unstable near zero -- use an absolute bound
        # instead: QAT learned well on its own terms (far below random) and isn't far behind PTQ's fp32.
        random_baseline_ce = float(np.log(len(set(self.val_y.numpy().tolist()))))
        self.assertLess(qat_fp32_ce, 0.1 * random_baseline_ce)  # QAT alone: real learning, not noise
        self.assertLess(qat_fp32_ce - ptq_fp32_ce, 0.05)  # and it isn't far behind normally-trained fp32


class QATComposesWithGradLeafTest(unittest.TestCase):
    """A QAT-wrapped module drops into GradLeaf's existing fit/estimate interface unmodified -- no
    changes to mixle/models/grad_leaf.py, exactly the "compose via wrapping" contract this module's
    docstring claims. This is the mechanism J4 (distillation students) and F1 (a real distributed
    trainer), once they exist, would also ride: both are separate, not-yet-built roadmap items, so
    this test proves the composition point generically instead of depending on either."""

    class _LMLogDensity(torch.nn.Module):
        """log_density(row) for a GradLeaf leaf: row = [context tokens..., next token], one array
        per observation -- the shape GradLeaf's default encoder already produces (seq_encode just
        concatenates rows), so no custom encoder is needed either."""

        def __init__(self, causal_lm, block: int):
            super().__init__()
            self.lm = causal_lm
            self.block = int(block)

        def log_density(self, rows):
            ctx = rows[:, : self.block]
            y = rows[:, self.block].long()
            logits = self.lm(ctx)
            return -F.cross_entropy(logits, y, reduction="none")

    def test_qat_wrapped_transformer_trains_through_grad_leaf(self):
        block = 8
        vocab, (train_x, train_y), _ = _corpus(block)
        rows = np.concatenate([train_x.numpy(), train_y.numpy().reshape(-1, 1).astype(np.float32)], axis=1)
        rows = rows[:120]  # keep the EM loop fast

        torch.manual_seed(0)
        lm = build_causal_lm(vocab=vocab, d_model=16, n_layer=1, n_head=2, block=block)
        apply_qat(lm)  # QAT hooked in BEFORE the module ever reaches GradLeaf -- no grad_leaf.py changes
        module = self._LMLogDensity(lm, block)

        before_ll = float(np.mean(GradLeaf(module).seq_log_density(rows)))
        fitted = optimize(list(rows), GradLeaf(module, m_steps=40, lr=3e-3), max_its=3, out=None)
        after_ll = float(np.mean(fitted.seq_log_density(rows)))

        self.assertTrue(any(isinstance(m, QATWrapper) for m in fitted.module.lm.modules()))
        self.assertGreater(after_ll, before_ll)  # the QAT-wrapped module actually learned, through GradLeaf


if __name__ == "__main__":
    unittest.main()
