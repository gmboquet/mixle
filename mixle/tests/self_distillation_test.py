"""Acceptance tests for mixle.models.self_distillation (roadmap J3: self-distillation during training).

Each test below is an acceptance criterion from the roadmap item, not just a smoke test:
    1. EMA-teacher weight-averaging math is exactly correct against a hand-computed EMA.
    2. stochastic_depth_forward produces a DIFFERENT partial-depth output when blocks are dropped, and an
       IDENTICAL one at drop_prob=0 (the degenerate-case sanity check).
    3. THE core acceptance criterion: J2's not-yet-built "ladder" is substituted, per the roadmap note,
       with G3's already-built `coarsen()` (PR #151) as the compressibility proxy. Train a small CausalLM
       two ways for the SAME number of steps -- plain vs. train_with_self_distillation -- then run
       `coarsen()` on BOTH at the SAME divergence budget and trust region, and confirm the J3-trained
       checkpoint compresses measurably better (lower total KL for the same accepted depth cut).
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.data.stream_token_source import stream_token_source
from mixle.models.coarsening import coarsen
from mixle.models.moment_propagation import GaussianLaw
from mixle.models.self_distillation import (
    EMATeacher,
    consistency_loss,
    stochastic_depth_forward,
    train_with_self_distillation,
)
from mixle.models.transformer import build_causal_lm

# --------------------------------------------------------------------------------------------------------
# shared synthetic-data helper: a small Markov-ish token stream, so there is real structure to learn
# --------------------------------------------------------------------------------------------------------


def _synthetic_token_stream(n: int, vocab: int, seed: int) -> np.ndarray:
    """A simple order-1 Markov chain over `vocab` tokens -- learnable structure (next-token distribution
    depends on the current token), so cross-entropy training has a real target to fit."""
    rng = np.random.RandomState(seed)
    trans = rng.dirichlet(np.full(vocab, 0.3), size=vocab)  # (vocab, vocab) row-stochastic transition matrix
    ids = np.zeros(n, dtype=np.int64)
    ids[0] = rng.randint(vocab)
    for i in range(1, n):
        ids[i] = rng.choice(vocab, p=trans[ids[i - 1]])
    return ids


def _empirical_input_law(model, token_ids: np.ndarray, block: int, rng: np.random.Generator) -> GaussianLaw:
    """The residual-stream distribution ENTERING model.blocks (i.e. after tok+pos embedding), estimated
    empirically from real synthetic contexts run through `model`'s OWN (trained) embedding -- exactly what
    `coarsen`'s `input_law` argument documents it should represent, computed the same way for both models
    under comparison so the divergence budget means the same thing for each.
    """
    n = len(token_ids) - block
    idx = rng.choice(n, size=min(256, n), replace=False)
    ctx = np.stack([token_ids[i : i + block] for i in idx])
    x = torch.as_tensor(ctx, dtype=torch.float32)
    with torch.no_grad():
        t = x.shape[1]
        pos = torch.arange(t)
        h = model.tok(x.long()) + model.pos(pos)[None, :, :]
    flat = h.reshape(-1, h.shape[-1]).numpy().astype(np.float64)
    mu = flat.mean(axis=0)
    covar = np.cov(flat, rowvar=False) + 1e-6 * np.eye(flat.shape[-1])
    return GaussianLaw(mu=mu, covar=covar)


class EMATeacherCorrectnessTest(unittest.TestCase):
    """Acceptance criterion: "EMA teacher's weight-averaging math is exactly correct -- a direct numerical
    check across a couple of update steps against a hand-computed EMA"."""

    def test_ema_update_matches_hand_computed_average(self):
        torch.manual_seed(0)
        model = build_causal_lm(vocab=11, d_model=8, n_layer=2, n_head=2, block=8)
        decay = 0.9
        teacher = EMATeacher(model, decay=decay)

        # hand-computed EMA of every tensor, tracked independently in plain torch/numpy
        expected = {k: v.detach().clone() for k, v in model.state_dict().items()}

        for step in range(3):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.05)  # simulate one optimizer step
            teacher.update(model)
            student_state = model.state_dict()
            for k in expected:
                expected[k] = decay * expected[k] + (1.0 - decay) * student_state[k].detach()

        teacher_state = teacher.ema_model.state_dict()
        for k, exp_v in expected.items():
            torch.testing.assert_close(teacher_state[k], exp_v, atol=1e-6, rtol=1e-5)

    def test_ema_teacher_is_frozen_and_forward_works(self):
        torch.manual_seed(1)
        model = build_causal_lm(vocab=11, d_model=8, n_layer=2, n_head=2, block=8)
        teacher = EMATeacher(model, decay=0.99)
        for p in teacher.ema_model.parameters():
            self.assertFalse(p.requires_grad)

        x = torch.randint(0, 11, (4, 8)).float()
        out = teacher.forward(x)
        self.assertEqual(tuple(out.shape), (4, 11))
        # predict is an alias for forward
        out2 = teacher.predict(x)
        torch.testing.assert_close(out, out2)


class StochasticDepthCorrectnessTest(unittest.TestCase):
    """Acceptance criterion: "verify stochastic_depth_forward actually produces a DIFFERENT (partial)
    output when blocks are dropped vs. the full-depth output, and that at drop_prob=0 the two outputs are
    IDENTICAL"."""

    def test_drop_prob_zero_gives_identical_outputs(self):
        torch.manual_seed(0)
        model = build_causal_lm(vocab=13, d_model=16, n_layer=4, n_head=2, block=8)
        model.eval()
        x = torch.randint(0, 13, (3, 8)).float()
        full, partial = stochastic_depth_forward(model, x, drop_prob=0.0)
        torch.testing.assert_close(full, partial, atol=0.0, rtol=0.0)

    def test_dropping_blocks_changes_the_output(self):
        torch.manual_seed(2)
        model = build_causal_lm(vocab=13, d_model=16, n_layer=6, n_head=2, block=8)
        model.eval()
        x = torch.randint(0, 13, (5, 8)).float()
        gen = torch.Generator().manual_seed(0)
        full, partial = stochastic_depth_forward(model, x, drop_prob=0.9, generator=gen)
        self.assertFalse(torch.allclose(full, partial))
        # sanity: full-depth output itself matches a bare model(x) call
        with torch.no_grad():
            direct = model(x)
        torch.testing.assert_close(full, direct)

    def test_consistency_loss_is_zero_for_identical_inputs_and_positive_otherwise(self):
        torch.manual_seed(3)
        a = torch.randn(4, 5)
        b = a.clone()
        self.assertAlmostEqual(float(consistency_loss(a, b, mode="mse")), 0.0, delta=1e-8)
        c = torch.randn(4, 5)
        self.assertGreater(float(consistency_loss(a, c, mode="mse")), 0.0)
        self.assertGreaterEqual(float(consistency_loss(a, c, mode="kl")), 0.0)


class J3CompressibilityAcceptanceTest(unittest.TestCase):
    """THE acceptance criterion (substituting G3's coarsen() for the not-yet-built J2 ladder, per the
    roadmap note): a J3-trained checkpoint must compress better under coarsen() at an EQUAL divergence
    budget/trust region than a plainly-trained checkpoint of the identical architecture, trained for the
    SAME number of steps on the SAME data.
    """

    def test_j3_trained_checkpoint_compresses_better_than_plain_under_g3_coarsen(self):
        vocab, d_model, n_head, n_layer, block = 17, 24, 3, 6, 16
        steps = 250
        batch_size = 16
        seed = 0

        token_ids = _synthetic_token_stream(n=20_000, vocab=vocab, seed=seed)

        def fresh_model():
            torch.manual_seed(seed)
            return build_causal_lm(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)

        # (a) plain training: identical architecture, identical data, identical step count -- ordinary
        # cross-entropy only, no self-distillation.
        plain_model = fresh_model()
        opt = torch.optim.Adam(plain_model.parameters(), lr=3e-3)
        plain_model.train()
        src = stream_token_source(token_ids, block=block, batch_size=batch_size, epochs=1, shuffle=True, seed=seed)
        for step, (ctx, nxt) in enumerate(src):
            if step >= steps:
                break
            x = torch.as_tensor(ctx, dtype=torch.float32)
            y = torch.as_tensor(nxt, dtype=torch.long)
            logits = plain_model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        plain_model.eval()

        # (b) J3 training: same architecture/data/step count, with EMA-teacher + stochastic-depth
        # self-distillation added on top of the same per-step cross-entropy.
        j3_model = fresh_model()
        j3_model = train_with_self_distillation(
            j3_model,
            data=lambda: stream_token_source(
                token_ids, block=block, batch_size=batch_size, epochs=1, shuffle=True, seed=seed
            ),
            steps=steps,
            ema_decay=0.99,
            drop_prob=0.3,
            consistency_weight=0.5,
            lr=3e-3,
            seed=seed,
        )
        j3_model.eval()

        # G3's coarsen(), called directly as the compressibility proxy for the not-yet-built J2 ladder --
        # SAME budget, trust region, and (per-model, but identically-constructed) input_law for both.
        rng = np.random.default_rng(seed)
        budget = 8.0
        trust_region = 8.0

        law_plain = _empirical_input_law(plain_model, token_ids, block, rng)
        law_j3 = _empirical_input_law(j3_model, token_ids, block, rng)

        result_plain = coarsen(plain_model, budget=budget, trust_region=trust_region, input_law=law_plain, seed=seed)
        result_j3 = coarsen(j3_model, budget=budget, trust_region=trust_region, input_law=law_j3, seed=seed)

        print(
            f"\n[J3 acceptance] plain: accepted_pairs={len(result_plain.accepted_pairs)} "
            f"total_kl={result_plain.total_kl:.6f} depth={result_plain.model.n_layer}\n"
            f"[J3 acceptance] j3:    accepted_pairs={len(result_j3.accepted_pairs)} "
            f"total_kl={result_j3.total_kl:.6f} depth={result_j3.model.n_layer}"
        )

        # Same architecture, same budget/trust region -> compare directly. The concrete metric: for the
        # SAME number of accepted merges (or more), the J3-trained model's accumulated divergence is
        # LOWER -- i.e. it needs less of the divergence budget to achieve the same (or a larger) depth cut.
        self.assertGreaterEqual(len(result_j3.accepted_pairs), len(result_plain.accepted_pairs))
        if len(result_j3.accepted_pairs) == len(result_plain.accepted_pairs):
            self.assertLess(result_j3.total_kl, result_plain.total_kl)


if __name__ == "__main__":
    unittest.main()
