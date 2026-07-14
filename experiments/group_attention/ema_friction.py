"""E10 acceptance: do EMA codebooks cut the VQ optimization friction to <= 1.5x dense?

The falsification measured the friction (RESULTS.md): the gradient-VQ quantized arm needed ~2x the
dense arm's step budget to break through on associative recall, with EMA codebooks named as the
standard untried remedy. This experiment quantifies the remedy on the SHIPPED mechanism's
ProductQuantizer (mixle.experimental.quantized_key_attention), not a reimplementation.

Task: associative recall, identical to train_quantized_keys.py (16 key-value pairs then a query;
learnable only through attention retrieval). Arms share architecture/params/seed/batch stream:

  A  dense attention                                    -- the budget yardstick
  B  PQ straight-through, codebook_update="gradient"    -- the measured ~2x friction
  C  PQ straight-through, codebook_update="ema"         -- the remedy under test

Acceptance (pre-stated): budget-to-target(C) <= 1.5 x budget-to-target(A), where the target is
held-out recall >= 0.95 evaluated every 100 steps (batch 64; budget = examples seen). Kill: C
needs more than 1.5x -- record the ratio honestly and keep "gradient" as the default.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mixle.experimental.quantized_key_attention import ProductQuantizer

torch.manual_seed(0)
N_KEYS, N_VALS, N_PAIRS = 32, 32, 16
VOCAB = N_KEYS + N_VALS + 1
SEQ = 2 * N_PAIRS + 2
D_MODEL, N_HEAD, D_HEAD, N_LAYER = 64, 2, 32, 2
BLOCKS, KCODES = 4, 16
TARGET, EVAL_EVERY, MAX_STEPS, BATCH = 0.95, 100, 6000, 64


def make_batch(bs, rng):
    keys = np.stack([rng.choice(N_KEYS, N_PAIRS, replace=False) for _ in range(bs)])
    vals = rng.randint(0, N_VALS, (bs, N_PAIRS))
    q_idx = rng.randint(0, N_PAIRS, bs)
    x = np.zeros((bs, SEQ), dtype=np.int64)
    x[:, 0 : 2 * N_PAIRS : 2] = keys
    x[:, 1 : 2 * N_PAIRS + 1 : 2] = vals + N_KEYS
    x[:, 2 * N_PAIRS] = N_KEYS + N_VALS
    x[:, 2 * N_PAIRS + 1] = keys[np.arange(bs), q_idx]
    return torch.as_tensor(x), torch.as_tensor(vals[np.arange(bs), q_idx])


class Block(nn.Module):
    def __init__(self, pq_kwargs):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))
        self.pq = ProductQuantizer(D_HEAD, n_blocks=BLOCKS, codes_per_block=KCODES, **pq_kwargs) if pq_kwargs else None

    def forward(self, h):
        b, t, _ = h.shape
        q, k, v = self.qkv(self.ln1(h)).split(D_MODEL, dim=-1)
        q = q.view(b, t, N_HEAD, D_HEAD).transpose(1, 2)
        k = k.view(b, t, N_HEAD, D_HEAD).transpose(1, 2)
        v = v.view(b, t, N_HEAD, D_HEAD).transpose(1, 2)
        commit = h.new_zeros(())
        if self.pq is not None:
            k, _, commit = self.pq(k)
        att = (q @ k.transpose(-2, -1)) / np.sqrt(D_HEAD)
        att = att.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
        h = h + self.proj((att.softmax(-1) @ v).transpose(1, 2).reshape(b, t, D_MODEL))
        h = h + self.mlp(self.ln2(h))
        return h, commit


class Model(nn.Module):
    def __init__(self, pq_kwargs):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(SEQ, D_MODEL)
        self.blocks = nn.ModuleList([Block(pq_kwargs) for _ in range(N_LAYER)])
        self.head = nn.Linear(D_MODEL, N_VALS)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))[None]
        commit = h.new_zeros(())
        for blk in self.blocks:
            h, c = blk(h)
            commit = commit + c
        return self.head(h[:, -1]), commit


def budget_to_target(pq_kwargs, tag):
    torch.manual_seed(1)
    model = Model(pq_kwargs)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.RandomState(0)
    xt, yt = make_batch(2000, np.random.RandomState(99))
    for step in range(1, MAX_STEPS + 1):
        x, y = make_batch(BATCH, rng)
        logits, commit = model(x)
        loss = F.cross_entropy(logits, y) + commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                acc = float((model(xt)[0].argmax(-1) == yt).float().mean())
            model.train()
            if step % 500 == 0 or acc >= TARGET:
                print(f"  {tag:22s} step {step:5d}  examples {step * BATCH:7d}  recall {acc:.3f}")
            if acc >= TARGET:
                return step * BATCH, acc
    return None, acc


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] not in ("--sweep-decay",):
        sys.exit(f"unknown argument {sys.argv[1]!r}; supported: --sweep-decay [dense_budget]")
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep-decay":
        # tune the REMEDY's own hyperparameter, never the task: dense budget from the main run
        dense_budget = int(sys.argv[2]) if len(sys.argv) > 2 else 83_200
        for decay in (0.95, 0.9, 0.8):
            b, _ = budget_to_target({"codebook_update": "ema", "ema_decay": decay}, f"C pq-ema d={decay}")
            ratio = (b / dense_budget) if b else float("inf")
            print(f"  decay {decay}: {b} examples -> {ratio:.2f}x dense")
        sys.exit(0)

    print(f"== EMA-codebook friction acceptance: budget to held-out recall >= {TARGET} ==")
    budget_dense, _ = budget_to_target(None, "A dense")
    budget_grad, _ = budget_to_target({"codebook_update": "gradient"}, "B pq-gradient")
    budget_ema, _ = budget_to_target({"codebook_update": "ema"}, "C pq-ema")

    print("\nbudgets (examples):", {"dense": budget_dense, "gradient": budget_grad, "ema": budget_ema})
    if budget_dense and budget_ema:
        ratio_grad = (budget_grad / budget_dense) if budget_grad else float("inf")
        ratio_ema = budget_ema / budget_dense
        verdict = "PASS" if ratio_ema <= 1.5 else "FAIL"
        print(f"friction: gradient {ratio_grad:.2f}x  ema {ratio_ema:.2f}x  (acceptance <= 1.50x) -> {verdict}")
    else:
        print("FAIL: an arm never reached the target inside the step cap")
