"""E10 falsification: does a transformer TRAINED with product-quantized keys live up?

Task: associative recall (16 key-value pairs, then a query key; predict its value) -- learnable
only through attention retrieval, so it is the sharpest cheap kill test for a key representation.

Arms (matched architecture/params/steps/seeds):
  A  dense attention (continuous keys)                       -- baseline
  B  PQ-STE keys: per-head keys snapped to learned per-block codebooks (VQ-VAE commitment loss),
     attention computed densely over quantized keys          -- quality cost of quantization
  C  cell aggregation over B's trained model at inference    -- exactness + the O(cells) speed win

Kill criteria (pre-stated):
  K1  B reaches >= 90% of A's final recall accuracy
  K2  C's attention output == B's dense forward within fp32 tolerance (1e-4)
  K3  C beats dense attention >= 5x per query at 64k context using B's trained code distribution
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cpu"  # deterministic, and the models are tiny
N_KEYS, N_VALS, N_PAIRS = 32, 32, 16
VOCAB = N_KEYS + N_VALS + 1  # +1 query marker
SEQ = 2 * N_PAIRS + 2
D_MODEL, N_HEAD, D_HEAD, N_LAYER = 64, 2, 32, 2
BLOCKS, SUB, KCODES = 4, 8, 16  # d_head = 4 blocks x 8 dims, 16 codes/block -> 16^4 cells/head


def make_batch(bs, rng):
    keys = np.stack([rng.choice(N_KEYS, N_PAIRS, replace=False) for _ in range(bs)])
    vals = rng.randint(0, N_VALS, (bs, N_PAIRS))
    q_idx = rng.randint(0, N_PAIRS, bs)
    x = np.zeros((bs, SEQ), dtype=np.int64)
    x[:, 0 : 2 * N_PAIRS : 2] = keys
    x[:, 1 : 2 * N_PAIRS + 1 : 2] = vals + N_KEYS
    x[:, 2 * N_PAIRS] = N_KEYS + N_VALS  # query marker
    x[:, 2 * N_PAIRS + 1] = keys[np.arange(bs), q_idx]
    y = vals[np.arange(bs), q_idx]  # value id to recall
    return torch.as_tensor(x), torch.as_tensor(y)


class PQ(nn.Module):
    """Per-block codebooks with straight-through quantization + VQ commitment losses."""

    def __init__(self):
        super().__init__()
        self.codebooks = nn.Parameter(torch.randn(BLOCKS, KCODES, SUB) / np.sqrt(D_HEAD))

    def forward(self, k):  # k: (..., D_HEAD)
        shape = k.shape
        kb = k.reshape(*shape[:-1], BLOCKS, SUB)
        d = ((kb.unsqueeze(-2) - self.codebooks) ** 2).sum(-1)  # (..., BLOCKS, KCODES)
        idx = d.argmin(-1)
        quant = torch.take_along_dim(self.codebooks.expand(*shape[:-1], -1, -1, -1),
                                     idx[..., None, None], dim=-2).squeeze(-2)
        commit = F.mse_loss(kb, quant.detach()) + 0.25 * F.mse_loss(quant, kb.detach())
        k_q = kb + (quant - kb).detach()  # straight-through
        return k_q.reshape(shape), idx, commit


class Block(nn.Module):
    def __init__(self, quantize):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))
        self.pq = PQ() if quantize else None

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
    def __init__(self, quantize):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(SEQ, D_MODEL)
        self.blocks = nn.ModuleList([Block(quantize) for _ in range(N_LAYER)])
        self.head = nn.Linear(D_MODEL, N_VALS)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))[None]
        commit = h.new_zeros(())
        for blk in self.blocks:
            h, c = blk(h)
            commit = commit + c
        return self.head(h[:, -1]), commit


def train(quantize, tag, steps=2500):
    torch.manual_seed(1)
    model = Model(quantize)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.RandomState(0)
    acc = 0.0
    for step in range(steps):
        x, y = make_batch(64, rng)
        logits, commit = model(x)
        loss = F.cross_entropy(logits, y) + commit
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                xt, yt = make_batch(2000, np.random.RandomState(99))
                acc = float((model(xt)[0].argmax(-1) == yt).float().mean())
            print(f"  {tag} step {step+1:5d}  loss={float(loss):.3f}  held-out recall={acc:.3f}")
    return model, acc


print("== K1: does quantized-key training retain retrieval? ==")
_, acc_dense = train(False, "A dense    ")
model_q, acc_quant = train(True, "B quantized")
print(f"K1: quantized/dense recall = {acc_quant:.3f}/{acc_dense:.3f} = {acc_quant/max(acc_dense,1e-9):.1%} "
      f"-> {'PASS' if acc_quant >= 0.9 * acc_dense else 'FAIL'}")

# ---- K2 + K3: cell aggregation on the TRAINED quantized model -------------------------------
print("\n== K2/K3: cell attention on the trained model ==")
blk = model_q.blocks[0]
with torch.no_grad():
    # a long context of random tokens through the trained embeddings -> real trained K/V streams
    for n_ctx in (4096, 16384, 65536):
        rng = np.random.RandomState(5)
        toks = torch.as_tensor(rng.randint(0, VOCAB, (1, n_ctx)))
        h = model_q.tok(toks) + model_q.pos(torch.arange(SEQ))[None, :1]  # pos broadcast: content keys
        q_, k_, v_ = blk.qkv(blk.ln1(h)).split(D_MODEL, dim=-1)
        k_ = k_.view(1, n_ctx, N_HEAD, D_HEAD).transpose(1, 2)
        v_ = v_.view(1, n_ctx, N_HEAD, D_HEAD).transpose(1, 2)
        kq, idx, _ = blk.pq(k_)
        head = 0
        K = kq[0, head]; V = v_[0, head]; codes = idx[0, head]  # (n, BLOCKS)
        cell_id = (codes * (KCODES ** torch.arange(BLOCKS))).sum(-1)
        uniq, inv = torch.unique(cell_id, return_inverse=True)
        counts = torch.bincount(inv, minlength=len(uniq)).double()
        vbar = torch.zeros(len(uniq), D_HEAD, dtype=torch.float64)
        vbar.index_add_(0, inv, V.double())
        vbar /= counts[:, None]
        kcell = torch.zeros(len(uniq), D_HEAD, dtype=torch.float64)
        kcell.index_add_(0, inv, K.double())
        kcell /= counts[:, None]  # cell key = the (shared) quantized key; mean of identical vectors

        query = q_.view(1, n_ctx, N_HEAD, D_HEAD)[0, -1, head].double()
        s_dense = (K.double() @ query) / np.sqrt(D_HEAD); s_dense -= s_dense.max()
        w = torch.exp(s_dense); out_dense = (w @ V.double()) / w.sum()
        s_cell = (kcell @ query) / np.sqrt(D_HEAD); s_cell -= s_cell.max()
        wc = counts * torch.exp(s_cell); out_cell = (wc @ vbar) / wc.sum()
        err = float((out_dense - out_cell).abs().max())

        reps = 50
        t0 = time.perf_counter()
        for _ in range(reps):
            s = (K @ query.float()) / np.sqrt(D_HEAD); s = s - s.max(); w = torch.exp(s); (w @ V) / w.sum()
        td = (time.perf_counter() - t0) / reps * 1e3
        kcf, vbf, cf = kcell.float(), vbar.float(), counts.float()
        t0 = time.perf_counter()
        for _ in range(reps):
            s = (kcf @ query.float()) / np.sqrt(D_HEAD); s = s - s.max(); w = cf * torch.exp(s); (w @ vbf) / w.sum()
        tc = (time.perf_counter() - t0) / reps * 1e3
        print(f"  n={n_ctx:6d}  occupied cells={len(uniq):5d}  K2 err={err:.2e}  "
              f"dense {td:7.3f}ms  cell {tc:7.3f}ms  speedup {td/tc:5.1f}x")
