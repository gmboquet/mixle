"""LLaVA-style stage-1 on toy volumes -- the existence proof.

The multimodal "stage-1" pattern, in a page: a frozen vision-ish encoder, a frozen toy LM, and a thin
TRAINABLE projection bridging the two -- fit end to end while both backbones stay bitwise frozen. Here
the "vision" is a synthetic 3-D "volume" (a toy stand-in for a scan/point-cloud/voxel grid) with a
planted structure that determines a short caption; a small frozen ``Conv3d`` stack encodes it; a small
frozen toy LM (embedding + GRU cell + output head, randomly initialized and never trained -- no
pretrained weights needed, this is about the WIRING, not the language model) scores the caption tokens
conditioned on that embedding through the projection. This mirrors ``examples/peft_lora_grad_leaf.py``'s
pattern (a causal LM's next-token log-likelihood dropped straight into ``GradLeaf.log_density``) but
swaps the real HF checkpoint for a fully synthetic, dependency-free pair of backbones so the receipt
runs in milliseconds with only ``torch`` installed.

Design note on A2 (``build_projection_leaf``, PR #127, not yet merged into this branch): that leaf is a
CONTRASTIVE/InfoNCE bridge between two embedding spaces with no ground-truth targets -- exactly the
right shape for retrieval, but not for THIS claim. Stage-1 LLaVA-style pretraining supervises the
projection with real caption tokens via teacher-forced next-token cross-entropy, i.e. a GENERATIVE
``p(caption | volume)``, which is what ``GradLeaf`` (a bare module's ``log_density(x) -> (n,)``, the
same bridge the peft example rides) gives for free once the caption-conditioned log-likelihood is
computed inside the module. So this example writes its own small inline projection rather than
depending on #127 -- it is also the more independent, mergeable choice while that PR is still open.

Run: ``python examples/multimodal_stage1_demo.py``
"""

from __future__ import annotations

import numpy as np

from mixle.inference.estimation import optimize
from mixle.models import GradLeaf

SIZE = 8  # toy volumes are SIZE x SIZE x SIZE voxels
# caption vocabulary: BLOB, STRIPE are "subject" tokens; LEFT, RIGHT, EOS complete the two-token caption
BLOB, STRIPE, LEFT, RIGHT, EOS = range(5)
VOCAB = 5
CAPTIONS = {0: (BLOB, LEFT), 1: (BLOB, RIGHT), 2: (STRIPE, EOS)}  # class -> two-token caption


def synthetic_volume(label: int, rng: np.random.RandomState) -> np.ndarray:
    """A planted structure that determines the caption: a blob in the left/right half, or a stripe."""
    vol = 0.05 * rng.randn(SIZE, SIZE, SIZE).astype("float32")
    if label in (0, 1):  # blob-left / blob-right: a small bright cube in one half of the volume
        lo, hi = (0, SIZE // 2 - 2) if label == 0 else (SIZE // 2, SIZE - 2)
        cx, cy, cz = rng.randint(lo, hi + 1), rng.randint(1, SIZE - 2), rng.randint(1, SIZE - 2)
        vol[cx : cx + 2, cy : cy + 2, cz : cz + 2] += 1.0
    else:  # stripe: a full-width bright plane at a random depth
        d = rng.randint(0, SIZE)
        vol[d, :, :] += 1.0
    return vol


def build_dataset(n_per_class: int, seed: int) -> list[np.ndarray]:
    """Each row is a flat vector: the volume's voxels followed by its two caption token ids -- the
    single homogeneous array ``GradLeaf``'s ``module.log_density(x) -> (n,)`` contract expects."""
    rng = np.random.RandomState(seed)
    rows = []
    for label in CAPTIONS:
        for _ in range(n_per_class):
            volume = synthetic_volume(label, rng)
            tokens = np.asarray(CAPTIONS[label], dtype="float32")
            rows.append(np.concatenate([volume.reshape(-1), tokens]))
    rng.shuffle(rows)
    return rows


def build_module(seed: int = 0):
    """Frozen conv encoder -> trainable projection -> frozen toy LM, all in one ``log_density`` module."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    emb_dim, hidden = 8, 16

    encoder = nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool3d(2),
        nn.Conv3d(4, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
    )  # (n, 1, S, S, S) -> (n, emb_dim)
    token_embed = nn.Embedding(VOCAB, emb_dim)
    cell = nn.GRUCell(emb_dim, hidden)
    head = nn.Linear(hidden, VOCAB)
    for module in (encoder, token_embed, cell, head):  # the backbones: frozen, forever
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    projection = nn.Sequential(nn.Linear(emb_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))  # trainable

    class Stage1CaptionLeaf(nn.Module):
        """``log_density(x) -> (n,)``: caption log-likelihood conditioned on the volume, via the
        frozen encoder -> trainable projection -> frozen toy-LM path. Owns all four pieces so a
        bitwise freeze check just walks ``named_parameters()`` by prefix after fitting."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder, self.token_embed, self.cell, self.head = encoder, token_embed, cell, head
            self.projection = projection

        def train(self, mode: bool = True) -> Stage1CaptionLeaf:
            super().train(mode)
            for module in (self.encoder, self.token_embed, self.cell, self.head):
                module.eval()  # frozen backbones never leave eval(), regardless of the M-step's train()
            return self

        def log_density(self, x):
            n = x.shape[0]
            volume = x[:, : SIZE**3].reshape(n, 1, SIZE, SIZE, SIZE)
            tokens = x[:, SIZE**3 :].long()  # (n, 2): the two caption token ids
            h0 = self.projection(self.encoder(volume))  # ONLY trainable step: embedding -> LM init state
            logits0 = self.head(h0)
            ll0 = torch.log_softmax(logits0, dim=-1)[torch.arange(n), tokens[:, 0]]
            h1 = self.cell(self.token_embed(tokens[:, 0]), h0)
            logits1 = self.head(h1)
            ll1 = torch.log_softmax(logits1, dim=-1)[torch.arange(n), tokens[:, 1]]
            return ll0 + ll1  # summed two-token caption log-likelihood == log density of the caption

    return Stage1CaptionLeaf()


def backbone_param_names(module) -> list[str]:
    return [n for n, _ in module.named_parameters() if not n.startswith("projection.")]


def main() -> None:
    import torch

    data = build_dataset(n_per_class=30, seed=0)
    module = build_module(seed=0)

    before = {n: p.detach().clone() for n, p in module.named_parameters()}

    leaf = GradLeaf(module, m_steps=200, lr=5e-2)
    stacked = np.stack(data)
    before_ll = float(np.mean(leaf.seq_log_density(stacked)))

    fitted = optimize(data, leaf, max_its=6, out=None)

    after_ll = float(np.mean(fitted.seq_log_density(stacked)))
    after = dict(fitted.module.named_parameters())

    backbone_names = backbone_param_names(module)
    proj_names = [n for n in before if n.startswith("projection.")]
    frozen_ok = all(torch.equal(before[n], after[n]) for n in backbone_names)
    moved = any(not torch.equal(before[n], after[n]) for n in proj_names)

    print(f"backbone params bitwise unchanged: {frozen_ok} ({len(backbone_names)} tensors checked)")
    print(f"projection params moved: {moved} ({len(proj_names)} tensors checked)")
    print(f"mean caption log-likelihood before fit: {before_ll:.4f}")
    print(f"mean caption log-likelihood after fit:  {after_ll:.4f}")
    assert frozen_ok, "a frozen backbone parameter changed during fit"
    assert moved, "the projection never trained"
    assert after_ll > before_ll, "the fit made no progress"
    print("OK: only the projection trained; both backbones are bitwise frozen; caption likelihood improved.")


if __name__ == "__main__":
    main()
