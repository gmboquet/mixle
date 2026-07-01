"""A causal decoder-only Transformer as a torch module -- the engine behind the declarative AR-LM surface.

Built lazily by the :class:`mixle.ppl.core.Transformer` predictor token. ``forward(x)`` takes a ``(batch, block)``
context of token ids (accepted as float so it rides the ``SoftmaxNeuralLeaf`` float path, cast to long inside)
and returns next-token logits ``(batch, vocab)`` from the last position. So
``Categorical(logits=Transformer(out=V))`` is *exactly* next-token prediction ``p(token | context)``, fit by the
standard ``estimate()`` loop whose cross-entropy is ``-log p`` -- no new training machinery.

Attention is ``F.scaled_dot_product_attention`` (the FlashAttention dispatch on CUDA). At frontier scale the
same module is what a vendored TorchTitan/Megatron trainer shards (FSDP2/TP/PP); here it runs single-process.
"""

from __future__ import annotations

from typing import Any


def build_causal_lm(
    vocab: int, d_model: int = 128, n_layer: int = 3, n_head: int = 4, block: int = 64, embedding: Any = None
) -> Any:
    """Build a causal decoder-only Transformer LM (token+pos embeddings, pre-norm blocks, weight-tied head).

    ``embedding`` optionally injects a *shared* token ``nn.Embedding`` (``vocab x d_model``) to use in place of a
    fresh one -- so several language models can tie the same word embedding and train it jointly (the weight-tied
    head follows it). Its shape must match ``(vocab, d_model)``.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from mixle.models.embedding import resolve_embedding

    embedding = resolve_embedding(embedding, vocab, d_model)  # SharedEmbedding | nn.Embedding | None -> module

    class CausalAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.h = n_head
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x: Any) -> Any:
            b, t, d = x.shape
            qkv = self.qkv(x).reshape(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
            o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)  # FlashAttention seam (CUDA)
            return self.proj(o.transpose(1, 2).reshape(b, t, d))

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.attn = CausalAttention()
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

        def forward(self, x: Any) -> Any:
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class CausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = embedding if embedding is not None else nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(block, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.head.weight = self.tok.weight  # weight tying

        def forward(self, x: Any) -> Any:
            x = x.long()
            t = x.shape[1]
            pos = torch.arange(t, device=x.device)
            h = self.tok(x) + self.pos(pos)[None, :, :]
            for blk in self.blocks:
                h = blk(h)
            return self.head(self.ln(h))[:, -1]  # next-token logits from the last position -> (batch, vocab)

    return CausalLM()
