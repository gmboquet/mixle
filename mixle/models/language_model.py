"""``LM`` -- a declarative autoregressive language model tying the frontier-LLM stack into one usable object.

A causal Transformer trained on a token stream::

    lm = LM(vocab=V, d_model=256, n_layer=6, n_head=8, block=128)
    lm.fit(token_ids, epochs=3, batch_size=64, device="mps")          # pretrain (single process)
    lm.fit(token_ids, distributed=True, precision="bf16")             # or distributed under torchrun (FSDP2 on CUDA)
    text = lm.generate(prompt_ids, n=200, temperature=0.8)            # autoregressive sampling
    nll  = lm.nll(held_out_ids)                                       # bits/token on held-out data

``fit`` runs the non-buffering streaming estimator (``mixle.models.streaming_transformer_leaf``); with
``distributed=True`` it dispatches through ``StreamingTokenEncodedData`` (per-rank shard, in-backward all-reduce,
FSDP2/ZeRO-3 + bf16 + DCP on CUDA). The multi-stage pipeline (CPT-with-EWC, SFT loss-mask, DPO) is provided by
``mixle.models.continual`` / the responsibility-weight channel / ``mixle.models.dpo_leaf``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _torch() -> Any:
    import torch

    return torch


class LM:
    """A causal-Transformer language model with a small declarative surface: ``fit`` / ``generate`` / ``nll``."""

    def __init__(
        self,
        vocab: int,
        *,
        d_model: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        block: int = 128,
        device: str = "cpu",
        embedding: Any = None,
    ) -> None:
        from mixle.models.transformer import build_causal_lm

        self.vocab = int(vocab)
        self.block = int(block)
        self.device = device
        # embedding=CategoricalEmbedding ties one word embedding across LMs (e.g. a mixture's per-cluster experts)
        self.module = build_causal_lm(self.vocab, d_model, n_layer, n_head, self.block, embedding=embedding)

    def fit(
        self,
        token_ids: Any,
        *,
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 3e-3,
        distributed: bool = False,
        precision: str = "fp32",
        shuffle: bool = True,
    ) -> LM:
        """Pretrain (or continue) on a token-id array via the streaming estimator; the corpus is never buffered."""
        if distributed:
            from mixle.models.streaming_transformer_leaf import StreamingTransformerLeafEstimator
            from mixle.stats.compute.sequence import seq_estimate
            from mixle.utils.parallel.torch_neural import StreamingTokenEncodedData

            handle = StreamingTokenEncodedData(
                token_ids, block=self.block, batch_size=batch_size, epochs=epochs, shuffle=shuffle, precision=precision
            )
            est = StreamingTransformerLeafEstimator(self.module, lr=lr, device=self.device)
            self.module = seq_estimate(handle, est, None).module
        else:
            from mixle.data.stream_token_source import stream_token_source
            from mixle.models.streaming_transformer_leaf import stream_fit

            src = stream_token_source(
                token_ids, block=self.block, batch_size=batch_size, epochs=epochs, shuffle=shuffle
            )
            self.module = stream_fit(self.module, src, lr=lr, device=self.device)[0].module
        return self

    def generate(
        self, prompt_ids: Any, n: int = 200, *, temperature: float = 1.0, greedy: bool = False, seed: int = 0
    ) -> list:
        """Autoregressively extend ``prompt_ids`` by ``n`` tokens (greedy, or temperature-sampled)."""
        torch = _torch()
        rng = np.random.RandomState(seed)
        self.module.to(self.device).eval()
        w = [int(t) for t in prompt_ids]
        out = list(w)
        for _ in range(int(n)):
            win = w[-self.block :]
            if len(win) < self.block:
                win = [0] * (self.block - len(win)) + win
            with torch.no_grad():
                logits = self.module(torch.as_tensor([win], dtype=torch.float32).to(self.device))[0].cpu().numpy()
            if greedy:
                nxt = int(logits.argmax())
            else:
                p = np.exp((logits - logits.max()) / max(temperature, 1e-6))
                p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            w.append(nxt)
            out.append(nxt)
        self.module.train()
        return out

    def nll(self, token_ids: Any) -> float:
        """Mean next-token negative log-likelihood (nats/token) on a token-id array."""
        from mixle.models.streaming_transformer_leaf import StreamingTransformerLeaf

        ids = np.asarray(token_ids)
        ctx = np.stack([ids[i : i + self.block] for i in range(len(ids) - self.block)]).astype("float32")
        leaf = StreamingTransformerLeaf(self.module, self.device)
        return float(-np.mean(leaf.seq_log_density((ctx, ids[self.block :]))))
