"""Differentiable product-of-experts fusion for structured multimodal evidence aggregation.

Dense cross-attention fuses N tokens in O(N^2). When the tokens are (conditionally) independent evidence about
a shared latent -- the common case for aggregating many partial observations (image patches, sensors, views) --
precision-weighted product-of-experts fuses them in O(N) with few parameters:
the fusion inductive bias is built in, not learned. Each expert is a diagonal
Gaussian ``N(mu_i, diag(1/prec_i))`` over the latent, and
the posterior is their normalized product::

    prec_fused = sum_i prec_i + prec_prior          # precisions add
    mu_fused   = (sum_i prec_i * mu_i) / prec_fused  # precision-weighted mean

The reference benchmark in ``examples/structured_fusion_vlm.py`` reports that
PoE fusion matches a cross-attention block's accuracy with fewer parameters and
faster training on exchangeable-evidence tasks.

Boundary condition: PoE fusion is permutation-invariant and assumes conditional independence, so it cannot
model token order or pairwise interactions. On a task that depends on a specific pair or position, attention
reaches ~0.96 while PoE sits at chance. Use structured fusion where evidence is exchangeable, and attention
where relational interactions are part of the signal.

mixle.reason's exact core (:class:`GaussianBelief`) does this fusion in closed form for *inference*; this is the
torch, end-to-end-trainable version -- the encoders that emit the experts are learned, the fusion stays exact.
Torch is imported lazily.
"""

from __future__ import annotations

from typing import Any


def fusion_flops(n_tokens: int, latent_dim: int, *, attention: bool = False) -> int:
    """Approximate multiply-adds to fuse ``n_tokens`` into one latent.

    Product-of-experts fusion is O(N*M); attention is O(N^2*M).
    """
    if attention:
        return n_tokens * n_tokens * latent_dim  # the QK^T score matrix dominates
    return n_tokens * latent_dim  # one precision-weighted accumulate per token


def _build():
    import torch
    import torch.nn as nn

    class ProductOfExpertsFusion(nn.Module):
        """Fuse per-token diagonal-Gaussian experts into one latent posterior. Parameter-free, O(N), exact.

        ``forward(mu, log_prec)`` takes ``(batch, n_tokens, latent)`` expert means and log-precisions and
        returns ``(fused_mu, fused_prec)`` -- each ``(batch, latent)``. A unit prior keeps it well-posed when
        every expert is uncertain. Differentiable, so the encoders emitting the experts train through it.
        """

        def __init__(self, prior_prec: float = 1.0) -> None:
            super().__init__()
            self.prior_prec = float(prior_prec)

        def forward(self, mu: Any, log_prec: Any) -> tuple[Any, Any]:
            prec = torch.nn.functional.softplus(log_prec)  # (b, n, m) >= 0
            fused_prec = prec.sum(dim=1) + self.prior_prec  # precisions add (+ prior)
            fused_mu = (prec * mu).sum(dim=1) / fused_prec  # precision-weighted mean
            return fused_mu, fused_prec

    class StructuredFusionClassifier(nn.Module):
        """A from-scratch multimodal classifier: shared per-token encoder -> PoE fusion -> linear head.

        The Level-3 architecture in miniature: replace the minimal encoder with real modality encoders
        such as a patch CNN or text embedder, and the same structured fusion aggregates their evidence in O(N).
        """

        def __init__(self, token_dim: int, latent_dim: int, n_classes: int, hidden: int = 32) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(token_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * latent_dim))
            self.latent_dim = int(latent_dim)
            self.fusion = ProductOfExpertsFusion()
            self.head = nn.Linear(2 * latent_dim, n_classes)  # posterior mean + log-precision -> logits

        def forward(self, tokens: Any) -> Any:  # tokens: (batch, n_tokens, token_dim)
            h = self.encoder(tokens)
            mu, log_prec = h[..., : self.latent_dim], h[..., self.latent_dim :]
            fused_mu, fused_prec = self.fusion(mu, log_prec)
            return self.head(torch.cat([fused_mu, torch.log(fused_prec)], dim=-1))

    class HybridFusionClassifier(nn.Module):
        """Attention for the relations, structured PoE for the aggregation -- the accuracy/compute sweet spot.

        Pure PoE fusion is permutation-invariant and misses token interactions; a full ViT models them but pays
        O(N^2) per layer and pools with a CLS token. This runs a small number of attention layers to inject the
        relational structure PoE lacks, then aggregates with the parameter-free precision-weighted readout.
        In the CIFAR patch benchmark, one attention layer plus PoE readout outperforms a same-budget ViT while
        using less compute than a deeper ViT.

        ``n_tokens`` is required (positional embeddings); ``attn_layers`` trades cost for relational capacity.
        """

        def __init__(
            self,
            token_dim: int,
            latent_dim: int,
            n_classes: int,
            n_tokens: int,
            *,
            attn_layers: int = 1,
            heads: int = 4,
            hidden: int = 32,
        ) -> None:
            super().__init__()
            self.latent_dim = int(latent_dim)
            self.encoder = nn.Sequential(nn.Linear(token_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * latent_dim))
            self.proj = nn.Linear(2 * latent_dim, latent_dim)
            self.pos = nn.Parameter(0.02 * torch.randn(1, n_tokens, latent_dim))
            self.attn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(latent_dim, heads, 2 * latent_dim, batch_first=True), attn_layers
            )
            self.to_expert = nn.Linear(latent_dim, 2 * latent_dim)
            self.fusion = ProductOfExpertsFusion()
            self.head = nn.Linear(2 * latent_dim, n_classes)

        def forward(self, tokens: Any) -> Any:
            t = self.proj(self.encoder(tokens)) + self.pos  # per-token latent + position
            t = self.attn(t)  # relational mixing (O(N^2) but only a few layers)
            h = self.to_expert(t)
            fused_mu, fused_prec = self.fusion(h[..., : self.latent_dim], h[..., self.latent_dim :])
            return self.head(torch.cat([fused_mu, torch.log(fused_prec)], dim=-1))

    return ProductOfExpertsFusion, StructuredFusionClassifier, HybridFusionClassifier


def __getattr__(name: str) -> Any:
    built = ("ProductOfExpertsFusion", "StructuredFusionClassifier", "HybridFusionClassifier")
    if name in built:
        poe, clf, hyb = _build()
        globals().update(dict(zip(built, (poe, clf, hyb))))
        return {"ProductOfExpertsFusion": poe, "StructuredFusionClassifier": clf, "HybridFusionClassifier": hyb}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
