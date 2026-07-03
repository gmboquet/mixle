"""Differentiable product-of-experts fusion -- structured multimodal fusion you train from scratch.

Dense cross-attention fuses N tokens in O(N^2). When the tokens are (conditionally) independent evidence about
a shared latent -- the common case for aggregating many partial observations (image patches, sensors, views) --
precision-weighted product-of-experts fuses them in O(N) with almost no parameters: the fusion inductive bias
is built in, not learned. Each expert is a diagonal Gaussian ``N(mu_i, diag(1/prec_i))`` over the latent, and
the posterior is their normalized product::

    prec_fused = sum_i prec_i + prec_prior          # precisions add
    mu_fused   = (sum_i prec_i * mu_i) / prec_fused  # precision-weighted mean

Measured from scratch on a laptop (examples/structured_fusion_vlm.py): PoE fusion matches a cross-attention
block's accuracy at ~2.6x fewer parameters and ~7x faster training on exchangeable-evidence tasks.

The honest boundary: PoE fusion is PERMUTATION-INVARIANT and assumes conditional independence -- it cannot
model token order or pairwise interactions. On a task that depends on a specific pair or position, attention
reaches ~0.96 while PoE sits at chance. So the design is: structured fusion where evidence is exchangeable (to
cut the quadratic cost), attention for the relational parts.

mixle.reason's exact core (:class:`GaussianBelief`) does this fusion in closed form for *inference*; this is the
torch, end-to-end-trainable version -- the encoders that emit the experts are learned, the fusion stays exact.
Torch is imported lazily.
"""

from __future__ import annotations

from typing import Any


def fusion_flops(n_tokens: int, latent_dim: int, *, attention: bool = False) -> int:
    """Rough multiply-adds to fuse ``n_tokens`` into one latent. PoE is O(N*M); attention is O(N^2*M)."""
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

        The Level-3 architecture in miniature -- swap the toy encoder for real modality encoders (a small
        patch CNN, a text embedder) and the same structured fusion aggregates their evidence in O(N).
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

    return ProductOfExpertsFusion, StructuredFusionClassifier


def __getattr__(name: str) -> Any:
    if name in ("ProductOfExpertsFusion", "StructuredFusionClassifier"):
        poe, clf = _build()
        globals()["ProductOfExpertsFusion"], globals()["StructuredFusionClassifier"] = poe, clf
        return {"ProductOfExpertsFusion": poe, "StructuredFusionClassifier": clf}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
