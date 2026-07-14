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

Workstream L (cross-model adjudication): the same precision-weighted product-of-experts rule above, applied
not to learned tokens but to scalar claims from independent *external* models (a CMIP climate projection, a
hydrology emulator, ...). :func:`fuse_claims` fuses a list of :class:`ModelClaim` into one :class:`FusedBelief`,
flags when two models disagree beyond a standardized-distance threshold, and -- on disagreement -- adjudicates
via an IC-6-shaped ``verifier`` plus the ``language_bridge`` conformal claim score before ever emitting a fused
point, so a driller-facing cross-model number is never quietly averaged out of a real disagreement.

L8 (multi-climate-model ensemble fusion) does not add new fusion math: a real climate question is never
answered by one model, it is answered by an ensemble (CMIP members, AI emulators, ...) with uneven skill
against held-out observations. :func:`skill_weighted_fuse` maps each :class:`ClimateMember`'s ``skill`` onto
:func:`fuse_claims`'s ``reliability`` slot, so the ensemble posterior is the same precision-weighted
product-of-experts rule with higher-skill members earning proportionally more weight -- disagreement/abstention
are inherited unchanged from L5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def fusion_flops(n_tokens: int, latent_dim: int, *, attention: bool = False) -> int:
    """Approximate multiply-adds to fuse ``n_tokens`` into one latent.

    Product-of-experts fusion is O(N*M); attention is O(N^2*M).
    """
    if attention:
        return n_tokens * n_tokens * latent_dim  # the QK^T score matrix dominates
    return n_tokens * latent_dim  # one precision-weighted accumulate per token


@dataclass(frozen=True)
class ModelClaim:
    """One external model's scalar claim about a shared quantity, with the provenance it must carry.

    ``variance`` is that model's own uncertainty about ``value`` (physical units, not log-precision);
    ``reliability`` is a prior trust weight (1.0 = taken at face value) folded into the fused precision
    the same way L8's per-member ``skill`` later will. ``model_id``/``version``/``content_hash`` mirror
    IC-7's ``ProvenancedResult`` fields so every fused belief traces back to the calls that produced it.
    """

    value: float
    variance: float
    model_id: str
    version: str
    content_hash: str
    reliability: float = 1.0


@dataclass(frozen=True)
class FusedBelief:
    """The precision-weighted fusion of several :class:`ModelClaim`, with attribution and an honesty gate.

    ``weights`` is each model's share of the fused precision (sums to 1). ``disagreement`` fires when the
    worst pairwise standardized distance between two claims exceeds ``sigma_flag`` (default 3-sigma);
    ``abstained`` is only ever ``True`` when ``disagreement`` is ``True`` AND no single claim clears the
    conformal accept bar under cross-model adjudication -- the mean/variance are still the precision-weighted
    values even then, but callers MUST check ``abstained`` before surfacing ``mean`` as a driller-facing number.
    """

    mean: float
    variance: float
    weights: dict[str, float]
    disagreement: bool
    abstained: bool
    provenance: dict[str, Any]


def _max_pairwise_standardized_distance(claims: list[ModelClaim]) -> float:
    """``max_{i<j} |value_i - value_j| / sqrt(variance_i + variance_j)`` -- 0.0 for a single claim."""
    worst = 0.0
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a, b = claims[i], claims[j]
            z = abs(a.value - b.value) / math.sqrt(a.variance + b.variance)
            worst = max(worst, z)
    return worst


def _any_claim_clears_accept_bar(
    claims: list[ModelClaim], *, verifier: Any = None, n_samples: int = 200, seed: int = 0, width_sigma: float = 1.0
) -> bool:
    """Cross-model adjudication for a disagreeing claim set: does *any* claim survive scrutiny?

    Two independent checks, either of which can clear a claim (so it is not folded into an abstain):

    1. IC-6 ``verifier`` (if supplied): ``verifier.verify(claim, context) -> Verdict``-shaped, duck-typed
       so this module never imports ``mixle_mlops`` (E10 owns the real physical/calibration verifiers).
    2. ``language_bridge.claim_score`` (frozen at ``reason/language_bridge.py:146``): build a
       ``width_sigma``-sigma interval :class:`~mixle.reason.language_bridge.Claim` around each model's value
       and score it -- via the *other* models' synthetic posteriors -- for coverage-per-unit-width. Under
       real disagreement (the only path that reaches this function) every other model's mass sits many sigma
       away from a claim's own interval, so its cross-model coverage is 0 and the score cannot clear any
       positive bar; near-agreeing claims (the common case that never reaches here because disagreement is
       False) would instead see substantial cross-coverage. A claim "clears the bar" iff its score against at
       least one other model's posterior is strictly positive.
    """
    from mixle.reason.language_bridge import Claim, claim_score

    if len(claims) < 2:
        return True  # nothing to adjudicate against -- a lone claim cannot disagree with itself

    rng = np.random.default_rng(seed)
    posteriors = {c.model_id: rng.normal(c.value, math.sqrt(c.variance), n_samples) for c in claims}

    for claim in claims:
        if verifier is not None:
            context = {
                "claims": [
                    {"model_id": o.model_id, "value": o.value, "variance": o.variance} for o in claims if o is not claim
                ]
            }
            candidate = {"model_id": claim.model_id, "value": claim.value, "variance": claim.variance}
            verdict = verifier.verify(candidate, context)
            if getattr(verdict, "passed", False):
                return True

        sd = math.sqrt(claim.variance)
        interval = Claim(field=claim.model_id, lo=claim.value - width_sigma * sd, hi=claim.value + width_sigma * sd)
        for other in claims:
            if other.model_id == claim.model_id:
                continue
            score = claim_score(interval, posterior=posteriors[other.model_id], n_samples=n_samples, seed=seed)
            if score > 0.0:
                return True
    return False


def fuse_claims(
    claims: list[ModelClaim],
    *,
    prior_prec: float = 0.0,
    sigma_flag: float = 3.0,
    verifier: Any = None,
) -> FusedBelief:
    """Precision-weighted product-of-experts fusion of independent external-model claims (workstream L5).

    Exactly the rule stated at the top of this module (``prec_fused = sum(prec_i) + prior_prec``,
    ``mean = sum(prec_i * value_i) / prec_fused``), applied to scalar :class:`ModelClaim`\\ s instead of
    learned tokens: ``prec_i = reliability_i / variance_i``. On disagreement (worst pairwise standardized
    distance ``> sigma_flag``), the fused point is only trusted once at least one claim clears cross-model
    adjudication (:func:`_any_claim_clears_accept_bar`); otherwise ``abstained=True`` and the caller must not
    surface ``mean`` as a resolved answer. ``provenance`` records every claim's id/version/content_hash/weight
    so a fused belief is always attributable back to the models that produced it.
    """
    if not claims:
        raise ValueError("fuse_claims needs at least one ModelClaim")
    for c in claims:
        if c.variance <= 0:
            raise ValueError(f"ModelClaim {c.model_id!r} has non-positive variance {c.variance!r}")

    precisions = [c.reliability / c.variance for c in claims]
    total_prec = sum(precisions)
    prec_fused = total_prec + prior_prec
    mean = sum(p * c.value for p, c in zip(precisions, claims)) / prec_fused
    variance = 1.0 / prec_fused
    weights = {c.model_id: p / total_prec for p, c in zip(precisions, claims)}

    max_z = _max_pairwise_standardized_distance(claims)
    disagreement = max_z > sigma_flag
    abstained = disagreement and not _any_claim_clears_accept_bar(claims, verifier=verifier)

    provenance = {
        "claims": [
            {
                "model_id": c.model_id,
                "version": c.version,
                "content_hash": c.content_hash,
                "weight": weights[c.model_id],
            }
            for c in claims
        ],
        "max_pairwise_standardized_distance": max_z,
        "sigma_flag": sigma_flag,
    }
    return FusedBelief(
        mean=mean,
        variance=variance,
        weights=weights,
        disagreement=disagreement,
        abstained=abstained,
        provenance=provenance,
    )


@dataclass(frozen=True)
class ClimateMember:
    """One external climate model's projection in a multi-model ensemble (workstream L8).

    A ``ClimateMember`` is a CMIP ensemble member or an AI emulator's projection of a shared climate quantity.
    ``skill`` is a per-member inverse validation error against held-out observations (1.0 = neutral trust) --
    it plays exactly the role :class:`ModelClaim`'s ``reliability`` plays in L5, because L8 folds skill into
    L5's frozen precision-weighted rule rather than inventing new fusion math. ``model_id``/``version``/
    ``content_hash`` mirror IC-7's ``ProvenancedResult`` fields so every ensemble member traces back to the
    domain-model call that produced it.
    """

    value: float
    variance: float
    model_id: str
    version: str
    content_hash: str
    skill: float = 1.0


def skill_weighted_fuse(
    members: list[ClimateMember],
    *,
    sigma_flag: float = 3.0,
    verifier: Any = None,
) -> FusedBelief:
    """Skill-weighted Bayesian model averaging of a multi-climate-model ensemble (workstream L8/DR-ALG L8).

    Each :class:`ClimateMember` maps onto an L5 :class:`ModelClaim` with ``reliability = skill``, then
    :func:`fuse_claims` (the frozen precision-weighted product-of-experts rule) does the actual fusion:
    ``prec_i = skill_i / variance_i`` and the BMA posterior weight is ``skill_i * prec_i / sum_j skill_j *
    prec_j``, so a higher-skill ensemble member earns proportionally more weight in the fused projection.
    Disagreement/abstention are inherited unchanged from L5: the max pairwise standardized distance flags at
    ``sigma_flag`` (default 3-sigma) and, on disagreement, routes through the same IC-6 ``verifier`` +
    ``language_bridge`` adjudication before a fused ensemble point is ever trusted. ``provenance`` records
    every member's ``{model_id, version, content_hash, weight, skill}`` so the fused projection driving
    L2/L3/L7 is always attributable back to the ensemble members that produced it.
    """
    if not members:
        raise ValueError("skill_weighted_fuse needs at least one ClimateMember")

    claims = [
        ModelClaim(
            value=m.value,
            variance=m.variance,
            model_id=m.model_id,
            version=m.version,
            content_hash=m.content_hash,
            reliability=m.skill,
        )
        for m in members
    ]
    fused = fuse_claims(claims, sigma_flag=sigma_flag, verifier=verifier)

    skill_by_id = {m.model_id: m.skill for m in members}
    for entry in fused.provenance["claims"]:
        entry["skill"] = skill_by_id[entry["model_id"]]
    return fused


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
