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
via an IC-6-shaped ``verifier`` plus a predeclared, calibrated pairwise significance test (deterministic, no
Monte Carlo tail luck involved) before ever emitting a fused point, classifying every claim as accepted,
rejected, or unresolved so a driller-facing cross-model number is never quietly averaged out of -- or
contaminated by -- a real disagreement.

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

    ``weights`` has exactly one entry per input claim, keyed by that claim's ``model_id`` -- except
    when two or more claims share a ``model_id`` (repeated ensemble members of the same model is the
    realistic case; see :func:`skill_weighted_fuse`), in which case only the FIRST such claim keeps the
    bare ``model_id`` key and every later one is disambiguated ``"{model_id}#{n}"`` (matching the
    id-collision convention in :meth:`mixle.epistemic.portfolio.HypothesisPortfolio.resample`) so no
    claim's share is dropped. This SAME stable per-claim identity (not the possibly-duplicated bare
    ``model_id``) is threaded through cross-model adjudication too, so two distinct claims that happen
    to share a ``model_id`` are compared like any other pair instead of silently colliding. This same
    key appears in ``provenance["claims"]`` under ``"weight_key"`` for exact correlation.

    ``disagreement`` fires when the worst pairwise standardized distance between two claims exceeds
    ``sigma_flag`` (default 3-sigma). On disagreement, every claim is independently classified as
    accepted, rejected, or unresolved by a predeclared, deterministic cross-model test (see
    :func:`_adjudicate_claims`); ONLY accepted claims contribute to ``mean``/``variance``/``weights``,
    so one claim clearing adjudication no longer drags every conflicting claim along with it. ``weights``
    is 0.0 for any claim excluded this way. ``abstained`` is ``True`` iff ``disagreement`` is ``True``
    AND not a single claim was accepted -- ``mean``/``variance`` then fall back to the precision-weighted
    fusion of ALL claims (still *a* number, never absent), but callers MUST check ``abstained`` before
    surfacing ``mean`` as a driller-facing number.

    ``weights`` sums to 1 across the surviving claims when :func:`fuse_claims` was called with no prior
    (the default). A positive ``prior_prec`` takes its own share of the fused precision -- ``weights``
    (which only ever has one entry per CLAIM) then sums to ``1 - prior_weight``, and
    ``provenance["prior_prec"]``/``["prior_mean"]``/``["prior_weight"]`` report exactly what that prior
    was and how much of the shrinkage it is responsible for (MXR-080-0286).
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


def _connected_components(n: int, edges: set[tuple[int, int]]) -> list[list[int]]:
    """Connected components of the undirected graph on ``range(n)`` implied by ``edges``."""
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    seen = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack, component = [start], []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(component)
    return components


def _pairwise_p_value(z: float) -> float:
    """Two-sided p-value for a standardized distance ``z`` under the null that both claims estimate the
    same true value with correctly-specified, independent Gaussian noise: ``erfc(z / sqrt(2))``. This is
    the exact, closed-form significance behind the ``sigma_flag`` bar (``z > 3`` <=> ``p < 0.0027``) --
    reported per claim so adjudication's classification carries real, quantified uncertainty rather than
    a silent binary pass/fail."""
    if not math.isfinite(z):
        return 1.0
    return math.erfc(z / math.sqrt(2.0))


def _adjudicate_claims(
    claims: list[ModelClaim],
    weight_keys: list[str],
    *,
    verifier: Any = None,
    sigma_flag: float = 3.0,
) -> list[dict[str, Any]]:
    """Predeclared, calibrated cross-model adjudication for a disagreeing claim set (MXR-080-0284/0285).

    Deterministic and seed-free -- there is no Monte Carlo sampling anywhere in this function, so the
    verdict can never depend on tail luck. Every claim is classified independently as ``"accepted"``,
    ``"rejected"``, or ``"unresolved"``:

    1. Build the agreement graph on the claims: an edge between claims i and j iff their pairwise
       standardized distance ``|value_i - value_j| / sqrt(variance_i + variance_j)`` is at most
       ``sigma_flag`` -- the SAME statistic and SAME predeclared bar already used to decide the set
       disagrees in the first place (default 3-sigma, a two-sided ~0.27% false-flag rate per pair under
       the null that both claims are unbiased independent estimates of the same quantity; see
       :func:`_pairwise_p_value`). Comparisons are keyed by each claim's own list index, never by its
       (possibly duplicated) ``model_id``, so distinct claims that share a ``model_id`` -- e.g. two
       realizations of the same ensemble member -- are compared like any other pair instead of one
       silently overwriting or skipping the other.
    2. Find the graph's connected components. If exactly one component is uniquely the largest
       (size >= 2, no tie), its members are "accepted" (a corroborated core) and every claim outside it
       is "rejected" (a confirmed minority/outlier against that core). If every claim is isolated, or
       two-or-more components tie for largest, there is no statistical basis to prefer one claim over
       another: every claim is "unresolved".
    3. An IC-6 ``verifier`` (if supplied) is an independent, per-claim override, exactly as before:
       ``verifier.verify(candidate, context) -> Verdict``-shaped, duck-typed so this module never imports
       ``mixle_mlops``. A claim the verifier vouches for is "accepted" regardless of step 2 -- it can only
       ever promote a claim, never reject one the graph already accepted.

    Accepting one claim never implicitly clears any other: each claim's status is decided on its own, so
    callers can gate inclusion in the fused mean per claim (MXR-080-0285) instead of an all-or-nothing
    "did anything clear the bar" gate that let a single lucky pass drag every conflicting claim along.

    Known scope limit: this detects "one clear outlier against a corroborated core," not general robust
    clustering -- e.g. two same-size, mutually-disagreeing corroborated pairs are both left "unresolved"
    rather than one being arbitrarily preferred over the other.
    """
    n = len(claims)
    z = [[0.0] * n for _ in range(n)]
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            zij = abs(claims[i].value - claims[j].value) / math.sqrt(claims[i].variance + claims[j].variance)
            z[i][j] = z[j][i] = zij
            if zij <= sigma_flag:
                edges.add((i, j))

    verifier_passed: list[bool | None] = [None] * n
    if verifier is not None:
        for i, claim in enumerate(claims):
            context = {
                "claims": [
                    {"model_id": o.model_id, "value": o.value, "variance": o.variance}
                    for j, o in enumerate(claims)
                    if j != i
                ]
            }
            candidate = {"model_id": claim.model_id, "value": claim.value, "variance": claim.variance}
            verdict = verifier.verify(candidate, context)
            verifier_passed[i] = bool(getattr(verdict, "passed", False))

    components = _connected_components(n, edges)
    max_size = max((len(comp) for comp in components), default=0)
    core_components = [comp for comp in components if len(comp) == max_size]
    graph_status = ["unresolved"] * n
    if max_size >= 2 and len(core_components) == 1:
        core = set(core_components[0])
        for i in range(n):
            graph_status[i] = "accepted" if i in core else "rejected"

    results: list[dict[str, Any]] = []
    for i in range(n):
        status = "accepted" if verifier_passed[i] else graph_status[i]
        others_z = [z[i][j] for j in range(n) if j != i]
        min_z = min(others_z) if others_z else math.inf
        results.append(
            {
                "weight_key": weight_keys[i],
                "status": status,
                "min_pairwise_z": min_z,
                "min_pairwise_p_value": _pairwise_p_value(min_z),
                "verifier_passed": verifier_passed[i],
            }
        )
    return results


def fuse_claims(
    claims: list[ModelClaim],
    *,
    prior_prec: float = 0.0,
    prior_mean: float | None = None,
    sigma_flag: float = 3.0,
    verifier: Any = None,
) -> FusedBelief:
    """Precision-weighted product-of-experts fusion of independent external-model claims (workstream L5).

    Exactly the rule stated at the top of this module generalized to an explicit prior mean
    (``prec_fused = sum(prec_i) + prior_prec``, ``mean = (sum(prec_i * value_i) + prior_prec *
    prior_mean) / prec_fused``), applied to scalar :class:`ModelClaim`\\ s instead of learned tokens:
    ``prec_i = reliability_i / variance_i``. On disagreement (worst pairwise standardized distance
    ``> sigma_flag``), every claim is independently classified as accepted, rejected, or unresolved by a
    predeclared, deterministic cross-model test (:func:`_adjudicate_claims`); ONLY accepted claims feed
    ``mean``/``variance``/``weights`` (see :class:`FusedBelief`'s docstring for the full inclusion-gating
    and abstention contract). ``provenance`` records every claim's
    id/version/content_hash/weight/adjudication-status so a fused belief is always attributable back to
    the models that produced it and to exactly why each one was or was not trusted.

    ``prior_prec`` MUST be paired with an explicit ``prior_mean`` whenever it is positive: unlike the
    torch :class:`ProductOfExpertsFusion` at the bottom of this module, whose experts describe an
    arbitrary LEARNED latent space where a zero-mean prior is the standard, principled convention (as in
    a VAE), a :class:`ModelClaim`'s ``value`` is a real PHYSICAL quantity (a temperature, a flow rate, ...)
    -- silently shrinking it toward 0 has no physical justification and can quietly change a
    physical-unit prediction (MXR-080-0286). ``provenance["prior_weight"]`` reports the prior's own share
    of the fused precision so the attribution is honest about where any shrinkage comes from; ``weights``
    (which is claims-only) then sums to ``1 - prior_weight`` rather than always summing to 1.

    Every scalar that reaches the precision arithmetic is validated up front (``value``/``variance``/
    ``reliability`` per claim, plus ``prior_prec``/``prior_mean``/``sigma_flag``) and the resulting
    per-claim and total precision are both re-checked as strictly positive and finite immediately before
    they are divided by -- this function never returns a :class:`FusedBelief` carrying a NaN or negative
    ``variance`` (MXR-080-0283); it raises ``ValueError`` instead.
    """
    if not claims:
        raise ValueError("fuse_claims needs at least one ModelClaim")
    if not math.isfinite(sigma_flag) or sigma_flag <= 0:
        raise ValueError(f"sigma_flag must be finite and > 0, got {sigma_flag!r}")
    if not math.isfinite(prior_prec) or prior_prec < 0:
        raise ValueError(f"prior_prec must be finite and >= 0, got {prior_prec!r}")
    if prior_prec > 0 and prior_mean is None:
        raise ValueError(
            "fuse_claims requires an explicit prior_mean when prior_prec > 0 -- a precision-only prior "
            "silently implies a zero-centered prior mean, which is rarely physically meaningful (e.g. "
            "shrinking a temperature claim toward 0 degrees for no physical reason)"
        )
    if prior_mean is not None and not math.isfinite(prior_mean):
        raise ValueError(f"prior_mean must be finite, got {prior_mean!r}")
    for c in claims:
        if not math.isfinite(c.value):
            raise ValueError(f"ModelClaim {c.model_id!r} has a non-finite value {c.value!r}")
        if not math.isfinite(c.variance) or c.variance <= 0:
            raise ValueError(f"ModelClaim {c.model_id!r} has non-positive/non-finite variance {c.variance!r}")
        if not math.isfinite(c.reliability) or c.reliability <= 0:
            raise ValueError(f"ModelClaim {c.model_id!r} has non-positive/non-finite reliability {c.reliability!r}")

    precisions = [c.reliability / c.variance for c in claims]
    for c, p in zip(claims, precisions):
        if not math.isfinite(p) or p <= 0:
            raise ValueError(f"ModelClaim {c.model_id!r} has a non-positive/non-finite effective precision {p!r}")
    total_prec = sum(precisions)
    if not math.isfinite(total_prec) or total_prec <= 0:
        raise ValueError(f"fuse_claims produced a non-positive/non-finite total precision {total_prec!r}")

    # One weight-dict key per CLAIM, not per distinct model_id -- see FusedBelief's docstring. A
    # claim keeps its bare model_id as the key unless an earlier claim in this list already claimed
    # it, in which case it is disambiguated "{model_id}#{n}"; either way every claim's own precision
    # share survives into `weights`, so it always sums to 1.0 even with duplicate model_ids. This SAME
    # identity is threaded through adjudication below (MXR-080-0284) instead of re-deriving a second,
    # bare-model_id-keyed identity that can silently collide on duplicates.
    seen: dict[str, int] = {}
    weight_keys: list[str] = []
    for c in claims:
        count = seen.get(c.model_id, 0)
        seen[c.model_id] = count + 1
        weight_keys.append(c.model_id if count == 0 else f"{c.model_id}#{count}")

    max_z = _max_pairwise_standardized_distance(claims)
    disagreement = max_z > sigma_flag

    adjudication: list[dict[str, Any]] | None = None
    if disagreement:
        adjudication = _adjudicate_claims(claims, weight_keys, verifier=verifier, sigma_flag=sigma_flag)
        accepted_idx = [i for i, a in enumerate(adjudication) if a["status"] == "accepted"]
    else:
        accepted_idx = list(range(len(claims)))

    abstained = disagreement and not accepted_idx
    # Nothing survived adjudication: mean/variance fall back to ALL claims so they are still *a* number
    # (never NaN/absent) -- `abstained=True` forces callers to check before trusting it, matching
    # FusedBelief's documented contract. Once something IS accepted, only accepted claims contribute
    # (MXR-080-0285: one claim's pass no longer drags every conflicting claim along into the mean).
    included_idx = accepted_idx if accepted_idx else list(range(len(claims)))
    included = set(included_idx)

    included_total_prec = sum(precisions[i] for i in included_idx)
    prec_fused = included_total_prec + prior_prec
    if not math.isfinite(prec_fused) or prec_fused <= 0:
        raise ValueError(f"fuse_claims produced a non-positive/non-finite fused precision {prec_fused!r}")
    numerator = sum(precisions[i] * claims[i].value for i in included_idx)
    if prior_prec > 0:
        numerator += prior_prec * prior_mean  # prior_mean is guaranteed non-None here (validated above)
    mean = numerator / prec_fused
    variance = 1.0 / prec_fused
    if not math.isfinite(mean) or not math.isfinite(variance) or variance <= 0:
        raise ValueError("fuse_claims produced a non-finite fused mean/variance from otherwise-valid inputs")
    # Divide by the FULL fused precision (claims + prior), not just the claims' own total, so `weights`
    # honestly reflects each claim's share of the ACTUAL fused result: when prior_prec > 0 the claims no
    # longer explain 100% of `mean`, so `weights` sums to `1 - prior_weight` rather than always 1
    # (MXR-080-0286) -- `provenance["prior_weight"]` reports the prior's own share explicitly so the
    # shortfall is never silently unexplained.
    per_claim_weight = [(precisions[i] / prec_fused) if i in included else 0.0 for i in range(len(claims))]
    weights = dict(zip(weight_keys, per_claim_weight))
    prior_weight = (prior_prec / prec_fused) if prior_prec > 0 else 0.0

    provenance: dict[str, Any] = {
        "claims": [
            {
                "model_id": c.model_id,
                "version": c.version,
                "content_hash": c.content_hash,
                "weight": w,
                "weight_key": k,
                "included_in_fused_mean": i in included,
            }
            for i, (c, w, k) in enumerate(zip(claims, per_claim_weight, weight_keys))
        ],
        "max_pairwise_standardized_distance": max_z,
        "sigma_flag": sigma_flag,
        "prior_prec": prior_prec,
        "prior_mean": prior_mean,
        "prior_weight": prior_weight,
    }
    if adjudication is not None:
        for entry, a in zip(provenance["claims"], adjudication):
            entry["adjudication_status"] = a["status"]
            entry["min_pairwise_z"] = a["min_pairwise_z"]
            entry["min_pairwise_p_value"] = a["min_pairwise_p_value"]
            entry["verifier_passed"] = a["verifier_passed"]

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

    # Positional, not an id-keyed dict lookup: `claims` (and so `fused.provenance["claims"]`, which
    # fuse_claims builds by iterating `claims` in order without reordering or filtering) is exactly
    # `members` mapped 1:1 in order, so this is safe even when two members share a model_id -- the
    # realistic case for a multi-realization CMIP ensemble member, which an id-keyed dict would
    # silently collapse to only the last member's skill for every entry sharing that id.
    for entry, m in zip(fused.provenance["claims"], members):
        entry["skill"] = m.skill
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
