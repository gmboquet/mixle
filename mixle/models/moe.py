"""Mixture-of-experts transformer MLP -- the neural half of ConditionalJIT's structural adaptation (roadmap H2).

The user-stated goal "the model adapts its structure as it trains," applied to
:class:`~mixle.models.transformer.Block`'s MLP: instead of one dense feed-forward network every token
runs through, :class:`MoEBlock` gives each token a per-token choice of ``N`` expert feed-forward
networks via a learned linear gate, so the network's *effective* structure (which parameters a given
token's forward pass touches) is decided at train/inference time rather than fixed at init.

**This is the same "mixture" idea mixle already has a name for.** :class:`mixle.stats.latent.mixture.MixtureDistribution`
defines ``P(Y) = sum_k P(Y|Z=k) P(Z=k)``: a *soft* responsibility (posterior ``P(Z=k|Y)``, fit by EM)
selects which of ``K`` homogeneous component distributions explains a data point. A gradient-trained
MoE gate is the same combinator with a different fitting mechanism: ``softmax(W x)`` plays the role of
the responsibility ``P(Z=k|Y)`` and each expert MLP plays the role of a mixture component, but the
routing distribution is trained end-to-end by gradient descent through a load-balancing auxiliary loss
rather than by EM's alternating E/M steps (MoE's per-token hard top-k selection is also a discrete
argmax over an otherwise continuous responsibility, unlike EM's fully soft E-step). That structural
identity is not just a metaphor: the gate's ``(n_tokens, n_experts)`` softmax output has *exactly* the
shape of the token-by-component responsibility matrix ``z`` that
:func:`mixle.utils.hvis.topology.model_fit_health`/:func:`~mixle.utils.hvis.topology.fuzzy_nerve` consume for
probabilistic mixtures, so :func:`expert_collapse_receipt` below feeds routing weights into
``fuzzy_nerve`` directly -- the exact same overlap-nerve computation HViS uses to flag a mixture's
merged/shattered component regimes, re-aimed at expert routing statistics instead of clustering
posteriors. See that function's docstring for the re-aimed semantics.

Two entry points:

* :class:`MoEBlock` -- drop-in replacement for :class:`~mixle.models.transformer.Block` (attention
  unchanged; the dense MLP is replaced by :class:`MoEMLP`, ``N`` expert MLPs plus a top-k linear gate
  and the standard Switch-Transformer load-balancing auxiliary loss).
* :func:`upcycle_dense_to_moe` -- turn an already-trained dense ``Block`` into an ``MoEBlock`` by
  copying attention unchanged and initializing every expert as a near-copy of the dense MLP (the
  standard "sparse upcycling" trick: Komatsuzaki et al., 2023), carrying a function-preservation-style
  receipt (how close the freshly-upcycled model's output is to the original dense block's output,
  before any MoE-specific training happens).
* :func:`expert_collapse_receipt` -- the balance/collapse receipt: "merged" (routing collapsed onto a
  handful of experts) and "shattered" (routing so unstable round-to-round that no expert receives a
  consistent, learnable token distribution), reusing :func:`mixle.utils.hvis.topology.fuzzy_nerve`.

Torch is imported lazily/guarded exactly like ``transformer.py`` so this module still imports (as a
no-op) when torch is not installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = ["expert_collapse_receipt", "upcycle_dense_to_moe"]

if _HAS_TORCH:

    class MoEMLP(nn.Module):
        """``N`` expert MLPs (same shape as :class:`~mixle.models.transformer.Block`'s dense MLP) plus a
        linear top-k routing gate.

        ``expert_hidden`` defaults to ``4 * d_model`` -- the same hidden width as the dense MLP it
        replaces -- so with ``top_k=1`` the ACTIVE compute per token (one expert's forward pass) matches
        the dense MLP's compute exactly: total capacity grows with ``n_experts`` while active FLOPs/token
        stays fixed, which is the standard "matched-FLOPs" MoE-vs-dense comparison methodology (Switch
        Transformer / GShard).

        After every ``forward`` the dense (pre-top-k) gate softmax is cached on
        ``self.last_gate_probs`` (``(n_tokens, n_experts)``, detached) and the Switch-style
        load-balancing auxiliary loss on ``self.last_aux_loss`` -- both are what
        :func:`expert_collapse_receipt` / the training loop consume.
        """

        def __init__(
            self,
            d_model: int,
            n_experts: int,
            *,
            top_k: int = 1,
            expert_hidden: int | None = None,
            aux_loss_weight: float = 0.01,
        ) -> None:
            super().__init__()
            if not (1 <= top_k <= n_experts):
                raise ValueError(f"top_k={top_k} must be in [1, n_experts={n_experts}]")
            self.d_model = int(d_model)
            self.n_experts = int(n_experts)
            self.top_k = int(top_k)
            self.expert_hidden = int(expert_hidden) if expert_hidden is not None else 4 * self.d_model
            self.aux_loss_weight = float(aux_loss_weight)
            self.gate = nn.Linear(d_model, n_experts, bias=False)
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_model, self.expert_hidden), nn.GELU(), nn.Linear(self.expert_hidden, d_model)
                    )
                    for _ in range(n_experts)
                ]
            )
            self.last_gate_probs: Any = None
            self.last_aux_loss: Any = None

        def forward(self, x: Any) -> Any:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            n_tokens = flat.shape[0]

            logits = self.gate(flat)  # (n_tokens, n_experts)
            probs = F.softmax(logits, dim=-1)
            self.last_gate_probs = probs.detach()

            top_w, top_idx = probs.topk(self.top_k, dim=-1)  # (n_tokens, top_k) each
            # NOT renormalized to sum to 1: the raw gate probability of each selected expert is the
            # combine weight (Switch Transformer's y = p_i(x) * FFN_i(x)), which is what carries a
            # gradient back to the gate's parameters through the main task loss -- a top_k=1
            # renormalization would collapse every selected weight to the constant 1.0 and cut that path,
            # leaving the gate learnable only through the auxiliary load-balance loss.
            out = flat.new_zeros(n_tokens, self.d_model)
            for e, expert in enumerate(self.experts):
                # tokens that picked expert e in ANY of their top_k slots
                slot = top_idx == e
                token_mask = slot.any(dim=-1)
                if not bool(token_mask.any()):
                    continue
                weight = (top_w * slot).sum(dim=-1)[token_mask]  # combine weight if e appears once (top_k<=n_experts)
                out[token_mask] += weight.unsqueeze(-1) * expert(flat[token_mask])

            # Switch-Transformer load-balance auxiliary loss: n_experts * sum_e f_e * P_e, minimized when
            # both the DISPATCH fraction f_e and the mean GATE probability P_e are uniform over experts.
            top1 = top_idx[:, 0]
            f = torch.zeros(self.n_experts, device=x.device, dtype=probs.dtype)
            f.scatter_add_(0, top1, torch.ones_like(top1, dtype=probs.dtype))
            f = f / max(n_tokens, 1)
            p = probs.mean(dim=0)
            aux = self.n_experts * torch.sum(f * p) * self.aux_loss_weight
            self.last_aux_loss = aux

            return out.reshape(shape)

    class MoEBlock(nn.Module):
        """Drop-in replacement for :class:`mixle.models.transformer.Block`: identical pre-norm attention,
        MoE-routed MLP in place of the dense one. Same call signature as ``Block`` plus the MoE
        hyperparameters, so it can replace entries of a ``CausalLM.blocks`` ``ModuleList`` directly.
        """

        def __init__(
            self,
            d_model: int,
            n_head: int,
            n_experts: int,
            *,
            top_k: int = 1,
            expert_hidden: int | None = None,
            aux_loss_weight: float = 0.01,
        ) -> None:
            super().__init__()
            from mixle.models.transformer import CausalAttention

            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.attn = CausalAttention(d_model, n_head)
            self.mlp = MoEMLP(
                d_model,
                n_experts,
                top_k=top_k,
                expert_hidden=expert_hidden,
                aux_loss_weight=aux_loss_weight,
            )

        def forward(self, x: Any) -> Any:
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

        @property
        def aux_loss(self) -> Any:
            """The most recent forward's load-balancing auxiliary loss (add this to the training objective)."""
            return self.mlp.last_aux_loss

        @property
        def routing_weights(self) -> Any:
            """The most recent forward's dense gate softmax, ``(n_tokens, n_experts)``, detached -- feed a
            sequence of these (one per training round) to :func:`expert_collapse_receipt`."""
            return self.mlp.last_gate_probs


def upcycle_dense_to_moe(
    dense_block: Any,
    n_experts: int,
    *,
    top_k: int = 1,
    seed: int = 0,
    noise_std: float = 0.01,
    probe_tokens: int = 64,
) -> tuple[Any, dict]:
    """ "Sparse upcycling" (Komatsuzaki et al., 2023): build a fresh ``MoEBlock`` whose attention is
    copied unchanged from ``dense_block`` and whose ``n_experts`` expert MLPs are each initialized as a
    near-copy of ``dense_block``'s trained dense MLP (exact weights + small seeded Gaussian
    perturbation per expert, so experts start distinguishable rather than identical dead-gradient
    copies). Unlike H1's growth operators this is NOT exactly function-preserving -- the gate's
    top-``k`` hard selection is a nonlinearity the dense path never had, so upcycled output only
    APPROXIMATES the original dense output. That approximation is measured, not assumed: a fixed probe
    batch is pushed through both blocks and the relative L2 output gap is returned in the receipt.

    Returns ``(moe_block, receipt)`` where ``receipt`` has ``relative_output_diff`` (the measured gap,
    expected small but nonzero), ``n_experts``, ``top_k``, ``noise_std``, and ``seed``.
    """
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("upcycle_dense_to_moe requires torch")

    d_model = dense_block.ln1.normalized_shape[0]
    n_head = dense_block.attn.h
    dense_hidden = dense_block.mlp[0].out_features

    gen = torch.Generator().manual_seed(int(seed))
    moe_block = MoEBlock(d_model, n_head, n_experts, top_k=top_k, expert_hidden=dense_hidden)

    moe_block.ln1.load_state_dict(dense_block.ln1.state_dict())
    moe_block.ln2.load_state_dict(dense_block.ln2.state_dict())
    moe_block.attn.load_state_dict(dense_block.attn.state_dict())

    dense_mlp_state = dense_block.mlp.state_dict()
    with torch.no_grad():
        for expert in moe_block.mlp.experts:
            expert.load_state_dict(dense_mlp_state)
            for p in expert.parameters():
                p.add_(torch.randn(p.shape, generator=gen) * noise_std)

    with torch.no_grad():
        probe = torch.randn(1, probe_tokens, d_model, generator=gen)
        dense_block.eval()
        moe_block.eval()
        dense_out = dense_block(probe)
        moe_out = moe_block(probe)
        gap = torch.linalg.norm(moe_out - dense_out) / torch.linalg.norm(dense_out).clamp_min(1.0e-12)

    receipt = {
        "relative_output_diff": float(gap),
        "n_experts": int(n_experts),
        "top_k": int(top_k),
        "noise_std": float(noise_std),
        "seed": int(seed),
        "probe_tokens": int(probe_tokens),
    }
    return moe_block, receipt


def expert_collapse_receipt(
    routing_history: Sequence[Any],
    *,
    merged_effective_frac: float = 0.5,
    shattered_instability: float = 0.35,
    shattered_edge_threshold: float = 0.3,
    shattered_edge_frac: float = 0.5,
) -> dict:
    """Load-balance / expert-collapse receipt, reusing :func:`mixle.utils.hvis.topology.fuzzy_nerve` --
    the SAME overlap-nerve computation ``model_fit_health`` uses to flag a mixture's merged/shattered
    component regimes -- re-aimed at MoE routing statistics.

    ``routing_history`` is a sequence of per-round gate softmax matrices (each ``(n_tokens_r,
    n_experts)``, e.g. ``MoEBlock.routing_weights`` collected once per training step/round). Each
    matrix has exactly the shape of the token-by-component responsibility matrix ``z`` that
    ``fuzzy_nerve`` was built for, so it is fed in directly -- no adapter needed.

    Two failure modes, re-aimed from the original clustering semantics:

    * **merged** -- ``fuzzy_nerve``'s ``masses`` (per-expert claimed-token mass, pooled over every
      round) are so concentrated that the *effective* number of experts in use,
      ``exp(entropy(utilization))``, drops below ``merged_effective_frac * n_experts``. This is the
      routing analogue of the original merged-regime detector: instead of "one COMPONENT secretly
      covers two regimes," it is "routing has secretly collapsed onto fewer experts than exist,"
      measured with the same entropy-of-mass machinery.
    * **shattered** -- the original shattered detector flagged near-duplicate components via
      ``fuzzy_nerve`` edge weight; here that is generalized across time: (a) per-ROUND utilization is
      so unstable (large round-to-round total-variation distance in per-expert mass fractions) that no
      expert sees a consistent token distribution to specialize on, and/or (b) the POOLED nerve has
      strong overlap edges (``fuzzy_nerve``'s literal near-duplicate-component signal) across a large
      fraction of expert pairs, meaning experts are not actually claiming distinguishable token sets.

    A well-balanced run (near-uniform utilization, stable round to round) trips neither flag.
    """
    from mixle.utils.hvis.topology import fuzzy_nerve

    if not routing_history:
        raise ValueError("expert_collapse_receipt requires at least one routing-weight snapshot")

    rounds = [np.asarray(r, dtype=np.float64) for r in routing_history]
    n_experts = rounds[0].shape[1]
    if any(r.shape[1] != n_experts for r in rounds):
        raise ValueError("every round in routing_history must have the same number of experts")

    per_round_util = []
    for r in rounds:
        nerve = fuzzy_nerve(r)
        masses = nerve["masses"]
        total = masses.sum()
        per_round_util.append(masses / total if total > 0 else np.full(n_experts, 1.0 / n_experts))

    pooled = np.concatenate(rounds, axis=0)
    pooled_nerve = fuzzy_nerve(pooled)
    pooled_masses = pooled_nerve["masses"]
    pooled_total = pooled_masses.sum()
    utilization = pooled_masses / pooled_total if pooled_total > 0 else np.full(n_experts, 1.0 / n_experts)

    eps = 1.0e-12
    entropy = float(-np.sum(utilization * np.log(utilization + eps)))
    effective_experts = float(np.exp(entropy))
    merged = effective_experts < merged_effective_frac * n_experts

    if len(per_round_util) >= 2:
        tv_dists = [
            0.5 * float(np.abs(per_round_util[i] - per_round_util[i + 1]).sum()) for i in range(len(per_round_util) - 1)
        ]
        instability = float(np.mean(tv_dists))
    else:
        instability = 0.0

    n_pairs = n_experts * (n_experts - 1) / 2
    strong_edges = [w for w in pooled_nerve["edges"].values() if w >= shattered_edge_threshold]
    edge_frac = (len(strong_edges) / n_pairs) if n_pairs > 0 else 0.0

    shattered = instability > shattered_instability or edge_frac > shattered_edge_frac

    diagnosis = []
    if merged:
        diagnosis.append(
            f"routing MERGED: effective experts in use {effective_experts:.2f} of {n_experts} "
            f"(< {merged_effective_frac:.0%} threshold) -- utilization has collapsed onto a handful of experts."
        )
    if shattered:
        diagnosis.append(
            f"routing SHATTERED: round-to-round utilization instability {instability:.2f} "
            f"(threshold {shattered_instability:.2f}) or expert-overlap edge fraction {edge_frac:.2f} "
            f"(threshold {shattered_edge_frac:.2f}) -- no expert is seeing a consistent, learnable token distribution."
        )
    if not diagnosis:
        diagnosis.append("routing balanced: no merged or shattered regime detected.")

    return {
        "n_experts": int(n_experts),
        "n_rounds": len(rounds),
        "utilization": utilization.tolist(),
        "effective_experts": effective_experts,
        "entropy": entropy,
        "instability": instability,
        "edge_fraction": edge_frac,
        "merged": bool(merged),
        "shattered": bool(shattered),
        "diagnosis": diagnosis,
    }
