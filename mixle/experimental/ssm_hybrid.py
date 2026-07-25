"""E5 (part 2): the hybrid block -- local attention + selective-scan SSM + E2's moment-closure far field,
composed in ONE ``ContextMechanism``, with an explicit routing-mass receipt. See
``notes/designs/E5.md`` for the full design: why these three mechanisms (exact short-range, smooth
long-range, sparse extreme-long-tail) are complementary rather than redundant, the exact fusion
architecture (near+far combined by E2's own joint softmax; that combined attention branch and the SSM
branch fused by a separate learned 2-way gate). The routing receipt reports the forward pass's gate and
attention mass; it deliberately does not claim to attribute the fused output, because branch magnitude
and cancellation make routing coefficients insufficient for output attribution.

Reuses, without reimplementing:

- ``mixle.experimental.context_spine``: ``_rope_angles``/``_apply_rope``/``SlidingWindowState`` (E1's near
  field, exactly the code path ``SlidingWindowSpine.step`` uses).
- ``mixle.experimental.moment_closure_attention``: ``ClusterBank``/``_empty_cluster_bank``/
  ``mgf_cluster_attention``/``ingest_cluster_batch`` (E2's far field, verbatim).
- ``mixle.experimental.selective_scan``: ``_scan_layer``/``_s4d_real_a_log_init``/``_dt_bias_init`` (E5 part
  1's S6 recurrence and its verified init, verbatim -- the ONE scan implementation, not a second one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from mixle.experimental.context_spine import SlidingWindowState
from mixle.experimental.graduation import REGISTRY, ExperimentalMechanism
from mixle.experimental.moment_closure_attention import (
    ClusterBank,
    _empty_cluster_bank,
    _representation_receipt,
    ingest_cluster_batch,
    mgf_cluster_attention,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

if _HAS_TORCH:
    from mixle.experimental.context_spine import _apply_rope, _rope_angles
    from mixle.experimental.selective_scan import _dt_bias_init, _s4d_real_a_log_init, _scan_layer

__all__ = ["HybridState", "HybridBlock"]


@dataclass
class HybridState:
    """Per-layer carried state: E1's near-field cache, E2's far-field ``ClusterBank``, and E5 part 1's SSM
    hidden state -- one list per mechanism, indexed by layer, matching E1/E2's existing per-layer-list
    convention so nothing about the state shape is new."""

    near: SlidingWindowState
    banks: list[Any] = field(default_factory=list)
    ssm_h: list[Any] = field(default_factory=list)
    batch_size: int = 1


if _HAS_TORCH:

    class HybridBlock(nn.Module):
        """``ContextMechanism`` (E1 protocol): per layer, per position, combines (a) E1-style windowed
        exact attention, (b) E2's far-field ``ClusterBank`` mixture attention -- (a)+(b) joined by ONE
        softmax, per E2.md section 3.3 -- and (c) a selective-scan SSM branch (E5 part 1's ``_scan_layer``),
        fused with the combined attention output via a learned per-position 2-way softmax gate (notes/
        designs/E5.md section 2). ``report()`` exposes descriptive routing mass after a ``step()`` call,
        an instance-level side channel populated by ``step`` the same way
        ``MomentClosureAttention.last_misfit``/``last_receipts`` are. Routing mass is not an output
        attribution measurement.
        """

        def __init__(
            self,
            vocab: int,
            *,
            d_model: int = 32,
            n_layer: int = 2,
            n_head: int = 2,
            window: int = 16,
            d_state: int = 16,
            ssm_expand: int = 2,
            max_clusters: int = 4,
            birth_threshold: float = -2.0,
            merge_threshold: float | None = None,
        ) -> None:
            super().__init__()
            for name, value in {
                "vocab": vocab,
                "d_model": d_model,
                "n_layer": n_layer,
                "n_head": n_head,
                "window": window,
                "d_state": d_state,
                "ssm_expand": ssm_expand,
                "max_clusters": max_clusters,
            }.items():
                if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
                    raise ValueError(f"{name} must be a positive exact integer.")
            if d_model % n_head:
                raise ValueError("d_model must be divisible by n_head.")
            if (d_model // n_head) % 2:
                raise ValueError("d_model / n_head must be even for RoPE.")
            if not math.isfinite(float(birth_threshold)):
                raise ValueError("birth_threshold must be finite.")
            if merge_threshold is not None and not math.isfinite(float(merge_threshold)):
                raise ValueError("merge_threshold must be finite when provided.")

            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = self.d_model // self.n_head
            self.window = int(window)
            self.d_state = int(d_state)
            self.ssm_expand = int(ssm_expand)
            self.d_inner = self.ssm_expand * self.d_model
            self.max_clusters = int(max_clusters)
            self.birth_threshold = float(birth_threshold)
            self.merge_threshold = merge_threshold

            self.tok = nn.Embedding(vocab, d_model)
            self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
            self.mlp = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
                    for _ in range(n_layer)
                ]
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.head.weight = self.tok.weight  # weight tying, matching every other Track-E mechanism

            # -- attention branch (near + far, E1 qkv + E2 cluster bank) --------------------------------
            self.qkv = nn.ModuleList([nn.Linear(d_model, 3 * d_model) for _ in range(n_layer)])
            self.attn_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layer)])

            # -- SSM branch (E5 part 1's _scan_layer, same params SelectiveScan.__init__ builds) ---------
            self.in_proj_ssm = nn.ModuleList([nn.Linear(d_model, self.d_inner) for _ in range(n_layer)])
            self.W_delta = nn.ModuleList([nn.Linear(self.d_inner, self.d_inner) for _ in range(n_layer)])
            self.W_B = nn.ModuleList([nn.Linear(self.d_inner, d_state) for _ in range(n_layer)])
            self.W_C = nn.ModuleList([nn.Linear(self.d_inner, d_state) for _ in range(n_layer)])
            self.out_proj_ssm = nn.ModuleList([nn.Linear(self.d_inner, d_model) for _ in range(n_layer)])
            self.A_log = nn.Parameter(
                torch.stack([_s4d_real_a_log_init(self.d_inner, d_state) for _ in range(n_layer)])
            )
            self.A_log._no_weight_decay = True
            self.D = nn.Parameter(torch.ones(n_layer, self.d_inner))
            self.D._no_weight_decay = True
            with torch.no_grad():
                for layer in range(n_layer):
                    self.W_delta[layer].bias.copy_(_dt_bias_init(self.d_inner))
                    self.W_delta[layer].bias._no_reinit = True

            # -- fusion gate: per-position 2-way softmax over (attention branch, SSM branch) --------------
            self.gate = nn.ModuleList([nn.Linear(d_model, 2) for _ in range(n_layer)])

            self.last_routing_mass: dict[str, float] = {}
            self.last_receipts: list[dict] = []

        def init_state(self, batch_size: int, *, device: str = "cpu") -> HybridState:
            if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or int(batch_size) <= 0:
                raise ValueError("batch_size must be a positive exact integer.")
            batch_size = int(batch_size)
            dev = torch.device(device)
            if dev != self.tok.weight.device:
                raise ValueError("state device must match the model device.")
            near = SlidingWindowState(
                cache_k=[None] * self.n_layer,
                cache_v=[None] * self.n_layer,
                pos=0,
                batch_size=batch_size,
                n_head=self.n_head,
                head_dim=self.head_dim,
                device=dev,
            )
            banks = [
                _empty_cluster_bank(
                    self.n_head,
                    self.max_clusters,
                    self.head_dim,
                    device=dev,
                    dtype=self.tok.weight.dtype,
                )
                for _ in range(self.n_layer)
            ]
            ssm_h = [
                torch.zeros(
                    batch_size,
                    self.d_inner,
                    self.d_state,
                    device=dev,
                    dtype=self.A_log.dtype,
                )
                for _ in range(self.n_layer)
            ]
            return HybridState(near=near, banks=banks, ssm_h=ssm_h, batch_size=batch_size)

        def detach(self, state: HybridState) -> HybridState:
            near = SlidingWindowState(
                cache_k=[t.detach() if t is not None else None for t in state.near.cache_k],
                cache_v=[t.detach() if t is not None else None for t in state.near.cache_v],
                pos=state.near.pos,
            )
            banks = [b.detach() for b in state.banks]
            ssm_h = [h.detach() if h is not None else None for h in state.ssm_h]
            return HybridState(near=near, banks=banks, ssm_h=ssm_h, batch_size=state.batch_size)

        def step(self, state: HybridState, chunk: tuple[Any, Any]) -> tuple[HybridState, Any]:
            if not isinstance(state, HybridState):
                raise TypeError("state must be a HybridState.")
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                raise TypeError("chunk must be an (input_tokens, target_tokens) tuple.")
            x, y = chunk
            if not torch.is_tensor(x) or not torch.is_tensor(y) or x.dtype != torch.long or y.dtype != torch.long:
                raise TypeError("input and target tokens must be torch.long tensors.")
            if x.ndim != 2 or y.shape != x.shape or x.shape[0] == 0 or x.shape[1] == 0:
                raise ValueError("input and target tokens must have equal non-empty (batch, time) shape.")
            b, t = x.shape
            if state.batch_size != b:
                raise ValueError(f"state batch_size={state.batch_size} does not match chunk batch_size={b}.")
            if isinstance(state.near.pos, bool) or not isinstance(state.near.pos, Integral) or state.near.pos < 0:
                raise ValueError("state.near.pos must be a non-negative exact integer.")
            if (
                len(state.near.cache_k) != self.n_layer
                or len(state.near.cache_v) != self.n_layer
                or len(state.banks) != self.n_layer
                or len(state.ssm_h) != self.n_layer
            ):
                raise ValueError("state must contain exactly one near cache, far bank, and SSM state per layer.")
            if x.device != self.tok.weight.device or y.device != x.device:
                raise ValueError("state, model, input, and target tensors must be on the same device.")
            if bool(((x < 0) | (x >= self.vocab) | (y < 0) | (y >= self.vocab)).any().item()):
                raise ValueError(f"input and target token IDs must lie in [0, {self.vocab}).")
            device = x.device
            query_positions = torch.arange(state.near.pos, state.near.pos + t, device=device)

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            new_banks: list[ClusterBank] = []
            new_ssm_h: list[Any] = []
            receipts: list[dict] = []
            local_routing_masses: list[float] = []
            far_routing_masses: list[float] = []
            ssm_routing_masses: list[float] = []

            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)

                # ---- attention branch: E1 near field + E2 far field, one joint softmax ----------------
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

                cache_k, cache_v = state.near.cache_k[layer], state.near.cache_v[layer]
                if cache_k is not None:
                    if (
                        cache_v is None
                        or cache_k.shape != cache_v.shape
                        or cache_k.ndim != 4
                        or cache_k.shape[0] != b
                        or cache_k.shape[2:] != (self.n_head, self.head_dim)
                        or cache_k.shape[1] > self.window
                        or cache_k.device != device
                        or cache_k.dtype != k.dtype
                    ):
                        raise ValueError(f"layer {layer} near K/V cache has an invalid layout.")
                    cache_len = cache_k.shape[1]
                    _representation_receipt(
                        state.banks[layer],
                        batch_size=b,
                        positions_seen=state.near.pos,
                        near_tokens_per_stream=cache_len,
                        evicted_tokens_per_stream=0,
                    )
                    key_positions = torch.arange(state.near.pos - cache_len, state.near.pos + t, device=device)
                    k_full = torch.cat([cache_k, k], dim=1)
                    v_full = torch.cat([cache_v, v], dim=1)
                else:
                    if cache_v is not None:
                        raise ValueError(f"layer {layer} has a partial near cache.")
                    cache_len = 0
                    _representation_receipt(
                        state.banks[layer],
                        batch_size=b,
                        positions_seen=state.near.pos,
                        near_tokens_per_stream=0,
                        evicted_tokens_per_stream=0,
                    )
                    key_positions = query_positions
                    k_full, v_full = k, v

                sin_q, cos_q = _rope_angles(query_positions, self.head_dim)
                sin_k, cos_k = _rope_angles(key_positions, self.head_dim)
                q_rope = _apply_rope(q, sin_q, cos_q)
                k_full_rope = _apply_rope(k_full, sin_k, cos_k)

                delta = query_positions[:, None] - key_positions[None, :]
                allowed = (delta >= 0) & (delta < self.window)
                near_mask = torch.zeros(t, key_positions.shape[0], device=device)
                near_mask = near_mask.masked_fill(~allowed, float("-inf"))

                qh = q_rope.transpose(1, 2)
                kh = k_full_rope.transpose(1, 2)
                vh = v_full.transpose(1, 2)
                near_logits = (qh @ kh.transpose(-2, -1)) / (self.head_dim**0.5)
                near_logits = near_logits + near_mask[None, None]

                bank = state.banks[layer]
                expected_bank_shape = (self.n_head, self.max_clusters, self.head_dim)
                if (
                    not isinstance(bank, ClusterBank)
                    or bank.mu_k.shape != expected_bank_shape
                    or bank.mu_v.shape != expected_bank_shape
                    or bank.count.shape != expected_bank_shape[:2]
                    or bank.sigma_kk.shape != expected_bank_shape
                    or bank.sigma_vk.shape != (*expected_bank_shape, self.head_dim)
                    or bank.mu_k.device != device
                    or bank.mu_k.dtype != k.dtype
                ):
                    raise ValueError(f"layer {layer} far-field bank has an invalid layout.")

                # Advance the far bank at each query's exact window crossing. A post-chunk update leaves
                # a gap for early queries in large chunks; a pre-chunk update exposes future evictions.
                # This scan makes each past token visible in exactly one of the near/far representations.
                keep = min(self.window, k_full.shape[1])
                evicted = k_full.shape[1] - keep
                processed = 0
                ingestion_receipts: list[dict[str, Any]] = []
                attention_outputs: list[Any] = []
                near_masses: list[Any] = []
                far_masses: list[Any] = []
                for query_index in range(t):
                    target = max(0, cache_len + query_index + 1 - self.window)
                    if target > evicted:
                        raise RuntimeError("local-window eviction schedule is inconsistent with the far-field bank.")
                    while processed < target:
                        bank, ingestion = ingest_cluster_batch(
                            bank,
                            k_full[:, processed : processed + 1],
                            v_full[:, processed : processed + 1],
                            birth_threshold=self.birth_threshold,
                            merge_threshold=self.merge_threshold,
                        )
                        ingestion_receipts.append(ingestion)
                        processed += 1

                    query_near_logits = near_logits[:, :, query_index : query_index + 1]
                    far_out, far_logits = mgf_cluster_attention(q[:, query_index : query_index + 1], bank)
                    if bank.n_clusters > 0:
                        far_logits_bh = far_logits.permute(0, 3, 1, 2)
                        combined = torch.cat([query_near_logits, far_logits_bh], dim=-1)
                        weights = combined.softmax(dim=-1)
                        near_w = weights[..., : key_positions.shape[0]]
                        far_w = weights[..., key_positions.shape[0] :]
                        near_out = near_w @ vh
                        far_out_bh = far_out.permute(0, 3, 1, 2, 4)
                        far_contrib = torch.einsum("bhtc,bhtcd->bhtd", far_w, far_out_bh)
                        attention_outputs.append(near_out + far_contrib)
                        near_masses.append(near_w.sum(dim=-1))
                        far_masses.append(far_w.sum(dim=-1))
                    else:
                        near_w = query_near_logits.softmax(dim=-1)
                        attention_outputs.append(near_w @ vh)
                        near_masses.append(near_w.sum(dim=-1))
                        far_masses.append(near_w.new_zeros(b, self.n_head, 1))

                if processed != evicted:
                    raise RuntimeError("far-field bank did not consume every token that left the local window.")
                attn_out = torch.cat(attention_outputs, dim=2).transpose(1, 2).reshape(b, t, self.d_model)
                near_mass = torch.cat(near_masses, dim=2)
                far_mass = torch.cat(far_masses, dim=2)

                attn_out = self.attn_proj[layer](attn_out)

                # ---- SSM branch: E5 part 1's _scan_layer, shared, not reimplemented -------------------
                u = self.in_proj_ssm[layer](hn)
                h_ssm_last, y_ssm = _scan_layer(
                    u,
                    self.A_log[layer],
                    self.W_delta[layer],
                    self.W_B[layer],
                    self.W_C[layer],
                    self.D[layer],
                    state.ssm_h[layer],
                )
                ssm_out = self.out_proj_ssm[layer](y_ssm)

                # ---- fusion: learned 2-way gate over (attention branch, SSM branch) --------------------
                gate_logits = self.gate[layer](hn)  # (b, t, 2)
                gate_w = gate_logits.softmax(dim=-1)
                g_attn, g_ssm = gate_w[..., 0], gate_w[..., 1]  # each (b, t)
                mix_out = g_attn.unsqueeze(-1) * attn_out + g_ssm.unsqueeze(-1) * ssm_out

                h = h + mix_out
                h = h + self.mlp[layer](self.ln2[layer](h))

                new_cache_k.append(k_full[:, -keep:])
                new_cache_v.append(v_full[:, -keep:])
                new_ssm_h.append(h_ssm_last)

                receipt = {
                    "query_aligned": True,
                    "ingestion_steps": len(ingestion_receipts),
                    "birthed": any(item["birthed"] for item in ingestion_receipts),
                    "merged": [pair for item in ingestion_receipts for pair in item["merged"]],
                    "soft_update_applied": any(item["soft_update_applied"] for item in ingestion_receipts),
                    "expected_ingested_tokens_per_head": b * evicted,
                    "count_conserved": all(item["count_conserved"] for item in ingestion_receipts),
                }
                receipt["accounting"] = _representation_receipt(
                    bank,
                    batch_size=b,
                    positions_seen=state.near.pos + t,
                    near_tokens_per_stream=keep,
                    evicted_tokens_per_stream=evicted,
                )
                receipt["accounting"]["query_aligned"] = True
                receipt["accounting"]["dual_visible_position_count"] = 0
                new_banks.append(bank)
                receipts.append(receipt)

                # Routing coefficients are averaged jointly, preserving correlation between the fusion
                # gate and the near/far attention split. They are not output attribution measurements.
                g_attn_by_head = g_attn.detach().unsqueeze(1)
                local_routing_masses.append(float((g_attn_by_head * near_mass.detach()).mean()))
                far_routing_masses.append(float((g_attn_by_head * far_mass.detach()).mean()))
                ssm_routing_masses.append(float(g_ssm.detach().mean()))

            logits = self.head(self.ln_f(h))
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))

            new_near = SlidingWindowState(cache_k=new_cache_k, cache_v=new_cache_v, pos=state.near.pos + t)
            new_state = HybridState(
                near=new_near,
                banks=new_banks,
                ssm_h=new_ssm_h,
                batch_size=state.batch_size,
            )

            self.last_receipts = receipts
            self.last_routing_mass = {
                "local_routing_mass": float(sum(local_routing_masses) / len(local_routing_masses)),
                "far_field_routing_mass": float(sum(far_routing_masses) / len(far_routing_masses)),
                "ssm_routing_mass": float(sum(ssm_routing_masses) / len(ssm_routing_masses)),
            }
            return new_state, loss

        def report(self) -> dict[str, float]:
            """Return routing mass from the most recent ``step()`` call.

            The three masses sum to one because they read the fusion gate and conditional near/far
            attention weights. They describe routing only. They do not estimate fractional output
            attribution, which would additionally depend on branch vector magnitude and cancellation.
            """
            return dict(self.last_routing_mass)

        def log_density(self, x: Any, y: Any) -> Any:
            """Return conditional sequence log likelihoods ``sum_t log p(y_t | x_<=t)``.

            ``x`` and ``y`` are equal-shape, non-empty ``(n_sequences, time)`` long tensors. Each row is
            scored with a fresh unbatched state so cluster birth and merge cannot couple observations.
            """
            if not torch.is_tensor(x) or not torch.is_tensor(y) or x.dtype != torch.long or y.dtype != torch.long:
                raise TypeError("x and y must be torch.long tensors.")
            if x.ndim != 2 or y.shape != x.shape or x.shape[0] == 0 or x.shape[1] == 0:
                raise ValueError("x and y must have equal non-empty (n_sequences, time) shape.")
            out = []
            for i in range(x.shape[0]):
                state = self.init_state(1, device=str(x.device))
                _, mean_nll = self.step(state, (x[i : i + 1], y[i : i + 1]))
                out.append(-mean_nll * x.shape[1])
            return torch.stack(out)

    REGISTRY.register(ExperimentalMechanism(name="ssm_hybrid"))
