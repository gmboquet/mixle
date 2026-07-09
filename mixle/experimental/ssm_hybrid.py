"""E5 (part 2): the hybrid block -- local attention + selective-scan SSM + E2's moment-closure far field,
composed in ONE ``ContextMechanism``, with a real per-mechanism contribution receipt. See
``notes/designs/E5.md`` for the full design: why these three mechanisms (exact short-range, smooth
long-range, sparse extreme-long-tail) are complementary rather than redundant, the exact fusion
architecture (near+far combined by E2's own joint softmax; that combined attention branch and the SSM
branch fused by a separate learned 2-way gate), and why the contribution receipt is a real softmax-mass
reading rather than a fabricated importance score.

Reuses, without reimplementing:

- ``mixle.experimental.context_spine``: ``_rope_angles``/``_apply_rope``/``SlidingWindowState`` (E1's near
  field, exactly the code path ``SlidingWindowSpine.step`` uses).
- ``mixle.experimental.moment_closure_attention``: ``ClusterBank``/``_empty_cluster_bank``/
  ``mgf_cluster_attention``/``cluster_responsibilities``/``update_cluster_bank``/``birth_and_merge`` (E2's
  far field, verbatim).
- ``mixle.experimental.selective_scan``: ``_scan_layer``/``_s4d_real_a_log_init``/``_dt_bias_init`` (E5 part
  1's S6 recurrence and its verified init, verbatim -- the ONE scan implementation, not a second one).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mixle.experimental.context_spine import SlidingWindowState
from mixle.experimental.graduation import REGISTRY, ExperimentalMechanism
from mixle.experimental.moment_closure_attention import (
    ClusterBank,
    _empty_cluster_bank,
    birth_and_merge,
    cluster_responsibilities,
    mgf_cluster_attention,
    update_cluster_bank,
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


if _HAS_TORCH:

    class HybridBlock(nn.Module):
        """``ContextMechanism`` (E1 protocol): per layer, per position, combines (a) E1-style windowed
        exact attention, (b) E2's far-field ``ClusterBank`` mixture attention -- (a)+(b) joined by ONE
        softmax, per E2.md section 3.3 -- and (c) a selective-scan SSM branch (E5 part 1's ``_scan_layer``),
        fused with the combined attention output via a learned per-position 2-way softmax gate (notes/
        designs/E5.md section 2). ``report()`` exposes the real per-mechanism contribution receipt after a
        ``step()`` call (section 3), an instance-level side channel populated by ``step`` the same way
        ``MomentClosureAttention.last_misfit``/``last_receipts`` are.
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
            assert d_model % n_head == 0
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
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

            self.last_contributions: dict[str, float] = {}
            self.last_receipts: list[dict] = []

        def init_state(self, batch_size: int, *, device: str = "cpu") -> HybridState:
            del batch_size
            dev = torch.device(device)
            near = SlidingWindowState(cache_k=[None] * self.n_layer, cache_v=[None] * self.n_layer, pos=0)
            banks = [
                _empty_cluster_bank(self.n_head, self.max_clusters, self.head_dim, device=dev, dtype=torch.float32)
                for _ in range(self.n_layer)
            ]
            return HybridState(near=near, banks=banks, ssm_h=[None] * self.n_layer)

        def detach(self, state: HybridState) -> HybridState:
            near = SlidingWindowState(
                cache_k=[t.detach() if t is not None else None for t in state.near.cache_k],
                cache_v=[t.detach() if t is not None else None for t in state.near.cache_v],
                pos=state.near.pos,
            )
            banks = [b.detach() for b in state.banks]
            ssm_h = [h.detach() if h is not None else None for h in state.ssm_h]
            return HybridState(near=near, banks=banks, ssm_h=ssm_h)

        def step(self, state: HybridState, chunk: tuple[Any, Any]) -> tuple[HybridState, Any]:
            x, y = chunk
            b, t = x.shape
            device = x.device
            query_positions = torch.arange(state.near.pos, state.near.pos + t, device=device)

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            new_banks: list[ClusterBank] = []
            new_ssm_h: list[Any] = []
            receipts: list[dict] = []
            local_shares: list[float] = []
            far_shares: list[float] = []
            ssm_shares: list[float] = []

            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)

                # ---- attention branch: E1 near field + E2 far field, one joint softmax ----------------
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

                cache_k, cache_v = state.near.cache_k[layer], state.near.cache_v[layer]
                if cache_k is not None:
                    cache_len = cache_k.shape[1]
                    key_positions = torch.arange(state.near.pos - cache_len, state.near.pos + t, device=device)
                    k_full = torch.cat([cache_k, k], dim=1)
                    v_full = torch.cat([cache_v, v], dim=1)
                else:
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
                far_out, far_logits = mgf_cluster_attention(q, bank)
                n_c = bank.n_clusters

                if n_c > 0:
                    far_logits_bh = far_logits.permute(0, 3, 1, 2)
                    combined = torch.cat([near_logits, far_logits_bh], dim=-1)
                    weights = combined.softmax(dim=-1)
                    near_w = weights[..., : key_positions.shape[0]]
                    far_w = weights[..., key_positions.shape[0] :]
                    near_out = near_w @ vh
                    far_out_bh = far_out.permute(0, 3, 1, 2, 4)
                    far_contrib = torch.einsum("bhtc,bhtcd->bhtd", far_w, far_out_bh)
                    attn_out = (near_out + far_contrib).transpose(1, 2).reshape(b, t, self.d_model)
                    near_mass = near_w.sum(dim=-1)  # (b, n_head, t)
                    far_mass = far_w.sum(dim=-1)
                else:
                    weights = near_logits.softmax(dim=-1)
                    attn_out = (weights @ vh).transpose(1, 2).reshape(b, t, self.d_model)
                    near_mass = weights.sum(dim=-1)
                    far_mass = torch.zeros_like(near_mass)

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

                keep = self.window
                new_cache_k.append(k_full[:, -keep:])
                new_cache_v.append(v_full[:, -keep:])
                new_ssm_h.append(h_ssm_last)

                bank, receipt = birth_and_merge(
                    bank,
                    k.detach(),
                    v.detach(),
                    birth_threshold=self.birth_threshold,
                    merge_threshold=self.merge_threshold,
                )
                if bank.n_clusters > 0:
                    resp = cluster_responsibilities(k, bank)
                    bank = update_cluster_bank(bank, k, v, resp)
                new_banks.append(bank)
                receipts.append(receipt)

                # ---- contribution receipt (notes/designs/E5.md section 3): real softmax-mass reading --
                attn_share = float(g_attn.detach().mean())
                ssm_share = float(g_ssm.detach().mean())
                local_shares.append(attn_share * float(near_mass.detach().mean()))
                far_shares.append(attn_share * float(far_mass.detach().mean()))
                ssm_shares.append(ssm_share)

            logits = self.head(self.ln_f(h))
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))

            new_near = SlidingWindowState(cache_k=new_cache_k, cache_v=new_cache_v, pos=state.near.pos + t)
            new_state = HybridState(near=new_near, banks=new_banks, ssm_h=new_ssm_h)

            self.last_receipts = receipts
            self.last_contributions = {
                "local": float(sum(local_shares) / len(local_shares)),
                "far_field": float(sum(far_shares) / len(far_shares)),
                "ssm": float(sum(ssm_shares) / len(ssm_shares)),
            }
            return new_state, loss

        def report(self) -> dict[str, float]:
            """The per-mechanism contribution receipt from the most recent ``step()`` call: fractional
            share of the fused output attributable to each of (``local``, ``far_field``, ``ssm``), summing
            to 1.0 by construction (notes/designs/E5.md section 3) -- a real reading of the learned gate's
            and joint softmax's own weights, not a fabricated importance score."""
            return dict(self.last_contributions)

        def log_density(self, x: Any, y: Any) -> Any:
            """``x, y``: ``(n, T)`` long tensors. Returns ``-mean_per_position_nll`` per row, each scored
            independently (fresh state per row) via one ``init_state`` + ``step`` call, exactly the
            ``SelectiveScan.log_density`` convention (see notes/designs/E5.md section 4 for the caveat this
            inherits from E2: cluster birth/merge is only independent across rows because each row is scored
            with its own fresh, unbatched stream, not scored together)."""
            out = []
            for i in range(x.shape[0]):
                state = self.init_state(1, device=str(x.device))
                _, mean_nll = self.step(state, (x[i : i + 1], y[i : i + 1]))
                out.append(-mean_nll)
            return torch.stack(out)

    REGISTRY.register(ExperimentalMechanism(name="ssm_hybrid"))
