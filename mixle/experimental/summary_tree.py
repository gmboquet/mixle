"""E4: hierarchical summary tree + multi-scale objective -- see ``notes/designs/E4.md`` for the full
derivation (persistent append-only tree over evicted tokens, tree-path positional encoding, the
predict-the-summary auxiliary loss, the stop-gradient horizon receipt). This module implements that
note section-by-section; see the note's "Implementation notes vs. this design" section for the small,
honestly-documented places this module simplifies the note's scheme for tractability.

**What this is.** E1's :class:`~mixle.experimental.context_spine.SlidingWindowSpine` keeps an exact
but bounded KV window; anything evicted from that window is gone. E4 keeps it: every evicted token is
folded, one at a time, into a persistent tree of learned summaries via mixed-radix carry propagation
(the fast-multipole-method structure -- near field exact, far field via a bounded number of
increasingly coarse representatives -- applied to token history). Tree depth grows only as
``log_fanout(evicted_count)``, so the far-field attention set stays bounded regardless of how much
history has streamed through.

**Positional encoding.** RoPE's ``q . k`` dependence on ``i - j`` is well-conditioned only when
``i - j`` is a small, well-scaled number (E1's window); a far-field summary node represents a *range*
of possibly billions of original positions, and there is no single ``j`` to rotate by. E4 replaces
RoPE for the far field with (a) a content channel -- a level embedding plus a sibling-slot embedding
summed into the node's summary before it's used as an attention key -- and (b) a relative bias
channel -- a learned scalar indexed by tree distance (``lca_depth``, see below), the ALiBi/T5-bias
shape of mechanism but indexed by tree distance instead of linear offset. Near-field window tokens
keep ordinary RoPE unchanged (E1's ``_rope_angles``/``_apply_rope``, reused verbatim).

**Predict-the-summary auxiliary loss.** Every node, the moment it's finalized, is scored against the
exact additive token-id histogram of the leaves it covers via one shared linear head
(``d_model -> vocab``) and cross-entropy against the normalized histogram. This is direct supervision
independent of whether any future query ever attends to that node -- for a node many levels up the
tree that a training run may never query again, this is its compressor's only gradient.

**Stop-gradient horizon (receipted).** A node moves from ``live`` to ``archived`` once ``H`` further
nodes have finalized at its own level after it; ``.summary`` is ``.detach()``-ed at that moment and
never un-detaches. Archived nodes stay forward-visible (attendable) but stop receiving gradient. See
:meth:`SummaryTreeSpine.detach` and ``mixle/tests/summary_tree_test.py`` for the exact-accounting
receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from mixle.experimental.context_spine import SlidingWindowState
from mixle.experimental.graduation import REGISTRY, ExperimentalMechanism

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

if _HAS_TORCH:
    from mixle.experimental.context_spine import _apply_rope, _rope_angles

__all__ = [
    "TreeNode",
    "SummaryTreeState",
    "SummaryTreeSpine",
    "digits_of",
    "lca_depth",
]


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("mixle.experimental.summary_tree requires torch.")


def digits_of(n: int, base: int) -> tuple[int, ...]:
    """Base-``base`` digits of ``n``, least-significant first. ``digits_of(0, base) == (0,)``."""
    if n == 0:
        return (0,)
    out = []
    while n > 0:
        out.append(n % base)
        n //= base
    return tuple(out)


def lca_depth(query_pos: int, node_level: int, node_g: int, fanout: int, *, max_climb: int = 64) -> int:
    """Tree distance between a query at absolute position ``query_pos`` and a node covering the
    contiguous leaf range ``[g * fanout**level, (g + 1) * fanout**level)`` (``g`` = the node's own
    0-based sequential index among nodes finalized at its level -- see E4.md's carry-propagation
    section). Returns how many levels above ``node_level`` the query's ancestor chain must climb
    before it lands in the same subtree as the node -- see ``notes/designs/E4.md``'s "Implementation
    notes" section for why this integer recurrence is the exact algebraic equivalent of "matching
    leading digits of the two paths" without materializing ``query_pos``'s (potentially ~1e9-long)
    digit expansion. Pure function of ``(query_pos, node_level, node_g, fanout)`` -- no dependence on
    how the stream was chunked, which is what makes it stable under re-chunking (Acceptance §3)."""
    m = node_level
    q_anc = query_pos // (fanout**m)
    n_anc = node_g
    while q_anc != n_anc and (m - node_level) < max_climb:
        m += 1
        q_anc = query_pos // (fanout**m)
        n_anc = node_g // (fanout ** (m - node_level))
    return m - node_level


@dataclass
class TreeNode:
    """One finalized node of the persistent summary tree (E4.md's ``TreeNode``).

    ``summary`` carries one entry per layer (each ``(batch, d_model)``) since every layer's tree is
    built from that layer's own ``(k, v)`` via that layer's own ``qkv`` projection, but the bookkeeping
    fields below (``histogram``/``path``/``level``/``g``) are identical across layers -- they describe
    which evicted tokens this node covers, not any layer-specific content -- so they're stored once.
    """

    summary: list[Any]  # len n_layer, each (batch, d_model), live (requires_grad) until detached
    histogram: Any  # (batch, vocab) exact additive sufficient statistic (predict-the-summary target)
    path: tuple[int, ...]  # base-fanout digits of `g`, least-significant first
    level: int  # 1 = first compressed level (groups of `fanout` evicted tokens)
    g: int  # this node's own 0-based sequential index among nodes finalized at `level`
    finalized_step: int
    finalized_index_within_level: int  # value of level_finalized_count[level] at finalization time
    detached: bool = False
    detached_at_finalized_count: int | None = None


@dataclass
class SummaryTreeState:
    """``ContextMechanism`` carried state: E1's exact near field plus the persistent far-field tree."""

    window: SlidingWindowState  # E1's near field, unmodified
    cached_ids: Any | None  # (batch, cache_len) token ids aligned with window's cache -- shared across layers
    pending_leaf: list  # buffered evicted (k_per_layer, v_per_layer, id) tuples not yet forming a level-1 node
    pending: list[list[TreeNode]]  # index i = level (i + 2)'s buffered children (level-1's buffer is pending_leaf)
    live: list[list[TreeNode]]  # index i = level (i + 1): finalized, not-yet-detached nodes
    archived: list[list[TreeNode]]  # index i = level (i + 1): finalized, detached nodes
    level_finalized_count: list[int]  # index i = level (i + 1): total nodes ever finalized at that level
    evicted_count: int = 0


def _ensure_level(lists: list[list], level: int) -> None:
    """Grow a per-level bookkeeping list (0-indexed by ``level - 1``) so index ``level - 1`` exists."""
    while len(lists) < level:
        lists.append([])


def _ensure_count_level(counts: list[int], level: int) -> None:
    while len(counts) < level:
        counts.append(0)


if _HAS_TORCH:

    class _Compressor(nn.Module):
        """Shared pooling module (one instance for every tree level, per E4.md's "weight-tied, don't
        grow parameters with scale" convention -- mirrors E1's ``head.weight = tok.weight``). A
        level-conditioned linear input adapter handles "children are raw ``(k, v)`` pairs" (level 1)
        vs. "children are summary vectors" (level > 1) without duplicating the pooling/MLP weights.
        """

        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.leaf_adapter = nn.Linear(2 * d_model, d_model)  # concat(k, v) -> d_model, level == 1
            self.node_adapter = nn.Linear(d_model, d_model)  # summary -> d_model, level > 1
            self.pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
            self.pool_key = nn.Linear(d_model, d_model)
            self.pool_val = nn.Linear(d_model, d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
            self.out_norm = nn.LayerNorm(d_model)

        def forward(self, children: Any, *, is_leaf_level: bool) -> Any:
            """``children``: ``(batch, fanout, 2*d_model)`` if ``is_leaf_level`` else ``(batch, fanout,
            d_model)``. Returns ``(batch, d_model)`` -- an attention-pool over the fanout children
            followed by an MLP (E4.md's ``Compressor_L``)."""
            content = self.leaf_adapter(children) if is_leaf_level else self.node_adapter(children)
            b = content.shape[0]
            q = self.pool_query.expand(b, 1, -1)
            k = self.pool_key(content)
            v = self.pool_val(content)
            attn = (q @ k.transpose(-2, -1)) / (content.shape[-1] ** 0.5)  # (b, 1, fanout)
            w = attn.softmax(dim=-1)
            pooled = (w @ v).squeeze(1)  # (b, d_model)
            return self.out_norm(pooled + self.mlp(pooled))

    class SummaryTreeSpine(nn.Module):
        """E4: ``SlidingWindowSpine``'s exact near field plus a persistent, bounded far-field tree of
        learned summaries, merged into one joint softmax per layer (E4.md's "Far-field attention:
        merging into E1's score"). See the module docstring for the positional encoding and auxiliary
        loss this adds on top of E1 unchanged.
        """

        def __init__(
            self,
            vocab: int,
            *,
            d_model: int = 32,
            n_layer: int = 2,
            n_head: int = 2,
            window: int = 16,
            fanout: int = 4,
            max_level_cap: int = 24,
            detach_horizon_nodes: int = 2,
            aux_weight: float = 0.1,
        ) -> None:
            super().__init__()
            _require_torch()
            assert d_model % n_head == 0
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
            self.window = int(window)
            self.fanout = int(fanout)
            assert self.fanout >= 2
            self.max_level_cap = int(max_level_cap)
            self.detach_horizon_nodes = int(detach_horizon_nodes)
            self.aux_weight = float(aux_weight)

            self.tok = nn.Embedding(vocab, d_model)
            self.qkv = nn.ModuleList([nn.Linear(d_model, 3 * d_model) for _ in range(n_layer)])
            self.proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layer)])
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
            self.head.weight = self.tok.weight

            self.compressor = _Compressor(d_model)  # one shared module across levels AND layers
            self.predict_head = nn.Linear(d_model, vocab)  # predict-the-summary aux loss head, shared

            # tree-path positional encoding (content channel, E4.md "Implementation notes" §1)
            self.level_embed = nn.Embedding(self.max_level_cap, d_model)
            self.slot_embed = nn.Embedding(self.fanout, d_model)
            nn.init.normal_(self.level_embed.weight, std=0.02)  # small init: content bias augments, doesn't dominate
            nn.init.normal_(self.slot_embed.weight, std=0.02)  # (matches _Compressor.pool_query's 0.02 convention)
            # relative bias channel: one learned scalar per lca_depth bucket (ALiBi/T5-bias shape)
            self.lca_bias = nn.Parameter(torch.zeros(self.max_level_cap))

            self.last_aux_loss: float = 0.0  # self-reported per-step signal, mean over nodes finalized this step

        # -----------------------------------------------------------------------------------------------
        # ContextMechanism protocol
        # -----------------------------------------------------------------------------------------------

        def init_state(self, batch_size: int, *, device: str = "cpu") -> SummaryTreeState:
            del batch_size
            window = SlidingWindowState(cache_k=[None] * self.n_layer, cache_v=[None] * self.n_layer, pos=0)
            return SummaryTreeState(
                window=window,
                cached_ids=None,
                pending_leaf=[],
                pending=[],
                live=[],
                archived=[],
                level_finalized_count=[],
                evicted_count=0,
            )

        def detach(self, state: SummaryTreeState) -> SummaryTreeState:
            window = SlidingWindowState(
                cache_k=[t.detach() if t is not None else None for t in state.window.cache_k],
                cache_v=[t.detach() if t is not None else None for t in state.window.cache_v],
                pos=state.window.pos,
            )
            cached_ids = state.cached_ids.detach() if state.cached_ids is not None else None

            # Pending buffers are also cut here -- see E4.md "Implementation notes" §3: a pending group
            # can span an arbitrary number of future steps, so leaving it attached across a TBPTT detach
            # boundary risks a second backward through already-freed graph once it finally finalizes.
            pending_leaf = [
                ([kk.detach() for kk in k_layers], [vv.detach() for vv in v_layers], ids)
                for (k_layers, v_layers, ids) in state.pending_leaf
            ]
            pending = [[self._detach_node(n, force=True) for n in level_pending] for level_pending in state.pending]

            live: list[list[TreeNode]] = []
            archived: list[list[TreeNode]] = [list(lvl) for lvl in state.archived]
            while len(archived) < len(state.live):
                archived.append([])
            for level_idx, level_live in enumerate(state.live):
                level = level_idx + 1
                finalized_total = (
                    state.level_finalized_count[level_idx] if level_idx < len(state.level_finalized_count) else 0
                )
                still_live: list[TreeNode] = []
                for node in level_live:
                    age = finalized_total - node.finalized_index_within_level
                    if age >= self.detach_horizon_nodes:
                        detached_node = self._detach_node(node, force=True)
                        detached_node.detached_at_finalized_count = finalized_total
                        archived[level_idx].append(detached_node)
                    else:
                        still_live.append(node)
                live.append(still_live)

            return SummaryTreeState(
                window=window,
                cached_ids=cached_ids,
                pending_leaf=pending_leaf,
                pending=pending,
                live=live,
                archived=archived,
                level_finalized_count=list(state.level_finalized_count),
                evicted_count=state.evicted_count,
            )

        @staticmethod
        def _detach_node(node: TreeNode, *, force: bool) -> TreeNode:
            if node.detached and not force:
                return node
            return replace(node, summary=[s.detach() for s in node.summary], detached=True)

        # -----------------------------------------------------------------------------------------------
        # Tree construction: mixed-radix carry propagation over evicted tokens (E4.md's construction
        # section) -- one token at a time, oldest first, so the result never depends on chunk boundaries.
        # -----------------------------------------------------------------------------------------------

        def _finalize_node(
            self, state: SummaryTreeState, level: int, children_summaries: list[list[Any]], histogram: Any
        ) -> TreeNode:
            """``children_summaries[layer]``: list of ``fanout`` ``(batch, d_model)`` tensors for that
            layer. Runs the shared ``compressor`` once per layer and stamps bookkeeping fields."""
            summaries_by_layer: list[Any] = []
            is_leaf_level = level == 1
            for layer in range(self.n_layer):
                stacked = torch.stack(children_summaries[layer], dim=1)  # (b, fanout, feat)
                summaries_by_layer.append(self.compressor(stacked, is_leaf_level=is_leaf_level))

            _ensure_count_level(state.level_finalized_count, level)
            g = state.level_finalized_count[level - 1]
            node = TreeNode(
                summary=summaries_by_layer,
                histogram=histogram,
                path=digits_of(g, self.fanout),
                level=level,
                g=g,
                finalized_step=state.evicted_count,
                finalized_index_within_level=g,
            )
            state.level_finalized_count[level - 1] += 1
            _ensure_level(state.live, level)
            state.live[level - 1].append(node)
            return node

        def _carry_propagate(self, state: SummaryTreeState, node: TreeNode) -> None:
            """A freshly finalized node at ``node.level`` becomes a pending child at ``node.level + 1``;
            once ``fanout`` of those accumulate, finalize the parent and recurse (E4.md's carry step)."""
            parent_level_idx = node.level - 1  # pending[parent_level_idx] holds children for level (node.level + 1)
            while len(state.pending) <= parent_level_idx:
                state.pending.append([])
            state.pending[parent_level_idx].append(node)
            if len(state.pending[parent_level_idx]) < self.fanout:
                return
            children = state.pending[parent_level_idx][: self.fanout]
            state.pending[parent_level_idx] = state.pending[parent_level_idx][self.fanout :]

            children_summaries = [[c.summary[layer] for c in children] for layer in range(self.n_layer)]
            histogram = children[0].histogram
            for c in children[1:]:
                histogram = histogram + c.histogram
            parent = self._finalize_node(state, node.level + 1, children_summaries, histogram)
            self._carry_propagate(state, parent)

        def _absorb_evicted_token(
            self, state: SummaryTreeState, k_layers: list[Any], v_layers: list[Any], token_id: Any
        ) -> None:
            """Feed ONE evicted token into the tree (E4.md construction step 1). ``k_layers``/``v_layers``:
            list of ``(batch, d_model)`` per layer (flattened across heads -- see module docstring).
            ``token_id``: ``(batch,)`` long tensor."""
            state.pending_leaf.append((k_layers, v_layers, token_id))
            if len(state.pending_leaf) < self.fanout:
                return
            group = state.pending_leaf[: self.fanout]
            state.pending_leaf = state.pending_leaf[self.fanout :]

            children_summaries = [
                [torch.cat([g[0][layer], g[1][layer]], dim=-1) for g in group] for layer in range(self.n_layer)
            ]
            histogram = None
            for _, _, tid in group:
                onehot = F.one_hot(tid, num_classes=self.vocab).to(children_summaries[0][0].dtype)
                histogram = onehot if histogram is None else histogram + onehot
            node = self._finalize_node(state, 1, children_summaries, histogram)
            self._carry_propagate(state, node)

        # -----------------------------------------------------------------------------------------------
        # Far-field attention set + predict-the-summary aux loss
        # -----------------------------------------------------------------------------------------------

        def _far_field_nodes(self, state: SummaryTreeState) -> list[TreeNode]:
            """``live ∪ archived`` across every level -- archived nodes stay forward-visible (E4.md:
            "stop-gradient only cuts backward"), still O(log_fanout(evicted_count)) total."""
            nodes: list[TreeNode] = []
            for lvl in state.live:
                nodes.extend(lvl)
            for lvl in state.archived:
                nodes.extend(lvl)
            return nodes

        def _content_bias(self, nodes: list[TreeNode], device: Any) -> Any:
            if not nodes:
                return None
            levels = torch.as_tensor([min(n.level - 1, self.max_level_cap - 1) for n in nodes], device=device)
            slots = torch.as_tensor([n.path[0] % self.fanout for n in nodes], device=device)
            return self.level_embed(levels) + self.slot_embed(slots)  # (n_far, d_model)

        def _lca_bias_matrix(self, query_positions: Any, nodes: list[TreeNode]) -> Any:
            """``(t, n_far)`` learned bias, one column per far-field node, indexed by tree distance
            from each query position (E4.md's relative channel -- see module docstring)."""
            device = query_positions.device
            qs = [int(p) for p in query_positions.tolist()]
            depths = [
                [
                    min(
                        lca_depth(q, n.level, n.g, self.fanout, max_climb=self.max_level_cap - 1),
                        self.max_level_cap - 1,
                    )
                    for n in nodes
                ]
                for q in qs
            ]
            idx = torch.as_tensor(depths, device=device, dtype=torch.long)  # (t, n_far)
            return self.lca_bias[idx]

        def _aux_loss_for_nodes(self, nodes: list[TreeNode]) -> Any | None:
            """Predict-the-summary aux loss (E4.md), evaluated ONCE per node at finalization -- ``nodes``
            here is always the set finalized this very step, never re-touched later."""
            if not nodes:
                return None
            losses = []
            for node in nodes:
                target = node.histogram / node.histogram.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                for layer in range(self.n_layer):
                    logits = self.predict_head(node.summary[layer])
                    losses.append(-(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean())
            return torch.stack(losses).mean()

        # -----------------------------------------------------------------------------------------------
        # step()
        # -----------------------------------------------------------------------------------------------

        def step(self, state: SummaryTreeState, chunk: tuple[Any, Any]) -> tuple[SummaryTreeState, Any]:
            x, y = chunk
            b, t = x.shape
            device = x.device
            query_positions = torch.arange(state.window.pos, state.window.pos + t, device=device)

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            far_nodes = self._far_field_nodes(state)
            content_bias = self._content_bias(far_nodes, device)  # (n_far, d_model) | None
            lca_bias_mat = self._lca_bias_matrix(query_positions, far_nodes) if far_nodes else None  # (t, n_far)

            evicted_k_by_layer: list[Any] = []
            evicted_v_by_layer: list[Any] = []

            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (b, t, n_head, head_dim)
                k_flat = k.reshape(b, t, self.d_model)  # pre-RoPE, flattened across heads -- tree leaf content
                v_flat = v.reshape(b, t, self.d_model)

                cache_k, cache_v = state.window.cache_k[layer], state.window.cache_v[layer]
                if cache_k is not None:
                    cache_len = cache_k.shape[1]
                    key_positions = torch.arange(state.window.pos - cache_len, state.window.pos + t, device=device)
                    k_full = torch.cat([cache_k, k], dim=1)
                    v_full = torch.cat([cache_v, v], dim=1)
                else:
                    key_positions = query_positions
                    k_full, v_full = k, v

                sin_q, cos_q = _rope_angles(query_positions, self.head_dim)
                sin_k, cos_k = _rope_angles(key_positions, self.head_dim)
                q_rope = _apply_rope(q, sin_q, cos_q)
                k_full_rope = _apply_rope(k_full, sin_k, cos_k)

                delta = query_positions[:, None] - key_positions[None, :]  # (t, len(keys))
                allowed = (delta >= 0) & (delta < self.window)
                near_mask = torch.zeros(t, key_positions.shape[0], device=device)
                near_mask = near_mask.masked_fill(~allowed, float("-inf"))

                qh = q_rope.transpose(1, 2)  # (b, n_head, t, head_dim)
                kh = k_full_rope.transpose(1, 2)
                vh = v_full.transpose(1, 2)
                near_logits = (qh @ kh.transpose(-2, -1)) / (self.head_dim**0.5)  # (b, n_head, t, len(keys))
                near_logits = near_logits + near_mask[None, None]

                if far_nodes:
                    summaries = torch.stack([n.summary[layer] for n in far_nodes], dim=1)  # (b, n_far, d_model)
                    far_input = summaries + content_bias[None, :, :]
                    far_qkv = self.qkv[layer](far_input).reshape(b, len(far_nodes), 3, self.n_head, self.head_dim)
                    k_far, v_far = far_qkv[:, :, 1], far_qkv[:, :, 2]  # no RoPE (E4.md: no single relative offset)
                    kfh = k_far.transpose(1, 2)  # (b, n_head, n_far, head_dim)
                    vfh = v_far.transpose(1, 2)
                    far_logits = (qh @ kfh.transpose(-2, -1)) / (self.head_dim**0.5)  # (b, n_head, t, n_far)
                    far_logits = far_logits + lca_bias_mat[None, None]

                    combined = torch.cat([near_logits, far_logits], dim=-1)
                    weights = combined.softmax(dim=-1)
                    near_w = weights[..., : key_positions.shape[0]]
                    far_w = weights[..., key_positions.shape[0] :]
                    out = (near_w @ vh + far_w @ vfh).transpose(1, 2).reshape(b, t, self.d_model)
                else:
                    weights = near_logits.softmax(dim=-1)
                    out = (weights @ vh).transpose(1, 2).reshape(b, t, self.d_model)

                h = h + self.proj[layer](out)
                h = h + self.mlp[layer](self.ln2[layer](h))

                keep = self.window
                new_cache_k.append(k_full[:, -keep:])
                new_cache_v.append(v_full[:, -keep:])
                evict_amt = max(0, k_full.shape[1] - keep)
                if evict_amt > 0:
                    evicted_k_by_layer.append(k_full[:, :evict_amt].reshape(b, evict_amt, self.d_model))
                    evicted_v_by_layer.append(v_full[:, :evict_amt].reshape(b, evict_amt, self.d_model))
                else:
                    evicted_k_by_layer.append(None)
                    evicted_v_by_layer.append(None)

            logits = self.head(self.ln_f(h))
            lm_loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))

            # ids for the evicted slice: the same absolute-position slice for every layer (window shared).
            ids_full = x if state.cached_ids is None else torch.cat([state.cached_ids, x], dim=1)
            evict_amt = max(0, ids_full.shape[1] - self.window)
            evicted_ids = ids_full[:, :evict_amt] if evict_amt > 0 else None
            new_cached_ids = ids_full[:, -self.window :]

            new_state = SummaryTreeState(
                window=SlidingWindowState(cache_k=new_cache_k, cache_v=new_cache_v, pos=state.window.pos + t),
                cached_ids=new_cached_ids,
                pending_leaf=list(state.pending_leaf),
                pending=[list(lvl) for lvl in state.pending],
                live=[list(lvl) for lvl in state.live],
                archived=[list(lvl) for lvl in state.archived],
                level_finalized_count=list(state.level_finalized_count),
                evicted_count=state.evicted_count + evict_amt,
            )

            finalized_this_step: list[TreeNode] = []
            if evicted_ids is not None and evict_amt > 0:
                for pos in range(evict_amt):
                    before_live_counts = [len(lvl) for lvl in new_state.live]
                    self._absorb_evicted_token(
                        new_state,
                        [evicted_k_by_layer[layer][:, pos] for layer in range(self.n_layer)],
                        [evicted_v_by_layer[layer][:, pos] for layer in range(self.n_layer)],
                        evicted_ids[:, pos],
                    )
                    for level_idx, before in enumerate(before_live_counts):
                        if len(new_state.live[level_idx]) > before:
                            finalized_this_step.append(new_state.live[level_idx][-1])
                    if len(new_state.live) > len(before_live_counts):
                        for lvl in new_state.live[len(before_live_counts) :]:
                            finalized_this_step.extend(lvl)

            aux_loss = self._aux_loss_for_nodes(finalized_this_step)
            if aux_loss is not None and self.aux_weight > 0:
                total_loss = lm_loss + self.aux_weight * aux_loss
                self.last_aux_loss = float(aux_loss.detach())
            else:
                total_loss = lm_loss
                self.last_aux_loss = 0.0

            return new_state, total_loss

        # -----------------------------------------------------------------------------------------------
        # GradLeaf citizenship (E4.md "GradLeaf citizenship")
        # -----------------------------------------------------------------------------------------------

        def log_density(self, x: Any, y: Any) -> Any:
            """``x, y``: ``(n, T)`` long tensors. Returns ``-mean_per_position_nll`` for each of the
            ``n`` sequences, each scored independently (state re-initialized per row) -- one
            non-streaming forward per row, computed by calling ``init_state`` + ``step`` once per row
            exactly as a length-``T``, single-chunk stream would (E5.md's "GradLeaf citizenship")."""
            # score by LM loss alone -- the aux loss is a training-time compressor-supervision signal,
            # not part of the sequence's density, so it's excluded here (temporarily zero aux_weight
            # rather than duplicate step()'s forward pass).
            saved_aux_weight = self.aux_weight
            self.aux_weight = 0.0
            try:
                out = []
                for i in range(x.shape[0]):
                    state = self.init_state(1, device=str(x.device))
                    _, loss = self.step(state, (x[i : i + 1], y[i : i + 1]))
                    out.append(-loss)
                return torch.stack(out)
            finally:
                self.aux_weight = saved_aux_weight

    REGISTRY.register(ExperimentalMechanism(name="summary_tree"))
