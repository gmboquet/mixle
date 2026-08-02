"""E4: hierarchical summary tree + multi-scale objective -- see ``notes/designs/E4.md`` for the full
derivation (persistent frontier tree over evicted tokens, tree-path positional encoding, the
predict-the-summary auxiliary loss, the stop-gradient horizon receipt). This module implements that
note section-by-section; see the note's "Implementation notes vs. this design" section for the small,
honestly-documented places this module simplifies the note's scheme for tractability.

**What this is.** E1's :class:`~mixle.experimental.context_spine.SlidingWindowSpine` keeps an exact
but bounded KV window; anything evicted from that window is gone. E4 keeps it: every evicted token is
folded, one at a time, into a bounded frontier of learned summaries via mixed-radix carry propagation
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

**Stop-gradient horizon (receipted).** A frontier node moves from ``live`` to ``archived`` once ``H``
further evicted tokens have arrived after it finalized; ``.summary`` is ``.detach()``-ed at that moment
and never un-detaches. Archived frontier nodes stay forward-visible but stop receiving gradient. See
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
    represented_leaves: int = 0
    detached: bool = False
    detached_at_finalized_count: int | None = None
    detached_at_evicted_count: int | None = None


@dataclass
class SummaryTreeState:
    """``ContextMechanism`` carried state: E1's exact near field plus the persistent far-field tree."""

    window: SlidingWindowState  # E1's near field, unmodified
    cached_ids: Any | None  # (batch, cache_len) token ids aligned with window's cache -- shared across layers
    pending_leaf: list  # buffered evicted (k_per_layer, v_per_layer, id) tuples not yet forming a level-1 node
    pending: list[list[TreeNode]]  # references to frontier children awaiting a same-level carry group
    live: list[list[TreeNode]]  # index i = level (i + 1): live nodes on the non-overlapping frontier
    archived: list[list[TreeNode]]  # index i = level (i + 1): detached nodes on that same frontier
    level_finalized_count: list[int]  # index i = level (i + 1): total nodes ever finalized at that level
    evicted_count: int = 0
    batch_size: int = 1
    receipt: dict[str, Any] | None = None


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
            # Public constructor argument checks, so not asserts: `python -O` strips asserts, and
            # both gate real architectural invariants -- an indivisible head split is lossy, and a
            # fanout below 2 is not a tree (MXR-080-1861).
            if n_head < 1 or d_model % n_head != 0:
                raise ValueError(
                    f"SummaryTreeSpine requires d_model divisible by a positive n_head, got "
                    f"d_model={d_model}, n_head={n_head}."
                )
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
            self.window = int(window)
            self.fanout = int(fanout)
            if self.fanout < 2:
                raise ValueError(
                    f"SummaryTreeSpine requires fanout >= 2, got {fanout}: a fanout of one never "
                    "reduces the sequence, so the summary tree has no levels to build."
                )
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
            # The same quantity as a live, aux_weight-scaled tensor -- ALREADY included in the loss the
            # matching step() returned, exposed for callers that discard that loss but must keep the aux
            # term trainable (a warm-up prefix whose LM targets are deliberately not trained on: nodes
            # finalize there and nowhere else, so dropping the prefix loss drops all aux gradient).
            # ``None`` when no node finalized or ``aux_weight == 0``. Adding it to a backward that already
            # includes the returned loss double-counts it.
            self.last_aux_term: Any | None = None

        # -----------------------------------------------------------------------------------------------
        # ContextMechanism protocol
        # -----------------------------------------------------------------------------------------------

        def init_state(self, batch_size: int, *, device: str = "cpu") -> SummaryTreeState:
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be a positive exact integer.")
            dev = torch.device(device)
            if dev != self.tok.weight.device:
                raise ValueError("state device must match the model device.")
            window = SlidingWindowState(
                cache_k=[None] * self.n_layer,
                cache_v=[None] * self.n_layer,
                pos=0,
                batch_size=batch_size,
                n_head=self.n_head,
                head_dim=self.head_dim,
                device=dev,
            )
            return SummaryTreeState(
                window=window,
                cached_ids=None,
                pending_leaf=[],
                pending=[],
                live=[],
                archived=[],
                level_finalized_count=[],
                evicted_count=0,
                batch_size=batch_size,
                receipt={
                    "evicted_tokens_per_stream": 0,
                    "represented_leaf_mass": 0,
                    "frontier_nodes": 0,
                    "stored_frontier_nodes": 0,
                    "pending_leaves": 0,
                    "storage_bound": 0,
                    "non_overlapping": True,
                    "conserved": True,
                },
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
                    age = state.evicted_count - node.finalized_step
                    if age >= self.detach_horizon_nodes:
                        detached_node = self._detach_node(node, force=True)
                        detached_node.detached_at_finalized_count = finalized_total
                        detached_node.detached_at_evicted_count = state.evicted_count
                        archived[level_idx].append(detached_node)
                    else:
                        still_live.append(node)
                live.append(still_live)

            # ``pending`` is an index into the same bounded frontier, not a second ownership store.
            # Rebind it to the detached/live replacements so the two lists never carry divergent copies.
            replacements = {
                (node.level, node.g): node
                for levels in (live, archived)
                for level_nodes in levels
                for node in level_nodes
            }
            pending = [
                [replacements[(node.level, node.g)] for node in level_pending] for level_pending in state.pending
            ]

            detached_state = SummaryTreeState(
                window=window,
                cached_ids=cached_ids,
                pending_leaf=pending_leaf,
                pending=pending,
                live=live,
                archived=archived,
                level_finalized_count=list(state.level_finalized_count),
                evicted_count=state.evicted_count,
                batch_size=state.batch_size,
            )
            detached_state.receipt = self._frontier_receipt(detached_state)
            return detached_state

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
            self,
            state: SummaryTreeState,
            level: int,
            children_summaries: list[list[Any]],
            histogram: Any,
            *,
            represented_leaves: int,
            finalized: list[TreeNode],
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
                represented_leaves=represented_leaves,
            )
            state.level_finalized_count[level - 1] += 1
            _ensure_level(state.live, level)
            state.live[level - 1].append(node)
            finalized.append(node)
            return node

        @staticmethod
        def _remove_frontier_node(state: SummaryTreeState, node: TreeNode) -> None:
            for levels in (state.live, state.archived):
                if len(levels) < node.level:
                    continue
                levels[node.level - 1] = [
                    candidate
                    for candidate in levels[node.level - 1]
                    if not (candidate.level == node.level and candidate.g == node.g)
                ]

        def _carry_propagate(
            self,
            state: SummaryTreeState,
            node: TreeNode,
            finalized: list[TreeNode],
        ) -> None:
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
            for child in children:
                self._remove_frontier_node(state, child)

            children_summaries = [[c.summary[layer] for c in children] for layer in range(self.n_layer)]
            histogram = children[0].histogram
            for c in children[1:]:
                histogram = histogram + c.histogram
            parent = self._finalize_node(
                state,
                node.level + 1,
                children_summaries,
                histogram,
                represented_leaves=sum(child.represented_leaves for child in children),
                finalized=finalized,
            )
            self._carry_propagate(state, parent, finalized)

        def _absorb_evicted_token(
            self, state: SummaryTreeState, k_layers: list[Any], v_layers: list[Any], token_id: Any
        ) -> list[TreeNode]:
            """Feed ONE evicted token into the tree (E4.md construction step 1). ``k_layers``/``v_layers``:
            list of ``(batch, d_model)`` per layer (flattened across heads -- see module docstring).
            ``token_id``: ``(batch,)`` long tensor."""
            state.pending_leaf.append((k_layers, v_layers, token_id))
            if len(state.pending_leaf) < self.fanout:
                return []
            group = state.pending_leaf[: self.fanout]
            state.pending_leaf = state.pending_leaf[self.fanout :]

            children_summaries = [
                [torch.cat([g[0][layer], g[1][layer]], dim=-1) for g in group] for layer in range(self.n_layer)
            ]
            histogram = None
            for _, _, tid in group:
                onehot = F.one_hot(tid, num_classes=self.vocab).to(children_summaries[0][0].dtype)
                histogram = onehot if histogram is None else histogram + onehot
            finalized: list[TreeNode] = []
            node = self._finalize_node(
                state,
                1,
                children_summaries,
                histogram,
                represented_leaves=self.fanout,
                finalized=finalized,
            )
            self._carry_propagate(state, node, finalized)
            return finalized

        # -----------------------------------------------------------------------------------------------
        # Far-field attention set + predict-the-summary aux loss
        # -----------------------------------------------------------------------------------------------

        @staticmethod
        def _stored_frontier_nodes(state: SummaryTreeState) -> list[TreeNode]:
            nodes: list[TreeNode] = []
            for levels in (state.live, state.archived):
                for level_nodes in levels:
                    nodes.extend(level_nodes)
            return nodes

        def _pending_leaf_nodes(self, state: SummaryTreeState) -> list[TreeNode]:
            nodes: list[TreeNode] = []
            first_position = state.evicted_count - len(state.pending_leaf)
            for offset, (k_layers, v_layers, token_id) in enumerate(state.pending_leaf):
                absolute_position = first_position + offset
                summaries = [
                    self.compressor.leaf_adapter(torch.cat([k_layers[layer], v_layers[layer]], dim=-1))
                    for layer in range(self.n_layer)
                ]
                histogram = F.one_hot(token_id, num_classes=self.vocab).to(summaries[0].dtype)
                nodes.append(
                    TreeNode(
                        summary=summaries,
                        histogram=histogram,
                        path=digits_of(absolute_position, self.fanout),
                        level=0,
                        g=absolute_position,
                        finalized_step=absolute_position + 1,
                        finalized_index_within_level=absolute_position,
                        represented_leaves=1,
                    )
                )
            return nodes

        def _frontier_receipt(self, state: SummaryTreeState) -> dict[str, Any]:
            stored = self._stored_frontier_nodes(state)
            keys = [(node.level, node.g) for node in stored]
            represented = sum(node.represented_leaves for node in stored) + len(state.pending_leaf)
            pending_keys = [(node.level, node.g) for level_pending in state.pending for node in level_pending]
            storage_bound = len(state.pending_leaf) + (self.fanout - 1) * len(state.level_finalized_count)
            non_overlapping = (
                len(keys) == len(set(keys))
                and len(pending_keys) == len(set(pending_keys))
                and set(pending_keys) == set(keys)
            )
            conserved = non_overlapping and represented == state.evicted_count
            receipt = {
                "evicted_tokens_per_stream": state.evicted_count,
                "represented_leaf_mass": represented,
                "frontier_nodes": len(stored) + len(state.pending_leaf),
                "stored_frontier_nodes": len(stored),
                "pending_leaves": len(state.pending_leaf),
                "storage_bound": storage_bound,
                "non_overlapping": non_overlapping,
                "conserved": conserved,
            }
            if not conserved or receipt["frontier_nodes"] > storage_bound:
                raise RuntimeError(
                    f"summary-tree frontier violates non-overlapping bounded-state accounting: {receipt}"
                )
            return receipt

        def _far_field_nodes(self, state: SummaryTreeState) -> list[TreeNode]:
            """Return the non-overlapping frontier plus incomplete leaves.

            A parent replaces its children, so no leaf is represented at multiple levels. Incomplete
            leaf groups remain visible through a single-token adapter until they form a level-1 node.
            """
            self._frontier_receipt(state)
            return self._stored_frontier_nodes(state) + self._pending_leaf_nodes(state)

        def _content_bias(self, nodes: list[TreeNode], device: Any) -> Any:
            if not nodes:
                return None
            levels = torch.as_tensor(
                [min(max(n.level - 1, 0), self.max_level_cap - 1) for n in nodes],
                device=device,
            )
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
            if not isinstance(state, SummaryTreeState):
                raise TypeError("state must be a SummaryTreeState.")
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                raise TypeError("chunk must be an (input_tokens, target_tokens) tuple.")
            x, y = chunk
            if not torch.is_tensor(x) or not torch.is_tensor(y) or x.dtype != torch.long or y.dtype != torch.long:
                raise TypeError("input and target tokens must be torch.long tensors.")
            if x.ndim != 2 or y.shape != x.shape or x.shape[0] == 0 or x.shape[1] == 0:
                raise ValueError("input and target tokens must have equal non-empty (batch, time) shape.")
            b, t = x.shape
            if b != state.batch_size:
                raise ValueError(f"state batch_size={state.batch_size} does not match chunk batch_size={b}.")
            if x.device != self.tok.weight.device or y.device != x.device:
                raise ValueError("state, model, input, and target tensors must be on the same device.")
            if bool(((x < 0) | (x >= self.vocab) | (y < 0) | (y >= self.vocab)).any().item()):
                raise ValueError(f"input and target token IDs must lie in [0, {self.vocab}).")

            # Single-token advancement makes the tree boundary part of the causal recurrence. This keeps
            # outputs and state invariant to how a caller chunks the same stream.
            if t > 1:
                current_state = state
                losses = []
                auxiliary_losses = []
                auxiliary_terms = []
                for query_index in range(t):
                    current_state, token_loss = self.step(
                        current_state,
                        (
                            x[:, query_index : query_index + 1],
                            y[:, query_index : query_index + 1],
                        ),
                    )
                    losses.append(token_loss)
                    auxiliary_losses.append(self.last_aux_loss)
                    if self.last_aux_term is not None:
                        auxiliary_terms.append(self.last_aux_term)
                self.last_aux_loss = float(sum(auxiliary_losses) / len(auxiliary_losses))
                # The returned loss is the MEAN over tokens, so the aux contribution it already carries
                # is the per-token sum divided by t -- not the mean over the finalizing tokens alone.
                self.last_aux_term = torch.stack(auxiliary_terms).sum() / t if auxiliary_terms else None
                return current_state, torch.stack(losses).mean()

            device = x.device
            query_positions = torch.arange(state.window.pos, state.window.pos + t, device=device)

            working_state = SummaryTreeState(
                window=state.window,
                cached_ids=state.cached_ids,
                pending_leaf=list(state.pending_leaf),
                pending=[list(level_pending) for level_pending in state.pending],
                live=[list(level_live) for level_live in state.live],
                archived=[list(level_archived) for level_archived in state.archived],
                level_finalized_count=list(state.level_finalized_count),
                evicted_count=state.evicted_count,
                batch_size=state.batch_size,
                receipt=dict(state.receipt or {}),
            )
            finalized_this_step: list[TreeNode] = []

            cached_ids = working_state.cached_ids
            cache_lengths = {cache.shape[1] for cache in working_state.window.cache_k if cache is not None}
            if len(cache_lengths) > 1:
                raise ValueError("all summary-tree layer caches must have the same length.")
            cache_len = next(iter(cache_lengths), 0)
            if cached_ids is None:
                if cache_len != 0 or any(cache is not None for cache in working_state.window.cache_v):
                    raise ValueError("summary-tree token IDs and K/V caches are not aligned.")
            elif cached_ids.shape != (b, cache_len):
                raise ValueError("summary-tree cached token IDs do not align with the K/V caches.")
            if cache_len > self.window:
                raise ValueError("summary-tree near cache exceeds the configured window.")

            # The token at distance ``window`` crosses into the far frontier before this query reads
            # memory. Keeping it until after the query would make it invisible for one step.
            if cache_len == self.window:
                if any(cache is None for cache in working_state.window.cache_k + working_state.window.cache_v):
                    raise ValueError("summary-tree has a partial per-layer K/V cache.")
                working_state.evicted_count += 1
                finalized_this_step.extend(
                    self._absorb_evicted_token(
                        working_state,
                        [
                            working_state.window.cache_k[layer][:, 0].reshape(b, self.d_model)
                            for layer in range(self.n_layer)
                        ],
                        [
                            working_state.window.cache_v[layer][:, 0].reshape(b, self.d_model)
                            for layer in range(self.n_layer)
                        ],
                        cached_ids[:, 0],
                    )
                )
                working_state.window = SlidingWindowState(
                    cache_k=[cache[:, 1:] for cache in working_state.window.cache_k],
                    cache_v=[cache[:, 1:] for cache in working_state.window.cache_v],
                    pos=working_state.window.pos,
                    batch_size=state.batch_size,
                    n_head=self.n_head,
                    head_dim=self.head_dim,
                    device=device,
                )
                working_state.cached_ids = cached_ids[:, 1:]

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            far_nodes = self._far_field_nodes(working_state)
            content_bias = self._content_bias(far_nodes, device)  # (n_far, d_model) | None
            lca_bias_mat = self._lca_bias_matrix(query_positions, far_nodes) if far_nodes else None  # (t, n_far)

            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (b, t, n_head, head_dim)

                cache_k, cache_v = working_state.window.cache_k[layer], working_state.window.cache_v[layer]
                if cache_k is not None:
                    cache_len = cache_k.shape[1]
                    key_positions = torch.arange(
                        state.window.pos - cache_len,
                        state.window.pos + t,
                        device=device,
                    )
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
                qh_far = q.transpose(1, 2)
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
                    far_logits = (qh_far @ kfh.transpose(-2, -1)) / (self.head_dim**0.5)  # (b, n_head, t, n_far)
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

                keep = min(self.window, k_full.shape[1])
                new_cache_k.append(k_full[:, -keep:])
                new_cache_v.append(v_full[:, -keep:])

            logits = self.head(self.ln_f(h))
            lm_loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))

            ids_full = x if working_state.cached_ids is None else torch.cat([working_state.cached_ids, x], dim=1)
            new_cached_ids = ids_full[:, -self.window :]

            new_state = SummaryTreeState(
                window=SlidingWindowState(
                    cache_k=new_cache_k,
                    cache_v=new_cache_v,
                    pos=state.window.pos + t,
                    batch_size=state.batch_size,
                    n_head=self.n_head,
                    head_dim=self.head_dim,
                    device=device,
                ),
                cached_ids=new_cached_ids,
                pending_leaf=working_state.pending_leaf,
                pending=working_state.pending,
                live=working_state.live,
                archived=working_state.archived,
                level_finalized_count=working_state.level_finalized_count,
                evicted_count=working_state.evicted_count,
                batch_size=state.batch_size,
            )
            new_state.receipt = self._frontier_receipt(new_state)

            aux_loss = self._aux_loss_for_nodes(finalized_this_step)
            if aux_loss is not None and self.aux_weight > 0:
                aux_term = self.aux_weight * aux_loss
                total_loss = lm_loss + aux_term
                self.last_aux_loss = float(aux_loss.detach())
                self.last_aux_term = aux_term
            else:
                total_loss = lm_loss
                self.last_aux_loss = 0.0
                self.last_aux_term = None

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
