"""Temporal (dynamic) graph grammar -- a distribution over graph SEQUENCES you can score, fit, and sample.

A dynamic graph is observed as a sequence of adjacency snapshots ``[A_0, A_1, ..., A_T]`` (binary,
undirected; nodes may be appended and edges added over time -- a growth process). The model is a Markov
chain over graphs whose transition kernel is a stochastic **motif-edit grammar**:

    given G_{t-1}, draw a number of new edges, and produce each by firing a grammar rule -- "add an edge
    that creates motif m" -- where the rule is chosen from the motif distribution ``w`` and an anchor is
    chosen uniformly among the non-edges of G_{t-1} that instantiate that motif.

So the grammar EDITS the graph over time, and its rule weights ARE the motif distribution it imposes. The
default motif family bins a candidate edge by how many triangles it would close (its number of common
neighbours: 0 = a bridge, 1, 2, 3+), i.e. a learnable triadic-closure profile; a custom mutually-exclusive
motif partition can be supplied instead. Because the bins are mutually exclusive each added edge has a
*single* motif, so scoring and fitting are exact (no per-edge latent -- the VRG/HRG grammars marginalise
over derivations; here the derivation is read off the snapshots).

Edges both FORM and DISSOLVE: an ADD grammar (``motif_weights`` / ``edge_rate``) draws new edges by the
motif each would create, and a separate REMOVE grammar (``remove_weights`` / ``edge_remove_rate``) deletes
existing edges by the motif each is part of (so e.g. growth can favour triadic closure while decay favours
bridges -- ties in dense neighbourhoods persist). Removal defaults off, so the constructor is backward
compatible and a pure-growth grammar still scores a deletion as -inf.

Adjacencies may be dense ``ndarray`` or ``scipy.sparse`` -- scoring and fitting never form the n*n bin
matrix (they touch only the changed edges and the wedge structure of ``A @ A``), so a 200k-node graph
scores in a fraction of a second where a dense adjacency would need hundreds of GB. (Sampling stays
dense/moderate-scale.) ``LabeledTemporalGraphGrammarDistribution`` attaches node attributes (location,
name, age, ...) and edge attributes (communication counts, channel, ...) as ordinary mixle distributions
scored as emissions on top of the topology -- the whole thing fits jointly with the full distribution
machinery (mixtures, every leaf family, the numba fusion).

Graphs may be **directed** (``directed=True``): the adjacency is asymmetric (i->j and j->i are distinct
edges), the candidate space is the full off-diagonal, and ``A @ A`` counts transitive i->k->j paths -- a
directed triadic-closure profile. **Weighted** edges are just an edge attribute: put a weight distribution
(Poisson volume, Gaussian strength, ...) in the labeled model's ``edge_dist``, so a directed + weighted +
attributed dynamic graph is a directed structure composed with node/edge emission models.

Nodes also LEAVE: ``ChurningTemporalGraphGrammarDistribution`` tracks stable node identities (each snapshot
is ``(adjacency, node_ids)``) so a transition can remove nodes -- those whose id disappears, their edges
vanishing with them -- before running the edit grammar on the surviving subgraph. Node churn is a thin
wrapper: identity alignment + a node-removal Poisson term on top of all the motif/edge machinery.

The dynamics carry a HIDDEN REGIME: ``LatentTemporalGraphGrammarDistribution`` is an HMM whose emission
models ARE the edit grammars -- a latent Markov state z_t selects which of K grammars governs transition t,
so the graph switches phases over time (bursty growth/densification, then fragmentation/decay -- dynamics a
single grammar cannot produce). The sequence likelihood marginalises the regime path by the forward
algorithm; EM (Baum-Welch) runs forward-backward then a per-regime weighted M-step reusing each grammar's
accumulator; ``decode`` (Viterbi) recovers the active regime at each step.

This is the temporal counterpart of the static vertex-/hyperedge-replacement grammars in this package.
Scope: undirected or directed, binary topology (attribute models carry weights/labels); edges add+remove,
nodes join+leave; dense or sparse scoring, dense or scalable (rejection) sampling; optional hidden regime.
Sparse-path churn and directed scalable sampling are the remaining extensions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_EPS = 1.0e-12


def _binarize(adj: Any) -> Any:
    """Return a binary upper-or-full adjacency as a CSR array (sparse) or float ndarray (dense)."""
    if sp.issparse(adj):
        a = adj.tocsr().copy()
        a.data[:] = 1.0
        return a
    return (np.asarray(adj, dtype=np.float64) > 0).astype(np.float64)


def _pad(adj: Any, n: int) -> Any:
    """Grow a (n0,n0) adjacency to (n,n) by appending isolated nodes (sparse or dense)."""
    n0 = adj.shape[0]
    if n == n0:
        return adj
    if sp.issparse(adj):
        out = sp.lil_array((n, n))
        out[:n0, :n0] = adj
        return out.tocsr()
    out = np.zeros((n, n), dtype=np.float64)
    out[:n0, :n0] = adj
    return out


def _edge_diff(prev: Any, cur: Any, directed: bool = False) -> tuple:
    """(added_i, added_j, removed_i, removed_j) between two binary adjacencies.

    ``prev`` is padded to ``cur``'s size; added = edges in cur not in prev, removed = edges in prev not in
    cur. Undirected reads the upper triangle (each edge once); directed reads the full off-diagonal (i->j
    and j->i are distinct edges). Works for sparse or dense and only touches the edges that actually
    changed."""
    n1 = cur.shape[0]
    pp = _pad(_binarize(prev), n1)
    cc = _binarize(cur)
    if sp.issparse(cur) or sp.issparse(prev):
        diff = sp.csr_array(cc) - sp.csr_array(pp)
        d = (diff if directed else sp.triu(diff, 1)).tocoo()
        added = d.data > 0
        removed = d.data < 0
        return d.row[added], d.col[added], d.row[removed], d.col[removed]
    delta = cc - pp  # directed: full off-diagonal (diagonal is 0 -- no self-loops); undirected: upper tri
    d = delta if directed else np.triu(delta, 1)
    ai, aj = np.where(d > 0)
    ri, rj = np.where(d < 0)
    return ai, aj, ri, rj


# --- motifs ---------------------------------------------------------------------------------------
class CommonNeighbourMotif:
    """A motif rule keyed by how many common neighbours a candidate edge has (triangles it would close).

    ``bins`` is an increasing list of thresholds; bin ``b`` covers common-neighbour counts in
    ``[bins[b], bins[b+1])`` with the last bin open-ended. The default ``[0, 1, 2, 3]`` gives the
    interpretable {bridge, closes-1, closes-2, closes-3+} partition. A non-edge falls in exactly one bin,
    so the motifs partition every candidate edge.
    """

    def __init__(self, bins: Sequence[int] = (0, 1, 2, 3), directed: bool = False) -> None:
        self.bins = tuple(int(b) for b in bins)
        self.directed = bool(directed)  # directed: A@A counts transitive i->k->j paths; candidates = full off-diagonal
        self.names = [f"cn>={self.bins[-1]}" if i == len(self.bins) - 1 else f"cn={b}" for i, b in enumerate(self.bins)]

    @property
    def num_motifs(self) -> int:
        """Number of mutually exclusive motif bins."""
        return len(self.bins)

    def assign(self, adj: np.ndarray, on_edges: bool = False) -> np.ndarray:
        """Motif bin of every candidate pair (and -1 on non-candidates / diagonal).

        The common-neighbour count of a pair (i, j) is ``(A @ A)[i, j]`` -- for a non-edge, how many
        triangles adding it would CLOSE; for an existing edge, how many triangles it is PART of. Binning by
        ``self.bins`` gives its motif. With ``on_edges=False`` the candidates are the non-edges (addition);
        with ``on_edges=True`` they are the existing edges (removal). Non-candidates and the diagonal -> -1.
        """
        n = adj.shape[0]
        cn = adj @ adj  # common-neighbour counts
        b = np.searchsorted(self.bins, cn, side="right") - 1  # bin index per pair
        b = np.clip(b, 0, len(self.bins) - 1).astype(np.int64)
        non_candidate = (adj == 0) if on_edges else (adj > 0)  # removal scores edges; addition scores non-edges
        b[non_candidate | np.eye(n, dtype=bool)] = -1
        return b

    def _bin(self, cn_vals: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.bins, cn_vals, side="right") - 1, 0, len(self.bins) - 1)

    def counts_and_binner(self, adj: Any, on_edges: bool) -> tuple:
        """Return (candidate_counts[M], lookup(i, j) -> motif index) WITHOUT forming the n*n bin matrix.

        Sparse-scalable: only the existing edges (O(m)) and the non-edges that close a triangle (O(wedges) =
        ``A @ A``'s nonzeros) are ever enumerated; the bridge count (cn=0 non-edges) is the analytic
        remainder ``pairs - edges - wedge_non_edges``. The lookup reads ``(A @ A)[i, j]`` for the handful of
        observed edges. (For graphs with mega-hubs the wedge set itself is large -- the documented limit.)
        """
        adj = _binarize(adj)
        n = adj.shape[0]
        cn = adj @ adj
        counts = np.zeros(self.num_motifs, dtype=np.float64)
        total_pairs = n * (n - 1) if self.directed else n * (n - 1) / 2  # off-diagonal candidate pairs
        if sp.issparse(adj):
            au = (adj if self.directed else sp.triu(adj, 1)).tocoo()
            cu = cn.tocsr().copy() if self.directed else sp.triu(cn, 1).tocsr()
            if self.directed:
                cu.setdiag(0)  # drop i->k->i (the diagonal is not a candidate edge)
                cu.eliminate_zeros()
            edge_mask = sp.csr_array((np.ones(au.nnz), (au.row, au.col)), shape=(n, n)) if au.nnz else None
            if on_edges:
                if au.nnz:
                    np.add.at(counts, self._bin(np.asarray(cu[au.row, au.col]).ravel()), 1.0)
            else:
                non_edge_cn = cu if edge_mask is None else (cu - cu.multiply(edge_mask))
                non_edge_cn.eliminate_zeros()
                vals = non_edge_cn.tocoo().data
                counts[0] += total_pairs - au.nnz - vals.size  # bridges = pairs - edges - wedge non-edges
                if vals.size:
                    np.add.at(counts, self._bin(vals), 1.0)
            csr = cn.tocsr()

            def lookup(ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
                return self._bin(np.asarray(csr[ii, jj]).ravel()) if len(ii) else np.zeros(0, dtype=np.int64)
        else:
            offdiag = ~np.eye(n, dtype=bool)
            cand_mask = offdiag if self.directed else np.triu(np.ones((n, n), dtype=bool), 1)
            sel = cand_mask & ((adj > 0) if on_edges else (adj == 0))
            np.add.at(counts, self._bin(cn[sel]), 1.0)

            def lookup(ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
                return self._bin(cn[ii, jj]) if len(ii) else np.zeros(0, dtype=np.int64)

        return counts, lookup


# --- distribution ---------------------------------------------------------------------------------
class TemporalGraphGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """Distribution over dynamic graphs (sequences of adjacency snapshots) under a motif-edit grammar."""

    def __init__(
        self,
        motif_weights: Sequence[float],
        edge_rate: float = 1.0,
        node_rate: float = 0.0,
        remove_weights: Sequence[float] | None = None,
        edge_remove_rate: float = 0.0,
        motif: CommonNeighbourMotif | None = None,
        directed: bool = False,
        name: str | None = None,
    ) -> None:
        self.motif = motif if motif is not None else CommonNeighbourMotif(directed=directed)
        self.directed = self.motif.directed
        m = self.motif.num_motifs

        def _norm(w: Sequence[float] | None) -> np.ndarray:
            a = np.ones(m) if w is None else np.asarray(w, dtype=np.float64)
            if a.shape[0] != m:
                raise ValueError("motif weights must have one entry per motif bin (%d)." % m)
            return a / a.sum()

        self.motif_weights = _norm(motif_weights)  # ADDITION grammar (which motifs grow)
        self.remove_weights = _norm(remove_weights)  # REMOVAL grammar (which motifs decay)
        self.log_w = np.log(np.clip(self.motif_weights, _EPS, None))
        self.log_rw = np.log(np.clip(self.remove_weights, _EPS, None))
        self.edge_rate = float(edge_rate)
        self.edge_remove_rate = float(edge_remove_rate)
        self.node_rate = float(node_rate)
        self.name = name

    def __str__(self) -> str:
        return (
            "TemporalGraphGrammarDistribution(add_w=%s, edge_rate=%s, remove_w=%s, edge_remove_rate=%s, node_rate=%s)"
            % (
                np.array2string(self.motif_weights, precision=3),
                self.edge_rate,
                np.array2string(self.remove_weights, precision=3),
                self.edge_remove_rate,
                self.node_rate,
            )
        )

    def _edit_log_density(self, edit_bins: np.ndarray, log_w: np.ndarray, rate: float, cand: np.ndarray) -> float:
        """Shared add/remove term: per-motif Poisson(rate*w_m) x uniform anchor among motif-m candidates.

        ``edit_bins`` is the motif index of each edited edge. Equivalent to Poisson(total; rate) x
        Multinomial(w) x (1/cand_m per edit). Returns -inf if an edit's candidate pool is empty (e.g. a
        removal when rate == 0)."""
        k = len(edit_bins)
        lp = k * math.log(rate + _EPS) - rate - math.lgamma(k + 1)
        for mtf in edit_bins:
            if cand[mtf] <= 0:
                return float("-inf")
            lp += log_w[mtf] - math.log(cand[mtf])
        return lp

    def transition_components(self, prev: Any, cur: Any) -> tuple:
        """The PARAMETER-INDEPENDENT decomposition of a transition: ``(new_nodes, add_bins, add_cand,
        rem_bins, rem_cand, valid)``. Depends only on the graph pair and the motif, NOT on the grammar's
        weights/rates -- so K regimes sharing a motif can compute the (expensive A@A) decomposition ONCE and
        score it K times via :meth:`score_components`. ``valid`` is False for an impossible node removal."""
        n0, n1 = prev.shape[0], cur.shape[0]
        if n1 < n0:  # fewer nodes -> a node was removed, which the bare grammar does not model
            return 0, None, None, None, None, False
        new_nodes = n1 - n0
        ai, aj, ri, rj = _edge_diff(prev, cur, self.directed)
        add_cand, add_lookup = self.motif.counts_and_binner(_pad(prev, n1), on_edges=False)
        rem_cand, rem_lookup = self.motif.counts_and_binner(prev, on_edges=True)
        add_bins = add_lookup(np.asarray(ai), np.asarray(aj))
        rem_bins = rem_lookup(np.asarray(ri), np.asarray(rj))
        return new_nodes, add_bins, add_cand, rem_bins, rem_cand, True

    def score_components(self, components: tuple) -> float:
        """Score a precomputed :meth:`transition_components` decomposition under THIS grammar's parameters."""
        new_nodes, add_bins, add_cand, rem_bins, rem_cand, valid = components
        if not valid:
            return float("-inf")
        if len(rem_bins) and self.edge_remove_rate <= 0.0:  # a deletion under a no-removal (growth) grammar
            return float("-inf")
        lp = new_nodes * math.log(self.node_rate + _EPS) - self.node_rate - math.lgamma(new_nodes + 1)
        lp += self._edit_log_density(add_bins, self.log_w, self.edge_rate, add_cand)
        if lp == float("-inf"):
            return lp
        lp += self._edit_log_density(rem_bins, self.log_rw, self.edge_remove_rate, rem_cand)
        return lp

    def _transition_log_density(self, prev: Any, cur: Any) -> float:
        """log p(G_t | G_{t-1}): node-growth + an ADD grammar over new edges + a REMOVE grammar over deleted
        edges, each a per-motif Poisson scored against the PREVIOUS graph's structure (so order within a
        step is irrelevant). Works on dense OR sparse adjacencies. Node removal is not modelled -> -inf."""
        return self.score_components(self.transition_components(prev, cur))

    def log_density(self, x: Sequence[np.ndarray]) -> float:
        """Log-density of one dynamic graph: the sum of transition log-densities over the snapshot chain.

        ``x`` is a sequence of binary adjacency matrices -- dense ``ndarray`` or ``scipy.sparse`` (large
        graphs). The initial graph is taken as given (its marginal is not modelled, matching how the static
        grammars treat their start symbol)."""
        snaps = list(x)
        if len(snaps) < 2:
            return 0.0
        return float(sum(self._transition_log_density(snaps[t - 1], snaps[t]) for t in range(1, len(snaps))))

    def seq_encode(self, x: Sequence[Sequence[np.ndarray]]) -> Sequence[Sequence[np.ndarray]]:
        """Return dynamic graph sequences unchanged for sequence scoring."""
        return x

    def seq_log_density(self, x: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        """Score a batch of dynamic graph sequences."""
        return np.asarray([self.log_density(seq) for seq in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> TemporalGraphGrammarSampler:
        """Return a sampler for dynamic graph sequences."""
        return TemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> TemporalGraphGrammarEstimator:
        """Return the closed-form estimator for this motif grammar."""
        return TemporalGraphGrammarEstimator(self.motif, pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the pass-through data encoder for graph sequences."""
        return TemporalGraphGrammarDataEncoder()


# --- sampler --------------------------------------------------------------------------------------
class TemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for dynamic graph sequences generated by a motif-edit grammar."""

    def __init__(self, dist: TemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample_one(
        self, num_steps: int = 10, seed_graph: np.ndarray | None = None, n_init: int = 5
    ) -> list[np.ndarray]:
        """Run a derivation: from a seed graph, apply ``num_steps`` of grammar-sampled edits."""
        d = self.dist
        if seed_graph is None:
            adj = np.zeros((n_init, n_init), dtype=np.float64)
        elif sp.issparse(seed_graph):
            adj = seed_graph.toarray().astype(np.float64)  # sampler is dense/moderate-scale (see module doc)
        else:
            adj = np.asarray(seed_graph, dtype=np.float64).copy()
        snaps = [adj.copy()]
        for _ in range(num_steps):
            new_nodes = self.rng.poisson(d.node_rate)
            if new_nodes:
                n = adj.shape[0]
                big = np.zeros((n + new_nodes, n + new_nodes), dtype=np.float64)
                big[:n, :n] = adj
                adj = big
            self._edge_edit_step(adj)
            snaps.append(adj.copy())
        return snaps

    def _edge_edit_step(self, adj: np.ndarray) -> None:
        """Apply one step of the ADD + REMOVE edge grammars to ``adj`` in place (dense).

        Batch, pre-step motif assignment (NO within-step refresh) so the realized motif distribution matches
        the weights and equals what the scorer reads off the snapshots. Per motif m the edit count is
        Poisson(rate * w_m) -- the multinomial split of a Poisson(rate) total. Additions and removals both
        act on the start-of-step graph (disjoint -- non-edges vs edges)."""
        d = self.dist
        cand_mask = ~np.eye(adj.shape[0], dtype=bool) if d.directed else np.triu(np.ones(adj.shape, dtype=bool), 1)
        add_bins = d.motif.assign(adj, on_edges=False)
        rem_bins = d.motif.assign(adj, on_edges=True)
        toggles = []  # (i, j, value) applied after both grammars are sampled, against the pre-step graph
        for m in range(d.motif.num_motifs):
            ai, aj = np.where((add_bins == m) & cand_mask)
            if ai.shape[0]:
                ka = min(self.rng.poisson(d.edge_rate * d.motif_weights[m]), ai.shape[0])
                for idx in self.rng.choice(ai.shape[0], size=ka, replace=False):
                    toggles.append((ai[idx], aj[idx], 1.0))
            ri, rj = np.where((rem_bins == m) & cand_mask)
            if ri.shape[0] and d.edge_remove_rate > 0.0:
                kr = min(self.rng.poisson(d.edge_remove_rate * d.remove_weights[m]), ri.shape[0])
                for idx in self.rng.choice(ri.shape[0], size=kr, replace=False):
                    toggles.append((ri[idx], rj[idx], 0.0))
        for i, j, v in toggles:
            if d.directed:
                adj[i, j] = v
            else:
                adj[i, j] = adj[j, i] = v

    def sample_one_scalable(
        self,
        num_steps: int = 10,
        seed_edges: Sequence[tuple] | None = None,
        n_init: int = 5,
        max_reject: int = 64,
    ) -> list:
        """Sample a dynamic graph for a LARGE sparse graph -- never materialises the n*n adjacency.

        The dense :meth:`sample_one` is exact but O(n^2) in space (the full bin matrix). This path keeps the
        graph as an edge set and emits ``scipy.sparse`` snapshots, costing O(edges + wedges) per step:

        * triangle-closing motifs (cn>=1) are exactly the wedge non-edges -- the nonzeros of ``A @ A`` that
          aren't edges -- so they are enumerated directly from the wedge structure, never the full pair grid;
        * bridges (cn=0) dominate a sparse graph and can't be enumerated, so they are **rejection-sampled**:
          draw random pairs and accept those that are neither an edge nor a wedge (acceptance ~ 1 when the
          graph is sparse). ``max_reject`` attempts per bridge bound the loop; a shortfall is explicit
          capping (same realized-rate semantics as the dense sampler when a motif's anchors run out).

        Growth+removal; undirected or directed (directed: ordered i->j edges, full off-diagonal candidates,
        ``A@A`` = transitive i->k->j wedges). Returns a list of ``csr_array`` snapshots. The realized motif
        distribution matches the weights, so a model fit on these snapshots recovers the grammar.
        """
        d = self.dist
        directed = d.directed

        def canon(i: int, j: int) -> tuple:
            return (i, j) if directed else ((i, j) if i < j else (j, i))

        n = n_init if seed_edges is None else (max(max(e) for e in seed_edges) + 1 if seed_edges else n_init)
        edges = set() if seed_edges is None else {canon(i, j) for i, j in seed_edges}
        snaps = [self._csr(edges, n, directed)]
        for _ in range(num_steps):
            n += int(self.rng.poisson(d.node_rate))  # new isolated nodes
            a = self._csr(edges, n, directed)
            if directed:
                cnu = (a @ a).tocsr().copy()  # full transitive-path counts (i->k->j)
                cnu.setdiag(0)
                cnu.eliminate_zeros()
            else:
                cnu = sp.triu(a @ a, 1).tocsr()  # upper-tri common-neighbour counts (the wedge structure)
            em = self._edge_mask(edges, n)  # 1 at existing edges (canonical positions)
            edge_cn = cnu.multiply(em) if em is not None else None  # cn on existing edges (for removal binning)
            non_edge = cnu - edge_cn if edge_cn is not None else cnu  # cn on non-edges = the wedge non-edges
            non_edge.eliminate_zeros()
            nec = non_edge.tocoo()
            w_i, w_j, w_bin = nec.row, nec.col, d.motif._bin(nec.data)  # wedge non-edges + their motif bin (>=1)
            wedge_keys = w_i.astype(np.int64) * n + w_j  # encoded for O(log) membership in the bridge rejection
            wedge_keys.sort()
            add, remove = [], []
            # triangle motifs (m>=1): the wedge non-edges in each bin, vectorised
            for m in range(1, d.motif.num_motifs):
                pool = np.where(w_bin == m)[0]
                if pool.size:
                    k = min(int(self.rng.poisson(d.edge_rate * d.motif_weights[m])), pool.size)
                    pick = pool[self.rng.choice(pool.size, size=k, replace=False)]
                    add += list(zip(w_i[pick].tolist(), w_j[pick].tolist()))
            # bridges (m=0): rejection-sample random non-edge / non-wedge pairs
            if n > 1:
                k0 = int(self.rng.poisson(d.edge_rate * d.motif_weights[0]))
                chosen: set = set()
                for _try in range(k0 * max_reject):
                    if len(chosen) >= k0:
                        break
                    i, j = int(self.rng.randint(n)), int(self.rng.randint(n))
                    if i == j:
                        continue
                    key = canon(i, j)
                    enc = key[0] * n + key[1]
                    pos = np.searchsorted(wedge_keys, enc)
                    is_wedge = pos < wedge_keys.size and wedge_keys[pos] == enc
                    if key in edges or is_wedge or key in chosen:
                        continue
                    chosen.add(key)
                add += list(chosen)
            # removals: existing edges binned by their cn (enumerated -- O(edges))
            if d.edge_remove_rate > 0.0 and edges:
                ec = edge_cn.tocoo() if edge_cn is not None else None
                rby: dict = {0: list(edges)}  # default every edge to the bridge bin, then reassign triangle edges
                if ec is not None and ec.nnz:
                    ekeys = {(int(i), int(j)) for i, j in zip(ec.row, ec.col)}
                    rby[0] = [e for e in edges if e not in ekeys]
                    ebins = d.motif._bin(ec.data)
                    for i, j, b in zip(ec.row, ec.col, ebins):
                        rby.setdefault(int(b), []).append((int(i), int(j)))
                for m in range(d.motif.num_motifs):
                    pool = rby.get(m, [])
                    if pool:
                        k = min(int(self.rng.poisson(d.edge_remove_rate * d.remove_weights[m])), len(pool))
                        remove += [pool[idx] for idx in self.rng.choice(len(pool), size=k, replace=False)]
            edges |= set(add)
            edges -= set(remove)
            snaps.append(self._csr(edges, n, directed))
        return snaps

    @staticmethod
    def _edge_mask(edges: set, n: int) -> Any:
        if not edges:
            return None
        ij = np.fromiter((c for e in edges for c in e), dtype=np.int64, count=2 * len(edges)).reshape(-1, 2)
        return sp.csr_array((np.ones(len(edges)), (ij[:, 0], ij[:, 1])), shape=(n, n))

    @staticmethod
    def _csr(edges: set, n: int, directed: bool = False) -> Any:
        if not edges:
            return sp.csr_array((n, n))
        ij = np.fromiter((c for e in edges for c in e), dtype=np.int64, count=2 * len(edges)).reshape(-1, 2)
        if directed:  # ordered edges, asymmetric adjacency
            return sp.csr_array((np.ones(len(edges)), (ij[:, 0], ij[:, 1])), shape=(n, n))
        rows = np.concatenate([ij[:, 0], ij[:, 1]])
        cols = np.concatenate([ij[:, 1], ij[:, 0]])
        return sp.csr_array((np.ones(rows.size), (rows, cols)), shape=(n, n))

    def sample(self, size: int | None = None, *, num_steps: int = 10, n_init: int = 5) -> Any:
        """Draw one sequence or a list of sequences from the grammar."""
        if size is None:
            return self.sample_one(num_steps=num_steps, n_init=n_init)
        return [self.sample_one(num_steps=num_steps, n_init=n_init) for _ in range(size)]


# --- estimator / accumulator ----------------------------------------------------------------------
class TemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate per-motif edge counts + step/edge/node totals -- the exact sufficient statistics."""

    def __init__(self, motif: CommonNeighbourMotif) -> None:
        self.motif = motif
        self.add_counts = np.zeros(motif.num_motifs, dtype=np.float64)
        self.rem_counts = np.zeros(motif.num_motifs, dtype=np.float64)
        self.edges = 0.0
        self.rem_edges = 0.0
        self.nodes = 0.0
        self.steps = 0.0

    def update(self, x: Sequence[Any], weight: float, estimate: Any | None) -> None:
        """Accumulate sufficient statistics from one dynamic graph sequence."""
        snaps = list(x)  # adjacencies may be dense ndarrays or scipy.sparse
        for t in range(1, len(snaps)):
            prev, cur = snaps[t - 1], snaps[t]
            ai, aj, ri, rj = _edge_diff(prev, cur, self.motif.directed)
            _, add_lookup = self.motif.counts_and_binner(_pad(prev, cur.shape[0]), on_edges=False)
            _, rem_lookup = self.motif.counts_and_binner(prev, on_edges=True)
            for m in add_lookup(np.asarray(ai), np.asarray(aj)):
                self.add_counts[m] += weight
            for m in rem_lookup(np.asarray(ri), np.asarray(rj)):
                self.rem_counts[m] += weight
            self.edges += weight * len(ai)
            self.rem_edges += weight * len(ri)
            self.nodes += weight * (cur.shape[0] - prev.shape[0])
            self.steps += weight

    def seq_update(self, x: Sequence[Sequence[np.ndarray]], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted sufficient statistics from a batch of sequences."""
        for seq, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.update(seq, float(w), estimate)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        self.seq_update(x, weights, None)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted sequence."""
        self.update(x, weight, None)

    def combine(self, suff_stat: tuple) -> TemporalGraphGrammarAccumulator:
        """Merge serialized sufficient statistics into this accumulator."""
        ac, rc, e, re, n, s = suff_stat
        self.add_counts += ac
        self.rem_counts += rc
        self.edges += e
        self.rem_edges += re
        self.nodes += n
        self.steps += s
        return self

    def value(self) -> tuple:
        """Return serialized sufficient statistics for estimation or merging."""
        return self.add_counts.copy(), self.rem_counts.copy(), self.edges, self.rem_edges, self.nodes, self.steps

    def from_value(self, x: tuple) -> TemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        self.add_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.rem_counts = np.asarray(x[1], dtype=np.float64).copy()
        self.edges, self.rem_edges, self.nodes, self.steps = float(x[2]), float(x[3]), float(x[4]), float(x[5])
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return TemporalGraphGrammarDataEncoder()


class TemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for temporal graph grammar accumulators."""

    def __init__(self, motif: CommonNeighbourMotif) -> None:
        self.motif = motif

    def make(self) -> TemporalGraphGrammarAccumulator:
        """Create a fresh temporal graph grammar accumulator."""
        return TemporalGraphGrammarAccumulator(self.motif)


class TemporalGraphGrammarEstimator(ParameterEstimator):
    """Learn the motif distribution (rule weights) + edge/node rates from observed dynamic graphs."""

    def __init__(
        self, motif: CommonNeighbourMotif | None = None, pseudo_count: float | None = None, name: str | None = None
    ) -> None:
        self.motif = motif if motif is not None else CommonNeighbourMotif()
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> TemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by the estimator."""
        return TemporalGraphGrammarAccumulatorFactory(self.motif)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> TemporalGraphGrammarDistribution:
        """Estimate grammar weights and rates from sufficient statistics."""
        add_counts, rem_counts, edges, rem_edges, nodes, steps = suff_stat

        def _w(counts: np.ndarray) -> np.ndarray:
            c = np.asarray(counts, dtype=np.float64).copy()
            if self.pseudo_count is not None:
                c = c + float(self.pseudo_count)
            return c / c.sum() if c.sum() > 0 else np.ones(self.motif.num_motifs) / self.motif.num_motifs

        return TemporalGraphGrammarDistribution(
            _w(add_counts),
            edges / steps if steps > 0 else 1.0,
            nodes / steps if steps > 0 else 0.0,
            remove_weights=_w(rem_counts),
            edge_remove_rate=rem_edges / steps if steps > 0 else 0.0,
            motif=self.motif,
            name=self.name,
        )


# --- encoder --------------------------------------------------------------------------------------
class TemporalGraphGrammarDataEncoder(DataSequenceEncoder):
    """Pass-through encoder for dynamic graph sequence observations."""

    def seq_encode(self, x: Sequence[Sequence[np.ndarray]]) -> Sequence[Sequence[np.ndarray]]:
        """Return graph sequences unchanged."""
        return x

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TemporalGraphGrammarDataEncoder)


# --- labelled (attributed) dynamic graphs ---------------------------------------------------------
def _emission_ll(dist: Any, records: Sequence[Any]) -> float:
    if dist is None or not records:
        return 0.0
    enc = dist.dist_to_encoder().seq_encode(list(records))
    return float(np.sum(dist.seq_log_density(enc)))


class LabeledTemporalGraphGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """A dynamic graph whose nodes and edges carry attributes.

    Composes a structural :class:`TemporalGraphGrammarDistribution` (the topology over time) with two
    ordinary mixle distributions: ``node_dist`` over per-node attribute records (location, name, age, ... --
    typically a ``CompositeDistribution`` of leaves or a mixture) and ``edge_dist`` over per-edge attribute
    records (communication counts, channel, weight, ...). An observation is ``(snapshots, node_features,
    edge_features)``: the adjacency chain, one attribute record per node, and one per added edge. The
    likelihood factorises -- structure x node attributes x edge attributes -- so the attribute models are
    fit (and scored) with the full mixle distribution machinery (mixtures, fusion, all leaf families).
    """

    def __init__(
        self,
        structure: TemporalGraphGrammarDistribution,
        node_dist: SequenceEncodableProbabilityDistribution | None = None,
        edge_dist: SequenceEncodableProbabilityDistribution | None = None,
        name: str | None = None,
    ) -> None:
        self.structure = structure
        self.node_dist = node_dist
        self.edge_dist = edge_dist
        self.name = name

    def __str__(self) -> str:
        return "LabeledTemporalGraphGrammarDistribution(structure=%s, node_dist=%s, edge_dist=%s)" % (
            self.structure,
            self.node_dist,
            self.edge_dist,
        )

    def log_density(self, x: tuple) -> float:
        """Score one attributed dynamic graph observation."""
        snaps, node_features, edge_features = x
        return (
            self.structure.log_density(snaps)
            + _emission_ll(self.node_dist, node_features)
            + _emission_ll(self.edge_dist, edge_features)
        )

    def seq_encode(self, x: Sequence[tuple]) -> Sequence[tuple]:
        """Return attributed dynamic graph observations unchanged."""
        return x

    def seq_log_density(self, x: Sequence[tuple]) -> np.ndarray:
        """Score a batch of attributed dynamic graph observations."""
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> LabeledTemporalGraphGrammarSampler:
        """Return a sampler for attributed dynamic graph observations."""
        return LabeledTemporalGraphGrammarSampler(self, seed)

    def estimator(self, **kw: Any) -> LabeledTemporalGraphGrammarEstimator:
        """Return the estimator for structure, node attributes, and edge attributes."""
        return LabeledTemporalGraphGrammarEstimator(
            self.structure.estimator(**kw),
            None if self.node_dist is None else self.node_dist.estimator(),
            None if self.edge_dist is None else self.edge_dist.estimator(),
            name=self.name,
        )

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the pass-through graph encoder."""
        return TemporalGraphGrammarDataEncoder()


class LabeledTemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for attributed dynamic graph observations."""

    def __init__(self, dist: LabeledTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.struct = dist.structure.sampler(self.rng.randint(2**31))

    def sample_one(self, **kw: Any) -> tuple:
        """Draw one attributed dynamic graph observation."""
        snaps = self.struct.sample_one(**kw)
        n_final = snaps[-1].shape[0]
        directed = getattr(self.dist.structure, "directed", False)
        num_added = sum(len(_edge_diff(snaps[t - 1], snaps[t], directed)[0]) for t in range(1, len(snaps)))
        node_features = (
            list(self.dist.node_dist.sampler(self.rng.randint(2**31)).sample(size=n_final))
            if self.dist.node_dist is not None
            else []
        )
        edge_features = (
            list(self.dist.edge_dist.sampler(self.rng.randint(2**31)).sample(size=num_added))
            if self.dist.edge_dist is not None and num_added
            else []
        )
        return snaps, node_features, edge_features

    def sample(self, size: int | None = None, **kw: Any) -> Any:
        """Draw one observation or a list of observations."""
        if size is None:
            return self.sample_one(**kw)
        return [self.sample_one(**kw) for _ in range(size)]


class LabeledTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for structure, node-attribute, and edge-attribute sufficient statistics."""

    def __init__(self, structure_acc: Any, node_acc: Any, edge_acc: Any) -> None:
        self.structure_acc = structure_acc
        self.node_acc = node_acc
        self.edge_acc = edge_acc

    def update(self, x: tuple, weight: float, estimate: Any | None) -> None:
        """Accumulate sufficient statistics from one attributed graph observation."""
        snaps, node_features, edge_features = x
        self.structure_acc.update(snaps, weight, None if estimate is None else estimate.structure)
        if self.node_acc is not None and node_features:
            nd = None if estimate is None else estimate.node_dist
            enc = nd.dist_to_encoder().seq_encode(list(node_features)) if nd is not None else node_features
            self.node_acc.seq_update(enc, np.full(len(node_features), weight), nd)
        if self.edge_acc is not None and edge_features:
            ed = None if estimate is None else estimate.edge_dist
            enc = ed.dist_to_encoder().seq_encode(list(edge_features)) if ed is not None else edge_features
            self.edge_acc.seq_update(enc, np.full(len(edge_features), weight), ed)

    def seq_update(self, x: Sequence[tuple], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted sufficient statistics from a batch."""
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.update(obs, float(w), estimate)

    def initialize(self, x: tuple, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted observation."""
        self.update(x, weight, None)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple) -> LabeledTemporalGraphGrammarAccumulator:
        """Merge serialized attributed-graph sufficient statistics."""
        s, n, e = suff_stat
        self.structure_acc.combine(s)
        if self.node_acc is not None:
            self.node_acc.combine(n)
        if self.edge_acc is not None:
            self.edge_acc.combine(e)
        return self

    def value(self) -> tuple:
        """Return serialized attributed-graph sufficient statistics."""
        return (
            self.structure_acc.value(),
            None if self.node_acc is None else self.node_acc.value(),
            None if self.edge_acc is None else self.edge_acc.value(),
        )

    def from_value(self, x: tuple) -> LabeledTemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        self.structure_acc.from_value(x[0])
        if self.node_acc is not None:
            self.node_acc.from_value(x[1])
        if self.edge_acc is not None:
            self.edge_acc.from_value(x[2])
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return TemporalGraphGrammarDataEncoder()


class LabeledTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for attributed temporal graph grammar accumulators."""

    def __init__(self, structure_factory: Any, node_factory: Any, edge_factory: Any) -> None:
        self.structure_factory = structure_factory
        self.node_factory = node_factory
        self.edge_factory = edge_factory

    def make(self) -> LabeledTemporalGraphGrammarAccumulator:
        """Create a fresh attributed temporal graph grammar accumulator."""
        return LabeledTemporalGraphGrammarAccumulator(
            self.structure_factory.make(),
            None if self.node_factory is None else self.node_factory.make(),
            None if self.edge_factory is None else self.edge_factory.make(),
        )


class LabeledTemporalGraphGrammarEstimator(ParameterEstimator):
    """Estimator for attributed temporal graph grammars."""

    def __init__(
        self, structure_estimator: Any, node_estimator: Any = None, edge_estimator: Any = None, name: str | None = None
    ) -> None:
        self.structure_estimator = structure_estimator
        self.node_estimator = node_estimator
        self.edge_estimator = edge_estimator
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> LabeledTemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return LabeledTemporalGraphGrammarAccumulatorFactory(
            self.structure_estimator.accumulator_factory(),
            None if self.node_estimator is None else self.node_estimator.accumulator_factory(),
            None if self.edge_estimator is None else self.edge_estimator.accumulator_factory(),
        )

    def estimate(self, nobs: float | None, suff_stat: tuple) -> LabeledTemporalGraphGrammarDistribution:
        """Estimate structure and attribute distributions from sufficient statistics."""
        s_val, n_val, e_val = suff_stat
        return LabeledTemporalGraphGrammarDistribution(
            self.structure_estimator.estimate(nobs, s_val),
            None if self.node_estimator is None else self.node_estimator.estimate(nobs, n_val),
            None if self.edge_estimator is None else self.edge_estimator.estimate(nobs, e_val),
            name=self.name,
        )


__all__ = [
    "CommonNeighbourMotif",
    "TemporalGraphGrammarDistribution",
    "TemporalGraphGrammarSampler",
    "TemporalGraphGrammarEstimator",
    "TemporalGraphGrammarAccumulator",
    "TemporalGraphGrammarAccumulatorFactory",
    "TemporalGraphGrammarDataEncoder",
    "LabeledTemporalGraphGrammarDistribution",
    "LabeledTemporalGraphGrammarSampler",
    "LabeledTemporalGraphGrammarEstimator",
    "LabeledTemporalGraphGrammarAccumulator",
    "LabeledTemporalGraphGrammarAccumulatorFactory",
    "HomophilyTemporalGraphGrammarDistribution",
    "HomophilyTemporalGraphGrammarSampler",
    "HomophilyTemporalGraphGrammarEstimator",
    "HomophilyTemporalGraphGrammarAccumulator",
    "HomophilyTemporalGraphGrammarAccumulatorFactory",
    "ChurningTemporalGraphGrammarDistribution",
    "ChurningTemporalGraphGrammarSampler",
    "ChurningTemporalGraphGrammarEstimator",
    "ChurningTemporalGraphGrammarAccumulator",
    "ChurningTemporalGraphGrammarAccumulatorFactory",
    "LatentTemporalGraphGrammarDistribution",
    "LatentTemporalGraphGrammarSampler",
    "LatentTemporalGraphGrammarEstimator",
    "LatentTemporalGraphGrammarAccumulator",
    "LatentTemporalGraphGrammarAccumulatorFactory",
    "LatentAttributedTemporalGraphGrammarDistribution",
    "LatentAttributedTemporalGraphGrammarSampler",
    "LatentAttributedTemporalGraphGrammarEstimator",
    "LatentAttributedTemporalGraphGrammarAccumulator",
    "LatentAttributedTemporalGraphGrammarAccumulatorFactory",
    "LatentChurningTemporalGraphGrammarDistribution",
    "LatentChurningTemporalGraphGrammarSampler",
    "LatentChurningTemporalGraphGrammarEstimator",
    "LatentChurningTemporalGraphGrammarAccumulator",
    "LatentChurningTemporalGraphGrammarAccumulatorFactory",
    "regime_moment_init",
]


# --- homophily: attribute-conditioned edge formation ----------------------------------------------
class HomophilyTemporalGraphGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """A growth grammar whose edge formation depends on node ATTRIBUTES, not just structure (homophily).

    Each node carries a categorical ``type`` (community / location-bucket / ...). The per-step number of new
    edges of motif ``m`` between an (unordered) type pair (a, b) is ``Poisson(rate[m, a, b])``, placed
    uniformly among the candidate non-edges of that motif and type pair. Making ``rate[m, a, a]`` larger
    than ``rate[m, a, b]`` is homophily ("similar nodes connect more"); the rate tensor is the learnable
    coupling between attributes and topology. New nodes draw their type from ``type_weights``.

    Observation: ``(snapshots, node_types)`` -- the adjacency chain plus an int type per node. Exact and
    closed-form: the rate tensor is just edge counts per (motif, type-pair) over steps, and the type
    distribution is node-type counts. (Phase: growth-only, dense; add+remove and sparse compose with the
    machinery above and are the natural extensions.)
    """

    def __init__(
        self,
        rate: np.ndarray,
        type_weights: Sequence[float],
        node_rate: float = 0.0,
        motif: CommonNeighbourMotif | None = None,
        name: str | None = None,
    ) -> None:
        self.motif = motif if motif is not None else CommonNeighbourMotif()
        self.rate = np.asarray(rate, dtype=np.float64)  # (M, K, K), symmetric in the last two axes
        self.M, self.K = self.rate.shape[0], self.rate.shape[1]
        tw = np.asarray(type_weights, dtype=np.float64)
        self.type_weights = tw / tw.sum()
        self.log_tw = np.log(np.clip(self.type_weights, _EPS, None))
        self.node_rate = float(node_rate)
        self.name = name

    def __str__(self) -> str:
        return "HomophilyTemporalGraphGrammarDistribution(K=%d, type_w=%s, node_rate=%s)" % (
            self.K,
            np.array2string(self.type_weights, precision=3),
            self.node_rate,
        )

    def _pair_axes(self, ii: np.ndarray, jj: np.ndarray, types: np.ndarray) -> tuple:
        ti, tj = types[ii], types[jj]
        return np.minimum(ti, tj), np.maximum(ti, tj)

    def _cand_counts(self, padded: Any, types: np.ndarray) -> np.ndarray:
        b = self.motif.assign(padded, on_edges=False)  # (n,n) non-edge motif bins, -1 elsewhere
        ut = np.triu(np.ones(b.shape, dtype=bool), 1)
        ii, jj = np.where(ut & (b >= 0))
        a, bb = self._pair_axes(ii, jj, types)
        cand = np.zeros((self.M, self.K, self.K), dtype=np.float64)
        np.add.at(cand, (b[ii, jj], a, bb), 1.0)
        return cand

    def _transition_log_density(self, prev: Any, cur: Any, types: np.ndarray) -> float:
        n0, n1 = prev.shape[0], cur.shape[0]
        if n1 < n0:
            return float("-inf")
        ai, aj, ri, rj = _edge_diff(prev, cur)
        if len(ri):  # growth-only homophily phase
            return float("-inf")
        padded = _pad(prev, n1)
        cand = self._cand_counts(padded, types)
        _, lookup = self.motif.counts_and_binner(padded, on_edges=False)
        new_nodes = n1 - n0
        lp = new_nodes * math.log(self.node_rate + _EPS) - self.node_rate - math.lgamma(new_nodes + 1)
        lp -= float(self.rate.sum())  # the -rate Poisson normaliser over every (motif, type-pair) cell
        if len(ai):
            m = lookup(np.asarray(ai), np.asarray(aj))
            a, b = self._pair_axes(np.asarray(ai), np.asarray(aj), types)
            for mm, aa, bb in zip(m.tolist(), a.tolist(), b.tolist()):
                if self.rate[mm, aa, bb] <= 0 or cand[mm, aa, bb] <= 0:
                    return float("-inf")
                lp += math.log(self.rate[mm, aa, bb]) - math.log(cand[mm, aa, bb])  # weight x uniform anchor
        return lp

    def log_density(self, x: tuple) -> float:
        """Score one homophily dynamic graph observation."""
        snaps, types = x
        types = np.asarray(types, dtype=np.int64)
        lp = float(np.sum(self.log_tw[types]))  # node-type likelihood (each node's type ~ Categorical)
        lp += sum(self._transition_log_density(snaps[t - 1], snaps[t], types) for t in range(1, len(snaps)))
        return lp

    def seq_encode(self, x: Sequence[tuple]) -> Sequence[tuple]:
        """Return homophily observations unchanged for sequence scoring."""
        return x

    def seq_log_density(self, x: Sequence[tuple]) -> np.ndarray:
        """Score a batch of homophily dynamic graph observations."""
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> HomophilyTemporalGraphGrammarSampler:
        """Return a sampler for homophily dynamic graph observations."""
        return HomophilyTemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> HomophilyTemporalGraphGrammarEstimator:
        """Return the estimator for homophily rates and node-type weights."""
        return HomophilyTemporalGraphGrammarEstimator(self.M, self.K, self.motif, pseudo_count, self.name)

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the pass-through graph encoder."""
        return TemporalGraphGrammarDataEncoder()


class HomophilyTemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for homophily temporal graph observations."""

    def __init__(self, dist: HomophilyTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample_one(self, num_steps: int = 8, seed_graph: np.ndarray | None = None, n_init: int = 8) -> tuple:
        """Draw one homophily dynamic graph observation."""
        d = self.dist
        adj = np.zeros((n_init, n_init)) if seed_graph is None else np.asarray(seed_graph, dtype=np.float64).copy()
        types = list(self.rng.choice(d.K, size=adj.shape[0], p=d.type_weights))
        snaps = [adj.copy()]
        for _ in range(num_steps):
            new_nodes = self.rng.poisson(d.node_rate)
            if new_nodes:
                n = adj.shape[0]
                big = np.zeros((n + new_nodes, n + new_nodes))
                big[:n, :n] = adj
                adj = big
                types += list(self.rng.choice(d.K, size=new_nodes, p=d.type_weights))
            tarr = np.asarray(types)
            b = d.motif.assign(adj, on_edges=False)
            ut = np.triu(np.ones(adj.shape, dtype=bool), 1)
            for m in range(d.M):
                ii, jj = np.where((b == m) & ut)
                if not ii.shape[0]:
                    continue
                a, bb = np.minimum(tarr[ii], tarr[jj]), np.maximum(tarr[ii], tarr[jj])
                for aa in range(d.K):
                    for cc in range(aa, d.K):
                        sel = (a == aa) & (bb == cc)
                        idx = np.where(sel)[0]
                        if not idx.shape[0]:
                            continue
                        k = min(self.rng.poisson(d.rate[m, aa, cc]), idx.shape[0])
                        for p in self.rng.choice(idx, size=k, replace=False):
                            adj[ii[p], jj[p]] = adj[jj[p], ii[p]] = 1.0
            snaps.append(adj.copy())
        return snaps, np.asarray(types, dtype=np.int64)

    def sample(self, size: int | None = None, **kw: Any) -> Any:
        """Draw one observation or a list of observations."""
        if size is None:
            return self.sample_one(**kw)
        return [self.sample_one(**kw) for _ in range(size)]


class HomophilyTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for homophily edge-rate and node-type sufficient statistics."""

    def __init__(self, M: int, K: int, motif: CommonNeighbourMotif) -> None:
        self.M, self.K, self.motif = M, K, motif
        self.edge_counts = np.zeros((M, K, K), dtype=np.float64)
        self.type_counts = np.zeros(K, dtype=np.float64)
        self.nodes = 0.0
        self.steps = 0.0

    def update(self, x: tuple, weight: float, estimate: Any | None) -> None:
        """Accumulate sufficient statistics from one homophily observation."""
        snaps, types = x
        types = np.asarray(types, dtype=np.int64)
        np.add.at(self.type_counts, types, weight)
        for t in range(1, len(snaps)):
            prev, cur = snaps[t - 1], snaps[t]
            ai, aj, _, _ = _edge_diff(prev, cur)
            _, lookup = self.motif.counts_and_binner(_pad(prev, cur.shape[0]), on_edges=False)
            if len(ai):
                m = lookup(np.asarray(ai), np.asarray(aj))
                a = np.minimum(types[np.asarray(ai)], types[np.asarray(aj)])
                b = np.maximum(types[np.asarray(ai)], types[np.asarray(aj)])
                np.add.at(self.edge_counts, (m, a, b), weight)
            self.nodes += weight * (cur.shape[0] - prev.shape[0])
            self.steps += weight

    def seq_update(self, x: Sequence[tuple], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted sufficient statistics from a batch."""
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.update(obs, float(w), estimate)

    def initialize(self, x: tuple, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted observation."""
        self.update(x, weight, None)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple) -> HomophilyTemporalGraphGrammarAccumulator:
        """Merge serialized homophily sufficient statistics."""
        ec, tc, n, s = suff_stat
        self.edge_counts += ec
        self.type_counts += tc
        self.nodes += n
        self.steps += s
        return self

    def value(self) -> tuple:
        """Return serialized homophily sufficient statistics."""
        return self.edge_counts.copy(), self.type_counts.copy(), self.nodes, self.steps

    def from_value(self, x: tuple) -> HomophilyTemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        self.edge_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.type_counts = np.asarray(x[1], dtype=np.float64).copy()
        self.nodes, self.steps = float(x[2]), float(x[3])
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return TemporalGraphGrammarDataEncoder()


class HomophilyTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for homophily temporal graph grammar accumulators."""

    def __init__(self, M: int, K: int, motif: CommonNeighbourMotif) -> None:
        self.M, self.K, self.motif = M, K, motif

    def make(self) -> HomophilyTemporalGraphGrammarAccumulator:
        """Create a fresh homophily accumulator."""
        return HomophilyTemporalGraphGrammarAccumulator(self.M, self.K, self.motif)


class HomophilyTemporalGraphGrammarEstimator(ParameterEstimator):
    """Estimator for homophily temporal graph grammars."""

    def __init__(
        self,
        M: int,
        K: int,
        motif: CommonNeighbourMotif | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
    ) -> None:
        self.M, self.K = M, K
        self.motif = motif if motif is not None else CommonNeighbourMotif()
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> HomophilyTemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return HomophilyTemporalGraphGrammarAccumulatorFactory(self.M, self.K, self.motif)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> HomophilyTemporalGraphGrammarDistribution:
        """Estimate homophily rates, type weights, and node rate from sufficient statistics."""
        edge_counts, type_counts, nodes, steps = suff_stat
        rate = np.asarray(edge_counts, dtype=np.float64) / steps if steps > 0 else np.asarray(edge_counts)
        tc = np.asarray(type_counts, dtype=np.float64).copy()
        if self.pseudo_count is not None:
            tc = tc + float(self.pseudo_count)
        type_weights = tc / tc.sum() if tc.sum() > 0 else np.ones(self.K) / self.K
        return HomophilyTemporalGraphGrammarDistribution(
            rate, type_weights, nodes / steps if steps > 0 else 0.0, motif=self.motif, name=self.name
        )


# --- node churn: removal + addition with identity tracking -----------------------------------------
def _node_removal_logp(rate: float, n_prev: int, k_removed: int) -> float:
    """log p(remove k of n_prev nodes) = Poisson(k; rate) x uniform over which k-subset."""
    if k_removed > n_prev:
        return float("-inf")
    lp = k_removed * math.log(rate + _EPS) - rate - math.lgamma(k_removed + 1)
    return lp - math.lgamma(n_prev + 1) + math.lgamma(k_removed + 1) + math.lgamma(n_prev - k_removed + 1)


def _align_by_ids(prev_adj: Any, prev_ids: Sequence[int], cur_adj: Any, cur_ids: Sequence[int]) -> tuple:
    """Align two snapshots by stable node id. Returns (prev_surviving_subgraph, cur_reordered, num_removed).

    Removed nodes = ids in prev but not cur; their incident edges vanish with them (not counted as edge
    removals). ``cur`` is reordered so the surviving nodes (in prev order) come first and the genuinely-new
    nodes are appended -- exactly the ``prev' -> cur`` layout the edit grammar expects (shared nodes keep
    their index, new nodes at the end)."""
    pid, cid = list(prev_ids), list(cur_ids)
    cpos = {nid: k for k, nid in enumerate(cid)}
    pset = set(pid)
    surv = [k for k, nid in enumerate(pid) if nid in cpos]  # prev positions of survivors, in prev order
    surv_ids = [pid[k] for k in surv]
    new = [k for k, nid in enumerate(cid) if nid not in pset]  # cur positions of brand-new nodes
    num_removed = len(pid) - len(surv)
    order = [cpos[nid] for nid in surv_ids] + new
    if sp.issparse(prev_adj) or sp.issparse(cur_adj):  # keep large churned graphs sparse through the alignment
        pa, ca = sp.csr_array(prev_adj), sp.csr_array(cur_adj)
        si, oi = np.asarray(surv, dtype=np.int64), np.asarray(order, dtype=np.int64)
        prev_surv = pa[si, :][:, si] if surv else sp.csr_array((0, 0))
        cur_reord = ca[oi, :][:, oi] if order else sp.csr_array((0, 0))
        return prev_surv, cur_reord, num_removed
    pa = np.asarray(prev_adj, dtype=np.float64)
    ca = np.asarray(cur_adj, dtype=np.float64)
    prev_surv = pa[np.ix_(surv, surv)] if surv else np.zeros((0, 0))
    cur_reord = ca[np.ix_(order, order)] if order else np.zeros((0, 0))
    return prev_surv, cur_reord, num_removed


class ChurningTemporalGraphGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """Dynamic graph where nodes both JOIN and LEAVE, tracked by stable identity.

    Each snapshot is ``(adjacency, node_ids)`` -- ``node_ids[i]`` is the persistent identity of row i. A
    transition first **removes** nodes (those whose id disappears; count ~ Poisson(node_remove_rate), chosen
    uniformly, their edges vanishing with them), then runs the wrapped edit grammar on the surviving
    subgraph (which also appends new nodes + adds/removes edges). So churn is a thin wrapper: identity
    alignment + a node-removal Poisson term on top of all the existing motif/edge machinery. Scoring and
    fitting accept dense or ``scipy.sparse`` adjacencies (the id alignment slices either); the sampler is dense.
    """

    def __init__(
        self,
        edit_grammar: TemporalGraphGrammarDistribution,
        node_remove_rate: float = 0.0,
        name: str | None = None,
    ) -> None:
        self.edit_grammar = edit_grammar
        self.node_remove_rate = float(node_remove_rate)
        self.name = name

    def __str__(self) -> str:
        return "ChurningTemporalGraphGrammarDistribution(node_remove_rate=%s, edit=%s)" % (
            self.node_remove_rate,
            self.edit_grammar,
        )

    def _node_removal_log_density(self, n_prev: int, k_removed: int) -> float:
        return _node_removal_logp(self.node_remove_rate, n_prev, k_removed)

    def log_density(self, x: Sequence[tuple]) -> float:
        """Score one identity-tracked churning graph sequence."""
        snaps = list(x)
        if len(snaps) < 2:
            return 0.0
        lp = 0.0
        for t in range(1, len(snaps)):
            pa, pid = snaps[t - 1]
            ca, cid = snaps[t]
            prev_surv, cur_reord, num_removed = _align_by_ids(pa, pid, ca, cid)
            lp += self._node_removal_log_density(len(pid), num_removed)
            if lp == float("-inf"):
                return lp
            lp += self.edit_grammar._transition_log_density(prev_surv, cur_reord)
        return lp

    def seq_encode(self, x: Sequence[Any]) -> Sequence[Any]:
        """Return churning graph observations unchanged for scoring."""
        return x

    def seq_log_density(self, x: Sequence[Any]) -> np.ndarray:
        """Score a batch of churning graph observations."""
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> ChurningTemporalGraphGrammarSampler:
        """Return a sampler for churning graph observations."""
        return ChurningTemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ChurningTemporalGraphGrammarEstimator:
        """Return the estimator for the wrapped edit grammar and node churn rate."""
        return ChurningTemporalGraphGrammarEstimator(
            self.edit_grammar.estimator(pseudo_count=pseudo_count), name=self.name
        )

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the pass-through graph encoder."""
        return TemporalGraphGrammarDataEncoder()


class ChurningTemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for identity-tracked churning temporal graph observations."""

    def __init__(self, dist: ChurningTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.edit_sampler = dist.edit_grammar.sampler(self.rng.randint(2**31))

    def sample_one(self, num_steps: int = 10, seed_graph: np.ndarray | None = None, n_init: int = 8) -> list:
        """Draw one churning graph sequence."""
        d = self.dist
        adj = np.zeros((n_init, n_init)) if seed_graph is None else np.asarray(seed_graph, dtype=np.float64).copy()
        ids = list(range(adj.shape[0]))
        next_id = adj.shape[0]
        snaps = [(adj.copy(), list(ids))]
        for _ in range(num_steps):
            # 1) remove nodes (uniformly), dropping their incident edges
            n = adj.shape[0]
            k_rem = min(int(self.rng.poisson(d.node_remove_rate)), n)
            if k_rem:
                drop = set(self.rng.choice(n, size=k_rem, replace=False).tolist())
                keep = [i for i in range(n) if i not in drop]
                adj = adj[np.ix_(keep, keep)] if keep else np.zeros((0, 0))
                ids = [ids[i] for i in keep]
            # 2) add new nodes (node_rate), with fresh ids
            new_nodes = int(self.rng.poisson(d.edit_grammar.node_rate))
            if new_nodes:
                m = adj.shape[0]
                big = np.zeros((m + new_nodes, m + new_nodes))
                big[:m, :m] = adj
                adj = big
                ids += list(range(next_id, next_id + new_nodes))
                next_id += new_nodes
            # 3) edge edits via the wrapped grammar (same realized motif distribution as the scorer)
            if adj.shape[0]:
                self.edit_sampler._edge_edit_step(adj)
            snaps.append((adj.copy(), list(ids)))
        return snaps

    def sample(self, size: int | None = None, **kw: Any) -> Any:
        """Draw one sequence or a list of sequences."""
        if size is None:
            return self.sample_one(**kw)
        return [self.sample_one(**kw) for _ in range(size)]


class ChurningTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for churning graph grammar and node-removal sufficient statistics."""

    def __init__(self, edit_acc: Any) -> None:
        self.edit_acc = edit_acc
        self.removed = 0.0
        self.steps = 0.0

    def update(self, x: Sequence[tuple], weight: float, estimate: Any | None) -> None:
        """Accumulate sufficient statistics from one churning graph sequence."""
        snaps = list(x)
        edit_est = None if estimate is None else estimate.edit_grammar
        for t in range(1, len(snaps)):
            pa, pid = snaps[t - 1]
            ca, cid = snaps[t]
            prev_surv, cur_reord, num_removed = _align_by_ids(pa, pid, ca, cid)
            self.removed += weight * num_removed
            self.steps += weight
            self.edit_acc.update([prev_surv, cur_reord], weight, edit_est)  # one transition's edge/node stats

    def seq_update(self, x: Sequence[Any], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted sufficient statistics from a batch."""
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.update(obs, float(w), estimate)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted sequence."""
        self.update(x, weight, None)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple) -> ChurningTemporalGraphGrammarAccumulator:
        """Merge serialized churning sufficient statistics."""
        e, r, s = suff_stat
        self.edit_acc.combine(e)
        self.removed += r
        self.steps += s
        return self

    def value(self) -> tuple:
        """Return serialized churning sufficient statistics."""
        return self.edit_acc.value(), self.removed, self.steps

    def from_value(self, x: tuple) -> ChurningTemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        self.edit_acc.from_value(x[0])
        self.removed, self.steps = float(x[1]), float(x[2])
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return TemporalGraphGrammarDataEncoder()


class ChurningTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for churning temporal graph grammar accumulators."""

    def __init__(self, edit_factory: Any) -> None:
        self.edit_factory = edit_factory

    def make(self) -> ChurningTemporalGraphGrammarAccumulator:
        """Create a fresh churning accumulator."""
        return ChurningTemporalGraphGrammarAccumulator(self.edit_factory.make())


class ChurningTemporalGraphGrammarEstimator(ParameterEstimator):
    """Estimator for identity-tracked churning temporal graph grammars."""

    def __init__(self, edit_estimator: Any, name: str | None = None) -> None:
        self.edit_estimator = edit_estimator
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> ChurningTemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return ChurningTemporalGraphGrammarAccumulatorFactory(self.edit_estimator.accumulator_factory())

    def estimate(self, nobs: float | None, suff_stat: tuple) -> ChurningTemporalGraphGrammarDistribution:
        """Estimate the wrapped edit grammar and node-removal rate."""
        edit_val, removed, steps = suff_stat
        return ChurningTemporalGraphGrammarDistribution(
            self.edit_estimator.estimate(nobs, edit_val),
            node_remove_rate=removed / steps if steps > 0 else 0.0,
            name=self.name,
        )


# --- latent-regime dynamics: an HMM over graph-edit grammars ---------------------------------------
def _grammar_forward_backward(log_b: np.ndarray, log_init: np.ndarray, log_trans: np.ndarray) -> tuple:
    """Standard log-space forward-backward over the regime chain. Returns (loglik, gamma(T,K), xi(T-1,K,K)).

    ``log_b[t, k]`` is the log-density of transition t under regime k. gamma/xi are None for a zero-probability
    sequence (some transition impossible under every regime)."""
    from scipy.special import logsumexp

    t_steps, k = log_b.shape
    if t_steps == 0:
        return 0.0, np.zeros((0, k)), np.zeros((0, k, k))
    la = np.empty((t_steps, k))
    la[0] = log_init + log_b[0]
    for t in range(1, t_steps):
        la[t] = log_b[t] + logsumexp(la[t - 1][:, None] + log_trans, axis=0)
    log_p = float(logsumexp(la[-1]))
    if not np.isfinite(log_p):
        return log_p, None, None
    lb = np.zeros((t_steps, k))
    for t in range(t_steps - 2, -1, -1):
        lb[t] = logsumexp(log_trans + (log_b[t + 1] + lb[t + 1])[None, :], axis=1)
    gamma = np.exp(la + lb - log_p)
    xi = np.zeros((max(t_steps - 1, 0), k, k))
    for t in range(t_steps - 1):
        xi[t] = np.exp(la[t][:, None] + log_trans + (log_b[t + 1] + lb[t + 1])[None, :] - log_p)
    return log_p, gamma, xi


class LatentTemporalGraphGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """A dynamic graph whose edit grammar is governed by a hidden, time-evolving REGIME.

    A latent state z_t (a Markov chain: ``initial_probs`` pi, ``transition_matrix`` A) selects which of K
    edit grammars governs transition t. So the graph can switch regimes over time -- e.g. a bursty growth /
    densification phase, then a fragmentation / decay phase -- dynamics a single grammar cannot express. The
    sequence likelihood marginalises the regime path by the forward algorithm; emissions are the per-
    transition edit log-densities of each regime's grammar, so this is an HMM whose emission models are the
    graph-edit grammars and EM reuses each grammar's weighted accumulator for the M-step.

    Observation = a plain list of adjacency snapshots (same as the base grammar) -- the regime is latent.
    ``decode`` returns the most likely regime active at each transition (Viterbi).
    """

    def __init__(
        self,
        states: Sequence[TemporalGraphGrammarDistribution],
        initial_probs: Sequence[float] | None = None,
        transition_matrix: Sequence[Sequence[float]] | None = None,
        name: str | None = None,
    ) -> None:
        self.states = list(states)
        self.k = len(self.states)
        ip = np.ones(self.k) / self.k if initial_probs is None else np.asarray(initial_probs, dtype=np.float64)
        self.initial_probs = ip / ip.sum()
        if transition_matrix is None:
            self.transition_matrix = np.ones((self.k, self.k)) / self.k
        else:
            tm = np.asarray(transition_matrix, dtype=np.float64)
            self.transition_matrix = tm / tm.sum(axis=1, keepdims=True)
        self.log_init = np.log(np.clip(self.initial_probs, _EPS, None))
        self.log_trans = np.log(np.clip(self.transition_matrix, _EPS, None))
        self.name = name

    def __str__(self) -> str:
        return "LatentTemporalGraphGrammarDistribution(K=%d, A=%s)" % (
            self.k,
            np.array2string(self.transition_matrix, precision=2),
        )

    def _shared_motif(self) -> bool:
        m0 = self.states[0].motif
        return all((s.motif.bins == m0.bins and s.motif.directed == m0.directed) for s in self.states)

    def _emission_logb(self, snaps: Sequence[Any]) -> np.ndarray:
        """(T, K) per-transition, per-regime log-densities (T = number of transitions).

        When the regimes share a motif (the common case) the expensive A@A decomposition of each transition
        is computed ONCE and scored across all K regimes -- O(T) heavy work instead of O(T*K)."""
        t_steps = len(snaps) - 1
        log_b = np.empty((t_steps, self.k))
        if self._shared_motif():
            for t in range(t_steps):
                comp = self.states[0].transition_components(snaps[t], snaps[t + 1])
                for k, st in enumerate(self.states):
                    log_b[t, k] = st.score_components(comp)
        else:
            for t in range(t_steps):
                for k, st in enumerate(self.states):
                    log_b[t, k] = st._transition_log_density(snaps[t], snaps[t + 1])
        return log_b

    def log_density(self, x: Sequence[Any]) -> float:
        """Score one dynamic graph sequence with regimes marginalized out."""
        snaps = list(x)
        if len(snaps) < 2:
            return 0.0
        return _grammar_forward_backward(self._emission_logb(snaps), self.log_init, self.log_trans)[0]

    def decode(self, x: Sequence[Any]) -> list:
        """Viterbi: the most likely regime governing each transition."""
        snaps = list(x)
        log_b = self._emission_logb(snaps)
        t_steps = log_b.shape[0]
        if t_steps == 0:
            return []
        v = np.empty((t_steps, self.k))
        ptr = np.zeros((t_steps, self.k), dtype=np.int64)
        v[0] = self.log_init + log_b[0]
        for t in range(1, t_steps):
            scores = v[t - 1][:, None] + self.log_trans
            ptr[t] = scores.argmax(axis=0)
            v[t] = log_b[t] + scores.max(axis=0)
        path = [int(v[-1].argmax())]
        for t in range(t_steps - 1, 0, -1):
            path.append(int(ptr[t][path[-1]]))
        return path[::-1]

    def seq_encode(self, x: Sequence[Any]) -> Sequence[Any]:
        """Return latent-regime graph observations unchanged for scoring."""
        return x

    def seq_log_density(self, x: Sequence[Any]) -> np.ndarray:
        """Score a batch of latent-regime graph observations."""
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> LatentTemporalGraphGrammarSampler:
        """Return a sampler for latent-regime graph sequences."""
        return LatentTemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> LatentTemporalGraphGrammarEstimator:
        """Return the Baum-Welch estimator for the latent-regime grammar."""
        return LatentTemporalGraphGrammarEstimator(
            [st.estimator(pseudo_count=pseudo_count) for st in self.states], pseudo_count=pseudo_count, name=self.name
        )

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the pass-through graph encoder."""
        return TemporalGraphGrammarDataEncoder()


class LatentTemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for regime-switching temporal graph grammars."""

    def __init__(self, dist: LatentTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.sub = [st.sampler(self.rng.randint(2**31)) for st in dist.states]

    def sample_one(self, num_steps: int = 10, seed_graph: np.ndarray | None = None, n_init: int = 5) -> list:
        """Draw one latent-regime graph sequence."""
        d = self.dist
        adj = np.zeros((n_init, n_init)) if seed_graph is None else np.asarray(seed_graph, dtype=np.float64).copy()
        snaps = [adj.copy()]
        z = int(self.rng.choice(d.k, p=d.initial_probs))
        for _ in range(num_steps):
            st = d.states[z]
            new_nodes = int(self.rng.poisson(st.node_rate))
            if new_nodes:
                n = adj.shape[0]
                big = np.zeros((n + new_nodes, n + new_nodes))
                big[:n, :n] = adj
                adj = big
            self.sub[z]._edge_edit_step(adj)  # active regime's edit grammar
            snaps.append(adj.copy())
            z = int(self.rng.choice(d.k, p=d.transition_matrix[z]))  # regime evolves
        return snaps

    def sample(self, size: int | None = None, **kw: Any) -> Any:
        """Draw one sequence or a list of sequences."""
        if size is None:
            return self.sample_one(**kw)
        return [self.sample_one(**kw) for _ in range(size)]


class LatentTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for latent-regime graph grammar EM sufficient statistics."""

    def __init__(self, k: int, state_accs: Sequence[Any]) -> None:
        self.k = k
        self.state_accs = list(state_accs)
        self.init_counts = np.zeros(k, dtype=np.float64)
        self.trans_counts = np.zeros((k, k), dtype=np.float64)

    def _accumulate(self, snaps: list, weight: float, gamma: np.ndarray, xi: np.ndarray, estimate: Any) -> None:
        self.init_counts += weight * gamma[0]
        if xi.shape[0]:
            self.trans_counts += weight * xi.sum(axis=0)
        for kk in range(self.k):
            est_k = None if estimate is None else estimate.states[kk]
            for t in range(len(snaps) - 1):
                w = weight * gamma[t, kk]
                if w > 0:
                    self.state_accs[kk].update([snaps[t], snaps[t + 1]], w, est_k)

    def update(self, x: Sequence[Any], weight: float, estimate: Any | None) -> None:
        """Accumulate posterior-weighted sufficient statistics for one sequence."""
        snaps = list(x)
        if len(snaps) < 2:
            return
        log_b = estimate._emission_logb(snaps)
        _, gamma, xi = _grammar_forward_backward(log_b, estimate.log_init, estimate.log_trans)
        if gamma is None:
            return
        self._accumulate(snaps, weight, gamma, xi, estimate)

    def initialize(self, x: Sequence[Any], weight: float, rng: RandomState | None) -> None:
        """Initialize latent-regime sufficient statistics with random soft assignments."""
        snaps = list(x)
        if len(snaps) < 2:
            return
        rng = rng if rng is not None else RandomState()
        t_steps = len(snaps) - 1
        gamma = rng.dirichlet(np.ones(self.k), size=t_steps)  # random soft regime assignment to seed EM
        xi = np.zeros((max(t_steps - 1, 0), self.k, self.k))
        for t in range(t_steps - 1):
            xi[t] = np.outer(gamma[t], gamma[t + 1])
        self._accumulate(snaps, weight, gamma, xi, None)

    def seq_update(self, x: Sequence[Any], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate posterior-weighted sufficient statistics from a batch."""
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.update(obs, float(w), estimate)

    def seq_initialize(self, x: Sequence[Any], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.initialize(obs, float(w), rng)

    def combine(self, suff_stat: tuple) -> LatentTemporalGraphGrammarAccumulator:
        """Merge serialized latent-regime sufficient statistics."""
        ic, tc, states = suff_stat
        self.init_counts += ic
        self.trans_counts += tc
        for acc, sv in zip(self.state_accs, states):
            acc.combine(sv)
        return self

    def value(self) -> tuple:
        """Return serialized latent-regime sufficient statistics."""
        return self.init_counts.copy(), self.trans_counts.copy(), [acc.value() for acc in self.state_accs]

    def from_value(self, x: tuple) -> LatentTemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        self.init_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.trans_counts = np.asarray(x[1], dtype=np.float64).copy()
        for acc, sv in zip(self.state_accs, x[2]):
            acc.from_value(sv)
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return TemporalGraphGrammarDataEncoder()


class LatentTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for latent temporal graph grammar accumulators."""

    def __init__(self, k: int, state_factories: Sequence[Any]) -> None:
        self.k = k
        self.state_factories = list(state_factories)

    def make(self) -> LatentTemporalGraphGrammarAccumulator:
        """Create a fresh latent-regime accumulator."""
        return LatentTemporalGraphGrammarAccumulator(self.k, [f.make() for f in self.state_factories])


class LatentTemporalGraphGrammarEstimator(ParameterEstimator):
    """EM (Baum-Welch) for the regime-switching grammar: forward-backward E-step, per-regime weighted M-step."""

    def __init__(
        self, state_estimators: Sequence[Any], pseudo_count: float | None = None, name: str | None = None
    ) -> None:
        self.state_estimators = list(state_estimators)
        self.k = len(self.state_estimators)
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> LatentTemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return LatentTemporalGraphGrammarAccumulatorFactory(
            self.k, [est.accumulator_factory() for est in self.state_estimators]
        )

    def estimate(self, nobs: float | None, suff_stat: tuple) -> LatentTemporalGraphGrammarDistribution:
        """Estimate regime priors, transitions, and state grammars from EM statistics."""
        init_counts, trans_counts, state_vals = suff_stat
        pc = 0.0 if self.pseudo_count is None else float(self.pseudo_count)
        ip = init_counts + pc
        ip = ip / ip.sum() if ip.sum() > 0 else np.ones(self.k) / self.k
        tm = trans_counts + pc
        row = tm.sum(axis=1, keepdims=True)
        tm = np.where(row > 0, tm / np.where(row > 0, row, 1.0), 1.0 / self.k)
        states = [est.estimate(nobs, sv) for est, sv in zip(self.state_estimators, state_vals)]
        return LatentTemporalGraphGrammarDistribution(states, ip, tm, name=self.name)


# --- regime-switching ATTRIBUTES: a latent regime over structure AND node/edge attributes -----------
class LatentAttributedTemporalGraphGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """A regime-switching dynamic graph where the hidden regime drives the STRUCTURE *and* the ATTRIBUTES.

    Each of K regimes carries a full edit grammar plus (optionally) a node-attribute distribution and an
    edge-attribute distribution, all switched by one latent Markov state z_t. So a single regime change can
    densify the topology AND spike communication volume / shift node properties together -- e.g. an "active"
    phase with bursty triadic closure and high message counts, vs a "quiet" phase. The per-transition
    emission under regime k is ``structure_k(transition) + node_attrs_k(nodes added this step) +
    edge_attrs_k(edges added this step)``; the sequence likelihood marginalises the regime path by the
    forward algorithm and EM does forward-backward + a per-regime weighted M-step over each piece.

    Observation = ``(snapshots, node_features, edge_features)`` where ``node_features[t]`` / ``edge_features[t]``
    are the attribute records of the nodes / edges that appear at transition t (lists, length = #transitions).
    """

    def __init__(
        self,
        structures: Sequence[TemporalGraphGrammarDistribution],
        node_dists: Sequence[Any] | None = None,
        edge_dists: Sequence[Any] | None = None,
        initial_probs: Sequence[float] | None = None,
        transition_matrix: Sequence[Sequence[float]] | None = None,
        name: str | None = None,
    ) -> None:
        self.structures = list(structures)
        self.k = len(self.structures)
        self.node_dists = None if node_dists is None else list(node_dists)
        self.edge_dists = None if edge_dists is None else list(edge_dists)
        ip = np.ones(self.k) / self.k if initial_probs is None else np.asarray(initial_probs, dtype=np.float64)
        self.initial_probs = ip / ip.sum()
        if transition_matrix is None:
            self.transition_matrix = np.ones((self.k, self.k)) / self.k
        else:
            tm = np.asarray(transition_matrix, dtype=np.float64)
            self.transition_matrix = tm / tm.sum(axis=1, keepdims=True)
        self.log_init = np.log(np.clip(self.initial_probs, _EPS, None))
        self.log_trans = np.log(np.clip(self.transition_matrix, _EPS, None))
        self.name = name

    def __str__(self) -> str:
        return "LatentAttributedTemporalGraphGrammarDistribution(K=%d, node=%s, edge=%s)" % (
            self.k,
            self.node_dists is not None,
            self.edge_dists is not None,
        )

    def _shared_motif(self) -> bool:
        m0 = self.structures[0].motif
        return all((s.motif.bins == m0.bins and s.motif.directed == m0.directed) for s in self.structures)

    def _emission_logb(self, x: tuple) -> np.ndarray:
        snaps, node_features, edge_features = x
        t_steps = len(snaps) - 1
        log_b = np.empty((t_steps, self.k))
        shared = self._shared_motif()
        for t in range(t_steps):
            comp = self.structures[0].transition_components(snaps[t], snaps[t + 1]) if shared else None
            nf = node_features[t] if node_features else []
            ef = edge_features[t] if edge_features else []
            for k in range(self.k):
                st = self.structures[k]
                ll = st.score_components(comp) if shared else st._transition_log_density(snaps[t], snaps[t + 1])
                if self.node_dists is not None:
                    ll += _emission_ll(self.node_dists[k], nf)
                if self.edge_dists is not None:
                    ll += _emission_ll(self.edge_dists[k], ef)
                log_b[t, k] = ll
        return log_b

    def log_density(self, x: tuple) -> float:
        """Score one attributed graph sequence with regimes marginalized out."""
        snaps = x[0]
        if len(snaps) < 2:
            return 0.0
        return _grammar_forward_backward(self._emission_logb(x), self.log_init, self.log_trans)[0]

    def decode(self, x: tuple) -> list:
        """Viterbi: the most likely regime governing each transition (jointly explaining structure+attrs)."""
        log_b = self._emission_logb(x)
        t_steps = log_b.shape[0]
        if t_steps == 0:
            return []
        v = np.empty((t_steps, self.k))
        ptr = np.zeros((t_steps, self.k), dtype=np.int64)
        v[0] = self.log_init + log_b[0]
        for t in range(1, t_steps):
            scores = v[t - 1][:, None] + self.log_trans
            ptr[t] = scores.argmax(axis=0)
            v[t] = log_b[t] + scores.max(axis=0)
        path = [int(v[-1].argmax())]
        for t in range(t_steps - 1, 0, -1):
            path.append(int(ptr[t][path[-1]]))
        return path[::-1]

    def seq_encode(self, x: Sequence[Any]) -> Sequence[Any]:
        """Return attributed latent-regime observations unchanged for scoring."""
        return x

    def seq_log_density(self, x: Sequence[Any]) -> np.ndarray:
        """Score a batch of attributed latent-regime observations."""
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> LatentAttributedTemporalGraphGrammarSampler:
        """Return a sampler for attributed latent-regime graph sequences."""
        return LatentAttributedTemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> LatentAttributedTemporalGraphGrammarEstimator:
        """Return the EM estimator for structure and attribute regime models."""
        return LatentAttributedTemporalGraphGrammarEstimator(
            [st.estimator(pseudo_count=pseudo_count) for st in self.structures],
            None if self.node_dists is None else [d.estimator() for d in self.node_dists],
            None if self.edge_dists is None else [d.estimator() for d in self.edge_dists],
            pseudo_count=pseudo_count,
            name=self.name,
        )

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the pass-through graph encoder."""
        return TemporalGraphGrammarDataEncoder()


class LatentAttributedTemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for regime-switching attributed temporal graph grammars."""

    def __init__(self, dist: LatentAttributedTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.sub = [st.sampler(self.rng.randint(2**31)) for st in dist.structures]

    def sample_one(self, num_steps: int = 10, seed_graph: np.ndarray | None = None, n_init: int = 5) -> tuple:
        """Draw one attributed latent-regime graph observation."""
        d = self.dist
        adj = np.zeros((n_init, n_init)) if seed_graph is None else np.asarray(seed_graph, dtype=np.float64).copy()
        snaps = [adj.copy()]
        node_features: list = []
        edge_features: list = []
        z = int(self.rng.choice(d.k, p=d.initial_probs))
        for _ in range(num_steps):
            st = d.structures[z]
            n_before = adj.shape[0]
            new_nodes = int(self.rng.poisson(st.node_rate))
            if new_nodes:
                big = np.zeros((n_before + new_nodes, n_before + new_nodes))
                big[:n_before, :n_before] = adj
                adj = big
            before = adj.copy()
            self.sub[z]._edge_edit_step(adj)
            num_added = len(_edge_diff(before, adj, st.directed)[0])
            nf = (
                list(d.node_dists[z].sampler(self.rng.randint(2**31)).sample(size=new_nodes))
                if d.node_dists is not None and new_nodes
                else []
            )
            ef = (
                list(d.edge_dists[z].sampler(self.rng.randint(2**31)).sample(size=num_added))
                if d.edge_dists is not None and num_added
                else []
            )
            node_features.append(nf)
            edge_features.append(ef)
            snaps.append(adj.copy())
            z = int(self.rng.choice(d.k, p=d.transition_matrix[z]))
        return snaps, node_features, edge_features

    def sample(self, size: int | None = None, **kw: Any) -> Any:
        """Draw one observation or a list of observations."""
        if size is None:
            return self.sample_one(**kw)
        return [self.sample_one(**kw) for _ in range(size)]


class LatentAttributedTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for attributed latent-regime graph grammar EM statistics."""

    def __init__(self, k: int, struct_accs: Sequence[Any], node_accs: Any, edge_accs: Any) -> None:
        self.k = k
        self.struct_accs = list(struct_accs)
        self.node_accs = node_accs
        self.edge_accs = edge_accs
        self.init_counts = np.zeros(k, dtype=np.float64)
        self.trans_counts = np.zeros((k, k), dtype=np.float64)

    def _accumulate(self, x: tuple, weight: float, gamma: np.ndarray, xi: np.ndarray, estimate: Any) -> None:
        snaps, node_features, edge_features = x
        self.init_counts += weight * gamma[0]
        if xi.shape[0]:
            self.trans_counts += weight * xi.sum(axis=0)
        for kk in range(self.k):
            s_est = None if estimate is None else estimate.structures[kk]
            for t in range(len(snaps) - 1):
                w = weight * gamma[t, kk]
                if w <= 0:
                    continue
                self.struct_accs[kk].update([snaps[t], snaps[t + 1]], w, s_est)
                if self.node_accs is not None and node_features and node_features[t]:
                    nd = None if estimate is None else estimate.node_dists[kk]
                    for r in node_features[t]:  # per-record update (raw values; works with or without an estimate)
                        self.node_accs[kk].update(r, w, nd)
                if self.edge_accs is not None and edge_features and edge_features[t]:
                    ed = None if estimate is None else estimate.edge_dists[kk]
                    for r in edge_features[t]:
                        self.edge_accs[kk].update(r, w, ed)

    def update(self, x: tuple, weight: float, estimate: Any | None) -> None:
        """Accumulate posterior-weighted statistics for one attributed sequence."""
        if len(x[0]) < 2:
            return
        _, gamma, xi = _grammar_forward_backward(estimate._emission_logb(x), estimate.log_init, estimate.log_trans)
        if gamma is None:
            return
        self._accumulate(x, weight, gamma, xi, estimate)

    def initialize(self, x: tuple, weight: float, rng: RandomState | None) -> None:
        """Initialize attributed latent-regime statistics with random soft assignments."""
        if len(x[0]) < 2:
            return
        rng = rng if rng is not None else RandomState()
        t_steps = len(x[0]) - 1
        gamma = rng.dirichlet(np.ones(self.k), size=t_steps)
        xi = np.zeros((max(t_steps - 1, 0), self.k, self.k))
        for t in range(t_steps - 1):
            xi[t] = np.outer(gamma[t], gamma[t + 1])
        self._accumulate(x, weight, gamma, xi, None)

    def seq_update(self, x: Sequence[Any], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate posterior-weighted statistics from a batch."""
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.update(obs, float(w), estimate)

    def seq_initialize(self, x: Sequence[Any], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.initialize(obs, float(w), rng)

    def combine(self, suff_stat: tuple) -> LatentAttributedTemporalGraphGrammarAccumulator:
        """Merge serialized attributed latent-regime sufficient statistics."""
        ic, tc, sv, nv, ev = suff_stat
        self.init_counts += ic
        self.trans_counts += tc
        for acc, v in zip(self.struct_accs, sv):
            acc.combine(v)
        if self.node_accs is not None:
            for acc, v in zip(self.node_accs, nv):
                acc.combine(v)
        if self.edge_accs is not None:
            for acc, v in zip(self.edge_accs, ev):
                acc.combine(v)
        return self

    def value(self) -> tuple:
        """Return serialized attributed latent-regime sufficient statistics."""
        return (
            self.init_counts.copy(),
            self.trans_counts.copy(),
            [acc.value() for acc in self.struct_accs],
            None if self.node_accs is None else [acc.value() for acc in self.node_accs],
            None if self.edge_accs is None else [acc.value() for acc in self.edge_accs],
        )

    def from_value(self, x: tuple) -> LatentAttributedTemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        self.init_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.trans_counts = np.asarray(x[1], dtype=np.float64).copy()
        for acc, v in zip(self.struct_accs, x[2]):
            acc.from_value(v)
        if self.node_accs is not None:
            for acc, v in zip(self.node_accs, x[3]):
                acc.from_value(v)
        if self.edge_accs is not None:
            for acc, v in zip(self.edge_accs, x[4]):
                acc.from_value(v)
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return TemporalGraphGrammarDataEncoder()


class LatentAttributedTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for attributed latent-regime graph grammar accumulators."""

    def __init__(self, k: int, struct_factories: Sequence[Any], node_factories: Any, edge_factories: Any) -> None:
        self.k = k
        self.struct_factories = list(struct_factories)
        self.node_factories = node_factories
        self.edge_factories = edge_factories

    def make(self) -> LatentAttributedTemporalGraphGrammarAccumulator:
        """Create a fresh attributed latent-regime accumulator."""
        return LatentAttributedTemporalGraphGrammarAccumulator(
            self.k,
            [f.make() for f in self.struct_factories],
            None if self.node_factories is None else [f.make() for f in self.node_factories],
            None if self.edge_factories is None else [f.make() for f in self.edge_factories],
        )


class LatentAttributedTemporalGraphGrammarEstimator(ParameterEstimator):
    """EM for the regime-switching attributed grammar: forward-backward E-step, per-regime weighted M-step
    over structure + node attrs + edge attrs together."""

    def __init__(
        self,
        structure_estimators: Sequence[Any],
        node_estimators: Any = None,
        edge_estimators: Any = None,
        pseudo_count: float | None = None,
        name: str | None = None,
    ) -> None:
        self.structure_estimators = list(structure_estimators)
        self.k = len(self.structure_estimators)
        self.node_estimators = node_estimators
        self.edge_estimators = edge_estimators
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> LatentAttributedTemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return LatentAttributedTemporalGraphGrammarAccumulatorFactory(
            self.k,
            [est.accumulator_factory() for est in self.structure_estimators],
            None if self.node_estimators is None else [est.accumulator_factory() for est in self.node_estimators],
            None if self.edge_estimators is None else [est.accumulator_factory() for est in self.edge_estimators],
        )

    def estimate(self, nobs: float | None, suff_stat: tuple) -> LatentAttributedTemporalGraphGrammarDistribution:
        """Estimate regime priors, transitions, structures, and attribute models."""
        init_counts, trans_counts, sv, nv, ev = suff_stat
        pc = 0.0 if self.pseudo_count is None else float(self.pseudo_count)
        ip = init_counts + pc
        ip = ip / ip.sum() if ip.sum() > 0 else np.ones(self.k) / self.k
        tm = trans_counts + pc
        row = tm.sum(axis=1, keepdims=True)
        tm = np.where(row > 0, tm / np.where(row > 0, row, 1.0), 1.0 / self.k)
        structures = [est.estimate(nobs, v) for est, v in zip(self.structure_estimators, sv)]
        node_dists = (
            None
            if self.node_estimators is None
            else [est.estimate(nobs, v) for est, v in zip(self.node_estimators, nv)]
        )
        edge_dists = (
            None
            if self.edge_estimators is None
            else [est.estimate(nobs, v) for est, v in zip(self.edge_estimators, ev)]
        )
        return LatentAttributedTemporalGraphGrammarDistribution(
            structures, node_dists, edge_dists, ip, tm, name=self.name
        )


# --- latent regime + node churn: regimes that also switch the turnover rate ------------------------
class LatentChurningTemporalGraphGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """A regime-switching dynamic graph where the hidden regime also governs NODE TURNOVER.

    Combines the latent regime HMM with identity-tracked node churn: each of K regimes carries a full edit
    grammar AND its own node-removal rate, so the graph can switch between e.g. a stable phase (slow
    turnover, triadic growth) and a churn phase (fast member departure, fragmentation). Each snapshot is
    ``(adjacency, node_ids)``; per transition the active regime first removes nodes (those whose id
    disappears) then edits the surviving subgraph, and the per-transition emission under regime k is
    ``node_removal_k(#removed) + grammar_k(edit on the aligned surviving subgraph)``. The sequence likelihood
    marginalises the regime path by the forward algorithm; EM does forward-backward then a per-regime
    weighted M-step over both the grammar and the turnover rate. ``decode`` recovers the active regime.

    Observation = ``(snapshots, node_ids)`` where ``node_ids`` is a list of per-snapshot id arrays. Dense.
    """

    def __init__(
        self,
        states: Sequence[TemporalGraphGrammarDistribution],
        node_remove_rates: Sequence[float] | None = None,
        initial_probs: Sequence[float] | None = None,
        transition_matrix: Sequence[Sequence[float]] | None = None,
        name: str | None = None,
    ) -> None:
        self.states = list(states)
        self.k = len(self.states)
        self.node_remove_rates = (
            np.zeros(self.k) if node_remove_rates is None else np.asarray(node_remove_rates, dtype=np.float64)
        )
        ip = np.ones(self.k) / self.k if initial_probs is None else np.asarray(initial_probs, dtype=np.float64)
        self.initial_probs = ip / ip.sum()
        if transition_matrix is None:
            self.transition_matrix = np.ones((self.k, self.k)) / self.k
        else:
            tm = np.asarray(transition_matrix, dtype=np.float64)
            self.transition_matrix = tm / tm.sum(axis=1, keepdims=True)
        self.log_init = np.log(np.clip(self.initial_probs, _EPS, None))
        self.log_trans = np.log(np.clip(self.transition_matrix, _EPS, None))
        self.name = name

    def __str__(self) -> str:
        return "LatentChurningTemporalGraphGrammarDistribution(K=%d, remove_rates=%s)" % (
            self.k,
            np.array2string(self.node_remove_rates, precision=2),
        )

    def _shared_motif(self) -> bool:
        m0 = self.states[0].motif
        return all((s.motif.bins == m0.bins and s.motif.directed == m0.directed) for s in self.states)

    def _aligned(self, x: Sequence[tuple]) -> list:
        snaps = list(x)  # list of (adjacency, node_ids) tuples
        return [
            (*_align_by_ids(snaps[t][0], snaps[t][1], snaps[t + 1][0], snaps[t + 1][1]), len(snaps[t][1]))
            for t in range(len(snaps) - 1)
        ]

    def _emission_logb(self, x: tuple, aligned: list | None = None) -> np.ndarray:
        aligned = aligned if aligned is not None else self._aligned(x)
        log_b = np.empty((len(aligned), self.k))
        shared = self._shared_motif()
        for t, (prev_surv, cur_reord, num_removed, n_prev) in enumerate(aligned):
            comp = self.states[0].transition_components(prev_surv, cur_reord) if shared else None
            for k in range(self.k):
                st = self.states[k]
                struct = st.score_components(comp) if shared else st._transition_log_density(prev_surv, cur_reord)
                log_b[t, k] = _node_removal_logp(self.node_remove_rates[k], n_prev, num_removed) + struct
        return log_b

    def log_density(self, x: tuple) -> float:
        """Score one identity-tracked sequence with regimes marginalized out."""
        if len(x) < 2:
            return 0.0
        return _grammar_forward_backward(self._emission_logb(x), self.log_init, self.log_trans)[0]

    def decode(self, x: tuple) -> list:
        """Return the most likely churn/edit regime for each transition."""
        log_b = self._emission_logb(x)
        t_steps = log_b.shape[0]
        if t_steps == 0:
            return []
        v = np.empty((t_steps, self.k))
        ptr = np.zeros((t_steps, self.k), dtype=np.int64)
        v[0] = self.log_init + log_b[0]
        for t in range(1, t_steps):
            scores = v[t - 1][:, None] + self.log_trans
            ptr[t] = scores.argmax(axis=0)
            v[t] = log_b[t] + scores.max(axis=0)
        path = [int(v[-1].argmax())]
        for t in range(t_steps - 1, 0, -1):
            path.append(int(ptr[t][path[-1]]))
        return path[::-1]

    def seq_encode(self, x: Sequence[Any]) -> Sequence[Any]:
        """Return latent churning observations unchanged for scoring."""
        return x

    def seq_log_density(self, x: Sequence[Any]) -> np.ndarray:
        """Score a batch of latent churning graph observations."""
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> LatentChurningTemporalGraphGrammarSampler:
        """Return a sampler for latent churning graph sequences."""
        return LatentChurningTemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> LatentChurningTemporalGraphGrammarEstimator:
        """Return the EM estimator for regime-specific edit grammars and churn rates."""
        return LatentChurningTemporalGraphGrammarEstimator(
            [st.estimator(pseudo_count=pseudo_count) for st in self.states], pseudo_count=pseudo_count, name=self.name
        )

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the pass-through graph encoder."""
        return TemporalGraphGrammarDataEncoder()


class LatentChurningTemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for regime-switching churning temporal graph grammars."""

    def __init__(self, dist: LatentChurningTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.sub = [st.sampler(self.rng.randint(2**31)) for st in dist.states]

    def sample_one(self, num_steps: int = 10, seed_graph: np.ndarray | None = None, n_init: int = 8) -> tuple:
        """Draw one latent churning graph sequence."""
        d = self.dist
        adj = np.zeros((n_init, n_init)) if seed_graph is None else np.asarray(seed_graph, dtype=np.float64).copy()
        ids = list(range(adj.shape[0]))
        next_id = adj.shape[0]
        snaps = [(adj.copy(), list(ids))]
        z = int(self.rng.choice(d.k, p=d.initial_probs))
        for _ in range(num_steps):
            n = adj.shape[0]
            k_rem = min(int(self.rng.poisson(d.node_remove_rates[z])), n)
            if k_rem:
                drop = set(self.rng.choice(n, size=k_rem, replace=False).tolist())
                keep = [i for i in range(n) if i not in drop]
                adj = adj[np.ix_(keep, keep)] if keep else np.zeros((0, 0))
                ids = [ids[i] for i in keep]
            new_nodes = int(self.rng.poisson(d.states[z].node_rate))
            if new_nodes:
                m = adj.shape[0]
                big = np.zeros((m + new_nodes, m + new_nodes))
                big[:m, :m] = adj
                adj = big
                ids += list(range(next_id, next_id + new_nodes))
                next_id += new_nodes
            if adj.shape[0]:
                self.sub[z]._edge_edit_step(adj)
            snaps.append((adj.copy(), list(ids)))
            z = int(self.rng.choice(d.k, p=d.transition_matrix[z]))
        return snaps

    def sample(self, size: int | None = None, **kw: Any) -> Any:
        """Draw one sequence or a list of sequences."""
        if size is None:
            return self.sample_one(**kw)
        return [self.sample_one(**kw) for _ in range(size)]


class LatentChurningTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for latent churning graph grammar EM sufficient statistics."""

    def __init__(self, k: int, state_accs: Sequence[Any]) -> None:
        self.k = k
        self.state_accs = list(state_accs)
        self.init_counts = np.zeros(k, dtype=np.float64)
        self.trans_counts = np.zeros((k, k), dtype=np.float64)
        self.removed = np.zeros(k, dtype=np.float64)
        self.steps = np.zeros(k, dtype=np.float64)

    def _accumulate(self, aligned: list, weight: float, gamma: np.ndarray, xi: np.ndarray, estimate: Any) -> None:
        self.init_counts += weight * gamma[0]
        if xi.shape[0]:
            self.trans_counts += weight * xi.sum(axis=0)
        for kk in range(self.k):
            s_est = None if estimate is None else estimate.states[kk]
            for t, (prev_surv, cur_reord, num_removed, _n_prev) in enumerate(aligned):
                w = weight * gamma[t, kk]
                if w <= 0:
                    continue
                self.state_accs[kk].update([prev_surv, cur_reord], w, s_est)
                self.removed[kk] += w * num_removed
                self.steps[kk] += w

    def update(self, x: tuple, weight: float, estimate: Any | None) -> None:
        """Accumulate posterior-weighted statistics for one churning sequence."""
        if len(x) < 2:
            return
        aligned = estimate._aligned(x)
        _, gamma, xi = _grammar_forward_backward(
            estimate._emission_logb(x, aligned), estimate.log_init, estimate.log_trans
        )
        if gamma is None:
            return
        self._accumulate(aligned, weight, gamma, xi, estimate)

    def initialize(self, x: tuple, weight: float, rng: RandomState | None) -> None:
        """Initialize latent churning statistics with random soft assignments."""
        if len(x) < 2:
            return
        rng = rng if rng is not None else RandomState()
        snaps = list(x)  # list of (adjacency, node_ids) tuples
        aligned = [
            (*_align_by_ids(snaps[t][0], snaps[t][1], snaps[t + 1][0], snaps[t + 1][1]), len(snaps[t][1]))
            for t in range(len(snaps) - 1)
        ]
        t_steps = len(aligned)
        gamma = rng.dirichlet(np.ones(self.k), size=t_steps)
        xi = np.zeros((max(t_steps - 1, 0), self.k, self.k))
        for t in range(t_steps - 1):
            xi[t] = np.outer(gamma[t], gamma[t + 1])
        self._accumulate(aligned, weight, gamma, xi, None)

    def seq_update(self, x: Sequence[Any], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate posterior-weighted statistics from a batch."""
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.update(obs, float(w), estimate)

    def seq_initialize(self, x: Sequence[Any], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.initialize(obs, float(w), rng)

    def combine(self, suff_stat: tuple) -> LatentChurningTemporalGraphGrammarAccumulator:
        """Merge serialized latent churning sufficient statistics."""
        ic, tc, rem, st, states = suff_stat
        self.init_counts += ic
        self.trans_counts += tc
        self.removed += rem
        self.steps += st
        for acc, sv in zip(self.state_accs, states):
            acc.combine(sv)
        return self

    def value(self) -> tuple:
        """Return serialized latent churning sufficient statistics."""
        return (
            self.init_counts.copy(),
            self.trans_counts.copy(),
            self.removed.copy(),
            self.steps.copy(),
            [acc.value() for acc in self.state_accs],
        )

    def from_value(self, x: tuple) -> LatentChurningTemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        self.init_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.trans_counts = np.asarray(x[1], dtype=np.float64).copy()
        self.removed = np.asarray(x[2], dtype=np.float64).copy()
        self.steps = np.asarray(x[3], dtype=np.float64).copy()
        for acc, sv in zip(self.state_accs, x[4]):
            acc.from_value(sv)
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return TemporalGraphGrammarDataEncoder()


class LatentChurningTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for latent churning temporal graph grammar accumulators."""

    def __init__(self, k: int, state_factories: Sequence[Any]) -> None:
        self.k = k
        self.state_factories = list(state_factories)

    def make(self) -> LatentChurningTemporalGraphGrammarAccumulator:
        """Create a fresh latent churning accumulator."""
        return LatentChurningTemporalGraphGrammarAccumulator(self.k, [f.make() for f in self.state_factories])


class LatentChurningTemporalGraphGrammarEstimator(ParameterEstimator):
    """Estimator for regime-switching churning temporal graph grammars."""

    def __init__(
        self, state_estimators: Sequence[Any], pseudo_count: float | None = None, name: str | None = None
    ) -> None:
        self.state_estimators = list(state_estimators)
        self.k = len(self.state_estimators)
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> LatentChurningTemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return LatentChurningTemporalGraphGrammarAccumulatorFactory(
            self.k, [est.accumulator_factory() for est in self.state_estimators]
        )

    def estimate(self, nobs: float | None, suff_stat: tuple) -> LatentChurningTemporalGraphGrammarDistribution:
        """Estimate regime priors, transitions, grammars, and churn rates."""
        init_counts, trans_counts, removed, steps, state_vals = suff_stat
        pc = 0.0 if self.pseudo_count is None else float(self.pseudo_count)
        ip = init_counts + pc
        ip = ip / ip.sum() if ip.sum() > 0 else np.ones(self.k) / self.k
        tm = trans_counts + pc
        row = tm.sum(axis=1, keepdims=True)
        tm = np.where(row > 0, tm / np.where(row > 0, row, 1.0), 1.0 / self.k)
        states = [est.estimate(nobs, sv) for est, sv in zip(self.state_estimators, state_vals)]
        rates = np.where(steps > 0, removed / np.where(steps > 0, steps, 1.0), 0.0)
        return LatentChurningTemporalGraphGrammarDistribution(states, rates, ip, tm, name=self.name)


# --- moment-based regime initialisation (identifiability-grounded EM seeding) -----------------------
def _regime_signatures(proto: Any, obs: Any) -> np.ndarray:
    """Per-transition signature (T, F): the OBSERVED edit derivation summary that identifies a regime.

    Because the motif partition is mutually exclusive the derivation is observed, so each transition exposes
    its own sufficient statistics -- per-motif add/remove counts, node growth, and (if attributed) attribute
    means. Regimes are, by the identifiability argument, separated in this signature space, so clustering the
    signatures seeds EM near the true solution instead of at random."""
    regimes = getattr(proto, "states", None) or proto.structures
    attributed = hasattr(proto, "node_dists") and proto.node_dists is not None
    edge_attr = hasattr(proto, "edge_dists") and proto.edge_dists is not None
    snaps = obs[0] if (attributed or edge_attr) else obs
    m = regimes[0].motif.num_motifs
    nf = obs[1] if attributed else None
    ef = obs[2] if edge_attr else None
    out = []
    for t in range(len(snaps) - 1):
        nn, add_bins, _ac, rem_bins, _rc, valid = regimes[0].transition_components(snaps[t], snaps[t + 1])
        a = np.bincount(add_bins, minlength=m).astype(float) if valid and len(add_bins) else np.zeros(m)
        r = np.bincount(rem_bins, minlength=m).astype(float) if valid and len(rem_bins) else np.zeros(m)
        feat = [*a.tolist(), *r.tolist(), float(nn if valid else 0)]
        if attributed:
            feat.append(_records_mean(nf[t]) if nf and t < len(nf) else 0.0)
        if edge_attr:
            feat.append(_records_mean(ef[t]) if ef and t < len(ef) else 0.0)
        out.append(feat)
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, 2 * m + 1))


def _records_mean(records: Sequence[Any]) -> float:
    vals = []
    for r in records:
        try:
            vals.append(float(r))
        except (TypeError, ValueError):
            try:
                vals.append(float(r[0]))
            except (TypeError, ValueError, IndexError):
                pass
    return float(np.mean(vals)) if vals else 0.0


def _kmeans_labels(x: np.ndarray, k: int, rng: RandomState, iters: int = 25) -> np.ndarray:
    if x.shape[0] <= k:
        return np.arange(x.shape[0]) % k
    centers = x[rng.choice(x.shape[0], size=k, replace=False)]
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for c in range(k):
            members = x[labels == c]
            if members.shape[0]:
                centers[c] = members.mean(axis=0)
    return labels


def regime_moment_init(estimator: Any, proto: Any, data: Sequence[Any], k: int, seed: int | None = None) -> Any:
    """Seed a regime-switching grammar EM by clustering observed per-transition edit signatures.

    Returns an initial distribution whose regimes are the k-means clusters of the (identifiable) transition
    signatures. Because the derivation is observed, these signatures are sufficient statistics that separate
    the regimes, so this avoids the local optima of random-restart EM. ``proto`` is any distribution of the
    target class (used only for its motif/attribute structure); ``estimator`` produces the fitted result."""
    rng = RandomState(seed)
    sigs, spans = [], []
    for obs in data:
        s = _regime_signatures(proto, obs)
        sigs.append(s)
        spans.append(s.shape[0])
    x = np.vstack([s for s in sigs if s.shape[0]]) if any(spans) else np.zeros((0, 1))
    xs = (x - x.mean(axis=0)) / (x.std(axis=0) + 1.0e-9) if x.shape[0] else x
    labels = _kmeans_labels(xs, k, rng)
    acc = estimator.accumulator_factory().make()
    off = 0
    for obs, span in zip(data, spans):
        if span == 0:
            continue
        lab = labels[off : off + span]
        off += span
        gamma = np.eye(k)[lab]  # hard one-hot responsibilities from the clustering
        xi = np.zeros((max(span - 1, 0), k, k))
        for t in range(span - 1):
            xi[t] = np.outer(gamma[t], gamma[t + 1])
        acc._accumulate(obs, 1.0, gamma, xi, None)  # Latent/Attributed _accumulate take the observation directly
    return estimator.estimate(len(data), acc.value())
