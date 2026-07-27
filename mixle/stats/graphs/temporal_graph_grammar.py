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
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp
from numpy.random import RandomState
from scipy.stats import poisson

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_EPS = 1.0e-12


def _binarize(adj: Any, directed: bool = False) -> Any:
    """Validate and own a binary adjacency without dense/sparse semantic drift."""
    if sp.issparse(adj):
        a = sp.csr_array(adj, dtype=np.float64).copy()
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("temporal graph adjacency must be square.")
        a.sum_duplicates()
        a.eliminate_zeros()
        if np.any(~np.isfinite(a.data)) or np.any(a.data != 1.0):
            raise ValueError("temporal graph adjacency must contain exact binary values 0/1.")
        if np.any(a.diagonal() != 0.0):
            raise ValueError("temporal graph adjacency cannot contain self-loops.")
        if not directed and (a != a.T).nnz:
            raise ValueError("undirected temporal graph adjacency must be symmetric.")
        return a
    a = np.array(adj, dtype=np.float64, copy=True)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("temporal graph adjacency must be square.")
    if np.any(~np.isfinite(a)) or np.any((a != 0.0) & (a != 1.0)):
        raise ValueError("temporal graph adjacency must contain finite exact binary values 0/1.")
    if np.any(np.diag(a) != 0.0):
        raise ValueError("temporal graph adjacency cannot contain self-loops.")
    if not directed and not np.array_equal(a, a.T):
        raise ValueError("undirected temporal graph adjacency must be symmetric.")
    return a


def _pad(adj: Any, n: int) -> Any:
    """Grow a (n0,n0) adjacency to (n,n) by appending isolated nodes (sparse or dense)."""
    n0 = adj.shape[0]
    if n < n0:
        raise ValueError("cannot pad an adjacency to a smaller node count.")
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
    pp = _pad(_binarize(prev, directed=directed), n1)
    cc = _binarize(cur, directed=directed)
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
        checked = []
        for value in bins:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("motif bins must be exact non-Boolean integers.")
            checked.append(int(value))
        if not checked or checked[0] != 0 or any(right <= left for left, right in zip(checked, checked[1:])):
            raise ValueError("motif bins must be nonempty, begin at zero, and be strictly increasing.")
        if not isinstance(directed, (bool, np.bool_)):
            raise ValueError("directed must be Boolean.")
        self.bins = tuple(checked)
        self.directed = bool(directed)
        self.names = tuple(
            f"cn>={self.bins[-1]}" if i == len(self.bins) - 1 else f"cn={b}"
            for i, b in enumerate(self.bins)
        )

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
        adj = _binarize(adj, directed=self.directed)
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
def _finite_nonnegative(value: Any, *, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked) or checked < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return checked


def _exact_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an exact non-Boolean integer.")
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be non-negative.")
    return checked


def _exact_positive_int(value: Any, *, name: str) -> int:
    checked = _exact_nonnegative_int(value, name=name)
    if checked < 1:
        raise ValueError(f"{name} must be positive.")
    return checked


def _capped_poisson_subset_log_prob(selected: int, candidates: int, rate: float) -> float:
    """Log probability of one uniformly selected subset under min(Poisson(rate), candidates)."""
    if selected < 0 or candidates < 0 or selected > candidates:
        return float("-inf")
    if candidates == 0:
        return 0.0 if selected == 0 else float("-inf")
    if rate == 0.0:
        return 0.0 if selected == 0 else float("-inf")
    if selected < candidates:
        log_count = (
            math.lgamma(candidates + 1)
            - math.lgamma(selected + 1)
            - math.lgamma(candidates - selected + 1)
        )
        return -rate + selected * math.log(rate) - math.lgamma(selected + 1) - log_count
    return float(poisson.logsf(candidates - 1, rate))


@dataclass(frozen=True)
class TemporalGraphApproximationReceipt:
    """Auditable shortfall record for sparse rejection sampling."""

    exact: bool
    bridge_requested: int
    bridge_realized: int
    bridge_shortfall: int
    max_reject: int


@dataclass(frozen=True)
class ApproximateTemporalGraphSample:
    """A separately typed sparse approximation; unwrap snapshots explicitly."""

    snapshots: tuple[Any, ...]
    receipt: TemporalGraphApproximationReceipt

    def __iter__(self):
        return iter(self.snapshots)

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, item):
        return self.snapshots[item]


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
        if not isinstance(directed, (bool, np.bool_)):
            raise ValueError("directed must be Boolean.")
        if motif is not None and not isinstance(motif, CommonNeighbourMotif):
            raise ValueError("motif must be a CommonNeighbourMotif or None.")
        if motif is not None and motif.directed != bool(directed):
            raise ValueError("motif.directed must agree with directed.")
        source_motif = motif if motif is not None else CommonNeighbourMotif(directed=directed)
        self.motif = CommonNeighbourMotif(source_motif.bins, directed=source_motif.directed)
        self.directed = self.motif.directed
        m = self.motif.num_motifs

        def _norm(w: Sequence[float] | None) -> np.ndarray:
            a = np.ones(m, dtype=np.float64) if w is None else np.array(w, dtype=np.float64, copy=True)
            if a.ndim != 1 or a.shape[0] != m:
                raise ValueError("motif weights must have one entry per motif bin (%d)." % m)
            if np.any(~np.isfinite(a)) or np.any(a < 0.0) or float(a.sum()) <= 0.0:
                raise ValueError("motif weights must be finite, non-negative, and have positive total.")
            normalized = a / a.sum()
            normalized.setflags(write=False)
            return normalized

        self.motif_weights = _norm(motif_weights)  # ADDITION grammar (which motifs grow)
        self.remove_weights = _norm(remove_weights)  # REMOVAL grammar (which motifs decay)
        self.log_w = np.full(m, -math.inf, dtype=np.float64)
        self.log_rw = np.full(m, -math.inf, dtype=np.float64)
        self.log_w[self.motif_weights > 0.0] = np.log(self.motif_weights[self.motif_weights > 0.0])
        self.log_rw[self.remove_weights > 0.0] = np.log(self.remove_weights[self.remove_weights > 0.0])
        self.log_w.setflags(write=False)
        self.log_rw.setflags(write=False)
        self.edge_rate = _finite_nonnegative(edge_rate, name="edge_rate")
        self.edge_remove_rate = _finite_nonnegative(edge_remove_rate, name="edge_remove_rate")
        self.node_rate = _finite_nonnegative(node_rate, name="node_rate")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
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

    def _edit_log_density(
        self,
        edit_bins: np.ndarray,
        weights: np.ndarray,
        rate: float,
        cand: np.ndarray,
    ) -> float:
        """Score the exact capped-Poisson, without-replacement subset law."""
        bins = np.asarray(edit_bins)
        candidates = np.asarray(cand, dtype=np.float64)
        if bins.ndim != 1 or candidates.shape != (self.motif.num_motifs,):
            raise ValueError("temporal edit components have incompatible motif shapes.")
        if bins.size and (
            not np.issubdtype(bins.dtype, np.integer)
            or int(bins.min()) < 0
            or int(bins.max()) >= self.motif.num_motifs
        ):
            raise ValueError("temporal edit bins must be in motif support.")
        if np.any(~np.isfinite(candidates)) or np.any(candidates < 0.0) or np.any(candidates != np.floor(candidates)):
            raise ValueError("temporal motif candidate counts must be finite non-negative integers.")
        selected = np.bincount(bins.astype(np.int64), minlength=self.motif.num_motifs)
        return float(
            sum(
                _capped_poisson_subset_log_prob(int(selected[m]), int(candidates[m]), rate * float(weights[m]))
                for m in range(self.motif.num_motifs)
            )
        )

    def transition_components(self, prev: Any, cur: Any) -> tuple:
        """The PARAMETER-INDEPENDENT decomposition of a transition: ``(new_nodes, add_bins, add_cand,
        rem_bins, rem_cand, valid)``. Depends only on the graph pair and the motif, NOT on the grammar's
        weights/rates -- so K regimes sharing a motif can compute the (expensive A@A) decomposition ONCE and
        score it K times via :meth:`score_components`. ``valid`` is False for an impossible node removal."""
        prev = _binarize(prev, directed=self.directed)
        cur = _binarize(cur, directed=self.directed)
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
        if isinstance(new_nodes, (bool, np.bool_)) or int(new_nodes) != new_nodes or new_nodes < 0:
            raise ValueError("new_nodes must be a non-negative integer.")
        if self.node_rate == 0.0:
            lp = 0.0 if new_nodes == 0 else float("-inf")
        else:
            lp = new_nodes * math.log(self.node_rate) - self.node_rate - math.lgamma(new_nodes + 1)
        lp += self._edit_log_density(add_bins, self.motif_weights, self.edge_rate, add_cand)
        if lp == float("-inf"):
            return lp
        lp += self._edit_log_density(rem_bins, self.remove_weights, self.edge_remove_rate, rem_cand)
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
        if isinstance(x, ApproximateTemporalGraphSample):
            raise ValueError("approximate scalable samples must be explicitly unwrapped before scoring.")
        snaps = list(x)
        if not snaps:
            raise ValueError("temporal graph observations must contain at least one snapshot.")
        snaps = [_binarize(snapshot, directed=self.directed) for snapshot in snaps]
        if len(snaps) < 2:
            return 0.0
        return float(sum(self._transition_log_density(snaps[t - 1], snaps[t]) for t in range(1, len(snaps))))

    def seq_encode(self, x: Sequence[Sequence[np.ndarray]]) -> Sequence[Sequence[np.ndarray]]:
        """Return dynamic graph sequences unchanged for sequence scoring."""
        return tuple(tuple(_binarize(snapshot, directed=self.directed) for snapshot in seq) for seq in x)

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
        return TemporalGraphGrammarDataEncoder(self.directed)


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
        num_steps = _exact_nonnegative_int(num_steps, name="num_steps")
        n_init = _exact_nonnegative_int(n_init, name="n_init")
        if seed_graph is None:
            adj = np.zeros((n_init, n_init), dtype=np.float64)
        elif sp.issparse(seed_graph):
            adj = _binarize(seed_graph, directed=d.directed).toarray()
        else:
            adj = _binarize(seed_graph, directed=d.directed)
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
    ) -> ApproximateTemporalGraphSample:
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
        num_steps = _exact_nonnegative_int(num_steps, name="num_steps")
        n_init = _exact_nonnegative_int(n_init, name="n_init")
        max_reject = _exact_positive_int(max_reject, name="max_reject")
        directed = d.directed

        def canon(i: int, j: int) -> tuple:
            return (i, j) if directed else ((i, j) if i < j else (j, i))

        edges = set()
        if seed_edges is not None:
            for edge in seed_edges:
                if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                    raise ValueError("seed_edges entries must be node-index pairs.")
                i = _exact_nonnegative_int(edge[0], name="seed edge endpoint")
                j = _exact_nonnegative_int(edge[1], name="seed edge endpoint")
                if i == j:
                    raise ValueError("seed_edges cannot contain self-loops.")
                edges.add(canon(i, j))
        n = max(n_init, max((max(edge) + 1 for edge in edges), default=0))
        snaps = [self._csr(edges, n, directed)]
        requested_bridges = 0
        realized_bridges = 0
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
                total_pairs = n * (n - 1) if directed else n * (n - 1) // 2
                bridge_candidates = max(0, total_pairs - len(edges) - int(wedge_keys.size))
                target_bridges = min(k0, bridge_candidates)
                requested_bridges += target_bridges
                chosen: set = set()
                for _try in range(target_bridges * max_reject):
                    if len(chosen) >= target_bridges:
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
                realized_bridges += len(chosen)
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
        receipt = TemporalGraphApproximationReceipt(
            exact=False,
            bridge_requested=requested_bridges,
            bridge_realized=realized_bridges,
            bridge_shortfall=requested_bridges - realized_bridges,
            max_reject=max_reject,
        )
        return ApproximateTemporalGraphSample(tuple(snaps), receipt)

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

    def sample(self, size: int | None = None, *, num_steps: int = 10, n_init: int = 5, batched: bool = True) -> Any:
        """Draw one sequence or a list of sequences from the grammar."""
        if size is None:
            return self.sample_one(num_steps=num_steps, n_init=n_init)
        sample_size = _exact_nonnegative_int(size, name="size")
        return [self.sample_one(num_steps=num_steps, n_init=n_init) for _ in range(sample_size)]


# --- estimator / accumulator ----------------------------------------------------------------------
class TemporalGraphGrammarStatistics(NamedTuple):
    schema_version: int
    bins: tuple[int, ...]
    directed: bool
    add_counts: np.ndarray
    rem_counts: np.ndarray
    edges: float
    rem_edges: float
    nodes: float
    steps: float


def _validate_temporal_statistics(
    value: Any,
    motif: CommonNeighbourMotif,
) -> TemporalGraphGrammarStatistics:
    if not isinstance(value, TemporalGraphGrammarStatistics) or value.schema_version != 1:
        raise ValueError("temporal graph statistics must use schema version 1.")
    if value.bins != motif.bins or value.directed != motif.directed:
        raise ValueError("temporal graph statistics use an incompatible motif configuration.")
    add_counts = np.array(value.add_counts, dtype=np.float64, copy=True)
    rem_counts = np.array(value.rem_counts, dtype=np.float64, copy=True)
    if add_counts.shape != (motif.num_motifs,) or rem_counts.shape != (motif.num_motifs,):
        raise ValueError("temporal motif-count vectors have incompatible shapes.")
    scalars = tuple(float(v) for v in (value.edges, value.rem_edges, value.nodes, value.steps))
    if (
        np.any(~np.isfinite(add_counts))
        or np.any(~np.isfinite(rem_counts))
        or np.any(add_counts < 0.0)
        or np.any(rem_counts < 0.0)
        or any(not np.isfinite(v) or v < 0.0 for v in scalars)
    ):
        raise ValueError("temporal graph sufficient statistics must be finite and non-negative.")
    edges, rem_edges, nodes, steps = scalars
    if not np.isclose(float(add_counts.sum()), edges, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("temporal add_counts must sum to edges.")
    if not np.isclose(float(rem_counts.sum()), rem_edges, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("temporal rem_counts must sum to rem_edges.")
    return TemporalGraphGrammarStatistics(
        1,
        motif.bins,
        motif.directed,
        add_counts,
        rem_counts,
        edges,
        rem_edges,
        nodes,
        steps,
    )


class TemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate per-motif edge counts + step/edge/node totals -- the exact sufficient statistics."""

    def __init__(self, motif: CommonNeighbourMotif) -> None:
        if not isinstance(motif, CommonNeighbourMotif):
            raise ValueError("motif must be a CommonNeighbourMotif.")
        self.motif = CommonNeighbourMotif(motif.bins, directed=motif.directed)
        self.add_counts = np.zeros(self.motif.num_motifs, dtype=np.float64)
        self.rem_counts = np.zeros(self.motif.num_motifs, dtype=np.float64)
        self.edges = 0.0
        self.rem_edges = 0.0
        self.nodes = 0.0
        self.steps = 0.0

    def update(self, x: Sequence[Any], weight: float, estimate: Any | None) -> None:
        """Accumulate sufficient statistics from one dynamic graph sequence."""
        if isinstance(x, ApproximateTemporalGraphSample):
            raise ValueError("approximate scalable samples must be explicitly unwrapped before fitting.")
        snaps = [_binarize(snapshot, directed=self.motif.directed) for snapshot in x]
        if not snaps:
            raise ValueError("temporal graph observations must contain at least one snapshot.")
        checked_weight = _finite_nonnegative(weight, name="weight")
        pending_add = np.zeros_like(self.add_counts)
        pending_rem = np.zeros_like(self.rem_counts)
        pending_edges = pending_rem_edges = pending_nodes = pending_steps = 0.0
        for t in range(1, len(snaps)):
            prev, cur = snaps[t - 1], snaps[t]
            if cur.shape[0] < prev.shape[0]:
                raise ValueError("bare temporal graph grammar does not support node removal.")
            ai, aj, ri, rj = _edge_diff(prev, cur, self.motif.directed)
            _, add_lookup = self.motif.counts_and_binner(_pad(prev, cur.shape[0]), on_edges=False)
            _, rem_lookup = self.motif.counts_and_binner(prev, on_edges=True)
            for m in add_lookup(np.asarray(ai), np.asarray(aj)):
                pending_add[m] += checked_weight
            for m in rem_lookup(np.asarray(ri), np.asarray(rj)):
                pending_rem[m] += checked_weight
            pending_edges += checked_weight * len(ai)
            pending_rem_edges += checked_weight * len(ri)
            pending_nodes += checked_weight * (cur.shape[0] - prev.shape[0])
            pending_steps += checked_weight
        self.add_counts += pending_add
        self.rem_counts += pending_rem
        self.edges += pending_edges
        self.rem_edges += pending_rem_edges
        self.nodes += pending_nodes
        self.steps += pending_steps

    def seq_update(self, x: Sequence[Sequence[np.ndarray]], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted sufficient statistics from a batch of sequences."""
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the temporal batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = TemporalGraphGrammarAccumulator(self.motif)
        for seq, weight in zip(x, checked_weights):
            pending.update(seq, float(weight), estimate)
        self.combine(pending.value())

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        self.seq_update(x, weights, None)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted sequence."""
        self.update(x, weight, None)

    def combine(self, suff_stat: TemporalGraphGrammarStatistics) -> TemporalGraphGrammarAccumulator:
        """Merge serialized sufficient statistics into this accumulator."""
        checked = _validate_temporal_statistics(suff_stat, self.motif)
        self.add_counts += checked.add_counts
        self.rem_counts += checked.rem_counts
        self.edges += checked.edges
        self.rem_edges += checked.rem_edges
        self.nodes += checked.nodes
        self.steps += checked.steps
        return self

    def value(self) -> TemporalGraphGrammarStatistics:
        """Return serialized sufficient statistics for estimation or merging."""
        return TemporalGraphGrammarStatistics(
            1,
            self.motif.bins,
            self.motif.directed,
            self.add_counts.copy(),
            self.rem_counts.copy(),
            self.edges,
            self.rem_edges,
            self.nodes,
            self.steps,
        )

    def from_value(self, x: TemporalGraphGrammarStatistics) -> TemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        checked = _validate_temporal_statistics(x, self.motif)
        self.add_counts = checked.add_counts
        self.rem_counts = checked.rem_counts
        self.edges = checked.edges
        self.rem_edges = checked.rem_edges
        self.nodes = checked.nodes
        self.steps = checked.steps
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return TemporalGraphGrammarDataEncoder(self.motif.directed)


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
        source = motif if motif is not None else CommonNeighbourMotif()
        if not isinstance(source, CommonNeighbourMotif):
            raise ValueError("motif must be a CommonNeighbourMotif.")
        self.motif = CommonNeighbourMotif(source.bins, directed=source.directed)
        self.pseudo_count = None if pseudo_count is None else _finite_nonnegative(pseudo_count, name="pseudo_count")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> TemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by the estimator."""
        return TemporalGraphGrammarAccumulatorFactory(self.motif)

    def estimate(
        self,
        nobs: float | None,
        suff_stat: TemporalGraphGrammarStatistics,
    ) -> TemporalGraphGrammarDistribution:
        """Estimate grammar weights and rates from sufficient statistics."""
        checked = _validate_temporal_statistics(suff_stat, self.motif)
        add_counts, rem_counts = checked.add_counts, checked.rem_counts
        edges, rem_edges, nodes, steps = checked.edges, checked.rem_edges, checked.nodes, checked.steps
        if steps <= 0.0:
            raise ValueError("cannot estimate a temporal graph grammar without transition evidence.")

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
            directed=self.motif.directed,
            name=self.name,
        )


# --- encoder --------------------------------------------------------------------------------------
class TemporalGraphGrammarDataEncoder(DataSequenceEncoder):
    """Pass-through encoder for dynamic graph sequence observations."""

    def __init__(self, directed: bool = False) -> None:
        self.directed = bool(directed)

    def seq_encode(self, x: Sequence[Sequence[np.ndarray]]) -> Sequence[Sequence[np.ndarray]]:
        """Validate and own every adjacency in a temporal graph batch."""
        return tuple(
            tuple(_binarize(snapshot, directed=self.directed) for snapshot in sequence)
            for sequence in x
        )

    def row_count(self, x: Any) -> int:
        """Return the number of temporal sequences in an encoded payload."""
        if not isinstance(x, tuple):
            raise ValueError("encoded temporal graph payload must be a tuple.")
        return len(x)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TemporalGraphGrammarDataEncoder) and other.directed == self.directed


# --- labelled (attributed) dynamic graphs ---------------------------------------------------------
def _emission_ll(dist: Any, records: Sequence[Any]) -> float:
    if dist is None or not records:
        return 0.0
    enc = dist.dist_to_encoder().seq_encode(list(records))
    return float(np.sum(dist.seq_log_density(enc)))


def _validate_labeled_observation(
    x: Any,
    structure: TemporalGraphGrammarDistribution,
    *,
    has_node_dist: bool,
    has_edge_dist: bool,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[tuple[Any, ...], ...]]:
    """Canonicalize an attributed observation and bind every record to a structural event."""
    if not isinstance(x, (tuple, list)) or len(x) != 3:
        raise ValueError("attributed temporal observations must be (snapshots, node_features, edge_features).")
    raw_snaps, raw_nodes, raw_edges = x
    if isinstance(raw_snaps, ApproximateTemporalGraphSample):
        raise ValueError("approximate scalable samples must be explicitly unwrapped before attribution.")
    try:
        snaps = tuple(_binarize(snapshot, directed=structure.directed) for snapshot in raw_snaps)
        node_features = tuple(raw_nodes)
    except TypeError as exc:
        raise ValueError("attributed temporal observation fields must be sequences.") from exc
    if not snaps:
        raise ValueError("attributed temporal observations must contain at least one snapshot.")
    structure.log_density(snaps)
    if has_node_dist:
        if len(node_features) != snaps[-1].shape[0]:
            raise ValueError("node_features must contain exactly one record per final node index.")
    elif node_features:
        raise ValueError("node_features must be empty when no node distribution is configured.")

    if has_edge_dist:
        try:
            edge_features = tuple(tuple(group) for group in raw_edges)
        except TypeError as exc:
            raise ValueError("edge_features must be a sequence of per-transition record sequences.") from exc
        if len(edge_features) != len(snaps) - 1:
            raise ValueError("edge_features must contain exactly one record group per transition.")
        for transition, (prev, cur, group) in enumerate(zip(snaps, snaps[1:], edge_features)):
            num_added = len(_edge_diff(prev, cur, structure.directed)[0])
            if len(group) != num_added:
                raise ValueError(
                    "edge feature group %d must contain exactly one record per added edge." % transition
                )
    else:
        try:
            edge_features = tuple(raw_edges)
        except TypeError as exc:
            raise ValueError("edge_features must be a sequence.") from exc
        if edge_features:
            raise ValueError("edge_features must be empty when no edge distribution is configured.")
        edge_features = ()
    return snaps, node_features, edge_features


class LabeledTemporalGraphGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """A dynamic graph whose nodes and edges carry attributes.

    Composes a structural :class:`TemporalGraphGrammarDistribution` (the topology over time) with two
    ordinary mixle distributions: ``node_dist`` over per-node attribute records (location, name, age, ... --
    typically a ``CompositeDistribution`` of leaves or a mixture) and ``edge_dist`` over per-edge attribute
    records (communication counts, channel, weight, ...). An observation is ``(snapshots, node_features,
    edge_features)``: the adjacency chain, one attribute record per final node index, and one edge-record
    group per transition containing one record per added edge in canonical ``_edge_diff`` order. The
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
        if not isinstance(structure, TemporalGraphGrammarDistribution):
            raise ValueError("structure must be a TemporalGraphGrammarDistribution.")
        for label, distribution in (("node_dist", node_dist), ("edge_dist", edge_dist)):
            if distribution is not None and not isinstance(distribution, SequenceEncodableProbabilityDistribution):
                raise ValueError(f"{label} must be a sequence-encodable probability distribution or None.")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
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
        snaps, node_features, edge_features = _validate_labeled_observation(
            x,
            self.structure,
            has_node_dist=self.node_dist is not None,
            has_edge_dist=self.edge_dist is not None,
        )
        flat_edges = tuple(record for group in edge_features for record in group)
        return (
            self.structure.log_density(snaps)
            + _emission_ll(self.node_dist, node_features)
            + _emission_ll(self.edge_dist, flat_edges)
        )

    def seq_encode(self, x: Sequence[tuple]) -> Sequence[tuple]:
        """Validate and encode a batch of event-aligned attributed observations."""
        return self.dist_to_encoder().seq_encode(x)

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
            structure=self.structure,
            node_encoder=None if self.node_dist is None else self.node_dist.dist_to_encoder(),
            edge_encoder=None if self.edge_dist is None else self.edge_dist.dist_to_encoder(),
            name=self.name,
        )

    def dist_to_encoder(self) -> LabeledTemporalGraphGrammarDataEncoder:
        """Return the event-aligned attributed graph encoder."""
        return LabeledTemporalGraphGrammarDataEncoder(
            self.structure,
            self.node_dist is not None,
            self.edge_dist is not None,
        )


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
        node_features = (
            list(self.dist.node_dist.sampler(self.rng.randint(2**31)).sample(size=n_final))
            if self.dist.node_dist is not None
            else []
        )
        edge_features = []
        if self.dist.edge_dist is not None:
            edge_sampler = self.dist.edge_dist.sampler(self.rng.randint(2**31))
            for prev, cur in zip(snaps, snaps[1:]):
                num_added = len(_edge_diff(prev, cur, directed)[0])
                edge_features.append(list(edge_sampler.sample(size=num_added)) if num_added else [])
        return snaps, node_features, edge_features

    def sample(self, size: int | None = None, **kw: Any) -> Any:
        """Draw one observation or a list of observations."""
        if size is None:
            return self.sample_one(**kw)
        sample_size = _exact_nonnegative_int(size, name="size")
        return [self.sample_one(**kw) for _ in range(sample_size)]


class LabeledTemporalGraphGrammarStatistics(NamedTuple):
    schema_version: int
    structure: Any
    node: Any
    edge: Any
    node_weight: float
    edge_weight: float


class LabeledTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for structure, node-attribute, and edge-attribute sufficient statistics."""

    def __init__(
        self,
        structure_acc: Any,
        node_acc: Any,
        edge_acc: Any,
        node_encoder: Any,
        edge_encoder: Any,
        factory: LabeledTemporalGraphGrammarAccumulatorFactory,
    ) -> None:
        self.structure_acc = structure_acc
        self.node_acc = node_acc
        self.edge_acc = edge_acc
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.factory = factory
        self.node_weight = 0.0
        self.edge_weight = 0.0

    def _validated_and_encoded(
        self,
        x: Any,
        estimate: Any | None,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], Any, Any]:
        structure = estimate.structure if estimate is not None else self.factory.structure
        snaps, nodes, edge_groups = _validate_labeled_observation(
            x,
            structure,
            has_node_dist=self.node_acc is not None,
            has_edge_dist=self.edge_acc is not None,
        )
        edges = tuple(record for group in edge_groups for record in group)
        node_encoder = (
            estimate.node_dist.dist_to_encoder()
            if estimate is not None and estimate.node_dist is not None
            else self.node_encoder
        )
        edge_encoder = (
            estimate.edge_dist.dist_to_encoder()
            if estimate is not None and estimate.edge_dist is not None
            else self.edge_encoder
        )
        node_encoded = node_encoder.seq_encode(nodes) if self.node_acc is not None else None
        edge_encoded = edge_encoder.seq_encode(edges) if self.edge_acc is not None else None
        return snaps, nodes, edges, node_encoded, edge_encoded

    def update(self, x: tuple, weight: float, estimate: Any | None) -> None:
        """Accumulate sufficient statistics from one attributed graph observation."""
        checked_weight = _finite_nonnegative(weight, name="weight")
        snaps, nodes, edges, node_encoded, edge_encoded = self._validated_and_encoded(x, estimate)
        pending = self.factory.make()
        pending.structure_acc.update(snaps, checked_weight, None if estimate is None else estimate.structure)
        if pending.node_acc is not None and nodes:
            pending.node_acc.seq_update(
                node_encoded,
                np.full(len(nodes), checked_weight),
                None if estimate is None else estimate.node_dist,
            )
            pending.node_weight = checked_weight * len(nodes)
        if pending.edge_acc is not None and edges:
            pending.edge_acc.seq_update(
                edge_encoded,
                np.full(len(edges), checked_weight),
                None if estimate is None else estimate.edge_dist,
            )
            pending.edge_weight = checked_weight * len(edges)
        self.combine(pending.value())

    def seq_update(self, x: Sequence[tuple], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted sufficient statistics from a batch."""
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the attributed batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = self.factory.make()
        for obs, weight in zip(x, checked_weights):
            pending.update(obs, float(weight), estimate)
        self.combine(pending.value())

    def initialize(self, x: tuple, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted observation."""
        checked_weight = _finite_nonnegative(weight, name="weight")
        snaps, nodes, edges, node_encoded, edge_encoded = self._validated_and_encoded(x, None)
        pending = self.factory.make()
        pending.structure_acc.initialize(snaps, checked_weight, rng)
        if pending.node_acc is not None and nodes:
            pending.node_acc.seq_initialize(node_encoded, np.full(len(nodes), checked_weight), rng)
            pending.node_weight = checked_weight * len(nodes)
        if pending.edge_acc is not None and edges:
            pending.edge_acc.seq_initialize(edge_encoded, np.full(len(edges), checked_weight), rng)
            pending.edge_weight = checked_weight * len(edges)
        self.combine(pending.value())

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the attributed batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = self.factory.make()
        for obs, weight in zip(x, checked_weights):
            pending.initialize(obs, float(weight), rng)
        self.combine(pending.value())

    def combine(self, suff_stat: LabeledTemporalGraphGrammarStatistics) -> LabeledTemporalGraphGrammarAccumulator:
        """Merge serialized attributed-graph sufficient statistics."""
        if not isinstance(suff_stat, LabeledTemporalGraphGrammarStatistics) or suff_stat.schema_version != 1:
            raise ValueError("labeled temporal graph statistics must use schema version 1.")
        node_weight = _finite_nonnegative(suff_stat.node_weight, name="node_weight")
        edge_weight = _finite_nonnegative(suff_stat.edge_weight, name="edge_weight")
        self.structure_acc.combine(suff_stat.structure)
        if self.node_acc is not None:
            if suff_stat.node is None:
                raise ValueError("node statistics are required by the configured node model.")
            self.node_acc.combine(suff_stat.node)
        elif suff_stat.node is not None or node_weight != 0.0:
            raise ValueError("node statistics are incompatible with a model that has no node distribution.")
        if self.edge_acc is not None:
            if suff_stat.edge is None:
                raise ValueError("edge statistics are required by the configured edge model.")
            self.edge_acc.combine(suff_stat.edge)
        elif suff_stat.edge is not None or edge_weight != 0.0:
            raise ValueError("edge statistics are incompatible with a model that has no edge distribution.")
        self.node_weight += node_weight
        self.edge_weight += edge_weight
        return self

    def value(self) -> LabeledTemporalGraphGrammarStatistics:
        """Return serialized attributed-graph sufficient statistics."""
        return LabeledTemporalGraphGrammarStatistics(
            1,
            self.structure_acc.value(),
            None if self.node_acc is None else self.node_acc.value(),
            None if self.edge_acc is None else self.edge_acc.value(),
            self.node_weight,
            self.edge_weight,
        )

    def from_value(self, x: LabeledTemporalGraphGrammarStatistics) -> LabeledTemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        fresh = self.factory.make()
        fresh.combine(x)
        self.structure_acc = fresh.structure_acc
        self.node_acc = fresh.node_acc
        self.edge_acc = fresh.edge_acc
        self.node_weight = fresh.node_weight
        self.edge_weight = fresh.edge_weight
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> LabeledTemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return self.factory.encoder


class LabeledTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for attributed temporal graph grammar accumulators."""

    def __init__(
        self,
        structure: TemporalGraphGrammarDistribution,
        structure_factory: Any,
        node_factory: Any,
        edge_factory: Any,
        node_encoder: Any,
        edge_encoder: Any,
    ) -> None:
        self.structure = structure
        self.structure_factory = structure_factory
        self.node_factory = node_factory
        self.edge_factory = edge_factory
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.encoder = LabeledTemporalGraphGrammarDataEncoder(
            structure,
            node_factory is not None,
            edge_factory is not None,
        )

    def make(self) -> LabeledTemporalGraphGrammarAccumulator:
        """Create a fresh attributed temporal graph grammar accumulator."""
        return LabeledTemporalGraphGrammarAccumulator(
            self.structure_factory.make(),
            None if self.node_factory is None else self.node_factory.make(),
            None if self.edge_factory is None else self.edge_factory.make(),
            self.node_encoder,
            self.edge_encoder,
            self,
        )


class LabeledTemporalGraphGrammarEstimator(ParameterEstimator):
    """Estimator for attributed temporal graph grammars."""

    def __init__(
        self,
        structure_estimator: Any,
        node_estimator: Any = None,
        edge_estimator: Any = None,
        *,
        structure: TemporalGraphGrammarDistribution,
        node_encoder: Any = None,
        edge_encoder: Any = None,
        name: str | None = None,
    ) -> None:
        self.structure_estimator = structure_estimator
        self.node_estimator = node_estimator
        self.edge_estimator = edge_estimator
        self.structure = structure
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> LabeledTemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return LabeledTemporalGraphGrammarAccumulatorFactory(
            self.structure,
            self.structure_estimator.accumulator_factory(),
            None if self.node_estimator is None else self.node_estimator.accumulator_factory(),
            None if self.edge_estimator is None else self.edge_estimator.accumulator_factory(),
            self.node_encoder,
            self.edge_encoder,
        )

    def estimate(
        self,
        nobs: float | None,
        suff_stat: LabeledTemporalGraphGrammarStatistics,
    ) -> LabeledTemporalGraphGrammarDistribution:
        """Estimate structure and attribute distributions from sufficient statistics."""
        if not isinstance(suff_stat, LabeledTemporalGraphGrammarStatistics) or suff_stat.schema_version != 1:
            raise ValueError("labeled temporal graph statistics must use schema version 1.")
        node_weight = _finite_nonnegative(suff_stat.node_weight, name="node_weight")
        edge_weight = _finite_nonnegative(suff_stat.edge_weight, name="edge_weight")
        if self.node_estimator is not None and node_weight <= 0.0:
            raise ValueError("cannot estimate node attributes without aligned node records.")
        if self.edge_estimator is not None and edge_weight <= 0.0:
            raise ValueError("cannot estimate edge attributes without aligned edge records.")
        return LabeledTemporalGraphGrammarDistribution(
            self.structure_estimator.estimate(nobs, suff_stat.structure),
            None
            if self.node_estimator is None
            else self.node_estimator.estimate(node_weight, suff_stat.node),
            None
            if self.edge_estimator is None
            else self.edge_estimator.estimate(edge_weight, suff_stat.edge),
            name=self.name,
        )


class LabeledTemporalGraphGrammarDataEncoder(DataSequenceEncoder):
    """Validate event alignment while retaining the attributed observation structure."""

    def __init__(
        self,
        structure: TemporalGraphGrammarDistribution,
        has_node_dist: bool,
        has_edge_dist: bool,
    ) -> None:
        self.structure = structure
        self.has_node_dist = has_node_dist
        self.has_edge_dist = has_edge_dist

    def seq_encode(self, x: Sequence[tuple]) -> tuple[tuple, ...]:
        return tuple(
            _validate_labeled_observation(
                observation,
                self.structure,
                has_node_dist=self.has_node_dist,
                has_edge_dist=self.has_edge_dist,
            )
            for observation in x
        )

    def row_count(self, x: Any) -> int:
        if not isinstance(x, tuple):
            raise ValueError("encoded labeled temporal graph payload must be a tuple.")
        return len(x)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LabeledTemporalGraphGrammarDataEncoder)
            and other.structure.motif.bins == self.structure.motif.bins
            and other.structure.directed == self.structure.directed
            and other.has_node_dist == self.has_node_dist
            and other.has_edge_dist == self.has_edge_dist
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
    "LabeledTemporalGraphGrammarDataEncoder",
    "LabeledTemporalGraphGrammarStatistics",
    "HomophilyTemporalGraphGrammarDistribution",
    "HomophilyTemporalGraphGrammarSampler",
    "HomophilyTemporalGraphGrammarEstimator",
    "HomophilyTemporalGraphGrammarAccumulator",
    "HomophilyTemporalGraphGrammarAccumulatorFactory",
    "HomophilyTemporalGraphGrammarDataEncoder",
    "HomophilyTemporalGraphGrammarStatistics",
    "ChurningTemporalGraphGrammarDistribution",
    "ChurningTemporalGraphGrammarSampler",
    "ChurningTemporalGraphGrammarEstimator",
    "ChurningTemporalGraphGrammarAccumulator",
    "ChurningTemporalGraphGrammarAccumulatorFactory",
    "ChurningTemporalGraphGrammarDataEncoder",
    "ChurningTemporalGraphGrammarStatistics",
    "LatentTemporalGraphGrammarDistribution",
    "LatentTemporalGraphGrammarSampler",
    "LatentTemporalGraphGrammarEstimator",
    "LatentTemporalGraphGrammarAccumulator",
    "LatentTemporalGraphGrammarAccumulatorFactory",
    "LatentTemporalGraphGrammarStatistics",
    "LatentAttributedTemporalGraphGrammarDistribution",
    "LatentAttributedTemporalGraphGrammarSampler",
    "LatentAttributedTemporalGraphGrammarEstimator",
    "LatentAttributedTemporalGraphGrammarAccumulator",
    "LatentAttributedTemporalGraphGrammarAccumulatorFactory",
    "LatentAttributedTemporalGraphGrammarDataEncoder",
    "LatentAttributedTemporalGraphGrammarStatistics",
    "LatentChurningTemporalGraphGrammarDistribution",
    "LatentChurningTemporalGraphGrammarSampler",
    "LatentChurningTemporalGraphGrammarEstimator",
    "LatentChurningTemporalGraphGrammarAccumulator",
    "LatentChurningTemporalGraphGrammarAccumulatorFactory",
    "regime_moment_init",
]


# --- homophily: attribute-conditioned edge formation ----------------------------------------------
def _validate_homophily_observation(
    x: Any,
    motif: CommonNeighbourMotif,
    num_types: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError("homophily observations must be (snapshots, node_types).")
    raw_snaps, raw_types = x
    try:
        snaps = tuple(
            _binarize(snapshot, directed=False).toarray()
            if sp.issparse(snapshot)
            else _binarize(snapshot, directed=False)
            for snapshot in raw_snaps
        )
    except TypeError as exc:
        raise ValueError("homophily snapshots must be a sequence.") from exc
    if not snaps:
        raise ValueError("homophily observations must contain at least one snapshot.")
    types = np.asarray(raw_types)
    if types.ndim != 1 or not np.issubdtype(types.dtype, np.integer):
        raise ValueError("node_types must be a one-dimensional exact integer vector.")
    types = np.array(types, dtype=np.int64, copy=True)
    if len(types) != snaps[-1].shape[0]:
        raise ValueError("node_types must contain exactly one type per final node index.")
    if np.any(types < 0) or np.any(types >= num_types):
        raise ValueError("node_types contain a value outside the configured type support.")
    for previous, current in zip(snaps, snaps[1:]):
        if current.shape[0] < previous.shape[0]:
            raise ValueError("homophily temporal graphs do not support node removal.")
        if len(_edge_diff(previous, current)[2]):
            raise ValueError("homophily temporal graphs do not support edge removal.")
    return snaps, types


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
        source_motif = motif if motif is not None else CommonNeighbourMotif()
        if not isinstance(source_motif, CommonNeighbourMotif) or source_motif.directed:
            raise ValueError("motif must be an undirected CommonNeighbourMotif.")
        self.motif = CommonNeighbourMotif(source_motif.bins)
        tw = np.array(type_weights, dtype=np.float64, copy=True)
        if tw.ndim != 1 or tw.size == 0:
            raise ValueError("type_weights must be a nonempty one-dimensional vector.")
        if np.any(~np.isfinite(tw)) or np.any(tw < 0.0) or float(tw.sum()) <= 0.0:
            raise ValueError("type_weights must be finite, non-negative, and have positive total.")
        self.K = len(tw)
        self.M = self.motif.num_motifs
        checked_rate = np.array(rate, dtype=np.float64, copy=True)
        if checked_rate.shape != (self.M, self.K, self.K):
            raise ValueError("rate must have shape (num_motifs, num_types, num_types).")
        if np.any(~np.isfinite(checked_rate)) or np.any(checked_rate < 0.0):
            raise ValueError("rate entries must be finite and non-negative.")
        if not np.array_equal(checked_rate, checked_rate.transpose(0, 2, 1)):
            raise ValueError("rate must be symmetric over its two unordered type axes.")
        checked_rate.setflags(write=False)
        self.rate = checked_rate
        normalized_tw = tw / tw.sum()
        normalized_tw.setflags(write=False)
        self.type_weights = normalized_tw
        self.log_tw = np.full(self.K, -math.inf, dtype=np.float64)
        self.log_tw[self.type_weights > 0.0] = np.log(self.type_weights[self.type_weights > 0.0])
        self.log_tw.setflags(write=False)
        self.node_rate = _finite_nonnegative(node_rate, name="node_rate")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
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
        if self.node_rate == 0.0:
            lp = 0.0 if new_nodes == 0 else float("-inf")
        else:
            lp = new_nodes * math.log(self.node_rate) - self.node_rate - math.lgamma(new_nodes + 1)
        selected = np.zeros((self.M, self.K, self.K), dtype=np.int64)
        if len(ai):
            m = lookup(np.asarray(ai), np.asarray(aj))
            a, b = self._pair_axes(np.asarray(ai), np.asarray(aj), types)
            np.add.at(selected, (m, a, b), 1)
        for motif_index in range(self.M):
            for left_type in range(self.K):
                for right_type in range(left_type, self.K):
                    lp += _capped_poisson_subset_log_prob(
                        int(selected[motif_index, left_type, right_type]),
                        int(cand[motif_index, left_type, right_type]),
                        float(self.rate[motif_index, left_type, right_type]),
                    )
        return lp

    def log_density(self, x: tuple) -> float:
        """Score one homophily dynamic graph observation."""
        snaps, types = _validate_homophily_observation(x, self.motif, self.K)
        lp = float(np.sum(self.log_tw[types]))  # node-type likelihood (each node's type ~ Categorical)
        lp += sum(self._transition_log_density(snaps[t - 1], snaps[t], types) for t in range(1, len(snaps)))
        return lp

    def seq_encode(self, x: Sequence[tuple]) -> Sequence[tuple]:
        """Validate and encode homophily observations."""
        return self.dist_to_encoder().seq_encode(x)

    def seq_log_density(self, x: Sequence[tuple]) -> np.ndarray:
        """Score a batch of homophily dynamic graph observations."""
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> HomophilyTemporalGraphGrammarSampler:
        """Return a sampler for homophily dynamic graph observations."""
        return HomophilyTemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> HomophilyTemporalGraphGrammarEstimator:
        """Return the estimator for homophily rates and node-type weights."""
        return HomophilyTemporalGraphGrammarEstimator(self.M, self.K, self.motif, pseudo_count, self.name)

    def dist_to_encoder(self) -> HomophilyTemporalGraphGrammarDataEncoder:
        """Return the validated homophily encoder."""
        return HomophilyTemporalGraphGrammarDataEncoder(self.motif, self.K)


class HomophilyTemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for homophily temporal graph observations."""

    def __init__(self, dist: HomophilyTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample_one(self, num_steps: int = 8, seed_graph: np.ndarray | None = None, n_init: int = 8) -> tuple:
        """Draw one homophily dynamic graph observation."""
        d = self.dist
        num_steps = _exact_nonnegative_int(num_steps, name="num_steps")
        n_init = _exact_nonnegative_int(n_init, name="n_init")
        if seed_graph is None:
            adj = np.zeros((n_init, n_init))
        else:
            canonical = _binarize(seed_graph)
            adj = canonical.toarray() if sp.issparse(canonical) else canonical
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
        sample_size = _exact_nonnegative_int(size, name="size")
        return [self.sample_one(**kw) for _ in range(sample_size)]


class HomophilyTemporalGraphGrammarStatistics(NamedTuple):
    schema_version: int
    bins: tuple[int, ...]
    num_types: int
    edge_counts: np.ndarray
    type_counts: np.ndarray
    nodes: float
    steps: float


def _validate_homophily_statistics(
    value: Any,
    motif: CommonNeighbourMotif,
    num_types: int,
) -> HomophilyTemporalGraphGrammarStatistics:
    if not isinstance(value, HomophilyTemporalGraphGrammarStatistics) or value.schema_version != 1:
        raise ValueError("homophily temporal statistics must use schema version 1.")
    if value.bins != motif.bins or value.num_types != num_types:
        raise ValueError("homophily temporal statistics use an incompatible model schema.")
    edge_counts = np.array(value.edge_counts, dtype=np.float64, copy=True)
    type_counts = np.array(value.type_counts, dtype=np.float64, copy=True)
    expected_shape = (motif.num_motifs, num_types, num_types)
    if edge_counts.shape != expected_shape or type_counts.shape != (num_types,):
        raise ValueError("homophily temporal statistics have incompatible count shapes.")
    nodes = float(value.nodes)
    steps = float(value.steps)
    if (
        np.any(~np.isfinite(edge_counts))
        or np.any(edge_counts < 0.0)
        or np.any(~np.isfinite(type_counts))
        or np.any(type_counts < 0.0)
        or not np.isfinite(nodes)
        or nodes < 0.0
        or not np.isfinite(steps)
        or steps < 0.0
    ):
        raise ValueError("homophily temporal statistics must be finite and non-negative.")
    if np.any(np.tril(edge_counts, -1) != 0.0):
        raise ValueError("homophily edge counts must use the upper-triangular unordered-type schema.")
    return HomophilyTemporalGraphGrammarStatistics(
        1,
        motif.bins,
        num_types,
        edge_counts,
        type_counts,
        nodes,
        steps,
    )


class HomophilyTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for homophily edge-rate and node-type sufficient statistics."""

    def __init__(self, M: int, K: int, motif: CommonNeighbourMotif) -> None:
        self.M, self.K = M, K
        self.motif = CommonNeighbourMotif(motif.bins)
        self.edge_counts = np.zeros((M, K, K), dtype=np.float64)
        self.type_counts = np.zeros(K, dtype=np.float64)
        self.nodes = 0.0
        self.steps = 0.0

    def update(self, x: tuple, weight: float, estimate: Any | None) -> None:
        """Accumulate sufficient statistics from one homophily observation."""
        snaps, types = _validate_homophily_observation(x, self.motif, self.K)
        checked_weight = _finite_nonnegative(weight, name="weight")
        pending_edges = np.zeros_like(self.edge_counts)
        pending_types = np.zeros_like(self.type_counts)
        pending_nodes = pending_steps = 0.0
        np.add.at(pending_types, types, checked_weight)
        for t in range(1, len(snaps)):
            prev, cur = snaps[t - 1], snaps[t]
            ai, aj, _, _ = _edge_diff(prev, cur)
            _, lookup = self.motif.counts_and_binner(_pad(prev, cur.shape[0]), on_edges=False)
            if len(ai):
                m = lookup(np.asarray(ai), np.asarray(aj))
                a = np.minimum(types[np.asarray(ai)], types[np.asarray(aj)])
                b = np.maximum(types[np.asarray(ai)], types[np.asarray(aj)])
                np.add.at(pending_edges, (m, a, b), checked_weight)
            pending_nodes += checked_weight * (cur.shape[0] - prev.shape[0])
            pending_steps += checked_weight
        self.edge_counts += pending_edges
        self.type_counts += pending_types
        self.nodes += pending_nodes
        self.steps += pending_steps

    def seq_update(self, x: Sequence[tuple], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted sufficient statistics from a batch."""
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the homophily batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = HomophilyTemporalGraphGrammarAccumulator(self.M, self.K, self.motif)
        for obs, weight in zip(x, checked_weights):
            pending.update(obs, float(weight), estimate)
        self.combine(pending.value())

    def initialize(self, x: tuple, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted observation."""
        self.update(x, weight, None)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        self.seq_update(x, weights, None)

    def combine(
        self,
        suff_stat: HomophilyTemporalGraphGrammarStatistics,
    ) -> HomophilyTemporalGraphGrammarAccumulator:
        """Merge serialized homophily sufficient statistics."""
        checked = _validate_homophily_statistics(suff_stat, self.motif, self.K)
        self.edge_counts += checked.edge_counts
        self.type_counts += checked.type_counts
        self.nodes += checked.nodes
        self.steps += checked.steps
        return self

    def value(self) -> HomophilyTemporalGraphGrammarStatistics:
        """Return serialized homophily sufficient statistics."""
        return HomophilyTemporalGraphGrammarStatistics(
            1,
            self.motif.bins,
            self.K,
            self.edge_counts.copy(),
            self.type_counts.copy(),
            self.nodes,
            self.steps,
        )

    def from_value(self, x: HomophilyTemporalGraphGrammarStatistics) -> HomophilyTemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        checked = _validate_homophily_statistics(x, self.motif, self.K)
        self.edge_counts = checked.edge_counts
        self.type_counts = checked.type_counts
        self.nodes = checked.nodes
        self.steps = checked.steps
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> HomophilyTemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return HomophilyTemporalGraphGrammarDataEncoder(self.motif, self.K)


class HomophilyTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for homophily temporal graph grammar accumulators."""

    def __init__(self, M: int, K: int, motif: CommonNeighbourMotif) -> None:
        self.M, self.K = M, K
        self.motif = CommonNeighbourMotif(motif.bins)

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
        source_motif = motif if motif is not None else CommonNeighbourMotif()
        if not isinstance(source_motif, CommonNeighbourMotif) or source_motif.directed:
            raise ValueError("motif must be an undirected CommonNeighbourMotif.")
        self.motif = CommonNeighbourMotif(source_motif.bins)
        self.M = _exact_positive_int(M, name="M")
        self.K = _exact_positive_int(K, name="K")
        if self.M != self.motif.num_motifs:
            raise ValueError("M must equal the configured motif count.")
        self.pseudo_count = None if pseudo_count is None else _finite_nonnegative(pseudo_count, name="pseudo_count")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> HomophilyTemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return HomophilyTemporalGraphGrammarAccumulatorFactory(self.M, self.K, self.motif)

    def estimate(
        self,
        nobs: float | None,
        suff_stat: HomophilyTemporalGraphGrammarStatistics,
    ) -> HomophilyTemporalGraphGrammarDistribution:
        """Estimate homophily rates, type weights, and node rate from sufficient statistics."""
        checked = _validate_homophily_statistics(suff_stat, self.motif, self.K)
        if checked.steps <= 0.0:
            raise ValueError("cannot estimate a homophily temporal grammar without transition evidence.")
        if float(checked.type_counts.sum()) <= 0.0:
            raise ValueError("cannot estimate homophily type weights without node-type evidence.")
        upper_rate = checked.edge_counts / checked.steps
        rate = upper_rate + upper_rate.transpose(0, 2, 1)
        diagonal = np.arange(self.K)
        rate[:, diagonal, diagonal] = upper_rate[:, diagonal, diagonal]
        tc = checked.type_counts.copy()
        if self.pseudo_count is not None:
            tc = tc + float(self.pseudo_count)
        type_weights = tc / tc.sum()
        return HomophilyTemporalGraphGrammarDistribution(
            rate,
            type_weights,
            checked.nodes / checked.steps,
            motif=self.motif,
            name=self.name,
        )


class HomophilyTemporalGraphGrammarDataEncoder(DataSequenceEncoder):
    """Validate homophily observations and retain their graph/type pairing."""

    def __init__(self, motif: CommonNeighbourMotif, num_types: int) -> None:
        self.motif = CommonNeighbourMotif(motif.bins)
        self.num_types = num_types

    def seq_encode(self, x: Sequence[tuple]) -> tuple[tuple, ...]:
        return tuple(_validate_homophily_observation(observation, self.motif, self.num_types) for observation in x)

    def row_count(self, x: Any) -> int:
        if not isinstance(x, tuple):
            raise ValueError("encoded homophily temporal graph payload must be a tuple.")
        return len(x)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, HomophilyTemporalGraphGrammarDataEncoder)
            and other.motif.bins == self.motif.bins
            and other.num_types == self.num_types
        )


# --- node churn: removal + addition with identity tracking -----------------------------------------
def _node_removal_logp(rate: float, n_prev: int, k_removed: int) -> float:
    """Exact law for a uniformly selected subset of ``min(Poisson(rate), n_prev)`` nodes."""
    return _capped_poisson_subset_log_prob(k_removed, n_prev, rate)


def _validate_identity_snapshot(snapshot: Any, directed: bool) -> tuple[Any, tuple[Any, ...]]:
    if not isinstance(snapshot, (tuple, list)) or len(snapshot) != 2:
        raise ValueError("identity-tracked snapshots must be (adjacency, node_ids).")
    adjacency, raw_ids = snapshot
    canonical = _binarize(adjacency, directed=directed)
    try:
        node_ids = tuple(raw_ids)
        unique_ids = set(node_ids)
    except TypeError as exc:
        raise ValueError("node_ids must be a sequence of hashable stable identities.") from exc
    if len(node_ids) != canonical.shape[0]:
        raise ValueError("node_ids must contain exactly one identity per adjacency row.")
    if len(unique_ids) != len(node_ids):
        raise ValueError("node_ids must be unique within every snapshot.")
    return canonical, node_ids


def _validate_churning_sequence(
    x: Any,
    directed: bool,
) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
    try:
        snapshots = tuple(_validate_identity_snapshot(snapshot, directed) for snapshot in x)
    except TypeError as exc:
        raise ValueError("churning temporal observations must be a sequence of snapshots.") from exc
    if not snapshots:
        raise ValueError("churning temporal observations must contain at least one snapshot.")
    seen = set(snapshots[0][1])
    active = seen.copy()
    for _, node_ids in snapshots[1:]:
        current = set(node_ids)
        arriving = current - active
        if arriving & seen:
            raise ValueError("a departed node identity cannot be reused as a newly arriving node.")
        seen.update(arriving)
        active = current
    return snapshots


def _align_by_ids(
    prev_adj: Any,
    prev_ids: Sequence[Any],
    cur_adj: Any,
    cur_ids: Sequence[Any],
    directed: bool = False,
) -> tuple:
    """Align two snapshots by stable node id. Returns (prev_surviving_subgraph, cur_reordered, num_removed).

    Removed nodes = ids in prev but not cur; their incident edges vanish with them (not counted as edge
    removals). ``cur`` is reordered so the surviving nodes (in prev order) come first and the genuinely-new
    nodes are appended -- exactly the ``prev' -> cur`` layout the edit grammar expects (shared nodes keep
    their index, new nodes at the end)."""
    prev_adj, checked_prev_ids = _validate_identity_snapshot((prev_adj, prev_ids), directed)
    cur_adj, checked_cur_ids = _validate_identity_snapshot((cur_adj, cur_ids), directed)
    pid, cid = list(checked_prev_ids), list(checked_cur_ids)
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
        if not isinstance(edit_grammar, TemporalGraphGrammarDistribution):
            raise ValueError("edit_grammar must be a TemporalGraphGrammarDistribution.")
        self.edit_grammar = edit_grammar
        self.node_remove_rate = _finite_nonnegative(node_remove_rate, name="node_remove_rate")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
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
        snaps = _validate_churning_sequence(x, self.edit_grammar.directed)
        if len(snaps) < 2:
            return 0.0
        lp = 0.0
        for t in range(1, len(snaps)):
            pa, pid = snaps[t - 1]
            ca, cid = snaps[t]
            prev_surv, cur_reord, num_removed = _align_by_ids(
                pa,
                pid,
                ca,
                cid,
                self.edit_grammar.directed,
            )
            lp += self._node_removal_log_density(len(pid), num_removed)
            if lp == float("-inf"):
                return lp
            lp += self.edit_grammar._transition_log_density(prev_surv, cur_reord)
        return lp

    def seq_encode(self, x: Sequence[Any]) -> Sequence[Any]:
        """Validate and encode identity-tracked graph observations."""
        return self.dist_to_encoder().seq_encode(x)

    def seq_log_density(self, x: Sequence[Any]) -> np.ndarray:
        """Score a batch of churning graph observations."""
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> ChurningTemporalGraphGrammarSampler:
        """Return a sampler for churning graph observations."""
        return ChurningTemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ChurningTemporalGraphGrammarEstimator:
        """Return the estimator for the wrapped edit grammar and node churn rate."""
        return ChurningTemporalGraphGrammarEstimator(
            self.edit_grammar.estimator(pseudo_count=pseudo_count),
            edit_grammar=self.edit_grammar,
            name=self.name,
        )

    def dist_to_encoder(self) -> ChurningTemporalGraphGrammarDataEncoder:
        """Return the validated identity-tracked graph encoder."""
        return ChurningTemporalGraphGrammarDataEncoder(self.edit_grammar.directed)


class ChurningTemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for identity-tracked churning temporal graph observations."""

    def __init__(self, dist: ChurningTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.edit_sampler = dist.edit_grammar.sampler(self.rng.randint(2**31))

    def sample_one(self, num_steps: int = 10, seed_graph: np.ndarray | None = None, n_init: int = 8) -> list:
        """Draw one churning graph sequence."""
        d = self.dist
        num_steps = _exact_nonnegative_int(num_steps, name="num_steps")
        n_init = _exact_nonnegative_int(n_init, name="n_init")
        if seed_graph is None:
            adj = np.zeros((n_init, n_init))
        else:
            canonical = _binarize(seed_graph, directed=d.edit_grammar.directed)
            adj = canonical.toarray() if sp.issparse(canonical) else canonical
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
        sample_size = _exact_nonnegative_int(size, name="size")
        return [self.sample_one(**kw) for _ in range(sample_size)]


class ChurningTemporalGraphGrammarStatistics(NamedTuple):
    schema_version: int
    directed: bool
    edit: Any
    removed: float
    steps: float


def _validate_churning_statistics(
    value: Any,
    directed: bool,
) -> ChurningTemporalGraphGrammarStatistics:
    if not isinstance(value, ChurningTemporalGraphGrammarStatistics) or value.schema_version != 1:
        raise ValueError("churning temporal graph statistics must use schema version 1.")
    if value.directed != directed:
        raise ValueError("churning temporal graph statistics use incompatible directedness.")
    removed = float(value.removed)
    steps = float(value.steps)
    if not np.isfinite(removed) or removed < 0.0 or not np.isfinite(steps) or steps < 0.0:
        raise ValueError("churning temporal graph statistics must be finite and non-negative.")
    return ChurningTemporalGraphGrammarStatistics(1, directed, value.edit, removed, steps)


class ChurningTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for churning graph grammar and node-removal sufficient statistics."""

    def __init__(self, edit_acc: Any, directed: bool, factory: ChurningTemporalGraphGrammarAccumulatorFactory) -> None:
        self.edit_acc = edit_acc
        self.directed = directed
        self.factory = factory
        self.removed = 0.0
        self.steps = 0.0

    def update(self, x: Sequence[tuple], weight: float, estimate: Any | None) -> None:
        """Accumulate sufficient statistics from one churning graph sequence."""
        snaps = _validate_churning_sequence(x, self.directed)
        checked_weight = _finite_nonnegative(weight, name="weight")
        edit_est = None if estimate is None else estimate.edit_grammar
        pending = self.factory.make()
        for t in range(1, len(snaps)):
            pa, pid = snaps[t - 1]
            ca, cid = snaps[t]
            prev_surv, cur_reord, num_removed = _align_by_ids(pa, pid, ca, cid, self.directed)
            pending.removed += checked_weight * num_removed
            pending.steps += checked_weight
            pending.edit_acc.update(
                [prev_surv, cur_reord],
                checked_weight,
                edit_est,
            )
        self.combine(pending.value())

    def seq_update(self, x: Sequence[Any], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate weighted sufficient statistics from a batch."""
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the churning batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = self.factory.make()
        for observation, weight in zip(x, checked_weights):
            pending.update(observation, float(weight), estimate)
        self.combine(pending.value())

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted sequence."""
        self.update(x, weight, None)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: ChurningTemporalGraphGrammarStatistics) -> ChurningTemporalGraphGrammarAccumulator:
        """Merge serialized churning sufficient statistics."""
        checked = _validate_churning_statistics(suff_stat, self.directed)
        self.edit_acc.combine(checked.edit)
        self.removed += checked.removed
        self.steps += checked.steps
        return self

    def value(self) -> ChurningTemporalGraphGrammarStatistics:
        """Return serialized churning sufficient statistics."""
        return ChurningTemporalGraphGrammarStatistics(
            1,
            self.directed,
            self.edit_acc.value(),
            self.removed,
            self.steps,
        )

    def from_value(self, x: ChurningTemporalGraphGrammarStatistics) -> ChurningTemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        fresh = self.factory.make()
        fresh.combine(x)
        self.edit_acc = fresh.edit_acc
        self.removed = fresh.removed
        self.steps = fresh.steps
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> ChurningTemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return ChurningTemporalGraphGrammarDataEncoder(self.directed)


class ChurningTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for churning temporal graph grammar accumulators."""

    def __init__(self, edit_factory: Any, directed: bool) -> None:
        self.edit_factory = edit_factory
        self.directed = directed

    def make(self) -> ChurningTemporalGraphGrammarAccumulator:
        """Create a fresh churning accumulator."""
        return ChurningTemporalGraphGrammarAccumulator(self.edit_factory.make(), self.directed, self)


class ChurningTemporalGraphGrammarEstimator(ParameterEstimator):
    """Estimator for identity-tracked churning temporal graph grammars."""

    def __init__(
        self,
        edit_estimator: Any,
        edit_grammar: TemporalGraphGrammarDistribution | None = None,
        name: str | None = None,
    ) -> None:
        self.edit_estimator = edit_estimator
        if edit_grammar is not None and not isinstance(edit_grammar, TemporalGraphGrammarDistribution):
            raise ValueError("edit_grammar must be a TemporalGraphGrammarDistribution or None.")
        if edit_grammar is not None:
            self.directed = edit_grammar.directed
        elif hasattr(edit_estimator, "motif") and isinstance(edit_estimator.motif, CommonNeighbourMotif):
            self.directed = edit_estimator.motif.directed
        else:
            raise ValueError("edit_estimator must declare the edit grammar's motif directedness.")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> ChurningTemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return ChurningTemporalGraphGrammarAccumulatorFactory(
            self.edit_estimator.accumulator_factory(),
            self.directed,
        )

    def estimate(
        self,
        nobs: float | None,
        suff_stat: ChurningTemporalGraphGrammarStatistics,
    ) -> ChurningTemporalGraphGrammarDistribution:
        """Estimate the wrapped edit grammar and node-removal rate."""
        checked = _validate_churning_statistics(suff_stat, self.directed)
        if checked.steps <= 0.0:
            raise ValueError("cannot estimate a churning temporal grammar without transition evidence.")
        return ChurningTemporalGraphGrammarDistribution(
            self.edit_estimator.estimate(nobs, checked.edit),
            node_remove_rate=checked.removed / checked.steps,
            name=self.name,
        )


class ChurningTemporalGraphGrammarDataEncoder(DataSequenceEncoder):
    """Validate and encode identity-tracked temporal graph sequences."""

    def __init__(self, directed: bool) -> None:
        self.directed = bool(directed)

    def seq_encode(self, x: Sequence[Any]) -> tuple[tuple, ...]:
        return tuple(_validate_churning_sequence(observation, self.directed) for observation in x)

    def row_count(self, x: Any) -> int:
        if not isinstance(x, tuple):
            raise ValueError("encoded churning temporal graph payload must be a tuple.")
        return len(x)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ChurningTemporalGraphGrammarDataEncoder) and other.directed == self.directed


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


def _normalized_probability_vector(value: Any, size: int, *, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if np.any(~np.isfinite(result)) or np.any(result < 0.0) or float(result.sum()) <= 0.0:
        raise ValueError(f"{name} must be finite, non-negative, and have positive total.")
    result /= result.sum()
    result.setflags(write=False)
    return result


def _normalized_transition_matrix(value: Any, size: int) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (size, size):
        raise ValueError(f"transition_matrix must have shape ({size}, {size}).")
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("transition_matrix must contain finite non-negative entries.")
    row_totals = result.sum(axis=1)
    if np.any(row_totals <= 0.0):
        raise ValueError("every transition_matrix row must have positive total.")
    result /= row_totals[:, None]
    result.setflags(write=False)
    return result


def _structural_log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    result = np.full(probabilities.shape, -math.inf, dtype=np.float64)
    positive = probabilities > 0.0
    result[positive] = np.log(probabilities[positive])
    result.setflags(write=False)
    return result


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
        self.states = tuple(states)
        self.k = len(self.states)
        if self.k == 0 or any(not isinstance(state, TemporalGraphGrammarDistribution) for state in self.states):
            raise ValueError("states must be a nonempty sequence of temporal graph grammar distributions.")
        first_motif = self.states[0].motif
        if any(
            state.motif.bins != first_motif.bins or state.directed != self.states[0].directed
            for state in self.states[1:]
        ):
            raise ValueError("all latent states must use the same motif partition and directedness.")
        ip_source = np.ones(self.k) if initial_probs is None else initial_probs
        tm_source = np.ones((self.k, self.k)) if transition_matrix is None else transition_matrix
        self.initial_probs = _normalized_probability_vector(ip_source, self.k, name="initial_probs")
        self.transition_matrix = _normalized_transition_matrix(tm_source, self.k)
        self.log_init = _structural_log_probabilities(self.initial_probs)
        self.log_trans = _structural_log_probabilities(self.transition_matrix)
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
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
        snaps = self.dist_to_encoder().seq_encode([x])[0]
        if len(snaps) < 2:
            return 0.0
        return _grammar_forward_backward(self._emission_logb(snaps), self.log_init, self.log_trans)[0]

    def decode(self, x: Sequence[Any]) -> list:
        """Viterbi: the most likely regime governing each transition."""
        snaps = self.dist_to_encoder().seq_encode([x])[0]
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
        if not np.any(np.isfinite(v[-1])):
            raise ValueError("cannot decode a zero-probability temporal graph sequence.")
        path = [int(v[-1].argmax())]
        for t in range(t_steps - 1, 0, -1):
            path.append(int(ptr[t][path[-1]]))
        return path[::-1]

    def seq_encode(self, x: Sequence[Any]) -> Sequence[Any]:
        """Validate and encode latent-regime graph observations."""
        return self.dist_to_encoder().seq_encode(x)

    def seq_log_density(self, x: Sequence[Any]) -> np.ndarray:
        """Score a batch of latent-regime graph observations."""
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> LatentTemporalGraphGrammarSampler:
        """Return a sampler for latent-regime graph sequences."""
        return LatentTemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> LatentTemporalGraphGrammarEstimator:
        """Return the Baum-Welch estimator for the latent-regime grammar."""
        return LatentTemporalGraphGrammarEstimator(
            [st.estimator(pseudo_count=pseudo_count) for st in self.states],
            pseudo_count=pseudo_count,
            name=self.name,
        )

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the validated graph encoder."""
        return TemporalGraphGrammarDataEncoder(self.states[0].directed)


class LatentTemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for regime-switching temporal graph grammars."""

    def __init__(self, dist: LatentTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.sub = [st.sampler(self.rng.randint(2**31)) for st in dist.states]

    def sample_one(self, num_steps: int = 10, seed_graph: np.ndarray | None = None, n_init: int = 5) -> list:
        """Draw one latent-regime graph sequence."""
        d = self.dist
        num_steps = _exact_nonnegative_int(num_steps, name="num_steps")
        n_init = _exact_nonnegative_int(n_init, name="n_init")
        if seed_graph is None:
            adj = np.zeros((n_init, n_init))
        else:
            canonical = _binarize(seed_graph, directed=d.states[0].directed)
            adj = canonical.toarray() if sp.issparse(canonical) else canonical
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
        sample_size = _exact_nonnegative_int(size, name="size")
        return [self.sample_one(**kw) for _ in range(sample_size)]


class LatentTemporalGraphGrammarStatistics(NamedTuple):
    schema_version: int
    bins: tuple[int, ...]
    directed: bool
    num_states: int
    init_counts: np.ndarray
    trans_counts: np.ndarray
    state_values: tuple[Any, ...]
    accepted_weight: float
    rejected_weight: float
    transition_weight: float


def _validate_latent_temporal_statistics(
    value: Any,
    *,
    bins: tuple[int, ...],
    directed: bool,
    num_states: int,
) -> LatentTemporalGraphGrammarStatistics:
    if not isinstance(value, LatentTemporalGraphGrammarStatistics) or value.schema_version != 1:
        raise ValueError("latent temporal graph statistics must use schema version 1.")
    if value.bins != bins or value.directed != directed or value.num_states != num_states:
        raise ValueError("latent temporal graph statistics use an incompatible model schema.")
    init_counts = np.array(value.init_counts, dtype=np.float64, copy=True)
    trans_counts = np.array(value.trans_counts, dtype=np.float64, copy=True)
    state_values = tuple(value.state_values)
    if (
        init_counts.shape != (num_states,)
        or trans_counts.shape != (num_states, num_states)
        or len(state_values) != num_states
    ):
        raise ValueError("latent temporal graph statistics have incompatible state dimensions.")
    accepted_weight = float(value.accepted_weight)
    rejected_weight = float(value.rejected_weight)
    transition_weight = float(value.transition_weight)
    if (
        np.any(~np.isfinite(init_counts))
        or np.any(init_counts < 0.0)
        or np.any(~np.isfinite(trans_counts))
        or np.any(trans_counts < 0.0)
        or any(
            not np.isfinite(component) or component < 0.0
            for component in (accepted_weight, rejected_weight, transition_weight)
        )
    ):
        raise ValueError("latent temporal graph statistics must be finite and non-negative.")
    if not np.isclose(float(init_counts.sum()), accepted_weight, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("latent initial counts must sum to accepted_weight.")
    expected_transitions = transition_weight - accepted_weight
    if expected_transitions < -1.0e-10 or not np.isclose(
        float(trans_counts.sum()),
        max(0.0, expected_transitions),
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise ValueError("latent transition counts are incoherent with accepted transition weight.")
    return LatentTemporalGraphGrammarStatistics(
        1,
        bins,
        directed,
        num_states,
        init_counts,
        trans_counts,
        state_values,
        accepted_weight,
        rejected_weight,
        transition_weight,
    )


class LatentTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for latent-regime graph grammar EM sufficient statistics."""

    def __init__(
        self,
        k: int,
        state_accs: Sequence[Any],
        bins: tuple[int, ...],
        directed: bool,
        factory: LatentTemporalGraphGrammarAccumulatorFactory,
    ) -> None:
        self.k = k
        self.state_accs = list(state_accs)
        self.bins = bins
        self.directed = directed
        self.factory = factory
        self.init_counts = np.zeros(k, dtype=np.float64)
        self.trans_counts = np.zeros((k, k), dtype=np.float64)
        self.accepted_weight = 0.0
        self.rejected_weight = 0.0
        self.transition_weight = 0.0

    def _accumulate(self, snaps: list, weight: float, gamma: np.ndarray, xi: np.ndarray, estimate: Any) -> None:
        self.init_counts += weight * gamma[0]
        self.accepted_weight += weight
        self.transition_weight += weight * (len(snaps) - 1)
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
        if not isinstance(estimate, LatentTemporalGraphGrammarDistribution) or estimate.k != self.k:
            raise ValueError("latent temporal updates require a compatible current distribution.")
        snaps = list(TemporalGraphGrammarDataEncoder(self.directed).seq_encode([x])[0])
        if len(snaps) < 2:
            raise ValueError("latent temporal estimation requires at least one graph transition.")
        checked_weight = _finite_nonnegative(weight, name="weight")
        log_b = estimate._emission_logb(snaps)
        _, gamma, xi = _grammar_forward_backward(log_b, estimate.log_init, estimate.log_trans)
        pending = self.factory.make()
        if gamma is None:
            pending.rejected_weight = checked_weight
        else:
            pending._accumulate(snaps, checked_weight, gamma, xi, estimate)
        self.combine(pending.value())

    def initialize(self, x: Sequence[Any], weight: float, rng: RandomState | None) -> None:
        """Initialize latent-regime sufficient statistics with random soft assignments."""
        snaps = list(TemporalGraphGrammarDataEncoder(self.directed).seq_encode([x])[0])
        if len(snaps) < 2:
            raise ValueError("latent temporal initialization requires at least one graph transition.")
        checked_weight = _finite_nonnegative(weight, name="weight")
        rng = rng if rng is not None else RandomState(0)
        t_steps = len(snaps) - 1
        gamma = rng.dirichlet(np.ones(self.k), size=t_steps)  # random soft regime assignment to seed EM
        xi = np.zeros((max(t_steps - 1, 0), self.k, self.k))
        for t in range(t_steps - 1):
            xi[t] = np.outer(gamma[t], gamma[t + 1])
        pending = self.factory.make()
        pending._accumulate(snaps, checked_weight, gamma, xi, None)
        self.combine(pending.value())

    def seq_update(self, x: Sequence[Any], weights: np.ndarray, estimate: Any | None) -> None:
        """Accumulate posterior-weighted sufficient statistics from a batch."""
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the latent batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = self.factory.make()
        for observation, weight in zip(x, checked_weights):
            pending.update(observation, float(weight), estimate)
        self.combine(pending.value())

    def seq_initialize(self, x: Sequence[Any], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted batch."""
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the latent batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = self.factory.make()
        for observation, weight in zip(x, checked_weights):
            pending.initialize(observation, float(weight), rng)
        self.combine(pending.value())

    def combine(self, suff_stat: LatentTemporalGraphGrammarStatistics) -> LatentTemporalGraphGrammarAccumulator:
        """Merge serialized latent-regime sufficient statistics."""
        checked = _validate_latent_temporal_statistics(
            suff_stat,
            bins=self.bins,
            directed=self.directed,
            num_states=self.k,
        )
        self.init_counts += checked.init_counts
        self.trans_counts += checked.trans_counts
        for acc, state_value in zip(self.state_accs, checked.state_values):
            acc.combine(state_value)
        self.accepted_weight += checked.accepted_weight
        self.rejected_weight += checked.rejected_weight
        self.transition_weight += checked.transition_weight
        return self

    def value(self) -> LatentTemporalGraphGrammarStatistics:
        """Return serialized latent-regime sufficient statistics."""
        return LatentTemporalGraphGrammarStatistics(
            1,
            self.bins,
            self.directed,
            self.k,
            self.init_counts.copy(),
            self.trans_counts.copy(),
            tuple(acc.value() for acc in self.state_accs),
            self.accepted_weight,
            self.rejected_weight,
            self.transition_weight,
        )

    def from_value(self, x: LatentTemporalGraphGrammarStatistics) -> LatentTemporalGraphGrammarAccumulator:
        """Restore accumulator state from serialized sufficient statistics."""
        fresh = self.factory.make()
        fresh.combine(x)
        self.init_counts = fresh.init_counts
        self.trans_counts = fresh.trans_counts
        self.state_accs = fresh.state_accs
        self.accepted_weight = fresh.accepted_weight
        self.rejected_weight = fresh.rejected_weight
        self.transition_weight = fresh.transition_weight
        return self

    def key_merge(self, stats_dict: dict) -> None:
        """Merge keyed sufficient statistics; unused for this accumulator."""
        pass

    def key_replace(self, stats_dict: dict) -> None:
        """Replace keyed sufficient statistics; unused for this accumulator."""
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        """Return the encoder associated with this accumulator."""
        return TemporalGraphGrammarDataEncoder(self.directed)


class LatentTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for latent temporal graph grammar accumulators."""

    def __init__(
        self,
        k: int,
        state_factories: Sequence[Any],
        bins: tuple[int, ...],
        directed: bool,
    ) -> None:
        self.k = k
        self.state_factories = list(state_factories)
        self.bins = bins
        self.directed = directed

    def make(self) -> LatentTemporalGraphGrammarAccumulator:
        """Create a fresh latent-regime accumulator."""
        return LatentTemporalGraphGrammarAccumulator(
            self.k,
            [factory.make() for factory in self.state_factories],
            self.bins,
            self.directed,
            self,
        )


class LatentTemporalGraphGrammarEstimator(ParameterEstimator):
    """EM (Baum-Welch) for the regime-switching grammar: forward-backward E-step, per-regime weighted M-step."""

    def __init__(
        self, state_estimators: Sequence[Any], pseudo_count: float | None = None, name: str | None = None
    ) -> None:
        self.state_estimators = tuple(state_estimators)
        self.k = len(self.state_estimators)
        if self.k == 0 or any(
            not hasattr(estimator, "motif") or not isinstance(estimator.motif, CommonNeighbourMotif)
            for estimator in self.state_estimators
        ):
            raise ValueError("state_estimators must be a nonempty sequence with declared temporal motifs.")
        first_motif = self.state_estimators[0].motif
        if any(
            estimator.motif.bins != first_motif.bins or estimator.motif.directed != first_motif.directed
            for estimator in self.state_estimators[1:]
        ):
            raise ValueError("all state estimators must use the same motif partition and directedness.")
        self.bins = first_motif.bins
        self.directed = first_motif.directed
        self.pseudo_count = None if pseudo_count is None else _finite_nonnegative(pseudo_count, name="pseudo_count")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string or None.")
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> LatentTemporalGraphGrammarAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return LatentTemporalGraphGrammarAccumulatorFactory(
            self.k,
            [estimator.accumulator_factory() for estimator in self.state_estimators],
            self.bins,
            self.directed,
        )

    def estimate(
        self,
        nobs: float | None,
        suff_stat: LatentTemporalGraphGrammarStatistics,
    ) -> LatentTemporalGraphGrammarDistribution:
        """Estimate regime priors, transitions, and state grammars from EM statistics."""
        checked = _validate_latent_temporal_statistics(
            suff_stat,
            bins=self.bins,
            directed=self.directed,
            num_states=self.k,
        )
        if checked.rejected_weight > 0.0:
            raise ValueError(
                "cannot estimate a latent temporal grammar after rejecting zero-probability evidence "
                f"with total weight {checked.rejected_weight}."
            )
        if checked.accepted_weight <= 0.0:
            raise ValueError("cannot estimate a latent temporal grammar without accepted transition evidence.")
        pc = 0.0 if self.pseudo_count is None else float(self.pseudo_count)
        ip = checked.init_counts + pc
        if float(ip.sum()) <= 0.0:
            raise ValueError("latent initial-state counts have no estimable mass.")
        tm = checked.trans_counts + pc
        if np.any(tm.sum(axis=1) <= 0.0):
            raise ValueError("every latent transition row requires evidence or a positive pseudo_count.")
        state_weights = checked.init_counts + checked.trans_counts.sum(axis=0)
        states = [
            estimator.estimate(float(state_weights[index]), state_value)
            for index, (estimator, state_value) in enumerate(
                zip(self.state_estimators, checked.state_values)
            )
        ]
        return LatentTemporalGraphGrammarDistribution(states, ip, tm, name=self.name)


# --- regime-switching ATTRIBUTES: a latent regime over structure AND node/edge attributes -----------
def _validate_latent_attributed_observation(
    x: Any,
    structure: TemporalGraphGrammarDistribution,
    *,
    has_node_dists: bool,
    has_edge_dists: bool,
) -> tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...]]:
    if not isinstance(x, (tuple, list)) or len(x) != 3:
        raise ValueError(
            "latent attributed observations must be (snapshots, node_features, edge_features)."
        )
    raw_snaps, raw_nodes, raw_edges = x
    if isinstance(raw_snaps, ApproximateTemporalGraphSample):
        raise ValueError("approximate scalable samples must be explicitly unwrapped before attribution.")
    try:
        snaps = tuple(_binarize(snapshot, directed=structure.directed) for snapshot in raw_snaps)
    except TypeError as exc:
        raise ValueError("latent attributed snapshots must be a sequence.") from exc
    if not snaps:
        raise ValueError("latent attributed observations must contain at least one snapshot.")
    structure.log_density(snaps)
    transition_count = len(snaps) - 1

    def _groups(raw: Any, *, label: str, enabled: bool) -> tuple[tuple[Any, ...], ...]:
        try:
            groups = tuple(tuple(group) for group in raw)
        except TypeError as exc:
            raise ValueError(f"{label} must be a sequence of per-transition record sequences.") from exc
        if enabled:
            if len(groups) != transition_count:
                raise ValueError(f"{label} must contain exactly one record group per transition.")
        elif groups:
            raise ValueError(f"{label} must be empty when its regime distributions are not configured.")
        return groups

    node_features = _groups(raw_nodes, label="node_features", enabled=has_node_dists)
    edge_features = _groups(raw_edges, label="edge_features", enabled=has_edge_dists)
    for transition, (previous, current) in enumerate(zip(snaps, snaps[1:])):
        new_nodes = current.shape[0] - previous.shape[0]
        if new_nodes < 0:
            raise ValueError("latent attributed temporal graphs do not support node removal.")
        if has_node_dists and len(node_features[transition]) != new_nodes:
            raise ValueError(
                "node feature group %d must contain exactly one record per added node." % transition
            )
        num_added = len(_edge_diff(previous, current, structure.directed)[0])
        if has_edge_dists and len(edge_features[transition]) != num_added:
            raise ValueError(
                "edge feature group %d must contain exactly one record per added edge." % transition
            )
    return snaps, node_features, edge_features


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
        core = LatentTemporalGraphGrammarDistribution(
            structures,
            initial_probs,
            transition_matrix,
            name=name,
        )
        self.structures = core.states
        self.k = core.k

        def _children(value: Sequence[Any] | None, label: str) -> tuple[Any, ...] | None:
            if value is None:
                return None
            result = tuple(value)
            if len(result) != self.k or any(
                not isinstance(distribution, SequenceEncodableProbabilityDistribution)
                for distribution in result
            ):
                raise ValueError(f"{label} must contain exactly one sequence-encodable distribution per regime.")
            return result

        self.node_dists = _children(node_dists, "node_dists")
        self.edge_dists = _children(edge_dists, "edge_dists")
        self.initial_probs = core.initial_probs
        self.transition_matrix = core.transition_matrix
        self.log_init = core.log_init
        self.log_trans = core.log_trans
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
        snaps, node_features, edge_features = _validate_latent_attributed_observation(
            x,
            self.structures[0],
            has_node_dists=self.node_dists is not None,
            has_edge_dists=self.edge_dists is not None,
        )
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
        validated = _validate_latent_attributed_observation(
            x,
            self.structures[0],
            has_node_dists=self.node_dists is not None,
            has_edge_dists=self.edge_dists is not None,
        )
        snaps = validated[0]
        if len(snaps) < 2:
            return 0.0
        return _grammar_forward_backward(self._emission_logb(validated), self.log_init, self.log_trans)[0]

    def decode(self, x: tuple) -> list:
        """Viterbi: the most likely regime governing each transition (jointly explaining structure+attrs)."""
        validated = _validate_latent_attributed_observation(
            x,
            self.structures[0],
            has_node_dists=self.node_dists is not None,
            has_edge_dists=self.edge_dists is not None,
        )
        log_b = self._emission_logb(validated)
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
        if not np.any(np.isfinite(v[-1])):
            raise ValueError("cannot decode a zero-probability latent attributed sequence.")
        path = [int(v[-1].argmax())]
        for t in range(t_steps - 1, 0, -1):
            path.append(int(ptr[t][path[-1]]))
        return path[::-1]

    def seq_encode(self, x: Sequence[Any]) -> Sequence[Any]:
        """Validate and encode attributed latent-regime observations."""
        return self.dist_to_encoder().seq_encode(x)

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
            structures=self.structures,
            node_encoders=None if self.node_dists is None else [d.dist_to_encoder() for d in self.node_dists],
            edge_encoders=None if self.edge_dists is None else [d.dist_to_encoder() for d in self.edge_dists],
            pseudo_count=pseudo_count,
            name=self.name,
        )

    def dist_to_encoder(self) -> LatentAttributedTemporalGraphGrammarDataEncoder:
        """Return the event-aligned latent attributed encoder."""
        return LatentAttributedTemporalGraphGrammarDataEncoder(
            self.structures[0],
            self.node_dists is not None,
            self.edge_dists is not None,
        )


class LatentAttributedTemporalGraphGrammarSampler(DistributionSampler):
    """Sampler for regime-switching attributed temporal graph grammars."""

    def __init__(self, dist: LatentAttributedTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.sub = [st.sampler(self.rng.randint(2**31)) for st in dist.structures]

    def sample_one(self, num_steps: int = 10, seed_graph: np.ndarray | None = None, n_init: int = 5) -> tuple:
        """Draw one attributed latent-regime graph observation."""
        d = self.dist
        num_steps = _exact_nonnegative_int(num_steps, name="num_steps")
        n_init = _exact_nonnegative_int(n_init, name="n_init")
        if seed_graph is None:
            adj = np.zeros((n_init, n_init))
        else:
            canonical = _binarize(seed_graph, directed=d.structures[0].directed)
            adj = canonical.toarray() if sp.issparse(canonical) else canonical
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
        sample_size = _exact_nonnegative_int(size, name="size")
        return [self.sample_one(**kw) for _ in range(sample_size)]


class LatentAttributedTemporalGraphGrammarStatistics(NamedTuple):
    schema_version: int
    bins: tuple[int, ...]
    directed: bool
    num_states: int
    init_counts: np.ndarray
    trans_counts: np.ndarray
    structure_values: tuple[Any, ...]
    node_values: tuple[Any, ...] | None
    edge_values: tuple[Any, ...] | None
    node_weights: np.ndarray
    edge_weights: np.ndarray
    accepted_weight: float
    rejected_weight: float
    transition_weight: float


def _validate_latent_attributed_statistics(
    value: Any,
    *,
    bins: tuple[int, ...],
    directed: bool,
    num_states: int,
    has_node_models: bool,
    has_edge_models: bool,
) -> LatentAttributedTemporalGraphGrammarStatistics:
    if not isinstance(value, LatentAttributedTemporalGraphGrammarStatistics) or value.schema_version != 1:
        raise ValueError("latent attributed temporal statistics must use schema version 1.")
    if value.bins != bins or value.directed != directed or value.num_states != num_states:
        raise ValueError("latent attributed temporal statistics use an incompatible model schema.")
    init_counts = np.array(value.init_counts, dtype=np.float64, copy=True)
    trans_counts = np.array(value.trans_counts, dtype=np.float64, copy=True)
    node_weights = np.array(value.node_weights, dtype=np.float64, copy=True)
    edge_weights = np.array(value.edge_weights, dtype=np.float64, copy=True)
    structure_values = tuple(value.structure_values)
    node_values = None if value.node_values is None else tuple(value.node_values)
    edge_values = None if value.edge_values is None else tuple(value.edge_values)
    if (
        init_counts.shape != (num_states,)
        or trans_counts.shape != (num_states, num_states)
        or node_weights.shape != (num_states,)
        or edge_weights.shape != (num_states,)
        or len(structure_values) != num_states
        or (has_node_models and (node_values is None or len(node_values) != num_states))
        or (not has_node_models and node_values is not None)
        or (has_edge_models and (edge_values is None or len(edge_values) != num_states))
        or (not has_edge_models and edge_values is not None)
    ):
        raise ValueError("latent attributed temporal statistics have incompatible state dimensions.")
    accepted_weight = float(value.accepted_weight)
    rejected_weight = float(value.rejected_weight)
    transition_weight = float(value.transition_weight)
    arrays = (init_counts, trans_counts, node_weights, edge_weights)
    if any(np.any(~np.isfinite(array)) or np.any(array < 0.0) for array in arrays) or any(
        not np.isfinite(component) or component < 0.0
        for component in (accepted_weight, rejected_weight, transition_weight)
    ):
        raise ValueError("latent attributed temporal statistics must be finite and non-negative.")
    if not np.isclose(init_counts.sum(), accepted_weight, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("latent attributed initial counts must sum to accepted_weight.")
    if not np.isclose(
        trans_counts.sum(),
        max(0.0, transition_weight - accepted_weight),
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise ValueError("latent attributed transition counts are incoherent with transition_weight.")
    if not has_node_models and np.any(node_weights != 0.0):
        raise ValueError("node weights require configured node models.")
    if not has_edge_models and np.any(edge_weights != 0.0):
        raise ValueError("edge weights require configured edge models.")
    return LatentAttributedTemporalGraphGrammarStatistics(
        1,
        bins,
        directed,
        num_states,
        init_counts,
        trans_counts,
        structure_values,
        node_values,
        edge_values,
        node_weights,
        edge_weights,
        accepted_weight,
        rejected_weight,
        transition_weight,
    )


class LatentAttributedTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    """Fail-closed EM accumulator with event-aligned attribute statistics."""

    def __init__(
        self,
        struct_accs: Sequence[Any],
        node_accs: Sequence[Any] | None,
        edge_accs: Sequence[Any] | None,
        factory: LatentAttributedTemporalGraphGrammarAccumulatorFactory,
    ) -> None:
        self.factory = factory
        self.k = factory.k
        self.struct_accs = list(struct_accs)
        self.node_accs = None if node_accs is None else list(node_accs)
        self.edge_accs = None if edge_accs is None else list(edge_accs)
        self.init_counts = np.zeros(self.k)
        self.trans_counts = np.zeros((self.k, self.k))
        self.node_weights = np.zeros(self.k)
        self.edge_weights = np.zeros(self.k)
        self.accepted_weight = 0.0
        self.rejected_weight = 0.0
        self.transition_weight = 0.0

    def _accumulate(
        self,
        x: tuple,
        weight: float,
        gamma: np.ndarray,
        xi: np.ndarray,
        estimate: Any,
        *,
        initialize: bool,
        rng: RandomState | None,
    ) -> None:
        snaps, node_features, edge_features = x
        self.init_counts += weight * gamma[0]
        self.accepted_weight += weight
        self.transition_weight += weight * (len(snaps) - 1)
        if xi.shape[0]:
            self.trans_counts += weight * xi.sum(axis=0)
        for state_index in range(self.k):
            structure_estimate = None if estimate is None else estimate.structures[state_index]
            for transition in range(len(snaps) - 1):
                posterior_weight = weight * gamma[transition, state_index]
                if posterior_weight <= 0.0:
                    continue
                structure_obs = [snaps[transition], snaps[transition + 1]]
                if initialize:
                    self.struct_accs[state_index].initialize(structure_obs, posterior_weight, rng)
                else:
                    self.struct_accs[state_index].update(
                        structure_obs,
                        posterior_weight,
                        structure_estimate,
                    )
                if self.node_accs is not None and node_features[transition]:
                    records = node_features[transition]
                    encoder = (
                        self.factory.node_encoders[state_index]
                        if estimate is None
                        else estimate.node_dists[state_index].dist_to_encoder()
                    )
                    encoded = encoder.seq_encode(records)
                    weights = np.full(len(records), posterior_weight)
                    if initialize:
                        self.node_accs[state_index].seq_initialize(encoded, weights, rng)
                    else:
                        self.node_accs[state_index].seq_update(
                            encoded,
                            weights,
                            estimate.node_dists[state_index],
                        )
                    self.node_weights[state_index] += posterior_weight * len(records)
                if self.edge_accs is not None and edge_features[transition]:
                    records = edge_features[transition]
                    encoder = (
                        self.factory.edge_encoders[state_index]
                        if estimate is None
                        else estimate.edge_dists[state_index].dist_to_encoder()
                    )
                    encoded = encoder.seq_encode(records)
                    weights = np.full(len(records), posterior_weight)
                    if initialize:
                        self.edge_accs[state_index].seq_initialize(encoded, weights, rng)
                    else:
                        self.edge_accs[state_index].seq_update(
                            encoded,
                            weights,
                            estimate.edge_dists[state_index],
                        )
                    self.edge_weights[state_index] += posterior_weight * len(records)

    def update(self, x: tuple, weight: float, estimate: Any | None) -> None:
        if not isinstance(estimate, LatentAttributedTemporalGraphGrammarDistribution) or estimate.k != self.k:
            raise ValueError("latent attributed updates require a compatible current distribution.")
        validated = _validate_latent_attributed_observation(
            x,
            self.factory.structures[0],
            has_node_dists=self.node_accs is not None,
            has_edge_dists=self.edge_accs is not None,
        )
        if len(validated[0]) < 2:
            raise ValueError("latent attributed estimation requires at least one graph transition.")
        checked_weight = _finite_nonnegative(weight, name="weight")
        _, gamma, xi = _grammar_forward_backward(
            estimate._emission_logb(validated),
            estimate.log_init,
            estimate.log_trans,
        )
        pending = self.factory.make()
        if gamma is None:
            pending.rejected_weight = checked_weight
        else:
            pending._accumulate(
                validated,
                checked_weight,
                gamma,
                xi,
                estimate,
                initialize=False,
                rng=None,
            )
        self.combine(pending.value())

    def initialize(self, x: tuple, weight: float, rng: RandomState | None) -> None:
        validated = _validate_latent_attributed_observation(
            x,
            self.factory.structures[0],
            has_node_dists=self.node_accs is not None,
            has_edge_dists=self.edge_accs is not None,
        )
        if len(validated[0]) < 2:
            raise ValueError("latent attributed initialization requires at least one graph transition.")
        checked_weight = _finite_nonnegative(weight, name="weight")
        rng = rng if rng is not None else RandomState(0)
        transition_count = len(validated[0]) - 1
        gamma = rng.dirichlet(np.ones(self.k), size=transition_count)
        xi = np.zeros((max(transition_count - 1, 0), self.k, self.k))
        for transition in range(transition_count - 1):
            xi[transition] = np.outer(gamma[transition], gamma[transition + 1])
        pending = self.factory.make()
        pending._accumulate(
            validated,
            checked_weight,
            gamma,
            xi,
            None,
            initialize=True,
            rng=rng,
        )
        self.combine(pending.value())

    def seq_update(self, x: Sequence[Any], weights: np.ndarray, estimate: Any | None) -> None:
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be one-dimensional and aligned with the attributed latent batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = self.factory.make()
        for observation, weight in zip(x, checked_weights):
            pending.update(observation, float(weight), estimate)
        self.combine(pending.value())

    def seq_initialize(self, x: Sequence[Any], weights: np.ndarray, rng: RandomState | None) -> None:
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be one-dimensional and aligned with the attributed latent batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = self.factory.make()
        for observation, weight in zip(x, checked_weights):
            pending.initialize(observation, float(weight), rng)
        self.combine(pending.value())

    def combine(
        self,
        suff_stat: LatentAttributedTemporalGraphGrammarStatistics,
    ) -> LatentAttributedTemporalGraphGrammarAccumulator:
        checked = _validate_latent_attributed_statistics(
            suff_stat,
            bins=self.factory.bins,
            directed=self.factory.directed,
            num_states=self.k,
            has_node_models=self.node_accs is not None,
            has_edge_models=self.edge_accs is not None,
        )
        self.init_counts += checked.init_counts
        self.trans_counts += checked.trans_counts
        for accumulator, state_value in zip(self.struct_accs, checked.structure_values):
            accumulator.combine(state_value)
        if self.node_accs is not None:
            for accumulator, state_value in zip(self.node_accs, checked.node_values):
                accumulator.combine(state_value)
        if self.edge_accs is not None:
            for accumulator, state_value in zip(self.edge_accs, checked.edge_values):
                accumulator.combine(state_value)
        self.node_weights += checked.node_weights
        self.edge_weights += checked.edge_weights
        self.accepted_weight += checked.accepted_weight
        self.rejected_weight += checked.rejected_weight
        self.transition_weight += checked.transition_weight
        return self

    def value(self) -> LatentAttributedTemporalGraphGrammarStatistics:
        return LatentAttributedTemporalGraphGrammarStatistics(
            1,
            self.factory.bins,
            self.factory.directed,
            self.k,
            self.init_counts.copy(),
            self.trans_counts.copy(),
            tuple(accumulator.value() for accumulator in self.struct_accs),
            None
            if self.node_accs is None
            else tuple(accumulator.value() for accumulator in self.node_accs),
            None
            if self.edge_accs is None
            else tuple(accumulator.value() for accumulator in self.edge_accs),
            self.node_weights.copy(),
            self.edge_weights.copy(),
            self.accepted_weight,
            self.rejected_weight,
            self.transition_weight,
        )

    def from_value(
        self,
        x: LatentAttributedTemporalGraphGrammarStatistics,
    ) -> LatentAttributedTemporalGraphGrammarAccumulator:
        fresh = self.factory.make()
        fresh.combine(x)
        self.__dict__.update(fresh.__dict__)
        return self

    def key_merge(self, stats_dict: dict) -> None:
        pass

    def key_replace(self, stats_dict: dict) -> None:
        pass

    def acc_to_encoder(self) -> LatentAttributedTemporalGraphGrammarDataEncoder:
        return self.factory.encoder


class LatentAttributedTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(
        self,
        structures: Sequence[TemporalGraphGrammarDistribution],
        struct_factories: Sequence[Any],
        node_factories: Sequence[Any] | None,
        edge_factories: Sequence[Any] | None,
        node_encoders: Sequence[Any] | None,
        edge_encoders: Sequence[Any] | None,
    ) -> None:
        self.structures = tuple(structures)
        self.k = len(self.structures)
        self.bins = self.structures[0].motif.bins
        self.directed = self.structures[0].directed
        self.struct_factories = tuple(struct_factories)
        self.node_factories = None if node_factories is None else tuple(node_factories)
        self.edge_factories = None if edge_factories is None else tuple(edge_factories)
        self.node_encoders = None if node_encoders is None else tuple(node_encoders)
        self.edge_encoders = None if edge_encoders is None else tuple(edge_encoders)
        self.encoder = LatentAttributedTemporalGraphGrammarDataEncoder(
            self.structures[0],
            self.node_factories is not None,
            self.edge_factories is not None,
        )

    def make(self) -> LatentAttributedTemporalGraphGrammarAccumulator:
        return LatentAttributedTemporalGraphGrammarAccumulator(
            [factory.make() for factory in self.struct_factories],
            None if self.node_factories is None else [factory.make() for factory in self.node_factories],
            None if self.edge_factories is None else [factory.make() for factory in self.edge_factories],
            self,
        )


class LatentAttributedTemporalGraphGrammarEstimator(ParameterEstimator):
    def __init__(
        self,
        structure_estimators: Sequence[Any],
        node_estimators: Sequence[Any] | None = None,
        edge_estimators: Sequence[Any] | None = None,
        *,
        structures: Sequence[TemporalGraphGrammarDistribution],
        node_encoders: Sequence[Any] | None = None,
        edge_encoders: Sequence[Any] | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
    ) -> None:
        self.structures = tuple(structures)
        core = LatentTemporalGraphGrammarEstimator(structure_estimators, pseudo_count, name)
        self.structure_estimators = core.state_estimators
        self.k = core.k
        if len(self.structures) != self.k:
            raise ValueError("structures must contain exactly one prototype per structure estimator.")

        def _children(value: Sequence[Any] | None, label: str) -> tuple[Any, ...] | None:
            if value is None:
                return None
            result = tuple(value)
            if len(result) != self.k:
                raise ValueError(f"{label} must contain exactly one estimator per regime.")
            return result

        self.node_estimators = _children(node_estimators, "node_estimators")
        self.edge_estimators = _children(edge_estimators, "edge_estimators")
        self.node_encoders = _children(node_encoders, "node_encoders")
        self.edge_encoders = _children(edge_encoders, "edge_encoders")
        if (self.node_estimators is None) != (self.node_encoders is None):
            raise ValueError("node estimators and encoders must be configured together.")
        if (self.edge_estimators is None) != (self.edge_encoders is None):
            raise ValueError("edge estimators and encoders must be configured together.")
        self.bins = core.bins
        self.directed = core.directed
        self.pseudo_count = core.pseudo_count
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> LatentAttributedTemporalGraphGrammarAccumulatorFactory:
        return LatentAttributedTemporalGraphGrammarAccumulatorFactory(
            self.structures,
            [estimator.accumulator_factory() for estimator in self.structure_estimators],
            None
            if self.node_estimators is None
            else [estimator.accumulator_factory() for estimator in self.node_estimators],
            None
            if self.edge_estimators is None
            else [estimator.accumulator_factory() for estimator in self.edge_estimators],
            self.node_encoders,
            self.edge_encoders,
        )

    def estimate(
        self,
        nobs: float | None,
        suff_stat: LatentAttributedTemporalGraphGrammarStatistics,
    ) -> LatentAttributedTemporalGraphGrammarDistribution:
        checked = _validate_latent_attributed_statistics(
            suff_stat,
            bins=self.bins,
            directed=self.directed,
            num_states=self.k,
            has_node_models=self.node_estimators is not None,
            has_edge_models=self.edge_estimators is not None,
        )
        if checked.rejected_weight > 0.0:
            raise ValueError(
                "cannot estimate a latent attributed grammar after rejecting zero-probability evidence."
            )
        if checked.accepted_weight <= 0.0:
            raise ValueError("cannot estimate a latent attributed grammar without accepted transition evidence.")
        pc = 0.0 if self.pseudo_count is None else float(self.pseudo_count)
        ip = checked.init_counts + pc
        tm = checked.trans_counts + pc
        if np.any(tm.sum(axis=1) <= 0.0):
            raise ValueError("every latent transition row requires evidence or a positive pseudo_count.")
        structure_weights = checked.init_counts + checked.trans_counts.sum(axis=0)
        structures = [
            estimator.estimate(float(structure_weights[index]), value)
            for index, (estimator, value) in enumerate(
                zip(self.structure_estimators, checked.structure_values)
            )
        ]
        node_dists = None
        if self.node_estimators is not None:
            if np.any(checked.node_weights <= 0.0):
                raise ValueError("every latent node model requires aligned effective record weight.")
            node_dists = [
                estimator.estimate(float(checked.node_weights[index]), value)
                for index, (estimator, value) in enumerate(
                    zip(self.node_estimators, checked.node_values)
                )
            ]
        edge_dists = None
        if self.edge_estimators is not None:
            if np.any(checked.edge_weights <= 0.0):
                raise ValueError("every latent edge model requires aligned effective record weight.")
            edge_dists = [
                estimator.estimate(float(checked.edge_weights[index]), value)
                for index, (estimator, value) in enumerate(
                    zip(self.edge_estimators, checked.edge_values)
                )
            ]
        return LatentAttributedTemporalGraphGrammarDistribution(
            structures,
            node_dists,
            edge_dists,
            ip,
            tm,
            name=self.name,
        )


class LatentAttributedTemporalGraphGrammarDataEncoder(DataSequenceEncoder):
    def __init__(
        self,
        structure: TemporalGraphGrammarDistribution,
        has_node_dists: bool,
        has_edge_dists: bool,
    ) -> None:
        self.structure = structure
        self.has_node_dists = has_node_dists
        self.has_edge_dists = has_edge_dists

    def seq_encode(self, x: Sequence[Any]) -> tuple[tuple, ...]:
        return tuple(
            _validate_latent_attributed_observation(
                observation,
                self.structure,
                has_node_dists=self.has_node_dists,
                has_edge_dists=self.has_edge_dists,
            )
            for observation in x
        )

    def row_count(self, x: Any) -> int:
        if not isinstance(x, tuple):
            raise ValueError("encoded latent attributed temporal payload must be a tuple.")
        return len(x)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LatentAttributedTemporalGraphGrammarDataEncoder)
            and other.structure.motif.bins == self.structure.motif.bins
            and other.structure.directed == self.structure.directed
            and other.has_node_dists == self.has_node_dists
            and other.has_edge_dists == self.has_edge_dists
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
    its own sufficient statistics -- per-motif add/remove counts, node growth, node-removal count (if
    churning), and (if attributed) attribute means. Regimes are, by the identifiability argument, separated
    in this signature space, so clustering the signatures seeds EM near the true solution instead of at
    random."""
    regimes = getattr(proto, "states", None) or proto.structures
    attributed = hasattr(proto, "node_dists") and proto.node_dists is not None
    edge_attr = hasattr(proto, "edge_dists") and proto.edge_dists is not None
    churning = hasattr(proto, "node_remove_rates")
    snaps = obs[0] if (attributed or edge_attr) else obs
    m = regimes[0].motif.num_motifs
    nf = obs[1] if attributed else None
    ef = obs[2] if edge_attr else None
    out = []
    for t in range(len(snaps) - 1):
        if churning:
            # Churning snapshots are (adjacency, node_ids) tuples -- transition_components (like
            # LatentChurningTemporalGraphGrammarDistribution._emission_logb) needs the identity-aligned
            # surviving subgraphs, not the raw tuples. num_removed is itself a regime-discriminating
            # signal here (the whole point of this variant), so it's folded into the signature too.
            prev_surv, cur_reord, num_removed = _align_by_ids(
                snaps[t][0], snaps[t][1], snaps[t + 1][0], snaps[t + 1][1]
            )
        else:
            prev_surv, cur_reord = snaps[t], snaps[t + 1]
        nn, add_bins, _ac, rem_bins, _rc, valid = regimes[0].transition_components(prev_surv, cur_reord)
        a = np.bincount(add_bins, minlength=m).astype(float) if valid and len(add_bins) else np.zeros(m)
        r = np.bincount(rem_bins, minlength=m).astype(float) if valid and len(rem_bins) else np.zeros(m)
        feat = [*a.tolist(), *r.tolist(), float(nn if valid else 0)]
        if churning:
            feat.append(float(num_removed))
        if attributed:
            feat.append(_records_mean(nf[t]) if nf and t < len(nf) else 0.0)
        if edge_attr:
            feat.append(_records_mean(ef[t]) if ef and t < len(ef) else 0.0)
        out.append(feat)
    dim = 2 * m + 1 + int(churning) + int(attributed) + int(edge_attr)
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, dim))


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
    churning = hasattr(proto, "node_remove_rates")
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
        # Latent/Attributed _accumulate take the observation directly; Churning's takes the
        # already-identity-aligned (prev_surv, cur_reord, num_removed, n_prev) tuples instead
        # (see LatentChurningTemporalGraphGrammarAccumulator._accumulate).
        acc._accumulate(proto._aligned(obs) if churning else obs, 1.0, gamma, xi, None)
    return estimator.estimate(len(data), acc.value())
