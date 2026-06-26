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
name, age, ...) and edge attributes (communication counts, channel, ...) as ordinary pysp distributions
scored as emissions on top of the topology -- the whole thing fits jointly with the full distribution
machinery (mixtures, every leaf family, the numba fusion).

This is the temporal counterpart of the static vertex-/hyperedge-replacement grammars in this package.
Scope: undirected, binary (the attribute models carry the labels); edges add+remove, nodes appended; dense
or sparse. Node *removal* (needs identity tracking across snapshots), directed/weighted topology, and
scalable sampling (rejection-based) are the natural extensions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.random import RandomState

from pysp.stats.compute.pdist import (
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


def _edge_diff(prev: Any, cur: Any) -> tuple:
    """Upper-triangular (added_i, added_j, removed_i, removed_j) between two binary adjacencies.

    ``prev`` is padded to ``cur``'s size; added = edges in cur not in prev, removed = edges in prev not in
    cur. Works for sparse or dense and only ever touches the edges that actually changed."""
    n1 = cur.shape[0]
    pp = _pad(_binarize(prev), n1)
    cc = _binarize(cur)
    if sp.issparse(cur) or sp.issparse(prev):
        d = sp.triu(sp.csr_array(cc) - sp.csr_array(pp), 1).tocoo()
        added = d.data > 0
        removed = d.data < 0
        return d.row[added], d.col[added], d.row[removed], d.col[removed]
    d = np.triu(cc - pp, 1)
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

    def __init__(self, bins: Sequence[int] = (0, 1, 2, 3)) -> None:
        self.bins = tuple(int(b) for b in bins)
        self.names = [f"cn>={self.bins[-1]}" if i == len(self.bins) - 1 else f"cn={b}" for i, b in enumerate(self.bins)]

    @property
    def num_motifs(self) -> int:
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
        if sp.issparse(adj):
            au = sp.triu(adj, 1).tocoo()
            cu = sp.triu(cn, 1).tocsr()
            edge_mask = sp.csr_array((np.ones(au.nnz), (au.row, au.col)), shape=(n, n)) if au.nnz else None
            if on_edges:
                if au.nnz:
                    np.add.at(counts, self._bin(np.asarray(cu[au.row, au.col]).ravel()), 1.0)
            else:
                non_edge_cn = cu if edge_mask is None else (cu - cu.multiply(edge_mask))
                non_edge_cn.eliminate_zeros()
                vals = non_edge_cn.tocoo().data
                counts[0] += n * (n - 1) / 2 - au.nnz - vals.size  # bridges = pairs - edges - wedge non-edges
                if vals.size:
                    np.add.at(counts, self._bin(vals), 1.0)
            csr = cn.tocsr()

            def lookup(ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
                return self._bin(np.asarray(csr[ii, jj]).ravel()) if len(ii) else np.zeros(0, dtype=np.int64)
        else:
            ut = np.triu(np.ones((n, n), dtype=bool), 1)
            sel = ut & ((adj > 0) if on_edges else (adj == 0))
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
        name: str | None = None,
    ) -> None:
        self.motif = motif if motif is not None else CommonNeighbourMotif()
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

    def _transition_log_density(self, prev: Any, cur: Any) -> float:
        """log p(G_t | G_{t-1}): node-growth + an ADD grammar over new edges + a REMOVE grammar over deleted
        edges, each a per-motif Poisson scored against the PREVIOUS graph's structure (so order within a
        step is irrelevant). Works on dense OR sparse adjacencies. Node removal is not modelled -> -inf."""
        n0, n1 = prev.shape[0], cur.shape[0]
        if n1 < n0:  # node removal not modelled
            return float("-inf")
        new_nodes = n1 - n0
        ai, aj, ri, rj = _edge_diff(prev, cur)
        if len(ri) and self.edge_remove_rate <= 0.0:  # a deletion under a no-removal (growth) grammar
            return float("-inf")
        add_cand, add_lookup = self.motif.counts_and_binner(_pad(prev, n1), on_edges=False)
        lp = new_nodes * math.log(self.node_rate + _EPS) - self.node_rate - math.lgamma(new_nodes + 1)
        lp += self._edit_log_density(add_lookup(np.asarray(ai), np.asarray(aj)), self.log_w, self.edge_rate, add_cand)
        if lp == float("-inf"):
            return lp
        rem_cand, rem_lookup = self.motif.counts_and_binner(prev, on_edges=True)
        lp += self._edit_log_density(
            rem_lookup(np.asarray(ri), np.asarray(rj)), self.log_rw, self.edge_remove_rate, rem_cand
        )
        return lp

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
        return x

    def seq_log_density(self, x: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        return np.asarray([self.log_density(seq) for seq in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> TemporalGraphGrammarSampler:
        return TemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> TemporalGraphGrammarEstimator:
        return TemporalGraphGrammarEstimator(self.motif, pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        return TemporalGraphGrammarDataEncoder()


# --- sampler --------------------------------------------------------------------------------------
class TemporalGraphGrammarSampler(DistributionSampler):
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
            # batch, pre-step motif assignment (NO within-step refresh) so the realized motif distribution
            # matches the weights and equals what the scorer reads off the snapshots. Per motif m the edit
            # count is Poisson(rate * w_m) -- the multinomial split of a Poisson(rate) total. Additions and
            # removals both act on the start-of-step graph (disjoint -- non-edges vs edges).
            ut = np.triu(np.ones(adj.shape, dtype=bool), 1)
            add_bins = d.motif.assign(adj, on_edges=False)
            rem_bins = d.motif.assign(adj, on_edges=True)
            toggles = []  # (i, j, value) applied after both grammars are sampled, against the pre-step graph
            for m in range(d.motif.num_motifs):
                ai, aj = np.where((add_bins == m) & ut)
                if ai.shape[0]:
                    ka = min(self.rng.poisson(d.edge_rate * d.motif_weights[m]), ai.shape[0])
                    for idx in self.rng.choice(ai.shape[0], size=ka, replace=False):
                        toggles.append((ai[idx], aj[idx], 1.0))
                ri, rj = np.where((rem_bins == m) & ut)
                if ri.shape[0] and d.edge_remove_rate > 0.0:
                    kr = min(self.rng.poisson(d.edge_remove_rate * d.remove_weights[m]), ri.shape[0])
                    for idx in self.rng.choice(ri.shape[0], size=kr, replace=False):
                        toggles.append((ri[idx], rj[idx], 0.0))
            for i, j, v in toggles:
                adj[i, j] = adj[j, i] = v
            snaps.append(adj.copy())
        return snaps

    def sample(self, size: int | None = None, *, num_steps: int = 10, n_init: int = 5) -> Any:
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
        snaps = list(x)  # adjacencies may be dense ndarrays or scipy.sparse
        for t in range(1, len(snaps)):
            prev, cur = snaps[t - 1], snaps[t]
            ai, aj, ri, rj = _edge_diff(prev, cur)
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
        for seq, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.update(seq, float(w), estimate)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def combine(self, suff_stat: tuple) -> TemporalGraphGrammarAccumulator:
        ac, rc, e, re, n, s = suff_stat
        self.add_counts += ac
        self.rem_counts += rc
        self.edges += e
        self.rem_edges += re
        self.nodes += n
        self.steps += s
        return self

    def value(self) -> tuple:
        return self.add_counts.copy(), self.rem_counts.copy(), self.edges, self.rem_edges, self.nodes, self.steps

    def from_value(self, x: tuple) -> TemporalGraphGrammarAccumulator:
        self.add_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.rem_counts = np.asarray(x[1], dtype=np.float64).copy()
        self.edges, self.rem_edges, self.nodes, self.steps = float(x[2]), float(x[3]), float(x[4]), float(x[5])
        return self

    def key_merge(self, stats_dict: dict) -> None:
        pass

    def key_replace(self, stats_dict: dict) -> None:
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        return TemporalGraphGrammarDataEncoder()


class TemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, motif: CommonNeighbourMotif) -> None:
        self.motif = motif

    def make(self) -> TemporalGraphGrammarAccumulator:
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
        return TemporalGraphGrammarAccumulatorFactory(self.motif)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> TemporalGraphGrammarDistribution:
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
    def seq_encode(self, x: Sequence[Sequence[np.ndarray]]) -> Sequence[Sequence[np.ndarray]]:
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
    ordinary pysp distributions: ``node_dist`` over per-node attribute records (location, name, age, ... --
    typically a ``CompositeDistribution`` of leaves or a mixture) and ``edge_dist`` over per-edge attribute
    records (communication counts, channel, weight, ...). An observation is ``(snapshots, node_features,
    edge_features)``: the adjacency chain, one attribute record per node, and one per added edge. The
    likelihood factorises -- structure x node attributes x edge attributes -- so the attribute models are
    fit (and scored) with the full pysp distribution machinery (mixtures, fusion, all leaf families).
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
        snaps, node_features, edge_features = x
        return (
            self.structure.log_density(snaps)
            + _emission_ll(self.node_dist, node_features)
            + _emission_ll(self.edge_dist, edge_features)
        )

    def seq_encode(self, x: Sequence[tuple]) -> Sequence[tuple]:
        return x

    def seq_log_density(self, x: Sequence[tuple]) -> np.ndarray:
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> LabeledTemporalGraphGrammarSampler:
        return LabeledTemporalGraphGrammarSampler(self, seed)

    def estimator(self, **kw: Any) -> LabeledTemporalGraphGrammarEstimator:
        return LabeledTemporalGraphGrammarEstimator(
            self.structure.estimator(**kw),
            None if self.node_dist is None else self.node_dist.estimator(),
            None if self.edge_dist is None else self.edge_dist.estimator(),
            name=self.name,
        )

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        return TemporalGraphGrammarDataEncoder()


class LabeledTemporalGraphGrammarSampler(DistributionSampler):
    def __init__(self, dist: LabeledTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.struct = dist.structure.sampler(self.rng.randint(2**31))

    def sample_one(self, **kw: Any) -> tuple:
        snaps = self.struct.sample_one(**kw)
        n_final = snaps[-1].shape[0]
        num_added = sum(len(_edge_diff(snaps[t - 1], snaps[t])[0]) for t in range(1, len(snaps)))
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
        if size is None:
            return self.sample_one(**kw)
        return [self.sample_one(**kw) for _ in range(size)]


class LabeledTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(self, structure_acc: Any, node_acc: Any, edge_acc: Any) -> None:
        self.structure_acc = structure_acc
        self.node_acc = node_acc
        self.edge_acc = edge_acc

    def update(self, x: tuple, weight: float, estimate: Any | None) -> None:
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
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.update(obs, float(w), estimate)

    def initialize(self, x: tuple, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple) -> LabeledTemporalGraphGrammarAccumulator:
        s, n, e = suff_stat
        self.structure_acc.combine(s)
        if self.node_acc is not None:
            self.node_acc.combine(n)
        if self.edge_acc is not None:
            self.edge_acc.combine(e)
        return self

    def value(self) -> tuple:
        return (
            self.structure_acc.value(),
            None if self.node_acc is None else self.node_acc.value(),
            None if self.edge_acc is None else self.edge_acc.value(),
        )

    def from_value(self, x: tuple) -> LabeledTemporalGraphGrammarAccumulator:
        self.structure_acc.from_value(x[0])
        if self.node_acc is not None:
            self.node_acc.from_value(x[1])
        if self.edge_acc is not None:
            self.edge_acc.from_value(x[2])
        return self

    def key_merge(self, stats_dict: dict) -> None:
        pass

    def key_replace(self, stats_dict: dict) -> None:
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        return TemporalGraphGrammarDataEncoder()


class LabeledTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, structure_factory: Any, node_factory: Any, edge_factory: Any) -> None:
        self.structure_factory = structure_factory
        self.node_factory = node_factory
        self.edge_factory = edge_factory

    def make(self) -> LabeledTemporalGraphGrammarAccumulator:
        return LabeledTemporalGraphGrammarAccumulator(
            self.structure_factory.make(),
            None if self.node_factory is None else self.node_factory.make(),
            None if self.edge_factory is None else self.edge_factory.make(),
        )


class LabeledTemporalGraphGrammarEstimator(ParameterEstimator):
    def __init__(
        self, structure_estimator: Any, node_estimator: Any = None, edge_estimator: Any = None, name: str | None = None
    ) -> None:
        self.structure_estimator = structure_estimator
        self.node_estimator = node_estimator
        self.edge_estimator = edge_estimator
        self.name = name
        self.keys = None

    def accumulator_factory(self) -> LabeledTemporalGraphGrammarAccumulatorFactory:
        return LabeledTemporalGraphGrammarAccumulatorFactory(
            self.structure_estimator.accumulator_factory(),
            None if self.node_estimator is None else self.node_estimator.accumulator_factory(),
            None if self.edge_estimator is None else self.edge_estimator.accumulator_factory(),
        )

    def estimate(self, nobs: float | None, suff_stat: tuple) -> LabeledTemporalGraphGrammarDistribution:
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
        snaps, types = x
        types = np.asarray(types, dtype=np.int64)
        lp = float(np.sum(self.log_tw[types]))  # node-type likelihood (each node's type ~ Categorical)
        lp += sum(self._transition_log_density(snaps[t - 1], snaps[t], types) for t in range(1, len(snaps)))
        return lp

    def seq_encode(self, x: Sequence[tuple]) -> Sequence[tuple]:
        return x

    def seq_log_density(self, x: Sequence[tuple]) -> np.ndarray:
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> HomophilyTemporalGraphGrammarSampler:
        return HomophilyTemporalGraphGrammarSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> HomophilyTemporalGraphGrammarEstimator:
        return HomophilyTemporalGraphGrammarEstimator(self.M, self.K, self.motif, pseudo_count, self.name)

    def dist_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        return TemporalGraphGrammarDataEncoder()


class HomophilyTemporalGraphGrammarSampler(DistributionSampler):
    def __init__(self, dist: HomophilyTemporalGraphGrammarDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample_one(self, num_steps: int = 8, seed_graph: np.ndarray | None = None, n_init: int = 8) -> tuple:
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
        if size is None:
            return self.sample_one(**kw)
        return [self.sample_one(**kw) for _ in range(size)]


class HomophilyTemporalGraphGrammarAccumulator(SequenceEncodableStatisticAccumulator):
    def __init__(self, M: int, K: int, motif: CommonNeighbourMotif) -> None:
        self.M, self.K, self.motif = M, K, motif
        self.edge_counts = np.zeros((M, K, K), dtype=np.float64)
        self.type_counts = np.zeros(K, dtype=np.float64)
        self.nodes = 0.0
        self.steps = 0.0

    def update(self, x: tuple, weight: float, estimate: Any | None) -> None:
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
        for obs, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.update(obs, float(w), estimate)

    def initialize(self, x: tuple, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, x: Any, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple) -> HomophilyTemporalGraphGrammarAccumulator:
        ec, tc, n, s = suff_stat
        self.edge_counts += ec
        self.type_counts += tc
        self.nodes += n
        self.steps += s
        return self

    def value(self) -> tuple:
        return self.edge_counts.copy(), self.type_counts.copy(), self.nodes, self.steps

    def from_value(self, x: tuple) -> HomophilyTemporalGraphGrammarAccumulator:
        self.edge_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.type_counts = np.asarray(x[1], dtype=np.float64).copy()
        self.nodes, self.steps = float(x[2]), float(x[3])
        return self

    def key_merge(self, stats_dict: dict) -> None:
        pass

    def key_replace(self, stats_dict: dict) -> None:
        pass

    def acc_to_encoder(self) -> TemporalGraphGrammarDataEncoder:
        return TemporalGraphGrammarDataEncoder()


class HomophilyTemporalGraphGrammarAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, M: int, K: int, motif: CommonNeighbourMotif) -> None:
        self.M, self.K, self.motif = M, K, motif

    def make(self) -> HomophilyTemporalGraphGrammarAccumulator:
        return HomophilyTemporalGraphGrammarAccumulator(self.M, self.K, self.motif)


class HomophilyTemporalGraphGrammarEstimator(ParameterEstimator):
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
        return HomophilyTemporalGraphGrammarAccumulatorFactory(self.M, self.K, self.motif)

    def estimate(self, nobs: float | None, suff_stat: tuple) -> HomophilyTemporalGraphGrammarDistribution:
        edge_counts, type_counts, nodes, steps = suff_stat
        rate = np.asarray(edge_counts, dtype=np.float64) / steps if steps > 0 else np.asarray(edge_counts)
        tc = np.asarray(type_counts, dtype=np.float64).copy()
        if self.pseudo_count is not None:
            tc = tc + float(self.pseudo_count)
        type_weights = tc / tc.sum() if tc.sum() > 0 else np.ones(self.K) / self.K
        return HomophilyTemporalGraphGrammarDistribution(
            rate, type_weights, nodes / steps if steps > 0 else 0.0, motif=self.motif, name=self.name
        )
