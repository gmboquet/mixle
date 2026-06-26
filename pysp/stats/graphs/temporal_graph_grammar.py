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

This is the temporal counterpart of the static vertex-/hyperedge-replacement grammars in this package.
Scope: undirected, binary; edges add+remove, nodes appended. Node *removal* (needs identity tracking
across snapshots), directed/weighted graphs, and node attributes are natural extensions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
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

    def _edit_log_density(self, ii, jj, bins, log_w, rate, cand) -> float:
        """Shared add/remove term: per-motif Poisson(rate*w_m) x uniform anchor among motif-m candidates.

        Equivalent to Poisson(total; rate) x Multinomial(w) x (1/cand_m per edit). Returns -inf if an edit
        falls outside every motif or its candidate pool is empty (e.g. a removal when rate == 0)."""
        k = ii.shape[0]
        lp = k * math.log(rate + _EPS) - rate - math.lgamma(k + 1)
        for a, b in zip(ii.tolist(), jj.tolist()):
            mtf = bins[a, b]
            if mtf < 0 or cand[mtf] <= 0:
                return float("-inf")
            lp += log_w[mtf] - math.log(cand[mtf])
        return lp

    def _candidate_counts(self, bins: np.ndarray) -> np.ndarray:
        ut = np.triu(np.ones(bins.shape, dtype=bool), 1)
        return np.asarray([float(np.count_nonzero((bins == b) & ut)) for b in range(self.motif.num_motifs)])

    def _transition_log_density(self, prev: np.ndarray, cur: np.ndarray) -> float:
        """log p(G_t | G_{t-1}): node-growth + an ADD grammar over new edges + a REMOVE grammar over deleted
        edges, each a per-motif Poisson scored against the PREVIOUS graph's structure (so order within a
        step is irrelevant). Node removal is not modelled -> -inf."""
        n0, n1 = prev.shape[0], cur.shape[0]
        if n1 < n0:  # node removal not modelled
            return float("-inf")
        new_nodes = n1 - n0
        padded = np.zeros((n1, n1), dtype=prev.dtype)
        padded[:n0, :n0] = prev
        add_bins = self.motif.assign(padded, on_edges=False)  # motif of each non-edge (addition candidate)
        rem_bins = self.motif.assign(prev, on_edges=True)  # motif of each edge (removal candidate)
        ai, aj = np.where(np.triu((cur > 0) & (padded == 0), 1))  # new edges
        ri, rj = np.where(np.triu((prev > 0) & (cur[:n0, :n0] == 0), 1))  # deleted edges
        if ri.shape[0] and self.edge_remove_rate <= 0.0:  # a deletion under a no-removal (growth) grammar
            return float("-inf")
        lp = new_nodes * math.log(self.node_rate + _EPS) - self.node_rate - math.lgamma(new_nodes + 1)
        lp += self._edit_log_density(ai, aj, add_bins, self.log_w, self.edge_rate, self._candidate_counts(add_bins))
        if lp == float("-inf"):
            return lp
        lp += self._edit_log_density(
            ri, rj, rem_bins, self.log_rw, self.edge_remove_rate, self._candidate_counts(rem_bins)
        )
        return lp

    def log_density(self, x: Sequence[np.ndarray]) -> float:
        """Log-density of one dynamic graph: the sum of transition log-densities over the snapshot chain.

        ``x`` is a sequence of binary adjacency matrices (the initial graph is taken as given -- its
        marginal is not modelled, matching how the static grammars treat their start symbol)."""
        snaps = [np.asarray(a, dtype=np.float64) for a in x]
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

    def update(self, x: Sequence[np.ndarray], weight: float, estimate: Any | None) -> None:
        snaps = [np.asarray(a, dtype=np.float64) for a in x]
        for t in range(1, len(snaps)):
            prev, cur = snaps[t - 1], snaps[t]
            n0, n1 = prev.shape[0], cur.shape[0]
            padded = np.zeros((n1, n1), dtype=prev.dtype)
            padded[:n0, :n0] = prev
            add_bins = self.motif.assign(padded, on_edges=False)
            rem_bins = self.motif.assign(prev, on_edges=True)
            ai, aj = np.where(np.triu((cur > 0) & (padded == 0), 1))  # added edges
            ri, rj = np.where(np.triu((prev > 0) & (cur[:n0, :n0] == 0), 1))  # removed edges
            for a, b in zip(ai.tolist(), aj.tolist()):
                if add_bins[a, b] >= 0:
                    self.add_counts[add_bins[a, b]] += weight
            for a, b in zip(ri.tolist(), rj.tolist()):
                if rem_bins[a, b] >= 0:
                    self.rem_counts[rem_bins[a, b]] += weight
            self.edges += weight * ai.shape[0]
            self.rem_edges += weight * ri.shape[0]
            self.nodes += weight * (n1 - n0)
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


__all__ = [
    "CommonNeighbourMotif",
    "TemporalGraphGrammarDistribution",
    "TemporalGraphGrammarSampler",
    "TemporalGraphGrammarEstimator",
    "TemporalGraphGrammarAccumulator",
    "TemporalGraphGrammarAccumulatorFactory",
    "TemporalGraphGrammarDataEncoder",
]
