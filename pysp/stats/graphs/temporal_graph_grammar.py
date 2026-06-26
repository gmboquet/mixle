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

This is the temporal counterpart of the static vertex-/hyperedge-replacement grammars in this package.
Scope (phase 1): undirected, binary, growth-only (edges added, nodes appended). Edge/node *removal*,
directed/weighted graphs, and node attributes are natural extensions.
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

    def assign(self, adj: np.ndarray) -> np.ndarray:
        """Return an (n, n) int array giving the motif bin of every *non-edge* (and -1 on edges/diagonal).

        The common-neighbour count of a candidate edge (i, j) is ``(A @ A)[i, j]``; binning it by
        ``self.bins`` yields its motif. Existing edges and the diagonal are marked -1 (not candidates).
        """
        n = adj.shape[0]
        cn = adj @ adj  # common-neighbour counts
        b = np.searchsorted(self.bins, cn, side="right") - 1  # bin index per pair
        b = np.clip(b, 0, len(self.bins) - 1).astype(np.int64)
        mask = (adj > 0) | np.eye(n, dtype=bool)  # edges + diagonal are not candidate non-edges
        b[mask] = -1
        return b


# --- distribution ---------------------------------------------------------------------------------
class TemporalGraphGrammarDistribution(SequenceEncodableProbabilityDistribution):
    """Distribution over dynamic graphs (sequences of adjacency snapshots) under a motif-edit grammar."""

    def __init__(
        self,
        motif_weights: Sequence[float],
        edge_rate: float = 1.0,
        node_rate: float = 0.0,
        motif: CommonNeighbourMotif | None = None,
        name: str | None = None,
    ) -> None:
        self.motif = motif if motif is not None else CommonNeighbourMotif()
        w = np.asarray(motif_weights, dtype=np.float64)
        if w.shape[0] != self.motif.num_motifs:
            raise ValueError("motif_weights must have one entry per motif bin (%d)." % self.motif.num_motifs)
        self.motif_weights = w / w.sum()
        self.log_w = np.log(np.clip(self.motif_weights, _EPS, None))
        self.edge_rate = float(edge_rate)
        self.node_rate = float(node_rate)
        self.name = name

    def __str__(self) -> str:
        return "TemporalGraphGrammarDistribution(w=%s, edge_rate=%s, node_rate=%s)" % (
            np.array2string(self.motif_weights, precision=3),
            self.edge_rate,
            self.node_rate,
        )

    def _transition_log_density(self, prev: np.ndarray, cur: np.ndarray) -> float:
        """log p(G_t | G_{t-1}): node-growth + edge-count + per-edge motif terms (growth-only)."""
        n0, n1 = prev.shape[0], cur.shape[0]
        if n1 < n0 or not np.all(cur[:n0, :n0] >= prev):  # must be a growth step (no removals)
            return float("-inf")
        new_nodes = n1 - n0
        padded = np.zeros((n1, n1), dtype=prev.dtype)
        padded[:n0, :n0] = prev
        bins = self.motif.assign(padded)  # motif of every non-edge of the (padded) previous graph
        added = np.triu((cur > 0) & (padded == 0), 1)  # new undirected edges
        ii, jj = np.where(added)
        k = ii.shape[0]
        # candidate counts per motif bin (undirected upper triangle), for the uniform-anchor normaliser
        cand = np.zeros(self.motif.num_motifs, dtype=np.float64)
        ut = np.triu(np.ones((n1, n1), dtype=bool), 1)
        for b in range(self.motif.num_motifs):
            cand[b] = float(np.count_nonzero((bins == b) & ut))
        lp = 0.0
        # edge count ~ Poisson(edge_rate); node count ~ Poisson(node_rate)
        lp += k * math.log(self.edge_rate + _EPS) - self.edge_rate - math.lgamma(k + 1)
        lp += new_nodes * math.log(self.node_rate + _EPS) - self.node_rate - math.lgamma(new_nodes + 1)
        for a, b in zip(ii.tolist(), jj.tolist()):
            m = bins[a, b]
            if m < 0 or cand[m] <= 0:
                return float("-inf")
            lp += self.log_w[m] - math.log(cand[m])  # rule weight x uniform anchor among its candidates
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
            # matches the weights and equals what the scorer reads off the snapshots. Per motif m the edge
            # count is Poisson(edge_rate * w_m) -- the multinomial split of a Poisson(edge_rate) total.
            bins = d.motif.assign(adj)  # computed once, relative to the start-of-step graph
            ut = np.triu(np.ones(adj.shape, dtype=bool), 1)
            for m in range(d.motif.num_motifs):
                ai, aj = np.where((bins == m) & ut)
                if ai.shape[0] == 0:
                    continue
                km = min(self.rng.poisson(d.edge_rate * d.motif_weights[m]), ai.shape[0])
                if km <= 0:
                    continue
                for idx in self.rng.choice(ai.shape[0], size=km, replace=False):
                    i, j = ai[idx], aj[idx]
                    adj[i, j] = adj[j, i] = 1.0
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
        self.motif_counts = np.zeros(motif.num_motifs, dtype=np.float64)
        self.edges = 0.0
        self.nodes = 0.0
        self.steps = 0.0

    def update(self, x: Sequence[np.ndarray], weight: float, estimate: Any | None) -> None:
        snaps = [np.asarray(a, dtype=np.float64) for a in x]
        for t in range(1, len(snaps)):
            prev, cur = snaps[t - 1], snaps[t]
            n0, n1 = prev.shape[0], cur.shape[0]
            padded = np.zeros((n1, n1), dtype=prev.dtype)
            padded[:n0, :n0] = prev
            bins = self.motif.assign(padded)
            added = np.triu((cur > 0) & (padded == 0), 1)
            ii, jj = np.where(added)
            for a, b in zip(ii.tolist(), jj.tolist()):
                m = bins[a, b]
                if m >= 0:
                    self.motif_counts[m] += weight
            self.edges += weight * ii.shape[0]
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
        mc, e, n, s = suff_stat
        self.motif_counts += mc
        self.edges += e
        self.nodes += n
        self.steps += s
        return self

    def value(self) -> tuple:
        return self.motif_counts.copy(), self.edges, self.nodes, self.steps

    def from_value(self, x: tuple) -> TemporalGraphGrammarAccumulator:
        self.motif_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.edges, self.nodes, self.steps = float(x[1]), float(x[2]), float(x[3])
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
        motif_counts, edges, nodes, steps = suff_stat
        counts = np.asarray(motif_counts, dtype=np.float64).copy()
        if self.pseudo_count is not None:
            counts = counts + float(self.pseudo_count)
        weights = counts / counts.sum() if counts.sum() > 0 else np.ones(self.motif.num_motifs) / self.motif.num_motifs
        edge_rate = edges / steps if steps > 0 else 1.0
        node_rate = nodes / steps if steps > 0 else 0.0
        return TemporalGraphGrammarDistribution(weights, edge_rate, node_rate, motif=self.motif, name=self.name)


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
