"""Probabilistic context-free grammar with heterogeneous terminal emissions.

This module implements a small Chomsky-normal-form PCFG whose terminal rules
emit observations through arbitrary ``mixle.stats`` distributions. A terminal
distribution can be scalar-valued, tuple-valued, set-valued, sequence-valued,
or any other sequence-encodable distribution; each terminal rule keeps its own
encoder.

Data type: an observation is a finite sequence of terminal observations. The
grammar supports binary rules ``A -> B C`` and terminal rules ``A -> emission``.
Epsilon productions and unary nonterminal productions are intentionally not
modeled.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import (
    BufferedStream,
    LazyQuantizedEnumerationIndex,
    ProductEnumerator,
    best_first_union,
)
from mixle.inference.fisher import (
    FisherView,
    FixedFisherView,
    Path,
    SufficientStatisticVectorizer,
    _seq_encode_model,
    _structured_values_matrix,
    to_fisher,
)
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)


def _logsumexp_1d(vals: np.ndarray) -> float:
    """Fast log-sum-exp over a small 1-D array.

    scipy.special.logsumexp carries large per-call overhead (dtype promotion, array-API shims).
    The inside/inside-outside dynamic program calls it once per chart cell, so for many short
    parses that overhead dominates; this inline version is numerically equivalent for the 1-D,
    real-valued inputs used here.

    This is a deliberate hot-loop perf fork of the canonical ``mixle.utils.special.logsumexp``; do
    not repoint it -- the per-call overhead of the general version is exactly what this avoids.
    """
    m = vals.max()
    if m == -np.inf:
        return -np.inf
    return float(m + np.log(np.exp(vals - m).sum()))


BinaryRuleInput = dict[Any, Sequence[tuple[Any, Any, float]]] | Sequence[tuple[Any, Any, Any, float]]
TerminalRuleInput = dict[Any, Sequence[tuple[Any, float]]] | Sequence[tuple[Any, Any, float]]
EncodedPCFGData = tuple[np.ndarray, tuple[Any, ...]]


def _iter_binary_rules(binary_rules: BinaryRuleInput | None) -> Iterable[tuple[Any, Any, Any, float]]:
    if binary_rules is None:
        return []
    if isinstance(binary_rules, dict):
        rv = []
        for parent, rules in binary_rules.items():
            for left, right, prob in rules:
                rv.append((parent, left, right, float(prob)))
        return rv
    return [(parent, left, right, float(prob)) for parent, left, right, prob in binary_rules]


def _iter_terminal_rules(terminal_rules: TerminalRuleInput) -> Iterable[tuple[Any, Any, float]]:
    if isinstance(terminal_rules, dict):
        rv = []
        for parent, rules in terminal_rules.items():
            for emission, prob in rules:
                rv.append((parent, emission, float(prob)))
        return rv
    return [(parent, emission, float(prob)) for parent, emission, prob in terminal_rules]


def _log_normalized(values: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(values)


class HeterogeneousPCFGDistribution(SequenceEncodableProbabilityDistribution):
    """CNF PCFG whose terminal productions emit arbitrary distributions.

    Args:
        binary_rules: Either ``{parent: [(left, right, prob), ...]}`` or a flat
            sequence of ``(parent, left, right, prob)`` rules.
        terminal_rules: Either ``{parent: [(emission_dist, prob), ...]}`` or a
            flat sequence of ``(parent, emission_dist, prob)`` rules.
        start: Start nonterminal. If omitted, the first nonterminal is used.
        nonterminals: Optional explicit nonterminal order.
        name: Optional model name.

    Rule probabilities are normalized over all rules sharing a parent.
    """

    def __init__(
        self,
        binary_rules: BinaryRuleInput | None,
        terminal_rules: TerminalRuleInput,
        start: Any | None = None,
        nonterminals: Sequence[Any] | None = None,
        name: str | None = None,
    ) -> None:
        raw_binary = list(_iter_binary_rules(binary_rules))
        raw_terminal = list(_iter_terminal_rules(terminal_rules))
        if len(raw_terminal) == 0:
            raise ValueError("HeterogeneousPCFGDistribution requires at least one terminal rule.")

        nt_list: list[Any] = []
        seen = set()

        def add_nt(nt: Any) -> None:
            if nt not in seen:
                seen.add(nt)
                nt_list.append(nt)

        if nonterminals is not None:
            for nt in nonterminals:
                add_nt(nt)
        if start is not None:
            add_nt(start)
        for parent, left, right, _ in raw_binary:
            add_nt(parent)
            add_nt(left)
            add_nt(right)
        for parent, _, _ in raw_terminal:
            add_nt(parent)

        if len(nt_list) == 0:
            raise ValueError("HeterogeneousPCFGDistribution requires at least one nonterminal.")
        self.nonterminals = nt_list
        self.nt_to_idx = {nt: i for i, nt in enumerate(self.nonterminals)}
        self.num_nonterminals = len(self.nonterminals)
        self.start = self.nonterminals[0] if start is None else start
        if self.start not in self.nt_to_idx:
            raise ValueError("start nonterminal is not present in the grammar.")
        self.start_idx = self.nt_to_idx[self.start]
        self.name = name

        totals = np.zeros(self.num_nonterminals, dtype=np.float64)
        for parent, _, _, prob in raw_binary:
            if prob < 0.0:
                raise ValueError("binary rule probabilities must be non-negative.")
            totals[self.nt_to_idx[parent]] += prob
        for parent, _, prob in raw_terminal:
            if prob < 0.0:
                raise ValueError("terminal rule probabilities must be non-negative.")
            totals[self.nt_to_idx[parent]] += prob

        for i, total in enumerate(totals):
            if total <= 0.0:
                has_rule = any(self.nt_to_idx[parent] == i for parent, _, _, _ in raw_binary)
                has_rule = has_rule or any(self.nt_to_idx[parent] == i for parent, _, _ in raw_terminal)
                if has_rule:
                    raise ValueError("rule probabilities for each active parent must have positive total.")

        self.binary_rules: list[tuple[int, int, int, float]] = []
        self.binary_parents: list[int] = []
        self.binary_left: list[int] = []
        self.binary_right: list[int] = []
        self.binary_probs: list[float] = []
        self.binary_by_parent: list[list[int]] = [[] for _ in range(self.num_nonterminals)]
        for parent, left, right, prob in raw_binary:
            p = self.nt_to_idx[parent]
            l = self.nt_to_idx[left]
            r = self.nt_to_idx[right]
            q = prob / totals[p]
            rule_idx = len(self.binary_rules)
            self.binary_rules.append((p, l, r, q))
            self.binary_parents.append(p)
            self.binary_left.append(l)
            self.binary_right.append(r)
            self.binary_probs.append(q)
            self.binary_by_parent[p].append(rule_idx)

        self.terminal_rules: list[tuple[int, SequenceEncodableProbabilityDistribution, float]] = []
        self.terminal_parents: list[int] = []
        self.terminal_probs: list[float] = []
        self.emissions: list[SequenceEncodableProbabilityDistribution] = []
        self.terminal_by_parent: list[list[int]] = [[] for _ in range(self.num_nonterminals)]
        for parent, emission, prob in raw_terminal:
            p = self.nt_to_idx[parent]
            q = prob / totals[p]
            rule_idx = len(self.terminal_rules)
            self.terminal_rules.append((p, emission, q))
            self.terminal_parents.append(p)
            self.terminal_probs.append(q)
            self.emissions.append(emission)
            self.terminal_by_parent[p].append(rule_idx)

        self.binary_parents = np.asarray(self.binary_parents, dtype=np.int32)
        self.binary_left = np.asarray(self.binary_left, dtype=np.int32)
        self.binary_right = np.asarray(self.binary_right, dtype=np.int32)
        self.binary_probs = np.asarray(self.binary_probs, dtype=np.float64)
        self.log_binary_probs = _log_normalized(self.binary_probs)
        self.terminal_parents = np.asarray(self.terminal_parents, dtype=np.int32)
        self.terminal_probs = np.asarray(self.terminal_probs, dtype=np.float64)
        self.log_terminal_probs = _log_normalized(self.terminal_probs)
        self.num_binary_rules = len(self.binary_rules)
        self.num_terminal_rules = len(self.terminal_rules)

    def __str__(self) -> str:
        nts = repr(self.nonterminals)
        binary = repr(
            [
                (self.nonterminals[p], self.nonterminals[l], self.nonterminals[r], float(q))
                for p, l, r, q in self.binary_rules
            ]
        )
        terminal = repr([(self.nonterminals[p], str(d), float(q)) for p, d, q in self.terminal_rules])
        return (
            "HeterogeneousPCFGDistribution(binary_rules=%s, terminal_rules=%s, start=%s, "
            "nonterminals=%s, name=%s)" % (binary, terminal, repr(self.start), nts, repr(self.name))
        )

    def density(self, x: Sequence[Any]) -> float:
        """Return the probability density or mass at a single observation."""
        return math.exp(self.log_density(x))

    def _terminal_log_density(self, x: Sequence[Any]) -> np.ndarray:
        n = len(x)
        rv = np.empty((n, self.num_terminal_rules), dtype=np.float64)
        with np.errstate(divide="ignore"):
            for r, dist in enumerate(self.emissions):
                rv[:, r] = np.asarray([dist.log_density(xx) for xx in x], dtype=np.float64)
        return rv

    def _inside(self, terminal_log_density: np.ndarray) -> np.ndarray:
        n = terminal_log_density.shape[0]
        k = self.num_nonterminals
        inside = np.full((n, n + 1, k), -np.inf, dtype=np.float64)
        if n == 0:
            return inside

        for i in range(n):
            for parent in range(k):
                rules = self.terminal_by_parent[parent]
                if rules:
                    vals = self.log_terminal_probs[rules] + terminal_log_density[i, rules]
                    inside[i, i + 1, parent] = _logsumexp_1d(vals)

        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span
                for rule_idx in range(self.num_binary_rules):
                    parent = self.binary_parents[rule_idx]
                    left = self.binary_left[rule_idx]
                    right = self.binary_right[rule_idx]
                    vals = inside[i, i + 1 : j, left] + inside[i + 1 : j, j, right]
                    score = _logsumexp_1d(vals)
                    if np.isfinite(score):
                        inside[i, j, parent] = np.logaddexp(
                            inside[i, j, parent], self.log_binary_probs[rule_idx] + score
                        )
        return inside

    def _inside_outside(self, terminal_log_density: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        n = terminal_log_density.shape[0]
        inside = self._inside(terminal_log_density)
        terminal_post = np.zeros((n, self.num_terminal_rules), dtype=np.float64)
        binary_counts = np.zeros(self.num_binary_rules, dtype=np.float64)
        if n == 0:
            return -np.inf, terminal_post, binary_counts, inside

        ll = float(inside[0, n, self.start_idx])
        if not np.isfinite(ll):
            return ll, terminal_post, binary_counts, inside

        outside = np.full_like(inside, -np.inf)
        outside[0, n, self.start_idx] = 0.0

        for span in range(n, 1, -1):
            for i in range(n - span + 1):
                j = i + span
                for rule_idx in range(self.num_binary_rules):
                    parent = self.binary_parents[rule_idx]
                    parent_out = outside[i, j, parent]
                    if not np.isfinite(parent_out):
                        continue
                    left = self.binary_left[rule_idx]
                    right = self.binary_right[rule_idx]
                    rule_lp = self.log_binary_probs[rule_idx]
                    for split in range(i + 1, j):
                        left_in = inside[i, split, left]
                        right_in = inside[split, j, right]
                        if np.isfinite(right_in):
                            outside[i, split, left] = np.logaddexp(
                                outside[i, split, left], parent_out + rule_lp + right_in
                            )
                        if np.isfinite(left_in):
                            outside[split, j, right] = np.logaddexp(
                                outside[split, j, right], parent_out + rule_lp + left_in
                            )

        for i in range(n):
            for rule_idx in range(self.num_terminal_rules):
                parent = self.terminal_parents[rule_idx]
                v = (
                    outside[i, i + 1, parent]
                    + self.log_terminal_probs[rule_idx]
                    + terminal_log_density[i, rule_idx]
                    - ll
                )
                if np.isfinite(v):
                    terminal_post[i, rule_idx] = math.exp(v)

        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span
                for rule_idx in range(self.num_binary_rules):
                    parent = self.binary_parents[rule_idx]
                    parent_out = outside[i, j, parent]
                    if not np.isfinite(parent_out):
                        continue
                    left = self.binary_left[rule_idx]
                    right = self.binary_right[rule_idx]
                    rule_lp = self.log_binary_probs[rule_idx]
                    vals = parent_out + rule_lp + inside[i, i + 1 : j, left] + inside[i + 1 : j, j, right] - ll
                    good = np.isfinite(vals)
                    if np.any(good):
                        binary_counts[rule_idx] += float(np.exp(vals[good]).sum())

        return ll, terminal_post, binary_counts, inside

    def log_density(self, x: Sequence[Any]) -> float:
        """Return the log-density or log-mass at a single observation."""
        if len(x) == 0:
            return -np.inf
        terminal_ld = self._terminal_log_density(x)
        inside = self._inside(terminal_ld)
        return float(inside[0, len(x), self.start_idx])

    def _log_density_from_nonterminal(self, x: Sequence[Any], nt: int) -> float:
        if len(x) == 0:
            return -np.inf
        terminal_ld = self._terminal_log_density(x)
        inside = self._inside(terminal_ld)
        return float(inside[0, len(x), nt])

    def seq_log_density(self, x: EncodedPCFGData) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        lengths, enc_by_rule = x
        total = int(lengths.sum())
        terminal_ld = np.empty((total, self.num_terminal_rules), dtype=np.float64)
        for r, dist in enumerate(self.emissions):
            terminal_ld[:, r] = dist.seq_log_density(enc_by_rule[r])

        rv = np.empty(len(lengths), dtype=np.float64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
        for i, n in enumerate(lengths):
            if n == 0:
                rv[i] = -np.inf
            else:
                inside = self._inside(terminal_ld[offsets[i] : offsets[i + 1]])
                rv[i] = inside[0, int(n), self.start_idx]
        return rv

    def compute_capabilities(self):
        """Engine readiness intersected from the terminal emission distributions.

        The CKY inside dynamic program is expressed in ComputeEngine ops (see
        ``backend_seq_log_density``), so the grammar is engine-ready on whatever engines all of its
        terminal emission leaves support (numpy plus, e.g., torch for autograd/GPU).
        """
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        ready = intersect_engine_ready(tuple(self.emissions)) if self.emissions else ("numpy",)
        return DistributionCapabilities(engine_ready=ready, kernel_status="generic_latent")

    def _engine_inside_ll(self, engine, terminal_ld, n):
        """Log-likelihood of one length-``n`` sequence via an engine-routed CKY inside chart.

        ``terminal_ld`` is an (n, num_terminal_rules) engine array of per-position terminal
        log-densities. Each chart cell collects every (binary rule, split) contribution and reduces
        them with a single ``logsumexp`` per nonterminal - equivalent to the NumPy ``_inside``
        logaddexp accumulation, but in engine ops so it runs on numpy and torch (autograd/GPU).
        """
        num_states = self.num_nonterminals
        neg_inf = engine.asarray(-np.inf)
        inside = {}
        for i in range(n):
            cells = []
            for parent in range(num_states):
                rules = self.terminal_by_parent[parent]
                if rules:
                    vals = engine.asarray(self.log_terminal_probs[rules]) + terminal_ld[i, rules]
                    cells.append(engine.logsumexp(vals, axis=0))
                else:
                    cells.append(neg_inf)
            inside[(i, i + 1)] = engine.stack(cells, axis=0)

        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span
                contribs = [[] for _ in range(num_states)]
                for rule_idx in range(self.num_binary_rules):
                    parent = int(self.binary_parents[rule_idx])
                    left = int(self.binary_left[rule_idx])
                    right = int(self.binary_right[rule_idx])
                    log_p = engine.asarray(self.log_binary_probs[rule_idx])
                    for split in range(i + 1, j):
                        contribs[parent].append(log_p + inside[(i, split)][left] + inside[(split, j)][right])
                cells = []
                for parent in range(num_states):
                    if contribs[parent]:
                        cells.append(engine.logsumexp(engine.stack(contribs[parent], axis=0), axis=0))
                    else:
                        cells.append(neg_inf)
                inside[(i, j)] = engine.stack(cells, axis=0)

        return inside[(0, n)][self.start_idx]

    def backend_seq_log_density(self, x: EncodedPCFGData, engine) -> Any:
        """Engine-routed CKY inside scoring (numpy + torch).

        Terminal log-densities come from each emission leaf's own engine backend score, and the
        inside dynamic program runs in ComputeEngine ops. Per-sequence charts are still a Python
        loop (variable-length parses), so this trades the tuned NumPy ``_inside`` for engine
        portability and differentiability rather than raw speed.
        """
        from mixle.stats.compute.backend import backend_seq_log_density as _backend_sld

        lengths, enc_by_rule = x
        lengths = np.asarray(lengths)
        terminal_cols = [
            _backend_sld(self.emissions[r], enc_by_rule[r], engine) for r in range(self.num_terminal_rules)
        ]
        terminal_ld = engine.stack(terminal_cols, axis=1) if terminal_cols else engine.zeros((0, 0))
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
        out = []
        for i, n in enumerate(lengths):
            n = int(n)
            if n == 0:
                out.append(engine.asarray(-np.inf))
            else:
                out.append(self._engine_inside_ll(engine, terminal_ld[offsets[i] : offsets[i + 1]], n))
        return engine.stack(out, axis=0)

    def to_fisher(self, **kwargs):
        """Inside-outside Fisher view for the PCFG."""
        if hasattr(self, "terminal_rules") and hasattr(self, "_inside_outside"):
            return HeterogeneousPCFGFisherView(self)
        return super().to_fisher(**kwargs)

    def sampler(self, seed: int | None = None) -> HeterogeneousPCFGSampler:
        """Return a sampler for drawing observations from this distribution."""
        return HeterogeneousPCFGSampler(self, seed)

    def enumerator(self) -> HeterogeneousPCFGEnumerator:
        """Return an exact support enumerator for acyclic enumerable PCFGs."""
        return HeterogeneousPCFGEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> LazyQuantizedEnumerationIndex:
        """Build a bounded index with an inside-style DP over quantized derivation costs.

        Terminal emission distributions must themselves support ``quantized_index``.
        The DP counts grammar derivations by additive quantized bit cost and lazily
        unranks those derivations into emitted terminal sequences. For unambiguous
        grammars this is an index over distinct sequences; for ambiguous grammars the
        same sequence can be represented by multiple derivations, and the returned log
        probability is still the exact PCFG ``log_density`` of that sequence.
        """
        return _HeterogeneousPCFGQuantizedIndexBuilder(self, max_bits, bin_width_bits).build()

    def estimator(self, pseudo_count: float | None = None) -> HeterogeneousPCFGEstimator:
        """Return an estimator for fitting this distribution from data."""
        binary_rules = [
            (self.nonterminals[p], self.nonterminals[l], self.nonterminals[r], float(q))
            for p, l, r, q in self.binary_rules
        ]
        terminal_rules = [
            (self.nonterminals[p], dist.estimator(pseudo_count=pseudo_count), float(q))
            for p, dist, q in self.terminal_rules
        ]
        return HeterogeneousPCFGEstimator(
            binary_rules=binary_rules,
            terminal_rules=terminal_rules,
            start=self.start,
            nonterminals=self.nonterminals,
            pseudo_count=pseudo_count,
            name=self.name,
        )

    def dist_to_encoder(self) -> HeterogeneousPCFGDataEncoder:
        """Return the data encoder used by this distribution for vectorized methods."""
        return HeterogeneousPCFGDataEncoder([d.dist_to_encoder() for d in self.emissions])


class HeterogeneousPCFGEnumerator(DistributionEnumerator):
    """Exact best-first enumerator for acyclic heterogeneous PCFGs."""

    def __init__(self, dist: HeterogeneousPCFGDistribution) -> None:
        super().__init__(dist)
        self._streams: dict[int, BufferedStream] = {}
        self._building = set()
        self._stream = self._stream_for_nonterminal(dist.start_idx)
        self._pos = 0

    @staticmethod
    def _concat_children(values: tuple[Any, ...]) -> list[Any]:
        left, right = values
        return list(left) + list(right)

    def _terminal_stream(self, rule_idx: int, rule_lp: float):
        emission = self.dist.emissions[rule_idx]
        enum = child_enumerator(emission, "HeterogeneousPCFGDistribution.terminal_rules[%d]" % rule_idx)
        for value, lp in enum:
            yield ([value], float(rule_lp + lp))

    def _stream_for_nonterminal(self, nt: int) -> BufferedStream:
        if nt in self._streams:
            return self._streams[nt]
        if nt in self._building:
            raise EnumerationError(
                self.dist,
                reason="PCFG exact enumerator requires an acyclic grammar; use quantized_index for bounded counting",
            )

        self._building.add(nt)
        rule_streams: list[BufferedStream] = []
        try:
            for rule_idx in self.dist.terminal_by_parent[nt]:
                rule_lp = float(self.dist.log_terminal_probs[rule_idx])
                if rule_lp > -np.inf:
                    rule_streams.append(BufferedStream(self._terminal_stream(rule_idx, rule_lp)))

            for rule_idx in self.dist.binary_by_parent[nt]:
                rule_lp = float(self.dist.log_binary_probs[rule_idx])
                if rule_lp == -np.inf:
                    continue
                left = self._stream_for_nonterminal(int(self.dist.binary_left[rule_idx]))
                right = self._stream_for_nonterminal(int(self.dist.binary_right[rule_idx]))
                product = ProductEnumerator([left, right], combine=self._concat_children, offset=rule_lp)
                rule_streams.append(BufferedStream(product))
        finally:
            self._building.remove(nt)

        if not rule_streams:
            stream = BufferedStream(iter(()))
        elif len(rule_streams) == 1:
            stream = rule_streams[0]
        else:
            exact = lambda value, nt=nt: self.dist._log_density_from_nonterminal(value, nt)
            stream = BufferedStream(best_first_union(rule_streams, [0.0] * len(rule_streams), exact))
        self._streams[nt] = stream
        return stream

    def __next__(self) -> tuple[list[Any], float]:
        item = self._stream.get(self._pos)
        if item is None:
            raise StopIteration
        self._pos += 1
        return item


class _HeterogeneousPCFGQuantizedIndexBuilder:
    """Build and lazily unrank a quantized derivation-count index."""

    def __init__(self, dist: HeterogeneousPCFGDistribution, max_bits: float, bin_width_bits: float) -> None:
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative.")
        if bin_width_bits <= 0:
            raise ValueError("bin_width_bits must be positive.")
        self.dist = dist
        self.max_bits = float(max_bits)
        self.bin_width_bits = float(bin_width_bits)
        self.max_bin = int(math.floor(self.max_bits / self.bin_width_bits + 1.0e-12))
        self.truncated = False
        self.terminal_options: list[list[tuple[int, int, int, list[tuple[Any, float]]]]] = [
            [] for _ in range(dist.num_nonterminals)
        ]
        self.terminal_counts: list[list[int]] = [[0] * (self.max_bin + 1) for _ in range(dist.num_nonterminals)]
        self.counts: list[list[int]] = []

    def _qcost(self, log_prob: float) -> int | None:
        if log_prob == -np.inf:
            return None
        bits = max(0.0, -float(log_prob) / math.log(2.0))
        return int(math.ceil(bits / self.bin_width_bits - 1.0e-12))

    def _prepare_terminal_options(self) -> None:
        for rule_idx, (parent, emission, _) in enumerate(self.dist.terminal_rules):
            rule_cost = self._qcost(float(self.dist.log_terminal_probs[rule_idx]))
            if rule_cost is None:
                continue
            if rule_cost > self.max_bin:
                self.truncated = True
                continue
            try:
                index = emission.quantized_index(max_bits=self.max_bits, bin_width_bits=self.bin_width_bits)
            except EnumerationError as e:
                path = "HeterogeneousPCFGDistribution.terminal_rules[%d]" % rule_idx
                new_path = path if not e.path else "%s -> %s" % (path, e.path)
                raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None
            self.truncated = self.truncated or index.truncated

            by_cost: dict[int, list[tuple[Any, float]]] = defaultdict(list)
            for value, lp in index.iter_from():
                emission_cost = self._qcost(float(lp))
                if emission_cost is None:
                    continue
                total_cost = rule_cost + emission_cost
                if total_cost <= self.max_bin:
                    by_cost[emission_cost].append((value, float(lp)))
                else:
                    self.truncated = True

            for emission_cost in sorted(by_cost):
                items = by_cost[emission_cost]
                total_cost = rule_cost + emission_cost
                self.terminal_options[parent].append((rule_idx, emission_cost, total_cost, items))
                self.terminal_counts[parent][total_cost] += len(items)

    def _compute_counts(self) -> None:
        self.counts = [row[:] for row in self.terminal_counts]
        max_iters = max(64, 2 * (self.max_bin + 1) * max(1, self.dist.num_nonterminals))

        for _ in range(max_iters):
            next_counts = [row[:] for row in self.terminal_counts]
            for rule_idx in range(self.dist.num_binary_rules):
                rule_cost = self._qcost(float(self.dist.log_binary_probs[rule_idx]))
                if rule_cost is None:
                    continue
                if rule_cost > self.max_bin:
                    self.truncated = True
                    continue
                parent = int(self.dist.binary_parents[rule_idx])
                left = int(self.dist.binary_left[rule_idx])
                right = int(self.dist.binary_right[rule_idx])
                for left_bin, left_count in enumerate(self.counts[left]):
                    if left_count == 0:
                        continue
                    remaining = self.max_bin - rule_cost - left_bin
                    if remaining < 0:
                        self.truncated = True
                        continue
                    for right_bin, right_count in enumerate(self.counts[right][: remaining + 1]):
                        if right_count:
                            next_counts[parent][rule_cost + left_bin + right_bin] += left_count * right_count
                    if any(self.counts[right][remaining + 1 :]):
                        self.truncated = True
            if next_counts == self.counts:
                return
            self.counts = next_counts

        raise EnumerationError(self.dist, reason="quantized PCFG counting did not converge within the bit bound")

    def _unrank_nonterminal(self, nt: int, bin_id: int, offset: int) -> list[Any]:
        if bin_id < 0 or bin_id > self.max_bin or offset < 0:
            raise IndexError("offset outside indexed bin.")

        for _, _, total_cost, items in self.terminal_options[nt]:
            if total_cost != bin_id:
                continue
            if offset < len(items):
                return [items[offset][0]]
            offset -= len(items)

        for rule_idx in self.dist.binary_by_parent[nt]:
            rule_cost = self._qcost(float(self.dist.log_binary_probs[rule_idx]))
            if rule_cost is None:
                continue
            rem = bin_id - rule_cost
            if rem < 0:
                continue
            left = int(self.dist.binary_left[rule_idx])
            right = int(self.dist.binary_right[rule_idx])
            for left_bin in range(rem + 1):
                right_bin = rem - left_bin
                left_count = self.counts[left][left_bin]
                right_count = self.counts[right][right_bin]
                block = left_count * right_count
                if block == 0:
                    continue
                if offset >= block:
                    offset -= block
                    continue
                left_offset = offset // right_count
                right_offset = offset % right_count
                return self._unrank_nonterminal(left, left_bin, left_offset) + self._unrank_nonterminal(
                    right, right_bin, right_offset
                )

        raise IndexError("offset outside indexed bin.")

    def build(self) -> LazyQuantizedEnumerationIndex:
        self._prepare_terminal_options()
        self._compute_counts()
        counts = {
            b: self.counts[self.dist.start_idx][b]
            for b in range(self.max_bin + 1)
            if self.counts[self.dist.start_idx][b] > 0
        }

        def getter(bin_id: int, offset: int) -> tuple[list[Any], float]:
            value = self._unrank_nonterminal(self.dist.start_idx, bin_id, offset)
            return value, float(self.dist.log_density(value))

        return LazyQuantizedEnumerationIndex(
            counts, bin_width_bits=self.bin_width_bits, max_bits=self.max_bits, truncated=self.truncated, getter=getter
        )


class HeterogeneousPCFGSampler(DistributionSampler):
    """Sampler for HeterogeneousPCFGDistribution."""

    def __init__(
        self,
        dist: HeterogeneousPCFGDistribution,
        seed: int | None = None,
        max_depth: int = 100,
        max_steps: int = 10000,
    ) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.emission_samplers = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in dist.emissions]

    def _sample_nt(self, nt: int, depth: int, budget: list[int]) -> list[Any]:
        if depth > self.max_depth or budget[0] <= 0:
            raise RuntimeError("HeterogeneousPCFGSampler exceeded recursion limits.")
        budget[0] -= 1
        term_rules = self.dist.terminal_by_parent[nt]
        bin_rules = self.dist.binary_by_parent[nt]
        choices = [("t", r, float(self.dist.terminal_probs[r])) for r in term_rules]
        choices.extend(("b", r, float(self.dist.binary_probs[r])) for r in bin_rules)
        if not choices:
            raise RuntimeError("nonterminal has no productions.")
        probs = np.asarray([u[2] for u in choices], dtype=np.float64)
        probs = probs / probs.sum()
        kind, rule_idx, _ = choices[int(self.rng.choice(len(choices), p=probs))]
        if kind == "t":
            return [self.emission_samplers[rule_idx].sample()]
        left = int(self.dist.binary_left[rule_idx])
        right = int(self.dist.binary_right[rule_idx])
        return self._sample_nt(left, depth + 1, budget) + self._sample_nt(right, depth + 1, budget)

    def sample(self, size: int | None = None) -> list[Any] | list[list[Any]]:
        """Draw one sequence, or ``size`` independent sequences, from the grammar."""
        if size is not None:
            return [self.sample() for _ in range(size)]
        return self._sample_nt(self.dist.start_idx, 0, [self.max_steps])


class HeterogeneousPCFGAccumulator(SequenceEncodableStatisticAccumulator):
    """Inside-outside sufficient-statistic accumulator."""

    def __init__(
        self,
        emission_accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        num_binary_rules: int,
        terminal_parents: Sequence[int],
        binary_parents: Sequence[int],
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        self.emission_accumulators = list(emission_accumulators)
        self.num_terminal_rules = len(self.emission_accumulators)
        self.num_binary_rules = int(num_binary_rules)
        self.terminal_parents = np.asarray(terminal_parents, dtype=np.int32)
        self.binary_parents = np.asarray(binary_parents, dtype=np.int32)
        self.terminal_counts = np.zeros(self.num_terminal_rules, dtype=np.float64)
        self.binary_counts = np.zeros(self.num_binary_rules, dtype=np.float64)
        self.rule_key, self.emission_key = keys if keys is not None else (None, None)
        self._init_rng = False
        self._acc_rng: list[RandomState] | None = None

    def _rng_initialize(self, rng: RandomState) -> None:
        seeds = rng.randint(maxrandint, size=max(1, self.num_terminal_rules))
        self._acc_rng = [RandomState(seeds[i]) for i in range(self.num_terminal_rules)]
        self._init_rng = True

    def update(self, x: Sequence[Any], weight: float, estimate: HeterogeneousPCFGDistribution) -> None:
        """Update sufficient statistics for one observed sequence."""
        self.seq_update(estimate.dist_to_encoder().seq_encode([x]), np.asarray([weight]), estimate)

    def initialize(self, x: Sequence[Any], weight: float, rng: RandomState) -> None:
        """Randomly initialize sufficient statistics for one observed sequence."""
        if not self._init_rng:
            self._rng_initialize(rng)
        enc = self.acc_to_encoder().seq_encode([x])
        self.seq_initialize(enc, np.asarray([weight]), rng)

    def seq_initialize(self, x: EncodedPCFGData, weights: np.ndarray, rng: RandomState) -> None:
        """Randomly initialize rule and emission statistics for encoded sequences."""
        if not self._init_rng:
            self._rng_initialize(rng)
        lengths, enc_by_rule = x
        total = int(lengths.sum())
        token_weights = np.repeat(weights, lengths)
        terminal_weights = np.zeros((total, self.num_terminal_rules), dtype=np.float64)
        if total > 0 and self.num_terminal_rules > 0:
            raw = rng.random_sample((total, self.num_terminal_rules)) + 1.0e-3
            raw /= raw.sum(axis=1, keepdims=True)
            terminal_weights = raw * token_weights[:, None]
            self.terminal_counts += terminal_weights.sum(axis=0)

        if self.num_binary_rules > 0:
            for n, w in zip(lengths, weights):
                if n > 1 and w > 0.0:
                    raw = rng.random_sample(self.num_binary_rules) + 1.0e-3
                    raw /= raw.sum()
                    self.binary_counts += raw * float(n - 1) * w

        for r, acc in enumerate(self.emission_accumulators):
            acc.seq_initialize(enc_by_rule[r], terminal_weights[:, r], self._acc_rng[r])

    def seq_update(self, x: EncodedPCFGData, weights: np.ndarray, estimate: HeterogeneousPCFGDistribution) -> None:
        """Update encoded-sequence statistics with inside-outside posteriors."""
        lengths, enc_by_rule = x
        total = int(lengths.sum())
        terminal_ld = np.empty((total, estimate.num_terminal_rules), dtype=np.float64)
        for r, dist in enumerate(estimate.emissions):
            terminal_ld[:, r] = dist.seq_log_density(enc_by_rule[r])

        terminal_weights = np.zeros((total, estimate.num_terminal_rules), dtype=np.float64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
        for i, n in enumerate(lengths):
            if n == 0 or weights[i] == 0.0:
                continue
            start, stop = offsets[i], offsets[i + 1]
            _, term_post, bin_counts, _ = estimate._inside_outside(terminal_ld[start:stop])
            w = float(weights[i])
            terminal_weights[start:stop, :] = term_post * w
            self.terminal_counts += term_post.sum(axis=0) * w
            self.binary_counts += bin_counts * w

        for r, acc in enumerate(self.emission_accumulators):
            acc.seq_update(enc_by_rule[r], terminal_weights[:, r], estimate.emissions[r])

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray, Sequence[Any]]) -> HeterogeneousPCFGAccumulator:
        """Merge another accumulator value into this accumulator."""
        terminal_counts, binary_counts, emission_values = suff_stat
        self.terminal_counts += terminal_counts
        self.binary_counts += binary_counts
        for r, value in enumerate(emission_values):
            self.emission_accumulators[r].combine(value)
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
        """Return rule counts and emission sufficient statistics."""
        return self.terminal_counts, self.binary_counts, tuple(a.value() for a in self.emission_accumulators)

    def from_value(self, x: tuple[np.ndarray, np.ndarray, Sequence[Any]]) -> HeterogeneousPCFGAccumulator:
        """Replace this accumulator from a serialized sufficient-statistic value."""
        terminal_counts, binary_counts, emission_values = x
        self.terminal_counts = terminal_counts
        self.binary_counts = binary_counts
        for r, value in enumerate(emission_values):
            self.emission_accumulators[r].from_value(value)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed rule and emission statistics into ``stats_dict``."""
        if self.rule_key is not None:
            if self.rule_key in stats_dict:
                t, b = stats_dict[self.rule_key]
                stats_dict[self.rule_key] = (t + self.terminal_counts, b + self.binary_counts)
            else:
                stats_dict[self.rule_key] = (self.terminal_counts, self.binary_counts)
        if self.emission_key is not None:
            if self.emission_key in stats_dict:
                for r, acc in enumerate(stats_dict[self.emission_key]):
                    acc.combine(self.emission_accumulators[r].value())
            else:
                stats_dict[self.emission_key] = self.emission_accumulators
        for acc in self.emission_accumulators:
            acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace keyed rule and emission statistics from ``stats_dict``."""
        if self.rule_key is not None and self.rule_key in stats_dict:
            self.terminal_counts, self.binary_counts = stats_dict[self.rule_key]
        if self.emission_key is not None and self.emission_key in stats_dict:
            self.emission_accumulators = stats_dict[self.emission_key]
        for acc in self.emission_accumulators:
            acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> HeterogeneousPCFGDataEncoder:
        """Return an encoder compatible with this accumulator's emissions."""
        return HeterogeneousPCFGDataEncoder([a.acc_to_encoder() for a in self.emission_accumulators])


class HeterogeneousPCFGAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for HeterogeneousPCFGAccumulator."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        num_binary_rules: int,
        terminal_parents: Sequence[int],
        binary_parents: Sequence[int],
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        self.factories = list(factories)
        self.num_binary_rules = int(num_binary_rules)
        self.terminal_parents = np.asarray(terminal_parents, dtype=np.int32)
        self.binary_parents = np.asarray(binary_parents, dtype=np.int32)
        self.keys = keys

    def make(self) -> HeterogeneousPCFGAccumulator:
        """Create a fresh heterogeneous PCFG accumulator."""
        return HeterogeneousPCFGAccumulator(
            [f.make() for f in self.factories],
            self.num_binary_rules,
            self.terminal_parents,
            self.binary_parents,
            keys=self.keys,
        )


class HeterogeneousPCFGEstimator(ParameterEstimator):
    """Estimator for a fixed-topology heterogeneous PCFG."""

    def __init__(
        self,
        binary_rules: BinaryRuleInput | None,
        terminal_rules: TerminalRuleInput,
        start: Any | None = None,
        nonterminals: Sequence[Any] | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        self.raw_binary = list(_iter_binary_rules(binary_rules))
        self.raw_terminal = list(_iter_terminal_rules(terminal_rules))
        if len(self.raw_terminal) == 0:
            raise ValueError("HeterogeneousPCFGEstimator requires at least one terminal estimator.")
        prior_terminal_rules = [(parent, est, prob) for parent, est, prob in self.raw_terminal]
        self._prior = HeterogeneousPCFGDistribution(
            self.raw_binary, prior_terminal_rules, start=start, nonterminals=nonterminals, name=name
        )
        self.terminal_estimators = [est for _, est, _ in self.raw_terminal]
        self.start = self._prior.start
        self.nonterminals = self._prior.nonterminals
        self.pseudo_count = pseudo_count
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> HeterogeneousPCFGAccumulatorFactory:
        """Return an accumulator factory for this fixed grammar topology."""
        return HeterogeneousPCFGAccumulatorFactory(
            [e.accumulator_factory() for e in self.terminal_estimators],
            self._prior.num_binary_rules,
            self._prior.terminal_parents,
            self._prior.binary_parents,
            keys=self.keys,
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, Sequence[Any]]
    ) -> HeterogeneousPCFGDistribution:
        """Estimate rule probabilities and terminal emissions from statistics."""
        terminal_counts, binary_counts, emission_values = suff_stat
        emissions = [
            self.terminal_estimators[r].estimate(float(terminal_counts[r]), emission_values[r])
            for r in range(len(self.terminal_estimators))
        ]

        term_probs = terminal_counts.copy()
        bin_probs = binary_counts.copy()
        if self.pseudo_count is not None:
            term_probs += self.pseudo_count * self._prior.terminal_probs
            bin_probs += self.pseudo_count * self._prior.binary_probs

        for parent in range(self._prior.num_nonterminals):
            term_idx = self._prior.terminal_by_parent[parent]
            bin_idx = self._prior.binary_by_parent[parent]
            total = 0.0
            if term_idx:
                total += float(term_probs[term_idx].sum())
            if bin_idx:
                total += float(bin_probs[bin_idx].sum())
            if total <= 0.0:
                if term_idx:
                    term_probs[term_idx] = self._prior.terminal_probs[term_idx]
                if bin_idx:
                    bin_probs[bin_idx] = self._prior.binary_probs[bin_idx]
            else:
                if term_idx:
                    term_probs[term_idx] /= total
                if bin_idx:
                    bin_probs[bin_idx] /= total

        binary_rules = [
            (
                self._prior.nonterminals[int(self._prior.binary_parents[r])],
                self._prior.nonterminals[int(self._prior.binary_left[r])],
                self._prior.nonterminals[int(self._prior.binary_right[r])],
                float(bin_probs[r]),
            )
            for r in range(self._prior.num_binary_rules)
        ]
        terminal_rules = [
            (self._prior.nonterminals[int(self._prior.terminal_parents[r])], emissions[r], float(term_probs[r]))
            for r in range(self._prior.num_terminal_rules)
        ]
        return HeterogeneousPCFGDistribution(
            binary_rules=binary_rules,
            terminal_rules=terminal_rules,
            start=self.start,
            nonterminals=self.nonterminals,
            name=self.name,
        )


class InducedHeterogeneousPCFGEstimator(ParameterEstimator):
    """Overcomplete sparse PCFG structure learner.

    This estimator builds a finite grammar skeleton automatically:

    * ``K`` nonterminals named ``start, prefix1, ..., prefixK-1``.
    * every binary rule ``A -> B C``.
    * every terminal rule ``A -> terminal_family_j``.

    EM learns rule probabilities and terminal emission parameters. Rules whose
    expected count is below ``prune_threshold`` or whose normalized probability
    is below ``min_rule_prob`` are assigned probability zero, but the rule layout
    is kept stable so repeated calls to ``seq_estimate`` remain compatible.
    """

    def __init__(
        self,
        max_nonterminals: int,
        terminal_estimators: Sequence[ParameterEstimator],
        start: Any = "S",
        nonterminal_prefix: str = "NT",
        terminal_rule_mass: float = 0.5,
        rule_pseudo_count: float | None = 1.0e-3,
        prune_threshold: float = 0.0,
        min_rule_prob: float = 0.0,
        name: str | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        if max_nonterminals <= 0:
            raise ValueError("max_nonterminals must be positive.")
        if len(terminal_estimators) == 0:
            raise ValueError("terminal_estimators must contain at least one estimator.")
        if terminal_rule_mass <= 0.0 or terminal_rule_mass > 1.0:
            raise ValueError("terminal_rule_mass must be in (0, 1].")
        if prune_threshold < 0.0:
            raise ValueError("prune_threshold must be non-negative.")
        if min_rule_prob < 0.0:
            raise ValueError("min_rule_prob must be non-negative.")

        self.max_nonterminals = int(max_nonterminals)
        self.terminal_family_estimators = list(terminal_estimators)
        self.num_terminal_families = len(self.terminal_family_estimators)
        self.start = start
        self.nonterminal_prefix = nonterminal_prefix
        self.terminal_rule_mass = float(terminal_rule_mass)
        self.binary_rule_mass = 1.0 - self.terminal_rule_mass
        self.rule_pseudo_count = rule_pseudo_count
        self.prune_threshold = float(prune_threshold)
        self.min_rule_prob = float(min_rule_prob)
        self.name = name
        self.keys = keys

        self.nonterminals = [start] + ["%s%d" % (nonterminal_prefix, i) for i in range(1, self.max_nonterminals)]
        self.raw_binary = self._make_binary_rules()
        self.raw_terminal = self._make_terminal_rules(self.terminal_family_estimators)
        prior_terminal_rules = [(parent, est, prob) for parent, est, prob in self.raw_terminal]
        self._prior = HeterogeneousPCFGDistribution(
            self.raw_binary, prior_terminal_rules, start=self.start, nonterminals=self.nonterminals, name=self.name
        )
        self.terminal_estimators = [est for _, est, _ in self.raw_terminal]

    def _make_binary_rules(self) -> list[tuple[Any, Any, Any, float]]:
        if self.binary_rule_mass <= 0.0:
            return []
        prob = self.binary_rule_mass / float(self.max_nonterminals * self.max_nonterminals)
        return [
            (parent, left, right, prob)
            for parent in self.nonterminals
            for left in self.nonterminals
            for right in self.nonterminals
        ]

    def _make_terminal_rules(self, terminals: Sequence[Any]) -> list[tuple[Any, Any, float]]:
        prob = self.terminal_rule_mass / float(len(terminals))
        return [(parent, terminal, prob) for parent in self.nonterminals for terminal in terminals]

    def accumulator_factory(self) -> HeterogeneousPCFGAccumulatorFactory:
        """Return an accumulator factory for the induced grammar skeleton."""
        return HeterogeneousPCFGAccumulatorFactory(
            [e.accumulator_factory() for e in self.terminal_estimators],
            self._prior.num_binary_rules,
            self._prior.terminal_parents,
            self._prior.binary_parents,
            keys=self.keys,
        )

    def initial_model(
        self,
        terminal_distributions: Sequence[SequenceEncodableProbabilityDistribution],
        rng: RandomState | None = None,
        jitter: float = 0.0,
    ) -> HeterogeneousPCFGDistribution:
        """Create an overcomplete starting grammar without hand-written rules.

        ``terminal_distributions`` may contain one distribution per terminal
        family, in which case each nonterminal gets its own copy of the same
        family list, or one distribution per generated terminal rule.
        """
        if len(terminal_distributions) == self.num_terminal_families:
            emissions = [
                terminal_distributions[j] for _ in self.nonterminals for j in range(self.num_terminal_families)
            ]
        elif len(terminal_distributions) == self._prior.num_terminal_rules:
            emissions = list(terminal_distributions)
        else:
            raise ValueError(
                "terminal_distributions must have length %d or %d."
                % (self.num_terminal_families, self._prior.num_terminal_rules)
            )

        bin_probs = self._prior.binary_probs.copy()
        term_probs = self._prior.terminal_probs.copy()
        if jitter > 0.0:
            rng = RandomState() if rng is None else rng
            if len(bin_probs) > 0:
                bin_probs *= np.exp(rng.normal(scale=jitter, size=len(bin_probs)))
            term_probs *= np.exp(rng.normal(scale=jitter, size=len(term_probs)))
            self._normalize_rule_probs(term_probs, bin_probs, use_prior_on_empty=True)

        binary_rules = [
            (
                self._prior.nonterminals[int(self._prior.binary_parents[r])],
                self._prior.nonterminals[int(self._prior.binary_left[r])],
                self._prior.nonterminals[int(self._prior.binary_right[r])],
                float(bin_probs[r]),
            )
            for r in range(self._prior.num_binary_rules)
        ]
        terminal_rules = [
            (self._prior.nonterminals[int(self._prior.terminal_parents[r])], emissions[r], float(term_probs[r]))
            for r in range(self._prior.num_terminal_rules)
        ]
        return HeterogeneousPCFGDistribution(
            binary_rules=binary_rules,
            terminal_rules=terminal_rules,
            start=self.start,
            nonterminals=self.nonterminals,
            name=self.name,
        )

    def _normalize_rule_probs(
        self, term_probs: np.ndarray, bin_probs: np.ndarray, use_prior_on_empty: bool = True
    ) -> None:
        for parent in range(self._prior.num_nonterminals):
            term_idx = self._prior.terminal_by_parent[parent]
            bin_idx = self._prior.binary_by_parent[parent]
            total = 0.0
            if term_idx:
                total += float(term_probs[term_idx].sum())
            if bin_idx:
                total += float(bin_probs[bin_idx].sum())

            if total <= 0.0:
                if not use_prior_on_empty:
                    continue
                if term_idx:
                    term_probs[term_idx] = self._prior.terminal_probs[term_idx]
                if bin_idx:
                    bin_probs[bin_idx] = self._prior.binary_probs[bin_idx]
                total = 0.0
                if term_idx:
                    total += float(term_probs[term_idx].sum())
                if bin_idx:
                    total += float(bin_probs[bin_idx].sum())

            if total > 0.0:
                if term_idx:
                    term_probs[term_idx] /= total
                if bin_idx:
                    bin_probs[bin_idx] /= total

    def _apply_probability_floor(self, term_probs: np.ndarray, bin_probs: np.ndarray) -> None:
        if self.min_rule_prob <= 0.0:
            return
        for parent in range(self._prior.num_nonterminals):
            term_idx = self._prior.terminal_by_parent[parent]
            bin_idx = self._prior.binary_by_parent[parent]
            if term_idx:
                term_probs[term_idx] *= term_probs[term_idx] >= self.min_rule_prob
            if bin_idx:
                bin_probs[bin_idx] *= bin_probs[bin_idx] >= self.min_rule_prob
        self._normalize_rule_probs(term_probs, bin_probs, use_prior_on_empty=True)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, Sequence[Any]]
    ) -> HeterogeneousPCFGDistribution:
        """Estimate a sparse grammar from rule counts and emission statistics."""
        terminal_counts, binary_counts, emission_values = suff_stat
        emissions = [
            self.terminal_estimators[r].estimate(float(terminal_counts[r]), emission_values[r])
            for r in range(len(self.terminal_estimators))
        ]

        term_probs = terminal_counts.copy()
        bin_probs = binary_counts.copy()
        if self.rule_pseudo_count is not None and self.rule_pseudo_count > 0.0:
            term_probs += self.rule_pseudo_count * self._prior.terminal_probs
            bin_probs += self.rule_pseudo_count * self._prior.binary_probs

        if self.prune_threshold > 0.0:
            term_probs *= terminal_counts >= self.prune_threshold
            bin_probs *= binary_counts >= self.prune_threshold
        else:
            term_probs *= terminal_counts > 0.0
            bin_probs *= binary_counts > 0.0

        self._normalize_rule_probs(term_probs, bin_probs, use_prior_on_empty=True)
        self._apply_probability_floor(term_probs, bin_probs)

        binary_rules = [
            (
                self._prior.nonterminals[int(self._prior.binary_parents[r])],
                self._prior.nonterminals[int(self._prior.binary_left[r])],
                self._prior.nonterminals[int(self._prior.binary_right[r])],
                float(bin_probs[r]),
            )
            for r in range(self._prior.num_binary_rules)
        ]
        terminal_rules = [
            (self._prior.nonterminals[int(self._prior.terminal_parents[r])], emissions[r], float(term_probs[r]))
            for r in range(self._prior.num_terminal_rules)
        ]
        return HeterogeneousPCFGDistribution(
            binary_rules=binary_rules,
            terminal_rules=terminal_rules,
            start=self.start,
            nonterminals=self.nonterminals,
            name=self.name,
        )


class HeterogeneousPCFGDataEncoder(DataSequenceEncoder):
    """Encode iid PCFG observations by flattening terminal observations."""

    def __init__(self, terminal_encoders: Sequence[DataSequenceEncoder]) -> None:
        self.terminal_encoders = list(terminal_encoders)

    def __str__(self) -> str:
        return "HeterogeneousPCFGDataEncoder([%s])" % ",".join(str(e) for e in self.terminal_encoders)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HeterogeneousPCFGDataEncoder) and self.terminal_encoders == other.terminal_encoders

    def seq_encode(self, x: Sequence[Sequence[Any]]) -> EncodedPCFGData:
        """Flatten token sequences and encode them for every terminal family."""
        lengths = np.asarray([len(seq) for seq in x], dtype=np.int32)
        flat: list[Any] = []
        for seq in x:
            flat.extend(seq)
        enc_by_rule = tuple(enc.seq_encode(flat) for enc in self.terminal_encoders)
        return lengths, enc_by_rule


# --- Fisher view(s) co-located with this family ---
class HeterogeneousPCFGFisherView(FixedFisherView):
    """Inside-outside Fisher view for heterogeneous PCFGs.

    Coordinates are expected complete-data rule counts followed by terminal
    emission sufficient statistics gated by the posterior probability of the
    corresponding terminal rule at each token.  For finite enumerable PCFGs,
    the model Fisher is the exact observed Fisher covariance of these
    posterior-expected complete-data statistics under the model distribution.
    Recursive or infinite-support grammars should use observed_fisher_* on data.
    """

    _max_model_enum_terms = 100000
    _model_mass_tol = 1.0e-8

    def __init__(self, dist: Any) -> None:
        self.dist = dist
        self.child_views = [to_fisher(d) for d in dist.emissions]
        self._model_cache: tuple[np.ndarray, np.ndarray] | None = None
        super().__init__(dist, self._labels_from_children())

    def _labels_from_children(self) -> list[Path]:
        labels: list[Path] = [("terminal_rule", str(r)) for r in range(self.dist.num_terminal_rules)]
        labels.extend(("binary_rule", str(r)) for r in range(self.dist.num_binary_rules))
        for r, view in enumerate(self.child_views):
            labels.extend(("terminal_emission", str(r)) + label for label in view.vectorizer.labels)
        return labels

    def _refresh_labels(self) -> None:
        self.labels = self._labels_from_children()
        self.vectorizer = SufficientStatisticVectorizer(self.labels)
        self._model_cache = None

    def _matrix_from_values(self, values: Sequence[Any]) -> np.ndarray:
        if not values:
            return np.zeros((0, len(self.labels)), dtype=np.float64)

        terminal = np.vstack([np.asarray(v[0], dtype=np.float64).reshape(-1) for v in values])
        binary = np.vstack([np.asarray(v[1], dtype=np.float64).reshape(-1) for v in values])
        blocks = [terminal, binary]

        for r, view in enumerate(self.child_views):
            emission_values = [v[2][r] for v in values]
            blocks.append(_structured_values_matrix(view, emission_values))

        self._refresh_labels()
        return np.hstack(blocks)

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        model = self.dist if estimate is None else estimate
        enc = _seq_encode_model(model, list(data))
        return self._statistics_from_encoded(enc, estimate=model)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        model = self.dist if estimate is None else estimate
        lengths, enc_by_rule = enc_data
        lengths = np.asarray(lengths, dtype=np.int64)
        nobs = len(lengths)
        total = int(lengths.sum())

        terminal_ld = np.empty((total, model.num_terminal_rules), dtype=np.float64)
        child_stats: list[np.ndarray] = []
        for r, dist in enumerate(model.emissions):
            if total > 0:
                terminal_ld[:, r] = dist.seq_log_density(enc_by_rule[r])
                child_stats.append(self.child_views[r].seq_expected_statistics(enc_by_rule[r], estimate=dist))
            else:
                terminal_ld = np.empty((0, model.num_terminal_rules), dtype=np.float64)
                child_stats.append(np.zeros((0, len(self.child_views[r].vectorizer.labels)), dtype=np.float64))

        self._refresh_labels()
        terminal_counts = np.zeros((nobs, model.num_terminal_rules), dtype=np.float64)
        binary_counts = np.zeros((nobs, model.num_binary_rules), dtype=np.float64)
        emission_blocks = [np.zeros((nobs, stats.shape[1]), dtype=np.float64) for stats in child_stats]

        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        for i, n in enumerate(lengths):
            if n <= 0:
                continue
            start = int(offsets[i])
            stop = int(offsets[i + 1])
            _, terminal_post, binary_count, _ = model._inside_outside(terminal_ld[start:stop])
            terminal_counts[i] = terminal_post.sum(axis=0)
            binary_counts[i] = binary_count
            for r, stats in enumerate(child_stats):
                if stats.shape[1] > 0:
                    emission_blocks[r][i] = np.dot(terminal_post[:, r], stats[start:stop])

        blocks = [terminal_counts, binary_counts]
        blocks.extend(emission_blocks)
        return np.hstack(blocks) if blocks else np.zeros((nobs, 0), dtype=np.float64)

    def _enumerated_model_mean_cov(self) -> tuple[np.ndarray, np.ndarray]:
        if self._model_cache is not None:
            return self._model_cache

        values: list[Any] = []
        probs: list[float] = []
        try:
            iterator = iter(self.dist.enumerator())
            exhausted = False
            for _ in range(self._max_model_enum_terms):
                try:
                    value, log_prob = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                if np.isfinite(log_prob):
                    values.append(value)
                    probs.append(float(math.exp(log_prob)))
            if not exhausted:
                raise NotImplementedError(
                    "PCFG model Fisher requires finite enumerable support; use observed_fisher_information()."
                )
        except NotImplementedError:
            raise
        except Exception as exc:
            raise NotImplementedError(
                "PCFG model Fisher requires finite enumerable support; use observed_fisher_information()."
            ) from exc

        if not values:
            raise NotImplementedError("PCFG model Fisher requires non-empty finite support.")

        weights = np.asarray(probs, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total) or abs(total - 1.0) > self._model_mass_tol:
            raise NotImplementedError("PCFG finite support did not sum to one; use observed_fisher_information().")
        weights /= total

        stats = self.expected_statistics_matrix(data=values)
        mean = np.dot(weights, stats)
        second = np.dot((weights[:, None] * stats).T, stats)
        cov = second - np.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        diag = np.maximum(np.diag(cov), 0.0)
        cov[np.diag_indices_from(cov)] = diag
        self._model_cache = (mean, cov)
        return self._model_cache

    def _model_mean(self) -> np.ndarray:
        return self._enumerated_model_mean_cov()[0]

    def _model_fisher(self) -> np.ndarray:
        return self._enumerated_model_mean_cov()[1]

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        """Return model Fisher information, falling back to observed statistics."""
        try:
            return FixedFisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge, **kwargs)
        except NotImplementedError:
            if stats is not None:
                return FisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge)
            raise

    def fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        """Whiten PCFG statistics under the requested Fisher metric."""
        try:
            return FixedFisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge, **kwargs
            )
        except NotImplementedError:
            if stats is None:
                raise
            return FisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge
            )
