"""Knowledge-graph-producing LLM uncertainty by marginalizing over graphs.

An LLM's raw likelihood is over strings, but many applications care about the
information asserted by those strings. This module has the model emit a
knowledge graph, represented as a set of triples, so equivalent information can
be canonicalized by exact graph equality. Answers are then obtained by
marginalizing over the graphs that produce them::

    P(outcome = c) = sum over graphs G with outcome(G) = c  of  P(G)

The reliability of a single fact is its edge marginal ``P(triple in G)``. Feed
those marginals to :class:`mixle.inference.ProbabilityCalibrator` to calibrate
against labeled truth when such labels are available.

``GraphLLM`` wraps any ``generate(prompt) -> str`` callable plus a
``parse(str) -> triples`` callable. It samples the model, canonicalizes each
generation to a graph, and marginalizes by Monte Carlo counting or by summing
sequence likelihoods when ``log_probs`` are supplied.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from mixle.inference.calibration import ProbabilityCalibrator, calibrate_probabilities

Triple = tuple  # (subject, relation, object) or any fixed-arity fact tuple


def canonical_graph(triples: Iterable[Any]) -> frozenset:
    """Return an order-independent, deduplicated graph representation."""
    canonical = []
    for triple in triples:
        if not isinstance(triple, (tuple, list)) or not triple:
            raise TypeError("every graph fact must be a non-empty tuple or list.")
        fact = tuple(triple)
        try:
            hash(fact)
        except TypeError as exc:
            raise TypeError("every graph fact and field must be hashable.") from exc
        canonical.append(fact)
    return frozenset(canonical)


@dataclass(frozen=True)
class GraphDistribution:
    """A distribution over knowledge graphs.

    ``graphs`` are the distinct canonical graphs observed; ``probs[i] = P(graphs[i])`` is the string
    distribution marginalized onto graphs (so it sums to 1 over distinct graphs). Every query is
    answered by marginalizing this distribution over the graphs that produce the queried outcome.
    """

    graphs: tuple[frozenset, ...]
    probs: np.ndarray

    def __post_init__(self) -> None:
        graphs = tuple(self.graphs)
        if not graphs:
            raise ValueError("GraphDistribution requires at least one graph.")
        if any(not isinstance(graph, frozenset) for graph in graphs):
            raise TypeError("graphs must contain canonical frozenset graph values.")
        for graph in graphs:
            for fact in graph:
                if not isinstance(fact, tuple) or not fact:
                    raise TypeError("canonical graph facts must be non-empty tuples.")
                try:
                    hash(fact)
                except TypeError as exc:
                    raise TypeError("canonical graph facts must be hashable.") from exc
        if len(set(graphs)) != len(graphs):
            raise ValueError("GraphDistribution graphs must be distinct.")
        probabilities = np.asarray(self.probs, dtype=float)
        if probabilities.shape != (len(graphs),):
            raise ValueError("probs must contain exactly one probability per graph.")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
            raise ValueError("graph probabilities must be finite and non-negative.")
        if not np.isclose(float(probabilities.sum()), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("graph probabilities must sum to one.")
        probabilities = probabilities.copy()
        probabilities.setflags(write=False)
        object.__setattr__(self, "graphs", graphs)
        object.__setattr__(self, "probs", probabilities)

    def marginalize(self, outcome: Callable[[frozenset], Hashable]) -> list[tuple[Any, float]]:
        """Return ``P(outcome = c) = sum_{G : outcome(G) = c} P(G)``.

        ``outcome`` maps a graph to a hashable value (a fact's object, a boolean property, an
        aggregate). Returns ``[(value, probability), ...]`` sorted by descending probability.
        """
        mass: dict[Any, float] = {}
        for g, p in zip(self.graphs, self.probs):
            v = outcome(g)
            mass[v] = mass.get(v, 0.0) + float(p)
        return sorted(mass.items(), key=lambda kv: -kv[1])

    def entropy(self, outcome: Callable[[frozenset], Hashable]) -> float:
        """Return entropy in nats of the marginal outcome distribution."""
        p = np.array([q for _, q in self.marginalize(outcome)], dtype=float)
        p = p[p > 0.0]
        return float(-np.sum(p * np.log(p)))

    def edge_marginals(self) -> dict[Triple, float]:
        """Return ``P(triple in G)`` for every asserted triple."""
        out: dict[Triple, float] = {}
        for g, p in zip(self.graphs, self.probs):
            for t in g:
                out[t] = out.get(t, 0.0) + float(p)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def fact_probability(self, triple: Any) -> float:
        """``P(triple in G)`` for one fact (0 if never asserted)."""
        t = tuple(triple)
        return float(sum(p for g, p in zip(self.graphs, self.probs) if t in g))

    def calibrated_edge_marginals(self, calibrator: ProbabilityCalibrator) -> dict[Triple, float]:
        """Map edge marginals through a fitted calibrator.

        A raw edge marginal is the model's internal assertion rate for a fact, not a probability that
        the fact is *true* -- a confidently-hallucinated fact has a high marginal yet is false. Fit the
        calibrator with :func:`fit_fact_calibrator` on labeled facts, then this reports, per fact, the
        empirical truth rate at that marginal. Confident hallucinations that
        look exactly like known facts still require an external check.
        """
        m = self.edge_marginals()
        keys = list(m)
        vals = calibrator.predict([m[k] for k in keys])
        return dict(zip(keys, (float(v) for v in vals)))

    def query(self, *prefix: Any) -> list[tuple[Any, float]]:
        """Return per-object fact marginals for triples matching ``prefix``.

        A graph may assert several matching objects; each is then an event with
        its own marginal and may have probability one. The result is therefore
        not renormalized into a categorical outcome. Use :meth:`functional_query`
        when the relation is declared single-valued and a categorical outcome,
        including no assertion, is required.
        """
        if not prefix:
            raise ValueError("query requires a non-empty fact prefix.")
        k = len(prefix)
        mass: dict[Any, float] = {}
        for g, p in zip(self.graphs, self.probs):
            objs = {t[k:] for t in g if len(t) > k and tuple(t[:k]) == tuple(prefix)}
            for o in objs:  # a graph asserting the fact contributes its full mass once
                val = o[0] if len(o) == 1 else o
                mass[val] = mass.get(val, 0.0) + float(p)
        return sorted(mass.items(), key=lambda kv: -kv[1])

    def functional_query(self, *prefix: Any, no_assertion: Hashable = None) -> list[tuple[Any, float]]:
        """Return a categorical outcome for an explicitly functional relation.

        Every graph must assert at most one matching suffix. Graphs with no
        match contribute to ``no_assertion`` rather than disappearing.
        """
        if not prefix:
            raise ValueError("functional_query requires a non-empty fact prefix.")
        k = len(prefix)
        mass: dict[Any, float] = {}
        for graph, probability in zip(self.graphs, self.probs):
            objects = {fact[k:] for fact in graph if len(fact) > k and tuple(fact[:k]) == tuple(prefix)}
            if len(objects) > 1:
                raise ValueError(f"relation prefix {prefix!r} is not functional in every graph.")
            if objects:
                suffix = next(iter(objects))
                value = suffix[0] if len(suffix) == 1 else suffix
            else:
                value = no_assertion
            mass[value] = mass.get(value, 0.0) + float(probability)
        return sorted(mass.items(), key=lambda item: -item[1])

    def most_likely_graph(self) -> tuple[frozenset, float]:
        """The single most probable graph and its probability."""
        i = int(np.argmax(self.probs))
        return self.graphs[i], float(self.probs[i])


class GraphLLM:
    """Turn a ``generate(prompt) -> str`` LLM into a distribution over knowledge graphs.

    Args:
        generate: ``callable(prompt) -> str`` for one stochastic generation.
        parse: ``callable(str) -> iterable[triple]`` to extract asserted facts.
            Generations that parse to the same triple set are treated as the
            same canonical graph.
        n: default number of samples per prompt.
    """

    def __init__(
        self,
        generate: Callable[[str], str],
        parse: Callable[[str], Iterable[Any]],
        *,
        n: int = 10,
    ) -> None:
        if not callable(generate) or not callable(parse):
            raise TypeError("generate and parse must be callable.")
        self.generate = generate
        self.parse = parse
        self.n = _positive_sample_count(n)

    def sample_graphs(self, prompt: str, n: int | None = None) -> list[frozenset]:
        """Sample ``n`` generations and parse each into a canonical graph."""
        count = self.n if n is None else _positive_sample_count(n)
        return [canonical_graph(self.parse(self.generate(prompt))) for _ in range(count)]

    def distribution(
        self,
        prompt: str,
        n: int | None = None,
        *,
        log_probs: Sequence[float] | None = None,
        graphs: Sequence[frozenset] | None = None,
        strings: Sequence[str] | None = None,
    ) -> GraphDistribution:
        """Sample, parse, and marginalize strings onto graphs.

        Marginalization uses Monte-Carlo counting by default (``P(G)`` = fraction of samples that
        parse to ``G``); pass ``log_probs`` (one ``log P(string)`` per sample) to instead sum the
        sequence likelihoods within each graph. This lower-variance estimator
        does not assume every string realizing a graph is equiprobable.
        """
        if graphs is None:
            if strings is not None:
                raise ValueError("strings may only be supplied with explicit graphs.")
            count = self.n if n is None else _positive_sample_count(n)
            generated = [self.generate(prompt) for _ in range(count)]
            if any(not isinstance(value, str) for value in generated):
                raise TypeError("generate must return strings.")
            gs = [canonical_graph(self.parse(value)) for value in generated]
            generated_strings = generated
        else:
            gs = list(graphs)
            generated_strings = list(strings) if strings is not None else None
            if n is not None:
                raise ValueError("n cannot be supplied together with explicit graphs.")
        if not gs:
            raise ValueError("no samples to form a graph distribution")
        if any(not isinstance(graph, frozenset) for graph in gs):
            raise TypeError("graphs must contain canonical frozenset values.")
        distinct: list[frozenset] = []
        index: dict[frozenset, int] = {}
        for g in gs:
            if g not in index:
                index[g] = len(distinct)
                distinct.append(g)
        if log_probs is not None:
            lp = np.asarray(log_probs, dtype=float).reshape(-1)
            if lp.size != len(gs):
                raise ValueError("log_probs must have one entry per sample")
            if np.isnan(lp).any() or np.isposinf(lp).any() or not np.isfinite(lp).any():
                raise ValueError("log_probs must contain valid log mass and cannot be all -inf.")
            if generated_strings is None:
                raise ValueError("strings are required with log_probs so duplicate sequences can be deduplicated.")
            if len(generated_strings) != len(gs) or any(not isinstance(value, str) for value in generated_strings):
                raise ValueError("strings must contain one generated string per graph and log probability.")
            unique_strings: dict[str, tuple[frozenset, float]] = {}
            for string, graph, log_probability in zip(generated_strings, gs, lp):
                if string in unique_strings:
                    previous_graph, previous_log_probability = unique_strings[string]
                    if graph != previous_graph or not np.isclose(log_probability, previous_log_probability):
                        raise ValueError("duplicate strings must map to the same graph and log probability.")
                    continue
                unique_strings[string] = (graph, float(log_probability))
            logmass = np.full(len(distinct), -np.inf)
            for g, l in unique_strings.values():
                i = index[g]
                logmass[i] = np.logaddexp(logmass[i], l)
            normalizer = float(logsumexp(logmass))
            if not np.isfinite(normalizer):
                raise ValueError("enumerated string log mass must contain at least one finite value.")
            probs = np.exp(logmass - normalizer)
        else:
            counts = np.zeros(len(distinct))
            for g in gs:
                counts[index[g]] += 1.0
            probs = counts / counts.sum()
        return GraphDistribution(distinct, probs)


def _positive_sample_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError("sample count must be a positive integer.")
    return int(value)


def fit_fact_calibrator(
    distributions: Iterable[GraphDistribution],
    truth: Callable[[Any], bool],
    *,
    method: str = "isotonic",
) -> ProbabilityCalibrator:
    """Fit ``edge marginal -> P(fact is true)`` over the facts asserted across many graph distributions.

    Turn the model's internal assertion rate (the edge marginal) into a calibrated probability of
    *truth*, learned against ground-truth labels. Collect every ``(triple, marginal)`` the model
    asserts, label it with ``truth(triple)``, and fit a :class:`~mixle.inference.ProbabilityCalibrator`.

    This does not by itself identify confident hallucinations: a false fact the
    model reliably emits can have a high marginal. Calibration can improve the
    aggregate reliability curve, but separating those cases requires an
    external signal such as retrieval or a checker.
    """
    scores, outcomes = [], []
    for d in distributions:
        for triple, marg in d.edge_marginals().items():
            scores.append(float(marg))
            outcomes.append(1.0 if truth(triple) else 0.0)
    if len(scores) < 2:
        raise ValueError("need at least two asserted facts across the distributions to calibrate")
    return calibrate_probabilities(np.asarray(scores), np.asarray(outcomes), method=method)
