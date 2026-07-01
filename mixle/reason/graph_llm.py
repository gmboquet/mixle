"""Knowledge-graph-producing LLM: UQ on the *information*, by marginalizing over graphs.

The likelihood an LLM emits is over *strings*; the thing we care about is the likelihood of the
*information*. Bridging them needs ``P(meaning) = sum over strings with that meaning of P(string)`` --
but "same meaning" over free text is fuzzy and its equivalence class is unbounded.

Have the model emit a **knowledge graph** (a set of triples) instead of prose and the difficulty
dissolves: the information *is* the generated object, and "same meaning" becomes **exact graph
equality** -- a computable equivalence, no embeddings or entailment needed. Then the answer to any
query is obtained by **marginalizing over the graphs (subgraphs) that produce it**::

    P(outcome = c) = sum over graphs G with outcome(G) = c  of  P(G)

and the reliability of a single fact is its **edge marginal** ``P(triple in G)`` -- exactly a
knowledge-graph edge posterior (feed it to :class:`mixle.inference.ProbabilityCalibrator` to calibrate
against truth; the per-graph samples are an ensemble, so BALD-style epistemic splits apply too).

``GraphLLM`` wraps any ``generate(prompt) -> str`` plus a ``parse(str) -> triples``; it samples the
model, canonicalizes each generation to a graph, and marginalizes -- by Monte-Carlo counting, or by
summing sequence likelihoods when ``log_probs`` are supplied (the correct, lower-variance estimator).
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
    """Canonical form of a graph: the frozenset of its triples (order-independent, dedup, hashable)."""
    return frozenset(tuple(t) for t in triples)


@dataclass(frozen=True)
class GraphDistribution:
    """A distribution over knowledge graphs -- the LLM's belief about the *information*.

    ``graphs`` are the distinct canonical graphs observed; ``probs[i] = P(graphs[i])`` is the string
    distribution marginalized onto graphs (so it sums to 1 over distinct graphs). Every query is
    answered by marginalizing this distribution over the graphs that produce the queried outcome.
    """

    graphs: list[frozenset]
    probs: np.ndarray

    def marginalize(self, outcome: Callable[[frozenset], Hashable]) -> list[tuple[Any, float]]:
        """``P(outcome = c) = sum_{G : outcome(G) = c} P(G)`` -- marginalize over subgraphs.

        ``outcome`` maps a graph to a hashable value (a fact's object, a boolean property, an
        aggregate). Returns ``[(value, probability), ...]`` sorted by descending probability.
        """
        mass: dict[Any, float] = {}
        for g, p in zip(self.graphs, self.probs):
            v = outcome(g)
            mass[v] = mass.get(v, 0.0) + float(p)
        return sorted(mass.items(), key=lambda kv: -kv[1])

    def entropy(self, outcome: Callable[[frozenset], Hashable]) -> float:
        """Entropy (nats) of the marginal ``P(outcome = .)`` -- the model's uncertainty about that query."""
        p = np.array([q for _, q in self.marginalize(outcome)], dtype=float)
        p = p[p > 0.0]
        return float(-np.sum(p * np.log(p)))

    def edge_marginals(self) -> dict[Triple, float]:
        """``P(triple in G)`` for every triple -- the per-fact reliability (a KG edge posterior)."""
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
        """Edge marginals mapped through a fitted calibrator -> a *calibrated* ``P(fact is true)``.

        A raw edge marginal is the model's internal assertion rate for a fact, not a probability that
        the fact is *true* -- a confidently-hallucinated fact has a high marginal yet is false. Fit the
        calibrator with :func:`fit_fact_calibrator` on labeled facts, then this reports, per fact, the
        empirical truth rate at that marginal. (Its residual limit -- confident hallucinations that look
        exactly like known facts -- is why an external check is needed; see the validation tests.)
        """
        m = self.edge_marginals()
        keys = list(m)
        vals = calibrator.predict([m[k] for k in keys])
        return dict(zip(keys, (float(v) for v in vals)))

    def query(self, *prefix: Any) -> list[tuple[Any, float]]:
        """Answer-completion posterior: ``P(object | prefix)`` over triples whose leading fields match.

        ``query("eiffel", "city")`` marginalizes over graphs, collecting the objects of every triple
        starting ``("eiffel", "city", ...)`` weighted by ``P(G)``, then renormalizes over the objects
        actually asserted. Returns ``[(object, probability), ...]`` best-first.
        """
        k = len(prefix)
        mass: dict[Any, float] = {}
        for g, p in zip(self.graphs, self.probs):
            objs = {t[k:] for t in g if len(t) > k and tuple(t[:k]) == tuple(prefix)}
            for o in objs:  # a graph asserting the fact contributes its full mass once
                val = o[0] if len(o) == 1 else o
                mass[val] = mass.get(val, 0.0) + float(p)
        total = sum(mass.values())
        if total <= 0.0:
            return []
        return sorted(((v, m / total) for v, m in mass.items()), key=lambda kv: -kv[1])

    def most_likely_graph(self) -> tuple[frozenset, float]:
        """The single most probable graph and its probability."""
        i = int(np.argmax(self.probs))
        return self.graphs[i], float(self.probs[i])


class GraphLLM:
    """Turn a ``generate(prompt) -> str`` LLM into a distribution over knowledge graphs.

    Args:
        generate: ``callable(prompt) -> str`` -- one stochastic generation.
        parse: ``callable(str) -> iterable[triple]`` -- extract the asserted facts (semantic parse /
            structured-output decode). Generations that parse to the same triple-set are the *same
            meaning* -- exact equality, no fuzzy matching.
        n: default number of samples per prompt.
    """

    def __init__(
        self,
        generate: Callable[[str], str],
        parse: Callable[[str], Iterable[Any]],
        *,
        n: int = 10,
    ) -> None:
        self.generate = generate
        self.parse = parse
        self.n = int(n)

    def sample_graphs(self, prompt: str, n: int | None = None) -> list[frozenset]:
        """Sample ``n`` generations and parse each into a canonical graph."""
        return [canonical_graph(self.parse(self.generate(prompt))) for _ in range(int(n or self.n))]

    def distribution(
        self,
        prompt: str,
        n: int | None = None,
        *,
        log_probs: Sequence[float] | None = None,
        graphs: Sequence[frozenset] | None = None,
    ) -> GraphDistribution:
        """Sample, parse, and marginalize strings onto graphs -> a :class:`GraphDistribution`.

        Marginalization uses Monte-Carlo counting by default (``P(G)`` = fraction of samples that
        parse to ``G``); pass ``log_probs`` (one ``log P(string)`` per sample) to instead sum the
        sequence likelihoods within each graph -- the lower-variance, estimator-correct form that
        does not assume every string realizing a graph is equiprobable.
        """
        gs = list(graphs) if graphs is not None else self.sample_graphs(prompt, n)
        if not gs:
            raise ValueError("no samples to form a graph distribution")
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
            logmass = np.full(len(distinct), -np.inf)
            for g, l in zip(gs, lp):
                i = index[g]
                logmass[i] = np.logaddexp(logmass[i], l)
            probs = np.exp(logmass - logsumexp(logmass))
        else:
            counts = np.zeros(len(distinct))
            for g in gs:
                counts[index[g]] += 1.0
            probs = counts / counts.sum()
        return GraphDistribution(distinct, probs)


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

    This does NOT rescue confident hallucination -- a fact the model reliably confabulates has a high
    marginal indistinguishable from a genuinely-known one, so calibration lowers the *overall* fact-ECE
    but cannot pull those specific facts down. Separating them needs a signal external to the model
    (retrieval / a checker); the validation tests quantify both the gain and this residual.
    """
    scores, outcomes = [], []
    for d in distributions:
        for triple, marg in d.edge_marginals().items():
            scores.append(float(marg))
            outcomes.append(1.0 if truth(triple) else 0.0)
    if len(scores) < 2:
        raise ValueError("need at least two asserted facts across the distributions to calibrate")
    return calibrate_probabilities(np.asarray(scores), np.asarray(outcomes), method=method)
