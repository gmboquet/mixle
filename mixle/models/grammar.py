"""Probabilistic context-free grammar fitting and parse inspection helpers.

The module wraps induced heterogeneous PCFG estimators with fit diagnostics,
likelihood evaluation, Viterbi parse reconstruction, and rule-table extraction
for small grammar-learning experiments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.inference import seq_estimate, seq_initialize
from mixle.models._result import FitResult
from mixle.stats import (
    HeterogeneousPCFGDistribution,
    InducedHeterogeneousPCFGEstimator,
    seq_encode,
)
from mixle.stats.compute.pdist import ParameterEstimator


@dataclass
class GrammarLearningResult(FitResult["HeterogeneousPCFGDistribution"]):
    """Fitted PCFG plus training and optional validation log-likelihood history."""


@dataclass
class PCFGParseNode:
    """Node in a Viterbi parse tree."""

    label: Any
    span: tuple[int, int]
    log_prob: float
    rule_index: int
    rule_type: str
    children: tuple[PCFGParseNode, ...] = ()
    value: Any = None

    def leaves(self) -> list[Any]:
        """Return terminal observations under this node."""
        if self.rule_type == "terminal":
            return [self.value]
        rv: list[Any] = []
        for child in self.children:
            rv.extend(child.leaves())
        return rv


def fit_induced_pcfg(
    data: Sequence[Sequence[Any]],
    terminal_estimators: Sequence[ParameterEstimator],
    max_nonterminals: int,
    initial_model: HeterogeneousPCFGDistribution | None = None,
    vdata: Sequence[Sequence[Any]] | None = None,
    max_its: int = 10,
    init_p: float = 1.0,
    seed: int | None = None,
    terminal_rule_mass: float = 0.5,
    rule_pseudo_count: float | None = 1.0e-3,
    prune_threshold: float = 0.0,
    min_rule_prob: float = 0.0,
    start: Any = "S",
    name: str | None = None,
) -> GrammarLearningResult:
    """Fit an induced heterogeneous PCFG and track train/validation likelihoods."""
    if len(data) == 0:
        raise ValueError("fit_induced_pcfg requires at least one sequence.")
    estimator = InducedHeterogeneousPCFGEstimator(
        max_nonterminals=max_nonterminals,
        terminal_estimators=terminal_estimators,
        start=start,
        terminal_rule_mass=terminal_rule_mass,
        rule_pseudo_count=rule_pseudo_count,
        prune_threshold=prune_threshold,
        min_rule_prob=min_rule_prob,
        name=name,
    )
    rng = np.random.RandomState(seed)
    if initial_model is None:
        enc_data = seq_encode(data, estimator=estimator)
        model = seq_initialize(enc_data, estimator, rng, p=init_p)
    else:
        model = initial_model
        enc_data = seq_encode(data, model=model)
    enc_vdata = None if vdata is None else seq_encode(vdata, model=model)
    history = [pcfg_log_likelihood(model, data)]
    validation_history = None if vdata is None else [pcfg_log_likelihood(model, vdata)]

    for _ in range(max(1, int(max_its))):
        model = seq_estimate(enc_data, estimator, model)
        history.append(pcfg_log_likelihood(model, data))
        if enc_vdata is not None:
            validation_history.append(float(np.sum(model.seq_log_density(enc_vdata[0][1]))))
    return GrammarLearningResult(model, history, validation_history)


def pcfg_log_likelihood(model: HeterogeneousPCFGDistribution, data: Sequence[Sequence[Any]]) -> float:
    """Return total PCFG log likelihood on raw sequences."""
    if len(data) == 0:
        return 0.0
    enc = model.dist_to_encoder().seq_encode(data)
    return float(np.sum(model.seq_log_density(enc)))


def viterbi_parse(model: HeterogeneousPCFGDistribution, sequence: Sequence[Any]) -> PCFGParseNode:
    """Return the maximum-probability CKY parse under a heterogeneous PCFG."""
    n = len(sequence)
    if n == 0:
        raise ValueError("viterbi_parse requires a non-empty sequence.")
    k = model.num_nonterminals
    scores = np.full((n, n + 1, k), -np.inf, dtype=np.float64)
    back: dict[tuple[int, int, int], tuple[Any, ...]] = {}

    for i, token in enumerate(sequence):
        for rule_idx, (parent, emission, _) in enumerate(model.terminal_rules):
            score = float(model.log_terminal_probs[rule_idx] + emission.log_density(token))
            if score > scores[i, i + 1, parent]:
                scores[i, i + 1, parent] = score
                back[(i, i + 1, parent)] = ("terminal", rule_idx, token)

    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span
            for rule_idx in range(model.num_binary_rules):
                parent = int(model.binary_parents[rule_idx])
                left = int(model.binary_left[rule_idx])
                right = int(model.binary_right[rule_idx])
                rule_lp = float(model.log_binary_probs[rule_idx])
                for split in range(i + 1, j):
                    score = rule_lp + scores[i, split, left] + scores[split, j, right]
                    if score > scores[i, j, parent]:
                        scores[i, j, parent] = score
                        back[(i, j, parent)] = ("binary", rule_idx, split, left, right)

    root_score = float(scores[0, n, model.start_idx])
    if not np.isfinite(root_score):
        raise ValueError("sequence has zero probability under the grammar.")
    return _build_parse_node(model, back, scores, 0, n, model.start_idx)


def grammar_rule_table(model: HeterogeneousPCFGDistribution) -> list[dict[str, Any]]:
    """Return a flat, inspectable rule table for learned PCFGs."""
    rows: list[dict[str, Any]] = []
    for idx, (parent, left, right, prob) in enumerate(model.binary_rules):
        rows.append(
            {
                "type": "binary",
                "rule_index": idx,
                "parent": model.nonterminals[parent],
                "left": model.nonterminals[left],
                "right": model.nonterminals[right],
                "probability": float(prob),
            }
        )
    for idx, (parent, emission, prob) in enumerate(model.terminal_rules):
        rows.append(
            {
                "type": "terminal",
                "rule_index": idx,
                "parent": model.nonterminals[parent],
                "emission": emission,
                "probability": float(prob),
            }
        )
    return rows


def _build_parse_node(
    model: HeterogeneousPCFGDistribution,
    back: dict[tuple[int, int, int], tuple[Any, ...]],
    scores: np.ndarray,
    i: int,
    j: int,
    nt: int,
) -> PCFGParseNode:
    entry = back[(i, j, nt)]
    if entry[0] == "terminal":
        _, rule_idx, token = entry
        return PCFGParseNode(
            label=model.nonterminals[nt],
            span=(i, j),
            log_prob=float(scores[i, j, nt]),
            rule_index=int(rule_idx),
            rule_type="terminal",
            value=token,
        )
    _, rule_idx, split, left, right = entry
    left_node = _build_parse_node(model, back, scores, i, int(split), int(left))
    right_node = _build_parse_node(model, back, scores, int(split), j, int(right))
    return PCFGParseNode(
        label=model.nonterminals[nt],
        span=(i, j),
        log_prob=float(scores[i, j, nt]),
        rule_index=int(rule_idx),
        rule_type="binary",
        children=(left_node, right_node),
    )
