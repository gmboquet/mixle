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
    data = _validated_sequences(data, "data")
    vdata = None if vdata is None else _validated_sequences(vdata, "vdata")
    max_nonterminals = _positive_int(max_nonterminals, "max_nonterminals")
    max_its = _positive_int(max_its, "max_its")
    init_p = _finite_scalar(init_p, "init_p")
    if not 0.0 < init_p <= 1.0:
        raise ValueError("init_p must lie in (0, 1]")
    terminal_rule_mass = _finite_scalar(terminal_rule_mass, "terminal_rule_mass")
    if not 0.0 < terminal_rule_mass <= 1.0:
        raise ValueError("terminal_rule_mass must lie in (0, 1]")
    if terminal_rule_mass == 1.0 and any(len(sequence) > 1 for sequence in data):
        raise ValueError("terminal_rule_mass=1 creates no binary rules and cannot model multi-token sequences")
    rule_pseudo_count = _optional_nonnegative(rule_pseudo_count, "rule_pseudo_count")
    prune_threshold = _nonnegative_finite(prune_threshold, "prune_threshold")
    min_rule_prob = _nonnegative_finite(min_rule_prob, "min_rule_prob")
    if min_rule_prob >= 1.0:
        raise ValueError("min_rule_prob must be less than 1")
    if not isinstance(terminal_estimators, Sequence) or isinstance(terminal_estimators, (str, bytes)):
        raise ValueError("terminal_estimators must be a non-empty sequence of ParameterEstimator objects")
    terminal_estimators = tuple(terminal_estimators)
    if not terminal_estimators:
        raise ValueError("terminal_estimators must be a non-empty sequence of ParameterEstimator objects")
    if any(not isinstance(estimator, ParameterEstimator) for estimator in terminal_estimators):
        raise TypeError("every terminal_estimators entry must be a ParameterEstimator")
    if name is not None and not isinstance(name, str):
        raise ValueError("name must be a string or None")
    try:
        hash(start)
    except TypeError as exc:
        raise ValueError("start must be hashable") from exc
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be None or an integer from 0 through 2**32 - 1")
        seed = int(seed)
        if not 0 <= seed <= np.iinfo(np.uint32).max:
            raise ValueError("seed must be None or an integer from 0 through 2**32 - 1")

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
        try:
            enc_data = seq_encode(data, estimator=estimator)
            model = seq_initialize(enc_data, estimator, rng, p=init_p)
        except (TypeError, ValueError) as exc:
            raise ValueError("data is incompatible with the supplied terminal estimators") from exc
    else:
        _validate_initial_model(initial_model, estimator)
        model = initial_model
        try:
            enc_data = seq_encode(data, model=model)
        except (TypeError, ValueError) as exc:
            raise ValueError("data is incompatible with initial_model's terminal schema") from exc
    try:
        enc_vdata = None if vdata is None else seq_encode(vdata, model=model)
    except (TypeError, ValueError) as exc:
        raise ValueError("vdata is incompatible with the fitted grammar terminal schema") from exc
    history = [pcfg_log_likelihood(model, data)]
    validation_history = None if vdata is None else [pcfg_log_likelihood(model, vdata)]
    if not np.isfinite(history[0]):
        raise ValueError("the initial grammar assigns non-finite total log likelihood to data")
    if validation_history is not None and not np.isfinite(validation_history[0]):
        raise ValueError("the initial grammar assigns non-finite total log likelihood to vdata")

    for iteration in range(max_its):
        model = seq_estimate(enc_data, estimator, model)
        train_likelihood = pcfg_log_likelihood(model, data)
        if not np.isfinite(train_likelihood):
            raise RuntimeError(f"grammar fit produced non-finite training likelihood at iteration {iteration + 1}")
        history.append(train_likelihood)
        if enc_vdata is not None:
            validation_likelihood = float(np.sum(model.seq_log_density(enc_vdata[0][1])))
            if not np.isfinite(validation_likelihood):
                raise RuntimeError(
                    f"grammar fit produced non-finite validation likelihood at iteration {iteration + 1}"
                )
            validation_history.append(validation_likelihood)
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


def _validated_sequences(data: Any, name: str) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)) or len(data) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of non-empty token sequences")
    result = []
    for index, sequence in enumerate(data):
        if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes)) or len(sequence) == 0:
            raise ValueError(f"{name}[{index}] must be a non-empty token sequence")
        result.append(tuple(sequence))
    return tuple(result)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar")
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_nonnegative(value: Any, name: str) -> float | None:
    return None if value is None else _nonnegative_finite(value, name)


def _validate_initial_model(
    model: Any,
    estimator: InducedHeterogeneousPCFGEstimator,
) -> None:
    if not isinstance(model, HeterogeneousPCFGDistribution):
        raise TypeError("initial_model must be a HeterogeneousPCFGDistribution")
    prior = estimator._prior
    structural_pairs = (
        ("nonterminals", list(model.nonterminals), list(prior.nonterminals)),
        ("binary parents", model.binary_parents.tolist(), prior.binary_parents.tolist()),
        ("binary left children", model.binary_left.tolist(), prior.binary_left.tolist()),
        ("binary right children", model.binary_right.tolist(), prior.binary_right.tolist()),
        ("terminal parents", model.terminal_parents.tolist(), prior.terminal_parents.tolist()),
    )
    for label, actual, expected in structural_pairs:
        if actual != expected:
            raise ValueError(f"initial_model {label} do not match the induced grammar skeleton")
    expected_encoder = estimator.accumulator_factory().make().acc_to_encoder()
    if model.dist_to_encoder() != expected_encoder:
        raise ValueError("initial_model terminal encoders do not match terminal_estimators")
