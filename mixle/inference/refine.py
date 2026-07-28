"""Diagnosis-directed correction compared with blind structure search.

If diagnosis-directed correction does not reach the same held-out target in
fewer trials than blind structure search (``learn_bayesian_network`` run with no
diagnosis, over growing prefixes of data) on the planted-fault benchmark, keep
blind search as the baseline.

Only the `add_edge` fix is translated to a concrete structural edit here, because that is the only fix
`mixle.inference.explain.diagnose` actually detects today (`upgrade_leaf`/`split_region`/`add_factor` are
recognized vocabulary with no detector wired yet -- see that module's docstring). A `FaultReport` naming
any other fix is reported as "not actionable," never guessed at.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.inference.bayesian_network import (
    HeterogeneousBayesianNetwork,
    _LinearGaussianFactor,
    _MarginalFactor,
    learn_bayesian_network,
)
from mixle.inference.explain import FaultReport, diagnose


def held_out_log_likelihood(model: Any, data: Sequence[tuple]) -> float:
    """Total log-density of a fitted network (or anything exposing ``dist_to_encoder``/``seq_log_density``)
    over held-out ``data`` -- the metric both paths below are compared on."""
    enc = model.dist_to_encoder().seq_encode(list(data))
    return float(np.sum(model.seq_log_density(enc)))


def _is_valid_score(value: float) -> bool:
    """Whether a held-out total is a score that can be compared at all.

    NaN means scoring FAILED, not that the model is bad: every ordered comparison against NaN is
    false, so ``after <= before`` is false and a candidate that could not be scored is reported as a
    verified improvement. ``+inf`` is likewise not a real total. ``-inf`` is a genuine, meaningful
    value here -- a model that assigns a held-out record zero probability -- so it stays comparable
    and simply loses.
    """
    return not (np.isnan(value) or value == np.inf)


def apply_add_edge_fix(
    model: HeterogeneousBayesianNetwork, fault: FaultReport, data: Sequence[tuple]
) -> HeterogeneousBayesianNetwork | None:
    """Apply only the fix ``diagnose`` actually named: add a linear-Gaussian edge between the two fields
    in ``fault.dominant`` (parsed from its ``"field[i]|...field[j]|..."`` shape), refit that one factor on
    ``data``, and return the corrected network -- every other factor is left untouched.

    Returns ``None`` (never guesses) when ``fault.suggested_fix`` is not ``"add_edge"``, or ``dominant``
    does not name exactly two fields (nothing dominant, or a shape this translation doesn't understand).

    ``fault.dominant`` names an undirected pair -- it carries no information about which field is the
    parent. Both orientations are fit and the better-fitting one on ``data`` is kept: field-index order
    (which field happens to have the smaller schema index) has no relationship to causal direction, so
    always taking ``parent, child = sorted(idx)`` risked silently adding the edge backwards.

    "Add edge" means ADD. The candidate keeps the child's existing parents, its categorical encoding
    and vector metadata, and is only built where doing so preserves the child's factor family; an
    orientation that would instead rewrite the child's model, or that would close a cycle, is reported
    as unsupported rather than performed under this function's name. ``None`` is returned when neither
    orientation is a semantics-preserving edit.
    """
    if fault.suggested_fix != "add_edge":
        return None
    idx = sorted({int(m) for m in re.findall(r"field\[(\d+)\]", fault.dominant)})
    if len(idx) != 2:
        return None
    cols = [[row[i] for row in data] for i in range(len(data[0]))]
    by_child = {f.child: f for f in model.factors}

    def _with_edge(parent: int, child: int) -> tuple[HeterogeneousBayesianNetwork, float] | None:
        """The child's factor refitted with ``parent`` ADDED, or None if that is not expressible here."""
        old = by_child.get(child)
        if old is None:
            return None
        # Only the linear-Gaussian family (and a continuous root, which is the zero-parent case of it)
        # can absorb another continuous parent without changing what the factor *is*. A GLM, a discrete
        # conditional table, or a vector CLG child would each be silently replaced by a scalar
        # linear-Gaussian -- a destructive model rewrite performed under the name "add edge".
        if not isinstance(old, (_LinearGaussianFactor, _MarginalFactor)):
            return None
        existing = list(getattr(old, "parents", []))
        if parent in existing or parent == child:
            return None  # already present / self-loop: nothing to add
        parents = existing + [parent]
        discrete = dict(getattr(old, "discrete", {}) or {})
        vec_dims = dict(getattr(old, "vec_dims", {}) or {})
        try:
            new_factor = _LinearGaussianFactor.fit(child, parents, cols, discrete, vec_dims=vec_dims)
            net = HeterogeneousBayesianNetwork([f for f in model.factors if f.child != child] + [new_factor])
            return net, held_out_log_likelihood(net, data)
        except Exception:  # noqa: BLE001 -- an inexpressible candidate is rejected, not a crash
            # Two distinct rejections land here, both meaning "this orientation is not a
            # semantics-preserving edit", which is exactly what this function reports as None:
            #   * The child is not actually a continuous field (a categorical root reaches
            #     `np.asarray(col, dtype=float)` inside fit and raises on its string levels) -- turning
            #     it into a scalar linear-Gaussian regression would be the destructive rewrite this
            #     guard exists to prevent.
            #   * The orientation closes a cycle, which the network constructor's topological sort
            #     rejects. Catching it per candidate keeps one cyclic orientation from aborting the
            #     search before the other, valid orientation has even been considered.
            return None

    candidates = [c for c in (_with_edge(idx[0], idx[1]), _with_edge(idx[1], idx[0])) if c is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[1])[0]


@dataclass
class TrialsToTarget:
    """How many trials a path needed to reach its target (``None`` = never reached), plus its score history."""

    n_trials: int | None
    final_model: Any
    history: list[float] = field(default_factory=list)


def directed_correction(
    model: Any,
    cases: Sequence[tuple],
    data: Sequence[tuple],
    held_out: Sequence[tuple],
    *,
    background: Sequence[tuple] | None = None,
) -> TrialsToTarget:
    """One diagnosis, one targeted edit, one verification.

    Costs exactly 1 trial if ``diagnose`` names an actionable fix and it verifiably improves held-out
    score over the original model; costs 0 (unreached) if the fix isn't actionable or doesn't verify --
    a correction that does not improve held-out is a failed diagnosis, logged as such, never silently
    kept anyway.

    A candidate is only "verified" against a comparison that actually happened. ``held_out`` must be
    nonempty, and both totals must be comparable scores: a NaN (scoring failed) fails ``after <=
    before`` just as a real improvement does, so without this check a model that could not be scored
    at all is returned as a successful one-trial correction.

    Raises:
        ValueError: if ``held_out`` is empty -- there is nothing to verify against.
    """
    held_out = list(held_out)
    if not held_out:
        raise ValueError("directed_correction needs a non-empty held-out set to verify a correction against")
    fault = diagnose(model, cases, background=background)
    fixed = apply_add_edge_fix(model, fault, data)
    before = held_out_log_likelihood(model, held_out)
    if fixed is None:
        return TrialsToTarget(n_trials=None, final_model=model, history=[before])
    after = held_out_log_likelihood(fixed, held_out)
    if not _is_valid_score(before) or not _is_valid_score(after) or after <= before:
        # Unreached, keeping the original model: either the correction did not improve held-out, or
        # one of the two scores is not a value an improvement can be established from.
        return TrialsToTarget(n_trials=None, final_model=model, history=[before, after])
    return TrialsToTarget(n_trials=1, final_model=fixed, history=[before, after])


def blind_search_trials_to_target(
    data: Sequence[tuple],
    held_out: Sequence[tuple],
    target_score: float,
    *,
    round_size: int = 10,
    max_rounds: int = 20,
    max_parents: int = 1,
    seed: int = 0,
) -> TrialsToTarget:
    """The blind baseline: NO diagnosis. Re-run ``learn_bayesian_network`` from scratch on growing
    prefixes of ``data`` (``round_size`` more examples each round, shuffled once up front) until the
    discovered structure's held-out score reaches ``target_score`` (typically
    :func:`directed_correction`'s own held-out score) or ``max_rounds`` is exhausted.
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(data))
    shuffled = [data[i] for i in order]
    history: list[float] = []
    model = None
    for round_idx in range(1, max_rounds + 1):
        n = min(round_idx * round_size, len(shuffled))
        model = learn_bayesian_network(shuffled[:n], max_parents=max_parents)
        score = held_out_log_likelihood(model, held_out)
        history.append(score)
        if score >= target_score:
            return TrialsToTarget(n_trials=round_idx, final_model=model, history=history)
        if n >= len(shuffled):
            break
    return TrialsToTarget(n_trials=None, final_model=model, history=history)
