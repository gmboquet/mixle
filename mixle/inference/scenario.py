"""``simulate(scenario) -> Simulator`` -- on-the-fly conditional simulators for special scenarios.

See ``notes/designs/M2.md`` for the full design: source selection (explicit / registry / auto-
designed), why interventions must compose with evidence as ``do()`` FIRST then ``condition()``
(and how that is realized for a ``HeterogeneousBayesianNetwork``, whose ``do()`` result M0's
``condition()`` does not dispatch on), the HMM/state-space temporal-rollout extension past M0's
conditioned window, and the plausibility receipt (the scenario's evidence log-density under the
UNMODIFIED base model).

Distinct from :mod:`mixle.inference.simulate` (`Scenario`/`Simulator`/`simulate(model)`), a
narrower, BN-intervention-only tool already consumed by ``mixle.substrate.act``: this module does
not touch that one, and is imported under its own names (`mixle.inference.scenario.simulate`, not
re-exported under the same bare name from ``mixle.inference``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork
from mixle.inference.causal import do as _bn_do
from mixle.inference.condition import (
    FieldPath,
    ImpossibleEvidenceError,
    Posterior,
    _generate_weighted,
    _is_gaussian_like,
    _norm_evidence,
    _norm_path,
    _rng_seed,
    _safe_log_density,
    condition,
    do,
)
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.compute.posterior import MarkovChainLatentPosterior
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

__all__ = ["Scenario", "SimulationReceipt", "FieldPosterior", "Simulator", "simulate"]


@dataclass
class Scenario:
    """``{evidence, interventions, priors, horizon}`` -- what to build a simulator for.

    ``priors`` is the generic model-source configuration slot (see ``notes/designs/M2.md``):
    ``priors["registry"] = (Registry, name, version="latest")`` to fetch a stored model, or
    ``priors["data"]`` (+ optional ``priors["llm"]``) to auto-design one from scarce scenario data
    via :func:`mixle.task.design.design_model`, when ``base`` is not given directly to
    :func:`simulate`.
    """

    evidence: dict[FieldPath | int, Any] = field(default_factory=dict)
    interventions: dict[FieldPath | int, Any] = field(default_factory=dict)
    priors: dict[str, Any] = field(default_factory=dict)
    horizon: int = 1


@dataclass
class SimulationReceipt:
    """What ``simulate()`` actually did: plausibility of the evidence, and the conditioning health."""

    plausibility: float | None  # log p_base(evidence); None when there is no evidence to score
    plausibility_method: str  # "exact" | "sir" | "no-evidence"
    method: str  # conditioning method for the working posterior: "exact" | "sir" | "none"
    evidence_status: str = "possible"  # "possible" | "impossible"
    ess: float | None = None
    ess_ratio: float | None = None
    composition_order: str = "do-then-condition"
    warnings: list[str] = field(default_factory=list)


class FieldPosterior:
    """An empirical marginal for one field, built from rollout draws (``.mean()``/``.std()``/``.sample()``)."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def mean(self) -> float:
        return float(np.mean(self._values))

    def std(self) -> float:
        return float(np.std(self._values))

    def sample(self, n: int = 1, *, seed: int | None = None) -> np.ndarray:
        rng = RandomState(seed)
        idx = rng.randint(0, len(self._values), size=int(n))
        return self._values[idx]


# --------------------------------------------------------------------------------------------- #
# source selection
# --------------------------------------------------------------------------------------------- #


def _resolve_base(base: Any | None, scenario: Scenario) -> Any:
    if base is not None:
        return base
    priors = scenario.priors
    if "registry" in priors:
        registry, name, *rest = priors["registry"]
        version = rest[0] if rest else "latest"
        return registry.get(name, version)[0]
    if "data" in priors:
        from mixle.task.design import design_model

        designed = design_model(priors["data"], priors.get("llm"), fallback=True)
        return designed.fit(priors["data"])
    raise ValueError(
        "simulate(): no model source -- pass `base=` explicitly, or set "
        "`scenario.priors['registry'] = (registry, name)` or `scenario.priors['data'] = records`."
    )


# --------------------------------------------------------------------------------------------- #
# do() FIRST, then condition()
# --------------------------------------------------------------------------------------------- #


def _extract(record: Any, path: FieldPath) -> Any:
    v = record
    for i in path:
        v = v[i]
    return v


def _validate_disjoint_scenario_paths(
    interventions: dict[FieldPath, Any],
    evidence: dict[FieldPath, Any],
) -> None:
    """Reject evidence that an intervention would replace wholly or in part."""
    overlaps = []
    for intervention_path in interventions:
        for evidence_path in evidence:
            prefix_length = min(len(intervention_path), len(evidence_path))
            if intervention_path[:prefix_length] == evidence_path[:prefix_length]:
                overlaps.append((intervention_path, evidence_path))
    if overlaps:
        detail = ", ".join(f"{intervention!r} vs {observed!r}" for intervention, observed in overlaps)
        raise ValueError(
            "scenario evidence and interventions must address disjoint fields; "
            f"overlap would overwrite evidence ({detail})."
        )


def _condition_after_do_bn(
    net: HeterogeneousBayesianNetwork,
    interventions: dict[int, Any],
    evidence: dict[FieldPath, Any],
    *,
    n_particles: int,
    seed: int | None,
) -> tuple[list[tuple], np.ndarray | None, SimulationReceipt]:
    """``condition()`` has no dispatch rule for ``InterventionalNetwork`` (M0's ``do()`` result for a
    BN) -- realize "do() first, then condition()" here directly, in one ancestral pass over
    ``net.order`` per particle: an intervened field is FIXED with no weight contribution (its
    factor/edge is never consulted -- graph surgery), an evidenced field is FIXED and its factor's
    ``log_density`` becomes an importance weight (Bayesian update), exactly M0's own BN branch of
    ``_generate_weighted`` -- except intervened fields skip the weight term, which is the entire
    difference between ``do()`` and ``condition()``.
    """
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1")
    rng = RandomState(seed)
    top = {p[0]: v for p, v in evidence.items() if len(p) == 1}
    if any(len(p) != 1 for p in evidence):
        raise NotImplementedError("nested evidence is not supported for a HeterogeneousBayesianNetwork scenario.")
    by_child = {f.child: f for f in net.factors}
    invalid = (set(interventions) | set(top)) - set(by_child)
    if invalid:
        raise ValueError(f"scenario field indices do not exist on this Bayesian network: {sorted(invalid)}.")
    n_fields = len(net.factors)
    records: list[tuple] = []
    log_weights = np.empty(n_particles, dtype=np.float64)
    for i in range(n_particles):
        vals: list[Any] = [None] * n_fields
        lw = 0.0
        for idx in net.order:
            f = by_child[idx]
            if idx in interventions:
                vals[idx] = interventions[idx]  # do(): fixed, edge severed, no weight
            elif idx in top:
                vals[idx] = top[idx]  # condition(): fixed, weighted by its own factor
                lw += _safe_log_density(f, tuple(vals))
            else:
                vals[idx] = f.sample(vals, rng)
        records.append(tuple(vals))
        log_weights[i] = lw

    warnings: list[str] = []
    if not top:
        w_norm = np.full(n_particles, 1.0 / n_particles)
        ess = float(n_particles)
    else:
        finite = np.isfinite(log_weights)
        if not finite.any():
            ess = 0.0
            warnings.append("all importance weights are zero; evidence has zero probability under do(...).")
            receipt = SimulationReceipt(
                plausibility=None,
                plausibility_method="no-evidence",
                method="sir",
                evidence_status="impossible",
                ess=ess,
                ess_ratio=0.0,
                warnings=warnings,
            )
            return records, None, receipt
        else:
            m = log_weights[finite].max()
            w = np.where(finite, np.exp(log_weights - m), 0.0)
            w_norm = w / w.sum()
            ess = float(1.0 / np.sum(w_norm**2))
    ess_ratio = ess / n_particles
    if top and ess_ratio < 0.01:
        warnings.append(f"ESS ratio {ess_ratio:.4f} < 0.01 threshold -- evidence may be near-impossible under do(...).")
    receipt = SimulationReceipt(
        plausibility=None,
        plausibility_method="no-evidence",
        method="sir" if top else "none",
        ess=ess,
        ess_ratio=ess_ratio,
        warnings=warnings,
    )
    return records, w_norm, receipt


# --------------------------------------------------------------------------------------------- #
# HMM / state-space temporal rollout past the conditioned window
# --------------------------------------------------------------------------------------------- #


def _hmm_forward_rollout(
    model: HiddenMarkovModelDistribution,
    evidence: dict[int, Any],
    horizon: int,
    n: int,
    rng: RandomState,
) -> list[list[Any]]:
    """``n`` full-horizon emission rows: exact smoothed state posterior over ``[0, t_max]`` (same
    construction M0's ``_condition_hmm`` uses), then ancestral forward continuation of the DRAWN
    joint state path from ``t_max+1`` to ``horizon-1`` -- "forward-filtered simulation" continued
    past the last clamp.
    """
    n_states = model.n_states
    t_max = max(evidence) if evidence else -1
    t_max = max(t_max, -1)
    log_b = np.zeros((t_max + 1, n_states), dtype=np.float64)
    for t in range(t_max + 1):
        if t in evidence:
            for k in range(n_states):
                log_b[t, k] = _safe_log_density(model.topics[k], evidence[t])
    q = MarkovChainLatentPosterior(model.log_w, model.log_transitions, log_b) if t_max >= 0 else None
    trans = np.exp(model.log_transitions)
    rows: list[list[Any]] = []
    for _ in range(n):
        if q is not None:
            z = list(q.sample(rng))
        else:
            z = [int(rng.choice(n_states, p=np.exp(model.log_w)))]
        row: list[Any] = [None] * max(horizon, t_max + 1)
        for t in range(t_max + 1):
            row[t] = evidence[t] if t in evidence else model.topics[int(z[t])].sampler(seed=_rng_seed(rng)).sample()
        state = int(z[t_max]) if t_max >= 0 else z[0]
        for t in range(t_max + 1, horizon):
            state = int(rng.choice(n_states, p=trans[state]))
            row[t] = model.topics[state].sampler(seed=_rng_seed(rng)).sample()
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------- #
# plausibility receipt: log p_base(evidence)
# --------------------------------------------------------------------------------------------- #


def _frozen_point_mass(record: Any) -> Any:
    """An immutable snapshot of a fully observed record, for the point-mass rollout path.

    ``rollout`` returned ``[self._determined] * n``: n references to ONE object, itself holding the
    caller's evidence values by reference. Mutating one draw therefore changed every draw and the
    caller's evidence, and a later rollout saw the edit (MXR-080-1850). Containers are converted, not
    copied, because a copy of a list is still a list.
    """
    if isinstance(record, np.ndarray):
        frozen = record.copy()
        if frozen.dtype.hasobject:
            # ``copy()`` duplicates the POINTERS an object array holds, and ``write=False`` only stops
            # an element being rebound -- neither stops ``frozen[0].append(...)`` reaching the
            # caller's list. Freeze what the pointers point at, in place, before sealing the buffer.
            for index in np.ndindex(frozen.shape):
                frozen[index] = _frozen_point_mass(frozen[index])
        frozen.setflags(write=False)
        return frozen
    if isinstance(record, Mapping):
        return MappingProxyType({key: _frozen_point_mass(item) for key, item in record.items()})
    if isinstance(record, (str, bytes)):
        return record
    if isinstance(record, (set, frozenset)):
        return frozenset(_frozen_point_mass(item) for item in record)
    if isinstance(record, Sequence):
        return tuple(_frozen_point_mass(item) for item in record)
    return record


def _exact_draw_count(n: Any) -> int:
    """``n`` as an exact positive integer; ``int(n)`` accepted True and truncated 2.7 (MXR-080-1850)."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n must be an exact positive integer, not {type(n).__name__}")
    count = int(n)
    if count < 1:
        raise ValueError("n must be >= 1")
    return count


def _exact_evidence_log_density(model: Any, top: dict[int, Any]) -> float | None:
    if not top or not callable(getattr(model, "marginal", None)):
        return None
    idx = sorted(top)
    try:
        if isinstance(model, CompositeDistribution):
            sub = model.marginal(idx)  # CompositeDistribution.marginal sorts internally too
            value = tuple(top[i] for i in idx)
        elif _is_gaussian_like(model):
            sub = model.marginal(idx)  # order-preserving
            value = np.array([float(top[i]) for i in idx], dtype=np.float64)
        else:
            return None
        return float(sub.log_density(value))
    except (ValueError, TypeError, NotImplementedError):
        return None


def _sir_evidence_log_density(model: Any, ev: dict[FieldPath, Any], *, n_particles: int, rng: RandomState) -> float:
    """SNIS estimate of ``log p_base(evidence)`` -- reuses M0's own generative decomposition
    (``_generate_weighted``) rather than a second per-combinator dispatch table (see M2.md)."""
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1")
    log_weights = np.empty(n_particles, dtype=np.float64)
    for i in range(n_particles):
        _, lw = _generate_weighted(model, ev, rng)
        log_weights[i] = lw
    finite = log_weights[np.isfinite(log_weights)]
    if finite.size == 0:
        return float("-inf")
    return float(logsumexp(log_weights[np.isfinite(log_weights)]) - np.log(n_particles))


def plausibility_receipt(
    base: Any, evidence: dict[FieldPath | int, Any], *, n_particles: int = 4096, seed: int | None = None
) -> tuple[float | None, str]:
    """``(log p_base(evidence), method)`` -- the scenario's evidence log-density under the base model."""
    if not evidence:
        return None, "no-evidence"
    ev = _norm_evidence(evidence)
    top = {p[0]: v for p, v in ev.items() if len(p) == 1}
    if len(top) == len(ev):
        exact = _exact_evidence_log_density(base, top)
        if exact is not None:
            return exact, "exact"
    rng = RandomState(seed)
    return _sir_evidence_log_density(base, ev, n_particles=n_particles, rng=rng), "sir"


# --------------------------------------------------------------------------------------------- #
# Simulator / simulate()
# --------------------------------------------------------------------------------------------- #


class Simulator:
    """A scenario resolved to a runnable generator, with a plausibility + conditioning receipt."""

    def __init__(
        self,
        *,
        base: Any,
        scenario: Scenario,
        seed: int | None,
        n_particles: int = 4096,
    ) -> None:
        self.base = base
        self.scenario = scenario
        self._seed = seed
        self._n_particles = int(n_particles)
        if self._n_particles < 1:
            raise ValueError("n_particles must be >= 1")
        self._rng = RandomState(seed)

        interventions = _norm_evidence(scenario.interventions) if scenario.interventions else {}
        evidence = _norm_evidence(scenario.evidence) if scenario.evidence else {}
        _validate_disjoint_scenario_paths(interventions, evidence)

        plausibility, plaus_method = plausibility_receipt(
            base, dict(evidence), n_particles=n_particles, seed=_rng_seed(self._rng)
        )

        self._bn_records: tuple[list[tuple], np.ndarray | None] | None = None
        self._hmm_model: HiddenMarkovModelDistribution | None = None
        self._hmm_evidence: dict[int, Any] = {}
        self._posterior: Posterior | None = None
        self._determined: tuple | None = None
        working = base

        if interventions:
            if isinstance(base, HeterogeneousBayesianNetwork):
                if any(len(path) != 1 for path in interventions):
                    raise NotImplementedError(
                        "nested interventions are not supported for a HeterogeneousBayesianNetwork scenario."
                    )
                iv = {p[0]: v for p, v in interventions.items()}
                _bn_do(base, iv)  # validates field indices are in range; raises ValueError otherwise
                records, w_norm, receipt = _condition_after_do_bn(
                    base, iv, evidence, n_particles=n_particles, seed=_rng_seed(self._rng)
                )
                self._bn_records = (records, w_norm)
                self.receipt = SimulationReceipt(
                    plausibility=plausibility,
                    plausibility_method=plaus_method,
                    method=receipt.method,
                    evidence_status=receipt.evidence_status,
                    ess=receipt.ess,
                    ess_ratio=receipt.ess_ratio,
                    warnings=receipt.warnings,
                )
                return
            working = do(base, dict(interventions))

        if isinstance(working, HiddenMarkovModelDistribution) and (evidence or scenario.horizon > 1):
            if any(len(p) != 1 for p in evidence):
                raise NotImplementedError(
                    "nested evidence is not supported for a HiddenMarkovModelDistribution scenario."
                )
            self._hmm_model = working
            self._hmm_evidence = {p[0]: v for p, v in evidence.items()}
            method = "exact" if evidence else "none"
            evidence_status = "possible"
            if evidence:
                evidence_posterior = condition(working, dict(evidence), method="exact")
                evidence_status = evidence_posterior.receipt.evidence_status
                if not evidence_posterior.possible:
                    self._posterior = evidence_posterior
                    self._hmm_model = None
            self.receipt = SimulationReceipt(
                plausibility=plausibility,
                plausibility_method=plaus_method,
                method=method,
                evidence_status=evidence_status,
            )
            return

        determined = _fully_observed_record(working, evidence)
        if determined is not None:
            # Evidence fixes every field, so there is no residual distribution to condition to --
            # CompositeDistribution.condition would be asked for the empty composite and reject it.
            # That is not a degenerate corner: scoring a fully specified scenario is the whole point
            # of the plausibility receipt, and the plausibility itself is already computed above by
            # _exact_evidence_log_density, independently of conditioning. The scenario is a point
            # mass on the evidence, so every rollout draw is that record.
            self._determined = _frozen_point_mass(determined)
            self.receipt = SimulationReceipt(
                plausibility=plausibility,
                plausibility_method=plaus_method,
                method="none",
                evidence_status="possible" if np.isfinite(plausibility) else "impossible",
            )
            return

        if evidence:
            self._posterior = condition(working, dict(evidence), n_particles=n_particles, seed=_rng_seed(self._rng))
            self.receipt = SimulationReceipt(
                plausibility=plausibility,
                plausibility_method=plaus_method,
                method=self._posterior.receipt.method,
                evidence_status=self._posterior.receipt.evidence_status,
                ess=self._posterior.receipt.ess,
                ess_ratio=self._posterior.receipt.ess_ratio,
                warnings=list(self._posterior.receipt.warnings),
            )
        else:
            self._working = working
            self.receipt = SimulationReceipt(plausibility=plausibility, plausibility_method=plaus_method, method="none")

    def rollout(self, n: int = 1) -> list[Any]:
        """``n`` draws from the resolved scenario (do() applied, evidence conditioned, seed-fixed)."""
        n = _exact_draw_count(n)
        if self.receipt.evidence_status == "impossible":
            raise ImpossibleEvidenceError(
                "Cannot roll out a scenario whose evidence has zero probability after intervention."
            )
        if self._bn_records is not None:
            records, w_norm = self._bn_records
            if w_norm is None:
                raise ImpossibleEvidenceError(
                    "Cannot roll out a scenario whose evidence has zero probability after intervention."
                )
            idx = self._rng.choice(len(records), size=n, replace=True, p=w_norm)
            return [records[j] for j in idx]
        if self._hmm_model is not None:
            return _hmm_forward_rollout(self._hmm_model, self._hmm_evidence, int(self.scenario.horizon), n, self._rng)
        if self._determined is not None:
            # A fully observed scenario is a point mass on its evidence; every draw is that record.
            return [self._determined] * n
        if self._posterior is not None:
            return self._posterior.sample(n, seed=_rng_seed(self._rng))
        sampler = self._working.sampler(seed=_rng_seed(self._rng))
        out = sampler.sample(n)
        return list(out) if not isinstance(out, list) else out

    def posterior(self, field: FieldPath | int, *, n: int = 2000) -> FieldPosterior:
        """An empirical marginal for one field, from ``n`` rollout draws (see class docstring)."""
        path = _norm_path(field)
        rows = self.rollout(n)
        values = np.array([float(_extract(r, path)) for r in rows], dtype=np.float64)
        return FieldPosterior(values)


def _fully_observed_record(working: Any, evidence: dict) -> tuple | None:
    """The evidence as a full record when it fixes every field of ``working``, else ``None``.

    Deliberately narrow: only a CompositeDistribution observed at every top-level index qualifies.
    Any other model, any nested path, or any partial observation returns None and takes the ordinary
    conditioning path, so this cannot change the behaviour of a scenario that has a real residual.
    """
    from mixle.stats.combinator.composite import CompositeDistribution

    if not isinstance(working, CompositeDistribution) or not evidence:
        return None
    if any(len(path) != 1 or not isinstance(path[0], (int, np.integer)) for path in evidence):
        return None
    observed = {int(path[0]): value for path, value in evidence.items()}
    if set(observed) != set(range(working.count)):
        return None
    return tuple(observed[i] for i in range(working.count))


def simulate(scenario: Scenario, *, base: Any | None = None, seed: int | None = None) -> Simulator:
    """Assemble a generative model for ``scenario`` and package it as a :class:`Simulator`.

    Source selection (explicit ``base``, registry, or auto-designed from ``scenario.priors``),
    ``do()``-then-``condition()`` composition, HMM temporal rollout, and the plausibility receipt
    are all covered in ``notes/designs/M2.md``.
    """
    resolved = _resolve_base(base, scenario)
    return Simulator(base=resolved, scenario=scenario, seed=seed)
