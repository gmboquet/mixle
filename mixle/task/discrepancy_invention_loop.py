"""L5: the discrepancy -> invention loop -- wiring epistemic discrepancy into structural invention.

The full chain named by the roadmap item, end to end:

.. parsed-literal::

    discrepancy_report(champion, held_out)      -- 1. is the fitted champion's predictive off?
    capacity ladder + ceiling_report(...)        -- 2. "tune it" (same family, needs more capacity/data)
                                                        vs. "the structure CLASS is exhausted" (invention
                                                        trigger): a capacity ladder of same-family refits is
                                                        fit and, unlike a bare ``ceiling_report`` call, its
                                                        *trend* is read -- meaningful gain from more capacity
                                                        means "tune it", a plateau below target means
                                                        ceiling-bound.
    propose_structure(candidates, ...)           -- 3. search the composition grammar for a genuinely richer
                                                        structure (only reached when ceiling-bound).
    mixle.epistemic.loop.step(..., action_space=...) -- 4. EIG picks the cheapest probe that would
                                                        distinguish the champion from the surviving proposals.
    challenger_beats_champion(...)               -- 5. the same anti-regression gate L1/L4/L6 reuse.
    EpistemicJournal                             -- 6. every step above is appended as a DecisionRecord with
                                                        a human-readable rationale; the ordered rationale
                                                        list *is* the replayable reasoning chain.
    design_prior.rank_design_families(...)       -- 7. novelty of an accepted proposal, scored as surprise
                                                        relative to the design prior's expectation for that
                                                        structural family (``achieved - expected``, or
                                                        ``+inf`` when the family has never been tried).

Every piece above is an EXISTING module (:mod:`mixle.epistemic.discrepancy`/``loop``/``journal``,
:mod:`mixle.task.imagine`, :mod:`mixle.task.design_prior`, :mod:`mixle.evolve.verify`); this module adds
no new core math, only the orchestration that chains them into one auditable loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.epistemic.discrepancy import DiscrepancyResult, discrepancy_report
from mixle.epistemic.journal import DecisionRecord, EpistemicJournal
from mixle.epistemic.loop import EpistemicStep, step
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio
from mixle.evolve.objective import Objective
from mixle.evolve.verify import Verdict, challenger_beats_champion
from mixle.task.design_prior import rank_design_families, record_accepted_recipe
from mixle.task.edge import DesignModel
from mixle.task.imagine import CeilingReport, ImagineResult, StructuralCandidate, ceiling_report, propose_structure


def _mean_log_density(model: Any, data: Sequence[Any]) -> float:
    return float(np.mean([model.log_density(x) for x in data]))


def _sample_one(dist: Any, rng: np.random.RandomState) -> float:
    """Draw a single scalar from ``dist`` via its ``.sampler(seed).sample(n)`` (or ``.sample(n)``) surface."""
    sampler_fn = getattr(dist, "sampler", None)
    if callable(sampler_fn):
        seed = int(rng.randint(0, 2**31 - 1))
        result = float(np.asarray(sampler_fn(seed).sample(1)).reshape(-1)[0])
    else:
        result = float(np.asarray(dist.sample(1)).reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("probe simulations must produce finite scalar observations")
    return result


def _record_stage(
    journal: EpistemicJournal,
    portfolio: HypothesisPortfolio,
    *,
    observation: Any,
    rationale: str,
    surprise: float = 0.0,
) -> DecisionRecord:
    """Append a non-probe stage to ``journal`` -- a real snapshot of ``portfolio`` plus a rationale.

    Unlike the EIG-probe stage, these stages (discrepancy / ceiling / proposal / gate) don't advance
    the belief trajectory -- they *reason about* it -- so :class:`~mixle.epistemic.loop.EpistemicStep`
    is constructed directly here rather than via :func:`~mixle.epistemic.loop.step`, whose UPDATE/ACT
    arrows would otherwise be invoked redundantly against an unchanging portfolio just to get a shell.
    """
    return journal.append(
        EpistemicStep(
            observation=observation,
            portfolio_before=portfolio,
            portfolio_after=portfolio,
            surprise=surprise,
            next_action=None,
            next_action_eig=None,
        ),
        rationale=rationale,
    )


def default_probe_action_space(data: Sequence[Any], *, n_probes: int = 5) -> list[float]:
    """A default set of candidate probe locations: quantiles of ``data``'s own range.

    Querying near a quantile of the already-observed data is the natural default "where would the
    next observation land" grid when the caller has no domain-specific probe design of their own.
    """
    if isinstance(n_probes, bool) or not isinstance(n_probes, (int, np.integer)) or n_probes < 1:
        raise ValueError("n_probes must be a positive integer")
    arr = np.asarray(list(data), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("probe data must be a non-empty sequence of finite scalars")
    qs = np.linspace(0.15, 0.85, n_probes)
    return [float(np.quantile(arr, q)) for q in qs]


def default_probe_simulate_fn(
    hypothesis: Hypothesis, action: float, rng: np.random.RandomState, *, window: float
) -> float:
    """Simulate "if we probed near ``action``, what would ``hypothesis`` predict we'd observe?"

    Rejection-samples from ``hypothesis.payload`` (a fitted distribution) until a draw lands within
    ``window`` of ``action``. If the budget is exhausted, returns the closest *actual model draw*;
    returning ``action`` itself would fabricate identical evidence for every hypothesis precisely
    where their predictive mass does not support the requested action.
    """
    if not np.isfinite(action) or not np.isfinite(window) or window < 0.0:
        raise ValueError("probe action and window must be finite, with window non-negative")
    closest: float | None = None
    closest_distance = float("inf")
    for _ in range(200):
        y = _sample_one(hypothesis.payload, rng)
        distance = abs(y - action)
        if distance < closest_distance:
            closest = y
            closest_distance = distance
        if distance <= window:
            return y
    if closest is None:  # pragma: no cover - the fixed positive loop count makes this unreachable
        raise RuntimeError("probe simulation produced no draws")
    return closest


def _acceptance_mass(payload: Any, action: float, window: float, *, quadrature_nodes: int) -> float:
    """``P(|Y - action| <= window | payload)`` -- the probe law's ACTION-DEPENDENT normalizer.

    Exact from ``payload.cdf`` when the fitted model exposes one (correct for discrete payloads too,
    up to the single atom at the closed lower endpoint), otherwise composite Simpson quadrature of
    ``exp(log_density)`` over ``[action - window, action + window]``. Quadrature rather than Monte
    Carlo on purpose: this normalizer is legitimately tiny (``~1e-9`` for a probe placed in a
    hypothesis's far tail, which is exactly the most discriminating kind of probe), and a Monte Carlo
    estimate of it would read zero there and divide the truncated density by nothing. Quadrature over
    a bounded interval has no such starvation regime, needs only the ``log_density`` every payload in
    this loop already has, and is deterministic -- so the EIG does not depend on a nuisance seed.
    """
    lo, hi = float(action) - float(window), float(action) + float(window)
    cdf = getattr(payload, "cdf", None)
    if callable(cdf):
        mass = float(cdf(hi)) - float(cdf(lo))
        if math.isfinite(mass) and mass >= 0.0:
            return min(mass, 1.0)
    if hi <= lo:
        return 0.0
    xs = np.linspace(lo, hi, quadrature_nodes)
    densities = np.array([math.exp(float(payload.log_density(float(x)))) for x in xs], dtype=np.float64)
    if not np.all(np.isfinite(densities)):
        raise ValueError(
            f"probe hypothesis payload {type(payload).__name__!r} produced a non-finite density on "
            f"[{lo!r}, {hi!r}]; the truncated probe law's normalizer cannot be computed from it"
        )
    coefficients = np.ones(quadrature_nodes, dtype=np.float64)
    coefficients[1:-1:2] = 4.0
    coefficients[2:-1:2] = 2.0
    mass = float((hi - lo) / (quadrature_nodes - 1) / 3.0 * float(np.dot(coefficients, densities)))
    return float(min(max(mass, 0.0), 1.0))


class TruncatedProbeLikelihood:
    """``p(y | h, a)`` for the probe :func:`default_probe_simulate_fn` actually performs (MXR-080-1896).

    The probe rejection-samples ``hypothesis``'s predictive until a draw lands within ``window`` of
    ``action``, so the law it realizes is the predictive TRUNCATED to ``[a - w, a + w]``::

        p(y | h, a) = p(y | h) / P(|Y - a| <= w | h)   for |y - a| <= w,   else 0

    and that normalizer depends on ``a``. The loop previously scored its EIG with
    ``exp(h.payload.log_density(y))`` -- the untruncated ``p(y|h)``, byte-identical for every action --
    so the quantity being maximized was not the information gain of the experiment being commissioned.
    Measured on a two-Gaussian portfolio with five candidate probe locations, that mismatch reported
    ``-1.15`` nats where the true action-conditioned EIG was ``+0.04``: negative, which an expected
    information gain cannot be.

    ``0.0`` outside the window is the honest value, not a guard: under the declared experiment such an
    observation is impossible, and it is only reachable at all through
    :func:`default_probe_simulate_fn`'s documented budget-exhaustion fallback.
    :func:`mixle.epistemic.loop._portfolio_eig_nmc` scores those draws as contributing nothing rather
    than refusing them.

    The normalizer is memoized per ``(hypothesis.id, action)``. That is exact for the way this class is
    used -- one instance per :func:`run_discrepancy_invention_loop` call, over one portfolio whose ids
    are unique (its constructor enforces that) and whose payloads are fitted once and never mutated --
    and it is what makes the cost tolerable, since the estimator asks for the same few normalizers
    ``n_outer * (1 + n_inner)`` times per action. Do NOT share an instance across portfolios that reuse
    hypothesis ids for different payloads.
    """

    def __init__(self, *, window: float, quadrature_nodes: int = 129) -> None:
        window = float(window)
        if not math.isfinite(window) or window < 0.0:
            raise ValueError("probe window must be finite and non-negative")
        if (
            isinstance(quadrature_nodes, bool)
            or not isinstance(quadrature_nodes, (int, np.integer))
            or quadrature_nodes < 3
            or quadrature_nodes % 2 == 0
        ):
            # Composite Simpson needs an odd node count (an even number of panels) of at least 3.
            raise ValueError("quadrature_nodes must be an odd integer >= 3")
        self.window = window
        self.quadrature_nodes = int(quadrature_nodes)
        self._mass: dict[tuple[str, float], float] = {}

    def acceptance_mass(self, hypothesis: Hypothesis, action: float) -> float:
        """``P(|Y - action| <= window | hypothesis)``, memoized."""
        key = (hypothesis.id, float(action))
        mass = self._mass.get(key)
        if mass is None:
            mass = _acceptance_mass(
                hypothesis.payload, float(action), self.window, quadrature_nodes=self.quadrature_nodes
            )
            self._mass[key] = mass
        return mass

    def __call__(self, hypothesis: Hypothesis, action: Any, observation: Any) -> float:
        y = float(observation)
        a = float(action)
        if not np.isfinite(y) or not np.isfinite(a):
            raise ValueError("probe likelihood requires finite scalar action and observation")
        if abs(y - a) > self.window:
            return 0.0
        mass = self.acceptance_mass(hypothesis, a)
        if mass <= 0.0:
            # The accepted set carries no probability under this hypothesis (a zero-width window, or a
            # payload that puts no mass at all in it), so the truncated law does not exist and the
            # experiment is unrealizable here. 0.0 says "this hypothesis cannot produce this outcome",
            # which the EIG estimator reads as "no discrimination" rather than dividing by zero.
            return 0.0
        return float(math.exp(float(hypothesis.payload.log_density(y))) / mass)


@dataclass
class InventionResult:
    """The full outcome of one discrepancy -> invention loop run, with its replayable journal."""

    verdict: str  # 'target_met' | 'tune' | 'ceiling_bound'
    discrepancy: DiscrepancyResult
    ceiling: CeilingReport
    ceiling_bound: bool
    capacity_ladder_scores: dict[str, float]
    imagine: ImagineResult | None
    novelty_scores: dict[str, float] = field(default_factory=dict)
    probe_action: Any | None = None
    probe_eig: float | None = None
    gate_verdict: Verdict | None = None
    adopted_structure: str | None = None
    evidence_indices: dict[str, tuple[int, ...]] = field(default_factory=dict)
    journal: EpistemicJournal = field(default_factory=EpistemicJournal)


def score_design_prior_surprise(name: str, achieved_score: float, design: DesignModel) -> float:
    """Novelty of an accepted structural family, as design-prior surprise: ``achieved - expected``.

    ``expected`` is :func:`mixle.task.design_prior.rank_design_families`'s recorded mean quality for
    ``name``'s family, or ``-inf`` (via ``rank_design_families``'s own ``default_score`` convention) if
    the family has never been tried before -- which makes it maximally surprising by construction:
    ``+inf`` rather than an arbitrary finite number, so "genuinely never-seen-before" is never confused
    with "merely better than a middling prior."
    """
    prior_score = dict(rank_design_families(design, candidates=[name]))[name]
    if not math.isfinite(prior_score):
        return float("inf")
    return float(achieved_score) - prior_score


def _capacity_ladder_verdict(
    champion_score: float, target: float, tuning_scores: dict[str, float], *, plateau_tol: float
) -> tuple[bool, float]:
    """Read the *trend* across a same-family capacity ladder: does more capacity meaningfully help?

    ``tune`` if the best-tuned variant reaches ``target`` outright, or closes at least
    ``plateau_tol`` of the champion's gap to target -- "more capacity/data within this family is still
    buying real ground." Otherwise the ladder has plateaued short of target: no amount of same-family
    tuning closes the gap, which is the actual "structure CLASS is exhausted" invention trigger (as
    opposed to a bare single-point ``ceiling_report`` check, which cannot tell a plateau from a family
    that just hasn't been tuned yet).
    """
    best_tuned_score = max([champion_score, *tuning_scores.values()]) if tuning_scores else champion_score
    gap = target - champion_score
    if best_tuned_score >= target:
        return True, best_tuned_score
    if gap <= 0:
        return True, best_tuned_score
    gain = best_tuned_score - champion_score
    return (gain >= plateau_tol * gap), best_tuned_score


def run_discrepancy_invention_loop(
    champion_fit: Callable[[Sequence[Any]], Any],
    train: Sequence[Any],
    held_out: Sequence[Any],
    target: float,
    candidates: Sequence[StructuralCandidate],
    *,
    objective: Objective,
    tuning_variants: Sequence[Callable[[Sequence[Any]], Any]] = (),
    plateau_tol: float = 0.15,
    design: DesignModel | None = None,
    probe_action_space: Sequence[float] | None = None,
    probe_window: float | None = None,
    probe_reweight_n: int = 20,
    seed: int = 0,
) -> InventionResult:
    """The end-to-end L5 loop: discrepancy receipt -> ceiling verdict -> proposal -> EIG probe -> gate -> journal.

    ``champion_fit`` fits an instance of the CURRENT structural family; ``tuning_variants`` are other
    fits within that SAME family (more capacity, different regularization, more data, ...) used to read
    the capacity ladder's trend. ``candidates`` is the composition-grammar search space handed to
    :func:`~mixle.task.imagine.propose_structure`, only consulted when the ladder is ceiling-bound.
    ``design`` is the persistent design prior (:mod:`mixle.task.design_prior`) an adopted structure's
    family gets recorded into; a fresh, empty one is used if omitted. Every stage is appended to the
    returned journal with a plain-English ``rationale`` -- ``journal.records`` (or
    :func:`reconstruct_reasoning_chain`) is the full, ordered, replayable audit trail.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not np.isfinite(target):
        raise ValueError("target must be finite")
    if not np.isfinite(plateau_tol) or not 0.0 <= plateau_tol <= 1.0:
        raise ValueError("plateau_tol must be finite and in [0, 1]")
    if (
        isinstance(probe_reweight_n, bool)
        or not isinstance(probe_reweight_n, (int, np.integer))
        or probe_reweight_n < 1
    ):
        raise ValueError("probe_reweight_n must be a positive integer")
    train = list(train)
    if not train:
        raise ValueError("train must be non-empty")
    candidates = list(candidates)
    rng = np.random.RandomState(int(seed))
    journal = EpistemicJournal()
    design = design if design is not None else DesignModel(signature="discrepancy-invention-loop", n_constraints=0)
    held_out = list(held_out)
    minimum_panel = max(2, len(candidates))
    if len(held_out) < 4 * minimum_panel:
        raise ValueError(
            "held_out needs four disjoint evidence panels with at least "
            f"{minimum_panel} rows each (got {len(held_out)} rows)"
        )
    evidence_panels = np.array_split(rng.permutation(len(held_out)), 4)
    ceiling_panel = [held_out[int(i)] for i in evidence_panels[0]]
    proposal_panel = [held_out[int(i)] for i in evidence_panels[1]]
    probe_panel = [held_out[int(i)] for i in evidence_panels[2]]
    gate_panel = [held_out[int(i)] for i in evidence_panels[3]]
    evidence_indices = {
        "ceiling": tuple(int(i) for i in evidence_panels[0]),
        "proposal": tuple(int(i) for i in evidence_panels[1]),
        "probe": tuple(int(i) for i in evidence_panels[2]),
        "gate": tuple(int(i) for i in evidence_panels[3]),
    }

    champion_model = champion_fit(train)
    champion_score = _mean_log_density(champion_model, ceiling_panel)

    # 1. discrepancy receipt: does the champion's predictive diverge from the real held-out data?
    discrepancy = discrepancy_report(champion_model, ceiling_panel, metric="auto")
    root_portfolio = HypothesisPortfolio([Hypothesis("champion", champion_model)], np.array([1.0]), w_open=0.0)
    root_surprise = root_portfolio.surprise_score(
        ceiling_panel[0], lambda h, y: float(np.exp(h.payload.log_density(y)))
    )
    _record_stage(
        journal,
        root_portfolio,
        observation=ceiling_panel[0],
        surprise=root_surprise,
        rationale=(
            f"discrepancy detected: metric={discrepancy.metric} value={discrepancy.value:.6g} "
            f"degraded={discrepancy.degraded}"
        ),
    )

    # 2. ceiling verdict: 'tune it' (capacity ladder still gaining) vs. 'ceiling-bound' (plateaued).
    ceiling = ceiling_report(champion_score, target)
    tuning_scores = {
        f"tune_variant_{i}": _mean_log_density(variant_fit(train), ceiling_panel)
        for i, variant_fit in enumerate(tuning_variants)
    }
    tune_helps, best_tuned_score = _capacity_ladder_verdict(
        champion_score, target, tuning_scores, plateau_tol=plateau_tol
    )

    if ceiling.met:
        verdict_label = "target_met"
    elif tune_helps:
        verdict_label = "tune"
    else:
        verdict_label = "ceiling_bound"
    ceiling_bound = verdict_label == "ceiling_bound"

    _record_stage(
        journal,
        root_portfolio,
        observation=champion_score,
        rationale=(
            f"ceiling verdict: {verdict_label} (champion_held_out={champion_score:.6g} "
            f"best_tuned={best_tuned_score:.6g} target={target:.6g}) -- "
            + (
                "the capacity ladder plateaus below target: no amount of same-family tuning closes the "
                "gap, this is the structural-invention trigger."
                if ceiling_bound
                else "same-family tuning still meaningfully closes the gap (or already meets target); "
                "no invention needed."
            )
        ),
    )

    result = InventionResult(
        verdict=verdict_label,
        discrepancy=discrepancy,
        ceiling=ceiling,
        ceiling_bound=ceiling_bound,
        capacity_ladder_scores=tuning_scores,
        imagine=None,
        evidence_indices=evidence_indices,
        journal=journal,
    )
    if not ceiling_bound:
        return result

    # 3. propose_structure: search the composition grammar for a genuinely richer structure.
    imagine = propose_structure(
        list(candidates),
        train,
        proposal_panel,
        ceiling,
        baseline_model=champion_model,
        seed=int(seed),
    )
    result.imagine = imagine
    _record_stage(
        journal,
        root_portfolio,
        observation=imagine.breaks_ceiling,
        rationale=(
            "structure proposal: "
            + (
                f"breaks the ceiling with {imagine.breaks_ceiling!r}"
                if imagine.breaks_ceiling
                else "no candidate broke the ceiling"
            )
            + "; verdicts="
            + ", ".join(
                f"{v.name}={'accepted' if v.accepted else 'rejected(' + v.reason + ')'}" for v in imagine.verdicts
            )
        ),
    )

    accepted = [v for v in imagine.verdicts if v.accepted]
    if not accepted:
        return result

    result.novelty_scores = {v.name: score_design_prior_surprise(v.name, v.held_out_score, design) for v in accepted}

    winning_name = imagine.breaks_ceiling or max(accepted, key=lambda v: v.held_out_score).name
    winning_model = imagine.fitted_models[winning_name]

    # 4. EIG probe: which experiment would most efficiently distinguish champion from the surviving
    #    proposals? Reweight on the real held-out evidence, then let ACT pick the highest-EIG probe.
    accepted_models = {verdict.name: imagine.fitted_models[verdict.name] for verdict in accepted}
    hyps = [Hypothesis("champion", champion_model)] + [Hypothesis(name, m) for name, m in accepted_models.items()]
    portfolio = HypothesisPortfolio(hyps, np.full(len(hyps), 1.0 / len(hyps)), w_open=0.0)

    def likelihood_fn(h: Hypothesis, y: Any) -> float:
        return float(np.exp(h.payload.log_density(y)))

    action_space = (
        list(probe_action_space) if probe_action_space is not None else default_probe_action_space(probe_panel)
    )
    if not action_space or not np.all(np.isfinite(np.asarray(action_space, dtype=np.float64))):
        raise ValueError("probe_action_space must contain finite scalar actions")
    window = (
        probe_window if probe_window is not None else float(np.std(np.asarray(probe_panel, dtype=np.float64))) * 0.5
    )
    if not np.isfinite(window) or window < 0.0:
        raise ValueError("probe_window must be finite and non-negative")

    def simulate_fn(h: Hypothesis, action: Any, r: np.random.RandomState) -> Any:
        return default_probe_simulate_fn(h, action, r, window=window)

    # MXR-080-1896: the density that scores the EIG must be the density of the law `simulate_fn`
    # draws from -- the predictive TRUNCATED to the probe window, whose normalizer depends on the
    # action. `likelihood_fn` above stays the UPDATE density: those observations are real held-out
    # rows, not action-conditioned probe outcomes, so conditioning them on a window would be wrong.
    probe_likelihood = TruncatedProbeLikelihood(window=window)

    subsample = probe_panel[: min(len(probe_panel), int(probe_reweight_n))]
    probe_outcome = None
    for i, y in enumerate(subsample):
        is_last = i == len(subsample) - 1
        probe_outcome = step(
            portfolio,
            y,
            likelihood_fn,
            action_space=action_space if is_last else None,
            simulate_fn=simulate_fn if is_last else None,
            action_likelihood=probe_likelihood if is_last else None,
            rng=rng,
        )
        portfolio = probe_outcome.portfolio_after

    result.probe_action = probe_outcome.next_action
    result.probe_eig = probe_outcome.next_action_eig
    journal.append(
        probe_outcome,
        action_considered=action_space,
        rationale=(
            f"EIG probe: cheapest distinguishing experiment is action={probe_outcome.next_action!r} "
            f"(EIG={probe_outcome.next_action_eig!r}), among candidates {list(accepted_models)}"
        ),
    )

    # 5. gate: does the winning proposal actually beat the champion (same anti-regression gate as L1/L4/L6)?
    gate_verdict = challenger_beats_champion(champion_model, winning_model, gate_panel, objective=objective)
    result.gate_verdict = gate_verdict
    _record_stage(
        journal,
        portfolio,
        observation=winning_name,
        rationale=(
            f"gate verdict: favored={gate_verdict.favored} promote={gate_verdict.promote} "
            f"delta={gate_verdict.delta:.6g} winning_structure={winning_name!r}"
        ),
    )

    if gate_verdict.promote:
        result.adopted_structure = winning_name
        winning_verdict = next(v for v in imagine.verdicts if v.name == winning_name)
        record_accepted_recipe(design, [0.0], winning_verdict.held_out_score, [], family=winning_name)

    return result


def reconstruct_reasoning_chain(journal: EpistemicJournal) -> list[str]:
    """The ordered list of every stage's rationale -- the human-readable replay of the full chain."""
    return [r.rationale for r in journal if r.rationale is not None]


__all__ = [
    "InventionResult",
    "TruncatedProbeLikelihood",
    "default_probe_action_space",
    "default_probe_simulate_fn",
    "reconstruct_reasoning_chain",
    "run_discrepancy_invention_loop",
    "score_design_prior_surprise",
]
