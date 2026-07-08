"""Concept discovery: the library itself under selection (CARD L6, the culmination of the L-track loop).

One loop -- **discrepancy -> propose -> verify -> adopt -> remember** -- run over the library of
*families* rather than over any one model's parameters. When several tasks recur with a residual
pattern no family in the starting library explains well, this module fits a genuinely new family to
that pattern, gates it as a challenger against the status quo, and -- only if it survives the gate --
admits it to a small, inspectable, reversible registry (:class:`ConceptLibrary`). A future task whose
signature matches records where the concept won (:mod:`mixle.task.design_prior`), so the search reaches
for it directly instead of re-discovering it from scratch. Today's systematic error becomes tomorrow's
primitive.

This module adds no new modeling capability -- it is pure wiring over five existing subsystems, each
already carrying its own contract:

* **discrepancy** -- :func:`mixle.epistemic.discrepancy.discrepancy_report` measures how far a fitted
  champion's predictive distribution is from held-out data; tracked across tasks, a sustained gap is
  the "recurring unmodeled residual" signal that triggers a proposal.
* **propose** -- :func:`mixle.utils.automatic.profiling._profile_series` (the automatic profiler that
  landed this session's six new univariate detectors) is asked which family it would recommend for the
  data; that recommendation is the candidate new family.
* **verify / adopt** -- :func:`mixle.evolve.verify.challenger_beats_champion` is the single gate: the
  proposed family is only admitted if it significantly, non-regressively beats the current champion on
  held-out data.
* **remember** -- :func:`mixle.task.design_prior.record_accepted_recipe` over a
  :class:`mixle.task.edge.DesignModel` persists *where* the concept won, tagged by a coarse task
  signature, so :meth:`ConceptLibrary.query` can recommend it for a matching future task.

``ConceptLibrary`` is intentionally its own small registry -- not a mutation of
:mod:`mixle.stats`'s hardcoded distribution list -- so admission (and, symmetrically, revocation) is a
local, receipted, reversible act rather than a change to shared global state.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.epistemic.discrepancy import discrepancy_report
from mixle.evolve.objective import Objective, nll_objective, pointwise_log_density
from mixle.evolve.verify import Verdict, challenger_beats_champion
from mixle.inference.estimation import optimize
from mixle.stats import GaussianDistribution, GumbelDistribution, LaplaceDistribution, SkewNormalDistribution
from mixle.task.design_prior import record_accepted_recipe
from mixle.task.edge import DesignModel
from mixle.utils.automatic.profiling import _profile_series

# Family name -> zero-argument constructor for an *unfitted* prototype of that family. Keyed by the
# same names the automatic profiler's ``recommendation`` field uses, so a profiler recommendation can
# be turned directly into a fittable model without a second, redundant search.
_FAMILY_CONSTRUCTORS: dict[str, Callable[[], Any]] = {
    "gaussian": lambda: GaussianDistribution(0.0, 1.0),
    "laplace": lambda: LaplaceDistribution(0.0, 1.0),
    "skew_normal": lambda: SkewNormalDistribution(0.0, 1.0, 0.0),
    "gumbel": lambda: GumbelDistribution(0.0, 1.0),
}


def register_family(name: str, constructor: Callable[[], Any]) -> None:
    """Extend the set of families the loop can propose/reuse (beyond the built-in three)."""
    _FAMILY_CONSTRUCTORS[name] = constructor


def known_families() -> tuple[str, ...]:
    return tuple(_FAMILY_CONSTRUCTORS)


def _fit_family(name: str, data: Sequence[float]) -> Any:
    ctor = _FAMILY_CONSTRUCTORS.get(name)
    if ctor is None:
        raise KeyError(f"unknown family {name!r}; call register_family() to add a constructor for it.")
    return optimize(list(data), ctor().estimator(), out=None)


def _mdl_gain_bits(champion_model: Any, challenger_model: Any, data: Sequence[float]) -> float:
    """Bits saved by ``challenger_model`` over ``champion_model`` encoding ``data`` (positive = better).

    ``sum(log p_challenger - log p_champion) / ln(2)`` -- the (negative) code-length delta between the
    two models' Shannon codes for the same held-out sample, i.e. the MDL gain from having the
    challenger's family available.
    """
    champ_ll = pointwise_log_density(champion_model, data)
    chal_ll = pointwise_log_density(challenger_model, data)
    return float(np.sum(chal_ll - champ_ll) / math.log(2.0))


def task_signature(data: Sequence[float]) -> str:
    """A coarse, reusable descriptor of a task's data-generating shape.

    Deliberately coarse (kind + skew sign, not a fingerprint of the exact data): the point is that
    *different* tasks sharing the same hidden family land on the *same* signature, so
    :meth:`ConceptLibrary.query` can recommend a concept discovered on an earlier task to a later one.
    """
    arr = np.asarray(data, dtype=float).reshape(-1)
    profile = _profile_series((), "root", list(arr))
    skew_bucket = "sym"
    if profile.numeric_var and profile.numeric_var > 0 and profile.numeric_mean is not None:
        m3 = float(np.mean((arr - profile.numeric_mean) ** 3))
        g1 = m3 / profile.numeric_var**1.5
        if g1 > 0.15:
            skew_bucket = "right_skew"
        elif g1 < -0.15:
            skew_bucket = "left_skew"
    return f"{profile.kind}:{skew_bucket}"


@dataclass(frozen=True)
class AdmissionEvent:
    """One receipted, timestamped-by-task-index entry in a :class:`ConceptLibrary`'s audit log."""

    action: str  # "admit" | "revoke"
    family: str
    task_index: int
    evidence: dict[str, Any] = field(default_factory=dict)


class ConceptLibrary:
    """A small, inspectable registry of admitted concept families -- the "library" of CARD L6.

    Starts with ``base_families`` (present from the outset, never revocable via :meth:`revoke` -- they
    are the starting library, not something this loop discovered). Everything admitted afterwards is a
    first-class, named, evidenced entry that can be queried by task signature and, if it turns out to
    be a mistake, genuinely revoked -- removed from both the active family set and the design-prior
    ledger that backs :meth:`query`, not merely hidden.
    """

    def __init__(self, base_families: Sequence[str] = ("gaussian",)) -> None:
        self._base: tuple[str, ...] = tuple(base_families)
        self._admitted: dict[str, dict[str, Any]] = {}
        self._design = DesignModel(signature="concept_discovery", n_constraints=0)
        self.history: list[AdmissionEvent] = []

    @property
    def families(self) -> tuple[str, ...]:
        """Every family currently in the library: the starting base plus everything still admitted."""
        return tuple(dict.fromkeys((*self._base, *self._admitted)))

    def is_admitted(self, family: str) -> bool:
        return family in self._admitted

    def evidence_for(self, family: str) -> dict[str, Any] | None:
        return self._admitted.get(family)

    def admit(
        self,
        family: str,
        evidence: dict[str, Any],
        *,
        task_signature: str,
        task_index: int,
        quality: float,
    ) -> None:
        """Admit ``family`` to the library, receipted with ``evidence`` and recorded in the
        design-prior ledger under ``task_signature`` so :meth:`query` can recommend it later."""
        self._admitted[family] = dict(evidence)
        record_accepted_recipe(self._design, [0.0], quality, [], family=family, task_signature=task_signature)
        self.history.append(AdmissionEvent("admit", family, task_index, dict(evidence)))

    def revoke(self, family: str, *, task_index: int = -1, reason: str = "") -> None:
        """Undo an admission: ``family`` leaves the active family set AND the design-prior ledger, so
        :meth:`query` can never recommend it again -- genuinely reversible, not soft-hidden."""
        if family not in self._admitted:
            raise KeyError(f"{family!r} is not an admitted concept (nothing to revoke).")
        del self._admitted[family]
        keep = [i for i, tag in enumerate(self._design.tags) if tag.get("family") != family]
        self._design.X = [self._design.X[i] for i in keep]
        self._design.quality = [self._design.quality[i] for i in keep]
        self._design.violations = [self._design.violations[i] for i in keep]
        self._design.tags = [self._design.tags[i] for i in keep]
        self.history.append(AdmissionEvent("revoke", family, task_index, {"reason": reason}))

    def query(self, signature: str) -> str | None:
        """The design-prior recommendation for ``signature``: the best-recorded, still-admitted family
        seen for a matching task signature, or ``None`` if nothing qualifies."""
        candidates = [
            (tag.get("family"), q)
            for tag, q in zip(self._design.tags, self._design.quality)
            if tag.get("family") in self._admitted and tag.get("task_signature") == signature
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda kv: kv[1])[0]


@dataclass
class TaskResult:
    """One task's pass through the loop -- the receipt the acceptance test reads."""

    task_index: int
    task_signature: str
    discrepancy: float
    challenger_family: str | None
    reused_concept: bool  # a previously-admitted concept was queried and tried on this task
    admitted_family: str | None  # a NEW family was admitted on this task
    verdict: Verdict | None
    mdl_gain_bits: float  # bits saved on held-out data by the (accepted) challenger over the champion


def run_concept_discovery_loop(
    tasks: Sequence[Sequence[float]],
    *,
    library: ConceptLibrary | None = None,
    champion_family: str = "gaussian",
    objective: Objective | None = None,
    train_frac: float = 0.6,
    recurrence_window: int = 2,
    discrepancy_threshold: float = 0.003,
    min_effect: float = 0.01,
    nonnested: bool = False,
) -> tuple[ConceptLibrary, list[TaskResult]]:
    """Run discrepancy -> propose -> verify -> adopt -> remember over ``tasks``, in order.

    For every task: fit the ``champion_family`` on a train split, measure discrepancy against a
    held-out split (:func:`mixle.epistemic.discrepancy.discrepancy_report`). If the library already has
    an admitted concept recommended for this task's signature (:meth:`ConceptLibrary.query`), that
    concept is tried directly -- no re-search. Otherwise, once high discrepancy has recurred for
    ``recurrence_window`` consecutive tasks sharing a signature with no admitted concept yet, the
    automatic profiler proposes a new family; it is gated via
    :func:`mixle.evolve.verify.challenger_beats_champion` and, only if it passes, admitted.

    Returns the (possibly newly-created) :class:`ConceptLibrary` and one :class:`TaskResult` per task.
    """
    library = library if library is not None else ConceptLibrary(base_families=(champion_family,))
    objective = objective if objective is not None else nll_objective()

    results: list[TaskResult] = []
    recurrence_by_signature: dict[str, int] = {}
    admitted_for_signature: set[str] = set()

    for task_index, data in enumerate(tasks):
        arr = np.asarray(data, dtype=float).reshape(-1)
        n_train = max(8, int(len(arr) * train_frac))
        train, held_out = arr[:n_train], arr[n_train:]
        sig = task_signature(train)

        champion = _fit_family(champion_family, train)
        disc = float(discrepancy_report(champion, held_out).value)

        challenger_family: str | None = None
        reused = False
        admitted_family: str | None = None
        verdict: Verdict | None = None
        mdl_gain = 0.0

        recommended = library.query(sig)
        if recommended is not None and recommended != champion_family:
            # remember: reuse a previously-admitted concept for this task signature, no re-discovery.
            reused = True
            challenger_family = recommended
            challenger = _fit_family(recommended, train)
            verdict = challenger_beats_champion(
                champion,
                challenger,
                held_out,
                objective=objective,
                nonnested=nonnested,
                min_effect=min_effect,
                require_calibration=False,
            )
            if verdict.promote:
                mdl_gain = _mdl_gain_bits(champion, challenger, held_out)
        else:
            recurrence_by_signature[sig] = (
                recurrence_by_signature.get(sig, 0) + 1 if disc > discrepancy_threshold else 0
            )
            if recurrence_by_signature[sig] >= recurrence_window and sig not in admitted_for_signature:
                # propose: ask the automatic profiler what family it would fit to this recurring pattern.
                profile = _profile_series((), "root", list(train))
                proposed = profile.recommendation
                if proposed in _FAMILY_CONSTRUCTORS and proposed != champion_family:
                    challenger_family = proposed
                    challenger = _fit_family(proposed, train)
                    # verify: does the proposed family actually beat the champion, held out?
                    verdict = challenger_beats_champion(
                        champion,
                        challenger,
                        held_out,
                        objective=objective,
                        nonnested=nonnested,
                        min_effect=min_effect,
                        require_calibration=False,
                    )
                    if verdict.promote:
                        gain = _mdl_gain_bits(champion, challenger, held_out)
                        # adopt + remember: admit the family, record where it won.
                        library.admit(
                            proposed,
                            {"verdict": verdict.as_dict(), "task_signature": sig, "task_index": task_index},
                            task_signature=sig,
                            task_index=task_index,
                            quality=gain,
                        )
                        admitted_for_signature.add(sig)
                        admitted_family = proposed
                        mdl_gain = gain
                        recurrence_by_signature[sig] = 0

        results.append(
            TaskResult(
                task_index=task_index,
                task_signature=sig,
                discrepancy=disc,
                challenger_family=challenger_family,
                reused_concept=reused,
                admitted_family=admitted_family,
                verdict=verdict,
                mdl_gain_bits=mdl_gain,
            )
        )

    return library, results


__all__ = [
    "AdmissionEvent",
    "ConceptLibrary",
    "TaskResult",
    "known_families",
    "register_family",
    "run_concept_discovery_loop",
    "task_signature",
]
