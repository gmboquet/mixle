"""Exact per-part attribution for scores and decision margins.

``explain`` decomposes a model's score for one observation.
``explain_margin`` and ``explain_margin_mixture`` decompose the decision margin
between two named hypotheses for evidence ledgers and receipts.

Because mixle models are *generative and structured*, a prediction's score decomposes exactly — no
surrogate models, no sampling approximations:

  * a Composite / Record factorizes over fields:      ``log p(x) = sum_i log p_i(x_i)``
  * a learned Bayesian network factorizes over nodes: ``log p(x) = sum_i log P(x_i | parents)``
  * a Mixture adds the latent view: per-component responsibilities, then the winner's field breakdown
    -- plus an explicit ``correction`` term for the logsumexp normalizer, since a mixture's total
    log-density is not a sum of any single component's parts (the one place these structures are not
    purely additive). The ledger always satisfies ``sum(v for _, v in parts) + correction == total``
    to machine precision -- that identity is the point, not an approximation of it.

``explain(model, x)`` returns those parts with their exact log-likelihood contributions, sorted so the
most *suspicious* part (lowest contribution) is first: "which field makes this record unlikely" is read
straight off the model rather than estimated::

    ex = explain(model, record)
    ex.parts             # [(name, log-contribution), ...] ascending (most anomalous first)
    ex.total             # == model.log_density(record), exactly
    ex.correction         # 0 for Composite/BN (purely additive); the logsumexp residual for Mixture
    ex.responsibilities  # mixtures: posterior over components

``explain_margin(model, answer, runner_up)`` decomposes ``log p(answer) - log p(runner_up)`` -- the
decision margin between two named hypotheses -- into the same kind of per-factor ledger:

  * Composite / BN: ``answer``/``runner_up`` are two full candidate records; a field/factor that does not
    differ between them contributes exactly 0, so a single corrupted field is visible as the one part
    that does not collapse to 0.
  * Mixture used as a generative classifier (``explain_margin_mixture``): ``answer``/``runner_up`` are
    component indices (e.g. class labels) at the same observed ``x``; the margin is
    ``(log_w[a] + log p_a(x)) - (log_w[b] + log p_b(x))`` -- the mixture's logsumexp normalizer cancels
    exactly in this subtraction, so the margin ledger needs no correction term (it is computed and
    asserted at 0.0, not assumed).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Explanation:
    """Exact additive attribution of a log-density (or a decision margin between two hypotheses).

    ``correction`` is the explicitly named non-additive residual -- 0.0 for purely additive structures
    (Composite, Bayesian network, and any margin comparison, since the normalizer cancels there); the
    logsumexp term for a Mixture's absolute ``log p(x)``. ``sum(v for _, v in parts) + correction`` equals
    ``total`` exactly, for every supported structure -- this is asserted to machine precision in tests.
    """

    total: float
    parts: list[tuple[str, float]] = field(default_factory=list)
    correction: float = 0.0
    responsibilities: np.ndarray | None = None
    component: int | None = None  # the most responsible component (mixtures)

    def most_anomalous(self, k: int = 3) -> list[tuple[str, float]]:
        """Return the top contribution terms by anomaly score order."""
        return self.parts[: int(k)]

    def ledger_sum(self) -> float:
        """``sum(parts) + correction`` -- should equal ``total`` to machine precision; see class docstring."""
        return float(sum(v for _, v in self.parts)) + self.correction

    def is_exact(self, atol: float = 1e-9) -> bool:
        """Return whether ledger parts plus correction reconstruct the total."""
        return abs(self.ledger_sum() - self.total) <= atol

    def summary(self) -> str:
        """Render a human-readable anomaly ledger summary."""
        lines = [f"log p(x) = {self.total:.3f}"]
        if self.responsibilities is not None:
            probs = ", ".join(f"{p:.3f}" for p in self.responsibilities)
            lines.append(f"  component posterior: [{probs}] (component {self.component})")
        lines += [f"  {name}: {v:+.3f}" for name, v in self.parts]
        if self.correction:
            lines.append(f"  correction (logsumexp normalizer): {self.correction:+.3f}")
        return "\n".join(lines)


def _composite_parts(dist: Any, x: Any, prefix: str = "field") -> list[tuple[str, float]]:
    return [(f"{prefix}[{i}]", float(d.log_density(xi))) for i, (d, xi) in enumerate(zip(dist.dists, x))]


def _composite_margin_parts(dist: Any, answer: Any, runner_up: Any, prefix: str = "field") -> list[tuple[str, float]]:
    return [
        (f"{prefix}[{i}]", float(d.log_density(a_i)) - float(d.log_density(r_i)))
        for i, (d, a_i, r_i) in enumerate(zip(dist.dists, answer, runner_up))
    ]


def explain(model: Any, x: Any) -> Explanation:
    """Exact per-part attribution of ``model.log_density(x)`` (see module docstring)."""
    # learned Bayesian network: one part per node's conditional factor
    if hasattr(model, "factors") and hasattr(model, "order"):
        parts = [(f"field[{f.child}]|parents{tuple(f.parents)}", float(f.log_density(x))) for f in model.factors]
        total = float(sum(v for _, v in parts))
        return Explanation(total, sorted(parts, key=lambda p: p[1]))

    # mixture: latent posterior, then the winning component's own breakdown when it is a composite.
    # The winner's raw (prior + field) score is not the mixture's true total (that is a logsumexp over
    # every component) -- the gap is the explicit, named correction term, not an omission.
    if hasattr(model, "components") and hasattr(model, "posterior"):
        resp = np.asarray(model.posterior(x), dtype=np.float64).reshape(-1)
        winner = int(np.argmax(resp))
        comp = model.components[winner]
        if hasattr(comp, "dists"):
            field_parts = _composite_parts(comp, x, prefix=f"component[{winner}].field")
        else:
            field_parts = [(f"component[{winner}]", float(comp.log_density(x)))]
        parts = [(f"component[{winner}].prior", float(model.log_w[winner])), *field_parts]
        total = float(model.log_density(x))
        correction = total - float(sum(v for _, v in parts))
        return Explanation(
            total,
            sorted(parts, key=lambda p: p[1]),
            correction=correction,
            responsibilities=resp,
            component=winner,
        )

    # composite / record: one part per field, summing exactly to the total
    if hasattr(model, "dists"):
        parts = _composite_parts(model, x)
        return Explanation(float(sum(v for _, v in parts)), sorted(parts, key=lambda p: p[1]))

    return Explanation(float(model.log_density(x)), [("model", float(model.log_density(x)))])


def explain_margin(model: Any, answer: Any, runner_up: Any) -> Explanation:
    """Exact per-part attribution of the decision margin ``log p(answer) - log p(runner_up)``.

    ``answer``/``runner_up`` are two full candidate records for Composite/Bayesian-network models, so a
    field that is identical between them contributes exactly 0 -- the diagnostic for a single corrupted
    field. For a Mixture used as a generative classifier, use :func:`explain_margin_mixture` instead
    (the margin there needs the observed point plus two component indices, not two full records).
    """
    if hasattr(model, "components") and hasattr(model, "log_w"):
        raise TypeError(
            "explain_margin needs two full candidate records; a Mixture's hypotheses are component "
            "indices at one observed point -- call explain_margin_mixture(model, x, answer, runner_up)."
        )

    if hasattr(model, "factors") and hasattr(model, "order"):
        parts = [
            (
                f"field[{f.child}]|parents{tuple(f.parents)}",
                float(f.log_density(answer)) - float(f.log_density(runner_up)),
            )
            for f in model.factors
        ]
        total = float(model.log_density(answer)) - float(model.log_density(runner_up))
        return Explanation(total, sorted(parts, key=lambda p: p[1]), correction=total - float(sum(v for _, v in parts)))

    if hasattr(model, "dists"):
        parts = _composite_margin_parts(model, answer, runner_up)
        total = float(model.log_density(answer)) - float(model.log_density(runner_up))
        return Explanation(total, sorted(parts, key=lambda p: p[1]), correction=total - float(sum(v for _, v in parts)))

    total = float(model.log_density(answer)) - float(model.log_density(runner_up))
    return Explanation(total, [("model", total)], correction=0.0)


def explain_margin_mixture(model: Any, x: Any, answer: int, runner_up: int) -> Explanation:
    """Exact per-part attribution of the decision margin between two Mixture components at one point ``x``.

    ``answer``/``runner_up`` are component indices (e.g. class labels) of a Mixture used as a generative
    classifier: the margin is ``(log_w[answer] + log p_answer(x)) - (log_w[runner_up] + log p_runner_up(x))``.
    The mixture's logsumexp normalizer -- the same term :func:`explain` must name as a correction for the
    absolute ``log p(x)`` -- cancels exactly in this subtraction, so the margin ledger needs no correction
    (computed and asserted at 0.0, not assumed away).
    """
    a_idx, r_idx = int(answer), int(runner_up)
    ca, cr = model.components[a_idx], model.components[r_idx]
    wa, wr = float(model.log_w[a_idx]), float(model.log_w[r_idx])
    if hasattr(ca, "dists") and hasattr(cr, "dists"):
        parts = [("prior", wa - wr)] + [
            (f"field[{i}]", float(da.log_density(xi)) - float(dr.log_density(xi)))
            for i, (da, dr, xi) in enumerate(zip(ca.dists, cr.dists, x))
        ]
    else:
        parts = [("prior", wa - wr), ("component_density", float(ca.log_density(x)) - float(cr.log_density(x)))]
    total = (wa + float(ca.log_density(x))) - (wr + float(cr.log_density(x)))
    correction = total - float(sum(v for _, v in parts))
    return Explanation(total, sorted(parts, key=lambda p: p[1]), correction=correction)


# --- diagnose: ledger -> FaultReport for refinement-loop criticism --------------------------------

_FIX_VOCAB = frozenset({"add_edge", "upgrade_leaf", "split_region", "add_factor"})

# Below these sizes there is not enough data to estimate a scale (MAD) or
# co-occurrence rate. ``diagnose`` reports an insufficient-data result instead
# of a high-confidence finding from numerical noise or a single case.
_MIN_BACKGROUND = 4
_MIN_CASES_FOR_COOCCURRENCE = 3


@dataclass
class FaultReport:
    """Structural diagnosis built from :func:`explain` ledgers over failing cases.

    ``dominant`` names the structural element most responsible, when one rises
    above ordinary case-to-case variability. ``suggested_fix`` is one of
    :data:`_FIX_VOCAB` or empty. ``evidence`` ranks elements by adverse
    contribution.
    """

    dominant: str
    evidence: list[tuple[str, float]] = field(default_factory=list)
    suggested_fix: str = ""
    receipt: dict[str, Any] = field(default_factory=dict)


def diagnose(
    model: Any,
    cases: Sequence[Any],
    *,
    background: Sequence[Any] | None = None,
    min_z: float = 1.0,
    co_occurrence_threshold: float = 0.5,
) -> FaultReport:
    """Aggregate :func:`explain` ledgers into a structural fault report.

    Each case's per-part contributions are compared to ``background`` (a reference sample of typical
    cases; defaults to ``cases`` themselves, though a real, separately-supplied background is needed to
    detect a fault that is systematic across every case, since self-baselining against the failing set
    cancels a shift common to all of it) via a robust z-score (median/MAD per part name), so "adverse"
    is relative to that part's own normal variability, not a raw log-density magnitude.

    A single part scoring far from baseline is not, by itself, evidence of a
    structural defect. The closed-vocabulary fault this function actively
    detects is a missing dependency: two parts that are adverse on the same
    cases more often than chance predicts, indicating a possible unmodeled
    edge. When no such co-anomalous pair is found, the report remains
    empty/low-severity rather than guessing a fix.
    """
    bg = list(background) if background is not None else list(cases)
    if len(bg) < _MIN_BACKGROUND or not cases:
        return FaultReport(
            dominant="", evidence=[], suggested_fix="", receipt={"n_cases": len(cases), "n_background": len(bg)}
        )

    bg_explanations = [explain(model, x) for x in bg]
    names = sorted({name for ex in bg_explanations for name, _ in ex.parts})
    baseline: dict[str, float] = {}
    scale: dict[str, float] = {}
    for name in names:
        vals = np.asarray([v for ex in bg_explanations for n, v in ex.parts if n == name], dtype=np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826  # normal-consistent MAD -> std-equivalent
        baseline[name], scale[name] = med, max(mad, 1e-9)

    rows: list[dict[str, float]] = []
    for x in cases:
        ex = explain(model, x)
        rows.append({name: max(0.0, (baseline[name] - v) / scale[name]) for name, v in ex.parts if name in baseline})

    mean_adverse = {name: float(np.mean([r.get(name, 0.0) for r in rows])) for name in names}
    ranked = sorted(mean_adverse.items(), key=lambda kv: -kv[1])

    dominant, fix, severity = "", "", 0.0
    if len(ranked) > 1 and len(cases) >= _MIN_CASES_FOR_COOCCURRENCE:
        top_name, second_name = ranked[0][0], ranked[1][0]
        both = sum(1 for r in rows if r.get(top_name, 0.0) > min_z and r.get(second_name, 0.0) > min_z)
        either = sum(1 for r in rows if r.get(top_name, 0.0) > min_z or r.get(second_name, 0.0) > min_z)
        co_occurrence = (both / either) if either else 0.0
        if either > 0 and co_occurrence >= co_occurrence_threshold:
            dominant, fix, severity = f"{top_name}+{second_name}", "add_edge", co_occurrence

    return FaultReport(
        dominant=dominant,
        evidence=ranked,
        suggested_fix=fix,
        receipt={"n_cases": len(cases), "n_background": len(bg), "severity": round(severity, 4)},
    )
