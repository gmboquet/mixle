"""``explain`` -- exact per-part attribution of a model's score for one observation, and
``explain_margin``/``explain_margin_mixture`` -- exact per-part attribution of a DECISION MARGIN between
two named hypotheses (workstream H1 of the 0.6.3 frontier-capability plan: the answer-with-receipts
evidence ledger).

Because mixle models are *generative and structured*, a prediction's score decomposes exactly — no
surrogate models, no sampling approximations:

  * a Composite / Record factorizes over fields:      ``log p(x) = sum_i log p_i(x_i)``
  * a learned Bayesian network factorizes over nodes: ``log p(x) = sum_i log P(x_i | parents)``
  * a Mixture adds the latent view: per-component responsibilities, then the winner's field breakdown
    -- PLUS an explicit ``correction`` term for the logsumexp normalizer, since a mixture's total
    log-density is not a sum of any single component's parts (the one place these structures are not
    purely additive). The ledger always satisfies ``sum(v for _, v in parts) + correction == total``
    to machine precision -- that identity is the point, not an approximation of it.

``explain(model, x)`` returns those parts with their exact log-likelihood contributions, sorted so the
most *suspicious* part (lowest contribution) is first — "WHICH field makes this record unlikely" is read
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
    component indices (e.g. class labels) at the SAME observed ``x``; the margin is
    ``(log_w[a] + log p_a(x)) - (log_w[b] + log p_b(x))`` -- the mixture's logsumexp normalizer cancels
    EXACTLY in this subtraction, so the margin ledger needs no correction term (it is computed and
    asserted at 0.0, not assumed).
"""

from __future__ import annotations

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
        return self.parts[: int(k)]

    def ledger_sum(self) -> float:
        """``sum(parts) + correction`` -- should equal ``total`` to machine precision; see class docstring."""
        return float(sum(v for _, v in self.parts)) + self.correction

    def is_exact(self, atol: float = 1e-9) -> bool:
        return abs(self.ledger_sum() - self.total) <= atol

    def summary(self) -> str:
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
    # The winner's raw (prior + field) score is NOT the mixture's true total (that is a logsumexp over
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
    absolute ``log p(x)`` -- cancels EXACTLY in this subtraction, so the margin ledger needs no correction
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
