"""``explain`` -- exact per-part attribution of a model's score for one observation.

Because mixle models are *generative and structured*, a prediction's score decomposes exactly — no
surrogate models, no sampling approximations:

  * a Composite / Record factorizes over fields:      ``log p(x) = sum_i log p_i(x_i)``
  * a learned Bayesian network factorizes over nodes: ``log p(x) = sum_i log P(x_i | parents)``
  * a Mixture adds the latent view: per-component responsibilities, then the winner's field breakdown.

``explain(model, x)`` returns those parts with their exact log-likelihood contributions, sorted so the
most *suspicious* part (lowest contribution) is first — "WHICH field makes this record unlikely" is read
straight off the model rather than estimated::

    ex = explain(model, record)
    ex.parts             # [(name, log-contribution), ...] ascending (most anomalous first)
    ex.total             # == model.log_density(record), exactly
    ex.responsibilities  # mixtures: posterior over components
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Explanation:
    """Exact additive attribution of ``log p(x)`` (plus the latent posterior for mixtures)."""

    total: float
    parts: list[tuple[str, float]] = field(default_factory=list)
    responsibilities: np.ndarray | None = None
    component: int | None = None  # the most responsible component (mixtures)

    def most_anomalous(self, k: int = 3) -> list[tuple[str, float]]:
        return self.parts[: int(k)]

    def summary(self) -> str:
        lines = [f"log p(x) = {self.total:.3f}"]
        if self.responsibilities is not None:
            probs = ", ".join(f"{p:.3f}" for p in self.responsibilities)
            lines.append(f"  component posterior: [{probs}] (component {self.component})")
        lines += [f"  {name}: {v:+.3f}" for name, v in self.parts]
        return "\n".join(lines)


def _composite_parts(dist: Any, x: Any, prefix: str = "field") -> list[tuple[str, float]]:
    return [(f"{prefix}[{i}]", float(d.log_density(xi))) for i, (d, xi) in enumerate(zip(dist.dists, x))]


def explain(model: Any, x: Any) -> Explanation:
    """Exact per-part attribution of ``model.log_density(x)`` (see module docstring)."""
    # learned Bayesian network: one part per node's conditional factor
    if hasattr(model, "factors") and hasattr(model, "order"):
        parts = [(f"field[{f.child}]|parents{tuple(f.parents)}", float(f.log_density(x))) for f in model.factors]
        total = float(sum(v for _, v in parts))
        return Explanation(total, sorted(parts, key=lambda p: p[1]))

    # mixture: latent posterior, then the winning component's own breakdown when it is a composite
    if hasattr(model, "components") and hasattr(model, "posterior"):
        resp = np.asarray(model.posterior(x), dtype=np.float64).reshape(-1)
        winner = int(np.argmax(resp))
        comp = model.components[winner]
        if hasattr(comp, "dists"):
            parts = _composite_parts(comp, x, prefix=f"component[{winner}].field")
        else:
            parts = [(f"component[{winner}]", float(comp.log_density(x)))]
        return Explanation(
            float(model.log_density(x)), sorted(parts, key=lambda p: p[1]), responsibilities=resp, component=winner
        )

    # composite / record: one part per field, summing exactly to the total
    if hasattr(model, "dists"):
        parts = _composite_parts(model, x)
        return Explanation(float(sum(v for _, v in parts)), sorted(parts, key=lambda p: p[1]))

    return Explanation(float(model.log_density(x)), [("model", float(model.log_density(x)))])
