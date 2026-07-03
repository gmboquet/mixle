"""``reason_discrete`` -- fuse multi-source evidence over a finite hypothesis set, with attribution.

The discrete sibling of :func:`mixle.reason.core.reason`: the latent is one of ``K`` alternatives
("which regime / fault / explanation"), each evidence source contributes a per-hypothesis
log-likelihood, and the answer is the exact posterior plus how many nats of uncertainty each source
removed. Sources can be raw log-likelihood vectors or **fitted mixle models** — one generative model
per hypothesis, scored on the raw observation (``model_evidence``) — so the same distributions you fit
elsewhere become reasoning evidence with no glue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.inference.belief import CategoricalBelief


def model_evidence(name: str, models: Any, x: Any) -> tuple[str, np.ndarray]:
    """Evidence from fitted mixle models: hypothesis ``k`` <-> ``models[k]``, scored on observation ``x``.

    Returns ``(name, log_lik)`` with ``log_lik[k] = models[k].log_density(x)``."""
    return name, np.asarray([float(m.log_density(x)) for m in models], dtype=np.float64)


@dataclass
class DiscreteAnswer:
    """The posterior over hypotheses plus per-source attribution (nats of entropy removed)."""

    belief: CategoricalBelief
    attribution: list[tuple[str, float]] = field(default_factory=list)

    @property
    def probs(self) -> np.ndarray:
        return self.belief.mean()

    def map(self) -> Any:
        return self.belief.map()

    def top(self, k: int = 3) -> list[tuple[Any, float]]:
        p = self.belief.probs
        order = np.argsort(-p)[: int(k)]
        return [(self.belief.labels[int(i)], float(p[i])) for i in order]

    def summary(self) -> str:
        lines = ["hypotheses: " + ", ".join(f"{h}={p:.3f}" for h, p in self.top(len(self.belief.labels)))]
        lines += [f"  {name}: removed {nats:+.2f} nats" for name, nats in self.attribution]
        lines.append(f"  residual entropy: {self.belief.entropy():.2f} nats")
        return "\n".join(lines)


def reason_discrete(prior: Any, evidence: Any) -> DiscreteAnswer:
    """Fold evidence into a categorical belief and return the posterior with per-source attribution.

    Args:
        prior: a :class:`CategoricalBelief`, an int ``K`` (uniform over ``K``), or a list of hypothesis
            labels (uniform over them).
        evidence: a sequence of ``(name, log_lik_vector)`` pairs — e.g. from :func:`model_evidence` —
            assimilated in order by exact Bayes.
    """
    if isinstance(prior, CategoricalBelief):
        belief = prior
    else:
        belief = CategoricalBelief.uniform(prior)
    attribution: list[tuple[str, float]] = []
    for name, ll in evidence:
        before = belief.entropy()
        belief = belief.update(ll)
        attribution.append((str(name), before - belief.entropy()))
    return DiscreteAnswer(belief, attribution)
