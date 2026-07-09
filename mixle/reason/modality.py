"""Structured modality views for belief-centered cross-modal reasoning.

Each modality is represented as its own typed Mixle sub-model: a
``SequenceEncodableProbabilityDistribution`` that can be scored and sampled,
with a declared symmetry group such as ``"translation"``, ``"permutation"``,
or ``"none"``. A :class:`ModalityView` keeps that structure visible until a
receiver explicitly asks for a task-specific representation.

This module provides the view contract and :class:`ModalityGraph`, which groups
multiple named views for one entity. Cross-modal claims should be compared
against an appropriate fixed-width baseline using measured accuracy and
calibration, not assumed from representation shape alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModalityView:
    """One modality as a typed structured belief: a real mixle distribution plus its symmetry group.

    ``dist`` is any fitted ``SequenceEncodableProbabilityDistribution``: for
    example a categorical model over labels, a Gaussian or Student-t model over
    measurements, a neural density over an embedding-shaped field, or a
    Bayesian network over a structured record. ``symmetry_group`` names the
    invariance the modality declares, such as ``"none"``, ``"translation"``,
    ``"permutation"``, or ``"rotation"``.
    """

    name: str
    dist: Any
    symmetry_group: str = "none"
    notes: list[str] = field(default_factory=list)

    def score(self, x: Any) -> float:
        """Return ``log p(x)`` under this modality's structured belief."""
        return float(self.dist.log_density(x))

    def sample(self, n: int = 1, *, seed: int | None = None) -> Any:
        """Draw from this modality's own sampler."""
        return self.dist.sampler(seed).sample(n)

    def seq_score(self, xs: Any) -> Any:
        """Return vectorized ``log p(x)`` over a batch via the modality encoder."""
        enc = self.dist.dist_to_encoder().seq_encode(list(xs))
        return self.dist.seq_log_density(enc)


@dataclass
class ModalityGraph:
    """A named collection of :class:`ModalityView` for one entity.

    Belief walks hop across this joint representation. A receiver reads named
    modalities and their own scores rather than an implicit shared vector.
    """

    views: dict[str, ModalityView] = field(default_factory=dict)

    def add(self, view: ModalityView) -> ModalityGraph:
        """Add a modality view and return the graph for chaining."""
        self.views[view.name] = view
        return self

    def __getitem__(self, name: str) -> ModalityView:
        return self.views[name]

    def __contains__(self, name: str) -> bool:
        return name in self.views

    def modalities(self) -> list[str]:
        """Return modality names in sorted order."""
        return sorted(self.views)

    def joint_score(self, observations: dict[str, Any]) -> dict[str, float]:
        """Return per-modality ``log p(x)`` for named observations."""
        return {name: self.views[name].score(x) for name, x in observations.items() if name in self.views}
