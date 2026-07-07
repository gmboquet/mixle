"""``ModalityView`` -- a modality as a structured belief node, not an embedding (workstream F1).

Mainstream cross-modal systems collapse every modality into one shared fixed-width vector and reason by
similarity in that space. That is lossy not because the vector is too small, but because it is a
*bottleneck with a fixed basis chosen before the question is known*, and it discards structure the next
hop depends on. Mixle's native object is a composable distribution, not a tensor, so the alternative is
concrete: each modality is its own typed mixle sub-model -- a real ``SequenceEncodableProbabilityDistribution``
that can be scored (``log p(x)``) and sampled -- with a declared symmetry group it is invariant to (e.g.
translation for a conv+pool image feature, permutation for a set, "none" for an ordered categorical
label). A :class:`ModalityView` never collapses evidence into a shared vector; if a receiver needs a
vector, that happens last, lazily, per-receiver (see :mod:`mixle.substrate.context`'s ``ContextPacket``,
workstream E) -- not as the primary representation every modality is forced through up front.

This module delivers the contract and the composition (:class:`ModalityGraph` fields multiple named
views into one joint), plus the falsifiable control every cross-modal claim in this plan must carry: a
structured belief must be compared against a distilled fixed-width bottleneck of the *same information*,
with the accuracy/calibration gap measured, not assumed (see ``modality_test.py``'s bottleneck test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModalityView:
    """One modality as a typed structured belief: a real mixle distribution plus its symmetry group.

    ``dist`` is any fitted ``SequenceEncodableProbabilityDistribution`` -- a Categorical over a discrete
    label, a Gaussian/StudentT over a measurement, a hybrid neural density (workstream A) over an
    embedding- or image-shaped field, a Bayesian network over a structured record. ``symmetry_group``
    names the invariance the modality's own encoder/leaf declares (e.g. ``"none"``, ``"translation"``,
    ``"permutation"``, ``"rotation"``) -- a modeling statement, not an accident of architecture (see
    workstream A3's quotient leaf, which this contract is written to accept once it lands).
    """

    name: str
    dist: Any
    symmetry_group: str = "none"
    notes: list[str] = field(default_factory=list)

    def score(self, x: Any) -> float:
        """``log p(x)`` under this modality's own structured belief -- never a similarity to a vector."""
        return float(self.dist.log_density(x))

    def sample(self, n: int = 1, *, seed: int | None = None) -> Any:
        """Draw from this modality's own belief (its sampler, not a decoded shared-space vector)."""
        return self.dist.sampler(seed).sample(n)

    def seq_score(self, xs: Any) -> Any:
        """Vectorized ``log p(x)`` over a batch, via the modality's own encoder -- for scoring many
        candidates without a Python loop when the wrapped distribution supports it."""
        enc = self.dist.dist_to_encoder().seq_encode(list(xs))
        return self.dist.seq_log_density(enc)


@dataclass
class ModalityGraph:
    """A named collection of :class:`ModalityView` for one entity -- the joint the belief walk (F3) hops
    across. Composition stays structured: a receiver reads named modalities and their own scores, never
    a concatenated shared vector (that would recreate the bottleneck this contract exists to avoid)."""

    views: dict[str, ModalityView] = field(default_factory=dict)

    def add(self, view: ModalityView) -> ModalityGraph:
        self.views[view.name] = view
        return self

    def __getitem__(self, name: str) -> ModalityView:
        return self.views[name]

    def __contains__(self, name: str) -> bool:
        return name in self.views

    def modalities(self) -> list[str]:
        return sorted(self.views)

    def joint_score(self, observations: dict[str, Any]) -> dict[str, float]:
        """Per-modality ``log p(x)`` for the given ``{modality_name: x}`` observations -- the exact,
        named per-factor decomposition a Composite gives (workstream H), one entry per modality."""
        return {name: self.views[name].score(x) for name, x in observations.items() if name in self.views}
