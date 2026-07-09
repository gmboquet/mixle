"""Cross-modal reasoning as conditional inference in a shared-latent joint (workstream L2).

The loop this module runs -- discrepancy -> propose -> verify -> adopt -> remember -- is generic
across altitudes; L2's edge is that "cross-modal" needs no bespoke machinery here. A composite over
heterogeneous fields (one Gaussian field standing in for an "image embedding," one Categorical field
standing in for a "text label," etc.) already ties every field to a single shared latent regime the
moment those fields are wrapped as one :class:`~mixle.stats.combinator.composite.CompositeDistribution`
and mixed over a component index via :class:`~mixle.stats.latent.mixture.MixtureDistribution` -- the
component index *is* the shared latent regime spanning modalities, and per-component parameter
``keys=`` ties (see :class:`~mixle.stats.latent.mixture.MixtureEstimator`) are the same mechanism used
to pool statistics across otherwise-independent per-modality estimators when fitting such a joint.

``MixtureDistribution.conditional`` already implements "condition on any subset, infer any other
subset" for exactly this shape of joint: it returns the full posterior mixture over the unobserved
coordinates, itself scoreable and sampleable. :class:`CrossModalJoint` is a thin, name-addressed
wrapper around that machinery so callers condition on modality NAMES ("image", "text", ...) instead of
bare composite field indices, and so a further subset of the remaining fields can be requested (not
just "everything unobserved").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.compute.pdist import SequenceEncodableProbabilityDistribution
from mixle.stats.latent.mixture import MixtureDistribution


@dataclass(frozen=True)
class CrossModalJoint:
    """A joint over named modalities sharing one latent regime (a mixture component index).

    ``names`` fixes the modality-name -> composite-field-index mapping; ``joint`` is a
    :class:`~mixle.stats.latent.mixture.MixtureDistribution` whose components are
    :class:`~mixle.stats.combinator.composite.CompositeDistribution` instances over ``len(names)``
    heterogeneous fields, in ``names`` order. The mixture weights are the shared latent's prior
    ``p(regime)``; each component is ``p(modality_0, modality_1, ... | regime=k)``.
    """

    names: tuple[str, ...]
    joint: MixtureDistribution

    @classmethod
    def from_components(
        cls,
        names: Sequence[str],
        component_fields: Sequence[Sequence[SequenceEncodableProbabilityDistribution]],
        weights: Sequence[float],
    ) -> CrossModalJoint:
        """Build a shared-latent joint from per-regime per-modality distributions.

        ``component_fields[k]`` is the sequence of ``len(names)`` per-modality distributions for
        latent regime ``k`` (in ``names`` order); ``weights[k]`` is ``p(regime=k)``. Each regime is
        wrapped as one :class:`CompositeDistribution` over the (heterogeneous) modality fields and the
        regimes are mixed, so the resulting joint's own component index is exactly the shared latent
        tying every modality together.
        """
        names = tuple(names)
        if any(len(fields) != len(names) for fields in component_fields):
            raise ValueError(
                f"every regime must supply one distribution per modality ({len(names)} modalities), "
                f"got field counts {[len(f) for f in component_fields]}"
            )
        components = [CompositeDistribution(list(fields)) for fields in component_fields]
        joint = MixtureDistribution(components, w=np.asarray(weights, dtype=np.float64))
        return cls(names=names, joint=joint)

    def _index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"unknown modality {name!r}; known modalities are {self.names!r}") from None

    def infer(
        self,
        observed: Mapping[str, Any],
        target: Sequence[str] | None = None,
    ) -> MixtureDistribution:
        """Posterior over ``target`` modalities given observed values for any OTHER subset.

        ``observed`` maps modality name -> its observed value, for any subset (including the empty
        set, which returns the marginal/prior). ``target`` names the modalities to infer the joint
        posterior over; defaults to every modality not in ``observed``. Every ``target`` name must be
        absent from ``observed`` (you cannot condition on and infer the same modality).

        Returns a :class:`~mixle.stats.latent.mixture.MixtureDistribution` over a
        ``len(target)``-tuple, in ``target`` order (a 1-modality target is a mixture over a
        1-tuple, matching :class:`CompositeDistribution`'s own convention for a single field).
        """
        observed_idx = {self._index_of(name): value for name, value in observed.items()}
        remaining = [name for name in self.names if name not in observed]
        if target is None:
            target = remaining
        target = list(target)
        missing = [name for name in target if name not in remaining]
        if missing:
            raise ValueError(
                f"target modalities must not be observed and must be known: bad target(s) {missing!r}, "
                f"observed={list(observed)!r}, known modalities={list(self.names)!r}"
            )
        posterior = self.joint.conditional(observed_idx)
        rel_idx = [remaining.index(name) for name in target]
        # Build the sub-composite directly off ``component.dists`` (rather than via
        # ``CompositeDistribution.marginal``, which always returns fields in SORTED index order) so
        # the returned tuple matches the caller's requested ``target`` order exactly.
        marginal_components = [
            CompositeDistribution([component.dists[i] for i in rel_idx]) for component in posterior.components
        ]
        return MixtureDistribution(marginal_components, w=posterior.w.copy())
