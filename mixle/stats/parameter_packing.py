"""Primary-parameter packing for extrapolation-based EM acceleration (SQUAREM).

This lives in the stats layer on purpose: the (extract, rebuild) handlers must name concrete
distribution families, which orchestration-level modules (mixle.inference.em -- see its module
docstring and compute_metadata_test's guard) are architecturally barred from doing. The consumer is
:class:`mixle.inference.em.SquaremEM`, which imports only the packer FUNCTION.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _squarem_leaf_handlers() -> dict[type, tuple[Callable[[Any], list[float]], Callable[[Any, list[float]], Any]]]:
    """Primary-parameter (extract, rebuild) pairs for :func:`squarem_packer`, keyed by leaf type.

    Coordinates are unconstrained (log for positive parameters, log-simplex for probabilities) so
    extrapolated points map back to valid parameters; rebuilding goes through CONSTRUCTORS so every
    derived constant is recomputed rather than left stale (the serialized state carries e.g. a
    Gaussian's ``log_const``, which naive state write-back would corrupt).
    """
    from mixle.stats import (
        CategoricalDistribution,
        ExponentialDistribution,
        GaussianDistribution,
        PoissonDistribution,
    )

    def cat_extract(d: Any) -> list[float]:
        return [float(np.log(max(d.pmap[key], 1e-300))) for key in sorted(d.pmap)]

    def cat_rebuild(d: Any, vals: list[float]) -> Any:
        logs = np.asarray(vals)
        p = np.exp(logs - logs.max())
        p /= p.sum()
        pmap = dict(zip(sorted(d.pmap), (float(v) for v in p)))
        return CategoricalDistribution(pmap, default_value=d.default_value, name=d.name)

    return {
        GaussianDistribution: (
            lambda d: [float(d.mu), float(np.log(d.sigma2))],
            lambda d, v: GaussianDistribution(float(v[0]), float(np.exp(v[1])), name=d.name),
        ),
        ExponentialDistribution: (
            lambda d: [float(np.log(d.beta))],
            lambda d, v: ExponentialDistribution(float(np.exp(v[0])), name=d.name),
        ),
        PoissonDistribution: (
            lambda d: [float(np.log(d.lam))],
            lambda d, v: PoissonDistribution(float(np.exp(v[0])), name=d.name),
        ),
        CategoricalDistribution: (cat_extract, cat_rebuild),
    }


def squarem_packer(
    model: Any,
) -> tuple[Callable[[Any], np.ndarray], Callable[[np.ndarray], Any]]:
    """Build ``(pack, unpack)`` for :class:`SquaremEM` over ``model``'s primary parameters.

    Supported out of the box: :class:`~mixle.stats.MixtureDistribution` whose components are
    Gaussian / Exponential / Poisson / Categorical leaves or Composites of them, with no priors
    attached (a MAP fit changes what the fixed point is; packing would silently ignore it).
    Anything else raises ``NotImplementedError`` with the escape hatch named: pass an explicit
    ``packer=(pack, unpack)`` to :class:`SquaremEM` for custom models.

    ``pack(model) -> theta`` and ``unpack(theta) -> model`` round-trip losslessly (asserted in
    tests); mixture weights travel as log-weights re-normalized on unpack.
    """
    from mixle.stats import CompositeDistribution, MixtureDistribution

    handlers = _squarem_leaf_handlers()

    def leaf_pair(d: Any) -> tuple[Callable[[Any], list[float]], Callable[[Any, list[float]], Any], int]:
        pair = handlers.get(type(d))
        if pair is None:
            raise NotImplementedError(
                "squarem_packer has no primary-parameter handler for %s; pass an explicit "
                "packer=(pack, unpack) to SquaremEM for this model." % type(d).__name__
            )
        if getattr(d, "prior", None) is not None:
            raise NotImplementedError(
                "squarem_packer does not extrapolate MAP fits (a %s carries a prior): the prior "
                "changes the fixed point, and packing only the likelihood parameters would silently "
                "drop it. Pass an explicit packer that includes the prior's contribution." % type(d).__name__
            )
        return pair[0], pair[1], len(pair[0](d))

    if not isinstance(model, MixtureDistribution):
        raise NotImplementedError(
            "squarem_packer currently supports MixtureDistribution models; pass an explicit "
            "packer=(pack, unpack) to SquaremEM for %s." % type(model).__name__
        )

    def component_factors(comp: Any) -> list[Any]:
        return list(comp.dists) if isinstance(comp, CompositeDistribution) else [comp]

    template = [component_factors(c) for c in model.components]
    is_composite = [isinstance(c, CompositeDistribution) for c in model.components]
    sizes = [[leaf_pair(f)[2] for f in factors] for factors in template]

    def pack(m: Any) -> np.ndarray:
        theta: list[float] = []
        for factors in (component_factors(c) for c in m.components):
            for f in factors:
                theta.extend(leaf_pair(f)[0](f))
        theta.extend(float(v) for v in np.log(np.maximum(np.asarray(m.w, dtype=np.float64), 1e-300)))
        return np.asarray(theta, dtype=np.float64)

    def unpack(theta: np.ndarray) -> Any:
        pos = 0
        comps = []
        for ci, factors in enumerate(template):
            rebuilt = []
            for fi, f in enumerate(factors):
                width = sizes[ci][fi]
                rebuilt.append(leaf_pair(f)[1](f, [float(v) for v in theta[pos : pos + width]]))
                pos += width
            comps.append(CompositeDistribution(tuple(rebuilt)) if is_composite[ci] else rebuilt[0])
        logw = np.asarray(theta[pos:], dtype=np.float64)
        w = np.exp(logw - logw.max())
        return MixtureDistribution(comps, list(w / w.sum()))

    return pack, unpack
