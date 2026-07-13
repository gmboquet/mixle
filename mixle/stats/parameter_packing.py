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
        LaplaceDistribution,
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
        LaplaceDistribution: (
            lambda d: [float(d.mu), float(np.log(d.b))],
            lambda d, v: LaplaceDistribution(float(v[0]), float(np.exp(v[1])), name=d.name, keys=d.keys),
        ),
        CategoricalDistribution: (cat_extract, cat_rebuild),
    }


def squarem_packer(
    model: Any,
) -> tuple[Callable[[Any], np.ndarray], Callable[[np.ndarray], Any]]:
    """Build ``(pack, unpack)`` for :class:`SquaremEM` over ``model``'s primary parameters.

    Supported out of the box: recursively nested :class:`~mixle.stats.MixtureDistribution` and
    :class:`~mixle.stats.CompositeDistribution` nodes with Gaussian / Laplace / Exponential /
    Poisson / Categorical leaves and no priors attached (a MAP fit changes the fixed point).
    Anything else raises ``NotImplementedError`` with the escape hatch named: pass an explicit
    ``packer=(pack, unpack)`` to :class:`SquaremEM` for custom models.

    ``pack(model) -> theta`` and ``unpack(theta) -> model`` round-trip losslessly (asserted in
    tests); mixture weights travel as log-weights re-normalized on unpack.
    """
    from mixle.stats import CompositeDistribution, MixtureDistribution

    handlers = _squarem_leaf_handlers()

    def prior_is_set(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (list, tuple)):
            return any(prior_is_set(child) for child in value)
        return True

    def ensure_no_prior(d: Any) -> None:
        getter = getattr(d, "get_prior", None)
        prior = getter() if callable(getter) else getattr(d, "prior", None)
        if prior_is_set(prior):
            raise NotImplementedError(
                "squarem_packer does not extrapolate MAP fits (a %s carries a prior): the prior "
                "changes the fixed point. Pass an explicit packer that includes the prior's "
                "contribution." % type(d).__name__
            )

    def leaf_pair(d: Any) -> tuple[Callable[[Any], list[float]], Callable[[Any, list[float]], Any], int]:
        pair = handlers.get(type(d))
        if pair is None:
            raise NotImplementedError(
                "squarem_packer has no primary-parameter handler for %s; pass an explicit "
                "packer=(pack, unpack) to SquaremEM for this model." % type(d).__name__
            )
        ensure_no_prior(d)
        return pair[0], pair[1], len(pair[0](d))

    if not isinstance(model, MixtureDistribution):
        raise NotImplementedError(
            "squarem_packer currently supports MixtureDistribution models; pass an explicit "
            "packer=(pack, unpack) to SquaremEM for %s." % type(model).__name__
        )

    def width(node: Any) -> int:
        ensure_no_prior(node)
        if isinstance(node, MixtureDistribution):
            return sum(width(child) for child in node.components) + node.num_components
        if isinstance(node, CompositeDistribution):
            return sum(width(child) for child in node.dists)
        return leaf_pair(node)[2]

    expected_width = width(model)

    def pack_node(node: Any, template: Any, theta: list[float]) -> None:
        if type(node) is not type(template):
            raise TypeError(
                "squarem_packer model structure changed from %s to %s." % (type(template).__name__, type(node).__name__)
            )
        ensure_no_prior(node)
        if isinstance(template, MixtureDistribution):
            if len(node.components) != len(template.components):
                raise ValueError("squarem_packer mixture arity changed.")
            for child, child_template in zip(node.components, template.components):
                pack_node(child, child_template, theta)
            theta.extend(float(v) for v in np.log(np.maximum(np.asarray(node.w, dtype=np.float64), 1e-300)))
            return
        if isinstance(template, CompositeDistribution):
            if len(node.dists) != len(template.dists):
                raise ValueError("squarem_packer composite arity changed.")
            for child, child_template in zip(node.dists, template.dists):
                pack_node(child, child_template, theta)
            return
        theta.extend(leaf_pair(node)[0](node))

    def unpack_node(template: Any, theta: np.ndarray, pos: int) -> tuple[Any, int]:
        if isinstance(template, MixtureDistribution):
            components = []
            for child in template.components:
                rebuilt, pos = unpack_node(child, theta, pos)
                components.append(rebuilt)
            logw = theta[pos : pos + template.num_components]
            pos += template.num_components
            weights = np.exp(logw - np.max(logw))
            weights /= weights.sum()
            return MixtureDistribution(components, weights, name=template.name), pos
        if isinstance(template, CompositeDistribution):
            dists = []
            for child in template.dists:
                rebuilt, pos = unpack_node(child, theta, pos)
                dists.append(rebuilt)
            return CompositeDistribution(tuple(dists)), pos
        extract, rebuild, leaf_width = leaf_pair(template)
        del extract
        values = [float(value) for value in theta[pos : pos + leaf_width]]
        return rebuild(template, values), pos + leaf_width

    def pack(m: Any) -> np.ndarray:
        theta: list[float] = []
        pack_node(m, model, theta)
        return np.asarray(theta, dtype=np.float64)

    def unpack(theta: np.ndarray) -> Any:
        values = np.asarray(theta, dtype=np.float64)
        if values.ndim != 1 or len(values) != expected_width:
            raise ValueError("squarem_packer expected a vector of length %d." % expected_width)
        rebuilt, pos = unpack_node(model, values, 0)
        if pos != expected_width:  # pragma: no cover - recursive width and rebuild are paired
            raise RuntimeError("squarem_packer internal width mismatch.")
        return rebuilt

    return pack, unpack
