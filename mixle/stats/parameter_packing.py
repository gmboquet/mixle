"""Primary-parameter packing for extrapolation-based EM acceleration (SQUAREM).

This lives in the stats layer on purpose: the (extract, rebuild) handlers must name concrete
distribution families, which orchestration-level modules (mixle.inference.em -- see its module
docstring and compute_metadata_test's guard) are architecturally barred from doing. The consumer is
:class:`mixle.inference.em.SquaremEM`, which imports only the packer FUNCTION.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Any

import numpy as np


def _categorical_coordinate_keys(dist: Any) -> list[Any]:
    return sorted(
        dist.pmap,
        key=lambda key: (
            type(key).__module__,
            type(key).__qualname__,
            repr(key),
        ),
    )


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
        keys = _categorical_coordinate_keys(d)
        probs = np.asarray([d.pmap[key] for key in keys], dtype=np.float64)
        if not np.isclose(float(probs.sum()), 1.0, rtol=1.0e-7, atol=1.0e-10):
            raise ValueError("squarem_packer requires categorical probabilities to sum to 1.")
        if not np.any(probs > 0.0):
            raise ValueError("squarem_packer requires at least one positive categorical probability.")
        return [float(np.log(value)) for value in probs if value > 0.0]

    def cat_rebuild(d: Any, vals: list[float]) -> Any:
        coordinate_keys = _categorical_coordinate_keys(d)
        active = [key for key in coordinate_keys if d.pmap[key] > 0.0]
        p = _normalized_positive_coordinates(vals, "categorical probabilities")
        if len(p) != len(active):
            raise ValueError("squarem_packer categorical coordinate width changed.")
        by_key = dict(zip(active, (float(value) for value in p)))
        pmap = {key: by_key.get(key, 0.0) for key in d.pmap}
        return CategoricalDistribution(
            pmap,
            default_value=d.default_value,
            name=d.name,
            keys=d.keys,
        )

    return {
        GaussianDistribution: (
            lambda d: [float(d.mu), float(np.log(d.sigma2))],
            lambda d, v: GaussianDistribution(
                float(v[0]),
                _positive_from_log(v[1], "Gaussian sigma2"),
                name=d.name,
                keys=d.keys,
            ),
        ),
        ExponentialDistribution: (
            lambda d: [float(np.log(d.beta))],
            lambda d, v: ExponentialDistribution(
                _positive_from_log(v[0], "Exponential beta"),
                name=d.name,
                keys=d.keys,
            ),
        ),
        PoissonDistribution: (
            lambda d: [float(np.log(d.lam))],
            lambda d, v: PoissonDistribution(
                _positive_from_log(v[0], "Poisson lambda"),
                name=d.name,
                keys=d.keys,
            ),
        ),
        LaplaceDistribution: (
            lambda d: [float(d.mu), float(np.log(d.b))],
            lambda d, v: LaplaceDistribution(
                float(v[0]),
                _positive_from_log(v[1], "Laplace scale"),
                name=d.name,
                keys=d.keys,
            ),
        ),
        CategoricalDistribution: (cat_extract, cat_rebuild),
    }


def _positive_from_log(value: float, name: str) -> float:
    try:
        result = math.exp(float(value))
    except OverflowError as exc:
        raise ValueError("%s coordinate overflowed." % name) from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("%s coordinate did not produce a finite positive value." % name)
    return result


def _normalized_positive_coordinates(values: Any, name: str) -> np.ndarray:
    logs = np.asarray(values, dtype=np.float64)
    if logs.ndim != 1 or logs.size == 0:
        raise ValueError("%s require a non-empty coordinate vector." % name)
    if np.any(~np.isfinite(logs)):
        raise ValueError("%s coordinates must be finite." % name)
    shifted = logs - logs.max()
    probs = np.exp(shifted)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("%s coordinates could not be normalized." % name)
    return probs / total


def _preserve_sequence_type(original: Any, values: list[Any]) -> Any:
    if isinstance(original, tuple):
        return tuple(values)
    if isinstance(original, list):
        return list(values)
    try:
        return type(original)(values)
    except (TypeError, ValueError):
        return list(values)


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
    from mixle.stats import (
        CategoricalDistribution,
        CompositeDistribution,
        MixtureDistribution,
    )

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

    def structure_fingerprint(node: Any) -> Any:
        ensure_no_prior(node)
        if isinstance(node, MixtureDistribution):
            weights = np.asarray(node.w, dtype=np.float64)
            return (
                "mixture",
                type(node),
                node.name,
                tuple(bool(value == 0.0) for value in weights),
                tuple(structure_fingerprint(child) for child in node.components),
            )
        if isinstance(node, CompositeDistribution):
            return (
                "composite",
                type(node),
                type(node.dists),
                tuple(structure_fingerprint(child) for child in node.dists),
            )
        leaf_pair(node)
        common = (
            "leaf",
            type(node),
            getattr(node, "name", None),
            getattr(node, "keys", None),
        )
        if isinstance(node, CategoricalDistribution):
            coordinate_keys = _categorical_coordinate_keys(node)
            return common + (
                tuple(coordinate_keys),
                tuple(bool(node.pmap[key] == 0.0) for key in coordinate_keys),
                float(node.default_value),
            )
        return common

    expected_structure = structure_fingerprint(model)

    def width(node: Any) -> int:
        ensure_no_prior(node)
        if isinstance(node, MixtureDistribution):
            active_count = int(np.count_nonzero(np.asarray(node.w, dtype=np.float64) > 0.0))
            if active_count == 0:
                raise ValueError("squarem_packer requires at least one active mixture component.")
            return sum(width(child) for child in node.components) + active_count
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
            weights = np.asarray(node.w, dtype=np.float64)
            theta.extend(float(np.log(value)) for value in weights if value > 0.0)
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
            active = np.asarray(template.w, dtype=np.float64) > 0.0
            active_count = int(np.count_nonzero(active))
            logw = theta[pos : pos + active_count]
            pos += active_count
            weights = np.zeros(template.num_components, dtype=np.float64)
            weights[active] = _normalized_positive_coordinates(logw, "mixture weights")
            rebuilt = copy.copy(template)
            rebuilt.components = _preserve_sequence_type(template.components, components)
            rebuilt.num_components = len(components)
            rebuilt.w = weights
            rebuilt.zw = weights == 0.0
            rebuilt.log_w = np.full(weights.shape, -np.inf, dtype=np.float64)
            rebuilt.log_w[~rebuilt.zw] = np.log(weights[~rebuilt.zw])
            return rebuilt, pos
        if isinstance(template, CompositeDistribution):
            dists = []
            for child in template.dists:
                rebuilt, pos = unpack_node(child, theta, pos)
                dists.append(rebuilt)
            rebuilt = copy.copy(template)
            rebuilt.dists = _preserve_sequence_type(template.dists, dists)
            rebuilt.count = len(dists)
            return rebuilt, pos
        extract, rebuild, leaf_width = leaf_pair(template)
        del extract
        values = [float(value) for value in theta[pos : pos + leaf_width]]
        return rebuild(template, values), pos + leaf_width

    def pack(m: Any) -> np.ndarray:
        if structure_fingerprint(m) != expected_structure:
            raise ValueError("squarem_packer model structure or fixed support changed.")
        theta: list[float] = []
        pack_node(m, model, theta)
        values = np.asarray(theta, dtype=np.float64)
        if values.shape != (expected_width,):
            raise ValueError(
                "squarem_packer packed %d coordinates; expected %d."
                % (values.size, expected_width)
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("squarem_packer produced non-finite coordinates.")
        return values

    def unpack(theta: np.ndarray) -> Any:
        try:
            values = np.asarray(theta, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("squarem_packer coordinates must be numeric.") from exc
        if values.ndim != 1 or len(values) != expected_width:
            raise ValueError("squarem_packer expected a vector of length %d." % expected_width)
        if np.any(~np.isfinite(values)):
            raise ValueError("squarem_packer coordinates must be finite.")
        rebuilt, pos = unpack_node(model, values, 0)
        if pos != expected_width:  # pragma: no cover - recursive width and rebuild are paired
            raise RuntimeError("squarem_packer internal width mismatch.")
        return rebuilt

    return pack, unpack
