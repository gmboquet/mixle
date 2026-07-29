"""Shared sampling helpers for vectorizing per-draw sampler loops.

The recurring pattern across mixture-like models is: draw a length-``size`` vector of component
indices, then sample each chosen component. The naive ``[comp_samplers[i].sample() for i in
comp_state]`` loop is slow; :func:`scatter_component_draws` instead samples each component once with
its assigned count and scatters the results back into draw order. Because every mixle component
sampler owns an independent ``RandomState`` and satisfies ``sample(n) == n`` sequential
``sample()`` calls, the scattered result is *bit-identical* to the per-draw loop, just far faster.
"""

from collections.abc import Iterable, Sequence
from itertools import islice
from operator import index
from typing import Any

import numpy as np


def _validated_draw_batch(drawn: Any, count: int, component: int) -> Any:
    """Return an indexable draw batch with exactly the requested leading size."""
    if isinstance(drawn, np.ndarray):
        if drawn.ndim == 0 or drawn.shape[0] != count:
            raise ValueError(
                f"component {component} sampler returned array shape {drawn.shape}, "
                f"expected leading shape ({count}, ...)"
            )
        return drawn
    try:
        actual = len(drawn)
    except TypeError:
        if not isinstance(drawn, Iterable):
            raise TypeError(f"component {component} sampler must return a sized or iterable batch") from None
        materialized = list(islice(iter(drawn), count + 1))
        actual = len(materialized)
        drawn = materialized
    if actual != count:
        raise ValueError(f"component {component} sampler returned {actual} draws, expected {count}")
    return drawn


def scatter_component_draws(comp_state: Any, comp_samplers: Sequence[Any], size: int) -> list[Any]:
    """Sample each component once (by its assigned count) and scatter into ``comp_state`` order.

    Args:
        comp_state: Length-``size`` array of component indices (already drawn).
        comp_samplers: One sampler per component; each must own an independent RNG.
        size: Number of draws.

    Returns:
        A length-``size`` list of draws, in the order given by ``comp_state``. When every component
        returns ndarrays (leaf / multivariate components), the list is backed by one contiguous array
        so the trailing sample shape (e.g. D-vectors) is preserved.
    """
    if isinstance(size, (bool, np.bool_)):
        raise ValueError("size must be a positive integer")
    try:
        size = index(size)
    except TypeError as exc:
        raise ValueError("size must be a positive integer") from exc
    if size <= 0:
        raise ValueError("size must be a positive integer")

    comp_state = np.asarray(comp_state)
    if comp_state.ndim != 1 or len(comp_state) != size:
        raise ValueError(f"comp_state must be one-dimensional with exactly size={size} entries")
    if not comp_samplers:
        raise ValueError("comp_samplers must contain at least one component sampler")
    if np.issubdtype(comp_state.dtype, np.bool_):
        raise ValueError("component assignments must be integer indices, not booleans")
    if np.issubdtype(comp_state.dtype, np.integer):
        if np.any(comp_state < 0) or np.any(comp_state >= len(comp_samplers)):
            raise ValueError(f"component assignments must be between 0 and {len(comp_samplers) - 1}")
        comp_state = comp_state.astype(np.intp, copy=False)
    elif comp_state.dtype == object:
        converted = np.empty(size, dtype=np.intp)
        for position, component in enumerate(comp_state):
            if isinstance(component, (bool, np.bool_)):
                raise ValueError("component assignments must be integer indices, not booleans")
            try:
                component_index = index(component)
            except TypeError as exc:
                raise ValueError("component assignments must be exact integer indices") from exc
            if component_index < 0 or component_index >= len(comp_samplers):
                raise ValueError(f"component assignments must be between 0 and {len(comp_samplers) - 1}")
            converted[position] = component_index
        comp_state = converted
    else:
        raise ValueError("component assignments must be exact integer indices")
    for component, sampler in enumerate(comp_samplers):
        if not callable(getattr(sampler, "sample", None)):
            raise TypeError(f"component sampler {component} has no callable sample method")

    positions_by_comp = {
        component: np.flatnonzero(comp_state == component)
        for component in range(len(comp_samplers))
        if np.any(comp_state == component)
    }
    draws_by_comp: dict[int, Any] = {}
    for component, positions in positions_by_comp.items():
        count = len(positions)
        drawn = comp_samplers[component].sample(size=count)
        draws_by_comp[component] = _validated_draw_batch(drawn, count, component)

    array_shapes = {drawn.shape[1:] for drawn in draws_by_comp.values() if isinstance(drawn, np.ndarray)}
    if len(array_shapes) == 1 and all(isinstance(drawn, np.ndarray) for drawn in draws_by_comp.values()):
        trailing_shape = next(iter(array_shapes))
        dtype = np.result_type(*(drawn.dtype for drawn in draws_by_comp.values()))
        out_arr = np.empty((size,) + trailing_shape, dtype=dtype)
        filled = np.zeros(size, dtype=bool)
        for component, drawn in draws_by_comp.items():
            positions = positions_by_comp[component]
            out_arr[positions] = drawn
            filled[positions] = True
        if not np.all(filled):
            raise RuntimeError("component draw scatter did not populate every requested output")
        return list(out_arr)

    out: list[Any] = [None] * size
    filled = np.zeros(size, dtype=bool)
    for component, drawn in draws_by_comp.items():
        for m, pos in enumerate(positions_by_comp[component]):
            out[pos] = drawn[m]
            filled[pos] = True
    if not np.all(filled):
        raise RuntimeError("component draw scatter did not populate every requested output")
    return out
