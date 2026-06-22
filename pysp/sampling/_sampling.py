"""Shared sampling helpers for vectorizing per-draw sampler loops.

The recurring pattern across mixture-like models is: draw a length-``size`` vector of component
indices, then sample each chosen component. The naive ``[comp_samplers[i].sample() for i in
comp_state]`` loop is slow; :func:`scatter_component_draws` instead samples each component once with
its assigned count and scatters the results back into draw order. Because every pysp component
sampler owns an independent ``RandomState`` and satisfies ``sample(n) == n`` sequential
``sample()`` calls, the scattered result is *bit-identical* to the per-draw loop, just far faster.
"""

from typing import Any

import numpy as np


def scatter_component_draws(comp_state: Any, comp_samplers: list, size: int) -> list[Any]:
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
    comp_state = np.asarray(comp_state)
    draws_by_comp: dict[int, Any] = {}
    all_array = True
    for c in range(len(comp_samplers)):
        count = int(np.count_nonzero(comp_state == c))
        if count:
            drawn = comp_samplers[c].sample(size=count)
            draws_by_comp[c] = drawn
            all_array = all_array and isinstance(drawn, np.ndarray)
    if all_array and draws_by_comp:
        sample = next(iter(draws_by_comp.values()))
        out_arr = np.empty((size,) + sample.shape[1:], dtype=sample.dtype)
        for c, drawn in draws_by_comp.items():
            out_arr[comp_state == c] = drawn
        return list(out_arr)
    out: list[Any] = [None] * size
    for c, drawn in draws_by_comp.items():
        for m, pos in enumerate(np.nonzero(comp_state == c)[0]):
            out[pos] = drawn[m]
    return out
