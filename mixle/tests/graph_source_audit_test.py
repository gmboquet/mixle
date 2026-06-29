"""Regression test for the vectorized ``_edge_counts`` in graph_source.

The original implementation iterated ``_edge_indices`` in pure Python over every
free edge position (O(n^2) Python-level steps + scalar ``__getitem__``). The fix
replaces this with single numpy reductions. This test asserts the vectorized
result is bit-identical to the reference loop across all four
``(directed, self_loops)`` cases.
"""

import numpy as np
import pytest

import mixle.stats  # noqa: F401  -- fully initialize the package to avoid a circular import


def _reference_edge_counts(adj, directed, self_loops):
    from mixle.data.sources.graph_source import _edge_indices

    total = 0.0
    successes = 0.0
    for i, j in _edge_indices(adj.shape[0], directed=directed, self_loops=self_loops):
        successes += adj[i, j]
        total += 1.0
    return total, successes


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("self_loops", [False, True])
@pytest.mark.parametrize("n", [0, 1, 2, 5, 9])
def test_edge_counts_matches_reference_loop(directed, self_loops, n):
    from mixle.data.sources.graph_source import _edge_counts

    rng = np.random.default_rng(1234 + n + 10 * directed + 100 * self_loops)
    adj = (rng.random((n, n)) < 0.4).astype(np.float64)
    expected_total, expected_successes = _reference_edge_counts(adj, directed, self_loops)
    total, successes = _edge_counts(adj, directed, self_loops)
    assert total == expected_total
    assert successes == expected_successes
