"""Regression test for the vectorized transition-count aggregation in
MarkovChainAccumulator.seq_update / seq_initialize.

The E-step previously accumulated transition weights with an explicit Python
loop over EVERY observed transition occurrence. That loop was replaced with a
single np.bincount over the flat (prev, next) index, scattering only the
distinct nonzero pairs into the sparse count map. These tests pin that the
vectorized path produces exactly the same pooled statistics as the row-by-row
update() path, and that the empty-transition boundary does not raise.
"""

import numpy as np

from pysp.stats.sequences.markov_chain import (
    MarkovChainDistribution,
    MarkovChainEstimator,
)


def _make_dist():
    init = {"a": 0.5, "b": 0.3, "c": 0.2}
    trans = {
        "a": {"a": 0.5, "b": 0.3, "c": 0.2},
        "b": {"a": 0.4, "b": 0.4, "c": 0.2},
        "c": {"a": 0.2, "b": 0.3, "c": 0.5},
    }
    return MarkovChainDistribution(init, trans)


def _row_by_row(dist, data, weights):
    """Reference accumulation via the scalar update() path."""
    acc = dist.estimator().accumulator_factory().make()
    for x, w in zip(data, weights):
        acc.update(x, w, dist)
    return acc


def test_seq_update_matches_row_by_row_pooled():
    dist = _make_dist()
    # Sequences chosen so many (prev, next) pairs repeat within the batch,
    # which is exactly what the np.bincount aggregation must collapse.
    data = [
        ["a", "a", "b", "c", "a", "a"],
        ["b", "b", "a", "a", "c", "c"],
        ["c", "a", "b", "b", "a", "c"],
    ]
    weights = np.array([1.0, 2.0, 0.5])

    enc = dist.dist_to_encoder().seq_encode(data)

    acc = dist.estimator().accumulator_factory().make()
    acc.seq_update(enc, weights, dist)

    ref = _row_by_row(dist, data, weights)

    # Initial-state counts pooled correctly.
    assert acc.init_count_map == ref.init_count_map

    # Transition counts pooled correctly (not just the first occurrence).
    assert set(acc.trans_count_map.keys()) == set(ref.trans_count_map.keys())
    for k1 in ref.trans_count_map:
        assert acc.trans_count_map[k1] == ref.trans_count_map[k1]


def test_seq_initialize_matches_row_by_row_pooled():
    dist = _make_dist()
    data = [
        ["a", "b", "a", "b", "c"],
        ["c", "c", "a", "a", "b"],
    ]
    weights = np.array([1.5, 0.75])

    enc = dist.dist_to_encoder().seq_encode(data)

    acc = dist.estimator().accumulator_factory().make()
    acc.seq_initialize(enc, weights, np.random.RandomState(0))

    ref = _row_by_row(dist, data, weights)

    assert acc.init_count_map == ref.init_count_map
    for k1 in ref.trans_count_map:
        assert acc.trans_count_map[k1] == ref.trans_count_map[k1]


def test_seq_update_empty_transitions_does_not_raise():
    # Hand-build an encoding with NO transitions (prev_x / next_x empty) to
    # exercise the len(prev_x) > 0 guard directly. (The encoder itself cannot
    # currently emit such a tuple, so we synthesize it from a real encode.)
    dist = _make_dist()
    sz, idx0, idx1, init_x, prev_x, next_x, inv_key_map, len_enc = (
        dist.dist_to_encoder().seq_encode([["a", "b"], ["c", "a"]])
    )

    empty = np.asarray([], dtype=prev_x.dtype)
    empty_enc = (
        sz,
        idx0,
        np.asarray([], dtype=idx1.dtype),
        init_x,
        empty,
        empty,
        inv_key_map,
        len_enc,
    )

    acc = dist.estimator().accumulator_factory().make()
    weights = np.array([1.0, 1.0])
    # Must not raise (np.bincount on empty / [:, 0] indexing would fail without
    # the guard) and must add no transition mass.
    acc.seq_update(empty_enc, weights, dist)

    assert acc.trans_count_map == {}


def test_estimator_constructs():
    # Sanity: the estimator wires up an accumulator factory.
    assert isinstance(_make_dist().estimator(), MarkovChainEstimator)
