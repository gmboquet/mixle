"""Production scheduling & block sequencing (H3): ultimate pit + time-phased extraction."""

from __future__ import annotations

import itertools

import numpy as np

from mixle.mine_planning import schedule_extraction, ultimate_pit


def _brute_force_ultimate_pit(value: np.ndarray, precedence: list[tuple[int, int]]) -> np.ndarray:
    """Exhaustive Lerchs-Grossmann reference: try every subset, keep the best precedence-closed one.

    Independent of :mod:`mixle.mine_planning`'s max-flow construction -- a subset is *closed* when
    every block in it also has its predecessors in it, and the reference is simply the
    highest-value closed subset over all ``2**n`` subsets. Fine for the small toy block model this
    test uses; not how a real pit optimizer would scale.
    """
    n = value.size
    best_val, best_mask = -np.inf, None
    for bits in itertools.product((0, 1), repeat=n):
        mask = np.array(bits, dtype=bool)
        if any(mask[b] and not mask[pred] for b, pred in precedence):
            continue
        val = float(value[mask].sum())
        if val > best_val:
            best_val, best_mask = val, mask
    return best_mask


def _toy_2d_block_model() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """A small two-pocket inverted-pyramid block model (a 2-D pit cross-section).

    Two independent 3-level pyramids side by side: surface blocks 0-2 / 3-5, middle blocks 6-7
    (under the left pyramid) / 8-9 (under the right), and one bottom block per pyramid -- 10 (low
    grade) and 11 (high grade). Standard 45-degree-slope precedence: a middle block requires the two
    surface blocks straddling it, a bottom block requires both middle blocks above it. The left
    pyramid's ore (block 10, value +5) is not worth the waste stripped to reach it (surface+middle
    strip cost 12); the right pyramid's ore (block 11, value +50) is. So the optimal ultimate pit is
    exactly the right-hand pyramid.
    """
    value = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -3.0, -3.0, -3.0, -3.0, 5.0, 50.0])
    precedence = [
        (6, 0), (6, 1),
        (7, 1), (7, 2),
        (8, 3), (8, 4),
        (9, 4), (9, 5),
        (10, 6), (10, 7),
        (11, 8), (11, 9),
    ]  # fmt: skip
    return value, precedence


def test_ultimate_pit_matches_lerchs_grossmann_reference():
    value, precedence = _toy_2d_block_model()
    reference_mask = _brute_force_ultimate_pit(value, precedence)

    mask = ultimate_pit(value, precedence)

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, reference_mask)
    # sanity: the right-hand (profitable) pyramid only, in full.
    np.testing.assert_array_equal(mask, [False, False, False, True, True, True, False, False, True, True, False, True])


def test_schedule_extraction_respects_precedence_and_capacity():
    # The right-hand pyramid alone, relabeled 0..5 (surface 0,1,2 / middle 3,4 / bottom 5), scheduled
    # over 3 periods with capacity for exactly 2 blocks each -- capacity exactly matches the 6 blocks
    # that are worth mining, so every period must be filled and every block eventually mined.
    block_value = np.array([-2.0, -2.0, -2.0, -3.0, -3.0, 50.0])
    precedence = [(3, 0), (3, 1), (4, 1), (4, 2), (5, 3), (5, 4)]
    mill_capacity = np.array([2.0, 2.0, 2.0])
    n_periods = 3

    npv, period = schedule_extraction(block_value, precedence, mill_capacity, n_periods, discount=0.1)

    assert period.shape == (6,)
    assert set(period.tolist()) <= {-1, 0, 1, 2}
    # every precedence arc honored: a mined block's predecessor is mined at or before it.
    for b, pred in precedence:
        if period[b] != -1:
            assert period[pred] != -1, f"block {b} mined without predecessor {pred}"
            assert period[pred] <= period[b], f"block {b} mined before its predecessor {pred}"
    # capacity never exceeded, in any period.
    for t in range(n_periods):
        mined_this_period = int((period == t).sum())
        assert mined_this_period <= mill_capacity[t], f"period {t} over capacity: {mined_this_period}"
    # mining the whole pyramid nets +38 undiscounted and capacity exactly admits all 6 blocks, so the
    # value-maximizing schedule mines everything (nothing left permanently unmined).
    assert np.all(period != -1)
    assert npv > 0.0


def test_schedule_extraction_can_leave_unprofitable_blocks_unmined():
    # Same shape as the pit test's left-hand pyramid: the ore block (value +5) does not cover the
    # waste (2 surface + 2 middle at cost 2 each -- total 8), so an optimal schedule mines nothing.
    block_value = np.array([-2.0, -2.0, -2.0, -3.0, -3.0, 5.0])
    precedence = [(3, 0), (3, 1), (4, 1), (4, 2), (5, 3), (5, 4)]
    mill_capacity = np.array([2.0, 2.0, 2.0])

    npv, period = schedule_extraction(block_value, precedence, mill_capacity, 3, discount=0.0)

    np.testing.assert_array_equal(period, [-1, -1, -1, -1, -1, -1])
    assert npv == 0.0
