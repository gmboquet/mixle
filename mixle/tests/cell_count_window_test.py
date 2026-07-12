"""Integer cell aggregation (E10 support structures): SlidingCellWindow's exact group eviction and
CellCountTree's O(log n) range queries -- torch-free, so these run on the fast lanes.

The property under test is the group claim from ``experiments/group_attention/RESULTS.md``: per-cell
integer counts live in Z^cells, so eviction/range-subtraction is EXACT -- counts after any operation
sequence equal a from-scratch recount, bit for bit, with fully-cancelled cells genuinely absent. Value
sums are float64 running sums and only promise recompute-level closeness, which is asserted at 1e-9.
"""

import math

import numpy as np
import pytest

from mixle.experimental.quantized_key_attention import CellCountTree, SlidingCellWindow


class SlidingCellWindowTest:
    def test_counts_match_recount_exactly_and_value_sums_closely_over_a_long_stream(self):
        rng = np.random.RandomState(0)
        window = SlidingCellWindow(window=64, value_dim=3)
        stream = [(int(rng.randint(0, 12)), rng.normal(size=3)) for _ in range(5000)]
        for i, (cell_id, value) in enumerate(stream):
            window.push(cell_id=cell_id, value=value)
            if i % 617 == 0:  # spot-check against brute-force recomputation mid-stream
                live = stream[max(0, i + 1 - 64) : i + 1]
                expected_counts: dict[int, int] = {}
                expected_sums: dict[int, np.ndarray] = {}
                for cid, val in live:
                    expected_counts[cid] = expected_counts.get(cid, 0) + 1
                    expected_sums[cid] = expected_sums.get(cid, np.zeros(3)) + val
                counts, sums = window.totals()
                assert counts == expected_counts  # integers: EXACT equality, no tolerance
                assert set(sums) == set(expected_sums)
                for cid in sums:
                    np.testing.assert_allclose(sums[cid], expected_sums[cid], atol=1e-9)

    def test_eviction_returns_the_scrolled_out_token_and_cancelled_cells_vanish(self):
        window = SlidingCellWindow(window=2, value_dim=1)
        assert window.push(cell_id=5, value=[1.0]) is None
        assert window.push(cell_id=6, value=[2.0]) is None
        evicted = window.push(cell_id=6, value=[3.0])
        assert evicted is not None and evicted[0] == 5 and float(evicted[1][0]) == 1.0
        assert 5 not in window.counts, "a cell whose count reaches zero must be deleted, not left at 0"
        assert window.counts == {6: 2} and len(window) == 2

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            SlidingCellWindow(window=0, value_dim=1)
        with pytest.raises(ValueError):
            SlidingCellWindow(window=4, value_dim=2).push(cell_id=0, value=[1.0, 2.0, 3.0])


class CellCountTreeTest:
    def test_arbitrary_range_queries_match_brute_force_exactly(self):
        rng = np.random.RandomState(1)
        capacity = 2048
        tree = CellCountTree(capacity)
        events = [(int(rng.randint(0, capacity)), int(rng.randint(0, 30))) for _ in range(3000)]
        for pos, cell_id in events:
            tree.add(pos, cell_id)
        for _ in range(50):
            lo = int(rng.randint(0, capacity))
            hi = int(rng.randint(lo, capacity + 1))
            expected: dict[int, int] = {}
            for pos, cell_id in events:
                if lo <= pos < hi:
                    expected[cell_id] = expected.get(cell_id, 0) + 1
            assert tree.range_counts(lo, hi) == expected

    def test_node_touches_are_logarithmic_receipted_not_asserted_from_theory(self):
        capacity = 4096
        tree = CellCountTree(capacity)
        bound = int(math.log2(capacity)) + 1
        rng = np.random.RandomState(2)
        for _ in range(200):
            tree.add(int(rng.randint(0, capacity)), cell_id=1)
            assert tree.last_touches <= bound
        for _ in range(50):
            lo = int(rng.randint(0, capacity))
            hi = int(rng.randint(lo, capacity + 1))
            tree.range_counts(lo, hi)
            assert tree.last_touches <= 2 * bound  # two prefix walks

    def test_group_inverse_cancellation_removes_cells_entirely(self):
        tree = CellCountTree(16)
        tree.add(2, cell_id=7)
        tree.add(9, cell_id=7)
        assert tree.range_counts(0, 16) == {7: 2}
        assert tree.range_counts(3, 9) == {}, "fully-cancelled cells must be absent, not zero-valued"
        assert tree.range_counts(2, 10) == {7: 2}

    def test_rejects_bad_inputs(self):
        tree = CellCountTree(8)
        with pytest.raises(ValueError):
            tree.add(8, cell_id=0)
        with pytest.raises(ValueError):
            tree.range_counts(5, 3)
        with pytest.raises(ValueError):
            CellCountTree(0)
