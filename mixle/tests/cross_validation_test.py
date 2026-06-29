"""Cross-validation fold generators (mixle.inference.cross_validation)."""

import unittest

import numpy as np

from mixle.inference import (
    blocked_kfold,
    group_kfold,
    kfold,
    leave_one_group_out,
    leave_one_out,
    nested_kfold,
    purged_kfold,
    spatial_block_kfold,
    stratified_kfold,
    time_series_split,
)


def _assert_partition(folds, n):
    seen = np.concatenate([te for _, te in folds])
    np.testing.assert_array_equal(np.sort(seen), np.arange(n))
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0


class KFoldTest(unittest.TestCase):
    def test_partition_and_sizes(self):
        folds = kfold(23, 5)
        _assert_partition(folds, 23)
        # 23 = 5+5+5+4+4 (the first n % k folds get one extra)
        self.assertEqual(sorted(len(te) for _, te in folds), [4, 4, 5, 5, 5])

    def test_shuffle_changes_membership_but_keeps_partition(self):
        a = kfold(20, 4, shuffle=False)
        b = kfold(20, 4, shuffle=True, seed=1)
        _assert_partition(b, 20)
        self.assertFalse(np.array_equal(a[0][1], b[0][1]))

    def test_blocked_is_contiguous(self):
        folds = blocked_kfold(20, 4)
        # first test fold is the first contiguous block
        np.testing.assert_array_equal(folds[0][1], np.arange(5))

    def test_leave_one_out(self):
        folds = leave_one_out(6)
        self.assertEqual(len(folds), 6)
        for _, te in folds:
            self.assertEqual(len(te), 1)

    def test_invalid_n_splits(self):
        with self.assertRaises(ValueError):
            kfold(5, 1)


class StratifiedTest(unittest.TestCase):
    def test_preserves_class_proportions(self):
        y = np.array([0] * 60 + [1] * 20)
        folds = stratified_kfold(y, 5, seed=0)
        _assert_partition(folds, 80)
        for _, te in folds:
            self.assertAlmostEqual(y[te].mean(), 0.25, delta=0.05)


class GroupTest(unittest.TestCase):
    def test_group_kfold_no_leakage(self):
        groups = np.repeat(np.arange(10), 6)
        folds = group_kfold(groups, 5)
        _assert_partition(folds, 60)
        for tr, te in folds:
            self.assertEqual(len(np.intersect1d(groups[tr], groups[te])), 0)

    def test_leave_one_group_out(self):
        groups = np.array([0, 0, 1, 1, 1, 2])
        folds = leave_one_group_out(groups)
        self.assertEqual(len(folds), 3)
        # the single-group test fold equals all rows of that group
        np.testing.assert_array_equal(folds[0][1], np.array([0, 1]))

    def test_group_kfold_too_many_splits(self):
        with self.assertRaises(ValueError):
            group_kfold(np.array([0, 0, 1, 1]), 5)


class TemporalTest(unittest.TestCase):
    def test_time_series_train_precedes_test(self):
        folds = time_series_split(100, 5)
        self.assertEqual(len(folds), 5)
        for tr, te in folds:
            self.assertLess(tr.max(), te.min())

    def test_time_series_gap_buffer(self):
        folds = time_series_split(100, 4, gap=5)
        for tr, te in folds:
            self.assertGreaterEqual(te.min() - tr.max(), 5)

    def test_time_series_sliding_window(self):
        folds = time_series_split(100, 4, max_train_size=10)
        for tr, _ in folds:
            self.assertLessEqual(len(tr), 10)

    def test_purged_embargo_removes_neighbours(self):
        folds = purged_kfold(50, 5, embargo=3)
        _assert_partition(folds, 50)
        for tr, te in folds:
            # no training index within `embargo` of any test index
            if len(tr) and len(te):
                dmin = np.abs(tr[:, None] - te[None, :]).min()
                self.assertGreater(dmin, 3)


class SpatialTest(unittest.TestCase):
    def test_partition_and_block_integrity(self):
        rng = np.random.RandomState(0)
        coords = rng.rand(300, 2)
        folds = spatial_block_kfold(coords, 5, n_side=6, seed=0)
        _assert_partition(folds, 300)

    def test_nearby_points_share_fold(self):
        # two tight clusters should each land entirely in one fold
        rng = np.random.RandomState(1)
        # two tight clusters separated by a non-integer multiple of the cell, so each falls wholly
        # inside one grid block (and thus one fold)
        c1 = rng.normal([0.5, 0.5], 0.01, (50, 2))
        c2 = rng.normal([7.3, 7.3], 0.01, (50, 2))
        coords = np.vstack([c1, c2])
        folds = spatial_block_kfold(coords, 5, block_size=1.0, seed=0)
        test_fold = {}
        for f, (_, te) in enumerate(folds):
            for i in te:
                test_fold[i] = f
        self.assertEqual(len({test_fold[i] for i in range(50)}), 1)
        self.assertEqual(len({test_fold[i] for i in range(50, 100)}), 1)


class NestedTest(unittest.TestCase):
    def test_inner_folds_index_into_outer_train(self):
        nf = nested_kfold(60, outer_splits=5, inner_splits=4)
        self.assertEqual(len(nf), 5)
        for fold in nf:
            self.assertEqual(len(fold.inner), 4)
            outer_train = set(fold.train.tolist())
            for itr, ite in fold.inner:
                self.assertTrue(set(itr.tolist()) <= outer_train)
                self.assertTrue(set(ite.tolist()) <= outer_train)
                self.assertEqual(len(np.intersect1d(ite, fold.test)), 0)


if __name__ == "__main__":
    unittest.main()
