"""Structural model-decomposition planner (C2): tree sizing + axis choice + unit partition over devices."""

import unittest

import numpy as np

import mixle.stats as stats
from mixle.stats.compute.decomposition import DecompAxis, ReductionOp
from mixle.utils.parallel.model_decomposition import decompose_model, shard_children, size_model_tree
from mixle.utils.parallel.planner import DeviceSpec, Resources


def _devices(n, mem=8 * 1024**3, throughput=1.0):
    return Resources(
        devices=tuple(DeviceSpec(name=f"d{i}", kind="cpu", memory_bytes=mem, throughput=throughput) for i in range(n))
    )


def _mixture(k):
    return stats.MixtureDistribution([stats.GaussianDistribution(float(i), 1.0) for i in range(k)], [1.0 / k] * k)


class SizingTest(unittest.TestCase):
    def test_hmm_own_params_scale_with_state_count_squared(self):
        # the dense S*S transition block is the structural cost the reflective walk silently zeroed
        def hmm(s):
            return stats.HiddenMarkovModelDistribution(
                [stats.CategoricalDistribution({"a": 0.5, "b": 0.5}) for _ in range(s)],
                [1.0 / s] * s,
                (np.ones((s, s)) / s).tolist(),
            )

        small = size_model_tree(hmm(2)).own_param_bytes
        big = size_model_tree(hmm(8)).own_param_bytes
        self.assertGreater(big, 8 * small)  # transitions grow ~16x (8^2 / 2^2), not silently 0

    def test_tree_recurses_into_children(self):
        comp = stats.CompositeDistribution((_mixture(3), stats.GaussianDistribution(0.0, 1.0)))
        sized = size_model_tree(comp)
        self.assertEqual(sized.axis, DecompAxis.FACTOR)
        self.assertEqual(len(sized.children), 2)
        self.assertEqual(sized.children[0].axis, DecompAxis.COMPONENT)  # the inner mixture
        self.assertEqual(sized.children[0].num_units, 3)

    def test_shard_children(self):
        self.assertEqual(len(shard_children(_mixture(4))), 4)
        self.assertEqual(len(shard_children(stats.GaussianDistribution(0.0, 1.0))), 0)


class DecomposeTest(unittest.TestCase):
    def _assert_partitions(self, dec, num_units):
        ranges = [(c.start, c.stop) for c in dec.cuts]
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], num_units)
        for (_, s0), (s1, _) in zip(ranges, ranges[1:]):
            self.assertEqual(s0, s1)  # contiguous, no gap/overlap
        self.assertEqual(sum(b - a for a, b in ranges), num_units)

    def test_mixture_component_cuts(self):
        dec = decompose_model(_mixture(10), _devices(3), n_data=20)
        self.assertTrue(dec.is_model_parallel)
        self.assertEqual(dec.axis, DecompAxis.COMPONENT)
        self.assertEqual(dec.reduction, ReductionOp.LOGSUMEXP_RESPONSIBILITY)
        self.assertEqual(len(dec.cuts), 3)
        self._assert_partitions(dec, 10)

    def test_wide_composite_factor_cuts(self):
        comp = stats.CompositeDistribution(tuple(stats.GaussianDistribution(float(i), 1.0) for i in range(8)))
        dec = decompose_model(comp, _devices(4), n_data=20)
        self.assertEqual(dec.axis, DecompAxis.FACTOR)
        self.assertEqual(len(dec.cuts), 4)
        self._assert_partitions(dec, 8)

    def test_leaf_is_not_model_parallel(self):
        dec = decompose_model(stats.GaussianDistribution(0.0, 1.0), _devices(4))
        self.assertEqual(dec.axis, DecompAxis.NONE)
        self.assertFalse(dec.is_model_parallel)

    def test_refuses_model_parallel_when_data_parallel_wins(self):
        # small mixture (5 comps), fits replicated, large N -> data-parallel preferred, no model cut
        dec = decompose_model(_mixture(5), _devices(4), n_data=100_000)
        self.assertEqual(dec.axis, DecompAxis.NONE)
        self.assertIn("data-parallel", dec.rationale)

    def test_throughput_weighted_partition(self):
        # one fast device should get more components
        res = Resources(
            devices=(
                DeviceSpec(name="fast", kind="cpu", memory_bytes=8 * 1024**3, throughput=3.0),
                DeviceSpec(name="slow", kind="cpu", memory_bytes=8 * 1024**3, throughput=1.0),
            )
        )
        dec = decompose_model(_mixture(12), res, n_data=20)
        self._assert_partitions(dec, 12)
        sizes = {c.device.name: c.stop - c.start for c in dec.cuts}
        self.assertGreater(sizes["fast"], sizes["slow"])


class WholeTreeCostTest(unittest.TestCase):
    """The planner searches the whole tree by COMPUTE COST, not just the root by unit count."""

    def test_subtree_work_recurses_and_excludes_nothing(self):
        from mixle.utils.parallel.model_decomposition import subtree_work

        leaf = stats.GaussianDistribution(0.0, 1.0)
        comp = stats.CompositeDistribution((_mixture(3), leaf))
        self.assertGreater(subtree_work(comp), subtree_work(_mixture(3)))  # composite includes the leaf too
        self.assertGreater(subtree_work(_mixture(3)), subtree_work(leaf))  # a 3-mixture out-weighs one leaf

    def test_finds_axis_nested_below_the_root(self):
        # the root is a thin 2-factor composite; the real axis is the 12-component mixture one level down
        comp = stats.CompositeDistribution((_mixture(12), stats.GaussianDistribution(0.0, 1.0)))
        dec = decompose_model(comp, _devices(4), n_data=40)
        self.assertTrue(dec.is_model_parallel)
        self.assertEqual(dec.axis, DecompAxis.COMPONENT)  # the nested mixture, not the root FACTOR axis
        self.assertEqual(dec.num_units, 12)

    def test_thin_imbalanced_parent_loses_to_balanced_child(self):
        from mixle.utils.parallel.model_decomposition import best_parallel_axis

        # root composite has 2 wildly imbalanced units [heavy mixture, tiny leaf]; the balanced 12-way
        # mixture inside must win despite the parent's larger total_work (a fat unit caps parent speedup)
        comp = stats.CompositeDistribution((_mixture(12), stats.GaussianDistribution(0.0, 1.0)))
        best = best_parallel_axis(comp, max_workers=4)
        self.assertEqual(best.num_units, 12)
        self.assertEqual(best.axis, DecompAxis.COMPONENT)

    def test_cost_aware_axis_prefers_heavier_units(self):
        from mixle.utils.parallel.model_decomposition import best_parallel_axis

        # a 4-way mixture of D=20 MVGaussians (heavy) vs an 8-way mixture of 2-cat (cheap): cost wins
        heavy = stats.MixtureDistribution(
            [stats.MultivariateGaussianDistribution([0.0] * 20, np.eye(20).tolist()) for _ in range(4)], [0.25] * 4
        )
        cheap = stats.MixtureDistribution(
            [stats.CategoricalDistribution({"a": 0.5, "b": 0.5}) for _ in range(8)], [1 / 8] * 8
        )
        best = best_parallel_axis(stats.CompositeDistribution((heavy, cheap)), max_workers=4)
        self.assertEqual(best.num_units, 4)  # the heavy 4-way axis, not the wider-but-cheap 8-way one

    def test_cost_weighted_cuts_balance_work_not_count(self):
        # two fast + two slow devices: the cost partition should give the fast devices more components
        res = Resources(
            devices=(
                DeviceSpec(name="f0", kind="cpu", memory_bytes=8 * 1024**3, throughput=3.0),
                DeviceSpec(name="s0", kind="cpu", memory_bytes=8 * 1024**3, throughput=1.0),
            )
        )
        dec = decompose_model(_mixture(12), res, n_data=20)
        sizes = {c.device.name: c.stop - c.start for c in dec.cuts}
        self.assertGreater(sizes["f0"], sizes["s0"])


if __name__ == "__main__":
    unittest.main()
