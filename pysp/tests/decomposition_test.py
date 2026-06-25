"""Model-parallel decomposition contract (C1.P0): atomic default + descriptor/lookup."""

import unittest

import pysp.stats as stats
from pysp.stats.compute.decomposition import (
    DecompAxis,
    Decomposition,
    ReductionOp,
    decomposition_for,
    register_decomposition,
)


class DecompositionContractTest(unittest.TestCase):
    def test_atomic_default(self):
        d = Decomposition.atomic()
        self.assertEqual(d.axis, DecompAxis.NONE)
        self.assertEqual(d.num_units, 1)
        self.assertEqual(d.reduction, ReductionOp.REPLICATE)
        self.assertTrue(d.exact)
        self.assertFalse(d.is_shardable)

    def test_is_shardable(self):
        self.assertTrue(Decomposition(axis=DecompAxis.COMPONENT, num_units=4, reduction=ReductionOp.SUM).is_shardable)
        self.assertFalse(  # one unit is not worth splitting
            Decomposition(axis=DecompAxis.COMPONENT, num_units=1, reduction=ReductionOp.SUM).is_shardable
        )

    def test_every_distribution_reports_atomic_by_default(self):
        # P0: the contract is opt-in; nothing has overridden decomposition() yet, so all report atomic.
        for d in (
            stats.GaussianDistribution(0.0, 1.0),
            stats.CategoricalDistribution({"a": 0.5, "b": 0.5}),
            stats.PoissonDistribution(2.0),
            stats.CompositeDistribution((stats.GaussianDistribution(0.0, 1.0), stats.PoissonDistribution(1.0))),
            stats.MixtureDistribution(
                [stats.GaussianDistribution(0.0, 1.0), stats.GaussianDistribution(3.0, 1.0)], [0.5, 0.5]
            ),
        ):
            self.assertEqual(decomposition_for(d).axis, DecompAxis.NONE, msg=type(d).__name__)
            self.assertFalse(decomposition_for(d).is_shardable, msg=type(d).__name__)

    def test_base_hook_returns_atomic(self):
        self.assertEqual(stats.GaussianDistribution(0.0, 1.0).decomposition(), Decomposition.atomic())

    def test_decomposition_for_accepts_class_and_instance(self):
        self.assertEqual(decomposition_for(stats.GaussianDistribution).axis, DecompAxis.NONE)  # class
        self.assertEqual(decomposition_for(stats.GaussianDistribution(0.0, 1.0)).axis, DecompAxis.NONE)  # instance

    def test_registry_override_via_mro(self):
        class _Marker:
            pass

        class _Sub(_Marker):
            pass

        register_decomposition(_Marker, Decomposition(axis=DecompAxis.FACTOR, num_units=3, reduction=ReductionOp.SUM))
        try:
            self.assertEqual(decomposition_for(_Sub()).axis, DecompAxis.FACTOR)  # found via MRO
            self.assertEqual(decomposition_for(_Sub()).num_units, 3)
        finally:
            from pysp.stats.compute.decomposition import _DECOMPOSITIONS

            _DECOMPOSITIONS.pop(_Marker, None)  # don't pollute the global registry


if __name__ == "__main__":
    unittest.main()
