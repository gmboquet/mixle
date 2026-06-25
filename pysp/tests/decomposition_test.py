"""Model-parallel decomposition contract (C1.P0 + C1.P1): descriptors, Shardable, additive-fold conformance."""

import unittest

import numpy as np

import pysp.stats as stats
from pysp.capability import Shardable, supports
from pysp.stats.compute.decomposition import (
    DecompAxis,
    Decomposition,
    ReductionOp,
    decomposition_for,
    register_decomposition,
)


def _close(a, b) -> bool:
    """Recursively compare suff-stat payloads: counts exact, float sums equal up to float reassociation
    (single-pass adds vs per-shard adds + combine differ only in the last ULPs -- that IS the guarantee)."""
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) or isinstance(a, float) or isinstance(b, float):
        return np.allclose(
            np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-9, atol=1e-12, equal_nan=True
        )
    return a == b


def _fold_value(dist, data):
    """Single-node accumulator value() over the full data."""
    est = dist.estimator()
    enc = dist.dist_to_encoder().seq_encode(data)
    acc = est.accumulator_factory().make()
    acc.seq_update(enc, np.ones(len(data)), dist)
    return acc.value()


def _sharded_fold_value(dist, data, k):
    """Two data shards, each accumulated independently, then combine()'d -- the additive monoid."""
    est = dist.estimator()
    enc_a = dist.dist_to_encoder().seq_encode(data[:k])
    enc_b = dist.dist_to_encoder().seq_encode(data[k:])
    a = est.accumulator_factory().make()
    a.seq_update(enc_a, np.ones(k), dist)
    b = est.accumulator_factory().make()
    b.seq_update(enc_b, np.ones(len(data) - k), dist)
    a.combine(b.value())
    return a.value()


class ContractTest(unittest.TestCase):
    def test_atomic_default_and_is_shardable(self):
        self.assertEqual(Decomposition.atomic().axis, DecompAxis.NONE)
        self.assertFalse(Decomposition.atomic().is_shardable)
        self.assertTrue(Decomposition(axis=DecompAxis.COMPONENT, num_units=4, reduction=ReductionOp.SUM).is_shardable)

    def test_leaves_report_atomic(self):
        for d in (stats.GaussianDistribution(0.0, 1.0), stats.CategoricalDistribution({"a": 0.5, "b": 0.5})):
            self.assertEqual(decomposition_for(d).axis, DecompAxis.NONE, msg=type(d).__name__)
            self.assertFalse(supports(d, Shardable), msg=type(d).__name__)

    def test_registry_override_via_mro(self):
        class _Marker:
            pass

        class _Sub(_Marker):
            pass

        register_decomposition(_Marker, Decomposition(axis=DecompAxis.FACTOR, num_units=3, reduction=ReductionOp.SUM))
        try:
            self.assertEqual(decomposition_for(_Sub()).num_units, 3)
        finally:
            from pysp.stats.compute.decomposition import _DECOMPOSITIONS

            _DECOMPOSITIONS.pop(_Marker, None)


class DescriptorTest(unittest.TestCase):
    def test_composite_and_record_factor(self):
        comp = stats.CompositeDistribution(
            (
                stats.GaussianDistribution(0.0, 1.0),
                stats.PoissonDistribution(1.0),
                stats.CategoricalDistribution({"a": 1.0}),
            )
        )
        dc = decomposition_for(comp)
        self.assertEqual((dc.axis, dc.num_units, dc.reduction), (DecompAxis.FACTOR, 3, ReductionOp.SUM))
        self.assertTrue(supports(comp, Shardable))
        rec = stats.RecordDistribution({"x": stats.GaussianDistribution(0.0, 1.0), "y": stats.PoissonDistribution(1.0)})
        self.assertEqual(decomposition_for(rec).axis, DecompAxis.FACTOR)
        self.assertEqual(decomposition_for(rec).num_units, 2)

    def test_sequence_names_data_axis(self):
        seq = stats.SequenceDistribution(stats.PoissonDistribution(2.0))
        self.assertEqual(decomposition_for(seq).axis, DecompAxis.SEQUENCE)
        self.assertEqual(decomposition_for(seq).reduction, ReductionOp.SUM)

    def test_mixture_component_axis(self):
        mix = stats.MixtureDistribution([stats.GaussianDistribution(float(i), 1.0) for i in range(5)], [0.2] * 5)
        dc = decomposition_for(mix)
        self.assertEqual(dc.axis, DecompAxis.COMPONENT)
        self.assertEqual(dc.num_units, 5)
        self.assertEqual(dc.reduction, ReductionOp.LOGSUMEXP_RESPONSIBILITY)
        self.assertEqual(dc.engine_axis, 0)  # homogeneous -> stacked/DTensor path
        self.assertTrue(supports(mix, Shardable))

    def test_heterogeneous_mixture_no_stacked_axis(self):
        het = stats.HeterogeneousMixtureDistribution(
            [stats.GaussianDistribution(0.0, 1.0), stats.PoissonDistribution(2.0)], [0.5, 0.5]
        )
        dc = decomposition_for(het)
        self.assertEqual(dc.axis, DecompAxis.COMPONENT)
        self.assertIsNone(dc.engine_axis)  # not homogeneous -> host-shard executor mode


class AdditiveFoldConformanceTest(unittest.TestCase):
    """The load-bearing guarantee: splitting the additive suff-stat payload and recombining with the
    same combine() monoid is bit-identical to the single-node fold (up to nothing -- integer/float adds)."""

    def test_composite_fold(self):
        rng = np.random.RandomState(0)
        comp = stats.CompositeDistribution((stats.GaussianDistribution(0.0, 1.0), stats.PoissonDistribution(2.0)))
        data = [(float(rng.randn()), int(rng.poisson(2))) for _ in range(200)]
        self.assertTrue(_close(_fold_value(comp, data), _sharded_fold_value(comp, data, 73)))

    def test_mixture_fold(self):
        rng = np.random.RandomState(1)
        mix = stats.MixtureDistribution(
            [stats.GaussianDistribution(-2.0, 1.0), stats.GaussianDistribution(2.0, 1.0)], [0.5, 0.5]
        )
        data = [float(rng.randn() + (4 * (rng.rand() < 0.5) - 2)) for _ in range(200)]
        self.assertTrue(_close(_fold_value(mix, data), _sharded_fold_value(mix, data, 91)))


if __name__ == "__main__":
    unittest.main()
