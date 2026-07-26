"""Nested fusion must preserve parameter-dependent component encodings."""

import unittest
from unittest.mock import patch

import numpy as np

from mixle.stats import (
    CompositeDistribution,
    GaussianDistribution,
    MixtureDistribution,
)
from mixle.stats.compute.fused_nested import (
    _marshal,
    analyze_nested,
    fused_nested_seq_log_density,
)
from mixle.stats.compute.pdist import DataSequenceEncoder
from mixle.utils.optional_deps import HAS_NUMBA


class _OffsetEncoder(DataSequenceEncoder):
    """Test encoder whose payload depends on one distribution parameter."""

    def __init__(self, offset):
        self.offset = float(offset)

    def __eq__(self, other):
        return isinstance(other, _OffsetEncoder) and other.offset == self.offset

    def seq_encode(self, x):
        return np.asarray(x, dtype=np.float64) + self.offset


def _parameterized_gaussian_encoder(dist):
    return _OffsetEncoder(dist.mu)


def _model():
    inner = MixtureDistribution(
        [GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0)],
        [0.4, 0.6],
    )
    return CompositeDistribution((inner, GaussianDistribution(1.0, 2.0)))


class NestedEncoderCompatibilityTest(unittest.TestCase):
    def test_same_model_shape_with_unequal_encoders_gets_distinct_slots(self):
        model = _model()
        rows = [(-1.0, 2.0), (0.5, -3.0), (2.0, 0.0)]
        with patch.object(GaussianDistribution, "dist_to_encoder", _parameterized_gaussian_encoder):
            encoded = model.dist_to_encoder().seq_encode(rows)
            root, ctx = analyze_nested(model)
            data, _ = _marshal(model, root, ctx, encoded)

        self.assertEqual(len(ctx.slots), 3)
        np.testing.assert_array_equal(data[0], np.asarray([-1.0, 0.5, 2.0]))
        np.testing.assert_array_equal(data[1], np.asarray([3.0, 4.5, 6.0]))
        np.testing.assert_array_equal(data[2], np.asarray([3.0, -2.0, 1.0]))

    @unittest.skipUnless(HAS_NUMBA, "nested fused kernels require numba")
    def test_fused_score_matches_each_components_own_encoding(self):
        model = _model()
        rows = [(-1.0, 2.0), (0.5, -3.0), (2.0, 0.0)]
        with patch.object(GaussianDistribution, "dist_to_encoder", _parameterized_gaussian_encoder):
            encoded = model.dist_to_encoder().seq_encode(rows)
            expected = model.seq_log_density(encoded)
            actual = fused_nested_seq_log_density(model, encoded)

        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
