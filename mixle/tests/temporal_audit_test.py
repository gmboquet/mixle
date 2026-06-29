"""Regression test for the PeriodicTime accumulator/encoder contract.

Before the fix, ``PeriodicTimeAccumulator`` / ``PeriodicTimeDataEncoder`` were plain classes that did
not subclass ``SequenceEncodableStatisticAccumulator`` / ``DataSequenceEncoder`` and were missing
``scale`` / ``key_merge`` / ``key_replace``. Streaming/online EM calls ``accumulator.scale(rho)``, which
raised ``AttributeError``, and ``PeriodicTimeDistribution(keys=...)`` could not pool statistics across a
shared key. This test exercises both paths: ``scale`` no longer raises and scales the linear stats, and
three accumulators that share a key over data {1, 2, 3} all carry the pooled statistics afterward.
"""

import math

import numpy as np

from mixle.stats.compute.pdist import DataSequenceEncoder, SequenceEncodableStatisticAccumulator
from mixle.stats.processes.temporal import (
    PeriodicTimeAccumulator,
    PeriodicTimeDataEncoder,
    PeriodicTimeDistribution,
    cyclic_phase,
)


def _make_acc(t, keys="shared"):
    est = PeriodicTimeDistribution("day", keys=keys).estimator()
    acc = est.accumulator_factory().make()
    acc.update(float(t), 1.0, None)
    return acc


def test_periodic_time_is_a_proper_accumulator_and_encoder():
    acc = _make_acc(1.0)
    enc = acc.acc_to_encoder()
    assert isinstance(acc, SequenceEncodableStatisticAccumulator)
    assert isinstance(enc, PeriodicTimeDataEncoder)
    assert isinstance(enc, DataSequenceEncoder)
    # Encoder equality is required for batching interchangeability.
    assert enc == PeriodicTimeDataEncoder(enc.period, enc.von_mises_encoder)


def test_scale_does_not_raise_and_scales_linear_stats():
    acc = _make_acc(1.0)
    count0, cos0, sin0 = acc.value()
    rv = acc.scale(2.0)  # streaming/online EM calls this with a decay rate
    assert rv is acc
    count1, cos1, sin1 = acc.value()
    assert math.isclose(count1, 2.0 * count0)
    assert math.isclose(cos1, 2.0 * cos0)
    assert math.isclose(sin1, 2.0 * sin0)


def test_key_merge_replace_pools_across_shared_key():
    accs = [_make_acc(1.0), _make_acc(2.0), _make_acc(3.0)]

    stats_dict = {}
    for acc in accs:
        acc.key_merge(stats_dict)

    assert "shared" in stats_dict, "key_merge never populated the shared key"

    pooled_count, pooled_cos, pooled_sin = stats_dict["shared"].value()
    phases = cyclic_phase([1.0, 2.0, 3.0], "day")
    assert math.isclose(pooled_count, 3.0)
    assert math.isclose(pooled_cos, float(np.sum(np.cos(phases))))
    assert math.isclose(pooled_sin, float(np.sum(np.sin(phases))))

    # key_replace must broadcast the pooled stats back into every accumulator (not the first batch).
    for acc in accs:
        acc.key_replace(stats_dict)

    for acc in accs:
        c, cos, sin = acc.value()
        assert math.isclose(c, 3.0)
        assert math.isclose(cos, float(np.sum(np.cos(phases))))
        assert math.isclose(sin, float(np.sum(np.sin(phases))))


def test_keys_none_is_a_no_op():
    acc = PeriodicTimeAccumulator(
        "day", PeriodicTimeDistribution("day").estimator().von_mises_estimator.accumulator_factory().make()
    )
    stats_dict = {}
    acc.key_merge(stats_dict)
    assert stats_dict == {}
    acc.key_replace(stats_dict)  # must not raise


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
