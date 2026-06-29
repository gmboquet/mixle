"""Regression test for MultivariateGaussian keyed parameter sharing.

Before the fix, ``key_merge`` had no else-branch seeding ``stats_dict[self.keys] = self``,
so the dict was never populated, ``combine`` never fired, and pooling silently no-opped.
``key_replace`` also passed the stored accumulator object (not its value tuple) to
``from_value`` and would crash. This test builds three accumulators that share a key over
data {1, 2, 3} and asserts all three carry the pooled statistics afterward.
"""

import numpy as np

from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianAccumulator


def _make_acc(x):
    acc = MultivariateGaussianAccumulator(dim=1, keys="shared")
    acc.update(np.asarray([float(x)]), 1.0, None)
    return acc


def test_multivariate_gaussian_key_merge_replace_pools():
    accs = [_make_acc(1), _make_acc(2), _make_acc(3)]

    stats_dict = {}
    for acc in accs:
        acc.key_merge(stats_dict)

    # The shared key must have been seeded and the pooled stats stored.
    assert "shared" in stats_dict, "key_merge never populated the shared key"

    pooled = stats_dict["shared"].value()
    # Pooled sum over {1,2,3} == 6, count == 3.
    assert float(pooled[0][0]) == 6.0
    assert float(pooled[2]) == 3.0

    # key_replace must broadcast the pooled stats back into every accumulator.
    for acc in accs:
        acc.key_replace(stats_dict)

    for acc in accs:
        s, _s2, c = acc.value()
        assert float(s[0]) == 6.0
        assert float(c) == 3.0
