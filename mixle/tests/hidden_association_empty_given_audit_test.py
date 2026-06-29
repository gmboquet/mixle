"""Regression: empty given-set must yield -inf (not NaN) in host log_density / EM update.

When the given-set ``x[0]`` is empty but the emitted-set ``x[1]`` is non-empty, the
per-emitted association normalizer ``ll`` stays ``-inf`` and the count ``cc`` stays 0,
so the old ``ll -= math.log(cc)`` computed ``-inf - (-inf) = NaN``. The vectorized
backend already returned ``-inf`` for this case, so host and backend disagreed, and the
EM update fed NaN posteriors into the conditional accumulator.
"""

import math
import unittest

import numpy as np

from mixle.stats.combinator.conditional import (
    ConditionalDistribution,
    ConditionalDistributionEstimator,
)
from mixle.stats.latent.hidden_association import (
    HiddenAssociationDistribution,
    HiddenAssociationEstimator,
)
from mixle.stats.univariate.discrete.categorical import (
    CategoricalDistribution,
    CategoricalEstimator,
)


def _make_dist():
    return HiddenAssociationDistribution(
        cond_dist=ConditionalDistribution(
            {
                "a": CategoricalDistribution({"x": 0.80, "y": 0.20}),
                "b": CategoricalDistribution({"x": 0.25, "y": 0.75}),
            }
        ),
        len_dist=CategoricalDistribution({0.0: 0.10, 2.0: 0.30, 3.0: 0.60}),
    )


class EmptyGivenAuditTestCase(unittest.TestCase):
    def test_log_density_empty_given_is_neg_inf_not_nan(self):
        dist = _make_dist()
        # Empty given-set, non-empty emitted-set: no association mass -> -inf.
        x = ([], [("x", 1.0), ("y", 2.0)])
        rv = dist.log_density(x)
        self.assertFalse(math.isnan(rv), "empty given-set produced NaN log-density")
        self.assertEqual(rv, -np.inf)

    def test_log_density_matches_backend_on_empty_given(self):
        dist = _make_dist()
        x = ([], [("x", 1.0)])
        host = dist.log_density(x)
        from mixle.engines import NUMPY_ENGINE

        backend = float(np.asarray(dist.backend_seq_log_density([x], NUMPY_ENGINE)).ravel()[0])
        self.assertEqual(host, backend)
        self.assertFalse(math.isnan(host))

    def test_em_update_empty_given_no_nan_stats(self):
        dist = _make_dist()
        est = HiddenAssociationEstimator(
            cond_estimator=ConditionalDistributionEstimator({"a": CategoricalEstimator(), "b": CategoricalEstimator()}),
            len_estimator=CategoricalEstimator(),
        )
        acc = est.accumulator_factory().make()
        # Mix a normal observation with an empty-given one; the empty one must not
        # corrupt the pooled conditional sufficient statistics with NaN.
        acc.update(([("a", 2.0), ("b", 1.0)], [("x", 1.0), ("y", 2.0)]), 1.0, dist)
        acc.update(([], [("x", 1.0)]), 1.0, dist)

        val = acc.cond_accumulator.value()
        flat = _flatten(val)
        self.assertFalse(np.isnan(flat).any(), "empty given-set corrupted EM sufficient statistics with NaN")


def _flatten(value):
    out = []
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, (tuple, list)):
            stack.extend(v)
        elif isinstance(v, dict):
            stack.extend(val for _, val in sorted(v.items(), key=lambda kv: str(kv[0])))
        elif v is None:
            continue
        else:
            try:
                out.append(np.asarray(v, dtype=np.float64).ravel())
            except (TypeError, ValueError):
                continue
    return np.concatenate(out) if out else np.zeros(0)


if __name__ == "__main__":
    unittest.main()
