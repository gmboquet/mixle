"""Regression: nonempty emissions with an empty given-set fail before inference.

When the given-set ``x[0]`` is empty but the emitted-set ``x[1]`` is non-empty, the
per-emitted association normalizer ``ll`` stays ``-inf`` and the count ``cc`` stays 0,
so the old ``ll -= math.log(cc)`` computed ``-inf - (-inf) = NaN``. The vectorized
backend already returned ``-inf`` for this case, so host and backend disagreed, and the
EM update fed NaN posteriors into the conditional accumulator.
"""

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
    def test_log_density_rejects_empty_given_for_nonempty_emissions(self):
        dist = _make_dist()
        x = ([], [("x", 1.0), ("y", 2.0)])
        with self.assertRaises(ValueError):
            dist.log_density(x)

    def test_backend_rejects_same_empty_given_schema(self):
        dist = _make_dist()
        x = ([], [("x", 1.0)])
        from mixle.engines import NUMPY_ENGINE

        with self.assertRaises(ValueError):
            dist.backend_seq_log_density([x], NUMPY_ENGINE)

    def test_em_update_rejects_empty_given_transactionally(self):
        dist = _make_dist()
        est = HiddenAssociationEstimator(
            cond_estimator=ConditionalDistributionEstimator({"a": CategoricalEstimator(), "b": CategoricalEstimator()}),
            len_estimator=CategoricalEstimator(),
        )
        acc = est.accumulator_factory().make()
        acc.update(([("a", 2.0), ("b", 1.0)], [("x", 1.0), ("y", 2.0)]), 1.0, dist)
        before = _flatten(acc.value()).copy()
        with self.assertRaises(ValueError):
            acc.update(([], [("x", 1.0)]), 1.0, dist)
        np.testing.assert_array_equal(_flatten(acc.value()), before)


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
