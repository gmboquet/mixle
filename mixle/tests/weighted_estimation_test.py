"""Weighted-estimation contract (worklist Q5.3).

A weight ``w`` on an observation must mean exactly what physically replicating that observation
``w`` times means. Weights enter estimation at the accumulator, as weighted sums of sufficient
statistics, so this is where the contract is pinned:

* **Replication fidelity** — the sufficient statistics from a weighted pass equal those from an
  integer-replicated pass. An accumulator that silently used ``1.0`` instead of ``weight`` (a
  dropped weight) fails here.
* **Zero-weight is a no-op** — an observation carried at weight ``0`` contributes nothing.
* **Scale invariance** — scaling every weight by a constant leaves the fitted distribution
  unchanged. An accumulator (or estimator) that normalized weights away, or a fit that treated the
  weight total as a sample size, fails here.

For the closed-form-MLE families the contract is additionally checked end to end at the *fitted
distribution* level (weighted fit == replicated fit, and scale-invariant). Iterative-MLE families
(e.g. Gamma, whose shape is solved by Newton iteration on the digamma function) are checked only at
the sufficient-statistic level: their fit is not bit-stable to a 1e-15 perturbation of an otherwise
identical sufficient statistic, which is a property of the solver, not of weight handling.
"""

import unittest

import numpy as np

import mixle.stats as st

# (name, distribution, data). Data is small and hand-picked to sit in each family's support.
_CLOSED_FORM = [
    ("Gaussian", st.GaussianDistribution(0.0, 1.0), [0.0, 1.0, 2.0, 3.0]),
    ("Poisson", st.PoissonDistribution(2.0), [0, 1, 2, 3]),
    ("Exponential", st.ExponentialDistribution(1.0), [0.5, 1.0, 2.0, 4.0]),
    ("Categorical", st.CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}), ["a", "b", "c", "a"]),
    ("Geometric", st.GeometricDistribution(0.4), [1, 2, 3, 4]),
    ("Binomial", st.BinomialDistribution(0.5, n=5), [0, 1, 2, 3]),
]

# Correct weight handling in the accumulator, but an iterative estimator that is not bit-stable to a
# ~1e-15 perturbation of the sufficient statistics -> sufficient-statistic checks only.
_SUFFSTAT_ONLY = [
    ("Gamma", st.GammaDistribution(2.0, 1.0), [0.5, 1.0, 2.0, 4.0]),
    (
        "DiagonalGaussian",
        st.DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0]),
        [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([3.0, 1.0])],
    ),
]

_WEIGHTS = [1.0, 2.0, 3.0, 1.0]


def _flatten(value):
    """Recursively flatten an accumulator ``value()`` (nested tuples/arrays/dicts) into a float vector."""
    out: list[float] = []

    def rec(o):
        if isinstance(o, dict):
            for v in o.values():
                rec(v)
        elif isinstance(o, (tuple, list)):
            for v in o:
                rec(v)
        elif isinstance(o, np.ndarray):
            out.extend(np.asarray(o, dtype=float).ravel().tolist())
        elif o is None:
            return
        else:
            try:
                out.append(float(o))
            except (TypeError, ValueError):
                return

    rec(value)
    return np.asarray(out, dtype=float)


def _weighted_suffstat(est, xs, ws):
    acc = est.accumulator_factory().make()
    for x, w in zip(xs, ws):
        acc.update(x, float(w), None)
    return acc.value()


def _replicated_suffstat(est, xs, ws):
    acc = est.accumulator_factory().make()
    for x, w in zip(xs, ws):
        for _ in range(int(w)):
            acc.update(x, 1.0, None)
    return acc.value()


def _weighted_fit(est, xs, ws):
    return est.estimate(float(np.sum(ws)), _weighted_suffstat(est, xs, ws))


def _replicated_fit(est, xs, ws):
    reps = float(sum(int(w) for w in ws))
    return est.estimate(reps, _replicated_suffstat(est, xs, ws))


class WeightedSufficientStatisticTest(unittest.TestCase):
    """The contract at the accumulator (all families, including iterative-MLE ones)."""

    def test_weight_equals_replication(self):
        for name, dist, xs in _CLOSED_FORM + _SUFFSTAT_ONLY:
            with self.subTest(family=name):
                est = dist.estimator()
                w = _flatten(_weighted_suffstat(est, xs, _WEIGHTS))
                r = _flatten(_replicated_suffstat(est, xs, _WEIGHTS))
                # a dropped weight (using 1.0 for every datum) cannot reproduce the replicated sums
                self.assertTrue(np.allclose(w, r, rtol=1e-9, atol=1e-9), f"{name}: {w} != {r}")

    def test_zero_weight_is_a_no_op(self):
        for name, dist, xs in _CLOSED_FORM + _SUFFSTAT_ONLY:
            with self.subTest(family=name):
                est = dist.estimator()
                base = _flatten(_weighted_suffstat(est, xs, _WEIGHTS))
                padded = _flatten(_weighted_suffstat(est, xs + [xs[0]], _WEIGHTS + [0.0]))
                self.assertTrue(np.allclose(base, padded, rtol=1e-9, atol=1e-9), f"{name}: zero weight changed it")


class WeightedFitTest(unittest.TestCase):
    """The contract end to end at the fitted distribution (closed-form-MLE families)."""

    def test_weighted_fit_equals_replicated_fit(self):
        for name, dist, xs in _CLOSED_FORM:
            with self.subTest(family=name):
                est = dist.estimator()
                self.assertEqual(
                    str(_weighted_fit(est, xs, _WEIGHTS)),
                    str(_replicated_fit(est, xs, _WEIGHTS)),
                    f"{name}: weighted fit != replicated fit",
                )

    def test_fit_is_weight_scale_invariant(self):
        for name, dist, xs in _CLOSED_FORM:
            with self.subTest(family=name):
                est = dist.estimator()
                base = _weighted_fit(est, xs, _WEIGHTS)
                scaled = _weighted_fit(est, xs, [5.0 * v for v in _WEIGHTS])
                self.assertEqual(str(base), str(scaled), f"{name}: fit changed under uniform weight scaling")


if __name__ == "__main__":
    unittest.main()
