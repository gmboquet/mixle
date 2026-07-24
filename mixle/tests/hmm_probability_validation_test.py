"""Constructor-time probability validation for HiddenMarkovModelDistribution.

``HiddenMarkovModelDistribution.__init__`` had zero validation on its three probability-array
parameters: ``w`` (initial hidden-state probabilities), ``transitions`` (the hidden-state
transition matrix), and ``taus`` (optional per-state mixture weights over ``topics``). All three
are documented and used as probability simplexes, but nothing rejected a NaN entry, a negative
entry, or an infinite entry -- each silently produced a NaN or otherwise invalid log-density
downstream (``log_w``/``log_transitions``/``log_taus`` are built with a bare ``np.log`` over the
raw array) instead of raising at construction, matching the bug class already fixed for
``CategoricalDistribution.pmap``, ``IntegerCategoricalDistribution.p_vec``, and
``MixtureDistribution``'s weights.

Deliberately NOT enforced here: each row of ``w``/``transitions``/``taus`` summing to 1.
Instrumenting the constructor and running the full existing test suite (2690 real
``HiddenMarkovModelDistribution`` constructions) found 42 constructions with a ``transitions`` row
that does not sum to 1 -- and every single one was either ordinary float64 fitting noise (e.g.
``0.9999999999999999``) or a row that sums to exactly 0. The zero-row case is not incidental:
``HiddenMarkovEstimator.estimate()`` (this module's own M-step) intentionally leaves the
transition row for a hidden state with no observed outgoing transition mass -- i.e. a state never
visited during fitting -- as all zeros rather than fabricating a uniform (or any other) row. A
hard row-sum-to-1 rejection would break that legitimate, live estimator output. None of the 42
offending constructions had a non-finite or negative entry, so finite+non-negative has no such
counterexample: it is unambiguous, unlike the sum-to-1 case. ``taus[i, :]`` is also fed directly
into ``MixtureDistribution(topics, taus[i, :])`` (see ``HiddenMarkovSampler.__init__``), and that
constructor does not require its weights to sum to 1 either, so the same scope choice keeps ``taus``
consistent with the class it is handed to.

This intentionally lives in its own file rather than in ``probability_range_validation_test.py``
(the file this session's prior sibling fixes for ``CategoricalDistribution``/``MixtureDistribution``
share) because that file is being extended concurrently by another change to
``MixtureDistribution``'s own validation.
"""

import unittest

import numpy as np

from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution


def _gaussian_topics():
    return [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)]


def _categorical_topics():
    return [
        CategoricalDistribution(pmap={"a": 0.8, "b": 0.2}),
        CategoricalDistribution(pmap={"a": 0.1, "b": 0.9}),
    ]


class HiddenMarkovInitialWeightValidationTestCase(unittest.TestCase):
    """Validation of ``w``, the initial hidden-state probability vector."""

    def test_nan_w_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "w"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [float("nan"), 0.5], [[0.9, 0.1], [0.2, 0.8]])

    def test_negative_w_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "w"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [-0.5, 1.5], [[0.9, 0.1], [0.2, 0.8]])

    def test_infinite_w_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "w"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [float("inf"), 0.5], [[0.9, 0.1], [0.2, 0.8]])

    def test_valid_w_still_constructs(self):
        hmm = HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]])
        self.assertTrue(np.isfinite(hmm.log_density([-2.0, -2.1])))


class HiddenMarkovTransitionsValidationTestCase(unittest.TestCase):
    """Validation of ``transitions``, the hidden-state transition matrix."""

    def test_nan_transitions_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "transitions"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[float("nan"), 0.1], [0.2, 0.8]])

    def test_negative_transitions_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "transitions"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[1.3, -0.3], [0.2, 0.8]])

    def test_infinite_transitions_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "transitions"):
            HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[float("inf"), 0.1], [0.2, 0.8]])

    def test_transitions_not_summing_to_one_construct_by_design(self):
        # See this module's docstring: a hard rejection here would break HiddenMarkovEstimator's own
        # M-step output for an unvisited hidden state. Pinned as deliberate, not an oversight.
        over = HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[0.9, 0.4], [0.2, 0.8]])  # row 0 = 1.3
        self.assertAlmostEqual(float(over.transitions[0].sum()), 1.3)
        self.assertTrue(np.isfinite(over.log_density([-2.0, -2.1])))

    def test_transitions_all_zero_row_still_constructs(self):
        # Matches HiddenMarkovEstimator.estimate()'s own output for a hidden state with no observed
        # outgoing transition mass: that row is left all-zero (sum 0), not uniform or renormalized.
        hmm = HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[0.0, 0.0], [0.2, 0.8]])
        self.assertEqual(float(hmm.transitions[0].sum()), 0.0)
        self.assertTrue(np.isfinite(hmm.log_density([-2.0, -2.1])))

    def test_valid_transitions_still_construct(self):
        hmm = HiddenMarkovModelDistribution(_gaussian_topics(), [0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]])
        self.assertTrue(np.isfinite(hmm.log_density([-2.0, -2.1])))
        for use_numba in (True, False):
            hmm_nb = HiddenMarkovModelDistribution(
                _gaussian_topics(), [0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]], use_numba=use_numba
            )
            enc = hmm_nb.dist_to_encoder().seq_encode([[-2.0, -2.1], [2.0, 1.9, 2.2]])
            ll = np.asarray(hmm_nb.seq_log_density(enc), dtype=float)
            self.assertTrue(np.all(np.isfinite(ll)), f"use_numba={use_numba}: {ll}")


class HiddenMarkovTausValidationTestCase(unittest.TestCase):
    """Validation of ``taus``, the optional per-state mixture weights over ``topics``."""

    def test_nan_taus_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "taus"):
            HiddenMarkovModelDistribution(
                _categorical_topics(),
                [0.6, 0.4],
                [[0.9, 0.1], [0.3, 0.7]],
                taus=[[float("nan"), 0.3], [0.2, 0.8]],
            )

    def test_negative_taus_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "taus"):
            HiddenMarkovModelDistribution(
                _categorical_topics(),
                [0.6, 0.4],
                [[0.9, 0.1], [0.3, 0.7]],
                taus=[[-0.1, 1.1], [0.2, 0.8]],
            )

    def test_infinite_taus_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "taus"):
            HiddenMarkovModelDistribution(
                _categorical_topics(),
                [0.6, 0.4],
                [[0.9, 0.1], [0.3, 0.7]],
                taus=[[float("inf"), 0.3], [0.2, 0.8]],
            )

    def test_taus_not_summing_to_one_construct_by_design(self):
        # See this module's docstring: taus[i, :] is fed straight into MixtureDistribution, which
        # itself does not require its weights to sum to 1 either -- pinned as deliberate.
        hmm = HiddenMarkovModelDistribution(
            _categorical_topics(),
            [0.6, 0.4],
            [[0.9, 0.1], [0.3, 0.7]],
            taus=[[0.3, 0.3], [0.2, 0.8]],  # row 0 sums to 0.6
        )
        self.assertAlmostEqual(float(hmm.taus[0].sum()), 0.6)
        self.assertTrue(np.isfinite(hmm.log_density(["a", "b"])))

    def test_valid_taus_still_construct(self):
        hmm = HiddenMarkovModelDistribution(
            _categorical_topics(),
            [0.6, 0.4],
            [[0.9, 0.1], [0.3, 0.7]],
            taus=[[0.7, 0.3], [0.2, 0.8]],
        )
        self.assertTrue(hmm.has_topics)
        self.assertTrue(np.isfinite(hmm.log_density(["a", "b", "a"])))


if __name__ == "__main__":
    unittest.main()
