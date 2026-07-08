"""Fit determinism: the estimation twin of sampler_seed_test.

sampler_seed_test pins that SAMPLING is seed-repeatable; nothing pinned that FITTING is. The gap
bit in practice: ``optimize``/``fit`` resolved ``rng=None`` to a fresh OS-entropy RandomState, so
any un-seeded fit whose family needs a randomized EM init (mixtures, HMMs, anything with latent
structure) returned a different model per call -- the root cause behind the structure-learning CI
flakes. The contract pinned here: an UN-SEEDED fit is deterministic (fixed default seed), a seeded
fit is repeatable, and the same holds through the ``fit`` wrapper. The catalog is representative
(closed-form leaves plus the randomized-init families that actually exercise the rng), not an
auto-discovered sweep -- entries are cheap and the point is the engine contract, not family
coverage.
"""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import fit
from mixle.inference.estimation import optimize


def _catalog():
    """(name, distribution) pairs; the estimator comes from dist.estimator(), data from its sampler."""
    return [
        ("gaussian", st.GaussianDistribution(1.0, 2.0)),
        ("gamma", st.GammaDistribution(2.0, 1.5)),
        ("poisson", st.PoissonDistribution(3.0)),
        ("categorical", st.CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})),
        ("diagonal_gaussian", st.DiagonalGaussianDistribution([0.0, 2.0], [1.0, 1.5])),
        (
            "composite",
            st.CompositeDistribution(
                [st.CategoricalDistribution({"x": 0.7, "y": 0.3}), st.GaussianDistribution(0.0, 1.0)]
            ),
        ),
        (
            "mixture",  # the family whose randomized init made un-seeded fits nondeterministic
            st.MixtureDistribution([st.GaussianDistribution(0.0, 1.0), st.GaussianDistribution(5.0, 1.0)], [0.5, 0.5]),
        ),
        (
            "hmm",
            st.HiddenMarkovModelDistribution(
                [
                    st.CategoricalDistribution({"a": 0.8, "b": 0.2}),
                    st.CategoricalDistribution({"a": 0.1, "b": 0.9}),
                ],
                [0.6, 0.4],
                [[0.7, 0.3], [0.2, 0.8]],
                len_dist=st.CategoricalDistribution({4: 1.0}),
                use_numba=False,
            ),
        ),
    ]


def _fingerprint(model, data):
    """The behavioral identity of a fit: its per-row log-densities on the training data."""
    enc = model.dist_to_encoder().seq_encode(data)
    return np.asarray(model.seq_log_density(enc), dtype=np.float64)


class FitSeedTest(unittest.TestCase):
    def test_unseeded_fit_is_deterministic(self):
        # rng=None must resolve to a FIXED seed: two identical un-seeded calls, bitwise-equal fits.
        # Before the fixed default this failed for 'mixture' and 'hmm' (fresh OS entropy per call).
        for name, dist in _catalog():
            with self.subTest(family=name):
                data = dist.sampler(seed=1).sample(150)
                a = optimize(data, dist.estimator(), max_its=8, out=None)
                b = optimize(data, dist.estimator(), max_its=8, out=None)
                np.testing.assert_array_equal(_fingerprint(a, data), _fingerprint(b, data))

    def test_seeded_fit_is_repeatable(self):
        for name, dist in _catalog():
            with self.subTest(family=name):
                data = dist.sampler(seed=2).sample(150)
                a = optimize(data, dist.estimator(), max_its=8, out=None, rng=np.random.RandomState(7))
                b = optimize(data, dist.estimator(), max_its=8, out=None, rng=np.random.RandomState(7))
                np.testing.assert_array_equal(_fingerprint(a, data), _fingerprint(b, data))

    def test_fit_wrapper_shares_the_contract(self):
        # fit() forwards rng to the same loop; pin it on the randomized-init family specifically
        dist = st.MixtureDistribution(
            [st.GaussianDistribution(0.0, 1.0), st.GaussianDistribution(5.0, 1.0)], [0.5, 0.5]
        )
        data = dist.sampler(seed=3).sample(150)
        a = fit(data, dist.estimator(), max_its=8, out=None)
        b = fit(data, dist.estimator(), max_its=8, out=None)
        np.testing.assert_array_equal(_fingerprint(a, data), _fingerprint(b, data))

    def test_distinct_seeds_still_diversify_restarts(self):
        # the fixed default must NOT collapse deliberate restart diversity: explicit distinct rngs
        # keep producing distinct initializations (they may or may not converge to the same optimum,
        # so assert on the INITIALIZATION path: one EM step from each seed differs on a fixture with
        # symmetric components, where init is the only symmetry breaker).
        dist = st.MixtureDistribution(
            [st.GaussianDistribution(-2.0, 1.0), st.GaussianDistribution(2.0, 1.0)], [0.5, 0.5]
        )
        data = dist.sampler(seed=4).sample(200)
        fits = [
            _fingerprint(optimize(data, dist.estimator(), max_its=1, out=None, rng=np.random.RandomState(s)), data)
            for s in range(6)
        ]
        distinct = {arr.tobytes() for arr in fits}
        self.assertGreater(len(distinct), 1)


if __name__ == "__main__":
    unittest.main()
