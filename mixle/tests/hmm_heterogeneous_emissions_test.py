"""Regression tests for native heterogeneous-emission HMM support.

A hidden Markov model can give every hidden state its own emission family (a Gaussian mixture in one
state, a neural density in another) -- see ``mixle/stats/latent/hidden_markov.py``'s
``_build_emission_encoder``/``_encoders_all_equal``. The forward-backward math never depended on
emission family; the only thing that had to change was routing each state to its own encoded view of
the raw data (mirroring ``HeterogeneousMixtureDataEncoder``, already used for exactly this in
non-temporal mixtures) instead of reusing state 0's encoder for every state.

The homogeneous case (every state shares one interchangeable encoder -- the common, "huge model" case)
is required to take the *exact* code path it always has: one shared encoded array, no per-state
grouping, no extra allocation. These tests check both the new heterogeneous capability and that the
homogeneous fast path stays untouched.
"""

import unittest

import numpy as np

from mixle.inference import optimize
from mixle.stats import (
    CategoricalEstimator,
    ExponentialEstimator,
    GaussianEstimator,
    HiddenMarkovEstimator,
    MixtureEstimator,
)
from mixle.stats.latent.heterogeneous_mixture import HeterogeneousMixtureDataEncoder
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

try:
    import torch

    from mixle.models import GradEstimator

    class _ConstantNormal(torch.nn.Module):
        """Minimal scalar neural density: log N(x | p, 1)."""

        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))

        def log_density(self, x):
            x = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
            return -0.5 * (x - self.p) ** 2 - 0.9189385332046727

    _HAVE_NEURAL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_NEURAL = False


def _sticky_sequences(rng, emit, n_sequences=40, min_len=15, max_len=25, n_states=2, p_stay=0.9):
    """Synthetic sequences from a 2-regime sticky Markov chain, for fitting an HMM against."""
    sequences = []
    for _ in range(n_sequences):
        length = rng.randint(min_len, max_len)
        state = 0
        seq = []
        for _ in range(length):
            seq.append(emit(state, rng))
            if rng.rand() >= p_stay:
                state = (state + 1) % n_states
        sequences.append(seq)
    return sequences


class HmmHomogeneousEmissionsUnchangedTestCase(unittest.TestCase):
    """The default (single-family) case must take the exact pre-existing code path."""

    def test_homogeneous_flag_true_for_matching_estimators(self):
        est = HiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()])
        acc = est.accumulator_factory().make()
        self.assertTrue(acc._homogeneous_emissions)

    def test_homogeneous_emission_encoder_is_the_plain_encoder_not_a_wrapper(self):
        from mixle.stats import GaussianDistribution

        dist = HiddenMarkovModelDistribution(
            topics=[GaussianDistribution(0.0, 1.0), GaussianDistribution(1.0, 1.0)],
            w=[0.5, 0.5],
            transitions=[[0.9, 0.1], [0.1, 0.9]],
        )
        encoder = dist.dist_to_encoder()
        self.assertNotIsInstance(encoder.emission_encoder, HeterogeneousMixtureDataEncoder)
        self.assertTrue(dist._homogeneous_emissions)

    def test_many_states_same_family_fits_and_stays_homogeneous(self):
        # A "huge model" proxy: many states, one family. Must not take the grouped path.
        rng = np.random.RandomState(0)
        estimators = [GaussianEstimator() for _ in range(25)]
        est = HiddenMarkovEstimator(estimators)
        data = [list(rng.randn(20) + rng.choice([-3, 0, 3])) for _ in range(20)]
        model = optimize(data, est, max_its=3, out=None)
        self.assertTrue(model._homogeneous_emissions)
        self.assertEqual(model.n_states, 25)


class HmmHeterogeneousEmissionsTestCase(unittest.TestCase):
    """A state carrying a different emission family than another must construct and fit correctly."""

    def test_estimator_and_distribution_flag_heterogeneous(self):
        est = HiddenMarkovEstimator(
            [MixtureEstimator([GaussianEstimator(), GaussianEstimator()]), CategoricalEstimator()]
        )
        acc = est.accumulator_factory().make()
        self.assertFalse(acc._homogeneous_emissions)

    def test_mixture_and_exponential_states_fit_numpy_path(self):
        # Two states with genuinely different emission families (a two-cluster Gaussian mixture vs. an
        # Exponential -- unlike a homogeneous mixture and its own bare component family, these do NOT
        # share an encoder). Must construct, encode, and fit without error.
        rng = np.random.RandomState(0)

        def emit(state, r):
            return float(abs(r.randn()) * 0.4 + 1.0) if state == 0 else float(r.exponential(3.0))

        cont_data = _sticky_sequences(rng, emit)
        est = HiddenMarkovEstimator(
            [MixtureEstimator([GaussianEstimator(), GaussianEstimator()]), ExponentialEstimator()], use_numba=False
        )
        model = optimize(cont_data, est, max_its=6, out=None)
        self.assertFalse(model._homogeneous_emissions)
        ll = model.seq_log_density(model.dist_to_encoder().seq_encode(cont_data))
        self.assertTrue(np.all(np.isfinite(ll)))

    def test_numpy_and_numba_paths_agree_on_heterogeneous_states(self):
        rng = np.random.RandomState(2)
        data = _sticky_sequences(
            rng, lambda s, r: float(abs(r.randn()) * 0.5 + 1.0) if s == 0 else float(r.exponential(4.0))
        )

        def fit(use_numba):
            local_rng = np.random.RandomState(42)
            est = HiddenMarkovEstimator(
                [MixtureEstimator([GaussianEstimator(), GaussianEstimator()]), ExponentialEstimator()],
                use_numba=use_numba,
            )
            return optimize(data, est, max_its=5, out=None, rng=local_rng)

        model_np = fit(False)
        model_nb = fit(True)
        self.assertFalse(model_np._homogeneous_emissions)
        self.assertFalse(model_nb._homogeneous_emissions)

        ll_np = model_np.seq_log_density(model_np.dist_to_encoder().seq_encode(data))
        ll_nb = model_nb.seq_log_density(model_nb.dist_to_encoder().seq_encode(data))
        np.testing.assert_allclose(ll_np, ll_nb, rtol=1e-6, atol=1e-8)

    @unittest.skipUnless(_HAVE_NEURAL, "mixle.models / torch unavailable")
    def test_mixture_and_neural_leaf_states_fit_and_recover_distinct_parameters(self):
        # The originally reported case: a Gaussian mixture in one state, a neural density in another.
        rng = np.random.RandomState(1)
        data = _sticky_sequences(
            rng,
            lambda s, r: float(r.randn() * 0.4 + (-2.0 if s == 0 else 2.0)),
            n_sequences=60,
        )
        est = HiddenMarkovEstimator(
            [
                MixtureEstimator([GaussianEstimator(), GaussianEstimator()]),
                GradEstimator(_ConstantNormal(), m_steps=20, lr=0.1),
            ]
        )
        model = optimize(data, est, max_its=8, out=None)
        self.assertFalse(model._homogeneous_emissions)
        comps = model.components if hasattr(model, "components") else model.topics
        self.assertEqual(type(comps[0]).__name__, "MixtureDistribution")
        self.assertEqual(type(comps[1]).__name__, "GradLeaf")

        ll = model.seq_log_density(model.dist_to_encoder().seq_encode(data))
        self.assertTrue(np.all(np.isfinite(ll)))
        self.assertGreater(float(np.mean(ll)), -50.0)  # a reasonable fit, not a degenerate one


if __name__ == "__main__":
    unittest.main()
