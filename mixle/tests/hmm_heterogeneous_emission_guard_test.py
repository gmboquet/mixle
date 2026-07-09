"""Regression tests for the HMM heterogeneous-emission guard.

A ``HiddenMarkovModelDistribution`` encodes every observation sequence with a single emission encoder
-- the first emission's (``topics[0].dist_to_encoder()`` / ``accumulators[0].acc_to_encoder()``) --
and scores every hidden state through it. If the states carry emissions of different families, the
others are scored through the first's encoder: usually a finite but WRONG log-likelihood (a silent
failure that "runs" and returns a number), occasionally a confusing error deep in encoding.

``HiddenMarkovEstimator`` and ``HiddenMarkovModelDistribution`` now reject that at construction time.
The check compares the *structure* each emission encoder produces (not the encoder's class), so
different families that encode identically -- Gaussian and Exponential both yield a plain ``(N,)``
float array -- are still allowed, while families that encode differently -- a neural ``GradLeaf``, a
Categorical, a Gamma -- are rejected.
"""

import unittest

from mixle.inference import optimize
from mixle.stats import (
    CategoricalDistribution,
    ExponentialEstimator,
    GaussianDistribution,
    GaussianEstimator,
    HiddenMarkovEstimator,
    MixtureEstimator,
)
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution

# The reported bug pairs a classical emission with a neural GradLeaf. mixle.models (and torch) may be
# unavailable (e.g. an unrelated import break on the release branch until the models fix lands), so the
# neural case is skipped rather than erroring collection.
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


def _mixture():
    return MixtureEstimator([GaussianEstimator(), GaussianEstimator()])


class HmmHeterogeneousEmissionGuardTestCase(unittest.TestCase):
    def test_estimator_rejects_mixture_and_categorical(self):
        # A continuous mixture and a discrete Categorical encode incompatibly; the estimator must
        # reject at construction, not silently return a fitted-but-wrong HMM.
        from mixle.stats import CategoricalEstimator

        with self.assertRaises(ValueError):
            HiddenMarkovEstimator([_mixture(), CategoricalEstimator()])

    def test_distribution_rejects_heterogeneous_topics(self):
        with self.assertRaises(ValueError):
            HiddenMarkovModelDistribution(
                [GaussianDistribution(0.0, 1.0), CategoricalDistribution({"a": 1.0})],
                w=[0.5, 0.5],
                transitions=[[0.9, 0.1], [0.1, 0.9]],
            )

    def test_error_message_names_offending_state_and_fix(self):
        from mixle.stats import CategoricalEstimator

        with self.assertRaises(ValueError) as ctx:
            HiddenMarkovEstimator([_mixture(), CategoricalEstimator()])
        message = str(ctx.exception)
        self.assertIn("emission 1", message)
        self.assertIn("emission 0", message)
        # points the user at the correct construction
        self.assertIn("Composite", message)
        self.assertIn("JointMixture", message)

    def test_homogeneous_emissions_allowed(self):
        # Same family in every state: the common, valid case must construct and fit unchanged.
        HiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()])
        est = HiddenMarkovEstimator([_mixture(), _mixture()])
        data = [[0.1, 0.2, 5.0, 5.1], [5.2, 0.0, 0.1, 4.9, 5.0]]
        model = optimize(data, est, max_its=3, out=None)
        self.assertEqual(model.n_states, 2)

    def test_same_encoding_different_family_allowed(self):
        # Gaussian and Exponential are different families but encode identically to a plain (N,) float
        # array, so an HMM can legitimately mix them. The guard must NOT reject this (no false positive).
        est = HiddenMarkovEstimator([GaussianEstimator(), ExponentialEstimator()])
        data = [[0.5, 1.0, 4.0, 3.5], [4.1, 0.2, 0.3, 3.9]]
        model = optimize(data, est, max_its=3, out=None)
        self.assertEqual(model.n_states, 2)

    @unittest.skipUnless(_HAVE_NEURAL, "mixle.models / torch unavailable")
    def test_estimator_rejects_neural_leaf_emission(self):
        # The reported bug: a Gaussian mixture in one state and a neural GradLeaf in another. Before
        # the guard, optimize() returned a finite but wrong log-likelihood; now construction raises.
        with self.assertRaises(ValueError):
            HiddenMarkovEstimator([_mixture(), GradEstimator(_ConstantNormal())])


if __name__ == "__main__":
    unittest.main()
