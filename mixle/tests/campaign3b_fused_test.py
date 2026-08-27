"""Fused diagonal-Gaussian scoring must agree with the Python scorer at every magnitude.

The fused kernel used to evaluate the natural-parameter expansion ``x*x*ca + x*cb + cc`` while
:meth:`DiagonalGaussianDistribution.log_density` had already been repaired to the centered
``-0.5*sum (x-mu)^2/covar``.  The two are algebraically equal and numerically unrelated on offset
data: at ``|x| ~ 1.7e9`` with unit variance the expansion returned ``+512.0`` per row -- a POSITIVE
log-density, impossible for a Gaussian whose maximum is ``log_c`` -- where the true value was about
``-4.57``.  Nothing warned, so the same model scored differently depending on whether the fused
engine was engaged.

These tests pin the centered form: fused and Python scoring, mixture evidence, and posteriors must
agree to a few ulp at offsets 0, 1e6, 1e9 and 1.7e9.

``mixle.stats.compute.fused_kernels`` imports numba through
:mod:`mixle.utils.optional_deps`, whose stand-in leaves the kernels running as ordinary Python when
numba is missing, so this module needs no ``importorskip`` for the numpy+scipy CI job.
"""

import unittest
import warnings

import numpy as np

from mixle.stats.compute.fused_kernels import CompiledMixture, build_kernel
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution

# The offsets the campaign brief names: exact at 0, ~1e-4 nats lost at 1e6, total loss past 1e9.
OFFSETS = (0.0, 1.0e6, 1.0e9, 1.7e9)


def _rows(array):
    return [row for row in array]


def _python_scores(dist, array):
    encoder = dist.dist_to_encoder()
    return np.asarray(dist.seq_log_density(encoder.seq_encode(_rows(array))))


def _fused_scores(dist, array):
    compiled = CompiledMixture(dist)
    return np.asarray(compiled.seq_log_density(compiled.encode(_rows(array)), model=dist))


class FusedDiagonalGaussianScoringTest(unittest.TestCase):
    """Parity between the fused kernel and the Python diagonal-Gaussian scorer."""

    def setUp(self):
        self.dim = 3
        self.covar = np.array([1.0, 2.0, 0.5])
        self.shape = np.array([0.25, -0.5, 1.0])
        self.base = np.random.default_rng(7).normal(size=(64, self.dim))

    def _model_and_data(self, offset):
        mu = np.full(self.dim, offset) + self.shape
        dist = DiagonalGaussianDistribution(mu, self.covar)
        return dist, self.base + offset

    def test_fused_matches_python_log_density_at_every_offset(self):
        """The measured defect: fused scoring diverged by up to 519 nats at 1.7e9."""
        for offset in OFFSETS:
            with self.subTest(offset=offset):
                dist, data = self._model_and_data(offset)
                expected = _python_scores(dist, data)
                got = _fused_scores(dist, data)
                worst = float(np.max(np.abs(got - expected)))
                self.assertLess(
                    worst,
                    1.0e-9,
                    "fused/Python divergence of %.6g nats at offset %.3g" % (worst, offset),
                )
                np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-12)

    def test_fused_log_density_never_exceeds_the_normalizer(self):
        """A Gaussian log-density is bounded above by ``log_c``; the expansion returned +512.0."""
        for offset in OFFSETS:
            with self.subTest(offset=offset):
                dist, data = self._model_and_data(offset)
                got = _fused_scores(dist, data)
                self.assertTrue(
                    np.all(got <= dist.log_c + 1.0e-9),
                    "fused log-density %.6g exceeded log_c=%.6g at offset %.3g"
                    % (float(np.max(got)), float(dist.log_c), offset),
                )

    def test_fused_scoring_is_shift_equivariant(self):
        """Shifting data and mean together must not move the score: it is a function of x-mu."""
        reference = _fused_scores(*self._model_and_data(0.0))
        for offset in OFFSETS[1:]:
            with self.subTest(offset=offset):
                shifted = _fused_scores(*self._model_and_data(offset))
                worst = float(np.max(np.abs(shifted - reference)))
                self.assertLess(worst, 1.0e-6, "shift of %.3g moved the fused score by %.6g nats" % (offset, worst))

    def test_fused_mixture_evidence_and_posteriors_match_python(self):
        """Wrong component scores flip responsibilities outright, so EM optimizes a fiction."""
        rng = np.random.default_rng(3)
        n = 200
        for offset in OFFSETS:
            with self.subTest(offset=offset):
                data = np.concatenate(
                    [
                        rng.normal(size=(n, 2)) + offset,
                        rng.normal(size=(n, 2)) * 0.5 + offset + 4.0,
                    ]
                )
                components = [
                    DiagonalGaussianDistribution(np.full(2, offset - 1.0), np.ones(2)),
                    DiagonalGaussianDistribution(np.full(2, offset + 5.0), np.ones(2)),
                ]
                model = MixtureDistribution(components, [0.5, 0.5])
                encoder = model.dist_to_encoder()
                python_encoding = encoder.seq_encode(_rows(data))
                expected = np.asarray(model.seq_log_density(python_encoding))

                compiled = CompiledMixture(model)
                encoding = compiled.encode(_rows(data))
                got = np.asarray(compiled.seq_log_density(encoding, model=model))
                np.testing.assert_allclose(got, expected, rtol=1.0e-12, atol=1.0e-10)

                component = np.asarray(model.seq_component_log_density(python_encoding))
                component = component + np.log(np.asarray(model.w))
                component -= component.max(axis=1, keepdims=True)
                reference = np.exp(component)
                reference /= reference.sum(axis=1, keepdims=True)
                posteriors = np.asarray(compiled.posteriors(encoding, model=model))
                worst = float(np.max(np.abs(posteriors - reference)))
                self.assertLess(worst, 1.0e-9, "posterior divergence %.6g at offset %.3g" % (worst, offset))

    def test_fused_scoring_emits_no_warning(self):
        """Parity is achieved by computing the right answer, not by warning about a wrong one."""
        dist, data = self._model_and_data(1.7e9)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fused_scores(dist, data)
        self.assertEqual([str(entry.message) for entry in caught], [])

    def test_kernel_is_fed_the_centered_parameters(self):
        """White-box guard: the kernel receives (mu, ca, log_c), never the expanded cb/cc.

        Feeding it ``cb``/``cc`` again would restore the cancelling form while every low-magnitude
        test still passed, so the parameter pack is pinned here rather than only its outputs.
        """
        dists = [
            DiagonalGaussianDistribution([0.0, 1.0], [1.0, 2.0]),
            DiagonalGaussianDistribution([3.0, -1.0], [0.5, 4.0]),
        ]
        builder = build_kernel(dists)
        mu, ca, logc = builder.params(dists)
        np.testing.assert_allclose(mu, np.array([d.mu for d in dists]))
        np.testing.assert_allclose(ca, np.array([-0.5 / d.covar for d in dists]))
        np.testing.assert_allclose(logc, np.array([d.log_c for d in dists]))
        self.assertEqual(mu.shape, (2, 2))
        self.assertEqual(logc.shape, (2,))

    def test_low_magnitude_scoring_is_unchanged(self):
        """The repair must not perturb the ordinary case the fused path was already exact on."""
        dist, data = self._model_and_data(0.0)
        np.testing.assert_array_equal(_fused_scores(dist, data), _python_scores(dist, data))


if __name__ == "__main__":
    unittest.main()
