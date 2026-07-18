"""Constructible neural-density families: VAE(dim=8, latent=2) instead of NeuralDensity(build_vae(...)).

Covers the two ergonomic asks: a distribution object you construct directly (no build_* + adapter double-wrap),
and composition with a classical density through one top-level estimator() (no per-component .estimator()).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.inference import optimize
from mixle.models import (
    MAF,
    VAE,
    DiscreteAR,
    Flow,
    NeuralDensity,
    NeuralDensityEstimator,
    build_vae,
)
from mixle.stats import MixtureDistribution, MultivariateGaussianDistribution
from mixle.utils.serialization import from_json, to_json


def _data(n=64, d=4, seed=0):
    rng = np.random.RandomState(seed)
    return [rng.randn(d).tolist() for _ in range(n)]


def test_vae_is_a_constructible_distribution_object():
    v = VAE(dim=4, latent=2)
    assert isinstance(v, NeuralDensity)  # it IS a leaf; composes anywhere NeuralDensity does
    xs = np.atleast_2d(np.asarray(_data(), dtype=float))
    ll = v.seq_log_density(xs)
    assert ll.shape == (len(xs),)
    assert np.isfinite(ll).all()
    assert "VAE" in str(v)


@pytest.mark.parametrize(
    "make",
    [
        lambda: VAE(dim=4, latent=2),
        lambda: Flow(dim=4),
        lambda: MAF(dim=4),
    ],
)
def test_families_construct_and_score(make):
    d = make()
    xs = np.atleast_2d(np.asarray(_data(), dtype=float))
    assert np.isfinite(d.seq_log_density(xs)).all()


def test_discrete_ar_constructs_over_discrete_vectors():
    rng = np.random.RandomState(1)
    x = [rng.randint(0, 4, size=3).tolist() for _ in range(48)]
    d = DiscreteAR(dim=3, cats=4)
    xs = np.atleast_2d(np.asarray(x, dtype=float))
    assert np.isfinite(d.seq_log_density(xs)).all()


def test_vae_composes_with_a_classical_density_one_estimator():
    """The headline case: a VAE and a classical Gaussian in one mixture, fit jointly with a single estimator()."""
    x = _data()
    g = MultivariateGaussianDistribution(np.zeros(4), np.eye(4))
    mix = MixtureDistribution([VAE(dim=4, latent=2), g], [0.5, 0.5])
    fitted = optimize(x, mix.estimator(), max_its=3, out=None)  # ONE estimator(), no per-component hop
    names = [type(c).__name__ for c in fitted.components]
    assert "NeuralDensity" in names and "MultivariateGaussianDistribution" in names
    enc = fitted.dist_to_encoder().seq_encode(x)
    assert np.isfinite(fitted.seq_log_density(enc)).all()


def test_neural_density_estimator_is_directly_constructible():
    """The second ask: build an estimator without a dist.estimator() hop."""
    est = NeuralDensityEstimator(build_vae(4, latent=2), m_steps=5)
    factory = est.accumulator_factory()
    acc = factory.make()
    for row in _data(n=16):
        acc.update(np.asarray(row, dtype=float), 1.0, None)
    fitted = est.estimate(16.0, acc.value())
    assert isinstance(fitted, NeuralDensity)


def test_family_json_round_trip_preserves_type_and_scores():
    from mixle.utils.serialization import trusted_deserialization

    a = VAE(dim=4, latent=2)
    with trusted_deserialization():  # embedded torch module: a self-produced, trusted round-trip
        b = from_json(to_json(a))
    assert type(b).__name__ == "VAE"
    xs = np.atleast_2d(np.asarray(_data(), dtype=float))
    assert np.allclose(a.seq_log_density(xs), b.seq_log_density(xs))


def test_mixture_with_a_family_round_trips():
    from mixle.utils.serialization import trusted_deserialization

    x = _data()
    g = MultivariateGaussianDistribution(np.zeros(4), np.eye(4))
    fitted = optimize(x, MixtureDistribution([VAE(dim=4, latent=2), g], [0.5, 0.5]).estimator(), max_its=2, out=None)
    with trusted_deserialization():  # embedded torch module: a self-produced, trusted round-trip
        back = from_json(to_json(fitted))
    enc = fitted.dist_to_encoder().seq_encode(x)
    assert np.allclose(fitted.seq_log_density(enc), back.seq_log_density(back.dist_to_encoder().seq_encode(x)))
