"""Workstream A2: the neural-leaf composition grid, made checkable.

Every neural leaf's own module docstring claims it "drops into a MixtureDistribution /
CompositeDistribution / HMM emission like any leaf." Mixture composition is already covered by
neural_leaf_serialization_test.py and friends; this file closes the HMM-emission gap (untested for
every leaf before this) and extends Composite-field coverage to a genuinely heterogeneous record mixing
multiple neural leaf types with a classical family -- with a JSON round trip proven in every case, not
assumed. See docs/neural-llm.rst's "Composition Grid" table for the full coverage matrix.

Workstream A2b closes the two cells the table originally left "not yet exercised": NeuralGaussian and
NeuralConditionalDensity (both conditional, p(y|x)) in a Composite field and an HMM emission.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.inference import optimize
from mixle.models import EnergyModel, Flow, build_energy_net, make_mlp
from mixle.models.mixture_density import NeuralConditionalDensity, build_mdn
from mixle.models.neural_leaf import NeuralGaussian
from mixle.models.softmax_leaf import NeuralCategorical
from mixle.stats import (
    CompositeDistribution,
    CompositeEstimator,
    GaussianDistribution,
    HiddenMarkovEstimator,
    HiddenMarkovModelDistribution,
    PoissonDistribution,
    PoissonEstimator,
)
from mixle.utils.serialization import from_json, to_json

pytestmark = pytest.mark.fast


def _unconditional_seqs(dim=2, n=30, seed=0):
    init = HiddenMarkovModelDistribution(
        [Flow(dim=dim), Flow(dim=dim)], [0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], len_dist=PoissonDistribution(6.0)
    )
    sampler = init.sampler(seed=seed)
    return init, [sampler.sample() for _ in range(n)]


def _conditional_seqs(x_dim=3, n_classes=2, n=30, seed=0):
    # NeuralCategorical has no unconditional sample() (it is p(y|x), never p(x)) -- an HMM whose every
    # state is such a leaf cannot use its own sampler, so the (x, y) sequences are supplied directly.
    rng = np.random.RandomState(seed)

    def rand_seq():
        t = rng.randint(3, 7)
        return [(rng.randn(x_dim).astype("float32"), int(rng.randint(0, n_classes))) for _ in range(t)]

    return [rand_seq() for _ in range(n)]


def _mlp_categorical(x_dim=3, n_classes=2, m_steps=15):
    return NeuralCategorical(make_mlp(x_dim, [8], n_classes), m_steps=m_steps)


class UnconditionalNeuralHmmTest:
    """An HMM whose emissions are flows -- the flagship claim from neural_density.py's docstring."""

    def test_fits_and_scores_finite(self):
        init, data = _unconditional_seqs()
        est = HiddenMarkovEstimator(
            [Flow(dim=2).estimator(), Flow(dim=2).estimator()], len_estimator=PoissonEstimator()
        )
        fitted = optimize(data, est, prev_estimate=init, max_its=2, out=None)
        assert isinstance(fitted, HiddenMarkovModelDistribution)
        assert np.isfinite(fitted.log_density(data[0]))

    def test_serializes(self):
        init, data = _unconditional_seqs()
        est = HiddenMarkovEstimator(
            [Flow(dim=2).estimator(), Flow(dim=2).estimator()], len_estimator=PoissonEstimator()
        )
        fitted = optimize(data, est, prev_estimate=init, max_its=2, out=None)
        ll = fitted.log_density(data[0])
        back = from_json(to_json(fitted))
        assert np.isclose(back.log_density(data[0]), ll)


class ConditionalNeuralHmmTest:
    """An HMM whose emissions are NeuralCategorical -- fitting works with no unconditional sampler."""

    def test_fits_on_externally_supplied_sequences(self):
        data = _conditional_seqs()
        init = HiddenMarkovModelDistribution(
            [_mlp_categorical(), _mlp_categorical()],
            [0.5, 0.5],
            [[0.9, 0.1], [0.1, 0.9]],
            len_dist=PoissonDistribution(5.0),
        )
        est = HiddenMarkovEstimator(
            [_mlp_categorical().estimator(), _mlp_categorical().estimator()], len_estimator=PoissonEstimator()
        )
        fitted = optimize(data, est, prev_estimate=init, max_its=2, out=None)
        assert np.isfinite(fitted.log_density(data[0]))

    def test_serializes(self):
        data = _conditional_seqs()
        init = HiddenMarkovModelDistribution(
            [_mlp_categorical(), _mlp_categorical()],
            [0.5, 0.5],
            [[0.9, 0.1], [0.1, 0.9]],
            len_dist=PoissonDistribution(5.0),
        )
        est = HiddenMarkovEstimator(
            [_mlp_categorical().estimator(), _mlp_categorical().estimator()], len_estimator=PoissonEstimator()
        )
        fitted = optimize(data, est, prev_estimate=init, max_its=2, out=None)
        ll = fitted.log_density(data[0])
        back = from_json(to_json(fitted))
        assert np.isclose(back.log_density(data[0]), ll)


class HeterogeneousNeuralCompositeTest:
    """One Composite record mixing two different neural leaf types with a classical family."""

    def _fields(self):
        flow = Flow(dim=2)
        energy = EnergyModel(build_energy_net(dim=2, hidden=16), m_steps=20)
        cat = _mlp_categorical()
        gauss = GaussianDistribution(0.0, 1.0)
        return flow, energy, cat, gauss

    def _rows(self, n=40, seed=0):
        rng = np.random.RandomState(seed)
        return [
            (
                rng.randn(2).astype("float32"),  # flow field: unconditional vector
                rng.randn(2).astype("float32"),  # energy field: unconditional vector
                (rng.randn(3).astype("float32"), int(rng.randint(0, 2))),  # categorical field: (x, y)
                float(rng.randn()),  # classical scalar field
            )
            for _ in range(n)
        ]

    def test_bare_composite_scores_finite(self):
        flow, energy, cat, gauss = self._fields()
        comp = CompositeDistribution((flow, energy, cat, gauss))
        rows = self._rows()
        assert np.isfinite(comp.log_density(rows[0]))

    def test_fits_jointly_by_em(self):
        flow, energy, cat, gauss = self._fields()
        est = CompositeEstimator((flow.estimator(), energy.estimator(), cat.estimator(), gauss.estimator()))
        rows = self._rows()
        fitted = optimize(rows, est, max_its=2, out=None)
        assert isinstance(fitted, CompositeDistribution)
        assert np.isfinite(fitted.log_density(rows[0]))

    def test_serializes(self):
        flow, energy, cat, gauss = self._fields()
        est = CompositeEstimator((flow.estimator(), energy.estimator(), cat.estimator(), gauss.estimator()))
        rows = self._rows()
        fitted = optimize(rows, est, max_its=2, out=None)
        ll = fitted.log_density(rows[0])
        back = from_json(to_json(fitted))
        assert np.isclose(back.log_density(rows[0]), ll)


def _mlp_gaussian(x_dim=3, m_steps=15):
    return NeuralGaussian(make_mlp(x_dim, [8], 1), noise=1.0, m_steps=m_steps)


def _mdn(x_dim=3, y_dim=1, m_steps=15):
    return NeuralConditionalDensity(build_mdn(x_dim, y_dim, k=2, hidden=8), m_steps=m_steps)


def _xy_rows(x_dim=3, n=40, seed=0):
    rng = np.random.RandomState(seed)
    return [(rng.randn(x_dim).astype("float32"), np.array([float(rng.randn())], dtype="float32")) for _ in range(n)]


class NeuralGaussianCompositeAndHmmTest:
    """A2b: closes the 'not yet exercised' NeuralGaussian x {Composite, HMM} cells."""

    def test_composite_field_scores_fits_and_serializes(self):
        ng, gauss = _mlp_gaussian(), GaussianDistribution(0.0, 1.0)
        comp = CompositeDistribution((ng, gauss))
        rows = [(xy, float(x)) for xy, x in zip(_xy_rows(), np.random.RandomState(1).randn(40))]
        assert np.isfinite(comp.log_density(rows[0]))
        est = CompositeEstimator((ng.estimator(), gauss.estimator()))
        fitted = optimize(rows, est, max_its=2, out=None)
        ll = fitted.log_density(rows[0])
        assert np.isfinite(ll)
        back = from_json(to_json(fitted))
        assert np.isclose(back.log_density(rows[0]), ll)

    def test_hmm_emission_fits_and_serializes(self):
        rng = np.random.RandomState(0)

        def rand_seq():
            t = rng.randint(3, 7)
            return [(rng.randn(3).astype("float32"), np.array([float(rng.randn())], dtype="float32")) for _ in range(t)]

        data = [rand_seq() for _ in range(20)]
        init = HiddenMarkovModelDistribution(
            [_mlp_gaussian(), _mlp_gaussian()], [0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], len_dist=PoissonDistribution(5.0)
        )
        est = HiddenMarkovEstimator(
            [_mlp_gaussian().estimator(), _mlp_gaussian().estimator()], len_estimator=PoissonEstimator()
        )
        fitted = optimize(data, est, prev_estimate=init, max_its=2, out=None)
        ll = fitted.log_density(data[0])
        assert np.isfinite(ll)
        back = from_json(to_json(fitted))
        assert np.isclose(back.log_density(data[0]), ll)


class NeuralConditionalDensityCompositeAndHmmTest:
    """A2b: closes the 'not yet exercised' NeuralConditionalDensity x {Composite, HMM} cells."""

    def test_composite_field_scores_fits_and_serializes(self):
        mdn, gauss = _mdn(), GaussianDistribution(0.0, 1.0)
        comp = CompositeDistribution((mdn, gauss))
        rows = [(xy, float(x)) for xy, x in zip(_xy_rows(), np.random.RandomState(1).randn(40))]
        assert np.isfinite(comp.log_density(rows[0]))
        est = CompositeEstimator((mdn.estimator(), gauss.estimator()))
        fitted = optimize(rows, est, max_its=2, out=None)
        ll = fitted.log_density(rows[0])
        assert np.isfinite(ll)
        back = from_json(to_json(fitted))
        assert np.isclose(back.log_density(rows[0]), ll)

    def test_hmm_emission_fits_and_serializes(self):
        rng = np.random.RandomState(0)

        def rand_seq():
            t = rng.randint(3, 7)
            return [(rng.randn(3).astype("float32"), np.array([float(rng.randn())], dtype="float32")) for _ in range(t)]

        data = [rand_seq() for _ in range(20)]
        init = HiddenMarkovModelDistribution(
            [_mdn(), _mdn()], [0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], len_dist=PoissonDistribution(5.0)
        )
        est = HiddenMarkovEstimator([_mdn().estimator(), _mdn().estimator()], len_estimator=PoissonEstimator())
        fitted = optimize(data, est, prev_estimate=init, max_its=2, out=None)
        ll = fitted.log_density(data[0])
        assert np.isfinite(ll)
        back = from_json(to_json(fitted))
        assert np.isclose(back.log_density(data[0]), ll)
