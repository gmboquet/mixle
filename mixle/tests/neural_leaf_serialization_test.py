"""Serialization + robustness regression tests for the neural-leaf families (mixle.models).

Covers the group-F2 audit fixes:
  1. the ``build_*`` helpers expose MODULE-LEVEL nn.Module classes, so a wrapped leaf (and any mixture holding
     one) survives ``pickle`` -- the prerequisite for distributed EM;
  2. every neural leaf has a working ``to_dict``/``from_dict`` and ``to_json``/``from_json`` that persists the
     module and reproduces ``log_density`` after reload;
  3. the VAE's ``log_density`` is deterministic (no per-call resample), so an EM log-likelihood is well-defined;
  4. non-finite input raises a clear error at the density boundary instead of silently returning NaN.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import mixle.stats as st  # noqa: E402
from mixle.inference import optimize  # noqa: E402
from mixle.models.energy import EnergyModel, build_energy_net  # noqa: E402
from mixle.models.mixture_density import (  # noqa: E402
    NeuralConditionalDensity,
    build_conditional_autoregressive_categorical,
    build_conditional_flow,
    build_mdn,
)
from mixle.models.neural import make_mlp  # noqa: E402
from mixle.models.neural_density import (  # noqa: E402
    NeuralDensity,
    build_autoregressive_categorical,
    build_coupling_flow,
    build_maf,
    build_vae,
)
from mixle.models.neural_leaf import NeuralGaussian  # noqa: E402
from mixle.models.softmax_leaf import NeuralCategorical  # noqa: E402


def _seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)


# --- unconditional density leaves: (name, distribution, encoder-input) --------------------------------------
def _unconditional_cases():
    _seed()
    x_cont = np.random.RandomState(0).randn(6, 2)
    x_disc = np.random.RandomState(1).randint(0, 3, (6, 2)).astype(float)
    return [
        ("coupling", NeuralDensity(build_coupling_flow(2, layers=4)), x_cont),
        ("maf", NeuralDensity(build_maf(2, blocks=2)), x_cont),
        ("vae", NeuralDensity(build_vae(2, latent=2, hidden=16)), x_cont),
        ("arcat", NeuralDensity(build_autoregressive_categorical(2, 3, hidden=16)), x_disc),
        ("energy", EnergyModel(build_energy_net(2, hidden=16)), x_cont),
    ]


# --- conditional density leaves over (x, y) pairs -----------------------------------------------------------
def _conditional_cases():
    _seed()
    x = np.random.RandomState(0).randn(6, 2)
    y_cont = np.random.RandomState(2).randn(6, 2)
    y_disc = np.random.RandomState(3).randint(0, 3, (6, 2)).astype(float)
    yint = np.random.RandomState(4).randint(0, 3, 6)
    return [
        ("gaussian", NeuralGaussian(make_mlp(2, [8], 2), noise=0.7), (x, np.random.RandomState(5).randn(6, 2))),
        ("categorical", NeuralCategorical(make_mlp(2, [8], 3)), (x, yint)),
        ("mdn", NeuralConditionalDensity(build_mdn(2, 2, k=3)), (x, y_cont)),
        ("cond_flow", NeuralConditionalDensity(build_conditional_flow(2, 2, layers=4)), (x, y_cont)),
        (
            "cond_arcat",
            NeuralConditionalDensity(build_conditional_autoregressive_categorical(2, 2, 3)),
            (x, y_disc),
        ),
    ]


class PickleRoundTripTest(unittest.TestCase):
    def test_unconditional_leaves_pickle(self):
        import pickle

        for name, dist, x in _unconditional_cases():
            ll = dist.seq_log_density(x)
            reloaded = pickle.loads(pickle.dumps(dist))
            self.assertTrue(np.allclose(reloaded.seq_log_density(x), ll, atol=1e-5), name)

    def test_conditional_leaves_pickle(self):
        import pickle

        for name, dist, enc in _conditional_cases():
            ll = dist.seq_log_density(enc)
            reloaded = pickle.loads(pickle.dumps(dist))
            self.assertTrue(np.allclose(reloaded.seq_log_density(enc), ll, atol=1e-5), name)

    def test_mixture_with_neural_leaf_pickles(self):
        # the concrete blocker the audit names: pickling a mixture that holds a neural leaf (distributed EM)
        _seed()
        mix = st.MixtureDistribution(
            [
                NeuralDensity(build_coupling_flow(2, layers=4)),
                st.MultivariateGaussianDistribution(np.zeros(2), np.eye(2)),
            ],
            [0.5, 0.5],
        )
        import pickle

        x = np.random.RandomState(0).randn(5, 2)
        enc = mix.dist_to_encoder().seq_encode([r for r in x])
        ll = mix.seq_log_density(enc)
        m2 = pickle.loads(pickle.dumps(mix))
        got = m2.seq_log_density(m2.dist_to_encoder().seq_encode([r for r in x]))
        self.assertTrue(np.allclose(got, ll, atol=1e-5))


class DictJsonRoundTripTest(unittest.TestCase):
    def _check(self, cls, dist, enc):
        from mixle.utils.serialization import trusted_deserialization

        ll = dist.seq_log_density(enc)
        with trusted_deserialization():  # embedded torch module: a self-produced, trusted round-trip
            d = cls.from_dict(dist.to_dict())
            self.assertTrue(np.allclose(d.seq_log_density(enc), ll, atol=1e-5), "to_dict")
            j = cls.from_json(dist.to_json())
            self.assertTrue(np.allclose(j.seq_log_density(enc), ll, atol=1e-5), "to_json")

    def test_unconditional_leaves_dict_json(self):
        classes = {"energy": EnergyModel}
        for name, dist, x in _unconditional_cases():
            self._check(classes.get(name, NeuralDensity), dist, x)

    def test_conditional_leaves_dict_json(self):
        classes = {"gaussian": NeuralGaussian, "categorical": NeuralCategorical}
        for name, dist, enc in _conditional_cases():
            self._check(classes.get(name, NeuralConditionalDensity), dist, enc)

    def test_mixture_with_neural_leaf_json(self):
        _seed()
        mix = st.MixtureDistribution(
            [NeuralDensity(build_maf(2, blocks=2)), st.MultivariateGaussianDistribution(np.zeros(2), np.eye(2))],
            [0.5, 0.5],
        )
        x = np.random.RandomState(0).randn(5, 2)
        enc = mix.dist_to_encoder().seq_encode([r for r in x])
        ll = mix.seq_log_density(enc)
        from mixle.utils.serialization import trusted_deserialization

        with trusted_deserialization():  # embedded torch module: a self-produced, trusted round-trip
            m2 = st.MixtureDistribution.from_json(mix.to_json())
        got = m2.seq_log_density(m2.dist_to_encoder().seq_encode([r for r in x]))
        self.assertTrue(np.allclose(got, ll, atol=1e-5))


class ModuleDeserializationTrustGateTest(unittest.TestCase):
    """A NeuralLeaf's JSON round-trip must refuse to execute its embedded module without explicit trust."""

    def test_from_json_refuses_without_trusted_deserialization(self):
        from mixle.utils.serialization import SerializationError

        dist = NeuralGaussian(_mlp_gaussian())
        payload = dist.to_json()
        with self.assertRaises(SerializationError):
            NeuralGaussian.from_json(payload)  # gate closed by default: no trust context entered

    def test_module_from_bytes_refuses_directly(self):
        from mixle.models._neural_serial import module_from_bytes, module_to_bytes
        from mixle.utils.serialization import SerializationError

        data = module_to_bytes(_mlp_gaussian())
        with self.assertRaises(SerializationError):
            module_from_bytes(data)

    def test_module_from_bytes_succeeds_inside_trusted_deserialization(self):
        from mixle.models._neural_serial import module_from_bytes, module_to_bytes
        from mixle.utils.serialization import trusted_deserialization

        module = _mlp_gaussian()
        data = module_to_bytes(module)
        with trusted_deserialization():
            restored = module_from_bytes(data)
        self.assertEqual(type(restored).__name__, type(module).__name__)


def _mlp_gaussian():
    from mixle.models.neural import make_mlp

    return make_mlp(input_dim=1, hidden_dims=[4], output_dim=2)


class NeuralCategoricalMinibatchTest(unittest.TestCase):
    def test_fixed_optimizer_budget_is_recorded(self):
        _seed()
        rng = np.random.RandomState(8)
        data = [(rng.randn(2).astype("float32"), int(i % 2)) for i in range(24)]
        leaf = NeuralCategorical(
            make_mlp(2, [8], 2),
            m_steps=20,
            batch_size=5,
            max_optimizer_steps=7,
        )
        fitted = optimize(data, leaf.estimator(), prev_estimate=leaf, max_its=1, out=None)
        self.assertEqual(fitted.fit_receipt["optimizer_steps"], 7)
        self.assertEqual(fitted.fit_receipt["batch_size"], 5)
        self.assertEqual(
            fitted.fit_receipt["gradient_estimator"],
            "N/B responsibility-weighted cross-entropy",
        )


class VAEDeterminismTest(unittest.TestCase):
    def test_log_density_deterministic_across_calls(self):
        # before the fix, log_density resampled z each call (~0.7 nats jitter) -> a non-monotone EM LL
        _seed()
        vae = NeuralDensity(build_vae(2, latent=3, hidden=16))
        x = np.random.RandomState(0).randn(20, 2)
        a = vae.seq_log_density(x)
        b = vae.seq_log_density(x)
        self.assertTrue(np.array_equal(a, b))  # bit-identical, not merely close

    def test_reloaded_vae_also_deterministic(self):
        _seed()
        vae = NeuralDensity(build_vae(2, latent=2, hidden=16))
        x = np.random.RandomState(1).randn(10, 2)
        import pickle

        v2 = pickle.loads(pickle.dumps(vae))
        self.assertTrue(np.array_equal(v2.seq_log_density(x), v2.seq_log_density(x)))
        self.assertTrue(np.allclose(v2.seq_log_density(x), vae.seq_log_density(x), atol=1e-5))


class MonotoneEMTest(unittest.TestCase):
    def test_neural_leaf_in_mixture_optimizes(self):
        # a neural leaf drops into a MixtureDistribution and EM runs to a finite, improved fit
        _seed()
        r = np.random.RandomState(0)
        hi = r.rand(300) < 0.5
        x = np.where(hi[:, None], r.randn(300, 2) * 0.3 + [3, 3], r.randn(300, 2) * 0.3 + [-3, -3])
        train = [row for row in x]
        est = st.MixtureEstimator(
            [NeuralDensity(build_coupling_flow(2, layers=6)).estimator(), st.MultivariateGaussianEstimator(dim=2)]
        )
        init = st.MixtureDistribution(
            [
                NeuralDensity(build_coupling_flow(2, layers=6)),
                st.MultivariateGaussianDistribution(np.zeros(2), np.eye(2)),
            ],
            [0.5, 0.5],
        )
        fit = optimize(train, est, prev_estimate=init, max_its=8, out=None)
        self.assertEqual(len(fit.components), 2)
        enc = fit.dist_to_encoder().seq_encode(train)
        self.assertTrue(np.isfinite(fit.seq_log_density(enc)).all())

    def test_vae_em_log_likelihood_is_well_defined(self):
        # with deterministic scoring, evaluating the fit's LL twice is bit-identical (no stochastic drift)
        _seed()
        r = np.random.RandomState(0)
        hi = r.rand(300) < 0.5
        x = np.where(hi[:, None], r.randn(300, 2) * 0.3 + [3, 3], r.randn(300, 2) * 0.3 + [-3, -3])
        train = [row for row in x]
        vae = NeuralDensity(build_vae(2, latent=2, hidden=32), m_steps=40, lr=5e-3)
        fit = optimize(train, vae.estimator(), prev_estimate=vae, max_its=6, out=None)
        enc = fit.dist_to_encoder().seq_encode(train)
        ll1 = float(np.sum(fit.seq_log_density(enc)))
        ll2 = float(np.sum(fit.seq_log_density(enc)))
        self.assertEqual(ll1, ll2)


class NonFiniteInputTest(unittest.TestCase):
    def test_unconditional_leaves_reject_nan(self):
        _seed()
        for name, dist, x in _unconditional_cases():
            bad = np.array(x, dtype=float)
            bad[0, 0] = np.nan
            with self.assertRaises(ValueError, msg=name):
                dist.seq_log_density(bad)

    def test_conditional_leaves_reject_nan(self):
        _seed()
        for name, dist, enc in _conditional_cases():
            x, y = enc
            bad_x = np.array(x, dtype=float)
            bad_x[0, 0] = np.inf
            with self.assertRaises(ValueError, msg=name):
                dist.seq_log_density((bad_x, y))


if __name__ == "__main__":
    unittest.main()
