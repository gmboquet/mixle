"""Engine-agnostic NUTS backend parity (``mixle.inference`` registry).

The same correlated-Gaussian posterior is expressed once per backend *contract* (a numpy fused
``value_and_grad``; an ``@njit`` fused ``value_and_grad``; a jax scalar ``logp``; a torch scalar
``logp``) and every *available* backend is asked to recover it. Each backend's test is skipped if
its engine is absent (mirroring ``mixle/tests/ppl_engine_test.py``); only the dependency-free numpy
backend always runs.
"""

import importlib.util
import unittest

import numpy as np

import mixle.inference as infer
from mixle.inference.backends import _KIND_PREFERENCE, available_backends, get_inference_backend, select_backend

HAS_NUMBA = importlib.util.find_spec("numba") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_JAX = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("numpyro") is not None

# A fixed correlated-Gaussian target shared by every contract.
_MU = np.array([1.0, -2.0, 0.5])
_COV = np.array([[1.0, 0.4, 0.0], [0.4, 2.0, 0.3], [0.0, 0.3, 0.5]])
_PREC = np.linalg.inv(_COV)


def _numpy_vg():
    mu, prec = _MU, _PREC

    def vg(x):
        x = np.asarray(x, dtype=float)
        d = x - mu
        return float(-0.5 * d @ prec @ d), -prec @ d

    return vg


def _njit_vg():
    from numba import njit

    mu = _MU.copy()
    prec = _PREC.copy()

    @njit  # NB: not cache=True -- caching this closure to disk goes stale across pytest reimports
    def vg(x):  # ("underlying object has vanished"), an intermittent flake; recompiling is cheap here
        d = x - mu
        return -0.5 * (d @ (prec @ d)), -(prec @ d)

    return vg


def _torch_logp():
    import torch

    mu_t = torch.as_tensor(_MU, dtype=torch.float64)
    prec_t = torch.as_tensor(_PREC, dtype=torch.float64)

    def logp(theta):
        d = theta - mu_t
        return -0.5 * (d @ (prec_t @ d))

    return logp


def _jax_logp():
    import jax.numpy as jnp

    mu_j = jnp.asarray(_MU)
    prec_j = jnp.asarray(_PREC)

    def logp(theta):
        d = theta - mu_j
        return -0.5 * (d @ (prec_j @ d))

    return logp


# Build per-backend targets only for the engines that are present.
_TARGETS = {"numpy": _numpy_vg}
if HAS_NUMBA:
    _TARGETS["numba"] = _njit_vg
if HAS_TORCH:
    _TARGETS["torch"] = _torch_logp
if HAS_JAX:
    _TARGETS["jax"] = _jax_logp


class RegistryTest(unittest.TestCase):
    def test_numpy_always_available(self):
        self.assertIn("numpy", available_backends())

    def test_auto_prefers_numpy(self):
        # No target-kind hint -> the dependency-free numpy path is the default.
        self.assertEqual(select_backend("auto"), "numpy")

    @unittest.skipUnless(HAS_NUMBA, "numba is not installed")
    def test_numba_available_when_installed(self):
        self.assertIn("numba", available_backends())

    @unittest.skipUnless(HAS_NUMBA, "numba is not installed")
    def test_explicit_backend_honored(self):
        self.assertEqual(select_backend("numba"), "numba")

    @unittest.skipUnless(HAS_NUMBA, "numba is not installed")
    def test_target_kind_hint_routes(self):
        self.assertEqual(select_backend("auto", target="njit_vg"), "numba")

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            get_inference_backend("does-not-exist")

    def test_unknown_target_kind_is_rejected_not_routed_to_numpy(self):
        # MXR-080-1637: a kind string the registry does not know says something about the target's
        # calling convention; falling through to numpy ignores it and hands the target to a backend
        # that will call it with the wrong protocol.
        for bad in ("nonsense", "", "torch-logp"):
            with self.assertRaises(ValueError):
                select_backend("auto", target=bad)

    def test_unavailable_target_kind_raises_instead_of_falling_back(self):
        # MXR-080-1637: a recognized kind whose engine is not installed must report that, not silently
        # return numpy -- a jax/torch scalar logp is not a numpy value_and_grad.
        unavailable = [k for k, pref in _KIND_PREFERENCE.items() if not any(n in available_backends() for n in pref)]
        if not unavailable:
            self.skipTest("every target kind has an available backend on this host")
        for kind in unavailable:
            with self.assertRaises(RuntimeError):
                select_backend("auto", target=kind)

    def test_available_target_kinds_still_route(self):
        for kind, pref in _KIND_PREFERENCE.items():
            if any(n in available_backends() for n in pref):
                self.assertIn(select_backend("auto", target=kind), pref)

    def test_no_hint_still_defaults_to_numpy(self):
        self.assertEqual(select_backend("auto", target=None), "numpy")

    def test_each_backend_declares_a_target_kind(self):
        for name in available_backends():
            self.assertIn(
                get_inference_backend(name).target_kind,
                {"numpy_vg", "njit_vg", "torch_logp", "jax_logp"},
            )


class BackendParityTest(unittest.TestCase):
    """Every available backend recovers the shared Gaussian posterior."""

    def _check_recovery(self, backend):
        target = _TARGETS[backend]()
        res = infer.nuts(target, backend=backend, dim=3, num_samples=2500, warmup=800, chains=1, rng=0)
        self.assertEqual(res.samples.shape, (2500, 3))
        self.assertTrue(np.all(np.isfinite(res.samples)), msg=f"{backend}: non-finite draws")
        self.assertEqual(res.extra["backend"], backend)
        # Posterior mean within a few MC standard errors per dimension.
        mc_se = np.sqrt(np.diag(_COV) / np.clip(res.ess, 1.0, None))
        err = np.abs(res.samples.mean(axis=0) - _MU)
        self.assertTrue(
            np.all(err < 4.0 * mc_se + 0.1),
            msg=f"{backend}: mean err={err}, 4*se={4 * mc_se}",
        )

    def test_numpy_recovers_posterior(self):
        self._check_recovery("numpy")

    @unittest.skipUnless(HAS_NUMBA, "numba is not installed")
    def test_numba_recovers_posterior(self):
        self._check_recovery("numba")

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch_recovers_posterior(self):
        self._check_recovery("torch")

    @unittest.skipUnless(HAS_JAX, "jax/numpyro not installed")
    def test_jax_recovers_posterior(self):
        self._check_recovery("jax")


class MultiChainRhatTest(unittest.TestCase):
    """Multi-chain R-hat < 1.05 for numpy and at least one other available backend."""

    def _check_rhat(self, backend):
        target = _TARGETS[backend]()
        res = infer.nuts(target, backend=backend, dim=3, num_samples=1500, warmup=700, chains=4, rng=1)
        self.assertEqual(res.chains.shape, (4, 1500, 3))
        self.assertTrue(np.all(res.rhat < 1.05), msg=f"{backend}: rhat={res.rhat}")
        self.assertTrue(np.all(res.ess > 100), msg=f"{backend}: ess={res.ess}")

    def test_numpy_rhat(self):
        self._check_rhat("numpy")

    @unittest.skipUnless(HAS_NUMBA, "numba is not installed")
    def test_numba_rhat(self):
        self._check_rhat("numba")

    @unittest.skipUnless(HAS_JAX, "jax/numpyro not installed")
    def test_jax_rhat(self):
        self._check_rhat("jax")


if __name__ == "__main__":
    unittest.main()
