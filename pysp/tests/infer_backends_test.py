"""Engine-agnostic NUTS backend parity (``pysp.infer`` registry).

The same correlated-Gaussian posterior is expressed once per backend *contract* (a numpy fused
``value_and_grad``; an ``@njit`` fused ``value_and_grad``; a jax scalar ``logp``; a torch scalar
``logp``) and every *available* backend is asked to recover it. Each backend's test is skipped if
its engine is absent (mirroring ``pysp/tests/ppl_engine_test.py``); the numpy and numba backends
must always run.
"""

import importlib.util
import unittest

import numpy as np
from numba import njit

import pysp.infer as infer
from pysp.infer.backends import available_backends, get_inference_backend, select_backend

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
    mu = _MU.copy()
    prec = _PREC.copy()

    @njit(cache=True)
    def vg(x):
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
_TARGETS = {"numpy": _numpy_vg, "numba": _njit_vg}
if HAS_TORCH:
    _TARGETS["torch"] = _torch_logp
if HAS_JAX:
    _TARGETS["jax"] = _jax_logp


class RegistryTest(unittest.TestCase):
    def test_numpy_and_numba_always_available(self):
        avail = available_backends()
        self.assertIn("numpy", avail)
        self.assertIn("numba", avail)

    def test_auto_prefers_numpy(self):
        # No target-kind hint -> the dependency-free numpy path is the default.
        self.assertEqual(select_backend("auto"), "numpy")

    def test_explicit_backend_honored(self):
        self.assertEqual(select_backend("numba"), "numba")

    def test_target_kind_hint_routes(self):
        self.assertEqual(select_backend("auto", target="njit_vg"), "numba")

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            get_inference_backend("does-not-exist")

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

    def test_numba_rhat(self):
        # numba is the always-available "other" backend (it's a core dep).
        self._check_rhat("numba")

    @unittest.skipUnless(HAS_JAX, "jax/numpyro not installed")
    def test_jax_rhat(self):
        self._check_rhat("jax")


if __name__ == "__main__":
    unittest.main()
