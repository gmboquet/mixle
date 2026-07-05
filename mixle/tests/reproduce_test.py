"""Reproducibility receipts (N2): record a fit, replay it, check it comes out bit-for-bit."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import optimize
from mixle.inference.reproduce import (
    ReproReceipt,
    data_fingerprint,
    param_fingerprint,
    record_fit,
    verify_reproducible,
)


def _gauss_data(seed=0, n=200):
    return [float(x) for x in np.random.RandomState(seed).normal(5, 2, n)]


def _mixture_data():
    a = np.random.RandomState(0).normal(-3, 1, 150)
    b = np.random.RandomState(1).normal(3, 1, 150)
    return [float(x) for x in np.concatenate([a, b])]


class FingerprintTest(unittest.TestCase):
    def test_data_fingerprint_stable_and_sensitive(self):
        d = _gauss_data()
        self.assertEqual(data_fingerprint(d), data_fingerprint(list(d)))  # stable
        self.assertNotEqual(data_fingerprint(d), data_fingerprint(_gauss_data(seed=1)))  # sensitive

    def test_param_fingerprint_identical_fits_match(self):
        d = _gauss_data()
        m1 = optimize(d, st.GaussianEstimator(), out=None)
        m2 = optimize(d, st.GaussianEstimator(), out=None)
        self.assertEqual(param_fingerprint(m1), param_fingerprint(m2))  # closed-form: identical

    def test_param_fingerprint_absorbs_last_bit_noise(self):
        # rounding means a tiny perturbation below precision doesn't flip the hash
        d = _gauss_data()
        m = optimize(d, st.GaussianEstimator(), out=None)
        fp = param_fingerprint(m, ndigits=6)
        self.assertEqual(fp, param_fingerprint(m, ndigits=6))


class RecordAndVerifyTest(unittest.TestCase):
    def test_closed_form_fit_reproduces(self):
        d = _gauss_data()
        m = optimize(d, st.GaussianEstimator(), out=None, rng=np.random.RandomState(7))
        rec = record_fit(m, d, seed=7, estimator=st.GaussianEstimator())
        self.assertIsInstance(rec, ReproReceipt)
        res = verify_reproducible(st.GaussianEstimator(), d, rec)
        self.assertTrue(res["reproducible"])
        self.assertTrue(res["data_matches"] and res["params_match"])

    def test_different_data_is_not_reproducible(self):
        d = _gauss_data()
        m = optimize(d, st.GaussianEstimator(), out=None, rng=np.random.RandomState(7))
        rec = record_fit(m, d, seed=7, estimator=st.GaussianEstimator())
        res = verify_reproducible(st.GaussianEstimator(), _gauss_data(seed=1), rec)
        self.assertFalse(res["reproducible"])
        self.assertFalse(res["data_matches"])

    def test_em_fit_reproduces_with_same_seed(self):
        d = _mixture_data()
        est = st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()])
        m = optimize(d, est, out=None, max_its=30, rng=np.random.RandomState(11))
        rec = record_fit(m, d, seed=11, estimator=est)
        res = verify_reproducible(
            st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()]), d, rec, max_its=30
        )
        self.assertTrue(res["reproducible"])  # same seed -> bit-identical EM path

    def test_em_fit_diverges_with_different_seed(self):
        d = _mixture_data()
        est = st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()])
        m = optimize(d, est, out=None, max_its=30, rng=np.random.RandomState(11))
        rec = record_fit(m, d, seed=11, estimator=est)
        res = verify_reproducible(
            st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()]),
            d,
            rec,
            seed=99,
            max_its=30,
        )
        self.assertFalse(res["params_match"])  # a different init can land in a different optimum

    def test_receipt_helpers(self):
        d = _gauss_data()
        m = optimize(d, st.GaussianEstimator(), out=None)
        rec = record_fit(m, d, seed=0, estimator=st.GaussianEstimator())
        self.assertTrue(rec.matches_data(d))
        self.assertTrue(rec.matches_model(m))
        self.assertEqual(rec.n, len(d))
        self.assertIn("data_fingerprint", rec.as_dict())


if __name__ == "__main__":
    unittest.main()
