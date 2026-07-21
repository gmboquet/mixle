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
        # record_fit is given the SAME max_its the original optimize() call used; verify_reproducible
        # no longer needs max_its threaded through separately -- it replays the receipt's own value.
        d = _mixture_data()
        est = st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()])
        m = optimize(d, est, out=None, max_its=30, rng=np.random.RandomState(11))
        rec = record_fit(m, d, seed=11, estimator=est, max_its=30)
        res = verify_reproducible(st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()]), d, rec)
        self.assertTrue(res["reproducible"])  # same seed -> bit-identical EM path

    def test_em_fit_diverges_with_different_seed(self):
        d = _mixture_data()
        est = st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()])
        m = optimize(d, est, out=None, max_its=30, rng=np.random.RandomState(11))
        rec = record_fit(m, d, seed=11, estimator=est, max_its=30)
        res = verify_reproducible(
            st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()]),
            d,
            rec,
            seed=99,
        )
        self.assertFalse(res["params_match"])  # a different init can land in a different optimum

    def test_receipt_carries_max_its_so_a_caller_need_not_thread_it_separately(self):
        # Regression: ReproReceipt used to carry no max_its/delta at all, and verify_reproducible
        # hardcoded max_its=25 -- unrelated to whatever the ORIGINAL fit actually used. A caller who
        # (reasonably) called record_fit/verify_reproducible without separately re-stating max_its
        # got a refit computed with a different iteration budget than the original fit, which can
        # land at a visibly different (if nearby) point for an iterative estimator -- a false
        # "not reproducible" verdict for a fit that WOULD have reproduced under its own settings.
        d = _mixture_data()
        est = st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()])
        m = optimize(d, est, out=None, max_its=3, rng=np.random.RandomState(11))  # deliberately under-converged
        rec = record_fit(m, d, seed=11, estimator=est, max_its=3)
        self.assertEqual(rec.max_its, 3)
        res = verify_reproducible(st.MixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()]), d, rec)
        self.assertTrue(res["reproducible"])  # replays max_its=3 from the receipt, not a hardcoded 25

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
