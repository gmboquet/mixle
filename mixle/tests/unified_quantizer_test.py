"""Acceptance tests for mixle.models.unified_quantizer (roadmap I1: one quantizer surface with a
method picker).

1. ``ExplicitDispatchTest`` -- each explicit ``method=`` value produces the SAME result as calling
   the underlying original function/class directly (the unification wrapper does not silently
   change behavior).
2. ``ModelZooAutoPickTest`` -- the core acceptance criterion: on a zoo of tensors with genuinely
   different value distributions, at a matched compressed-size budget, ``method="auto"``'s total
   reconstruction error is <= the best SINGLE fixed method's total error across the whole zoo.
3. ``ReceiptTest`` -- per-tensor receipts contain real, non-empty comparison data for every
   rejected method, and the chosen method's stated numbers match direct measurement.
"""

from __future__ import annotations

import unittest

import numpy as np

from mixle.engines.lns import LogNumberSystem
from mixle.models.sorted_profile_quantizer import fit_sorted_profile
from mixle.models.unified_quantizer import (
    METHODS,
    lns_quantize_array,
    quantize_tensor,
)
from mixle.stats import GaussianEstimator
from mixle.task.quantize import dequantize_symmetric
from mixle.task.quantize import quantize_dequantize_array as symmetric_quantize

# --- a small model zoo: genuinely different real-world-shaped value distributions -------------------


def _well_behaved_gaussian(rng: np.random.RandomState, n: int = 256) -> np.ndarray:
    """A well-behaved Gaussian-like weight tensor (e.g. a trained linear layer's weights)."""
    return rng.normal(0.0, 0.5, size=n)


def _heavy_tailed_optimizer_state(rng: np.random.RandomState, n: int = 256) -> np.ndarray:
    """A strictly-positive, heavy-tailed optimizer-state-like tensor (Adam second-moment EMA)."""
    v = np.zeros(n)
    for _ in range(30):
        g = rng.normal(0.0, 1.0, size=n) * (1.0 + 0.2 * np.abs(rng.standard_t(df=4, size=n)))
        v = 0.98 * v + 0.02 * g * g
    return v


def _bimodal_hostile(rng: np.random.RandomState, n: int = 256) -> np.ndarray:
    """A genuinely quantization-hostile bimodal tensor: two well-separated clusters, hostile to
    both a single Gaussian tail fit AND a single linear scale (the bulk of the dynamic range is
    "empty" between the two modes)."""
    half = n // 2
    lo = rng.normal(-5.0, 0.05, size=half)
    hi = rng.normal(5.0, 0.05, size=n - half)
    out = np.concatenate([lo, hi])
    rng.shuffle(out)
    return out


def _sparse_with_outliers(rng: np.random.RandomState, n: int = 256) -> np.ndarray:
    """Mostly-small values with a few extreme outliers (e.g. a gradient tensor with rare spikes)."""
    v = rng.normal(0.0, 0.05, size=n)
    n_outliers = max(1, n // 100)
    idx = rng.choice(n, size=n_outliers, replace=False)
    v[idx] = rng.choice([-1, 1], size=n_outliers) * rng.uniform(8.0, 15.0, size=n_outliers)
    return v


def _log_uniform_multiplicative(rng: np.random.RandomState, n: int = 256) -> np.ndarray:
    """A multiplicative-scale tensor spanning several orders of magnitude, with mixed sign -- the
    shape LNS's log-domain quantization is meant for."""
    mag = np.exp(rng.uniform(np.log(1e-3), np.log(1e2), size=n))
    sign = rng.choice([-1.0, 1.0], size=n)
    return sign * mag


_ZOO_BUILDERS = [
    ("gaussian", _well_behaved_gaussian),
    ("optimizer_state", _heavy_tailed_optimizer_state),
    ("bimodal_hostile", _bimodal_hostile),
    ("sparse_outliers", _sparse_with_outliers),
    ("log_uniform", _log_uniform_multiplicative),
]


def _build_zoo(seed: int = 0) -> list[tuple[str, np.ndarray]]:
    rng = np.random.RandomState(seed)
    return [(name, builder(rng)) for name, builder in _ZOO_BUILDERS]


class ExplicitDispatchTest(unittest.TestCase):
    """Each explicit method= dispatches to the exact same underlying primitive."""

    def setUp(self):
        self.rng = np.random.RandomState(1)
        self.tensor = self.rng.normal(0.0, 2.0, size=200)

    def test_int8_matches_symmetric_quantize_directly(self):
        qt = quantize_tensor(self.tensor, method="int8")
        wq_direct, scale_direct = symmetric_quantize(self.tensor, bits=8)
        np.testing.assert_array_equal(qt.payload.wq, wq_direct)
        self.assertEqual(qt.payload.scale, scale_direct)
        np.testing.assert_array_equal(qt.reconstruct(), dequantize_symmetric(wq_direct, scale_direct))

    def test_int4_matches_symmetric_quantize_directly(self):
        qt = quantize_tensor(self.tensor, method="int4")
        wq_direct, scale_direct = symmetric_quantize(self.tensor, bits=4)
        np.testing.assert_array_equal(qt.payload.wq, wq_direct)
        self.assertEqual(qt.payload.scale, scale_direct)

    def test_lns_matches_log_number_system_directly(self):
        qt = quantize_tensor(self.tensor, method="lns", bits=8)
        payload_direct = lns_quantize_array(self.tensor, bits=8)
        np.testing.assert_array_equal(qt.payload.codes, payload_direct.codes)
        self.assertEqual(qt.payload.step, payload_direct.step)
        self.assertEqual(qt.payload.center, payload_direct.center)
        # And the codes really are what LogNumberSystem.quantize produces on the log-magnitude data.
        log_mag = np.log(np.abs(self.tensor) + 1e-12)
        lns = LogNumberSystem(step=payload_direct.step)
        k_direct = np.clip(lns.quantize(log_mag - payload_direct.center), -127, 127).astype(np.int8)
        np.testing.assert_array_equal(payload_direct.codes, k_direct)

    def test_sorted_profile_matches_fit_sorted_profile_directly(self):
        qt = quantize_tensor(self.tensor, method="sorted_profile", top_k=5, tail_family=GaussianEstimator())
        direct = fit_sorted_profile(self.tensor, top_k=5, tail_family=GaussianEstimator())
        self.assertEqual(qt.payload.used_dense_fallback, direct.used_dense_fallback)
        self.assertEqual(qt.payload.goodness_of_fit, direct.goodness_of_fit)
        if not direct.used_dense_fallback:
            np.testing.assert_array_equal(qt.payload.permutation_indices, direct.permutation_indices)
            np.testing.assert_array_equal(qt.payload.top_k_values, direct.top_k_values)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            quantize_tensor(self.tensor, method="not_a_method")


class ModelZooAutoPickTest(unittest.TestCase):
    """The core I1 acceptance criterion: auto-pick's total error across the zoo is <= the best
    single fixed method's total error across the whole zoo, at a matched byte budget."""

    def test_auto_pick_beats_or_matches_best_single_method(self):
        zoo = _build_zoo(seed=42)
        bits = 8

        auto_total_error = 0.0
        auto_choices = {}
        fixed_totals = dict.fromkeys(METHODS, 0.0)

        for name, tensor in zoo:
            auto_qt = quantize_tensor(tensor, method="auto", bits=bits, top_k=max(1, len(tensor) // 100))
            auto_total_error += auto_qt.receipt.reconstruction_error
            auto_choices[name] = auto_qt.receipt.method

            for m in METHODS:
                fixed_qt = quantize_tensor(tensor, method=m, bits=bits, top_k=max(1, len(tensor) // 100))
                fixed_totals[m] += fixed_qt.receipt.reconstruction_error

        best_single_method = min(fixed_totals, key=fixed_totals.get)
        best_single_total = fixed_totals[best_single_method]

        print(f"\n[I1 model zoo] auto-pick total NMSE = {auto_total_error:.6g}")
        for m, total in sorted(fixed_totals.items(), key=lambda kv: kv[1]):
            print(f"[I1 model zoo] fixed method={m!r:16s} total NMSE = {total:.6g}")
        print(f"[I1 model zoo] per-tensor auto choices: {auto_choices}")
        print(f"[I1 model zoo] best single fixed method: {best_single_method!r} (total NMSE={best_single_total:.6g})")

        self.assertLessEqual(auto_total_error, best_single_total * (1.0 + 1e-9))
        # And the picker should not be trivially reducible to always-the-same-method on this zoo.
        self.assertGreater(len(set(auto_choices.values())), 1)


class ReceiptTest(unittest.TestCase):
    """Per-tensor receipts explain every choice with real, verifiable numbers."""

    def test_auto_receipt_has_real_comparison_data_for_rejected_methods(self):
        rng = np.random.RandomState(7)
        tensor = _heavy_tailed_optimizer_state(rng, n=256)
        qt = quantize_tensor(tensor, method="auto", bits=8, top_k=2)
        receipt = qt.receipt

        self.assertTrue(receipt.auto)
        self.assertEqual(set(receipt.candidates.keys()), set(METHODS))
        rejected = receipt.rejected()
        self.assertEqual(len(rejected), len(METHODS) - 1)
        for method, cand in rejected.items():
            self.assertEqual(cand.method, method)
            self.assertGreaterEqual(cand.nbytes, 0)
            self.assertTrue(np.isfinite(cand.reconstruction_error))
            self.assertTrue(np.isfinite(cand.reward) or cand.reward < 0)

        # The chosen method's stated numbers match direct .reconstruct() measurement.
        v_hat = qt.reconstruct().reshape(-1)
        v = np.asarray(tensor, dtype=np.float64).reshape(-1)
        measured_error = float(np.mean((v - v_hat) ** 2) / np.mean(v**2))
        self.assertAlmostEqual(receipt.reconstruction_error, measured_error, places=9)
        self.assertEqual(receipt.nbytes, qt.payload.nbytes())

    def test_explicit_receipt_is_not_auto_and_has_one_candidate(self):
        rng = np.random.RandomState(3)
        tensor = rng.normal(size=100)
        qt = quantize_tensor(tensor, method="int8")
        self.assertFalse(qt.receipt.auto)
        self.assertEqual(list(qt.receipt.candidates.keys()), ["int8"])
        self.assertEqual(qt.receipt.rejected(), {})


class IneligibleMethodsAreNeverAutoPickedTest(unittest.TestCase):
    """A method whose actual measured bytes exceed the target budget is marked ineligible and the
    picker never chooses it, even on a tensor where it would otherwise reconstruct well."""

    def test_oversized_sorted_profile_is_marked_ineligible_at_a_tiny_budget(self):
        rng = np.random.RandomState(9)
        tensor = rng.normal(size=5000)  # large n -> uint16/uint32 permutation indices
        qt = quantize_tensor(tensor, method="auto", bits=1, top_k=0)  # an unrealistically tiny budget
        sp_cand = qt.receipt.candidates["sorted_profile"]
        self.assertFalse(sp_cand.eligible)
        self.assertNotEqual(qt.receipt.method, "sorted_profile")


if __name__ == "__main__":
    unittest.main()
