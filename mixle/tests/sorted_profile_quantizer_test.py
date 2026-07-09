"""Acceptance tests for mixle.models.sorted_profile_quantizer (roadmap G4: sorted-profile
(permutation x monotone) quantizer).

Each test targets one line of the G4 acceptance criteria / honest-scope spec directly:

1. ``F6UseCaseTest`` -- F6 use case (synthetic Adam second-moment tensor) at loss parity with a MEASURED
   memory win (a real byte-count comparison, not an assumed one).
2. ``AnomalyDetectionTest`` -- the goodness-of-fit receipt fires on planted anomalies (both directions:
   fires on a corrupted/shifted tensor, does NOT fire on a genuinely similar one).
3. ``DenseFallbackTest`` -- the dense-fallback threshold triggers on a genuinely bad parametric fit
   (well-separated bimodal data against a unimodal family) and does NOT trigger on a well-behaved tensor.
4. ``ReconstructionCorrectnessTest`` -- head-exact reconstruction is exact; tail reconstruction error is
   within the goodness-of-fit-implied tolerance.
"""

import unittest

import numpy as np

from mixle.models.sorted_profile_quantizer import (
    DEFAULT_GOF_THRESHOLD,
    detect_anomaly,
    fit_sorted_profile,
    reconstruct,
)
from mixle.stats import GammaEstimator, GaussianEstimator


def _adam_second_moment_like(rng: np.random.RandomState, n: int, steps: int = 50, beta2: float = 0.98) -> np.ndarray:
    """A synthetic Adam-optimizer second-moment buffer: ``n`` independent parameters, each an exponential
    moving average of squared gradients over ``steps`` synthetic optimization steps -- ``v <- beta2*v +
    (1-beta2)*g**2``, exactly Adam's real update rule for its second-moment buffer. Per-step gradients are
    drawn from a heavy-tailed distribution (Gaussian scaled by an absolute Student-t factor, so occasional
    steps see much larger gradients), but the EMA itself is strictly positive and bounded away from zero (an
    average of many positive terms), matching a REAL optimizer state's behavior far better than a single
    squared-Gaussian draw would -- realistic enough to exercise a genuine Gamma-family tail fit, per F6's
    honest scope note in the module docstring.
    """
    v = np.zeros(n)
    for _ in range(steps):
        g = rng.normal(0.0, 1.0, size=n) * (1.0 + 0.15 * np.abs(rng.standard_t(df=4, size=n)))
        v = beta2 * v + (1.0 - beta2) * g * g
    return v


class F6UseCaseTest(unittest.TestCase):
    """F6 (optimizer states) use case: encode/reconstruct a synthetic Adam second-moment tensor, and
    measure BOTH a real memory footprint win and a real loss-parity proxy (not raw tensor MSE alone).

    Tensor size (16384, i.e. < 2**16) is chosen deliberately so permutation indices fit in uint16 (2
    bytes) against the original float32 values (4 bytes) -- the honest regime the R1 note flags as where
    this scheme actually wins on STORAGE (see the module docstring's "Hardware reality" paragraph): for
    tensors above 2**32 elements, or already-narrow dtypes, the win shrinks or disappears, which is why the
    module does not claim a blanket compression ratio.
    """

    def test_loss_parity_with_measured_memory_win(self):
        rng = np.random.RandomState(0)
        n = 16384
        v_true = _adam_second_moment_like(rng, n)

        encoding = fit_sorted_profile(v_true, top_k=int(0.01 * n), tail_family=GammaEstimator())
        self.assertFalse(encoding.used_dense_fallback, msg=f"unexpected dense fallback, GOF={encoding.goodness_of_fit}")

        v_hat = reconstruct(encoding).astype(np.float64)

        # --- (a) measured memory win: a real byte-count comparison ---
        dense_bytes = v_true.astype(np.float32).nbytes
        encoded_bytes = encoding.nbytes()
        compression_ratio = dense_bytes / encoded_bytes
        print(
            f"[F6] n={n}, dense={dense_bytes}B, encoded={encoded_bytes}B, "
            f"compression_ratio={compression_ratio:.2f}x, GOF(KS-D)={encoding.goodness_of_fit:.4f}"
        )
        self.assertGreater(compression_ratio, 1.5)

        # --- (b) loss parity: a real downstream proxy, not just raw MSE ---
        # Adam's parameter update is grad / (sqrt(v_hat) + eps); substituting the RECONSTRUCTED second
        # moment for the true one is exactly the perturbation F6 would actually inflict on training, so
        # comparing the resulting update vectors (not the raw v values) is the right proxy here -- v itself
        # only ever matters to Adam through this square root and eps-guarded reciprocal.
        grad = rng.normal(0.0, 1.0, size=n) * 0.1
        eps = 1e-8
        update_true = grad / (np.sqrt(v_true) + eps)
        update_hat = grad / (np.sqrt(np.clip(v_hat, 0.0, None)) + eps)

        relative_l2 = np.linalg.norm(update_hat - update_true) / np.linalg.norm(update_true)
        print(f"[F6] Adam-update relative L2 error (loss-parity proxy): {relative_l2:.4f}")
        # "loss parity" -- the substituted update should not meaningfully diverge from the true update
        self.assertLess(relative_l2, 0.1)

        # sanity: reconstruction error is not accidentally zero (i.e. the test is not vacuous)
        self.assertGreater(np.mean((v_hat - v_true) ** 2), 0.0)


class AnomalyDetectionTest(unittest.TestCase):
    """Goodness-of-fit receipt as an anomaly signal: fires on planted anomalies, does not fire on a
    genuinely similar tensor -- both directions checked.
    """

    def setUp(self):
        self.rng = np.random.RandomState(1)
        self.n = 16384
        self.reference_tensor = _adam_second_moment_like(self.rng, self.n)
        self.reference_encoding = fit_sorted_profile(
            self.reference_tensor, top_k=int(0.01 * self.n), tail_family=GammaEstimator()
        )
        self.assertFalse(self.reference_encoding.used_dense_fallback)

    def test_does_not_fire_on_a_similar_tensor(self):
        similar = _adam_second_moment_like(np.random.RandomState(2), self.n)
        report = detect_anomaly(similar, self.reference_encoding)
        print(f"[anomaly/similar] KS-D={report.ks_statistic:.4f} vs reference={report.reference_goodness_of_fit:.4f}")
        self.assertFalse(report.is_anomaly)

    def test_fires_on_a_burst_of_extreme_values(self):
        corrupted = _adam_second_moment_like(np.random.RandomState(3), self.n).copy()
        burst_idx = np.random.RandomState(4).choice(self.n, size=int(0.05 * self.n), replace=False)
        corrupted[burst_idx] += 50.0

        report = detect_anomaly(corrupted, self.reference_encoding)
        print(f"[anomaly/burst] KS-D={report.ks_statistic:.4f} vs reference={report.reference_goodness_of_fit:.4f}")
        self.assertTrue(report.is_anomaly)

    def test_fires_on_a_whole_distribution_shift(self):
        shifted = _adam_second_moment_like(np.random.RandomState(5), self.n) + 3.0

        report = detect_anomaly(shifted, self.reference_encoding)
        print(f"[anomaly/shift] KS-D={report.ks_statistic:.4f} vs reference={report.reference_goodness_of_fit:.4f}")
        self.assertTrue(report.is_anomaly)


class DenseFallbackTest(unittest.TestCase):
    """The dense-fallback threshold: correctly triggers on a tensor whose value distribution is a genuine
    bad fit for the chosen (unimodal) tail family, and does NOT trigger on a well-behaved tensor.
    """

    def test_triggers_on_well_separated_bimodal_data(self):
        rng = np.random.RandomState(6)
        n = 8192
        bimodal = np.concatenate([rng.normal(-5.0, 0.3, n // 2), rng.normal(5.0, 0.3, n // 2)])

        encoding = fit_sorted_profile(bimodal, top_k=10, tail_family=GaussianEstimator())
        print(f"[dense_fallback/bimodal] GOF(KS-D)={encoding.goodness_of_fit:.4f}, threshold={DEFAULT_GOF_THRESHOLD}")
        self.assertTrue(encoding.used_dense_fallback)
        self.assertGreater(encoding.goodness_of_fit, DEFAULT_GOF_THRESHOLD)

        # dense fallback must still round-trip exactly
        reconstructed = reconstruct(encoding)
        np.testing.assert_allclose(reconstructed, bimodal.astype(np.float32))

    def test_does_not_trigger_on_a_well_behaved_gaussian_tensor(self):
        rng = np.random.RandomState(7)
        n = 8192
        well_behaved = rng.normal(0.0, 1.0, size=n)

        encoding = fit_sorted_profile(well_behaved, top_k=10, tail_family=GaussianEstimator())
        print(
            f"[dense_fallback/well_behaved] GOF(KS-D)={encoding.goodness_of_fit:.4f}, threshold={DEFAULT_GOF_THRESHOLD}"
        )
        self.assertFalse(encoding.used_dense_fallback)
        self.assertLess(encoding.goodness_of_fit, DEFAULT_GOF_THRESHOLD)


class ReconstructionCorrectnessTest(unittest.TestCase):
    """Head (top-k outliers) reconstructs exactly; tail reconstruction error is within the
    goodness-of-fit-implied tolerance.
    """

    def test_head_is_reconstructed_exactly(self):
        rng = np.random.RandomState(8)
        n = 4096
        tensor = rng.normal(0.0, 1.0, size=n)
        top_k = 32
        encoding = fit_sorted_profile(tensor, top_k=top_k, tail_family=GaussianEstimator())
        self.assertFalse(encoding.used_dense_fallback)

        reconstructed = reconstruct(encoding)
        head_positions = encoding.top_k_indices.astype(np.int64)
        np.testing.assert_allclose(
            reconstructed[head_positions],
            tensor[head_positions].astype(np.float32),
            rtol=0.0,
            atol=1e-6,
        )
        # every original top-k value is bitwise-near-exact in the encoding itself, independent of placement
        np.testing.assert_allclose(
            np.sort(encoding.top_k_values), np.sort(tensor[head_positions].astype(np.float32)), atol=1e-6
        )

    def test_tail_error_within_goodness_of_fit_tolerance(self):
        rng = np.random.RandomState(9)
        n = 16384
        tensor = rng.normal(0.0, 1.0, size=n)
        encoding = fit_sorted_profile(tensor, top_k=16, tail_family=GaussianEstimator())
        self.assertFalse(encoding.used_dense_fallback)

        reconstructed = reconstruct(encoding).astype(np.float64)
        tail_positions = encoding.permutation_indices.astype(np.int64)

        # The KS receipt bounds the max discrepancy between the fitted CDF and the empirical CDF, i.e. a
        # bound on *quantile-position* error, not a direct per-element value bound -- so the tolerance
        # check here compares the *sorted* reconstructed tail against the *sorted* true tail (the same
        # profile-vs-profile comparison H4's profile_distance uses), scaled by the tensor's own spread.
        # RMSE (not max-abs) is the right summary here: the single largest error is always at the extreme
        # order statistics (the min/max of a sample of Gaussians), whose sampling variability is an
        # intrinsic property of extreme-value statistics -- present even for a PERFECT parametric fit --
        # not a symptom of the fit quality the goodness-of-fit receipt is actually about.
        true_sorted_tail = np.sort(tensor[tail_positions])
        reconstructed_sorted_tail = np.sort(reconstructed[tail_positions])
        rmse = float(np.sqrt(np.mean((reconstructed_sorted_tail - true_sorted_tail) ** 2)))
        spread = float(np.std(tensor))
        print(
            f"[reconstruction/tail] GOF(KS-D)={encoding.goodness_of_fit:.4f}, "
            f"rmse={rmse:.4f}, spread(std)={spread:.4f}, rmse/spread={rmse / spread:.4f}"
        )
        # a good KS fit (small D) should keep the reconstructed sorted profile close to the true one,
        # relative to the tensor's own scale
        self.assertLess(rmse, 0.1 * spread)


if __name__ == "__main__":
    unittest.main()
