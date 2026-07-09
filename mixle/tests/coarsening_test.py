"""Acceptance tests for mixle.models.coarsening (roadmap G3: coarsening operator R with per-scale
receipts).

Each test below IS an acceptance criterion from the roadmap item, not just a smoke test:
    1. A 2x depth cut on a real small LM stays within a STATED, held-out divergence budget, data-free
       (measured entirely via the propagated-law closed-form KL, never real data/activations).
    2. The closed-form per-scale receipt (:func:`gaussian_kl`) matches a direct sampling-based KL estimate.
    3. The receipt map correlates with REAL per-layer error, independently measured via a Monte Carlo
       forward-pass comparison of the actual teacher (two sequential blocks) vs. student (merged block).
    4. An artificially tiny trust region correctly rejects merges (keeps the original blocks).
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.coarsening import (
    coarsen,
    depth_merge,
    gaussian_kl,
    structure_project,
    width_merge,
)
from mixle.models.moment_propagation import GaussianLaw
from mixle.models.transformer import build_causal_lm

pytestmark = pytest.mark.fast


def _random_law(rng: np.random.Generator, d: int, scale: float = 0.5) -> GaussianLaw:
    mu = rng.normal(size=d) * 0.1
    a = rng.normal(size=(d, d)) * scale
    covar = a @ a.T + 0.1 * np.eye(d)
    return GaussianLaw(mu=mu, covar=covar)


def _scale_block_(blk, factor: float) -> None:
    """In-place scale a Block's weight (not bias) parameters -- used to synthesize block PAIRS with
    deliberately varying residual magnitude, so the depth-merge second-order truncation error (and hence
    its closed-form receipt) varies in a controlled way across test cases.
    """
    with torch.no_grad():
        for p_name in ("qkv", "proj"):
            getattr(blk.attn, p_name).weight.mul_(factor)
        blk.mlp[0].weight.mul_(factor)
        blk.mlp[2].weight.mul_(factor)


class DepthCutAcceptanceTest(unittest.TestCase):
    """Acceptance criterion: "2x depth cut on a small LM within stated held-out budget, data-free"."""

    def test_two_x_depth_cut_within_stated_budget(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        d_model, n_head, n_layer = 16, 2, 6
        model = build_causal_lm(vocab=23, d_model=d_model, n_layer=n_layer, n_head=n_head, block=16)
        law = _random_law(rng, d_model, scale=0.3)

        budget = 5.0
        result = coarsen(model, budget=budget, trust_region=budget, input_law=law)

        # 2x depth cut: n_layer=6 -> 3 merged blocks.
        self.assertEqual(result.model.n_layer, n_layer // 2)
        self.assertEqual(len(result.accepted_pairs), n_layer // 2)
        self.assertEqual(len(result.rejected_pairs), 0)

        # Data-free budget check: the ACCUMULATED closed-form KL (never touching real data) stays within
        # the stated budget.
        self.assertLessEqual(result.total_kl, budget)
        self.assertTrue(result.within_budget)
        for receipt in result.receipt_map.values():
            self.assertTrue(np.isfinite(receipt.kl_divergence))
            self.assertGreaterEqual(receipt.kl_divergence, 0.0)

        print(
            f"\n[G3 acceptance] depth {n_layer} -> {result.model.n_layer} "
            f"(2x cut), total_kl={result.total_kl:.6f} <= budget={budget}, "
            f"per-merge KLs={[r.kl_divergence for r in result.receipt_map.values()]}"
        )


class ClosedFormReceiptCorrectnessTest(unittest.TestCase):
    """Acceptance criterion: "the per-scale receipt is CLOSED-FORM ... cross-check against a direct
    numerical/sampling-based KL estimate for a couple of cases".
    """

    def _sampling_kl(self, p: GaussianLaw, q: GaussianLaw, rng: np.random.Generator, n: int = 400_000) -> float:
        samples = rng.multivariate_normal(mean=p.mu, cov=p.covar, size=n)
        log_p = p.seq_log_density(samples)
        log_q = q.seq_log_density(samples)
        return float(np.mean(log_p - log_q))

    def test_closed_form_matches_sampling_estimate_case_1(self):
        rng = np.random.default_rng(1)
        p = _random_law(rng, d=4, scale=0.4)
        q = _random_law(rng, d=4, scale=0.6)
        closed = gaussian_kl(p, q)
        sampled = self._sampling_kl(p, q, rng)
        self.assertAlmostEqual(closed, sampled, delta=0.05 * max(closed, 1.0))

    def test_closed_form_matches_sampling_estimate_case_2(self):
        rng = np.random.default_rng(2)
        p = _random_law(rng, d=6, scale=0.2)
        q = _random_law(rng, d=6, scale=0.2)
        closed = gaussian_kl(p, q)
        sampled = self._sampling_kl(p, q, rng)
        self.assertAlmostEqual(closed, sampled, delta=0.05 * max(closed, 1.0))

    def test_identical_laws_have_zero_kl(self):
        rng = np.random.default_rng(3)
        p = _random_law(rng, d=5)
        self.assertAlmostEqual(gaussian_kl(p, p), 0.0, delta=1e-8)


class ReceiptCorrelatesWithRealErrorTest(unittest.TestCase):
    """Acceptance criterion: "receipt map correlates with realized per-layer error" -- independently
    measure REAL per-layer error via Monte Carlo forward-pass comparison (teacher = real sequential
    block_a-then-block_b, student = the merged block), across several pairs with deliberately varying
    residual magnitude, and confirm the closed-form receipt's KL correlates with it.
    """

    def test_kl_receipt_correlates_with_monte_carlo_forward_pass_error(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(4)
        d_model, n_head = 16, 2
        law = _random_law(rng, d_model, scale=0.3)

        scales = [0.15, 0.4, 0.8, 1.3, 2.0]
        kls = []
        real_errors = []
        for i, scale in enumerate(scales):
            model = build_causal_lm(vocab=23, d_model=d_model, n_layer=2, n_head=n_head, block=16)
            block_a, block_b = model.blocks[0], model.blocks[1]
            _scale_block_(block_b, scale)

            merged, receipt = depth_merge(block_a, block_b, law, seed=100 + i)
            kls.append(receipt.kl_divergence)

            # REAL Monte Carlo forward-pass comparison: sample token-embedding-shaped vectors from the
            # SAME input_law, run the real teacher (block_a then block_b, sequentially) and the real
            # student (merged) and measure the actual output discrepancy.
            x = torch.as_tensor(rng.multivariate_normal(mean=law.mu, cov=law.covar, size=(64, 4)), dtype=torch.float32)
            with torch.no_grad():
                teacher_out = block_b(block_a(x))
                student_out = merged(x)
            err = float(torch.mean((teacher_out - student_out) ** 2))
            real_errors.append(err)

        kls = np.asarray(kls)
        real_errors = np.asarray(real_errors)
        correlation = float(np.corrcoef(kls, real_errors)[0, 1])

        print(
            f"\n[G3 acceptance] scales={scales}\n  receipt KLs={kls}\n  "
            f"real MC errors={real_errors}\n  pearson r={correlation:.4f}"
        )

        self.assertGreater(correlation, 0.7)


class TrustRegionRejectionTest(unittest.TestCase):
    """Acceptance criterion: "an artificially tiny trust-region budget correctly causes a bad merge to be
    rejected/skipped".
    """

    def test_tiny_trust_region_rejects_all_merges(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(5)
        d_model, n_head, n_layer = 16, 2, 4
        model = build_causal_lm(vocab=23, d_model=d_model, n_layer=n_layer, n_head=n_head, block=16)
        law = _random_law(rng, d_model, scale=0.3)

        result = coarsen(model, budget=1e-12, trust_region=1e-12, input_law=law)

        self.assertEqual(result.accepted_pairs, [])
        # Every adjacent pair is attempted-and-rejected as the walk advances one block at a time when
        # nothing merges (n_layer - 1 attempts for n_layer blocks), and every original block is kept.
        self.assertEqual(len(result.rejected_pairs), n_layer - 1)
        self.assertEqual(result.model.n_layer, n_layer)  # nothing merged -> depth unchanged
        for receipt in result.receipt_map.values():
            if not receipt.accepted and receipt.name == "depth_merge":
                self.assertGreater(receipt.kl_divergence, 1e-12)

    def test_generous_trust_region_accepts_the_same_merges(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(5)
        d_model, n_head, n_layer = 16, 2, 4
        model = build_causal_lm(vocab=23, d_model=d_model, n_layer=n_layer, n_head=n_head, block=16)
        law = _random_law(rng, d_model, scale=0.3)

        result = coarsen(model, budget=1e6, trust_region=1e6, input_law=law)

        self.assertEqual(len(result.accepted_pairs), n_layer // 2)
        self.assertEqual(result.rejected_pairs, [])
        self.assertEqual(result.model.n_layer, n_layer // 2)


class WidthMergeAndStructureProjectSmokeTest(unittest.TestCase):
    """Smoke coverage for the other two moves G3 composes (not the primary depth-cut acceptance
    criterion, but exercised so a regression in either is caught here too).
    """

    def test_width_merge_reduces_dimension_and_receipt_is_finite_nonnegative(self):
        rng = np.random.default_rng(6)
        law = _random_law(rng, d=10, scale=0.4)
        rep, receipt = width_merge(model=None, target_width=6, input_law=law)
        self.assertEqual(rep.merge.shape, (6, 10))
        self.assertEqual(rep.unmerge.shape, (10, 6))
        self.assertTrue(np.isfinite(receipt.kl_divergence))
        self.assertGreaterEqual(receipt.kl_divergence, 0.0)

    def test_width_merge_identity_width_has_zero_receipt(self):
        rng = np.random.default_rng(7)
        law = _random_law(rng, d=8, scale=0.4)
        _rep, receipt = width_merge(model=None, target_width=8, input_law=law)
        self.assertAlmostEqual(receipt.kl_divergence, 0.0, delta=1e-6)

    def test_structure_project_wraps_g2_low_rank_directly(self):
        rng = np.random.default_rng(8)
        w = rng.normal(size=(8, 8))
        a = rng.normal(size=(8, 8))
        sigma = a @ a.T + 0.1 * np.eye(8)
        from mixle.models.sigma_weighted_projection import sigma_weighted_low_rank

        what_direct = sigma_weighted_low_rank(w, sigma, rank=3)
        what_wrapped, receipt = structure_project(w, sigma, mode="low_rank", rank=3)
        np.testing.assert_allclose(what_direct, what_wrapped)
        self.assertEqual(receipt.mode, "low_rank")
        self.assertGreaterEqual(receipt.sigma_weighted_error, 0.0)


if __name__ == "__main__":
    unittest.main()
