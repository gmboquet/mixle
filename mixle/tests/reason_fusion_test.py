"""ProductOfExpertsFusion / StructuredFusionClassifier -- the trainable Level-3 fusion primitive.

Locks: the fusion equals the analytic Gaussian product-of-experts, it is permutation-invariant and O(N)
(not O(N^2)), it is differentiable, and the classifier learns an exchangeable-evidence task from scratch.
"""

import importlib.util
import unittest

import numpy as np

_HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ProductOfExpertsFusionTest(unittest.TestCase):
    def test_matches_analytic_gaussian_product_of_experts(self):
        import torch

        from mixle.reason import ProductOfExpertsFusion

        torch.manual_seed(0)
        mu = torch.randn(4, 5, 3)  # (batch, n_experts, latent)
        log_prec = torch.randn(4, 5, 3)
        prec = torch.nn.functional.softplus(log_prec)
        fused_mu, fused_prec = ProductOfExpertsFusion(prior_prec=1.0)(mu, log_prec)
        # analytic PoE: precisions add (+ prior), mean is precision-weighted
        want_prec = prec.sum(1) + 1.0
        want_mu = (prec * mu).sum(1) / want_prec
        torch.testing.assert_close(fused_prec, want_prec)
        torch.testing.assert_close(fused_mu, want_mu)

    def test_permutation_invariant_and_differentiable(self):
        import torch

        from mixle.reason import ProductOfExpertsFusion

        torch.manual_seed(1)
        mu = torch.randn(2, 6, 4, requires_grad=True)
        log_prec = torch.randn(2, 6, 4)
        f = ProductOfExpertsFusion()
        a_mu, _ = f(mu, log_prec)
        perm = torch.randperm(6)
        b_mu, _ = f(mu[:, perm], log_prec[:, perm])
        torch.testing.assert_close(a_mu, b_mu)  # order of the experts does not matter
        a_mu.sum().backward()  # gradients flow back to the experts (encoders train through fusion)
        self.assertIsNotNone(mu.grad)
        self.assertGreater(mu.grad.abs().sum().item(), 0.0)

    def test_fusion_flops_is_linear_not_quadratic(self):
        from mixle.reason import fusion_flops

        self.assertEqual(fusion_flops(64, 16), 64 * 16)  # PoE: O(N*M)
        self.assertEqual(fusion_flops(64, 16, attention=True), 64 * 64 * 16)  # attention: O(N^2*M)
        # the gap grows with token count -- the whole point at many-patch/many-token scale
        self.assertGreater(fusion_flops(256, 16, attention=True), 60 * fusion_flops(256, 16))

    def test_classifier_learns_exchangeable_evidence_from_scratch(self):
        import torch

        from mixle.reason import StructuredFusionClassifier

        torch.manual_seed(0)
        rng = np.random.RandomState(0)
        k, latent, n_tok, dtok = 6, 12, 16, 5
        protos = rng.randn(k, latent).astype(np.float32)
        proj = (rng.randn(n_tok, dtok, latent) * 0.6).astype(np.float32)

        def batch(n, seed):
            r = np.random.RandomState(seed)
            y = r.randint(0, k, n)
            x = np.einsum("ndl,bl->bnd", proj, protos[y]) + r.randn(n, n_tok, dtok).astype(np.float32) * 1.0
            return torch.tensor(x.astype(np.float32)), torch.tensor(y)

        model = StructuredFusionClassifier(dtok, latent, k)
        xtr, ytr = batch(1500, 1)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(60):
            for i in range(0, len(xtr), 128):
                loss = torch.nn.functional.cross_entropy(model(xtr[i : i + 128]), ytr[i : i + 128])
                opt.zero_grad()
                loss.backward()
                opt.step()
        xte, yte = batch(1000, 2)
        with torch.no_grad():
            acc = (model(xte).argmax(1) == yte).float().mean().item()
        self.assertGreater(acc, 0.8)  # fusing partial views recovers the class

    def test_hybrid_learns_a_relational_task_that_pure_poe_cannot(self):
        import torch

        from mixle.reason import HybridFusionClassifier, StructuredFusionClassifier

        n_tok, dtok = 8, 4

        def batch(n, seed):  # label depends on token POSITION -- pure PoE is permutation-invariant, blind
            r = np.random.RandomState(seed)
            x = r.randn(n, n_tok, dtok).astype(np.float32)
            y = ((x[:, 0] ** 2).sum(1) > (x[:, 1] ** 2).sum(1)).astype(np.int64)
            return torch.tensor(x), torch.tensor(y)

        def fit_acc(model, epochs):
            xtr, ytr = batch(3000, 1)
            opt = torch.optim.Adam(model.parameters(), lr=3e-3)
            for _ in range(epochs):
                for i in range(0, len(xtr), 128):
                    loss = torch.nn.functional.cross_entropy(model(xtr[i : i + 128]), ytr[i : i + 128])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            xte, yte = batch(1000, 2)
            with torch.no_grad():
                return (model(xte).argmax(1) == yte).float().mean().item()

        torch.manual_seed(0)
        hybrid = fit_acc(HybridFusionClassifier(dtok, 16, 2, n_tok, attn_layers=2), 20)
        torch.manual_seed(0)
        poe = fit_acc(StructuredFusionClassifier(dtok, 16, 2), 20)
        self.assertGreater(hybrid, 0.8)  # the attention layer supplies the relational structure...
        self.assertLess(poe, 0.6)  # ...that permutation-invariant PoE structurally cannot
        self.assertGreater(hybrid, poe + 0.2)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class PoEFusionValidationTest(unittest.TestCase):
    """Regression tests for MXR-080-0288: the torch fusion modules used to accept non-positive/non-finite
    prior_prec, mismatched mu/log_prec shapes, empty token axes, incompatible head counts, and runtime
    token counts different from the constructor's positional table -- silently broadcasting into a
    wrong-shaped or wrong-value product, or dividing by zero, instead of raising."""

    def test_negative_or_non_finite_prior_prec_rejected_at_construction(self):
        from mixle.reason import ProductOfExpertsFusion

        for bad in (-1.0, float("nan"), float("-inf")):
            with self.assertRaises(ValueError):
                ProductOfExpertsFusion(prior_prec=bad)

    def test_zero_prior_prec_still_allowed(self):
        from mixle.reason import ProductOfExpertsFusion

        ProductOfExpertsFusion(prior_prec=0.0)  # a legitimate "no prior" configuration -- must not raise

    def test_mismatched_mu_log_prec_shapes_rejected(self):
        import torch

        from mixle.reason import ProductOfExpertsFusion

        f = ProductOfExpertsFusion()
        with self.assertRaises(ValueError):
            f(torch.randn(2, 5, 3), torch.randn(2, 7, 3))  # different token counts
        with self.assertRaises(ValueError):
            f(torch.randn(2, 1, 3), torch.randn(2, 5, 3))  # broadcast-compatible but semantically wrong

    def test_non_rank_3_inputs_rejected(self):
        import torch

        from mixle.reason import ProductOfExpertsFusion

        f = ProductOfExpertsFusion()
        with self.assertRaises(ValueError):
            f(torch.randn(5, 3), torch.randn(5, 3))  # missing the batch axis

    def test_empty_token_axis_rejected(self):
        import torch

        from mixle.reason import ProductOfExpertsFusion

        f = ProductOfExpertsFusion()
        with self.assertRaises(ValueError):
            f(torch.randn(2, 0, 3), torch.randn(2, 0, 3))

    def test_zero_prior_and_fully_underflowed_precision_raises_not_nan(self):
        import torch

        from mixle.reason import ProductOfExpertsFusion

        f = ProductOfExpertsFusion(prior_prec=0.0)
        mu = torch.zeros(1, 3, 2)
        log_prec = torch.full((1, 3, 2), -1000.0)  # softplus(-1000) underflows to exactly 0.0
        with self.assertRaises(ValueError):
            f(mu, log_prec)

    def test_structured_classifier_rejects_non_positive_dimensions(self):
        from mixle.reason import StructuredFusionClassifier

        with self.assertRaises(ValueError):
            StructuredFusionClassifier(token_dim=0, latent_dim=4, n_classes=2)
        with self.assertRaises(ValueError):
            StructuredFusionClassifier(token_dim=4, latent_dim=-1, n_classes=2)
        with self.assertRaises(ValueError):
            StructuredFusionClassifier(token_dim=4, latent_dim=4, n_classes=0)

    def test_hybrid_classifier_rejects_incompatible_head_count(self):
        from mixle.reason import HybridFusionClassifier

        with self.assertRaises(ValueError):
            HybridFusionClassifier(token_dim=4, latent_dim=10, n_classes=2, n_tokens=8, heads=3)

    def test_hybrid_classifier_rejects_non_positive_dimensions(self):
        from mixle.reason import HybridFusionClassifier

        with self.assertRaises(ValueError):
            HybridFusionClassifier(token_dim=4, latent_dim=8, n_classes=2, n_tokens=0, heads=4)
        with self.assertRaises(ValueError):
            HybridFusionClassifier(token_dim=4, latent_dim=8, n_classes=2, n_tokens=8, attn_layers=0)

    def test_hybrid_classifier_rejects_runtime_token_count_mismatch(self):
        import torch

        from mixle.reason import HybridFusionClassifier

        model = HybridFusionClassifier(token_dim=4, latent_dim=8, n_classes=2, n_tokens=8, heads=4)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 4))  # built for n_tokens=8, called with 3
        model(torch.randn(2, 8, 4))  # the matching count must still work


if __name__ == "__main__":
    unittest.main()
