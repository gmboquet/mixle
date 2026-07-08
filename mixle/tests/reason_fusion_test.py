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


if __name__ == "__main__":
    unittest.main()
