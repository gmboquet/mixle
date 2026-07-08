"""Acceptance tests for mixle.models.sigma_weighted_projection (roadmap G2: Sigma-weighted structured
projections).

Each solver family gets an "optimality" test whose meaning is stated precisely up front (see each test's
docstring -- exact for low-rank, global-optimum-of-the-convex-subproblem for fixed-mask block-sparse,
local-optimum-of-the-alternating-scheme for 2:4, exact-linear-assignment for permutation), plus one
end-to-end acceptance test: a REAL propagated Sigma from G1 (mixle.models.moment_propagation) beats plain
unweighted SVD, at equal rank, on a real small transformer's weight matrix.
"""

import itertools
import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.moment_propagation import GaussianLaw, attention_law, layernorm_law
from mixle.models.sigma_weighted_projection import (
    _project_2_4,
    sigma_weighted_block_sparse,
    sigma_weighted_error,
    sigma_weighted_low_rank,
    sigma_weighted_permutation,
)
from mixle.models.transformer import build_causal_lm

pytestmark = pytest.mark.fast


def _random_sigma(rng: np.random.Generator, d: int, scale: float = 1.0) -> np.ndarray:
    a = rng.normal(size=(d, d)) * scale
    return a @ a.T + 0.1 * np.eye(d)


class LowRankOptimalityTest(unittest.TestCase):
    """Optimality meaning for the low-rank solver: EXACT. The whiten/SVD/un-whiten reduction is a genuine
    closed-form solution (Eckart-Young in the whitened space), so it must (a) match plain unweighted SVD
    exactly when Sigma = I (the objective degenerates to plain Frobenius low-rank), and (b) strictly beat
    plain SVD under the SAME weighted metric whenever Sigma is anisotropic.
    """

    def test_reduces_to_plain_svd_when_sigma_is_identity(self):
        rng = np.random.default_rng(0)
        d_out, d_in, rank = 6, 5, 3
        w = rng.normal(size=(d_out, d_in))

        what = sigma_weighted_low_rank(w, np.eye(d_in), rank)

        u, s, vt = np.linalg.svd(w, full_matrices=False)
        what_plain = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        np.testing.assert_allclose(what, what_plain, atol=1e-8)

    def test_beats_plain_svd_under_the_weighted_metric_on_anisotropic_sigma(self):
        rng = np.random.default_rng(1)
        d_out, d_in, rank = 8, 6, 2
        w = rng.normal(size=(d_out, d_in))
        sigma = _random_sigma(rng, d_in)

        what = sigma_weighted_low_rank(w, sigma, rank)
        err_weighted = sigma_weighted_error(w, what, sigma)

        u, s, vt = np.linalg.svd(w, full_matrices=False)
        what_plain = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        err_plain = sigma_weighted_error(w, what_plain, sigma)

        print(f"[low_rank] sigma-weighted solver error: {err_weighted:.6f}, plain-SVD error: {err_plain:.6f}")
        self.assertLess(err_weighted, err_plain)
        self.assertLessEqual(np.linalg.matrix_rank(what), rank)

    def test_matches_a_gradient_descent_refined_reference(self):
        """Independent numerical cross-check of the closed form: refine a random-init rank-r factorization
        by many steps of plain gradient descent directly on the stated objective and confirm it cannot beat
        (does no better than, within optimizer noise) the closed-form answer.
        """
        rng = np.random.default_rng(2)
        d_out, d_in, rank = 5, 4, 2
        w_np = rng.normal(size=(d_out, d_in))
        sigma_np = _random_sigma(rng, d_in, scale=0.7)

        closed_form_err = sigma_weighted_error(w_np, sigma_weighted_low_rank(w_np, sigma_np, rank), sigma_np)

        torch.manual_seed(0)
        w = torch.tensor(w_np, dtype=torch.float64)
        sigma = torch.tensor(sigma_np, dtype=torch.float64)
        u = torch.randn(d_out, rank, dtype=torch.float64, requires_grad=True)
        v = torch.randn(d_in, rank, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([u, v], lr=0.05)
        for _ in range(4000):
            opt.zero_grad()
            what = u @ v.T
            diff = w - what
            loss = torch.trace(diff @ sigma @ diff.T)
            loss.backward()
            opt.step()
        gd_err = float(loss.detach())

        print(f"[low_rank] closed-form error: {closed_form_err:.6f}, gradient-descent-refined error: {gd_err:.6f}")
        # the closed form is the TRUE optimum, so gradient descent -- a local, noisy search -- should not
        # beat it by more than a small numerical-optimization slack
        self.assertGreater(gd_err, closed_form_err - 1e-4)


class BlockSparseOptimalityTest(unittest.TestCase):
    """Optimality meaning for the block-sparse solver: for a FIXED support mask, the constraint set is a
    linear subspace and the objective is convex-quadratic, so alternating projection is plain convex
    projected-gradient descent -- it converges to the GLOBAL optimum of that (convex) subproblem, which has
    an independent closed form (per-output-row constrained weighted least squares) checked directly below.
    For the (non-convex, pattern-re-selected-every-step) 2:4 case, only convergence to a LOCAL optimum of
    the alternating scheme is claimed -- checked instead against a naive magnitude-only 2:4 baseline that
    does not adjust surviving weight VALUES for Sigma.
    """

    def test_fixed_mask_matches_closed_form_constrained_least_squares(self):
        rng = np.random.default_rng(3)
        d_out, d_in = 5, 7
        w = rng.normal(size=(d_out, d_in))
        sigma = _random_sigma(rng, d_in)
        mask = rng.random((d_out, d_in)) < 0.6
        # guarantee every row keeps at least one entry so the per-row closed form below is well posed
        for i in range(d_out):
            if not mask[i].any():
                mask[i, 0] = True

        what = sigma_weighted_block_sparse(w, sigma, mask, max_iter=1000, tol=1e-14)
        self.assertTrue(np.all(what[~mask] == 0.0))

        sigma_sym = 0.5 * (sigma + sigma.T)
        what_ref = np.zeros_like(w)
        for i in range(d_out):
            support = mask[i]
            rest = ~support
            if rest.any():
                sigma_ss = sigma_sym[np.ix_(support, support)]
                sigma_s_rest = sigma_sym[np.ix_(support, rest)]
                c = w[i, rest]
                d_support = -np.linalg.solve(sigma_ss, sigma_s_rest @ c)
                what_ref[i, support] = w[i, support] - d_support
            else:
                what_ref[i, support] = w[i, support]

        err_solver = sigma_weighted_error(w, what, sigma)
        err_ref = sigma_weighted_error(w, what_ref, sigma)
        print(f"[block_sparse/fixed_mask] solver error: {err_solver:.8f}, closed-form reference: {err_ref:.8f}")
        self.assertAlmostEqual(err_solver, err_ref, places=5)

    def test_two_four_beats_naive_magnitude_only_baseline(self):
        rng = np.random.default_rng(4)
        d_out, d_in = 4, 12
        w = rng.normal(size=(d_out, d_in))
        sigma = _random_sigma(rng, d_in, scale=1.5)

        what = sigma_weighted_block_sparse(w, sigma, "2:4", max_iter=300)
        # structural constraint respected: at most 2 nonzeros per contiguous group of 4
        groups_nnz = (what.reshape(d_out, d_in // 4, 4) != 0).sum(axis=-1)
        self.assertTrue(np.all(groups_nnz <= 2))

        naive = _project_2_4(w)  # magnitude-only baseline: picks the pattern but never re-optimizes values
        err_solver = sigma_weighted_error(w, what, sigma)
        err_naive = sigma_weighted_error(w, naive, sigma)
        print(f"[block_sparse/2:4] solver error: {err_solver:.6f}, naive-magnitude-only baseline: {err_naive:.6f}")
        self.assertLess(err_solver, err_naive)

        # convergence check: the alternating scheme should be monotonically non-increasing at the end
        what_more = sigma_weighted_block_sparse(w, sigma, "2:4", max_iter=600)
        self.assertLessEqual(
            sigma_weighted_error(w, what_more, sigma) + 1e-9,
            err_solver + 1e-6,
        )


class PermutationOptimalityTest(unittest.TestCase):
    """Optimality meaning for the permutation solver: the row-to-row matching problem is exactly a linear
    assignment problem, so the returned answer (Sinkhorn relaxation converged, then rounded by exact
    Hungarian assignment on the same cost matrix) is checked against true global optimality via brute-force
    enumeration over all n! permutations for small n.
    """

    def test_exact_recovery_of_a_known_permutation(self):
        rng = np.random.default_rng(5)
        n, d = 5, 4
        profile = rng.normal(size=(n, d))
        true_perm = rng.permutation(n)
        w = np.eye(n)[true_perm] @ profile
        sigma = _random_sigma(rng, d)

        what = sigma_weighted_permutation(w, sigma, profile, temperature=0.05, max_iter=200)
        np.testing.assert_allclose(what, w, atol=1e-8)
        self.assertAlmostEqual(sigma_weighted_error(w, what, sigma), 0.0, places=8)

    def test_matches_brute_force_global_optimum_on_a_noisy_case(self):
        rng = np.random.default_rng(6)
        n, d = 5, 4
        profile = rng.normal(size=(n, d))
        true_perm = rng.permutation(n)
        w = np.eye(n)[true_perm] @ profile + rng.normal(scale=0.3, size=(n, d))
        sigma = _random_sigma(rng, d)

        best_err = min(
            sigma_weighted_error(w, np.eye(n)[list(perm)] @ profile, sigma) for perm in itertools.permutations(range(n))
        )
        what = sigma_weighted_permutation(w, sigma, profile, temperature=0.05, max_iter=200)
        solver_err = sigma_weighted_error(w, what, sigma)

        print(f"[permutation] solver error: {solver_err:.6f}, brute-force global optimum: {best_err:.6f}")
        self.assertAlmostEqual(solver_err, best_err, places=6)


def _law(mu, covar) -> GaussianLaw:
    return GaussianLaw(mu=np.asarray(mu, dtype=float), covar=np.asarray(covar, dtype=float))


def _to_numpy(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def _block0_mlp_input_law(model, input_law: GaussianLaw) -> GaussianLaw:
    """Exact input law to ``model.blocks[0].mlp[0]`` -- i.e. the propagated law AFTER block 0's residual
    attention branch and its second LayerNorm -- replicating (by hand, from the same public G1 primitives
    :func:`propagate_moments` itself calls) the first half of one transformer block's forward pass. This
    is the REAL data-free covariance that ``blk.mlp[0].weight`` gets multiplied against.
    """
    blk = model.blocks[0]
    ln1_w, ln1_b = _to_numpy(blk.ln1.weight), _to_numpy(blk.ln1.bias)
    ln1_law, j_ln1 = layernorm_law(input_law, ln1_w, ln1_b, eps=blk.ln1.eps)

    qkv_w = _to_numpy(blk.attn.qkv.weight)
    qkv_b = _to_numpy(blk.attn.qkv.bias)
    proj_w = _to_numpy(blk.attn.proj.weight)
    proj_b = _to_numpy(blk.attn.proj.bias)
    attn_law, j_attn = attention_law(ln1_law, qkv_w, qkv_b, proj_w, proj_b, n_head=blk.attn.h)

    j_branch = j_attn @ j_ln1
    cross = input_law.covar @ j_branch.T
    mu1 = input_law.mu + attn_law.mu
    cov1 = input_law.covar + attn_law.covar + cross + cross.T
    x1 = _law(mu1, cov1)

    ln2_w, ln2_b = _to_numpy(blk.ln2.weight), _to_numpy(blk.ln2.bias)
    ln2_law, _ = layernorm_law(x1, ln2_w, ln2_b, eps=blk.ln2.eps)
    return ln2_law


class DataFreeSigmaBeatsPlainSvdTest(unittest.TestCase):
    """The G2 acceptance criterion: data-free Sigma (from G1's moment propagation) beats plain SVD at equal
    rank on a real small transformer's weight matrix, measured by the ACTUAL objective a data-free Sigma is
    supposed to optimize for -- the Sigma-weighted reconstruction error -- on the real ``blk.mlp[0].weight``
    of ``mixle.models.transformer.build_causal_lm``'s first block.
    """

    def test_sigma_weighted_solver_beats_plain_svd_on_real_mlp_weight(self):
        torch.manual_seed(11)
        model = build_causal_lm(vocab=12, d_model=16, n_layer=2, n_head=4, block=16)

        rng = np.random.default_rng(12)
        d_model = model.d_model
        mu0 = rng.normal(size=d_model) * 0.3
        a0 = rng.normal(size=(d_model, d_model)) * 0.2
        cov0 = a0 @ a0.T + 0.2 * np.eye(d_model)
        input_law = _law(mu0, cov0)

        sigma = _block0_mlp_input_law(model, input_law).covar
        w = _to_numpy(model.blocks[0].mlp[0].weight)  # (4*d_model, d_model)
        rank = 4

        what_weighted = sigma_weighted_low_rank(w, sigma, rank)
        err_weighted = sigma_weighted_error(w, what_weighted, sigma)

        u, s, vt = np.linalg.svd(w, full_matrices=False)
        what_plain = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        err_plain = sigma_weighted_error(w, what_plain, sigma)

        improvement = 1.0 - err_weighted / err_plain
        print(
            f"[acceptance/G2] real blocks[0].mlp[0].weight ({w.shape}), rank={rank}: "
            f"sigma-weighted error={err_weighted:.6f}, plain-SVD error={err_plain:.6f}, "
            f"relative improvement={improvement:.2%}"
        )

        self.assertLess(err_weighted, err_plain)
        # the real propagated Sigma off a genuine LayerNorm is meaningfully anisotropic -- require the
        # data-free solver to actually matter, not just win by float noise
        self.assertGreater(improvement, 0.05)


if __name__ == "__main__":
    unittest.main()
