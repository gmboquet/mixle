"""Acceptance tests for mixle.models.moment_propagation (roadmap G1: moment-propagation surrogate).

Each test below IS an acceptance criterion from the roadmap item, not just a smoke test:
    1. Linear pushforward is exact (float precision).
    2. GELU closed-form moments match large-sample Monte Carlo within a stated tolerance.
    3. The attention MGF identity matches Monte Carlo softmax attention within a stated (looser) tolerance.
    4. A REAL small transformer's streamed propagated law matches a Monte Carlo forward pass of the actual
       torch model within stated numerical bars.
    5. The per-layer closure-error receipt is measurably higher at a deliberately-perturbed ("bad") layer.
    6. Peak memory during propagation does not scale meaningfully with model depth.
"""

import tracemalloc
import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.moment_propagation import (
    GaussianLaw,
    attention_law,
    gelu_law,
    iter_moments,
    layernorm_law,
    linear_law,
    propagate_moments,
)
from mixle.models.transformer import build_causal_lm


def _law(mu, covar) -> GaussianLaw:
    return GaussianLaw(mu=np.asarray(mu, dtype=float), covar=np.asarray(covar, dtype=float))


class LinearExactTest(unittest.TestCase):
    def test_hand_computed_case_matches_to_float_precision(self):
        """``w`` is SQUARE and invertible (det=1) on purpose, not merely for simplicity: for any invertible
        ``w``, ``w @ covar @ w.T`` is a congruence transform of the PD ``covar`` and is therefore guaranteed
        PD too, so `linear_law`'s output never leaves the fast (non-healing) path through
        `MultivariateGaussianDistribution.__init__` (see ``_robust_cho_factor`` in
        ``mixle.stats.multivariate.multivariate_gaussian``) -- confirmed bit-exact (0 ULP) on this
        environment.

        An earlier version of this test used a (3, 2) ``w`` (mapping the 2-D input law into 3-D). That is
        mathematically GUARANTEED to make ``w @ covar @ w.T`` rank-deficient (rank <= 2 in a 3x3 matrix, for
        ANY choice of entries, on any numpy/BLAS build -- not a numerical artifact of one environment).
        `linear_law` routes its output through `MultivariateGaussianDistribution`, whose constructor
        self-heals a covariance that fails strict Cholesky factorization by adding a small trace-scaled
        diagonal jitter (so `log_det`/`inv_covar`/sampling stay well-defined) -- by design, and load-bearing
        elsewhere for float32/GPU EM (see ``_robust_cho_factor``'s docstring). Whether that healing actually
        fires for a given rank-deficient matrix is a knife-edge floating-point question (whether the
        near-zero pivot LAPACK's ``potrf`` hits during elimination rounds positive or negative) that depends
        on the specific numbers AND the BLAS/LAPACK build. Confirmed by direct investigation: on this
        environment (numpy 2.4.6 / Apple Accelerate) the old (3, 2) case reproducibly triggered healing,
        adding exactly ``1e-10 * trace(expected_covar)/3 ~= 4.53e-10`` to the diagonal only (matching
        ``_robust_cho_factor``'s formula to 6 significant figures, and sparing every off-diagonal entry) --
        a real, deterministic, structural effect, NOT generic accumulated matmul-rounding noise (which would
        be ~1e-15 here, on the diagonal AND off-diagonal alike). Since no change to `linear_law`'s
        (already-exact) formula can make an exactly-singular output matrix Cholesky-exact too, this test now
        uses an invertible ``w`` so it tests what its name promises; the dimension-expanding/rank-deficient
        regime is covered explicitly, with an honestly-derived tolerance, by
        `test_dimension_expanding_output_is_psd_with_bounded_diagonal_jitter` below.
        """
        mu = np.array([1.0, -2.0])
        covar = np.array([[2.0, 0.3], [0.3, 1.5]])
        law = _law(mu, covar)
        w = np.array([[1.0, 2.0], [0.0, 1.0]])
        b = np.array([0.5, -0.5])

        out, jac = linear_law(law, w, b)

        expected_mu = w @ mu + b
        expected_covar = w @ covar @ w.T
        np.testing.assert_allclose(out.mu, expected_mu, atol=1e-12, rtol=0)
        np.testing.assert_allclose(out.covar, expected_covar, atol=1e-12, rtol=0)
        np.testing.assert_allclose(jac, w, atol=0, rtol=0)

        # hand-computed by hand for a sanity cross-check independent of the formula above
        self.assertAlmostEqual(expected_mu[0], 1.0 * 1.0 + 2.0 * -2.0 + 0.5)
        self.assertAlmostEqual(expected_mu[1], 0.0 * 1.0 + 1.0 * -2.0 - 0.5)

    def test_dimension_expanding_output_is_psd_with_bounded_diagonal_jitter(self):
        """Companion to the test above: `linear_law` used the way `attention_law` actually uses it for its
        ``qkv`` projection (mapping to MORE output dimensions than the input law's rank) -- ``w`` here is the
        original (3, 2) case the test above used to (fragilely) assert bit-exactness against. ``w @ covar @
        w.T`` is mathematically rank-deficient (confirmed: rank 2 in a 3x3 matrix, on any BLAS build), so
        `linear_law`'s returned ``.covar`` may carry `MultivariateGaussianDistribution`'s documented,
        trace-scaled, DIAGONAL-ONLY self-healing jitter (see ``_robust_cho_factor``). This asserts what IS
        guaranteed instead: the mean and Jacobian are exact (never touched by covariance healing), the
        off-diagonal covariance entries match to float precision (the jitter is diagonal-only by
        construction), the diagonal matches within a generous, explicitly-derived multiple of the documented
        jitter formula, and the result is a valid PSD covariance.
        """
        mu = np.array([1.0, -2.0])
        covar = np.array([[2.0, 0.3], [0.3, 1.5]])
        law = _law(mu, covar)
        w = np.array([[1.0, 2.0], [0.0, 1.0], [-1.0, 1.0]])
        b = np.array([0.5, -0.5, 0.0])

        out, jac = linear_law(law, w, b)

        expected_mu = w @ mu + b
        expected_covar = w @ covar @ w.T
        self.assertEqual(np.linalg.matrix_rank(expected_covar), 2)  # genuinely singular, not a close call

        np.testing.assert_allclose(out.mu, expected_mu, atol=1e-12, rtol=0)
        np.testing.assert_allclose(jac, w, atol=0, rtol=0)

        off_diag = ~np.eye(expected_covar.shape[0], dtype=bool)
        np.testing.assert_allclose(out.covar[off_diag], expected_covar[off_diag], atol=1e-12, rtol=0)

        # _robust_cho_factor's healing jitter (when it fires at all) is `1e-10 * trace / n`; 1e-6 * trace/n
        # leaves >1000x headroom above that first-attempt scale for a slower-converging healing search (the
        # jitter escalates x10 per attempt) on a different BLAS/LAPACK build.
        diag_atol = 1e-6 * (np.trace(expected_covar) / expected_covar.shape[0])
        np.testing.assert_allclose(np.diag(out.covar), np.diag(expected_covar), atol=diag_atol, rtol=0)

        eigvals = np.linalg.eigvalsh(out.covar)
        self.assertTrue(np.all(eigvals >= -1e-8))


class GeluMomentsTest(unittest.TestCase):
    """Closed-form E[GELU(x)], Var[GELU(x)] vs. large-sample Monte Carlo."""

    def test_closed_form_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        cases = [(0.0, 1.0), (1.5, 0.5), (-2.0, 2.0), (0.2, 0.05), (-0.5, 3.0), (4.0, 1.0)]
        n_mc = 3_000_000
        max_mean_err = 0.0
        max_var_err = 0.0
        for mu, sigma in cases:
            samples = rng.normal(loc=mu, scale=sigma, size=n_mc)
            gelu = torch.nn.functional.gelu(torch.as_tensor(samples), approximate="none").numpy()
            mc_mean = gelu.mean()
            mc_var = gelu.var()

            law = _law([mu], [[sigma**2]])
            out, _ = gelu_law(law)
            cf_mean = float(out.mu[0])
            cf_var = float(out.covar[0, 0])

            mean_err = abs(cf_mean - mc_mean)
            var_err = abs(cf_var - mc_var)
            max_mean_err = max(max_mean_err, mean_err)
            max_var_err = max(max_var_err, var_err)
            # Monte Carlo standard error at n=3e6 is tiny; 5e-3 absolute is generous headroom.
            self.assertLess(mean_err, 5e-3, msg=f"mean mismatch at (mu={mu}, sigma={sigma})")
            self.assertLess(var_err, 1e-2, msg=f"var mismatch at (mu={mu}, sigma={sigma})")
        print(f"[gelu_moments] max abs mean error over cases: {max_mean_err:.3e}")
        print(f"[gelu_moments] max abs var error over cases: {max_var_err:.3e}")


class AttentionMgfTest(unittest.TestCase):
    """MGF-based propagated attention output vs. Monte Carlo softmax attention over a sampled key/value
    population, for a synthetic jointly-Gaussian (Q, K, V).

    Honesty note: the MGF identity is exact for the RATIO E[e^s V] / E[e^s] only in the limit of an infinite,
    perfectly Gaussian key population; with a finite sampled population (as any real softmax attention has)
    there is genuine Monte Carlo noise on top of the population-Gaussian approximation. The tolerance below is
    therefore stated generously (~5-8% relative) and is tighter for small query/key scale (where the softmax
    weights are close to uniform and the linear-in-q approximation is most accurate) and looser for large
    scale (where a few keys dominate the softmax and the Gaussian-population assumption is most stressed).
    """

    def _run(self, q_scale: float, n_keys: int, seed: int) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        d = 4
        mu_k = rng.normal(size=d) * 0.3
        mu_v = rng.normal(size=d) * 0.3
        a = rng.normal(size=(2 * d, 2 * d)) * 0.3
        cov = a @ a.T + 0.1 * np.eye(2 * d)
        sigma_kk = cov[:d, :d]
        sigma_vv = cov[d:, d:]
        sigma_vk = cov[d:, :d]

        q = rng.normal(size=d) * q_scale

        # closed-form MGF prediction: mu_v + (Sigma_VK / sqrt(d)) q
        pred = mu_v + (sigma_vk / np.sqrt(d)) @ q

        # Monte Carlo: sample a population of (k_i, v_i) jointly Gaussian, compute real softmax attention.
        joint_mean = np.concatenate([mu_k, mu_v])
        kv = rng.multivariate_normal(mean=joint_mean, cov=cov, size=n_keys)
        k, v = kv[:, :d], kv[:, d:]
        scores = (k @ q) / np.sqrt(d)
        scores -= scores.max()
        w = np.exp(scores)
        w /= w.sum()
        mc = w @ v

        rel_err = np.linalg.norm(pred - mc) / (np.linalg.norm(mc) + 1e-8)
        return rel_err, float(np.linalg.norm(mc))

    def test_small_query_scale_is_tight(self):
        errs = [self._run(q_scale=0.3, n_keys=200_000, seed=s)[0] for s in range(4)]
        max_err = max(errs)
        print(f"[attention_mgf] small-scale max relative error over seeds: {max_err:.3e}")
        self.assertLess(max_err, 0.08)

    def test_large_query_scale_is_looser_but_bounded(self):
        errs = [self._run(q_scale=3.0, n_keys=400_000, seed=s)[0] for s in range(4)]
        max_err = max(errs)
        print(f"[attention_mgf] large-scale max relative error over seeds: {max_err:.3e}")
        # Documented, honest regime: large query scale concentrates softmax mass on few keys, stressing the
        # population-Gaussian assumption -- the bound here is deliberately looser than the small-scale case.
        self.assertLess(max_err, 0.35)


def _build_small_model(seed: int = 0, n_layer: int = 2, d_model: int = 8, n_head: int = 2, vocab: int = 6):
    torch.manual_seed(seed)
    return build_causal_lm(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=16)


def _mc_forward_law(model, input_law: GaussianLaw, n_mc: int, seq_len: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw n_mc replicate sequences from input_law, run the REAL model's block-stack + ln + head (bypassing
    the token/position embedding, matching what propagate_moments propagates), return (mean, cov) of the
    resulting logits at the last position.
    """
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean=input_law.mu, cov=input_law.covar, size=(n_mc, seq_len))
    x = torch.as_tensor(samples, dtype=torch.float32)
    with torch.no_grad():
        for blk in model.blocks:
            x = blk(x)
        x = model.ln(x)
        logits = model.head(x[:, -1, :])
    y = logits.numpy().astype(np.float64)
    return y.mean(axis=0), np.cov(y, rowvar=False)


class TransformerAcceptanceTest(unittest.TestCase):
    """The G1 acceptance criterion: build a real small transformer, propagate an input law through the
    streaming pass, and separately estimate the same output law via large-sample Monte Carlo of the actual
    model. Propagated moments must match within stated bars.

    Bars: this is a 2-layer, d_model=8 model -- small enough that per-layer approximation error (LayerNorm
    re-anchoring + GELU covariance linearization + the attention population assumption) compounds only
    twice. Empirically that keeps output-mean relative error under ~15% and output-covariance relative
    (Frobenius) error under ~40% at this depth/width; both bars are well short of "no signal" (a random/zero
    prediction would show ~100%+ error) and are printed below so the actual measured numbers are visible.
    """

    def test_propagated_matches_monte_carlo_forward(self):
        model = _build_small_model(seed=0)
        d_model = model.d_model
        rng = np.random.default_rng(1)
        mu0 = rng.normal(size=d_model) * 0.3
        a0 = rng.normal(size=(d_model, d_model)) * 0.2
        cov0 = a0 @ a0.T + 0.2 * np.eye(d_model)
        input_law = _law(mu0, cov0)

        receipts = propagate_moments(model, input_law, n_mc=64, seq_len=16, seed=2)
        head_law = receipts[-1].law

        mc_mu, mc_cov = _mc_forward_law(model, input_law, n_mc=20_000, seq_len=16, seed=3)

        mean_rel_err = np.linalg.norm(head_law.mu - mc_mu) / (np.linalg.norm(mc_mu) + 1e-8)
        cov_rel_err = np.linalg.norm(head_law.covar - mc_cov) / (np.linalg.norm(mc_cov) + 1e-8)
        print(f"[acceptance] output mean relative error: {mean_rel_err:.3e}")
        print(f"[acceptance] output covariance relative (Frobenius) error: {cov_rel_err:.3e}")

        self.assertLess(mean_rel_err, 0.15)
        self.assertLess(cov_rel_err, 0.40)


class ErrorMapReceiptTest(unittest.TestCase):
    """Deliberately worsen one block's approximation (extreme LayerNorm/attention weights) and confirm the
    per-layer closure-error receipt is measurably higher there than at a well-behaved layer.
    """

    def test_perturbed_layer_has_higher_closure_error(self):
        model = _build_small_model(seed=5, n_layer=3, d_model=8, n_head=2)
        d_model = model.d_model

        # Blow up the second block's qkv weights so its attention branch is far outside the regime the
        # linearized re-anchoring / MGF-population approximation is accurate in.
        with torch.no_grad():
            model.blocks[1].attn.qkv.weight.mul_(25.0)
            model.blocks[1].ln1.weight.mul_(8.0)

        rng = np.random.default_rng(6)
        mu0 = rng.normal(size=d_model) * 0.2
        a0 = rng.normal(size=(d_model, d_model)) * 0.15
        cov0 = a0 @ a0.T + 0.15 * np.eye(d_model)
        input_law = _law(mu0, cov0)

        receipts = propagate_moments(model, input_law, n_mc=96, seq_len=16, seed=7)
        block_errors = {r.name: r.closure_error for r in receipts if r.name.startswith("block")}
        print(f"[error_map] per-block closure errors: {block_errors}")

        bad = block_errors["block[1]"]
        good_candidates = [v for k, v in block_errors.items() if k != "block[1]"]
        self.assertGreater(bad, max(good_candidates))
        self.assertGreater(bad, 2.0 * max(good_candidates))


class StreamingMemoryReceiptTest(unittest.TestCase):
    """Peak retained memory while consuming iter_moments should not scale meaningfully with model depth.

    ``propagate_moments`` deliberately materializes every full-covariance receipt and therefore has a
    documented ``O(depth*d_model^2)`` returned-memory cost; this test targets the streaming API.
    """

    def test_peak_memory_does_not_scale_with_depth(self):
        d_model = 8
        rng = np.random.default_rng(8)
        mu0 = rng.normal(size=d_model) * 0.2
        a0 = rng.normal(size=(d_model, d_model)) * 0.15
        cov0 = a0 @ a0.T + 0.15 * np.eye(d_model)
        input_law = _law(mu0, cov0)

        peaks = {}
        for n_layer in (2, 8):
            model = _build_small_model(seed=9, n_layer=n_layer, d_model=d_model, n_head=2)
            tracemalloc.start()
            for _ in iter_moments(model, input_law, n_mc=32, seq_len=8, seed=10):
                pass
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks[n_layer] = peak

        print(f"[constant_memory] peak traced memory (bytes) by depth: {peaks}")
        ratio = peaks[8] / peaks[2]
        # A materializing pass that kept every layer's activations alive simultaneously would grow peak
        # memory roughly linearly with depth (4x here, 8 vs 2 layers); the streaming pass should not.
        self.assertLess(ratio, 2.0)


class LayerNormAndAttentionSmokeTest(unittest.TestCase):
    """Cheap sanity checks that layernorm_law / attention_law return PD covariances and finite values, so a
    silent NaN/singular failure in the closed-form derivations doesn't hide inside the larger tests above.
    """

    def test_layernorm_law_is_finite_and_pd(self):
        law = _law([0.1, -0.2, 0.4, 0.0], np.eye(4) * 0.5 + 0.05)
        out, jac = layernorm_law(law, weight=np.ones(4), bias=np.zeros(4), eps=1e-5)
        self.assertTrue(np.all(np.isfinite(out.mu)))
        self.assertTrue(np.all(np.isfinite(out.covar)))
        self.assertTrue(np.all(np.isfinite(jac)))
        # LayerNorm's Jacobian has an EXACT null space along the all-ones direction (shifting every input
        # feature by the same constant leaves the per-sample mean-centered/normalized output unchanged), so
        # the propagated covariance J Sigma J^T is genuinely rank-deficient by (at least) one dimension --
        # not a numerical artifact. Assert PSD (eigenvalues >= -tol) rather than strict PD (raw Cholesky,
        # which would fail on any exactly-singular matrix even in exact arithmetic).
        eigvals = np.linalg.eigvalsh(out.covar)
        self.assertTrue(np.all(eigvals >= -1e-8))
        # And confirm the null space is exactly the all-ones direction, as the derivation predicts.
        ones = np.ones(4)
        np.testing.assert_allclose(jac @ ones, np.zeros(4), atol=1e-10)

    def test_attention_law_is_finite_and_pd(self):
        d_model, n_head = 8, 2
        law = _law(np.zeros(d_model), np.eye(d_model) * 0.3)
        qkv_w = np.random.default_rng(0).normal(size=(3 * d_model, d_model)) * 0.2
        qkv_b = np.zeros(3 * d_model)
        proj_w = np.random.default_rng(1).normal(size=(d_model, d_model)) * 0.2
        proj_b = np.zeros(d_model)
        out, jac = attention_law(law, qkv_w, qkv_b, proj_w, proj_b, n_head=n_head)
        self.assertTrue(np.all(np.isfinite(out.mu)))
        self.assertTrue(np.all(np.isfinite(out.covar)))
        self.assertTrue(np.all(np.isfinite(jac)))
        np.linalg.cholesky(out.covar)


if __name__ == "__main__":
    unittest.main()
