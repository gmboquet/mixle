"""Gaussian(-mixture) law propagation through the real causal transformer in :mod:`mixle.models.transformer`.

This is a *surrogate*: instead of running a forward pass on concrete activations, it pushes a Gaussian LAW
``x ~ N(mu, Sigma)`` (representing the distribution of a token's residual-stream vector under some input
distribution) analytically through each layer type the real ``CausalLM`` is built from -- ``nn.Linear``,
``nn.LayerNorm``, ``nn.GELU``, and :class:`mixle.models.transformer.CausalAttention` -- and returns the
propagated law at every layer together with a genuine, locally-computed "closure error" receipt: how far the
closed-form law is from a small Monte Carlo sample pushed through the REAL torch layer at that point.

Propagated laws are represented with this codebase's own
:class:`mixle.stats.multivariate.multivariate_gaussian.MultivariateGaussianDistribution` (mean + full
covariance), not a parallel numpy-only representation.

What is exact vs. approximate
------------------------------
* **Linear** (``y = Wx + b``): exact. A Gaussian pushed through an affine map is exactly Gaussian.
* **Attention**: the *output-given-query* map is derived to be exactly affine in the query (see
  :func:`attention_law`), via the MGF identity for jointly-Gaussian (K, V). Composed with the exactly-linear
  ``qkv``/``proj`` projections, the whole attention branch is an exact affine function of its LayerNorm'd
  input -- conditional on treating the key/value population as a single stationary Gaussian (see caveats in
  :func:`attention_law`).
* **LayerNorm**: nonlinear and data-dependent; propagated via a first-order Taylor ("re-anchoring") expansion
  of the true LayerNorm map around the input mean (see :func:`layernorm_law` for the closed-form Jacobian and
  documented failure modes).
* **GELU**: the first two output MOMENTS (mean and per-dimension variance) are exact closed-form expressions
  in ``(mu, sigma)`` (derived via Stein's lemma + the bivariate normal CDF, see :func:`gelu_law`); the
  OFF-diagonal output covariance starts from a first-order (Jacobian) linearization -- the same "delta
  method" used for LayerNorm's covariance push-forward -- then its approximate correlations are projected
  to the PSD cone under fixed exact marginal variances, with an explicit correction receipt.
* **Residual connections** (``x + branch(x)``): the two summands are correlated (both are functions of the
  same ``x``), which the propagation accounts for through a chained JACOBIAN of the branch mapping (composed
  from the exact/linearized per-layer Jacobians above), giving ``Cov(x, branch(x)) ~= Sigma_x @ J_branch^T``
  and hence an (approximately) correct ``Sigma_out = Sigma_x + Sigma_branch + Cov + Cov^T``.

Execution and retained-memory contract
--------------------------------------
:func:`iter_moments` mirrors the walking pattern of
:func:`mixle.inference.precision_plan.recommend_compute_precision`: it inspects the model's structure and
processes it piece by piece and yields each receipt before advancing. Its propagation working set is the
current full law plus one block's intermediates, while closure probes are processed in bounded batches and
reduced online rather than materialized as ``n_mc × seq_len × d_model`` tensors. Excluding the model itself
and any receipts retained by the caller, iterator memory is independent of depth and bounded by
``O(d_model² + probe_batch_size × seq_len × d_model)`` (with analogous output-width terms at the head).

:func:`propagate_moments` is the convenience materializer: it consumes that iterator into a list and
therefore intentionally retains one full mean/covariance law per layer. Its returned-memory cost is
``O(depth × d_model²)`` plus the final head law. Use :func:`iter_moments` when depth-independent retained
memory is required.

References
----------
* Stein's lemma (Gaussian integration by parts): ``E[(X-mu) g(X)] = sigma^2 E[g'(X)]`` for ``X ~ N(mu,
  sigma^2)`` -- used throughout to get closed forms for ``E[GELU(X)]`` and its derivative.
* The Gaussian-product / MGF identity ``E[Y e^{t^T X}] = M_X(t) (mu_Y + Sigma_YX t)`` for jointly Gaussian
  ``(X, Y)`` -- the exact identity behind :func:`attention_law`.
* Data-free quantization (DFQ) BatchNorm-based calibration is the closest prior art for the LayerNorm
  "re-anchoring" step: both re-derive a cheap closed-form summary of what a normalization layer does to a
  law, without touching real data.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from scipy.stats import multivariate_normal as _mvn
from scipy.stats import norm as _norm

from mixle.models.grad_leaf import _module_mode
from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

GaussianLaw = MultivariateGaussianDistribution


# --------------------------------------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------------------------------------


def _as_law(mu: np.ndarray, covar: np.ndarray) -> GaussianLaw:
    """Build a :class:`MultivariateGaussianDistribution` from a mean/covariance pair, symmetrizing first.

    ``MultivariateGaussianDistribution`` already self-heals a covariance that lost positive-definiteness to
    float round-off (see ``_robust_cho_factor``); symmetrizing here just keeps that jitter search cheap.
    """
    covar = np.asarray(covar, dtype=np.float64)
    covar = 0.5 * (covar + covar.T)
    return MultivariateGaussianDistribution(mu=np.asarray(mu, dtype=np.float64), covar=covar)


def _to_numpy(t: Any) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def _module_weight_bias(linear: Any) -> tuple[np.ndarray, np.ndarray | None]:
    w = _to_numpy(linear.weight)
    b = _to_numpy(linear.bias) if linear.bias is not None else None
    return w, b


@dataclass
class LayerMoments:
    """One layer's propagated law plus its locally-computed closure-error receipt."""

    index: int
    name: str
    law: GaussianLaw
    closure_error: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeluCovarianceReceipt:
    """Measured correction applied to keep linearized GELU correlations PSD at exact marginal variances."""

    correction_frobenius: float
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    iterations: int
    marginal_variance_max_error: float


def _exact_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


# --------------------------------------------------------------------------------------------------------
# 1. Linear (exact)
# --------------------------------------------------------------------------------------------------------


def linear_law(law: GaussianLaw, weight: np.ndarray, bias: np.ndarray | None = None) -> tuple[GaussianLaw, np.ndarray]:
    """Exact Gaussian push-forward of ``y = Wx + b``.

    ``x ~ N(mu, Sigma) => y ~ N(W mu + b, W Sigma W^T)`` exactly -- no approximation. Returns the new law
    together with the Jacobian ``dy/dx = W`` (used to chain residual cross-covariances).

    Note on ``out.covar`` when ``weight`` maps to MORE output dimensions than ``law.covar``'s rank (e.g. an
    expanding projection like the ``qkv`` weight in :func:`attention_law`): ``W Sigma W^T`` is then
    mathematically rank-deficient (PSD, not PD), and the returned law -- a
    :class:`MultivariateGaussianDistribution`, which self-heals a non-PD covariance with a small trace-scaled
    diagonal jitter so it stays Cholesky-factorizable for ``log_det``/``inv_covar``/sampling, see
    ``_robust_cho_factor`` in :mod:`mixle.stats.multivariate.multivariate_gaussian` -- may therefore differ
    from the exact ``W Sigma W^T`` by that jitter (diagonal-only, ``~1e-10 * trace/n``). This is NOT an error
    in the formula above -- verified bit-exact whenever the output is genuinely PD (e.g. square invertible
    ``weight``); see ``mixle/tests/moment_propagation_test.py::LinearExactTest`` for both regimes -- it is the
    unavoidable cost of representing an exactly-singular covariance with a factorization that requires one.
    """
    w = np.asarray(weight, dtype=np.float64)
    b = np.zeros(w.shape[0]) if bias is None else np.asarray(bias, dtype=np.float64)
    mu = w @ law.mu + b
    covar = w @ law.covar @ w.T
    return _as_law(mu=mu, covar=covar), w


# --------------------------------------------------------------------------------------------------------
# 2. LayerNorm (re-anchoring)
# --------------------------------------------------------------------------------------------------------


def layernorm_law(
    law: GaussianLaw, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> tuple[GaussianLaw, np.ndarray]:
    """Propagate a Gaussian law through ``LayerNorm``: ``y = weight * (x - m(x)) / sqrt(v(x) + eps) + bias``,
    where ``m(x) = mean_d(x)`` and ``v(x) = mean_d((x - m(x))^2)`` are the PER-SAMPLE (per-token) statistics
    LayerNorm computes over the feature axis -- this is the nonlinear, data-dependent step in the block.

    Derivation ("re-anchoring")
    ----------------------------
    LayerNorm has no closed-form pushforward of a full law in general (``m(x)`` and ``v(x)`` are themselves
    random, nonlinear functions of ``x``). We use a first-order Taylor expansion around the CURRENT mean
    ``mu``, i.e. anchor the (unknown, per-sample) normalization statistics at their EXPECTED values under the
    current law, not merely their values evaluated at the mean vector. ``m(x) = mean_d(x)`` is itself LINEAR in
    ``x``, so ``E[m(x)] = mean_d(mu)`` exactly -- no correction needed there. But ``v(x) = mean_d((x -
    m(x))^2)`` is QUADRATIC in ``x``, so evaluating it at ``mu`` alone (``v(mu) = mean((mu - m)^2)``) drops a
    systematic bias term: writing ``P = I - (1/d) 11^T`` for the feature-centering projector (symmetric,
    idempotent), ``v(x) = (1/d) x^T P x``, and the standard quadratic-form expectation identity gives

        ``E[v(x)] = v(mu) + (1/d) trace(P @ Sigma) = v(mu) + (1/d) (trace(Sigma) - (1/d) sum(Sigma))``.

    The second term is the "spread of x around its own per-token mean" contribution that ``v(mu)`` alone
    misses entirely; it is NOT a higher-order correction that can be dropped once ``Sigma`` is non-negligible
    relative to ``d`` -- for small ``d_model`` (e.g. an 8-wide toy model) it routinely dominates ``v(mu)``,
    which without this correction makes the anchored ``std = sqrt(v(mu) + eps)`` far too small and blows the
    propagated mean/covariance up by an order of magnitude relative to the true LayerNorm output law. We
    therefore re-anchor at the BIAS-CORRECTED ``v = E[v(x)]`` above (still a first-order/delta-method treatment
    of the covariance push-forward -- only the anchor point for ``v`` is corrected, not the linearization
    itself). This is the LLM analogue of BatchNorm-based data-free-quantization (DFQ) calibration, which
    likewise re-derives cheap closed-form layer statistics (there, running mean/var; here, the expected
    LayerNorm mean/var under the CURRENT propagated law) without touching real data.

    The mean is propagated by evaluating the true (nonlinear) LayerNorm map at ``mu`` but with the
    bias-corrected ``v``:

        ``mu_out = weight * (mu - m) / sqrt(v + eps) + bias``.

    The covariance is propagated via the JACOBIAN of LayerNorm evaluated at ``mu`` with the same
    bias-corrected ``v`` (a standard, textbook LayerNorm-backward-style derivative applied at the corrected
    anchor):

        ``d(norm_i)/d(x_j) |_{x=mu} = (1/sqrt(v+eps)) * (delta_ij - 1/d - (mu_i - m)(mu_j - m) / (d*(v+eps)))``
        ``J_ij = weight_i * d(norm_i)/d(x_j)``
        ``Sigma_out ~= J Sigma J^T``.

    Known failure modes (feeds the closure-error receipt)
    -------------------------------------------------------
    * **Small ``d_model``**: ``m(x)`` and ``v(x)`` are averages over only ``d_model`` features, so their
      sample-to-sample fluctuation around their (now bias-corrected) expectation -- which this delta-method
      approximation still ignores, since the Jacobian itself is frozen at the anchor -- is large relative to
      their magnitude when ``d_model`` is small. The approximation is progressively worse as ``d_model``
      shrinks, even after the mean-bias correction above.
    * **Heavy-tailed pre-norm activations**: the first-order Taylor expansion is only locally valid; if the
      input law has high kurtosis / is far from Gaussian in practice (despite being MODELED as Gaussian here),
      the true per-sample ``(m, v)`` can swing far from their Gaussian-law expectations, and the linearization
      degrades.
    * A deep stack of blocks compounds both effects: even a small per-layer LayerNorm error can accumulate
      across ``n_layer`` re-anchoring steps.
    """
    x = np.asarray(law.mu, dtype=np.float64)
    d = x.shape[0]
    m = float(x.mean())
    centered = x - m
    v_at_mean = float(np.mean(centered * centered))
    covar = np.asarray(law.covar, dtype=np.float64)
    # Bias correction: E[v(x)] = v(mu) + (1/d) * trace(P @ Sigma), P = I - (1/d) 11^T the centering
    # projector; trace(P @ Sigma) = trace(Sigma) - (1/d) * sum(Sigma) avoids materializing P explicitly.
    trace_sigma = float(np.trace(covar))
    sum_sigma = float(covar.sum())
    v_correction = (trace_sigma - sum_sigma / d) / d
    v = v_at_mean + v_correction
    std = np.sqrt(v + eps)

    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    mu_out = weight * (centered / std) + bias

    eye = np.eye(d)
    outer = np.outer(centered, centered)
    j0 = (eye - 1.0 / d - outer / (d * (v + eps))) / std
    jac = weight[:, None] * j0

    covar_out = jac @ law.covar @ jac.T
    return _as_law(mu=mu_out, covar=covar_out), jac


# --------------------------------------------------------------------------------------------------------
# 3. GELU (closed-form moments)
# --------------------------------------------------------------------------------------------------------


def _gelu_scalar_moments(mu: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-dimension closed-form ``E[GELU(x)]``, ``Var[GELU(x)]``, and ``d E[GELU(x)] / d mu`` for
    ``x_i ~ N(mu_i, var_i)`` independently, where ``GELU(x) = x * Phi(x)`` (the exact erf-based GELU that
    ``torch.nn.GELU()`` uses by default -- NOT the tanh approximation).

    Derivation via Stein's lemma
    -----------------------------
    For ``X ~ N(mu, sigma^2)``, Stein's lemma is ``E[(X - mu) g(X)] = sigma^2 E[g'(X)]``, i.e.
    ``E[X g(X)] = mu E[g(X)] + sigma^2 E[g'(X)]``. Writing ``s = sqrt(1 + sigma^2)``, ``P = Phi(mu / s)``,
    ``p = phi(mu / s)`` (``phi`` the standard normal PDF):

    * ``E[Phi(X)] = P``  (since ``Phi(X) = P(Z <= X)`` for independent standard normal ``Z``, and
      ``Z - X ~ N(-mu, s^2)``).
    * ``E[phi(X)] = p / s``  (density of a sum of independent Gaussians: ``phi`` convolved with ``N(mu,
      sigma^2)`` evaluated at 0 is ``N(mu; 0, s^2)``).
    * Applying Stein's lemma with ``g(X) = Phi(X)``, ``g'(X) = phi(X)``:
        ``E[X Phi(X)] = mu*P + (sigma^2/s)*p``  ->  this IS ``E[GELU(X)]``.
    * Applying Stein's lemma again with ``g(X) = phi(X)``, ``g'(X) = -X phi(X)``:
        ``E[X phi(X)] = mu * p / s^3``.
    * For the second moment ``E[X^2 Phi(X)^2] = E[GELU(X)^2]``, write ``Phi(X)^2 = P(Z1 <= X, Z2 <= X)``
      for independent standard normal ``Z1, Z2``: this is a bivariate normal CDF,
        ``E[Phi(X)^2] = Phi_2(mu/s, mu/s; rho = sigma^2/s^2)``
      (``Phi_2`` the standard bivariate normal CDF -- the "Owen's-T-style" closed form the derivation calls
      for; computed here via ``scipy.stats.multivariate_normal``). The cross term
      ``E[Phi(X) phi(X)]`` follows from writing ``phi(x) N(x; mu, sigma^2)`` as (up to a constant) another
      Gaussian density in ``x`` (product-of-Gaussians identity) and reducing to a univariate normal CDF:
        ``E[Phi(X) phi(X)] = (p / s) * Phi(mu / (s * sqrt(1 + 2*sigma^2)))``.
      Two applications of Stein's lemma to ``g(X) = X f(X)`` and then again to ``g(X) = f'(X)`` give the full
      (three-term, not two-term) identity
        ``E[X^2 f(X)] = (mu^2+sigma^2)*E[f] + 2*mu*sigma^2*E[f'] + sigma^4*E[f'']``
      for ``f(X) = Phi(X)^2``, ``f'(X) = 2 Phi(X) phi(X)``, ``f''(X) = 2 phi(X)^2 - 2 X Phi(X) phi(X)``. The
      extra ``E[phi(X)^2]`` and ``E[X Phi(X) phi(X)]`` terms this pulls in both reduce to closed form the same
      way: ``phi(x)^2`` is (up to a constant) another Gaussian density in ``x``, giving
        ``E[phi(X)^2] = exp(-mu^2/r^2) / (2*pi*r)``,  ``r = sqrt(1 + 2*sigma^2)``,
      and one more Stein's-lemma application (``E[X h(X)] = mu E[h] + sigma^2 E[h']`` with ``h = Phi*phi``,
      solved for the self-referential ``E[X h(X)]`` term that appears via ``h' = phi^2 - X Phi phi``) gives
        ``E[X Phi(X) phi(X)] = (mu*E[Phi(X)phi(X)] + sigma^2*E[phi(X)^2]) / (1 + sigma^2)``.
      (An earlier draft of this derivation dropped the ``sigma^4*E[f'']`` term entirely -- i.e. used only the
      two-term identity, which is exact for LINEAR ``f`` but not for the genuinely curved ``f = Phi^2`` here;
      that under-counted the variance, sometimes by an order of magnitude. The three-term identity above is
      cross-checked against numerical quadrature to <1e-9 absolute error for every case in the docstring and
      in ``mixle/tests/moment_propagation_test.py``.)
      Then ``Var[GELU(X)] = E[X^2 Phi(X)^2] - E[GELU(X)]^2``.
    * The exact derivative of the mean is, again by dominated convergence / Stein's lemma with
      ``GELU'(X) = Phi(X) + X phi(X)``:
        ``d E[GELU(X)] / d mu = E[GELU'(X)] = P + mu * p / s^3``.
      This is used as the (exact, for the diagonal; linearized off-diagonal) Jacobian for chaining
      covariances through GELU.

    Validated against large-sample Monte Carlo in ``mixle/tests/moment_propagation_test.py``.
    """
    mu = np.asarray(mu, dtype=np.float64)
    var = np.maximum(np.asarray(var, dtype=np.float64), 1e-12)
    sigma = np.sqrt(var)
    s = np.sqrt(1.0 + var)
    r = np.sqrt(1.0 + 2.0 * var)

    z = mu / s
    p_cdf = _norm.cdf(z)
    p_pdf = _norm.pdf(z)

    mean = mu * p_cdf + (var / s) * p_pdf

    rho = var / (s * s)
    ephi2 = np.empty_like(mu)
    for i in range(mu.shape[0]):
        a = float(z[i])
        rr = float(rho[i])
        ephi2[i] = _mvn(mean=[0.0, 0.0], cov=[[1.0, rr], [rr, 1.0]]).cdf([a, a])

    q = mu / (s * r)
    ephiphi = (p_pdf / s) * _norm.cdf(q)

    # E[phi(X)^2] and E[X Phi(X) phi(X)] -- the two extra terms the three-term Stein identity below needs
    # (see the module-level derivation note): phi(x)^2 is itself proportional to a Gaussian density in x.
    ephisq = np.exp(-(mu * mu) / (r * r)) / (2.0 * np.pi * r)
    exphiphi = (mu * ephiphi + var * ephisq) / (1.0 + var)
    e_fpp = 2.0 * ephisq - 2.0 * exphiphi  # E[f''(X)], f(X) = Phi(X)^2

    # Full three-term identity E[X^2 f(X)] = (mu^2+var)*E[f] + 2*mu*var*E[f'] + var^2*E[f'']
    # with E[f'] = 2*E[Phi(X)phi(X)] = 2*ephiphi, so the middle term is 4*mu*var*ephiphi.
    e_x2phi2 = (mu * mu + var) * ephi2 + 4.0 * mu * var * ephiphi + var * var * e_fpp
    variance = np.maximum(e_x2phi2 - mean * mean, 1e-12)

    jac_diag = p_cdf + mu * p_pdf / (s**3)

    _ = sigma  # kept for readability of the derivation above; sigma itself isn't needed past `var`
    return mean, variance, jac_diag


def _project_correlation_fixed_diagonal(
    covariance: np.ndarray,
    variance: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, GeluCovarianceReceipt]:
    """Project a covariance to the PSD cone through correlation space while preserving its diagonal."""
    std = np.sqrt(variance)
    correlation = covariance / np.outer(std, std)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    min_before = float(np.linalg.eigvalsh(correlation)[0])

    y = correlation.copy()
    delta = np.zeros_like(y)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        residual = y - delta
        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        psd = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        delta = psd - residual
        previous = y
        y = psd.copy()
        np.fill_diagonal(y, 1.0)
        if np.linalg.norm(y - previous, ord="fro") <= tol * max(1.0, np.linalg.norm(y, ord="fro")):
            break

    # A final positive-definite projection followed by diagonal renormalization is a congruence transform:
    # it preserves PSD while restoring unit diagonal. A tiny floor avoids downstream Cholesky "healing"
    # that would otherwise alter the exact scalar variances.
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (y + y.T))
    positive = (eigenvectors * np.maximum(eigenvalues, 1.0e-12)) @ eigenvectors.T
    diagonal_scale = np.sqrt(np.diag(positive))
    corrected_correlation = positive / np.outer(diagonal_scale, diagonal_scale)
    corrected_correlation = 0.5 * (corrected_correlation + corrected_correlation.T)
    np.fill_diagonal(corrected_correlation, 1.0)

    corrected = corrected_correlation * np.outer(std, std)
    np.fill_diagonal(corrected, variance)
    min_after = float(np.linalg.eigvalsh(corrected_correlation)[0])
    receipt = GeluCovarianceReceipt(
        correction_frobenius=float(np.linalg.norm(corrected - covariance, ord="fro")),
        min_eigenvalue_before=min_before,
        min_eigenvalue_after=min_after,
        iterations=iterations,
        marginal_variance_max_error=float(np.max(np.abs(np.diag(corrected) - variance))),
    )
    return corrected, receipt


def gelu_law(
    law: GaussianLaw,
    *,
    return_receipt: bool = False,
) -> tuple[GaussianLaw, np.ndarray] | tuple[GaussianLaw, np.ndarray, GeluCovarianceReceipt]:
    """Propagate a Gaussian law through elementwise ``GELU`` using the closed-form per-dimension moments in
    :func:`_gelu_scalar_moments`.

    The per-dimension MEAN and VARIANCE are exact closed-form expressions (no Monte Carlo, no approximation
    of the GELU functional form). The OFF-diagonal output covariance -- ``Cov(GELU(x_i), GELU(x_j))`` for
    ``i != j`` -- has no simple closed form (it needs the joint bivariate distribution of ``(x_i, x_j)``
    through a nonlinearity) and is instead approximated by the standard delta-method / Price's-theorem-style
    linearization ``Cov(y_i, y_j) ~= J_ii * J_jj * Cov(x_i, x_j)`` with ``J_ii = d E[GELU(x_i)]/d mu_i`` the
    EXACT mean-derivative from Stein's lemma. The diagonal is then overwritten with the exact closed-form
    variance. Because that splice need not be PSD, correlations are projected to the PSD cone under a
    fixed-unit-diagonal constraint and then rescaled by the exact marginal standard deviations. This
    preserves the scalar variances while correcting only the approximate correlation structure.

    Set ``return_receipt=True`` to receive a :class:`GeluCovarianceReceipt` measuring the correction,
    pre/post minimum correlation eigenvalues, iteration count, and marginal-variance preservation error.
    """
    mu = np.asarray(law.mu, dtype=np.float64)
    var = np.diag(law.covar).copy()
    mean, variance, jac_diag = _gelu_scalar_moments(mu, var)

    jac = np.diag(jac_diag)
    candidate = jac @ law.covar @ jac.T
    np.fill_diagonal(candidate, variance)
    covar, receipt = _project_correlation_fixed_diagonal(candidate, variance)
    result = _as_law(mu=mean, covar=covar)
    if return_receipt:
        return result, jac, receipt
    return result, jac


# --------------------------------------------------------------------------------------------------------
# 4. Attention (R2 / MGF identity)
# --------------------------------------------------------------------------------------------------------


def attention_law(
    law: GaussianLaw,
    qkv_weight: np.ndarray,
    qkv_bias: np.ndarray | None,
    proj_weight: np.ndarray,
    proj_bias: np.ndarray | None,
    n_head: int,
) -> tuple[GaussianLaw, np.ndarray]:
    """Propagate a Gaussian law through one :class:`mixle.models.transformer.CausalAttention` layer.

    Modeling assumption: the key/value population attended over (across sequence positions) is treated as a
    single STATIONARY Gaussian -- the same joint law as the query's -- rather than tracking per-position
    laws. This is the "R2 MGF" population-level approximation the roadmap specifies; it does not model the
    causal mask explicitly (a real causal mask makes early positions attend to a smaller, non-stationary
    population). That mismatch is a known limitation, checked directly against Monte Carlo softmax attention
    in ``mixle/tests/moment_propagation_test.py`` (tightest for a roughly-stationary population, looser for
    strongly non-stationary / short sequences).

    Derivation
    ----------
    For jointly Gaussian ``(K, V)`` (a key vector and its associated value vector from the SAME token) and a
    fixed query ``q``, the requested MGF identity is

        ``E[exp(q^T K / sqrt(d)) V] = exp(q^T mu_K/sqrt(d) + 0.5 q^T Sigma_KK q / d) * (mu_V + Sigma_VK q / sqrt(d))``

    which is the standard "``E[Y e^{t^T X}] = M_X(t) (mu_Y + Sigma_YX t)``" identity for jointly Gaussian
    ``(X, Y)`` with ``t = q / sqrt(d)``, ``X = K``, ``Y = V``. The attention DENOMINATOR (the softmax
    normalizer) is exactly the same MGF evaluated with ``V`` replaced by the constant ``1``:

        ``E[exp(q^T K / sqrt(d))] = exp(q^T mu_K/sqrt(d) + 0.5 q^T Sigma_KK q / d)``.

    Both share the identical exponential prefactor, so it CANCELS in the ratio:

        ``softmax-attention-output(q) ~= E[exp(q^T K/sqrt(d)) V] / E[exp(q^T K/sqrt(d))] = mu_V + (Sigma_VK / sqrt(d)) q``.

    This is a remarkably clean result: the MGF-approximated attention output, as a function of the query, is
    EXACTLY AFFINE in ``q`` (no exponential term survives). Since ``q`` itself has a propagated Gaussian law
    (the marginal of the joint (Q, K, V) law after the ``qkv`` projection), pushing that law through this
    affine map is exact (reusing :func:`linear_law`'s formula) -- there is no additional approximation beyond
    the population-stationarity assumption above and (for the covariance) the affine relation being evaluated
    with the ``Sigma_VK`` estimated from the SAME joint law used for the mean.

    Per head, per token position, ``y_h(q) = mu_{V,h} + (Sigma_{VK,h} / sqrt(d_head)) q_h``. Stacking heads
    into a block-diagonal map ``A`` (each head only reads its own query slice) and reusing the FULL
    (cross-head) query covariance from the ``qkv`` projection gives the cross-head covariance of the output
    "for free" -- cross-head correlation in ``Q`` (already present in ``Sigma_QQ``'s off-diagonal blocks)
    propagates through ``A Sigma_QQ A^T`` even though ``A`` itself only mixes within a head.
    """
    d_model = law.mu.shape[0]
    if d_model % n_head != 0:
        raise ValueError("d_model must be divisible by n_head")
    d_head = d_model // n_head

    qkv_law, _ = linear_law(law, qkv_weight, qkv_bias)
    cov = qkv_law.covar
    mu_q = qkv_law.mu[0:d_model]

    scale = 1.0 / np.sqrt(d_head)
    a_full = np.zeros((d_model, d_model), dtype=np.float64)
    b_full = np.zeros(d_model, dtype=np.float64)
    for h in range(n_head):
        sl = slice(h * d_head, (h + 1) * d_head)
        q_abs = slice(0 * d_model + h * d_head, 0 * d_model + (h + 1) * d_head)
        k_abs = slice(1 * d_model + h * d_head, 1 * d_model + (h + 1) * d_head)
        v_abs = slice(2 * d_model + h * d_head, 2 * d_model + (h + 1) * d_head)
        sigma_vk_h = cov[v_abs, k_abs]
        a_full[sl, sl] = sigma_vk_h * scale
        b_full[sl] = qkv_law.mu[v_abs]
        del q_abs  # only used for documentation of the index layout

    sigma_qq = cov[0:d_model, 0:d_model]
    y_mu = b_full + a_full @ mu_q
    y_cov = a_full @ sigma_qq @ a_full.T
    y_law = _as_law(mu=y_mu, covar=y_cov)

    out_law, j_proj = linear_law(y_law, proj_weight, proj_bias)

    w_qkv = np.asarray(qkv_weight, dtype=np.float64)
    w_q = w_qkv[0:d_model, :]
    jac_wrt_x = j_proj @ a_full @ w_q
    return out_law, jac_wrt_x


# --------------------------------------------------------------------------------------------------------
# 5. Per-layer closure-error receipts
# --------------------------------------------------------------------------------------------------------


def _law_discrepancy(law: GaussianLaw, mc_mu: np.ndarray, mc_cov: np.ndarray) -> float:
    """Compare a propagated law with online Monte Carlo mean/covariance summaries."""
    mu_err = np.linalg.norm(law.mu - mc_mu) / (np.linalg.norm(law.mu) + 1e-8)
    cov_err = np.linalg.norm(law.covar - mc_cov) / (np.linalg.norm(law.covar) + 1e-8)
    result = float(max(mu_err, cov_err))
    if not np.isfinite(result):
        raise ValueError("moment closure discrepancy became non-finite")
    return result


class _OnlineMoments:
    """Bounded-memory sufficient statistics for an unbiased empirical covariance."""

    def __init__(self, dim: int) -> None:
        self.count = 0
        self.total = np.zeros(dim, dtype=np.float64)
        self.cross = np.zeros((dim, dim), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.total.shape[0] or not np.all(np.isfinite(array)):
            raise ValueError("closure probe outputs must be a finite two-dimensional matrix")
        self.count += array.shape[0]
        self.total += array.sum(axis=0)
        self.cross += array.T @ array

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("closure probes require at least two samples")
        mean = self.total / self.count
        covariance = (self.cross - self.count * np.outer(mean, mean)) / (self.count - 1)
        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("closure probe moments became non-finite")
        return mean, covariance


def _module_device_dtype(module: Any) -> tuple[Any, Any]:
    tensors = list(module.parameters()) + list(module.buffers())
    if not tensors:
        return torch.device("cpu"), torch.float32
    reference = tensors[0]
    dtype = reference.dtype if reference.dtype.is_floating_point else torch.float32
    return reference.device, dtype


def _closure_error_pointwise(
    input_law: GaussianLaw,
    module: Any,
    propagated_law: GaussianLaw,
    rng: np.random.Generator,
    n_mc: int,
    probe_batch_size: int,
) -> float:
    """Closure-error receipt for a token-independent (pointwise) layer (LayerNorm / Linear head): draw
    ``n_mc`` iid samples from ``input_law`` and run them through the REAL torch module.
    """
    device, dtype = _module_device_dtype(module)
    moments = _OnlineMoments(propagated_law.mu.shape[0])
    with _module_mode(module, train=False), torch.no_grad():
        for start in range(0, n_mc, probe_batch_size):
            count = min(probe_batch_size, n_mc - start)
            samples = rng.multivariate_normal(mean=input_law.mu, cov=input_law.covar, size=count)
            x = torch.as_tensor(samples, dtype=dtype, device=device)
            y = module(x)
            if not isinstance(y, torch.Tensor) or tuple(y.shape) != (count, propagated_law.mu.shape[0]):
                raise ValueError(
                    "pointwise closure module output must have shape "
                    f"({count}, {propagated_law.mu.shape[0]}); got {getattr(y, 'shape', None)}"
                )
            moments.update(y.detach().to(device="cpu", dtype=torch.float64).numpy())
    return _law_discrepancy(propagated_law, *moments.finalize())


def _closure_error_block(
    input_law: GaussianLaw,
    blk: Any,
    propagated_law: GaussianLaw,
    rng: np.random.Generator,
    n_mc: int,
    seq_len: int = 16,
    probe_batch_size: int = 32,
) -> float:
    """Closure-error receipt for a full transformer :class:`~mixle.models.transformer.Block`.

    Draws ``n_mc`` REPLICATE sequences, each of ``seq_len`` iid tokens sampled from ``input_law`` (modeling
    the stationary population assumption used by :func:`attention_law`), runs each replicate through the REAL
    ``blk`` (with real causal masking), and takes the LAST position's output -- the position that attends
    causally over the full local population -- as one Monte Carlo draw of the block's output law. Comparing
    the resulting empirical (mean, covariance) across the ``n_mc`` replicates against the closed-form
    propagated law is a real, per-layer, locally-computed signal, not a placeholder constant.
    """
    device, dtype = _module_device_dtype(blk)
    moments = _OnlineMoments(propagated_law.mu.shape[0])
    with _module_mode(blk, train=False), torch.no_grad():
        for start in range(0, n_mc, probe_batch_size):
            count = min(probe_batch_size, n_mc - start)
            samples = rng.multivariate_normal(
                mean=input_law.mu,
                cov=input_law.covar,
                size=(count, seq_len),
            )
            x = torch.as_tensor(samples, dtype=dtype, device=device)
            output = blk(x)
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != (
                count,
                seq_len,
                propagated_law.mu.shape[0],
            ):
                raise ValueError(
                    "block closure output must have shape "
                    f"({count}, {seq_len}, {propagated_law.mu.shape[0]}); "
                    f"got {getattr(output, 'shape', None)}"
                )
            moments.update(output[:, -1, :].detach().to(device="cpu", dtype=torch.float64).numpy())
    return _law_discrepancy(propagated_law, *moments.finalize())


# --------------------------------------------------------------------------------------------------------
# 6. Execution contract: streaming iterator plus explicit list materializer
# --------------------------------------------------------------------------------------------------------


def _propagate_block(
    law: GaussianLaw,
    blk: Any,
    rng: np.random.Generator,
    n_mc: int,
    seq_len: int,
    probe_batch_size: int,
) -> tuple[GaussianLaw, float, dict[str, Any]]:
    """Propagate one transformer :class:`Block` (pre-norm attention + pre-norm MLP, both residual) and return
    ``(output_law, closure_error)``. Holds only ``law`` (the running state), ``blk`` (this block's weights),
    and this block's own intermediate laws/Jacobians -- nothing from earlier or later blocks.
    """
    ln1_w, ln1_b = _to_numpy(blk.ln1.weight), _to_numpy(blk.ln1.bias)
    ln1_law, j_ln1 = layernorm_law(law, ln1_w, ln1_b, eps=blk.ln1.eps)

    qkv_w, qkv_b = _module_weight_bias(blk.attn.qkv)
    proj_w, proj_b = _module_weight_bias(blk.attn.proj)
    attn_law, j_attn = attention_law(ln1_law, qkv_w, qkv_b, proj_w, proj_b, n_head=blk.attn.h)

    j_attn_branch = j_attn @ j_ln1
    cross1 = law.covar @ j_attn_branch.T
    mu1 = law.mu + attn_law.mu
    cov1 = law.covar + attn_law.covar + cross1 + cross1.T
    x1 = _as_law(mu=mu1, covar=cov1)

    ln2_w, ln2_b = _to_numpy(blk.ln2.weight), _to_numpy(blk.ln2.bias)
    ln2_law, j_ln2 = layernorm_law(x1, ln2_w, ln2_b, eps=blk.ln2.eps)

    lin1_w, lin1_b = _module_weight_bias(blk.mlp[0])
    lin1_law, j_lin1 = linear_law(ln2_law, lin1_w, lin1_b)
    gelu_out_law, j_gelu, gelu_receipt = gelu_law(lin1_law, return_receipt=True)
    lin2_w, lin2_b = _module_weight_bias(blk.mlp[2])
    lin2_law, j_lin2 = linear_law(gelu_out_law, lin2_w, lin2_b)

    j_mlp_branch = j_lin2 @ j_gelu @ j_lin1 @ j_ln2
    cross2 = x1.covar @ j_mlp_branch.T
    mu2 = x1.mu + lin2_law.mu
    cov2 = x1.covar + lin2_law.covar + cross2 + cross2.T
    x2 = _as_law(mu=mu2, covar=cov2)

    err = _closure_error_block(
        law,
        blk,
        x2,
        rng=rng,
        n_mc=n_mc,
        seq_len=seq_len,
        probe_batch_size=probe_batch_size,
    )
    return x2, err, {"gelu_covariance": gelu_receipt}


def iter_moments(
    model: Any,
    input_law: GaussianLaw,
    n_mc: int = 128,
    seq_len: int = 16,
    seed: int = 0,
    probe_batch_size: int = 32,
) -> Iterator[LayerMoments]:
    """Yield layer-local Gaussian laws without retaining earlier receipts.

    ``input_law`` models the distribution of a token's residual-stream vector ENTERING the block stack (i.e.
    after token + position embedding) -- an ``N(mu, Sigma)`` over ``R^{d_model}``.

    The iterator walks ``model.blocks`` one at a time, then the final ``model.ln`` and ``model.head``.
    Closure probes execute in the target module's device/dtype and evaluation mode, restore every
    submodule's prior mode, and reduce batches of at most ``probe_batch_size`` online.
    """
    if not _HAS_TORCH:
        raise RuntimeError("iter_moments requires torch (mixle.models.transformer is torch-only).")
    n_mc = _exact_int(n_mc, "n_mc", minimum=2)
    seq_len = _exact_int(seq_len, "seq_len", minimum=1)
    seed = _exact_int(seed, "seed", minimum=0)
    probe_batch_size = _exact_int(probe_batch_size, "probe_batch_size", minimum=1)

    rng = np.random.default_rng(seed)
    law = input_law

    for i, blk in enumerate(model.blocks):
        law, err, diagnostics = _propagate_block(
            law,
            blk,
            rng=rng,
            n_mc=n_mc,
            seq_len=seq_len,
            probe_batch_size=probe_batch_size,
        )
        yield LayerMoments(
            index=i,
            name=f"block[{i}]",
            law=law,
            closure_error=err,
            diagnostics=diagnostics,
        )

    ln_w, ln_b = _to_numpy(model.ln.weight), _to_numpy(model.ln.bias)
    ln_law, _ = layernorm_law(law, ln_w, ln_b, eps=model.ln.eps)
    ln_err = _closure_error_pointwise(
        law,
        model.ln,
        ln_law,
        rng=rng,
        n_mc=n_mc,
        probe_batch_size=probe_batch_size,
    )
    yield LayerMoments(index=len(model.blocks), name="ln_f", law=ln_law, closure_error=ln_err)

    head_w, head_b = _module_weight_bias(model.head)
    head_law, _ = linear_law(ln_law, head_w, head_b)
    head_err = _closure_error_pointwise(
        ln_law,
        model.head,
        head_law,
        rng=rng,
        n_mc=n_mc,
        probe_batch_size=probe_batch_size,
    )
    yield LayerMoments(index=len(model.blocks) + 1, name="head", law=head_law, closure_error=head_err)


def propagate_moments(
    model: Any,
    input_law: GaussianLaw,
    n_mc: int = 128,
    seq_len: int = 16,
    seed: int = 0,
    probe_batch_size: int = 32,
) -> list[LayerMoments]:
    """Materialize :func:`iter_moments` into a list of per-layer full-covariance receipts.

    Unlike the iterator, this convenience API intentionally retains
    ``O(depth × d_model²)`` returned law state. It remains useful when callers need random access to every
    layer; callers requiring depth-independent retained memory should consume :func:`iter_moments`.
    """
    return list(
        iter_moments(
            model,
            input_law,
            n_mc=n_mc,
            seq_len=seq_len,
            seed=seed,
            probe_batch_size=probe_batch_size,
        )
    )
