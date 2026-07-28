"""Model comparison: paired score differences and non-nested tests.

Is model A actually better than model B, or did it just win by chance on this sample? These tools answer
that from *paired, per-observation* held-out scores or log-likelihoods -- pairing removes the
observation-to-observation variance that swamps a comparison of two separate score totals:

  * :func:`paired_score_difference` -- the mean held-out score difference with a confidence interval and
    a paired test (works for any proper score from :mod:`mixle.inference.scoring`: CRPS, log score, ...).
  * :func:`vuong_test` -- the Vuong (1989) likelihood-ratio test for **non-nested** models, with an
    optional AIC/BIC complexity correction.
  * :func:`clarke_test` -- Clarke's distribution-free paired sign test, a robust alternative to Vuong
    when the log-likelihood-ratio distribution is non-normal.
  * :func:`compare_elpd` -- the standard LOO/WAIC comparison: the expected-log-predictive-density
    difference with the standard error of the *pointwise* difference (pair these with the ``pointwise``
    arrays from :func:`mixle.ppl.diagnostics.psis_loo`).

For scores lower is better; for log-likelihoods / elpd higher is better. Each result names the favored
model.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _as_paired_series(x: np.ndarray, name: str) -> np.ndarray:
    """``x`` as the documented ``(n,)`` per-observation array -- validated *before* any coercion.

    Every comparison here documents one-dimensional, per-observation input, but each used to call
    ``.ravel()`` first and validate the already-flattened result. Flattening silently redefines the
    experimental unit: a malformed ``(2, 3)`` pair became six independent observations, and the paired
    t route reported ``p=0.00593``, Vuong ``p=4.59e-06``, and Clarke ``p=0.03125`` off it. Real
    ``(model, fold)``, ``(chain, draw)`` or repeated-measures arrays flatten the same way, turning
    dependent measurements into pseudo-replicates that overstate precision -- and the returned dict
    carries no trace of the original shape, so nothing downstream can notice.

    Pairing is what these tests are for, so the observation axis has to be the caller's declared one.
    An intentional flattening is still available by passing ``x.ravel()`` explicitly.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D array of per-observation values, got shape {a.shape}. These are paired "
            "per-observation comparisons: flattening a higher-dimensional array would silently treat "
            f"{a.size} dependent measurements (folds, chains, outcomes, repeated measures) as independent "
            "observations and overstate precision. Reduce to one value per experimental unit, or pass "
            "an explicitly raveled array if the flattening is what you mean."
        )
    return a


def _validate_paired(a: np.ndarray, b: np.ndarray, *, min_n: int = 2) -> None:
    """Common guard for every paired per-observation comparison in this module.

    Without this, mismatched-length inputs silently broadcast (numpy raises only when the shapes
    are not broadcast-compatible at all) into a confidently-wrong verdict instead of an error, and
    n<min_n starves the ddof=1 standard deviation these functions all compute into NaN, which then
    propagates through a comparison (`stat > 0`, `p >= 0.05`) that is False for NaN either way and
    so silently resolves to a specific, seemingly-decisive favored side instead of "tie"/an error.
    """
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have the same shape, got {a.shape} and {b.shape}.")
    if a.shape[0] < min_n:
        raise ValueError(f"need at least {min_n} paired observations, got {a.shape[0]}.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired arrays must be finite (no NaN/Inf).")


def paired_score_difference(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    lower_is_better: bool = True,
    ci_level: float = 0.95,
) -> dict:
    """Mean paired held-out score difference with a CI and a paired t-test.

    Args:
        scores_a, scores_b: ``(n,)`` per-observation held-out scores for the two models (same
            observations, same order).
        lower_is_better: True for losses/scores (CRPS, log loss, pinball); False for higher-is-better
            metrics.
        ci_level: confidence level for the interval on the mean difference.

    Returns:
        ``{'mean_diff', 'se', 'ci_low', 'ci_high', 't', 'p_value', 'favored'}`` where ``mean_diff`` is
        ``mean(a - b)`` and ``favored`` is ``'A'`` / ``'B'`` / ``'tie'`` at the given level. A zero-
        variance paired difference (every observation agrees exactly) is maximally significant when
        the shared value is nonzero, not an automatic tie -- see the ``se == 0`` handling below.
    """
    a = _as_paired_series(scores_a, "scores_a")
    b = _as_paired_series(scores_b, "scores_b")
    _validate_paired(a, b)
    d = a - b
    n = d.shape[0]
    mean_diff = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    tcrit = stats.t.ppf(0.5 + ci_level / 2.0, n - 1)
    if se > 0:
        t_stat = mean_diff / se
    elif mean_diff != 0.0:
        # Every paired difference is identically the same nonzero value: se == 0 exactly, and
        # ci_low/ci_high below collapse to the single point `mean_diff` -- already reporting
        # certainty. `mean_diff / se` would naively be +-inf/nan depending on how it's evaluated;
        # the old `else 0.0` fallback instead threw away the sign and forced t=0/p=1 ("no evidence"),
        # exactly backwards -- zero dispersion with a nonzero mean is the STRONGEST evidence a paired
        # test can produce, not the weakest. Signed infinity is the honest t -> +-inf limit as
        # se -> 0 with a fixed nonzero numerator; mirrors geweke_z's identical denom==0-with-nonzero
        # -diff fix in mixle/inference/diagnostics.py.
        t_stat = np.inf if mean_diff > 0 else -np.inf
    else:
        t_stat = 0.0  # d is exactly zero everywhere -- a real tie, not just underpowered.
    p = float(2.0 * stats.t.sf(abs(t_stat), n - 1))
    favored = "tie"
    if p < 1.0 - ci_level:
        a_better = (mean_diff < 0) if lower_is_better else (mean_diff > 0)
        favored = "A" if a_better else "B"
    return {
        "mean_diff": mean_diff,
        "se": se,
        "ci_low": mean_diff - tcrit * se,
        "ci_high": mean_diff + tcrit * se,
        "t": float(t_stat),
        "p_value": p,
        "favored": favored,
    }


def _complexity_correction(correction: str, k_a: int, k_b: int, n: int) -> float:
    if correction == "none":
        return 0.0
    if correction == "aic":
        return float(k_a - k_b)
    if correction == "bic":
        return float((k_a - k_b) * np.log(n) / 2.0)
    raise ValueError("correction must be 'none', 'aic', or 'bic'.")


def vuong_test(
    loglik_a: np.ndarray,
    loglik_b: np.ndarray,
    *,
    k_a: int = 0,
    k_b: int = 0,
    correction: str = "none",
) -> dict:
    """Vuong's test for non-nested model selection.

    Compares two models by their pointwise log-likelihoods. Under the null that both are equally close
    to the truth, the statistic ``sqrt(n) * mean(m) / sd(m)`` (with ``m_i = ll_a_i - ll_b_i``, minus an
    optional complexity correction) is asymptotically standard normal. A large positive value favors A.

    Args:
        loglik_a, loglik_b: ``(n,)`` pointwise log-likelihoods of the two (non-nested) models.
        k_a, k_b: parameter counts, used only if ``correction`` is set.
        correction: ``"none"``, ``"aic"`` (subtract ``k_a - k_b``), or ``"bic"`` (subtract
            ``(k_a - k_b) log n / 2``) from the log-likelihood ratio.

    Returns:
        ``{'statistic', 'p_value', 'favored'}``.
    """
    la = _as_paired_series(loglik_a, "loglik_a")
    lb = _as_paired_series(loglik_b, "loglik_b")
    _validate_paired(la, lb)
    m = la - lb
    n = m.shape[0]
    lr = m.sum() - _complexity_correction(correction, k_a, k_b, n)
    omega = m.std(ddof=1)
    # Vuong's variance pretest: when the pointwise log-ratios are (nearly) constant the two models
    # are observationally indistinguishable and the ratio statistic is meaningless -- a tiny but
    # nonzero omega otherwise manufactures an enormous "significant" statistic from pure noise.
    scale = max(float(np.abs(m).max(initial=0.0)), 1.0)
    if omega <= 1e-12 * scale:
        return {"statistic": 0.0, "p_value": 1.0, "favored": "tie", "indistinguishable": True}
    stat = float(lr / (np.sqrt(n) * omega))
    p = float(2.0 * stats.norm.sf(abs(stat)))
    favored = "tie" if p >= 0.05 else ("A" if stat > 0 else "B")
    return {"statistic": stat, "p_value": p, "favored": favored, "indistinguishable": False}


def clarke_test(
    loglik_a: np.ndarray,
    loglik_b: np.ndarray,
    *,
    k_a: int = 0,
    k_b: int = 0,
    correction: str = "none",
) -> dict:
    """Clarke's distribution-free paired sign test for non-nested models.

    Counts how often model A's pointwise log-likelihood beats B's; under the null this count is
    ``Binomial(n, 0.5)``. More robust than :func:`vuong_test` when the per-observation log-ratio is
    heavy-tailed or skewed (where the normal approximation behind Vuong fails).

    Returns:
        ``{'statistic', 'p_value', 'favored', 'n'}`` -- ``statistic`` is the number of points favoring A.
    """
    la = _as_paired_series(loglik_a, "loglik_a")
    lb = _as_paired_series(loglik_b, "loglik_b")
    _validate_paired(la, lb)
    n = la.shape[0]
    d = la - lb - _complexity_correction(correction, k_a, k_b, n) / n
    b = int(np.sum(d > 0))
    nonzero = int(np.sum(d != 0))
    p = float(stats.binomtest(b, nonzero, 0.5).pvalue) if nonzero > 0 else 1.0
    favored = "tie" if p >= 0.05 else ("A" if b > nonzero / 2 else "B")
    return {"statistic": b, "p_value": p, "favored": favored, "n": nonzero}


def compare_elpd(pointwise_a: np.ndarray, pointwise_b: np.ndarray) -> dict:
    """Compare two models' expected log pointwise predictive density (LOO/WAIC).

    Takes the per-observation ``elpd`` contributions (the ``pointwise`` arrays returned by
    :func:`mixle.ppl.diagnostics.psis_loo` / ``waic``) and returns the elpd difference with the standard
    error of the *pointwise* difference -- the standard-error estimate for model comparison (a difference within ~2 SE
    of zero is not decisive).

    Args:
        pointwise_a, pointwise_b: ``(n,)`` per-observation elpd contributions (higher is better).

    Returns:
        ``{'elpd_diff', 'se', 'z', 'favored'}`` -- ``elpd_diff = sum(a - b)``. A zero-variance
        pointwise difference is maximally significant when ``elpd_diff`` is nonzero, not an automatic
        tie -- see the ``se == 0`` handling below (same degeneracy as :func:`paired_score_difference`).
    """
    a = _as_paired_series(pointwise_a, "pointwise_a")
    b = _as_paired_series(pointwise_b, "pointwise_b")
    _validate_paired(a, b)
    d = a - b
    n = d.shape[0]
    elpd_diff = float(d.sum())
    se = float(np.sqrt(n) * d.std(ddof=1))
    if se > 0:
        z = elpd_diff / se
    elif elpd_diff != 0.0:
        # Same zero-SE-with-nonzero-diff degeneracy as paired_score_difference (see its comment): a
        # deterministic, exactly-repeated pointwise difference has se == 0, and reporting `se: 0.0`
        # right next to a nonzero `elpd_diff` already claims certainty -- z must agree, not silently
        # fall back to "tie".
        z = np.inf if elpd_diff > 0 else -np.inf
    else:
        z = 0.0  # pointwise difference is exactly zero everywhere -- a real tie.
    favored = "tie" if abs(z) < 2.0 else ("A" if elpd_diff > 0 else "B")
    return {"elpd_diff": elpd_diff, "se": se, "z": float(z), "favored": favored}


__all__ = [
    "paired_score_difference",
    "vuong_test",
    "clarke_test",
    "compare_elpd",
]
