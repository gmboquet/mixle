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
        ``mean(a - b)`` and ``favored`` is ``'A'`` / ``'B'`` / ``'tie'`` at the given level.
    """
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    d = a - b
    n = d.shape[0]
    mean_diff = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    tcrit = stats.t.ppf(0.5 + ci_level / 2.0, n - 1)
    t_stat = mean_diff / se if se > 0 else 0.0
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
    la = np.asarray(loglik_a, dtype=float).ravel()
    lb = np.asarray(loglik_b, dtype=float).ravel()
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
    la = np.asarray(loglik_a, dtype=float).ravel()
    lb = np.asarray(loglik_b, dtype=float).ravel()
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
        ``{'elpd_diff', 'se', 'z', 'favored'}`` -- ``elpd_diff = sum(a - b)``.
    """
    a = np.asarray(pointwise_a, dtype=float).ravel()
    b = np.asarray(pointwise_b, dtype=float).ravel()
    d = a - b
    n = d.shape[0]
    elpd_diff = float(d.sum())
    se = float(np.sqrt(n) * d.std(ddof=1))
    z = elpd_diff / se if se > 0 else 0.0
    favored = "tie" if abs(z) < 2.0 else ("A" if elpd_diff > 0 else "B")
    return {"elpd_diff": elpd_diff, "se": se, "z": float(z), "favored": favored}


__all__ = [
    "paired_score_difference",
    "vuong_test",
    "clarke_test",
    "compare_elpd",
]
