"""Conformal prediction: distribution-free intervals with finite-sample coverage.

Conformal prediction wraps *any* point predictor in an interval (or set) guaranteed to contain the
truth with probability ``1 - alpha`` in finite samples, assuming only exchangeability -- no
distributional assumptions about the model or the noise. This module is the array-level toolkit
(operating on a ``fit_predict`` callable or precomputed residuals), complementing the PPL-fit wrappers
in :mod:`mixle.ppl.conformal`:

  * :func:`split_conformal` -- the fast split/inductive interval from a held-out calibration set, with
    optional one-sided (boundary) intervals.
  * :func:`jackknife_plus` / :func:`cv_plus` -- leave-one-out (CV+) intervals that use *all* the data
    for both fitting and calibration, with the J+/CV+ coverage guarantee (Barber et al. 2021).
  * :func:`mondrian_conformal` -- group-conditional intervals: a separate quantile per group, so
    coverage holds *within* each group, not just marginally.
  * :func:`weighted_conformal` -- covariate-shift-robust intervals, reweighting the calibration scores
    by the test/train density ratio (Tibshirani et al. 2019).

``fit_predict`` has the signature ``fit_predict(X_train, y_train, X_eval) -> y_hat`` so any estimator
plugs in.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ``ceil((n+1)(1-alpha))``-th smallest score (finite-sample conformal quantile).

    ``k`` reaches 0 exactly at ``alpha == 1.0`` (a valid boundary: 0% coverage requested, the most
    permissive threshold is never needed and the tightest one always is). ``s[k - 1]`` with ``k == 0``
    used to silently wrap around via Python's negative indexing to ``s[-1]`` -- the MAXIMUM score, the
    loosest threshold instead of the tightest -- breaking monotonicity in ``alpha`` right at the
    boundary and, for ``alpha > 1`` (invalid), either returning an arbitrary interior score or raising
    an uncaught ``IndexError``. Mirrors :func:`weighted_conformal`'s already-correct ``min(k, n-1)``
    convention, which returns the minimum score at this same boundary.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha!r}.")
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.shape[0]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    if k < 1:
        return float(s[0])
    return float(s[k - 1])


def split_conformal(
    cal_pred: np.ndarray,
    cal_y: np.ndarray,
    test_pred: np.ndarray,
    *,
    alpha: float = 0.1,
    side: str = "two-sided",
) -> tuple[np.ndarray, np.ndarray]:
    """Split (inductive) conformal interval from a calibration set.

    Args:
        cal_pred: ``(n,)`` model predictions on the calibration set.
        cal_y: ``(n,)`` calibration responses.
        test_pred: ``(m,)`` predictions at the test points.
        alpha: miscoverage level (``1 - alpha`` coverage).
        side: ``"two-sided"`` (``|y - yhat|`` score), ``"upper"`` (one-sided upper bound), or
            ``"lower"`` (one-sided lower bound).

    Returns:
        ``(lower, upper)`` arrays of length ``m`` (an unbounded side is ``-inf`` / ``+inf``).
    """
    cal_pred = np.asarray(cal_pred, dtype=float)
    cal_y = np.asarray(cal_y, dtype=float)
    test_pred = np.asarray(test_pred, dtype=float)
    if side == "two-sided":
        q = _conformal_quantile(np.abs(cal_y - cal_pred), alpha)
        return test_pred - q, test_pred + q
    if side == "upper":
        q = _conformal_quantile(cal_y - cal_pred, alpha)
        return np.full_like(test_pred, -np.inf), test_pred + q
    if side == "lower":
        q = _conformal_quantile(cal_pred - cal_y, alpha)
        return test_pred - q, np.full_like(test_pred, np.inf)
    raise ValueError("side must be 'two-sided', 'upper', or 'lower'.")


def jackknife_plus(
    x: np.ndarray,
    y: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    x_test: np.ndarray,
    *,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Jackknife+ intervals (leave-one-out), using all data for both fitting and calibration.

    For each training point ``i`` the model is refit without ``i``; ``R_i = |y_i - mu_{-i}(x_i)|`` is the
    LOO residual and ``mu_{-i}(x)`` the LOO prediction at a test point. The interval aggregates
    ``mu_{-i}(x) -/+ R_i`` across ``i`` (Barber et al. 2021), giving ~``1 - 2 alpha`` worst-case and
    ~``1 - alpha`` typical coverage without a data split. Costs ``n`` refits.

    Returns:
        ``(lower, upper)`` arrays of length ``len(x_test)``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    x_test = np.atleast_2d(np.asarray(x_test, dtype=float))
    n, m = x.shape[0], x_test.shape[0]
    loo_test = np.empty((n, m))
    resid = np.empty(n)
    idx = np.arange(n)
    for i in range(n):
        mask = idx != i
        eval_pts = np.vstack([x[i : i + 1], x_test])
        preds = np.asarray(fit_predict(x[mask], y[mask], eval_pts), dtype=float).ravel()
        resid[i] = abs(y[i] - preds[0])
        loo_test[i] = preds[1:]
    lower = np.quantile(loo_test - resid[:, None], alpha, axis=0, method="lower")
    upper = np.quantile(loo_test + resid[:, None], 1.0 - alpha, axis=0, method="higher")
    return lower, upper


def cv_plus(
    x: np.ndarray,
    y: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    x_test: np.ndarray,
    *,
    alpha: float = 0.1,
    n_folds: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """CV+ intervals: the K-fold analogue of :func:`jackknife_plus` (only ``n_folds`` refits).

    Each point's residual uses the model trained on the *other* folds, and the test prediction uses the
    same out-of-fold model. Much cheaper than Jackknife+ with nearly the same guarantee.

    Returns:
        ``(lower, upper)`` arrays of length ``len(x_test)``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    x_test = np.atleast_2d(np.asarray(x_test, dtype=float))
    n, m = x.shape[0], x_test.shape[0]
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    loo_test = np.empty((n, m))
    resid = np.empty(n)
    for fold in folds:
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        eval_pts = np.vstack([x[fold], x_test])
        preds = np.asarray(fit_predict(x[mask], y[mask], eval_pts), dtype=float).ravel()
        k = fold.shape[0]
        resid[fold] = np.abs(y[fold] - preds[:k])
        loo_test[fold] = np.tile(preds[k:], (k, 1))
    lower = np.quantile(loo_test - resid[:, None], alpha, axis=0, method="lower")
    upper = np.quantile(loo_test + resid[:, None], 1.0 - alpha, axis=0, method="higher")
    return lower, upper


def mondrian_conformal(
    cal_pred: np.ndarray,
    cal_y: np.ndarray,
    cal_groups: np.ndarray,
    test_pred: np.ndarray,
    test_groups: np.ndarray,
    *,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Mondrian (group-conditional) split conformal: a separate quantile per group.

    Calibrates the conformal quantile *within* each group (taxonomy), so coverage holds conditional on
    the group rather than only marginally -- the fix when error scale varies across known subpopulations.

    Args:
        cal_pred, cal_y, cal_groups: calibration predictions, responses, and group labels.
        test_pred, test_groups: test predictions and their group labels.
        alpha: miscoverage level.

    Returns:
        ``(lower, upper)`` arrays of length ``len(test_pred)``.
    """
    cal_pred = np.asarray(cal_pred, dtype=float)
    cal_y = np.asarray(cal_y, dtype=float)
    cal_groups = np.asarray(cal_groups)
    test_pred = np.asarray(test_pred, dtype=float)
    test_groups = np.asarray(test_groups)
    scores = np.abs(cal_y - cal_pred)
    qhat: dict = {}
    for g in np.unique(cal_groups):
        qhat[g] = _conformal_quantile(scores[cal_groups == g], alpha)
    q = np.array([qhat.get(g, np.inf) for g in test_groups])
    return test_pred - q, test_pred + q


def weighted_conformal(
    cal_pred: np.ndarray,
    cal_y: np.ndarray,
    test_pred: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float = 0.1,
    test_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Covariate-shift-weighted split conformal (Tibshirani et al. 2019).

    Under covariate shift the calibration and test inputs follow different distributions; reweighting
    the calibration scores by the likelihood ratio ``w(x) = p_test(x)/p_train(x)`` restores coverage.
    Uses the weighted empirical quantile of the calibration scores (each test point shares the same
    ``test_weight`` for its own potential score).

    Args:
        cal_pred, cal_y: calibration predictions and responses.
        test_pred: ``(m,)`` test predictions.
        weights: ``(n,)`` likelihood-ratio weights for the calibration points (need not be normalised).
        alpha: miscoverage level.
        test_weight: the weight assigned to a test point (usually the mean test/train ratio; ``1.0``
            when weights are self-normalised around the test density).

    Returns:
        ``(lower, upper)`` arrays of length ``m`` (a symmetric interval per test point).
    """
    cal_pred = np.asarray(cal_pred, dtype=float)
    cal_y = np.asarray(cal_y, dtype=float)
    test_pred = np.asarray(test_pred, dtype=float)
    w = np.asarray(weights, dtype=float)
    scores = np.abs(cal_y - cal_pred)
    order = np.argsort(scores)
    s_sorted = scores[order]
    w_sorted = w[order]
    total = w_sorted.sum() + test_weight
    cdf = np.cumsum(w_sorted) / total
    k = np.searchsorted(cdf, 1.0 - alpha)
    q = float(s_sorted[min(k, s_sorted.shape[0] - 1)]) if (cdf[-1] >= 1.0 - alpha) else float("inf")
    return test_pred - q, test_pred + q


def conformal_label_threshold(cal_prob_true: np.ndarray, *, alpha: float = 0.1) -> float:
    """Calibrate the LAC (least-ambiguous set-valued classifier) score threshold for ``1 - alpha`` coverage.

    The nonconformity score of a calibration point is ``1 - p_model[true_class]`` -- which needs the model's
    class scores to *rank* well, **not** to be a true probability (the whole point: a softmax over a ReLU net
    is not a describable random process, but conformal still gives a finite-sample coverage guarantee from how
    those scores behave on held-out, exchangeable data). Returns the conformal quantile ``qhat`` of the
    calibration scores; a class is admitted at test time iff ``1 - p[c] <= qhat`` (see :func:`conformal_label_sets`).

    Args:
        cal_prob_true: ``(n,)`` model score assigned to the *true* class of each calibration point.
        alpha: miscoverage level (``1 - alpha`` marginal coverage of the returned sets).

    Returns:
        ``qhat`` -- the score threshold (``+inf`` when ``n`` is too small for the requested ``alpha``).
    """
    scores = 1.0 - np.asarray(cal_prob_true, dtype=float)
    return _conformal_quantile(scores, alpha)


def conformal_label_sets(
    cal_prob_true: np.ndarray,
    test_prob: np.ndarray,
    *,
    alpha: float = 0.1,
    qhat: float | None = None,
) -> tuple[np.ndarray, float]:
    """Split-conformal prediction *sets* for a classifier: distribution-free ``1 - alpha`` label coverage.

    Calibrates a LAC threshold (:func:`conformal_label_threshold`) on the held-out true-class scores, then
    admits every class whose score clears it. The returned boolean mask has guaranteed marginal coverage: the
    true label is in the set with probability ``>= 1 - alpha``. A *singleton* set is a confident prediction; an
    *empty or multi-label* set is an explicit abstention -- the signal a cost-aware cascade escalates on.

    Args:
        cal_prob_true: ``(n,)`` score assigned to the true class of each calibration point.
        test_prob: ``(m, K)`` model class scores at the test points (rows need not sum to 1).
        alpha: miscoverage level.
        qhat: a precomputed threshold (e.g. from an earlier calibration); recomputed if ``None``.

    Returns:
        ``(sets, qhat)`` -- ``sets`` is an ``(m, K)`` boolean mask, ``qhat`` the threshold used.
    """
    if qhat is None:
        qhat = conformal_label_threshold(cal_prob_true, alpha=alpha)
    test_prob = np.asarray(test_prob, dtype=float)
    sets = (1.0 - test_prob) <= qhat
    return sets, float(qhat)


__all__ = [
    "split_conformal",
    "jackknife_plus",
    "cv_plus",
    "mondrian_conformal",
    "weighted_conformal",
    "conformal_label_threshold",
    "conformal_label_sets",
]
