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


def _alpha(alpha: float) -> float:
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or not 0.0 <= float(alpha) <= 1.0
    ):
        raise ValueError(f"alpha must be a finite number in [0.0, 1.0], got {alpha!r}.")
    return float(alpha)


def _finite_vector(name: str, values: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional finite numeric array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _feature_matrix(name: str, values: np.ndarray, *, n_features: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite two-dimensional feature matrix") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty shape (n_samples, n_features)")
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError(f"{name} has {array.shape[1]} features; expected {n_features}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _predictions(values: np.ndarray, expected: int, *, source: str) -> np.ndarray:
    predictions = _finite_vector(f"{source} predictions", values)
    if len(predictions) != expected:
        raise ValueError(f"{source} returned {len(predictions)} predictions; expected exactly {expected}")
    return predictions


def _probability_vector(name: str, values: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
    probabilities = _finite_vector(name, values, allow_empty=allow_empty)
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return probabilities


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
    alpha = _alpha(alpha)
    s = np.sort(_finite_vector("conformal calibration scores", scores))
    n = s.shape[0]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    if k < 1:
        return float(s[0])
    return float(s[k - 1])


def _jackknife_plus_bounds(lo_vals: np.ndarray, hi_vals: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Barber et al. (2021) finite-sample J+/CV+ endpoints from ``(n, m)`` per-fold bound matrices.

    Per test point, the lower endpoint is the ``floor(alpha (n+1))``-th smallest of
    ``mu_{-i}(x) - R_i`` and the upper the ``ceil((1-alpha)(n+1))``-th smallest of
    ``mu_{-i}(x) + R_i`` -- the explicit ``(n+1)``-based order statistics that carry the J+
    coverage guarantee (mirroring :func:`_conformal_quantile`; ``np.quantile``'s ``(n-1)``-based
    virtual indices do not). An out-of-range index (small ``n`` for the requested ``alpha``)
    yields the honest unbounded endpoint ``-inf`` / ``+inf``.
    """
    alpha = _alpha(alpha)
    lo_vals = np.asarray(lo_vals, dtype=float)
    hi_vals = np.asarray(hi_vals, dtype=float)
    if lo_vals.ndim != 2 or hi_vals.shape != lo_vals.shape or 0 in lo_vals.shape:
        raise ValueError("jackknife/CV+ bound matrices must have the same non-empty two-dimensional shape")
    if not np.all(np.isfinite(lo_vals)) or not np.all(np.isfinite(hi_vals)):
        raise ValueError("jackknife/CV+ bound matrices must be finite")
    n, m = lo_vals.shape
    k_lo = int(np.floor(alpha * (n + 1)))
    k_hi = int(np.ceil((1.0 - alpha) * (n + 1)))
    lower = np.full(m, -np.inf) if k_lo < 1 else np.sort(lo_vals, axis=0)[k_lo - 1]
    upper = np.full(m, np.inf) if k_hi > n else np.sort(hi_vals, axis=0)[k_hi - 1]
    return lower, upper


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
    cal_pred = _finite_vector("cal_pred", cal_pred)
    cal_y = _finite_vector("cal_y", cal_y)
    test_pred = _finite_vector("test_pred", test_pred)
    if cal_pred.shape != cal_y.shape:
        raise ValueError("cal_pred and cal_y must have matching one-dimensional shapes")
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
    _alpha(alpha)
    x = _feature_matrix("x", x)
    y = _finite_vector("y", y)
    x_test = _feature_matrix("x_test", x_test, n_features=x.shape[1])
    n, m = x.shape[0], x_test.shape[0]
    if n < 2:
        raise ValueError("jackknife_plus requires at least two training observations")
    if len(y) != n:
        raise ValueError("x and y must contain the same number of observations")
    if not callable(fit_predict):
        raise TypeError("fit_predict must be callable")
    loo_test = np.empty((n, m))
    resid = np.empty(n)
    idx = np.arange(n)
    for i in range(n):
        mask = idx != i
        eval_pts = np.vstack([x[i : i + 1], x_test])
        preds = _predictions(fit_predict(x[mask], y[mask], eval_pts), 1 + m, source="fit_predict")
        resid[i] = abs(y[i] - preds[0])
        loo_test[i] = preds[1:]
    return _jackknife_plus_bounds(loo_test - resid[:, None], loo_test + resid[:, None], alpha)


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
    _alpha(alpha)
    x = _feature_matrix("x", x)
    y = _finite_vector("y", y)
    x_test = _feature_matrix("x_test", x_test, n_features=x.shape[1])
    n, m = x.shape[0], x_test.shape[0]
    if len(y) != n:
        raise ValueError("x and y must contain the same number of observations")
    if isinstance(n_folds, (bool, np.bool_)) or not isinstance(n_folds, (int, np.integer)):
        raise ValueError("n_folds must be an integer in [2, n]")
    n_folds = int(n_folds)
    if not 2 <= n_folds <= n:
        raise ValueError("n_folds must be in [2, n]")
    if not callable(fit_predict):
        raise TypeError("fit_predict must be callable")
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    loo_test = np.empty((n, m))
    resid = np.empty(n)
    for fold in folds:
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        eval_pts = np.vstack([x[fold], x_test])
        preds = _predictions(fit_predict(x[mask], y[mask], eval_pts), len(fold) + m, source="fit_predict")
        k = fold.shape[0]
        resid[fold] = np.abs(y[fold] - preds[:k])
        loo_test[fold] = np.tile(preds[k:], (k, 1))
    return _jackknife_plus_bounds(loo_test - resid[:, None], loo_test + resid[:, None], alpha)


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
    cal_pred = _finite_vector("cal_pred", cal_pred)
    cal_y = _finite_vector("cal_y", cal_y)
    cal_groups = np.asarray(cal_groups)
    test_pred = _finite_vector("test_pred", test_pred)
    test_groups = np.asarray(test_groups)
    if cal_pred.shape != cal_y.shape:
        raise ValueError("cal_pred and cal_y must have matching one-dimensional shapes")
    if cal_groups.ndim != 1 or cal_groups.shape != cal_pred.shape:
        raise ValueError("cal_groups must be one-dimensional and match the calibration arrays")
    if test_groups.ndim != 1 or test_groups.shape != test_pred.shape:
        raise ValueError("test_groups must be one-dimensional and match test_pred")
    for name, labels in (("cal_groups", cal_groups), ("test_groups", test_groups)):
        for label in labels.tolist():
            if label is None or (isinstance(label, (float, np.floating)) and np.isnan(label)):
                raise ValueError(f"{name} cannot contain missing labels")
            try:
                hash(label)
            except TypeError as exc:
                raise ValueError(f"{name} must contain hashable scalar labels") from exc
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
    test_weight: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Covariate-shift-weighted split conformal (Tibshirani et al. 2019).

    Under covariate shift the calibration and test inputs follow different distributions; reweighting
    the calibration scores by the likelihood ratio ``w(x) = p_test(x)/p_train(x)`` restores coverage
    -- PER QUERY. The theorem places EACH test point's own ``w(x_test)`` alongside the calibration
    weights, so ``test_weight`` is REQUIRED and per-query: pass an ``(m,)`` array of each query's
    likelihood ratio, or a scalar as your explicit assertion that ``w(x_test)`` is the same for
    every query (true only when the shift does not move the test points' own ratios). An earlier
    signature defaulted to a single shared scalar and suggested "the mean test/train ratio"; on a
    two-point-covariate fixture with exact ratios w(0)=0.505 / w(1)=50 that default covered 0.589
    of 20,000 trials at nominal 0.90 while the query-specific weights covered 1.000
    (STAT-RR21-07) -- a shared scalar cannot apply the heavy query's required mass.

    Args:
        cal_pred, cal_y: calibration predictions and responses.
        test_pred: ``(m,)`` test predictions.
        weights: ``(n,)`` likelihood-ratio weights for the calibration points (need not be normalised).
        alpha: miscoverage level, in ``[0.0, 1.0]`` (``0.0`` and ``1.0`` are valid boundaries, as for
            :func:`_conformal_quantile`).
        test_weight: ``(m,)`` per-query likelihood ratios ``w(x_test_i)`` (or a scalar, asserting a
            constant ratio across these queries). Required.

    Returns:
        ``(lower, upper)`` arrays of length ``m`` (a symmetric interval per test point, each at its
        own weighted quantile).
    """
    alpha = _alpha(alpha)
    cal_pred = _finite_vector("cal_pred", cal_pred)
    cal_y = _finite_vector("cal_y", cal_y)
    test_pred = _finite_vector("test_pred", test_pred)
    w = _finite_vector("weights", weights)
    # This function computes its own weighted quantile rather than routing through
    # _conformal_quantile (the weighting has no unweighted equivalent there), so it needs its own
    # complete set of guards rather than inheriting _conformal_quantile's.
    if w.shape != cal_pred.shape or w.shape != cal_y.shape:
        raise ValueError(
            f"weights, cal_pred, and cal_y must have matching shape, got {w.shape}, {cal_pred.shape}, {cal_y.shape}."
        )
    if test_weight is None:
        raise ValueError(
            "test_weight is required: weighted conformal places each query's OWN likelihood ratio "
            "w(x_test) alongside the calibration weights (STAT-RR21-07 -- a shared default scalar "
            "covered 0.589 at nominal 0.90). Pass an (m,) array of per-query ratios, or a scalar "
            "as an explicit assertion that the ratio is constant across these queries."
        )
    if isinstance(test_weight, (bool, np.bool_)):
        raise ValueError("weights and test_weight must be finite numbers, not Booleans.")
    test_w = np.asarray(test_weight, dtype=float)
    if test_w.ndim == 0:
        test_w = np.full(test_pred.shape, float(test_w))
    if test_w.shape != test_pred.shape:
        raise ValueError(
            f"test_weight must be a scalar or match test_pred's shape {test_pred.shape}, got {test_w.shape}."
        )
    if not np.all(np.isfinite(test_w)):
        # checked before the sign check below: a NaN comparison is always False, so `NaN < 0.0` would
        # otherwise silently pass as "non-negative" and corrupt every downstream sum/cdf entry.
        raise ValueError("weights and test_weight must be finite.")
    if np.any(w < 0.0) or np.any(test_w < 0.0):
        raise ValueError("weights and test_weight must be non-negative likelihood ratios.")
    total_weights = w.sum() + test_w  # (m,) -- each query brings its own mass to the denominator
    if not np.all(np.isfinite(total_weights)) or np.any(total_weights <= 0.0):
        # w and test_w are already known finite and non-negative, so the only way to land here is
        # every weight (calibration and that query's own) being exactly zero -- a degenerate
        # reweighting whose CDF would be 0/0 = NaN, and NaN comparisons silently read as
        # "insufficient mass" (q = inf) instead of raising.
        raise ValueError("weighted_conformal requires a positive, finite total weight for every query.")
    scores = np.abs(cal_y - cal_pred)
    order = np.argsort(scores)
    s_sorted = scores[order]
    cumulative = np.cumsum(w[order])  # (n,) unnormalised calibration mass
    # Per query i: the smallest score s_(k) with cumulative_k >= (1 - alpha) * (W + w_test_i).
    # The query's own weight sits at +inf (its score is unknown), so it never helps reach the
    # level -- exactly the Tibshirani et al. construction, applied at each query's own ratio.
    required = (1.0 - alpha) * total_weights  # (m,)
    k = np.searchsorted(cumulative, required)
    reachable = cumulative[-1] >= required
    q = np.where(reachable, s_sorted[np.minimum(k, s_sorted.shape[0] - 1)], np.inf)
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
    scores = 1.0 - _probability_vector("cal_prob_true", cal_prob_true)
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
    alpha = _alpha(alpha)
    cal_prob_true = _probability_vector("cal_prob_true", cal_prob_true, allow_empty=qhat is not None)
    if qhat is None:
        qhat = conformal_label_threshold(cal_prob_true, alpha=alpha)
    elif (
        isinstance(qhat, (bool, np.bool_))
        or not isinstance(qhat, (int, float, np.integer, np.floating))
        or np.isnan(qhat)
        or qhat < 0.0
        or (not np.isposinf(qhat) and qhat > 1.0)
    ):
        raise ValueError("qhat must be a finite value in [0, 1] or positive infinity")
    test_prob = np.asarray(test_prob, dtype=float)
    if test_prob.ndim != 2 or test_prob.shape[0] == 0 or test_prob.shape[1] == 0:
        raise ValueError("test_prob must have non-empty shape (n_samples, n_classes)")
    if not np.all(np.isfinite(test_prob)):
        raise ValueError("test_prob must contain only finite probabilities")
    if np.any((test_prob < 0.0) | (test_prob > 1.0)):
        raise ValueError("test_prob must contain probabilities in [0, 1]")
    if not np.allclose(test_prob.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("each test_prob row must sum to 1")
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
