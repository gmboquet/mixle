"""Split (inductive) conformal prediction — distribution-free, finite-sample valid
prediction intervals and label sets around any already-fitted model.

Conformal prediction turns point predictions into calibrated sets using a held-out
*calibration* split, with a coverage guarantee that holds for any model and any data
distribution as long as the calibration and test points are exchangeable: a set built at
level ``alpha`` covers the truth with probability at least ``1 - alpha``.  A wrong model
only makes the sets wider, never breaks the guarantee.

The machinery is a nonconformity score plus one order statistic.  For regression the score
is the absolute residual ``|y - yhat|`` and the calibrated interval is
``predict(x) +/- qhat``; for classification the score is ``1 - p(true class | x)`` and the
label set is ``{y : 1 - p(y | x) <= tau}``.  Both reduce to the conformal quantile
``qhat`` / ``tau`` — the ``ceil((n + 1)(1 - alpha))`` smallest calibration score, the
``+1`` being the finite-sample correction.

:class:`ConformalRegressor` wraps a fitted :class:`~pysp.ppl.regression.RegressionResult`
(anything exposing ``predict(given)``); :class:`ConformalClassifier` wraps a matrix of
per-class probabilities (e.g. the posterior of a pysparkplug generative classifier).  The
:func:`conformal` helper is the one-liner entry point for the regression case.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def conformal_quantile(scores: Any, alpha: float) -> float:
    """The level-``alpha`` conformal quantile of calibration ``scores``.

    Returns the ``ceil((n + 1)(1 - alpha))`` smallest score (the finite-sample-corrected
    empirical ``1 - alpha`` quantile).  When ``alpha`` is too small for the calibration
    size — ``(n + 1)(1 - alpha) > n`` — no finite threshold gives the requested coverage
    and ``inf`` is returned, the honest "the set is everything" answer.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    if n == 0:
        raise ValueError("conformal calibration needs at least one score.")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


class ConformalRegressor:
    """Split-conformal prediction intervals around a fitted regression ``result``.

    Calibrates the absolute-residual nonconformity score on held-out ``(given, y_cal)`` and
    produces symmetric intervals ``predict(x) +/- qhat`` with marginal coverage at least
    ``1 - alpha``.  ``result`` is any object with a ``predict(given)`` method returning the
    fitted mean (a :class:`~pysp.ppl.regression.RegressionResult`, a location-scale result,
    or a GP regressor).
    """

    def __init__(self, result: Any, given: dict, y_cal: Any, *, alpha: float = 0.1) -> None:
        self.result = result
        self.alpha = float(alpha)
        yhat = np.asarray(result.predict(given), dtype=float).reshape(-1)
        y = np.asarray(y_cal, dtype=float).reshape(-1)
        if yhat.shape != y.shape:
            raise ValueError(f"calibration predictions {yhat.shape} and targets {y.shape} disagree.")
        self.scores = np.abs(y - yhat)
        self.qhat = conformal_quantile(self.scores, self.alpha)  # interval half-width

    def interval(self, given: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` arrays of the conformal interval at covariates ``given``."""
        center = np.asarray(self.result.predict(given), dtype=float).reshape(-1)
        return center - self.qhat, center + self.qhat

    def covers(self, given: dict, y: Any) -> np.ndarray:
        """Boolean array: does the interval at ``given`` contain each observed ``y``."""
        lo, hi = self.interval(given)
        y = np.asarray(y, dtype=float).reshape(-1)
        return (y >= lo) & (y <= hi)


class ConformalClassifier:
    """Split-conformal label sets from per-class probabilities.

    ``proba_cal`` is an ``(n, K)`` matrix of calibration probabilities ``p(y | x)`` (any
    proper classifier — a pysparkplug generative classifier's class posterior, a softmax,
    ...) and ``y_cal`` the integer labels.  The nonconformity score is ``1 - p(true)`` and
    the calibrated set keeps every label whose score is within the conformal quantile, so it
    covers the true label with probability at least ``1 - alpha`` and grows from one label
    (confident) to several (hedging) as the model is unsure.
    """

    def __init__(self, proba_cal: Any, y_cal: Any, *, alpha: float = 0.1) -> None:
        proba = np.asarray(proba_cal, dtype=float)
        y = np.asarray(y_cal, dtype=int).reshape(-1)
        if proba.ndim != 2 or proba.shape[0] != y.shape[0]:
            raise ValueError("proba_cal must be (n_calibration, n_classes) aligned with y_cal.")
        self.alpha = float(alpha)
        self.scores = 1.0 - proba[np.arange(y.size), y]
        self.tau = conformal_quantile(self.scores, self.alpha)

    def predict_set(self, proba: Any) -> np.ndarray:
        """Boolean ``(n, K)`` label-inclusion matrix at probabilities ``proba``."""
        return (1.0 - np.asarray(proba, dtype=float)) <= self.tau

    def covers(self, proba: Any, y: Any) -> np.ndarray:
        """Boolean array: is each true label ``y`` in the predicted set."""
        sets = self.predict_set(proba)
        y = np.asarray(y, dtype=int).reshape(-1)
        return sets[np.arange(y.size), y]

    def set_sizes(self, proba: Any) -> np.ndarray:
        """Number of labels in the predicted set for each row of ``proba``."""
        return self.predict_set(proba).sum(axis=1)


def conformal(result: Any, given: dict, y_cal: Any, *, alpha: float = 0.1) -> ConformalRegressor:
    """Split-conformal calibration of a fitted regression ``result`` into prediction intervals.

    One-liner over :class:`ConformalRegressor`::

        m = Normal(free * Field("x") + free, free).fit(y_tr, given={"x": x_tr})
        cp = conformal(m.result, {"x": x_cal}, y_cal, alpha=0.1)
        lo, hi = cp.interval({"x": x_te})
        cp.covers({"x": x_te}, y_te).mean()   # ~ 0.9
    """
    return ConformalRegressor(result, given, y_cal, alpha=alpha)
