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

:class:`ConformalRegressor` wraps a fitted :class:`~mixle.ppl.regression.RegressionResult`
(anything exposing ``predict(given)``); :class:`ConformalClassifier` wraps a matrix of
per-class probabilities (e.g. the posterior of a mixle generative classifier).  The
:func:`conformal` helper is the one-liner entry point for the regression case.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def conformal_quantile(scores: Any, alpha: float) -> float:
    """The level-``alpha`` conformal quantile of calibration ``scores``.

    Returns the ``ceil((n + 1)(1 - alpha))`` smallest score (the finite-sample-corrected
    empirical ``1 - alpha`` quantile).  When ``alpha`` is too small for the calibration
    size -- ``(n + 1)(1 - alpha) > n`` -- no finite threshold gives the requested coverage
    and ``inf`` is returned, representing an unconstrained prediction set.
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
    fitted mean (a :class:`~mixle.ppl.regression.RegressionResult`, a location-scale result,
    or a GP regressor).
    """

    def __init__(self, result: Any, y_cal: Any, *, given: dict, alpha: float = 0.1) -> None:
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

    def covers(self, y: Any, *, given: dict) -> np.ndarray:
        """Boolean array: does the interval at ``given`` contain each observed ``y``."""
        lo, hi = self.interval(given)
        y = np.asarray(y, dtype=float).reshape(-1)
        return (y >= lo) & (y <= hi)


class ConformalClassifier:
    """Split-conformal label sets from per-class probabilities.

    ``proba_cal`` is an ``(n, K)`` matrix of calibration probabilities ``p(y | x)`` (any
    proper classifier — a mixle generative classifier's class posterior, a softmax,
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


class ConformalQuantileRegressor:
    """Conformalized quantile regression (Romano, Patterson, Candes 2019).

    Combines two fitted quantile regressions (a lower and an upper conditional quantile) with a
    split-conformal calibration so the band has exact marginal coverage *and* the adaptive,
    heteroscedastic width of quantile regression — wide where the data is noisy, narrow where it is
    tight, unlike the constant-width absolute-residual band of :class:`ConformalRegressor`.

    The nonconformity score is the signed distance outside the predicted band,
    ``E_i = max(qlo(x_i) - y_i, y_i - qhi(x_i))`` (negative when ``y_i`` is comfortably inside), and
    the calibrated band is ``[qlo(x) - qhat, qhi(x) + qhat]`` with ``qhat`` the conformal quantile of
    the calibration scores. ``lo`` and ``hi`` are fitted quantile-regression results (from
    ``...fit(..., quantile=tau)``), typically at ``tau = alpha/2`` and ``1 - alpha/2``.
    """

    def __init__(self, lo: Any, hi: Any, y_cal: Any, *, given: dict, alpha: float = 0.1) -> None:
        self.lo = lo
        self.hi = hi
        self.alpha = float(alpha)
        y = np.asarray(y_cal, dtype=float).reshape(-1)
        qlo = np.asarray(lo.predict(given), dtype=float).reshape(-1)
        qhi = np.asarray(hi.predict(given), dtype=float).reshape(-1)
        self.scores = np.maximum(qlo - y, y - qhi)  # CQR nonconformity (negative when inside the band)
        self.qhat = conformal_quantile(self.scores, self.alpha)

    def interval(self, given: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` arrays of the calibrated adaptive band at covariates ``given``."""
        qlo = np.asarray(self.lo.predict(given), dtype=float).reshape(-1)
        qhi = np.asarray(self.hi.predict(given), dtype=float).reshape(-1)
        return qlo - self.qhat, qhi + self.qhat

    def covers(self, y: Any, *, given: dict) -> np.ndarray:
        """Boolean array: does the adaptive band at ``given`` contain each observed ``y``."""
        lo, hi = self.interval(given)
        y = np.asarray(y, dtype=float).reshape(-1)
        return (y >= lo) & (y <= hi)


class ConformalStructure:
    """Split-conformal credible sets over combinatorial structures (rankings, matchings, spanning
    trees, permutations, ...) from a fitted mixle distribution's exact log-density.

    The nonconformity of a structure ``s`` is ``-log p(s)``: the lower its model probability, the
    more surprising it is.  Calibrating on held-out true structures yields a log-probability
    threshold, and the conformal set is ``{s : log p(s) >= threshold}`` — it contains the true
    structure with probability at least ``1 - alpha`` whenever the calibration and test structures
    are exchangeable (for example iid draws), with no assumption on the model being correct.

    ``dist`` is any structure distribution exposing ``log_density`` (``PlackettLuceDistribution``,
    ``MallowsDistribution``, ``MatchingDistribution``, ``SpanningTreeDistribution``, ...);
    ``calibration`` is a sequence of observed structures.  Membership is always available; listing or
    counting the set additionally needs the distribution's exact ``enumerator()``.
    """

    def __init__(self, dist: Any, calibration: Any, *, alpha: float = 0.1) -> None:
        self.dist = dist
        self.alpha = float(alpha)
        self.scores = np.array([-float(dist.log_density(s)) for s in calibration], dtype=float)
        self.qhat = conformal_quantile(self.scores, self.alpha)  # largest admitted nonconformity

    @property
    def log_prob_threshold(self) -> float:
        """Structures with ``log p(s)`` at or above this value are in the conformal set."""
        return -self.qhat

    def contains(self, structure: Any) -> bool:
        """Is ``structure`` in the conformal set (its log-probability above the threshold)."""
        return bool(-float(self.dist.log_density(structure)) <= self.qhat)

    def covers(self, structures: Any) -> np.ndarray:
        """Boolean array: membership of each structure (use on held-out truths to check coverage)."""
        return np.array([self.contains(s) for s in structures], dtype=bool)

    def members(self) -> list:
        """List the structures in the conformal set, highest-probability first.

        Requires the distribution's exact ``enumerator()`` (raises ``EnumerationError`` otherwise).
        The enumerator yields structures in descending log-probability, so the scan stops at the
        threshold.
        """
        out = []
        for structure, log_p in self.dist.enumerator():
            if log_p < self.log_prob_threshold:
                break
            out.append(structure)
        return out

    def size(self) -> int:
        """Number of structures in the conformal set (needs the exact ``enumerator()``)."""
        return len(self.members())


class ConformalLinkPredictor:
    """Split-conformal candidate-neighbor sets for a random-graph model from its edge-probability
    matrix ``P`` (``P[i, j] = p(edge i--j)``, e.g. ``X @ X.T`` from a fitted RDPG, or an
    Erdos-Renyi / stochastic-block-model edge probability).

    The nonconformity of a present edge ``(i, j)`` is ``1 - P[i, j]``.  Calibrating on held-out true
    edges gives a threshold; the predicted neighbor set of a node keeps every candidate ``j`` with
    ``1 - P[i, j] <= tau``, so it contains a true neighbor with probability at least ``1 - alpha``
    over exchangeable held-out edges (a random split of the observed edges).
    """

    def __init__(self, edge_prob: Any, cal_edges: Any, *, alpha: float = 0.1) -> None:
        self.P = np.asarray(edge_prob, dtype=float)
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("edge_prob must be a square (n_nodes, n_nodes) probability matrix.")
        self.alpha = float(alpha)
        scores = np.array([1.0 - self.P[int(i), int(j)] for i, j in cal_edges], dtype=float)
        self.tau = conformal_quantile(scores, self.alpha)

    def neighbor_set(self, i: int, candidates: Any = None) -> np.ndarray:
        """Candidate nodes ``j`` in node ``i``'s conformal neighbor set."""
        row = self.P[int(i)]
        cand = np.arange(row.size) if candidates is None else np.asarray(candidates, dtype=int)
        return cand[(1.0 - row[cand]) <= self.tau]

    def covers(self, edges: Any) -> np.ndarray:
        """Boolean array: is each held-out true edge's endpoint in the predicted neighbor set."""
        return np.array([(1.0 - self.P[int(i), int(j)]) <= self.tau for i, j in edges], dtype=bool)

    def set_sizes(self, nodes: Any = None) -> np.ndarray:
        """Neighbor-set size per node (defaults to all nodes)."""
        nodes = range(self.P.shape[0]) if nodes is None else nodes
        return np.array([self.neighbor_set(i).size for i in nodes], dtype=int)


class ConformalKnowledgeGraph:
    """Split-conformal completion sets for a knowledge-graph model (any-slot UQ).

    Calibrating on held-out true triples turns the model's completion posterior into a *set* of
    candidate fillers that contains the true one with probability at least ``1 - alpha`` over
    exchangeable held-out triples.  ``slot`` selects which slot is predicted -- ``'tail'`` for
    ``(h, r, ?)``, ``'head'`` for ``(?, r, t)``, ``'relation'`` for ``(h, ?, t)`` -- using the model's
    ``tail_log_posterior`` / ``head_log_posterior`` / ``relation_log_posterior``.  The nonconformity of
    a triple is ``1 - p(true filler)``, so a confident model gives small completion sets and a recommended
    completion carries a coverage guarantee.
    """

    def __init__(self, kg: Any, calibration: Any, *, slot: str = "tail", alpha: float = 0.1) -> None:
        if slot not in ("tail", "head", "relation"):
            raise ValueError("slot must be 'tail', 'head', or 'relation'.")
        self.kg = kg
        self.slot = slot
        self.alpha = float(alpha)
        scores = [1.0 - float(np.exp(self._posterior(h, r, t)[self._truth(h, r, t)])) for h, r, t in calibration]
        self.tau = conformal_quantile(scores, self.alpha)

    def _posterior(self, h: int, r: int, t: int) -> np.ndarray:
        if self.slot == "tail":
            return self.kg.tail_log_posterior(int(h), int(r))
        if self.slot == "head":
            return self.kg.head_log_posterior(int(r), int(t))
        return self.kg.relation_log_posterior(int(h), int(t))

    def _truth(self, h: int, r: int, t: int) -> int:
        return int({"tail": t, "head": h, "relation": r}[self.slot])

    def completion_set(self, h: int | None = None, r: int | None = None, t: int | None = None) -> np.ndarray:
        """Candidate fillers in the conformal set for the missing slot of a query."""
        p = np.exp(self.kg.complete(h=h, r=r, t=t))
        return np.flatnonzero((1.0 - p) <= self.tau)

    def covers(self, triples: Any) -> np.ndarray:
        """Boolean array: is each held-out true triple's filler in the completion set."""
        return np.array(
            [(1.0 - np.exp(self._posterior(h, r, t)[self._truth(h, r, t)])) <= self.tau for h, r, t in triples],
            dtype=bool,
        )

    def set_sizes(self, triples: Any) -> np.ndarray:
        """Completion-set size for each query (the slot of each triple is treated as missing)."""
        out = []
        for h, r, t in triples:
            q = {"tail": (h, r, None), "head": (None, r, t), "relation": (h, None, t)}[self.slot]
            out.append(self.completion_set(*q).size)
        return np.array(out, dtype=int)


def conformal(result: Any, y_cal: Any, *, given: dict, alpha: float = 0.1) -> ConformalRegressor:
    """Split-conformal calibration of a fitted regression ``result`` into prediction intervals.

    Mirrors ``fit``'s convention (labels positional, ``given=`` keyword), a one-liner over
    :class:`ConformalRegressor`::

        m = Normal(free * Field("x") + free, free).fit(y_tr, given={"x": x_tr})
        cp = conformal(m.result, y_cal, given={"x": x_cal}, alpha=0.1)
        lo, hi = cp.interval({"x": x_te})
        cp.covers(y_te, given={"x": x_te}).mean()   # ~ 0.9
    """
    return ConformalRegressor(result, y_cal, given=given, alpha=alpha)
