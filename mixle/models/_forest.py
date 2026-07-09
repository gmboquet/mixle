"""Native numpy random forest (CART trees + bootstrap bagging) used by mixle.models.random_forest.

Pure numpy so mixle carries no scikit-learn dependency. Supports weighted observations (needed for EM
responsibilities), classification (Gini impurity, averaged leaf class distributions) and regression (weighted
variance reduction, averaged leaf means), per-split random feature subsets, and bootstrap resampling per tree.
Split finding is the standard sorted-cumulative-statistic scan: O(n log n) per feature per node.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_MAX_DEPTH = 1 << 20


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=-1, threshold=0.0, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf payload: class-probability vector (classification) or [mean] (regression)


def _resolve_max_features(max_features: Any, n_features: int, task: str) -> int:
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt" or (max_features == "auto" and task == "classification"):
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if max_features == "auto":  # regression default: one third of the features
            return max(1, n_features // 3)
        raise ValueError("unknown max_features %r" % (max_features,))
    if isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    return max(1, min(int(max_features), n_features))


class _DecisionTree:
    """A single weighted CART tree."""

    def __init__(self, task, n_classes, max_depth, min_samples_split, min_samples_leaf, max_features, rng):
        self.task = task
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng
        self.root: _Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> _DecisionTree:
        self.root = self._build(X, y, w, 0)
        return self

    def _leaf(self, y: np.ndarray, w: np.ndarray) -> _Node:
        if self.task == "classification":
            v = np.bincount(y, weights=w, minlength=self.n_classes).astype(float)
            s = v.sum()
            v = v / s if s > 0 else np.full(self.n_classes, 1.0 / self.n_classes)
            return _Node(value=v)
        ws = w.sum()
        mean = float(np.dot(w, y) / ws) if ws > 0 else float(np.mean(y))
        return _Node(value=np.array([mean]))

    def _build(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int) -> _Node:
        n = len(y)
        pure = (self.task == "classification" and len(np.unique(y)) <= 1) or (
            self.task == "regression" and np.ptp(y) == 0.0
        )
        # an all-zero-weight node (e.g. EM responsibilities -> 0) gives 0/0 NaN gains in _best_split; stop here
        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or n < 2 * self.min_samples_leaf
            or pure
            or w.sum() <= 0.0
        ):
            return self._leaf(y, w)

        feat, thr = self._best_split(X, y, w)
        if feat < 0:
            return self._leaf(y, w)
        mask = X[:, feat] <= thr
        left = self._build(X[mask], y[mask], w[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], w[~mask], depth + 1)
        return _Node(feature=feat, threshold=thr, left=left, right=right)

    def _best_split(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[int, float]:
        n, d = X.shape
        n_try = min(self.max_features, d)
        feats = self.rng.permutation(d)[:n_try]
        best_gain = 0.0
        best_feat = -1
        best_thr = 0.0
        ml = self.min_samples_leaf

        for f in feats:
            xs_full = X[:, f]
            order = np.argsort(xs_full, kind="mergesort")
            xs = xs_full[order]
            ws = w[order]
            distinct = xs[:-1] < xs[1:]  # only split between different feature values
            if not distinct.any():
                continue
            # left side after position i contains rows 0..i (i+1 rows); enforce min_samples_leaf by count
            pos = np.arange(n - 1)
            valid = distinct & (pos + 1 >= ml) & (n - pos - 1 >= ml)
            if not valid.any():
                continue

            if self.task == "classification":
                ys = y[order]
                onehot = np.zeros((n, self.n_classes))
                onehot[np.arange(n), ys] = ws
                cumW = np.cumsum(onehot, axis=0)
                totW = cumW[-1]
                cl = cumW[:-1]  # (n-1, K) left class weights
                cr = totW - cl
                wl = cl.sum(1)
                wr = cr.sum(1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    gl = 1.0 - (cl * cl).sum(1) / np.where(wl > 0, wl * wl, 1.0)
                    gr = 1.0 - (cr * cr).sum(1) / np.where(wr > 0, wr * wr, 1.0)
                parent = 1.0 - (totW * totW).sum() / (totW.sum() ** 2)
                child = (wl * gl + wr * gr) / (wl + wr)
            else:
                ys = y[order]
                sw = np.cumsum(ws)
                swy = np.cumsum(ws * ys)
                swyy = np.cumsum(ws * ys * ys)
                totW, totWY, totWYY = sw[-1], swy[-1], swyy[-1]
                wl = sw[:-1]
                wyl = swy[:-1]
                wyyl = swyy[:-1]
                wr = totW - wl
                wyr = totWY - wyl
                wyyr = totWYY - wyyl
                with np.errstate(divide="ignore", invalid="ignore"):
                    sse_l = wyyl - wyl * wyl / np.where(wl > 0, wl, 1.0)
                    sse_r = wyyr - wyr * wyr / np.where(wr > 0, wr, 1.0)
                parent = totWYY - totWY * totWY / totW
                child = sse_l + sse_r

            gain = parent - child
            gain = np.where(valid, gain, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                best_gain = float(gain[i])
                best_feat = int(f)
                best_thr = float((xs[i] + xs[i + 1]) / 2.0)

        return best_feat, best_thr

    def _apply(self, X: np.ndarray, out: np.ndarray) -> None:
        def rec(node: _Node, idx: np.ndarray) -> None:
            if node.left is None:
                out[idx] = node.value
                return
            go_left = X[idx, node.feature] <= node.threshold
            rec(node.left, idx[go_left])
            rec(node.right, idx[~go_left])

        rec(self.root, np.arange(len(X)))


class NativeRandomForest:
    """Bootstrap-bagged ensemble of weighted CART trees with a scikit-learn-like predict interface."""

    def __init__(
        self,
        task: str,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = "auto",
        random_state: int | None = None,
    ) -> None:
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = _MAX_DEPTH if max_depth is None else max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees: list[_DecisionTree] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> NativeRandomForest:
        """Fit the bagged tree ensemble from a weighted design matrix and target vector."""
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        rng = np.random.RandomState(self.random_state)
        mf = _resolve_max_features(self.max_features, d, self.task)

        if self.task == "classification":
            self.classes_, codes = np.unique(y, return_inverse=True)
            n_classes = len(self.classes_)
            yfit = codes
        else:
            n_classes = 0
            yfit = np.asarray(y, dtype=float)

        self.trees = []
        for _ in range(self.n_estimators):
            tree_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
            boot = tree_rng.randint(0, n, n)  # bootstrap sample (with replacement)
            tree = _DecisionTree(
                self.task, n_classes, self.max_depth, self.min_samples_split, self.min_samples_leaf, mf, tree_rng
            )
            tree.fit(X[boot], yfit[boot], w[boot])
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities averaged over all fitted trees."""
        X = np.asarray(X, dtype=float)
        acc = np.zeros((len(X), len(self.classes_)))
        buf = np.zeros((len(X), len(self.classes_)))
        for tree in self.trees:
            buf.fill(0.0)
            tree._apply(X, buf)
            acc += buf
        return acc / len(self.trees)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return log class probabilities, preserving ``-inf`` for impossible classes."""
        with np.errstate(divide="ignore"):
            return np.log(self.predict_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels for classification or mean predictions for regression."""
        X = np.asarray(X, dtype=float)
        if self.task == "classification":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        acc = np.zeros((len(X), 1))
        buf = np.zeros((len(X), 1))
        for tree in self.trees:
            buf.fill(0.0)
            tree._apply(X, buf)
            acc += buf
        return (acc / len(self.trees)).ravel()
