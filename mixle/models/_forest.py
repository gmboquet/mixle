"""Native numpy random forest (CART trees + bootstrap bagging) used by mixle.models.random_forest.

Pure numpy so mixle carries no scikit-learn dependency. Supports weighted observations (needed for EM
responsibilities), classification (Gini impurity, averaged leaf class distributions) and regression (weighted
variance reduction, averaged leaf means), per-split random feature subsets, and bootstrap resampling per tree.
Split finding is the standard sorted-cumulative-statistic scan: O(n log n) per feature per node.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_VALID_TASKS = frozenset({"classification", "regression"})


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=-1, threshold=0.0, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf payload: class-probability vector (classification) or [mean] (regression)


def _resolve_max_features(max_features: Any, n_features: int, task: str) -> int:
    if n_features <= 0:
        raise ValueError("n_features must be positive")
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
        if not np.isfinite(max_features) or not 0.0 < max_features <= 1.0:
            raise ValueError("float max_features must be finite and in (0, 1]")
        return max(1, int(max_features * n_features))
    if isinstance(max_features, (bool, np.bool_)) or not isinstance(max_features, (int, np.integer)):
        raise TypeError("max_features must be None, a supported string, a float in (0, 1], or a positive integer")
    if int(max_features) <= 0:
        raise ValueError("integer max_features must be positive")
    return min(int(max_features), n_features)


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validated_design(X: Any, *, n_features: int | None = None) -> np.ndarray:
    try:
        design = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a numeric two-dimensional design matrix") from exc
    if design.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix")
    if design.shape[1] == 0:
        raise ValueError("X must contain at least one feature")
    if n_features is not None and design.shape[1] != n_features:
        raise ValueError(f"X must contain exactly {n_features} features, got {design.shape[1]}")
    if not np.all(np.isfinite(design)):
        raise ValueError("X must contain only finite feature values")
    return design


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
        stack: list[tuple[_Node | None, str | None, np.ndarray, np.ndarray, np.ndarray, int]] = [
            (None, None, X, y, w, 0)
        ]
        while stack:
            parent, side, node_X, node_y, node_w, depth = stack.pop()
            n = len(node_y)
            pure = (self.task == "classification" and len(np.unique(node_y)) <= 1) or (
                self.task == "regression" and np.ptp(node_y) == 0.0
            )
            if (
                depth >= self.max_depth
                or n < self.min_samples_split
                or n < 2 * self.min_samples_leaf
                or pure
                or node_w.sum() <= 0.0
            ):
                node = self._leaf(node_y, node_w)
            else:
                feature, threshold = self._best_split(node_X, node_y, node_w)
                if feature < 0:
                    node = self._leaf(node_y, node_w)
                else:
                    node = _Node(feature=feature, threshold=threshold, left=_Node(), right=_Node())
                    mask = node_X[:, feature] <= threshold
                    stack.append((node, "right", node_X[~mask], node_y[~mask], node_w[~mask], depth + 1))
                    stack.append((node, "left", node_X[mask], node_y[mask], node_w[mask], depth + 1))
            if parent is None:
                self.root = node
            else:
                setattr(parent, side, node)
        if self.root is None:  # defensive: validated fitting always visits at least one node
            raise RuntimeError("tree fitting did not produce a root node")
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
        if self.root is None:
            raise RuntimeError("tree must be fitted before prediction")
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if node.left is None:
                out[idx] = node.value
                continue
            go_left = X[idx, node.feature] <= node.threshold
            stack.append((node.right, idx[~go_left]))
            stack.append((node.left, idx[go_left]))


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
        if task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(_VALID_TASKS)}, got {task!r}")
        self.task = task
        self.n_estimators = _positive_int(n_estimators, "n_estimators")
        self.max_depth = None if max_depth is None else _positive_int(max_depth, "max_depth")
        self.min_samples_split = _positive_int(min_samples_split, "min_samples_split", minimum=2)
        self.min_samples_leaf = _positive_int(min_samples_leaf, "min_samples_leaf")
        _resolve_max_features(max_features, 1, task)
        self.max_features = max_features
        if random_state is not None and (
            isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer))
        ):
            raise TypeError("random_state must be an integer or None")
        self.random_state = random_state
        self.trees: list[_DecisionTree] = []
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.oob_prediction_: np.ndarray | None = None
        self.oob_count_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> NativeRandomForest:
        """Fit the bagged tree ensemble from a weighted design matrix and target vector."""
        X = _validated_design(X)
        n, d = X.shape
        if n == 0:
            raise ValueError("X and y must contain at least one observation")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != n:
            raise ValueError(f"y must be a one-dimensional vector with {n} rows")
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if w.ndim != 1 or len(w) != n:
            raise ValueError(f"sample_weight must be a one-dimensional vector with {n} rows")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("sample_weight must contain only finite, non-negative values")
        if not np.any(w > 0.0):
            raise ValueError("sample_weight must contain at least one positive value")
        rng = np.random.RandomState(self.random_state)
        mf = _resolve_max_features(self.max_features, d, self.task)
        max_depth = n if self.max_depth is None else self.max_depth

        if self.task == "classification":
            if y.dtype.kind in {"f", "c"} and not np.all(np.isfinite(y)):
                raise ValueError("classification labels must be finite")
            if y.dtype.kind == "O" and any(label is None for label in y):
                raise ValueError("classification labels must not be None")
            try:
                self.classes_, codes = np.unique(y, return_inverse=True)
            except TypeError as exc:
                raise ValueError("classification labels must be mutually comparable scalar values") from exc
            n_classes = len(self.classes_)
            yfit = codes
        else:
            n_classes = 0
            try:
                yfit = np.asarray(y, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("regression targets must be numeric") from exc
            if not np.all(np.isfinite(yfit)):
                raise ValueError("regression targets must be finite")

        self.trees = []
        oob_shape = (n, n_classes) if self.task == "classification" else (n, 1)
        oob_sum = np.zeros(oob_shape, dtype=float)
        oob_count = np.zeros(n, dtype=int)
        for _ in range(self.n_estimators):
            tree_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
            boot = tree_rng.randint(0, n, n)  # bootstrap sample (with replacement)
            tree = _DecisionTree(
                self.task, n_classes, max_depth, self.min_samples_split, self.min_samples_leaf, mf, tree_rng
            )
            tree.fit(X[boot], yfit[boot], w[boot])
            self.trees.append(tree)
            in_bag = np.zeros(n, dtype=bool)
            in_bag[boot] = True
            oob_rows = np.flatnonzero(~in_bag)
            if len(oob_rows):
                tree_out = np.zeros((len(oob_rows), oob_shape[1]), dtype=float)
                tree._apply(X[oob_rows], tree_out)
                oob_sum[oob_rows] += tree_out
                oob_count[oob_rows] += 1
        self.n_features_in_ = d
        self.oob_count_ = oob_count
        self.oob_prediction_ = np.full(oob_shape, np.nan, dtype=float)
        covered = oob_count > 0
        self.oob_prediction_[covered] = oob_sum[covered] / oob_count[covered, None]
        if self.task == "regression":
            self.oob_prediction_ = self.oob_prediction_.ravel()
        return self

    def _prediction_design(self, X: Any) -> np.ndarray:
        if not self.trees or self.n_features_in_ is None:
            raise RuntimeError("forest must be fitted before prediction")
        return _validated_design(X, n_features=self.n_features_in_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities averaged over all fitted trees."""
        if self.task != "classification":
            raise ValueError("predict_proba is available only for classification forests")
        X = self._prediction_design(X)
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
        X = self._prediction_design(X)
        if self.task == "classification":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        acc = np.zeros((len(X), 1))
        buf = np.zeros((len(X), 1))
        for tree in self.trees:
            buf.fill(0.0)
            tree._apply(X, buf)
            acc += buf
        return (acc / len(self.trees)).ravel()
