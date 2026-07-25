"""Random forests as a conditional leaf in the mixle estimation framework.

A random forest is discriminative and is not fit by accumulating additive sufficient statistics or by EM, so it
does not look like the exponential-family leaves. It still fits the estimation contract cleanly if we treat it as
a *conditional* distribution p(y | x): the observation is a pair (x, y), the accumulator's "sufficient statistic"
is the buffered weighted design matrix, combine() concatenates the per-partition buffers (the map-reduce step is
the data shuffle), and estimate() trains the forest in a single non-EM pass over that buffer.

The result is a SequenceEncodableProbabilityDistribution whose seq_log_density returns log p(y | x), so a fitted
forest composes with seq_encode / seq_log_density / the top-level log_density helper, and can sit in a slot of a
composite/record model or act as a mixture-of-experts component. Because estimate() refits from scratch, run it
through optimize(..., max_its=1) (there is no likelihood for EM to iterate); for classification log-density is the
forest's predict_log_proba, for regression it is a Gaussian residual model with a globally estimated noise scale.

The forest itself is a native numpy CART + bagging ensemble (mixle.models._forest), so mixle carries no
scikit-learn dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.models._forest import NativeRandomForest
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

LOG_2PI = float(np.log(2.0 * np.pi))
_VALID_TASKS = frozenset({"classification", "regression"})


def _validated_features(x: Any, *, n_features: int | None = None) -> np.ndarray:
    try:
        X = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be a numeric two-dimensional matrix") from exc
    if X.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if X.shape[1] == 0:
        raise ValueError("features must contain at least one column")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"features must contain exactly {n_features} columns, got {X.shape[1]}")
    if not np.all(np.isfinite(X)):
        raise ValueError("features must contain only finite values")
    return X


def _validated_weights(weights: Any, n_rows: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != n_rows:
        raise ValueError(f"weights must be a one-dimensional vector with {n_rows} rows")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must contain only finite, non-negative values")
    return w


class RandomForestConditionalSampler(DistributionSampler):
    """Sampler for the conditional forest. p(y | x) cannot generate x, so the unconditional sample() is disabled;
    use sample_y(X) to draw targets given features."""

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
        """Raise because the conditional forest has no marginal model for ``x``."""
        raise NotImplementedError(
            "RandomForestConditional models p(y | x) and cannot generate x. Use sample_y(X) to draw y given x."
        )

    def sample_y(self, x: Any) -> np.ndarray:
        """Draw a target for each row of x: a class from predict_proba (classification) or mean+Gaussian-noise
        (regression)."""
        return self.dist.sample_y(x, self.rng)


class RandomForestConditional(SequenceEncodableProbabilityDistribution):
    """Fitted random forest viewed as a conditional distribution p(y | x).

    Observations are (x, y) pairs: x is a feature vector and y is a class label (classification) or a real target
    (regression). seq_log_density returns log p(y | x) -- predict_log_proba for classification, a Gaussian residual
    density with scale sigma for regression.
    """

    def __init__(
        self,
        forest: Any,
        task: str,
        sigma: float | None = None,
        n_features: int | None = None,
        name: str | None = None,
        keys: str | None = None,
        forest_spec: dict[str, Any] | None = None,
    ) -> None:
        if task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(_VALID_TASKS)}, got {task!r}")
        if not getattr(forest, "trees", None):
            raise ValueError("forest must be fitted and contain at least one tree")
        if n_features is None:
            n_features = getattr(forest, "n_features_in_", None)
        if isinstance(n_features, bool) or not isinstance(n_features, (int, np.integer)) or int(n_features) <= 0:
            raise ValueError("n_features must be a positive integer")
        if task == "regression" and (
            sigma is None or not np.isfinite(float(sigma)) or float(sigma) <= 0.0
        ):
            raise ValueError("regression sigma must be finite and positive")
        self.forest = forest
        self.task = task
        self.sigma = float(sigma) if sigma is not None else None
        self.sigma2 = self.sigma * self.sigma if self.sigma is not None else None
        self.n_features = int(n_features)
        self.name = name
        self.keys = keys
        if forest_spec is None:
            forest_spec = {
                "n_estimators": forest.n_estimators,
                "max_depth": forest.max_depth,
                "min_samples_split": forest.min_samples_split,
                "min_samples_leaf": forest.min_samples_leaf,
                "max_features": forest.max_features,
                "random_state": forest.random_state,
                "min_sigma": self.sigma if self.sigma is not None else 1.0e-3,
                "n_features": self.n_features,
            }
        self.forest_spec = dict(forest_spec)
        if task == "classification":
            self._class_pos = {c: i for i, c in enumerate(forest.classes_)}

    def __str__(self) -> str:
        return "RandomForestConditional(task=%s, n_features=%s, name=%s)" % (
            self.task,
            repr(self.n_features),
            repr(self.name),
        )

    def density(self, x: tuple[Any, Any]) -> float:
        """Return ``p(y | x)`` for one feature/target pair."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: tuple[Any, Any]) -> float:
        """Return ``log p(y | x)`` for one feature/target pair."""
        feat, target = x
        return float(self.seq_log_density((np.asarray([np.asarray(feat, dtype=float)]), np.asarray([target])))[0])

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return per-row conditional log densities for encoded ``(X, y)`` data."""
        X, y = x
        X = _validated_features(X, n_features=self.n_features)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"targets must be a one-dimensional vector with {len(X)} rows")
        if len(y) == 0:
            return np.zeros(0)
        if self.task == "classification":
            with np.errstate(divide="ignore"):
                # a forest leaf with no examples of a class gives proba 0 -> log p = -inf, which is correct
                logp = np.asarray(self.forest.predict_log_proba(X))
            cols = np.fromiter((self._class_pos.get(yi, -1) for yi in y), dtype=int, count=len(y))
            out = np.full(len(y), -np.inf)
            seen = cols >= 0
            rows = np.arange(len(y))[seen]
            out[seen] = logp[rows, cols[seen]]
            return out
        mu = np.asarray(self.forest.predict(X), dtype=float)
        resid = np.asarray(y, dtype=float) - mu
        return -0.5 * LOG_2PI - 0.5 * np.log(self.sigma2) - 0.5 * resid * resid / self.sigma2

    def sample_y(self, x: Any, rng: np.random.RandomState) -> np.ndarray:
        """Draw target values from the fitted conditional forest at feature rows ``x``."""
        X = _validated_features(x, n_features=self.n_features)
        if self.task == "classification":
            proba = np.asarray(self.forest.predict_proba(X))
            classes = self.forest.classes_
            return np.array([classes[rng.choice(len(classes), p=p)] for p in proba])
        mu = np.asarray(self.forest.predict(X), dtype=float)
        return mu + rng.normal(0.0, self.sigma, size=mu.shape)

    def sampler(self, seed: int | None = None) -> RandomForestConditionalSampler:
        """Return a conditional sampler for drawing targets given features."""
        return RandomForestConditionalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> RandomForestEstimator:
        """Return an estimator preserving the fitted model's complete specification."""
        return RandomForestEstimator(task=self.task, name=self.name, keys=self.keys, **self.forest_spec)

    def dist_to_encoder(self) -> RandomForestEncoder:
        """Return the encoder for feature/target observation pairs."""
        return RandomForestEncoder(self.n_features)


class RandomForestAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffers the weighted (x, y) design matrix; combine() concatenates partition buffers into the full training
    set that estimate() fits the forest on."""

    def __init__(
        self, keys: str | None = None, name: str | None = None, n_features: int | None = None
    ) -> None:
        self.keys = keys
        self.name = name
        self.n_features = n_features
        self._X: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._w: list[np.ndarray] = []

    def update(self, x: tuple[Any, Any], weight: float, estimate: RandomForestConditional | None) -> None:
        """Add one weighted feature/target observation to the training buffer."""
        feat, target = x
        X = _validated_features([feat], n_features=self.n_features)
        if self.n_features is None:
            self.n_features = X.shape[1]
        w = _validated_weights([weight], 1)
        self._X.append(X)
        self._y.append(np.asarray([target]))
        self._w.append(w)

    def initialize(self, x: tuple[Any, Any], weight: float, rng: np.random.RandomState | None) -> None:
        """Initialize from one observation using the ordinary update path."""
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: RandomForestConditional | None
    ) -> None:
        """Add an encoded batch and weights to the training buffer."""
        X, y = x
        y = np.asarray(y)
        raw_X = np.asarray(X)
        if y.ndim != 1:
            raise ValueError("targets must be a one-dimensional vector")
        if y.size == 0:
            if raw_X.ndim != 2 or raw_X.shape[0] != 0:
                raise ValueError("empty batches must contain an empty two-dimensional feature matrix and target vector")
            if self.n_features is not None and raw_X.shape[1] != self.n_features:
                raise ValueError(
                    f"features must contain exactly {self.n_features} columns, got {raw_X.shape[1]}"
                )
            _validated_weights(weights, 0)
            return
        X = _validated_features(X, n_features=self.n_features)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"targets must be a one-dimensional vector with {len(X)} rows")
        weights = _validated_weights(weights, len(X))
        if self.n_features is None:
            self.n_features = X.shape[1]
        self._X.append(X)
        self._y.append(y)
        self._w.append(weights)

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: Any) -> None:
        """Initialize from an encoded batch using the ordinary batch update path."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> RandomForestAccumulator:
        """Merge a buffered ``(X, y, weights)`` tuple from another accumulator."""
        if suff_stat is not None:
            X, y, w = suff_stat
            if len(y) > 0:
                self.seq_update((X, y), w, None)
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return the buffered design matrix, targets, and weights, or ``None`` if empty."""
        if not self._y:
            return None
        return (np.concatenate(self._X, axis=0), np.concatenate(self._y), np.concatenate(self._w))

    def from_value(self, x: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> RandomForestAccumulator:
        """Restore the accumulator from a buffered value tuple."""
        if x is None:
            self._X, self._y, self._w = [], [], []
        else:
            X, y, w = x
            self._X, self._y, self._w = [], [], []
            self.seq_update((X, y), w, None)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under ``keys`` when keyed accumulation is enabled."""
        if self.keys is not None:
            if self.keys in stats_dict:
                self.combine(stats_dict[self.keys])
            stats_dict[self.keys] = self.value()

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator from ``stats_dict`` under ``keys`` when present."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys])

    def acc_to_encoder(self) -> RandomForestEncoder:
        """Return the encoder expected by this accumulator."""
        return RandomForestEncoder(self.n_features)


class RandomForestAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for random-forest accumulators."""

    def __init__(
        self, name: str | None = None, keys: str | None = None, n_features: int | None = None
    ) -> None:
        self.name = name
        self.keys = keys
        self.n_features = n_features

    def make(self) -> RandomForestAccumulator:
        """Create a fresh random-forest accumulator."""
        return RandomForestAccumulator(name=self.name, keys=self.keys, n_features=self.n_features)


class RandomForestEstimator(ParameterEstimator):
    """Estimator that fits a native (numpy) random forest as a conditional leaf.

    task is explicitly 'classification' or 'regression'; target dtype is never used to guess semantics. The forest
    hyperparameters
    (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, random_state) are passed straight
    to the native ensemble. estimate() trains in one pass on the accumulated weighted data; there is no EM
    iteration, so drive it with optimize(max_its=1) or call the seq_encode / accumulate / estimate path directly.
    """

    def __init__(
        self,
        task: str | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = "auto",
        random_state: int | None = None,
        min_sigma: float = 1.0e-3,
        name: str | None = None,
        keys: str | None = None,
        n_features: int | None = None,
    ) -> None:
        if task not in _VALID_TASKS:
            raise ValueError(
                f"task must be explicitly one of {sorted(_VALID_TASKS)}; automatic dtype inference is unsafe"
            )
        validated = NativeRandomForest(
            task=task,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )
        if not np.isfinite(float(min_sigma)) or float(min_sigma) <= 0.0:
            raise ValueError("min_sigma must be finite and positive")
        if n_features is not None and (
            isinstance(n_features, bool)
            or not isinstance(n_features, (int, np.integer))
            or int(n_features) <= 0
        ):
            raise ValueError("n_features must be a positive integer or None")
        self.task = task
        self.n_estimators = validated.n_estimators
        self.max_depth = validated.max_depth
        self.min_samples_split = validated.min_samples_split
        self.min_samples_leaf = validated.min_samples_leaf
        self.max_features = validated.max_features
        self.random_state = validated.random_state
        self.min_sigma = float(min_sigma)
        self.name = name
        self.keys = keys
        self.n_features = None if n_features is None else int(n_features)

    def accumulator_factory(self) -> RandomForestAccumulatorFactory:
        """Return an accumulator factory for weighted feature/target buffers."""
        return RandomForestAccumulatorFactory(self.name, self.keys, self.n_features)

    def _forest_spec(self, n_features: int) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "min_sigma": self.min_sigma,
            "n_features": n_features,
        }

    def _make_forest(self, task: str) -> NativeRandomForest:
        return NativeRandomForest(
            task=task,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    ) -> RandomForestConditional:
        """Fit the native forest from buffered data and return it as a conditional leaf."""
        if suff_stat is None:
            raise ValueError("RandomForestEstimator.estimate requires at least one (x, y) observation.")
        if not isinstance(suff_stat, tuple) or len(suff_stat) != 3:
            raise ValueError("random-forest sufficient statistics must be an (X, y, weights) tuple")
        X, y, w = suff_stat
        X = _validated_features(X, n_features=self.n_features)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"targets must be a one-dimensional vector with {len(X)} rows")
        if len(y) == 0:
            raise ValueError("RandomForestEstimator.estimate requires at least one (x, y) observation.")
        w = _validated_weights(w, len(X))
        if not np.any(w > 0.0):
            raise ValueError("weights must contain at least one positive value")
        spec = self._forest_spec(X.shape[1])

        if self.task == "classification":
            forest = self._make_forest("classification").fit(X, y, sample_weight=w)
            return RandomForestConditional(
                forest,
                "classification",
                n_features=X.shape[1],
                name=self.name,
                keys=self.keys,
                forest_spec=spec,
            )

        try:
            y = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("regression targets must be numeric") from exc
        if not np.all(np.isfinite(y)):
            raise ValueError("regression targets must be finite")
        forest = self._make_forest("regression").fit(X, y, sample_weight=w)
        oob_prediction = np.asarray(forest.oob_prediction_, dtype=float)
        calibrated = np.isfinite(oob_prediction) & (w > 0.0)
        if np.count_nonzero(calibrated) < 2:
            raise ValueError(
                "regression uncertainty calibration requires at least two positive-weight observations "
                "with out-of-bag predictions; increase n_estimators or provide more data"
            )
        resid = y[calibrated] - oob_prediction[calibrated]
        calibration_weights = w[calibrated]
        var = float(np.sum(calibration_weights * resid * resid) / np.sum(calibration_weights))
        sigma = max(np.sqrt(var), self.min_sigma)
        return RandomForestConditional(
            forest,
            "regression",
            sigma=sigma,
            n_features=X.shape[1],
            name=self.name,
            keys=self.keys,
            forest_spec=spec,
        )


class RandomForestEncoder(DataSequenceEncoder):
    """Encodes a sequence of (x, y) observations into a (design-matrix, target-vector) pair."""

    def __init__(self, n_features: int | None = None) -> None:
        self.n_features = n_features

    def __str__(self) -> str:
        return f"RandomForestEncoder(n_features={self.n_features!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RandomForestEncoder) and self.n_features == other.n_features

    def seq_encode(self, x: list[tuple[Any, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Convert feature/target pairs into a design matrix and target vector."""
        if len(x) == 0:
            return (np.zeros((0, self.n_features or 0)), np.zeros(0))
        X = _validated_features([feat for feat, _ in x], n_features=self.n_features)
        y = np.asarray([target for _, target in x])
        return (X, y)
