"""Random forests as a conditional leaf in the pysp estimation framework.

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

sklearn provides the forest and is imported lazily inside estimate(), so importing this module -- or pysp.models --
does not require scikit-learn unless a forest is actually fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

LOG_2PI = float(np.log(2.0 * np.pi))


class RandomForestConditionalSampler(DistributionSampler):
    """Sampler for the conditional forest. p(y | x) cannot generate x, so the unconditional sample() is disabled;
    use sample_y(X) to draw targets given features."""

    def sample(self, size: int | None = None, *, batched: bool = True) -> Any:
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
    ) -> None:
        self.forest = forest
        self.task = task
        self.sigma = float(sigma) if sigma is not None else None
        self.sigma2 = self.sigma * self.sigma if self.sigma is not None else None
        self.n_features = n_features
        self.name = name
        self.keys = keys
        if task == "classification":
            self._class_pos = {c: i for i, c in enumerate(forest.classes_)}

    def __str__(self) -> str:
        return "RandomForestConditional(task=%s, n_features=%s, name=%s)" % (
            self.task,
            repr(self.n_features),
            repr(self.name),
        )

    def density(self, x: tuple[Any, Any]) -> float:
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: tuple[Any, Any]) -> float:
        feat, target = x
        return float(self.seq_log_density((np.asarray([np.asarray(feat, dtype=float)]), np.asarray([target])))[0])

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        X, y = x
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
        X = np.asarray(x, dtype=float)
        if self.task == "classification":
            proba = np.asarray(self.forest.predict_proba(X))
            classes = self.forest.classes_
            return np.array([classes[rng.choice(len(classes), p=p)] for p in proba])
        mu = np.asarray(self.forest.predict(X), dtype=float)
        return mu + rng.normal(0.0, self.sigma, size=mu.shape)

    def sampler(self, seed: int | None = None) -> RandomForestConditionalSampler:
        return RandomForestConditionalSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> RandomForestEstimator:
        return RandomForestEstimator(task=self.task, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> RandomForestEncoder:
        return RandomForestEncoder()


class RandomForestAccumulator(SequenceEncodableStatisticAccumulator):
    """Buffers the weighted (x, y) design matrix; combine() concatenates partition buffers into the full training
    set that estimate() fits the forest on."""

    def __init__(self, keys: str | None = None, name: str | None = None) -> None:
        self.keys = keys
        self.name = name
        self._X: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._w: list[np.ndarray] = []

    def update(self, x: tuple[Any, Any], weight: float, estimate: RandomForestConditional | None) -> None:
        feat, target = x
        self._X.append(np.asarray([np.asarray(feat, dtype=float)]))
        self._y.append(np.asarray([target]))
        self._w.append(np.asarray([weight], dtype=float))

    def initialize(self, x: tuple[Any, Any], weight: float, rng: np.random.RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(
        self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, estimate: RandomForestConditional | None
    ) -> None:
        X, y = x
        if len(y) == 0:
            return
        self._X.append(np.asarray(X, dtype=float))
        self._y.append(np.asarray(y))
        self._w.append(np.asarray(weights, dtype=float))

    def seq_initialize(self, x: tuple[np.ndarray, np.ndarray], weights: np.ndarray, rng: Any) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> RandomForestAccumulator:
        if suff_stat is not None:
            X, y, w = suff_stat
            if len(y) > 0:
                self._X.append(np.asarray(X, dtype=float))
                self._y.append(np.asarray(y))
                self._w.append(np.asarray(w, dtype=float))
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not self._y:
            return None
        return (np.concatenate(self._X, axis=0), np.concatenate(self._y), np.concatenate(self._w))

    def from_value(self, x: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> RandomForestAccumulator:
        if x is None:
            self._X, self._y, self._w = [], [], []
        else:
            X, y, w = x
            self._X, self._y, self._w = [np.asarray(X, dtype=float)], [np.asarray(y)], [np.asarray(w, dtype=float)]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                self.combine(stats_dict[self.keys])
            stats_dict[self.keys] = self.value()

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys])

    def acc_to_encoder(self) -> RandomForestEncoder:
        return RandomForestEncoder()


class RandomForestAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> RandomForestAccumulator:
        return RandomForestAccumulator(name=self.name, keys=self.keys)


class RandomForestEstimator(ParameterEstimator):
    """Estimator that fits a scikit-learn random forest as a conditional leaf.

    task is 'classification', 'regression', or 'auto' (inferred from the dtype of y). Extra keyword arguments are
    forwarded to the sklearn forest constructor (n_estimators, max_depth, random_state, ...). estimate() trains in
    one pass on the accumulated weighted data; there is no EM iteration, so drive it with optimize(max_its=1) or
    call the seq_encode / accumulate / estimate path directly.
    """

    def __init__(
        self,
        task: str = "auto",
        min_sigma: float = 1.0e-3,
        name: str | None = None,
        keys: str | None = None,
        **forest_kwargs: Any,
    ) -> None:
        self.task = task
        self.min_sigma = float(min_sigma)
        self.name = name
        self.keys = keys
        self.forest_kwargs = forest_kwargs

    def accumulator_factory(self) -> RandomForestAccumulatorFactory:
        return RandomForestAccumulatorFactory(self.name, self.keys)

    def _resolve_task(self, y: np.ndarray) -> str:
        if self.task != "auto":
            return self.task
        return "regression" if np.asarray(y).dtype.kind == "f" else "classification"

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    ) -> RandomForestConditional:
        if suff_stat is None or len(suff_stat[1]) == 0:
            raise ValueError("RandomForestEstimator.estimate requires at least one (x, y) observation.")
        X, y, w = suff_stat
        X = np.asarray(X, dtype=float)
        task = self._resolve_task(y)

        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier

            forest = RandomForestClassifier(**self.forest_kwargs)
            forest.fit(X, y, sample_weight=w)
            return RandomForestConditional(
                forest, "classification", n_features=X.shape[1], name=self.name, keys=self.keys
            )

        from sklearn.ensemble import RandomForestRegressor

        y = np.asarray(y, dtype=float)
        forest = RandomForestRegressor(**self.forest_kwargs)
        forest.fit(X, y, sample_weight=w)
        resid = y - forest.predict(X)
        wsum = float(np.sum(w))
        var = float(np.sum(w * resid * resid) / wsum) if wsum > 0 else float(np.mean(resid * resid))
        sigma = max(np.sqrt(var), self.min_sigma)
        return RandomForestConditional(
            forest, "regression", sigma=sigma, n_features=X.shape[1], name=self.name, keys=self.keys
        )


class RandomForestEncoder(DataSequenceEncoder):
    """Encodes a sequence of (x, y) observations into a (design-matrix, target-vector) pair."""

    def __str__(self) -> str:
        return "RandomForestEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RandomForestEncoder)

    def seq_encode(self, x: list[tuple[Any, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return (np.zeros((0, 0)), np.zeros(0))
        X = np.asarray([np.asarray(feat, dtype=float) for feat, _ in x], dtype=float)
        y = np.asarray([target for _, target in x])
        return (X, y)
