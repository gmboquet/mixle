"""Generic Fisher-geometry views for pysp distributions.

The classes here expose a common sufficient-statistic and Fisher-vector
interface without requiring every distribution to duplicate plumbing.  The
generic view is accumulator-backed: for ordinary models it returns observed
sufficient statistics, while latent models use the existing update/seq_update
E-step to return posterior-expected complete-data sufficient statistics.

Specialized distributions can later override to_fisher() with faster or
more canonical views, but this default gives every stats/bstats model a useful
and vectorizable baseline.

For latent-variable models, `fisher_information()` may expose a model or
complete-data Fisher metric supplied by the view.  When comparing observed
data, use `observed_fisher_information()` / `observed_fisher_vectors()`: these
center posterior-expected complete-data statistics into observed score vectors
and use their observed covariance as the metric.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

Path = tuple[str, ...]


class SufficientStatisticVectorizer:
    """Flatten nested sufficient-statistic structures into numeric vectors.

    Accumulators in pysp return tuples, arrays, dictionaries, and scalars.  This
    vectorizer learns a stable union schema from a collection of such structures
    and then maps each structure into the same numeric coordinate system.
    """

    def __init__(self, labels: Sequence[Path] | None = None) -> None:
        self.labels: list[Path] = list(labels) if labels is not None else []
        self._index = {k: i for i, k in enumerate(self.labels)}

    @staticmethod
    def _key_label(key: Any) -> str:
        return repr(key)

    @classmethod
    def _items(cls, value: Any, path: Path = ()) -> Iterable[tuple[Path, float]]:
        if value is None:
            return

        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
            if arr.dtype == object:
                for i, v in enumerate(arr.flat):
                    yield from cls._items(v, path + (str(i),))
            else:
                flat = arr.astype(np.float64, copy=False).ravel()
                for i, v in enumerate(flat):
                    yield path + (str(i),), float(v)
            return

        if isinstance(value, dict):
            for k in sorted(value.keys(), key=repr):
                yield from cls._items(value[k], path + (cls._key_label(k),))
            return

        if isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                yield from cls._items(v, path + (str(i),))
            return

        try:
            yield (path if path else ("value",)), float(value)
        except (TypeError, ValueError):
            return

    def fit(self, values: Sequence[Any]) -> SufficientStatisticVectorizer:
        labels = []
        seen = set()
        for value in values:
            for label, _ in self._items(value):
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
        self.labels = labels
        self._index = {k: i for i, k in enumerate(labels)}
        return self

    def partial_fit(self, values: Sequence[Any]) -> SufficientStatisticVectorizer:
        for value in values:
            for label, _ in self._items(value):
                if label not in self._index:
                    self._index[label] = len(self.labels)
                    self.labels.append(label)
        return self

    def transform(self, values: Sequence[Any], extend: bool = False) -> np.ndarray:
        if extend:
            self.partial_fit(values)

        mat = np.zeros((len(values), len(self.labels)), dtype=np.float64)
        for i, value in enumerate(values):
            for label, v in self._items(value):
                j = self._index.get(label)
                if j is not None:
                    mat[i, j] = v
        return mat

    def fit_transform(self, values: Sequence[Any]) -> np.ndarray:
        rows: list[list[tuple[int, float]]] = []
        labels: list[Path] = []
        index: dict[Path, int] = {}

        for value in values:
            row: list[tuple[int, float]] = []
            for label, v in self._items(value):
                j = index.get(label)
                if j is None:
                    j = len(labels)
                    index[label] = j
                    labels.append(label)
                row.append((j, v))
            rows.append(row)

        self.labels = labels
        self._index = index

        mat = np.zeros((len(rows), len(labels)), dtype=np.float64)
        for i, row in enumerate(rows):
            for j, v in row:
                mat[i, j] = v
        return mat

    def label_strings(self) -> list[str]:
        return [".".join(p) for p in self.labels]


class FisherView:
    """Accumulator-backed Fisher-geometry view of a distribution.

    Args:
        dist: pysp.stats or pysp.bstats distribution.
        estimator: Optional estimator.  When omitted, dist.estimator() is used.

    Notes:
        The generic encoded-data path obtains per-row statistics by replaying
        seq_update with one-hot weights.  That keeps the interface compatible
        with existing encoders, but it is a correctness fallback rather than a
        high-performance implementation.  Important families should override
        to_fisher() with direct seq_expected_statistics kernels.
    """

    def __init__(self, dist: Any, estimator: Any | None = None) -> None:
        self.dist = dist
        self.estimator = estimator if estimator is not None else self._make_estimator(dist)
        self.vectorizer = SufficientStatisticVectorizer()

    @staticmethod
    def _make_estimator(dist: Any) -> Any:
        estimator = dist.estimator()
        if estimator is None:
            raise NotImplementedError("%s does not provide an estimator for Fisher statistics" % type(dist).__name__)
        return estimator

    def _make_accumulator(self) -> Any:
        factory = self.estimator.accumulator_factory()
        return factory.make()

    def _default_estimate(self, estimate: Any | None) -> Any:
        return self.dist if estimate is None else estimate

    def structured_statistics(self, x: Any, estimate: Any | None = None, weight: float = 1.0) -> Any:
        """Structured sufficient stats for one observation.

        For latent-variable distributions, this is the posterior-expected
        complete-data sufficient statistic under estimate (or this view's
        distribution when estimate is omitted).
        """
        acc = self._make_accumulator()
        acc.update(x, weight, self._default_estimate(estimate))
        return acc.value()

    def expected_structured_statistics(self, x: Any, estimate: Any | None = None, weight: float = 1.0) -> Any:
        return self.structured_statistics(x, estimate=estimate, weight=weight)

    def sufficient_statistics(
        self, x: Any, estimate: Any | None = None, vectorizer: SufficientStatisticVectorizer | None = None
    ) -> np.ndarray:
        ss = self.structured_statistics(x, estimate=estimate)
        vec = vectorizer if vectorizer is not None else SufficientStatisticVectorizer().fit([ss])
        return vec.transform([ss])[0]

    def expected_sufficient_statistics(
        self, x: Any, estimate: Any | None = None, vectorizer: SufficientStatisticVectorizer | None = None
    ) -> np.ndarray:
        return self.sufficient_statistics(x, estimate=estimate, vectorizer=vectorizer)

    def _n_encoded(self, enc_data: Any, estimate: Any | None) -> int:
        model = self._default_estimate(estimate)
        if hasattr(model, "seq_log_density"):
            return int(len(model.seq_log_density(enc_data)))
        raise ValueError("encoded statistics require a model with seq_log_density")

    def _encode_data(self, data: Sequence[Any], estimate: Any | None) -> Any | None:
        model = self._default_estimate(estimate)
        try:
            return _seq_encode_model(model, data)
        except NotImplementedError:
            pass
        return None

    def seq_structured_statistics(self, enc_data: Any, estimate: Any | None = None) -> list[Any]:
        """Structured per-row stats from encoded data.

        This generic implementation is intentionally conservative and may be
        slow for large data.  It exists so every encoder-compatible model has a
        correct baseline.
        """
        model = self._default_estimate(estimate)
        n = self._n_encoded(enc_data, model)
        values = []
        for i in range(n):
            weights = np.zeros(n, dtype=np.float64)
            weights[i] = 1.0
            acc = self._make_accumulator()
            acc.seq_update(enc_data, weights, model)
            values.append(acc.value())
        return values

    def statistics_matrix(
        self,
        data: Sequence[Any] | None = None,
        enc_data: Any | None = None,
        estimate: Any | None = None,
        vectorizer: SufficientStatisticVectorizer | None = None,
        fit: bool = True,
    ) -> np.ndarray:
        """Return an n x d matrix of per-observation sufficient statistics."""
        if data is None and enc_data is None:
            raise ValueError("statistics_matrix requires data or enc_data")
        if data is not None and enc_data is not None:
            raise ValueError("pass only one of data or enc_data")

        if data is not None:
            data_values = list(data)
            values = [self.structured_statistics(x, estimate=estimate) for x in data_values]
        else:
            values = self.seq_structured_statistics(enc_data, estimate=estimate)

        vec = vectorizer if vectorizer is not None else self.vectorizer
        if fit:
            return vec.fit_transform(values)
        return vec.transform(values)

    def expected_statistics_matrix(
        self,
        data: Sequence[Any] | None = None,
        enc_data: Any | None = None,
        estimate: Any | None = None,
        vectorizer: SufficientStatisticVectorizer | None = None,
        fit: bool = True,
    ) -> np.ndarray:
        return self.statistics_matrix(data=data, enc_data=enc_data, estimate=estimate, vectorizer=vectorizer, fit=fit)

    def seq_expected_statistics(
        self,
        enc_data: Any,
        estimate: Any | None = None,
        vectorizer: SufficientStatisticVectorizer | None = None,
        fit: bool = True,
    ) -> np.ndarray:
        return self.expected_statistics_matrix(enc_data=enc_data, estimate=estimate, vectorizer=vectorizer, fit=fit)

    @staticmethod
    def _center(stats: np.ndarray, center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(stats, dtype=np.float64)
        mu = x.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
        return x - mu.reshape((1, -1)), mu

    def mean_statistics(self, stats: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        if stats is None:
            stats = self.statistics_matrix(**kwargs)
        return np.asarray(stats, dtype=np.float64).mean(axis=0)

    def score_center(self, stats: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        model_mean = getattr(self, "_model_mean", None)
        if model_mean is not None:
            try:
                return np.asarray(model_mean(), dtype=np.float64)
            except NotImplementedError:
                pass
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        return np.asarray(stats, dtype=np.float64).mean(axis=0)

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        """Empirical Fisher approximation from per-observation statistic vectors."""
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        centered, _ = self._center(stats)
        n = max(centered.shape[0], 1)
        if diagonal:
            return np.mean(centered * centered, axis=0) + ridge
        return np.dot(centered.T, centered) / float(n) + np.eye(centered.shape[1]) * ridge

    def fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return centered/whitened sufficient-statistic vectors.

        metric='identity' returns centered statistics, 'diagonal' divides by
        per-coordinate Fisher standard deviations, and 'full' applies an
        empirical full-matrix whitening transform.
        """
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        centered, _ = self._center(stats, center=center)

        if metric == "identity":
            return centered

        if metric == "diagonal":
            diag = fisher if fisher is not None else np.mean(centered * centered, axis=0)
            return centered / np.sqrt(np.asarray(diag, dtype=np.float64).reshape((1, -1)) + ridge)

        if metric == "full":
            info = fisher if fisher is not None else self.fisher_information(stats, diagonal=False, ridge=0.0)
            vals, vecs = np.linalg.eigh(np.asarray(info, dtype=np.float64))
            vals = np.maximum(vals, ridge)
            return np.dot(centered, np.dot(vecs, np.diag(1.0 / np.sqrt(vals))))

        raise ValueError("metric must be 'identity', 'diagonal', or 'full'")

    def observed_fisher_information(
        self,
        stats: np.ndarray | None = None,
        diagonal: bool = False,
        center: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        """Observed Fisher estimate from score vectors for observed data.

        For latent models the statistic rows are posterior-expected
        complete-data sufficient statistics, so centering them by their model
        expectation gives observed score vectors.  Their covariance is the
        observed Fisher metric used by Fisher-vector embeddings.  If a model
        center is unavailable, this falls back to the empirical mean.
        """
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        x = np.asarray(stats, dtype=np.float64)
        mu = self.score_center(stats=x) if center is None else np.asarray(center, dtype=np.float64)
        centered = x - mu.reshape((1, -1))
        n = max(centered.shape[0], 1)
        if diagonal:
            return np.mean(centered * centered, axis=0) + ridge
        return np.dot(centered.T, centered) / float(n) + np.eye(centered.shape[1]) * ridge

    def observed_fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        mu = self.score_center(stats=stats) if center is None else np.asarray(center, dtype=np.float64)

        if metric == "identity":
            return np.asarray(stats, dtype=np.float64) - mu.reshape((1, -1))

        if metric == "diagonal":
            diag = (
                fisher
                if fisher is not None
                else self.observed_fisher_information(stats=stats, diagonal=True, center=mu, ridge=0.0)
            )
            return self.fisher_vectors(stats=stats, metric=metric, center=mu, fisher=diag, ridge=ridge)

        if metric == "full":
            info = (
                fisher
                if fisher is not None
                else self.observed_fisher_information(stats=stats, diagonal=False, center=mu, ridge=0.0)
            )
            return self.fisher_vectors(stats=stats, metric=metric, center=mu, fisher=info, ridge=ridge)

        raise ValueError("metric must be 'identity', 'diagonal', or 'full'")

    def fisher_vector(
        self,
        x: Any,
        estimate: Any | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        vectorizer: SufficientStatisticVectorizer | None = None,
        ridge: float = 1.0e-8,
    ) -> np.ndarray:
        if vectorizer is None and not self.vectorizer.labels:
            stat = self.expected_sufficient_statistics(x, estimate=estimate)
        else:
            vec = vectorizer if vectorizer is not None else self.vectorizer
            stat = self.expected_sufficient_statistics(x, estimate=estimate, vectorizer=vec)
        return self.fisher_vectors(
            stats=np.reshape(stat, (1, -1)), metric=metric, center=center, fisher=fisher, ridge=ridge
        )[0]

    def natural_parameters(self) -> Any:
        """Return natural parameters when a specialized view provides them."""
        raise NotImplementedError("generic Fisher views do not expose canonical natural parameters")


class FixedFisherView(FisherView):
    """Distribution-specific Fisher view with fixed vector coordinates."""

    def __init__(self, dist: Any, labels: Sequence[Path]) -> None:
        self.dist = dist
        self.estimator = None
        self.labels = list(labels)
        self.vectorizer = SufficientStatisticVectorizer(self.labels)

    def _project_matrix(
        self, mat: np.ndarray, vectorizer: SufficientStatisticVectorizer | None, fit: bool
    ) -> np.ndarray:
        if vectorizer is None:
            return mat
        if fit:
            vectorizer.labels = list(self.labels)
            vectorizer._index = {k: i for i, k in enumerate(vectorizer.labels)}
            return mat

        out = np.zeros((mat.shape[0], len(vectorizer.labels)), dtype=np.float64)
        for j, label in enumerate(self.labels):
            k = vectorizer._index.get(label)
            if k is not None:
                out[:, k] = mat[:, j]
        return out

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        raise NotImplementedError

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        raise NotImplementedError

    def structured_statistics(self, x: Any, estimate: Any | None = None, weight: float = 1.0) -> Any:
        return self._statistics_from_data([x], estimate=estimate)[0] * weight

    def sufficient_statistics(
        self, x: Any, estimate: Any | None = None, vectorizer: SufficientStatisticVectorizer | None = None
    ) -> np.ndarray:
        mat = self._statistics_from_data([x], estimate=estimate)
        return self._project_matrix(mat, vectorizer, fit=vectorizer is None)[0]

    def seq_structured_statistics(self, enc_data: Any, estimate: Any | None = None) -> list[Any]:
        return [row for row in self._statistics_from_encoded(enc_data, estimate=estimate)]

    def statistics_matrix(
        self,
        data: Sequence[Any] | None = None,
        enc_data: Any | None = None,
        estimate: Any | None = None,
        vectorizer: SufficientStatisticVectorizer | None = None,
        fit: bool = True,
    ) -> np.ndarray:
        if data is None and enc_data is None:
            raise ValueError("statistics_matrix requires data or enc_data")
        if data is not None and enc_data is not None:
            raise ValueError("pass only one of data or enc_data")
        if data is not None:
            mat = self._statistics_from_data(list(data), estimate=estimate)
        else:
            mat = self._statistics_from_encoded(enc_data, estimate=estimate)
        return self._project_matrix(mat, vectorizer if vectorizer is not None else self.vectorizer, fit=fit)

    def _model_mean(self) -> np.ndarray:
        raise NotImplementedError

    def _model_fisher(self) -> np.ndarray:
        raise NotImplementedError

    def mean_statistics(self, stats: np.ndarray | None = None, model: bool = True, **kwargs: Any) -> np.ndarray:
        if model or stats is None:
            return self._model_mean()
        return np.asarray(stats, dtype=np.float64).mean(axis=0)

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        info = np.asarray(self._model_fisher(), dtype=np.float64)
        if diagonal:
            return np.diag(info) + ridge
        return info + np.eye(info.shape[0]) * ridge

    def fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        centered = np.asarray(stats, dtype=np.float64)
        mu = self._model_mean() if center is None else np.asarray(center, dtype=np.float64)
        centered = centered - mu.reshape((1, -1))

        if metric == "identity":
            return centered

        if metric == "diagonal":
            diag = np.diag(self._model_fisher()) if fisher is None else np.asarray(fisher, dtype=np.float64)
            return centered / np.sqrt(np.maximum(diag.reshape((1, -1)), 0.0) + ridge)

        if metric == "full":
            info = self._model_fisher() if fisher is None else np.asarray(fisher, dtype=np.float64)
            vals, vecs = np.linalg.eigh(info)
            vals = np.maximum(vals, ridge)
            return np.dot(centered, np.dot(vecs, np.diag(1.0 / np.sqrt(vals))))

        raise ValueError("metric must be 'identity', 'diagonal', or 'full'")


class CountFisherView(FixedFisherView):
    def __init__(
        self,
        dist: Any,
        mean_var_fn: Callable[[Any], tuple[float, float]],
        data_fn: Callable[[Any], np.ndarray],
        enc_fn: Callable[[Any], np.ndarray],
    ) -> None:
        super().__init__(dist, [("count",), ("sum",)])
        self._mean_var_fn = mean_var_fn
        self._data_fn = data_fn
        self._enc_fn = enc_fn

    @staticmethod
    def _matrix(x: Any) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        return np.column_stack((np.ones_like(xx, dtype=np.float64), xx))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        return self._matrix(self._data_fn(data))

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        return self._matrix(self._enc_fn(enc_data))

    def _model_mean(self) -> np.ndarray:
        mean, _ = self._mean_var_fn(self.dist)
        return np.asarray([1.0, mean], dtype=np.float64)

    def _model_fisher(self) -> np.ndarray:
        _, var = self._mean_var_fn(self.dist)
        info = np.zeros((2, 2), dtype=np.float64)
        info[1, 1] = max(float(var), 0.0)
        return info


class CompositeFisherView(FixedFisherView):
    def __init__(self, dist: Any) -> None:
        self.child_views = [to_fisher(d) for d in dist.dists]
        labels: list[Path] = []
        for i, view in enumerate(self.child_views):
            labels.extend((str(i),) + label for label in view.vectorizer.labels)
        super().__init__(dist, labels)

    def _refresh_labels(self) -> None:
        labels: list[Path] = []
        for i, view in enumerate(self.child_views):
            labels.extend((str(i),) + label for label in view.vectorizer.labels)
        self.labels = labels
        self.vectorizer = SufficientStatisticVectorizer(self.labels)

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        mats = []
        ests = [None] * len(self.child_views) if estimate is None else estimate.dists
        for i, view in enumerate(self.child_views):
            child_data = [x[i] for x in data]
            mats.append(view.expected_statistics_matrix(data=child_data, estimate=ests[i]))
        self._refresh_labels()
        return np.hstack(mats) if mats else np.zeros((len(data), 0), dtype=np.float64)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        mats = []
        ests = [None] * len(self.child_views) if estimate is None else estimate.dists
        for i, view in enumerate(self.child_views):
            mats.append(view.seq_expected_statistics(enc_data[i], estimate=ests[i]))
        self._refresh_labels()
        n = mats[0].shape[0] if mats else 0
        return np.hstack(mats) if mats else np.zeros((n, 0), dtype=np.float64)

    def _model_mean(self) -> np.ndarray:
        return np.concatenate([view.mean_statistics() for view in self.child_views])

    def _model_fisher(self) -> np.ndarray:
        blocks = [np.asarray(view.fisher_information(ridge=0.0), dtype=np.float64) for view in self.child_views]
        dim = sum(block.shape[0] for block in blocks)
        out = np.zeros((dim, dim), dtype=np.float64)
        pos = 0
        for block in blocks:
            n = block.shape[0]
            out[pos : pos + n, pos : pos + n] = block
            pos += n
        return out


class MixtureFisherView(FixedFisherView):
    """Complete-data Fisher view for finite mixture distributions.

    Coordinates are component assignment indicators followed by each
    component's sufficient statistics gated by that assignment.  Observed data
    map to posterior-expected complete-data statistics.
    """

    def __init__(self, dist: Any) -> None:
        self.child_views = [to_fisher(d) for d in dist.components]
        labels = self._labels_from_children()
        super().__init__(dist, labels)

    def _labels_from_children(self) -> list[Path]:
        labels: list[Path] = [("component", str(k)) for k in range(len(self.child_views))]
        for k, view in enumerate(self.child_views):
            labels.extend(("component_stat", str(k)) + label for label in view.vectorizer.labels)
        return labels

    def _refresh_labels(self) -> None:
        self.labels = self._labels_from_children()
        self.vectorizer = SufficientStatisticVectorizer(self.labels)

    def _posterior_from_data(self, data: Sequence[Any]) -> np.ndarray:
        return np.asarray([self.dist.posterior(x) for x in data], dtype=np.float64)

    def _posterior_from_encoded(self, enc_data: Any) -> np.ndarray:
        return np.asarray(self.dist.seq_posterior(enc_data), dtype=np.float64)

    def _component_stats_from_data(self, data: Sequence[Any]) -> list[np.ndarray]:
        return [view.expected_statistics_matrix(data=data) for view in self.child_views]

    def _component_stats_from_encoded(self, enc_data: Any) -> list[np.ndarray]:
        return [view.seq_expected_statistics(enc_data) for view in self.child_views]

    @staticmethod
    def _join_stats(z: np.ndarray, child_stats: Sequence[np.ndarray]) -> np.ndarray:
        blocks = [z]
        for k, stats in enumerate(child_stats):
            blocks.append(z[:, [k]] * stats)
        return np.hstack(blocks)

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        values = list(data)
        z = self._posterior_from_data(values)
        mats = self._component_stats_from_data(values)
        self._refresh_labels()
        return self._join_stats(z, mats)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        z = self._posterior_from_encoded(enc_data)
        mats = self._component_stats_from_encoded(enc_data)
        self._refresh_labels()
        return self._join_stats(z, mats)

    def structured_statistics(self, x: Any, estimate: Any | None = None, weight: float = 1.0) -> Any:
        z = self.dist.posterior(x) if estimate is None else estimate.posterior(x)
        child_values = tuple(z[k] * self.child_views[k].sufficient_statistics(x) for k in range(len(self.child_views)))
        return weight * z, child_values

    def _component_means(self) -> list[np.ndarray]:
        return [np.asarray(view.mean_statistics(), dtype=np.float64) for view in self.child_views]

    def _component_moments(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        means = self._component_means()
        infos = [np.asarray(view.fisher_information(ridge=0.0), dtype=np.float64) for view in self.child_views]
        return means, infos

    def _model_mean(self) -> np.ndarray:
        w = np.asarray(self.dist.w, dtype=np.float64)
        means = self._component_means()
        return np.concatenate([w] + [w[k] * means[k] for k in range(len(means))])

    def _model_fisher(self) -> np.ndarray:
        w = np.asarray(self.dist.w, dtype=np.float64)
        means, infos = self._component_moments()
        k_count = len(means)
        dims = [len(mu) for mu in means]
        offsets = []
        pos = k_count
        for dim in dims:
            offsets.append(pos)
            pos += dim

        out = np.zeros((pos, pos), dtype=np.float64)
        out[:k_count, :k_count] = np.diag(w) - np.outer(w, w)

        for i in range(k_count):
            for k in range(k_count):
                cov = ((w[k] if i == k else 0.0) - w[i] * w[k]) * means[k]
                s = offsets[k]
                e = s + dims[k]
                out[i, s:e] = cov
                out[s:e, i] = cov

        for k in range(k_count):
            sk = offsets[k]
            ek = sk + dims[k]
            muk = means[k]
            out[sk:ek, sk:ek] = w[k] * infos[k] + w[k] * (1.0 - w[k]) * np.outer(muk, muk)
            for l in range(k + 1, k_count):
                sl = offsets[l]
                el = sl + dims[l]
                block = -w[k] * w[l] * np.outer(muk, means[l])
                out[sk:ek, sl:el] = block
                out[sl:el, sk:ek] = block.T

        return out


class _PairProductFisherView(FixedFisherView):
    """Fisher view for a product of two component views."""

    def __init__(self, left: FisherView, right: FisherView) -> None:
        self.left = left
        self.right = right
        labels = [("0",) + label for label in left.vectorizer.labels]
        labels.extend(("1",) + label for label in right.vectorizer.labels)
        FixedFisherView.__init__(self, (left.dist, right.dist), labels)

    def _refresh_labels(self) -> None:
        labels = [("0",) + label for label in self.left.vectorizer.labels]
        labels.extend(("1",) + label for label in self.right.vectorizer.labels)
        self.labels = labels
        self.vectorizer = SufficientStatisticVectorizer(self.labels)

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        left_data = [x[0] for x in data]
        right_data = [x[1] for x in data]
        left_est = None if estimate is None else estimate[0]
        right_est = None if estimate is None else estimate[1]
        left = self.left.expected_statistics_matrix(data=left_data, estimate=left_est)
        right = self.right.expected_statistics_matrix(data=right_data, estimate=right_est)
        self._refresh_labels()
        return np.hstack((left, right))

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        left_est = None if estimate is None else estimate[0]
        right_est = None if estimate is None else estimate[1]
        left = self.left.seq_expected_statistics(enc_data[0], estimate=left_est)
        right = self.right.seq_expected_statistics(enc_data[1], estimate=right_est)
        self._refresh_labels()
        return np.hstack((left, right))

    def _model_mean(self) -> np.ndarray:
        return np.concatenate((self.left.mean_statistics(), self.right.mean_statistics()))

    def _model_fisher(self) -> np.ndarray:
        blocks = [
            np.asarray(self.left.fisher_information(ridge=0.0), dtype=np.float64),
            np.asarray(self.right.fisher_information(ridge=0.0), dtype=np.float64),
        ]
        dim = sum(block.shape[0] for block in blocks)
        out = np.zeros((dim, dim), dtype=np.float64)
        pos = 0
        for block in blocks:
            n = block.shape[0]
            out[pos : pos + n, pos : pos + n] = block
            pos += n
        return out


class JointMixtureFisherView(MixtureFisherView):
    """Complete-data Fisher view for joint mixtures without concrete proxies."""

    def __init__(self, dist: Any) -> None:
        self.pair_indices: list[tuple[int, int]] = []
        weights = []
        child_views = []
        for i, component1 in enumerate(dist.components1):
            if float(dist.w1[i]) <= 0.0:
                continue
            for j, component2 in enumerate(dist.components2):
                weight = float(dist.w1[i]) * float(dist.taus12[i, j])
                if weight <= 0.0:
                    continue
                self.pair_indices.append((i, j))
                weights.append(weight)
                child_views.append(_PairProductFisherView(to_fisher(component1), to_fisher(component2)))
        if not child_views:
            raise ValueError("JointMixtureFisherView requires at least one positive-weight component pair.")
        self.child_views = child_views
        self._pair_weights = np.asarray(weights, dtype=np.float64)
        self._pair_weights /= self._pair_weights.sum()
        labels = self._labels_from_children()
        FixedFisherView.__init__(self, dist, labels)

    @property
    def num_pairs(self) -> int:
        return len(self.pair_indices)

    def _pair_log_scores_from_data(self, data: Sequence[Any]) -> np.ndarray:
        rows = []
        for x in data:
            scores = []
            for i, j in self.pair_indices:
                scores.append(
                    self.dist.log_w1[i]
                    + self.dist.log_taus12[i, j]
                    + self.dist.components1[i].log_density(x[0])
                    + self.dist.components2[j].log_density(x[1])
                )
            rows.append(scores)
        return np.asarray(rows, dtype=np.float64)

    def _pair_log_scores_from_encoded(self, enc_data: Any) -> np.ndarray:
        sz, enc1, enc2 = enc_data
        scores = np.zeros((int(sz), len(self.pair_indices)), dtype=np.float64)
        left_cache: dict[int, np.ndarray] = {}
        right_cache: dict[int, np.ndarray] = {}
        for k, (i, j) in enumerate(self.pair_indices):
            if i not in left_cache:
                left_cache[i] = np.asarray(self.dist.components1[i].seq_log_density(enc1), dtype=np.float64)
            if j not in right_cache:
                right_cache[j] = np.asarray(self.dist.components2[j].seq_log_density(enc2), dtype=np.float64)
            scores[:, k] = self.dist.log_w1[i] + self.dist.log_taus12[i, j] + left_cache[i] + right_cache[j]
        return scores

    @staticmethod
    def _posterior_from_scores(scores: np.ndarray) -> np.ndarray:
        mx = np.max(scores, axis=1, keepdims=True)
        weights = np.exp(scores - mx)
        return weights / np.sum(weights, axis=1, keepdims=True)

    def _posterior_from_data(self, data: Sequence[Any]) -> np.ndarray:
        return self._posterior_from_scores(self._pair_log_scores_from_data(data))

    def _posterior_from_encoded(self, enc_data: Any) -> np.ndarray:
        return self._posterior_from_scores(self._pair_log_scores_from_encoded(enc_data))

    def log_density(self, x: Any) -> float:
        scores = self._pair_log_scores_from_data([x])[0]
        mx = float(np.max(scores))
        return float(mx + np.log(np.exp(scores - mx).sum()))

    def _component_stats_from_data(self, data: Sequence[Any]) -> list[np.ndarray]:
        return [view.expected_statistics_matrix(data=data) for view in self.child_views]

    def _component_stats_from_encoded(self, enc_data: Any) -> list[np.ndarray]:
        _, enc1, enc2 = enc_data
        return [view.seq_expected_statistics((enc1, enc2)) for view in self.child_views]

    def structured_statistics(self, x: Any, estimate: Any | None = None, weight: float = 1.0) -> Any:
        if estimate is not None and estimate is not self.dist:
            return to_fisher(estimate).structured_statistics(x, weight=weight)
        z = self._posterior_from_data([x])[0]
        child_values = tuple(z[k] * self.child_views[k].sufficient_statistics(x) for k in range(len(self.child_views)))
        return weight * z, child_values

    def _model_mean(self) -> np.ndarray:
        w = self._pair_weights
        means = self._component_means()
        return np.concatenate([w] + [w[k] * means[k] for k in range(len(means))])

    def _model_fisher(self) -> np.ndarray:
        w = self._pair_weights
        means, infos = self._component_moments()
        k_count = len(means)
        dims = [len(mu) for mu in means]
        offsets = []
        pos = k_count
        for dim in dims:
            offsets.append(pos)
            pos += dim

        out = np.zeros((pos, pos), dtype=np.float64)
        out[:k_count, :k_count] = np.diag(w) - np.outer(w, w)

        for i in range(k_count):
            for k in range(k_count):
                cov = ((w[k] if i == k else 0.0) - w[i] * w[k]) * means[k]
                s = offsets[k]
                e = s + dims[k]
                out[i, s:e] = cov
                out[s:e, i] = cov

        for k in range(k_count):
            sk = offsets[k]
            ek = sk + dims[k]
            muk = means[k]
            out[sk:ek, sk:ek] = w[k] * infos[k] + w[k] * (1.0 - w[k]) * np.outer(muk, muk)
            for l in range(k + 1, k_count):
                sl = offsets[l]
                el = sl + dims[l]
                block = -w[k] * w[l] * np.outer(muk, means[l])
                out[sk:ek, sl:el] = block
                out[sl:el, sk:ek] = block.T

        return out


def _is_null_dist(dist: Any) -> bool:
    return dist is None or type(dist).__name__ == "NullDistribution"


def _seq_encode_model(model: Any, data: Sequence[Any]) -> Any:
    if hasattr(model, "dist_to_encoder"):
        return model.dist_to_encoder().seq_encode(data)
    if hasattr(model, "seq_encode"):
        return model.seq_encode(data)
    raise NotImplementedError("%s does not provide sequence encoding" % type(model).__name__)


def _diag_info_from_view(view: FisherView) -> np.ndarray:
    info = np.asarray(view.fisher_information(ridge=0.0), dtype=np.float64)
    return np.diag(info) if info.ndim == 2 else info


def _full_info_from_view(view: FisherView) -> np.ndarray:
    info = np.asarray(view.fisher_information(ridge=0.0), dtype=np.float64)
    return np.diag(info) if info.ndim == 1 else info


def _second_diag_from_view(view: FisherView) -> np.ndarray:
    mu = np.asarray(view.mean_statistics(), dtype=np.float64)
    return _diag_info_from_view(view) + mu * mu


def _structured_values_matrix(view: FisherView, values: Sequence[Any]) -> np.ndarray:
    if not values:
        return np.zeros((0, len(view.vectorizer.labels)), dtype=np.float64)
    if isinstance(view, FixedFisherView):
        # IntegerCategorical views (now defined in pysp.stats.leaf.int_range) self-identify with a
        # class marker so this shared builder stays decoupled from that module (no import cycle).
        if getattr(view, "_fisher_integer_categorical", False):
            mat = np.zeros((len(values), len(view.vectorizer.labels)), dtype=np.float64)
            for i, value in enumerate(values):
                if not (isinstance(value, tuple) and len(value) == 2):
                    break
                try:
                    min_val = int(value[0])
                    counts = np.asarray(value[1], dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    break
                for offset, count in enumerate(counts):
                    j = view.key_index.get(min_val + offset)
                    if j is not None:
                        mat[i, j] = count
            else:
                return mat

        tmp = SufficientStatisticVectorizer().fit(values)
        mat = tmp.transform(values)
        if mat.shape[1] == len(view.vectorizer.labels):
            return mat
        return view.vectorizer.transform(values)
    vec = view.vectorizer
    if not vec.labels:
        vec.fit(values)
    return vec.transform(values)


def _finite_support_from_log_density(dist: Any, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(lo, hi + 1, dtype=np.int64)
    lp = np.asarray([dist.log_density(int(v)) for v in values], dtype=np.float64)
    good = np.isfinite(lp)
    values = values[good]
    lp = lp[good]
    if len(values) == 0:
        return values.astype(np.float64), np.zeros(0, dtype=np.float64)
    lp -= np.max(lp)
    p = np.exp(lp)
    p /= p.sum()
    return values.astype(np.float64), p


def _length_support(dist: Any, tol: float = 1.0e-12, max_terms: int = 20000) -> tuple[np.ndarray, np.ndarray] | None:
    if _is_null_dist(dist):
        return None
    tname = type(dist).__name__

    if tname == "IntegerCategoricalDistribution" and hasattr(dist, "p_vec"):
        values = np.arange(int(dist.min_val), int(dist.max_val) + 1, dtype=np.float64)
        probs = np.asarray(dist.p_vec, dtype=np.float64)
        total = probs.sum()
        return values, probs / total if total > 0.0 else np.ones_like(probs) / max(len(probs), 1)

    if tname == "IntegerCategoricalDistribution" and hasattr(dist, "prob_vec"):
        values = np.arange(int(dist.min_index), int(dist.max_index) + 1, dtype=np.float64)
        probs = np.asarray(dist.prob_vec, dtype=np.float64)
        total = probs.sum()
        return values, probs / total if total > 0.0 else np.ones_like(probs) / max(len(probs), 1)

    if tname == "CategoricalDistribution" and hasattr(dist, "pmap") and not getattr(dist, "no_default", False):
        try:
            items = sorted(((float(k), float(v)) for k, v in dist.pmap.items()), key=lambda u: u[0])
        except (TypeError, ValueError):
            return None
        values = np.asarray([u[0] for u in items], dtype=np.float64)
        probs = np.asarray([u[1] for u in items], dtype=np.float64)
        total = probs.sum()
        return values, probs / total if total > 0.0 else np.ones_like(probs) / max(len(probs), 1)

    if tname == "CategoricalDistribution" and hasattr(dist, "prob_map"):
        try:
            items = sorted(((float(k), float(v)) for k, v in dist.prob_map.items()), key=lambda u: u[0])
        except (TypeError, ValueError):
            return None
        values = np.asarray([u[0] for u in items], dtype=np.float64)
        probs = np.asarray([u[1] for u in items], dtype=np.float64)
        total = probs.sum()
        return values, probs / total if total > 0.0 else np.ones_like(probs) / max(len(probs), 1)

    if tname == "BernoulliDistribution" and hasattr(dist, "p"):
        p = float(dist.p)
        return np.asarray([0.0, 1.0]), np.asarray([1.0 - p, p])

    if tname == "BinomialDistribution" and hasattr(dist, "n"):
        shift = 0 if getattr(dist, "min_val", None) is None else int(dist.min_val)
        return _finite_support_from_log_density(dist, shift, shift + int(dist.n))

    if tname == "PoissonDistribution" and hasattr(dist, "lam"):
        lam = float(dist.lam)
        hi = int(max(32.0, math.ceil(lam + 12.0 * math.sqrt(max(lam, 1.0)) + 32.0)))
        while hi < max_terms:
            values = np.arange(0, hi + 1, dtype=np.int64)
            lp = np.asarray([dist.log_density(int(v)) for v in values], dtype=np.float64)
            probs = np.exp(lp[np.isfinite(lp)])
            values = values[np.isfinite(lp)]
            mass = probs.sum()
            if len(probs) and mass >= 1.0 - tol:
                return values.astype(np.float64), probs / mass
            if hi > lam + 20.0 * math.sqrt(max(lam, 1.0)) + 100.0:
                return values.astype(np.float64), probs / mass
            hi *= 2
        return _finite_support_from_log_density(dist, 0, max_terms)

    if tname == "GeometricDistribution" and hasattr(dist, "p"):
        p = float(dist.p)
        q = 1.0 - p
        if q <= 0.0:
            return np.asarray([1.0]), np.asarray([1.0])
        hi = int(min(max_terms, max(1, math.ceil(math.log(tol) / math.log(q)))))
        values = np.arange(1, hi + 1, dtype=np.float64)
        probs = p * np.power(q, values - 1.0)
        probs /= probs.sum()
        return values, probs

    return None


class EmpiricalMetricFixedFisherView(FixedFisherView):
    """Fixed-coordinate view whose whitening falls back to empirical Fisher."""

    def mean_statistics(self, stats: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        return np.asarray(stats, dtype=np.float64).mean(axis=0)

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        return FisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge)

    def fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        return FisherView.fisher_vectors(self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge)


class SequenceFisherView(FixedFisherView):
    """Structured Fisher view for iid sequence distributions."""

    def __init__(self, dist: Any) -> None:
        self.child_view = to_fisher(dist.dist)
        self.len_view = None if _is_null_dist(getattr(dist, "len_dist", None)) else to_fisher(dist.len_dist)
        super().__init__(dist, self._labels_from_children())

    def _labels_from_children(self) -> list[Path]:
        labels = [("element",) + label for label in self.child_view.vectorizer.labels]
        if self.len_view is not None:
            labels.extend(("length",) + label for label in self.len_view.vectorizer.labels)
        return labels

    def _refresh_labels(self) -> None:
        self.labels = self._labels_from_children()
        self.vectorizer = SufficientStatisticVectorizer(self.labels)

    @staticmethod
    def _lengths_from_encoded(enc_data: Any) -> np.ndarray:
        _, inv_len, nonzero, _, _ = enc_data
        lengths = np.zeros(len(inv_len), dtype=np.int64)
        nz = np.asarray(nonzero, dtype=bool)
        lengths[nz] = np.rint(1.0 / np.asarray(inv_len, dtype=np.float64)[nz]).astype(np.int64)
        return lengths

    def _aggregate_flat(
        self, flat_stats: np.ndarray, idx: np.ndarray, n: int, inv_len: np.ndarray | None
    ) -> np.ndarray:
        out = np.zeros((n, flat_stats.shape[1]), dtype=np.float64)
        if len(idx) == 0:
            return out
        weights = np.asarray(inv_len, dtype=np.float64)[idx] if self.dist.len_normalized else 1.0
        if np.isscalar(weights):
            np.add.at(out, idx, flat_stats)
        else:
            np.add.at(out, idx, flat_stats * weights[:, None])
        return out

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        enc = _seq_encode_model(self.dist if estimate is None else estimate, list(data))
        return self._statistics_from_encoded(enc, estimate=estimate)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        idx, inv_len, _, enc_seq, enc_len = enc_data
        n = len(inv_len)
        if len(idx):
            flat = self.child_view.seq_expected_statistics(enc_seq)
            elem = self._aggregate_flat(flat, np.asarray(idx, dtype=np.int64), n, inv_len)
        else:
            elem = np.zeros((n, len(self.child_view.mean_statistics())), dtype=np.float64)
        blocks = [elem]
        if self.len_view is not None:
            blocks.append(self.len_view.seq_expected_statistics(enc_len))
        self._refresh_labels()
        return np.hstack(blocks) if blocks else np.zeros((n, 0), dtype=np.float64)

    def _sequence_model_mean_cov(self) -> tuple[np.ndarray, np.ndarray]:
        support = _length_support(self.dist.len_dist)
        if support is None:
            raise NotImplementedError("sequence model Fisher requires a supported length distribution")
        lengths, probs = support
        child_mu = np.asarray(self.child_view.mean_statistics(), dtype=np.float64)
        child_cov = _full_info_from_view(self.child_view)
        child_outer = np.outer(child_mu, child_mu)

        elem_mean = np.zeros_like(child_mu)
        elem_second = np.zeros((len(child_mu), len(child_mu)), dtype=np.float64)
        elem_cond_means = []
        for n_float, p in zip(lengths, probs):
            n = max(int(round(n_float)), 0)
            if self.dist.len_normalized:
                if n > 0:
                    cond_mean = child_mu
                    cond_second = child_cov / float(n) + child_outer
                else:
                    cond_mean = np.zeros_like(child_mu)
                    cond_second = np.zeros_like(elem_second)
            else:
                cond_mean = float(n) * child_mu
                cond_second = float(n) * child_cov + float(n * n) * child_outer
            elem_mean += p * cond_mean
            elem_second += p * cond_second
            elem_cond_means.append(cond_mean)

        elem_cov = elem_second - np.outer(elem_mean, elem_mean)
        if self.len_view is not None:
            len_mat = self.len_view.expected_statistics_matrix(data=[int(round(v)) for v in lengths])
            len_mean = np.dot(probs, len_mat)
            len_second = np.dot((probs[:, None] * len_mat).T, len_mat)
            len_cov = len_second - np.outer(len_mean, len_mean)
            elem_len_second = np.zeros((len(child_mu), len(len_mean)), dtype=np.float64)
            for cond_mean, len_row, p in zip(elem_cond_means, len_mat, probs):
                elem_len_second += p * np.outer(cond_mean, len_row)
            cross = elem_len_second - np.outer(elem_mean, len_mean)

            mean = np.concatenate((elem_mean, len_mean))
            cov = np.zeros((len(mean), len(mean)), dtype=np.float64)
            d = len(child_mu)
            cov[:d, :d] = elem_cov
            cov[:d, d:] = cross
            cov[d:, :d] = cross.T
            cov[d:, d:] = len_cov
        else:
            mean = elem_mean
            cov = elem_cov

        cov = 0.5 * (cov + cov.T)
        diag = np.maximum(np.diag(cov), 0.0)
        cov[np.diag_indices_from(cov)] = diag
        return mean, cov

    def _model_mean(self) -> np.ndarray:
        return self._sequence_model_mean_cov()[0]

    def _model_fisher(self) -> np.ndarray:
        return self._sequence_model_mean_cov()[1]

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        try:
            return super().fisher_information(stats=stats, diagonal=diagonal, ridge=ridge, **kwargs)
        except NotImplementedError:
            return FisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge, **kwargs)

    def fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        try:
            return super().fisher_vectors(
                stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge, **kwargs
            )
        except NotImplementedError:
            if stats is None:
                stats = self.expected_statistics_matrix(**kwargs)
            return FisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge
            )


class MultinomialFisherView(SequenceFisherView):
    """Fisher view for bag/count observations with a count-weighted child model.

    The model Fisher uses the canonical multinomial/count sufficient-statistic
    moments that match estimation.  The repo's MultinomialDistribution
    log_density intentionally omits the multinomial coefficient in its
    enumerator score; that coefficient is a base-measure term, not an
    accumulator statistic.
    """

    def _labels_from_children(self) -> list[Path]:
        labels = [("value",) + label for label in self.child_view.vectorizer.labels]
        if self.len_view is not None:
            labels.extend(("length",) + label for label in self.len_view.vectorizer.labels)
        return labels

    def _aggregate_weighted_flat(
        self, flat_stats: np.ndarray, idx: np.ndarray, counts: np.ndarray, totals: np.ndarray
    ) -> np.ndarray:
        out = np.zeros((len(totals), flat_stats.shape[1]), dtype=np.float64)
        if len(idx) == 0:
            return out
        weights = np.asarray(counts, dtype=np.float64)
        if self.dist.len_normalized:
            totals = np.asarray(totals, dtype=np.float64)
            inv = np.zeros_like(totals, dtype=np.float64)
            nz = totals != 0.0
            inv[nz] = 1.0 / totals[nz]
            weights = weights * inv[idx]
        np.add.at(out, idx, flat_stats * weights[:, None])
        return out

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        idx, _, _, enc_seq, enc_len, counts, totals = enc_data
        idx = np.asarray(idx, dtype=np.int64)
        totals = np.asarray(totals, dtype=np.float64)
        if len(idx):
            flat = self.child_view.seq_expected_statistics(enc_seq)
            elem = self._aggregate_weighted_flat(flat, idx, np.asarray(counts, dtype=np.float64), totals)
        else:
            elem = np.zeros((len(totals), len(self.child_view.mean_statistics())), dtype=np.float64)
        blocks = [elem]
        if self.len_view is not None:
            blocks.append(self.len_view.seq_expected_statistics(enc_len))
        self._refresh_labels()
        return np.hstack(blocks) if blocks else np.zeros((len(totals), 0), dtype=np.float64)


class OptionalFisherView(EmpiricalMetricFixedFisherView):
    def __init__(self, dist: Any) -> None:
        self.child_view = to_fisher(dist.dist)
        self.has_gate = getattr(dist, "has_p", getattr(dist, "p", None) is not None)
        self._encoded_missing_first = hasattr(dist, "missing_value_is_nan")
        labels = []
        if self.has_gate:
            labels.extend([("missing",), ("present",)])
        labels.extend(("present_stat",) + label for label in self.child_view.vectorizer.labels)
        super().__init__(dist, labels)

    def _is_missing(self, x: Any) -> bool:
        if getattr(self.dist, "missing_value_is_nan", getattr(self.dist, "mv_is_nan", False)):
            return isinstance(x, (np.floating, float)) and np.isnan(x)
        return x == self.dist.missing_value or x is self.dist.missing_value

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        n = len(data)
        d = len(self.child_view.vectorizer.labels)
        child = np.zeros((n, d), dtype=np.float64)
        present_idx = []
        present_values = []
        gate = np.zeros((n, 2), dtype=np.float64) if self.has_gate else None
        for i, x in enumerate(data):
            missing = self._is_missing(x)
            if gate is not None:
                gate[i, 0 if missing else 1] = 1.0
            if not missing:
                present_idx.append(i)
                present_values.append(x)
        if present_values:
            child[np.asarray(present_idx, dtype=np.int64)] = self.child_view.expected_statistics_matrix(
                data=present_values
            )
        return np.hstack((gate, child)) if gate is not None else child

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        n, idx_a, idx_b, enc_child = enc_data
        z_idx, nz_idx = (idx_a, idx_b) if self._encoded_missing_first else (idx_b, idx_a)
        d = len(self.child_view.vectorizer.labels)
        child = np.zeros((n, d), dtype=np.float64)
        if len(nz_idx):
            child[np.asarray(nz_idx, dtype=np.int64)] = self.child_view.seq_expected_statistics(enc_child)
        if self.has_gate:
            gate = np.zeros((n, 2), dtype=np.float64)
            gate[np.asarray(z_idx, dtype=np.int64), 0] = 1.0
            gate[np.asarray(nz_idx, dtype=np.int64), 1] = 1.0
            return np.hstack((gate, child))
        return child

    def _model_mean(self) -> np.ndarray:
        if not self.has_gate:
            raise NotImplementedError
        p = float(self.dist.p)
        q = 1.0 - p
        return np.concatenate((np.asarray([p, q]), q * self.child_view.mean_statistics()))

    def _model_fisher(self) -> np.ndarray:
        if not self.has_gate:
            raise NotImplementedError
        p = float(self.dist.p)
        q = 1.0 - p
        mu = np.asarray(self.child_view.mean_statistics(), dtype=np.float64)
        info = np.asarray(self.child_view.fisher_information(ridge=0.0), dtype=np.float64)
        d = len(mu)
        out = np.zeros((2 + d, 2 + d), dtype=np.float64)
        gate_mean = np.asarray([p, q])
        out[:2, :2] = np.diag(gate_mean) - np.outer(gate_mean, gate_mean)
        out[0, 2:] = -p * q * mu
        out[2:, 0] = out[0, 2:]
        out[1, 2:] = p * q * mu
        out[2:, 1] = out[1, 2:]
        out[2:, 2:] = q * info + p * q * np.outer(mu, mu)
        return out

    def mean_statistics(self, stats: np.ndarray | None = None, model: bool = True, **kwargs: Any) -> np.ndarray:
        try:
            return FixedFisherView.mean_statistics(self, stats=stats, model=model, **kwargs)
        except NotImplementedError:
            return EmpiricalMetricFixedFisherView.mean_statistics(self, stats=stats, **kwargs)

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        try:
            return FixedFisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge, **kwargs)
        except NotImplementedError:
            return EmpiricalMetricFixedFisherView.fisher_information(
                self, stats=stats, diagonal=diagonal, ridge=ridge, **kwargs
            )

    def fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        try:
            return FixedFisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge, **kwargs
            )
        except NotImplementedError:
            return EmpiricalMetricFixedFisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge, **kwargs
            )


class WeightedFisherView(FixedFisherView):
    def __init__(self, dist: Any) -> None:
        self.child_view = to_fisher(dist.dist)
        super().__init__(dist, list(self.child_view.vectorizer.labels))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        values = [x[0] for x in data]
        weights = np.asarray([x[1] for x in data], dtype=np.float64)
        return self.child_view.expected_statistics_matrix(data=values) * weights[:, None]

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        enc_child, weights = enc_data
        return self.child_view.seq_expected_statistics(enc_child) * np.asarray(weights, dtype=np.float64)[:, None]

    def _model_mean(self) -> np.ndarray:
        return self.child_view.mean_statistics()

    def _model_fisher(self) -> np.ndarray:
        return np.asarray(self.child_view.fisher_information(ridge=0.0), dtype=np.float64)

    def score_center(self, stats: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        return np.asarray(stats, dtype=np.float64).mean(axis=0)


class SelectFisherView(EmpiricalMetricFixedFisherView):
    def __init__(self, dist: Any) -> None:
        self.child_views = [to_fisher(d) for d in dist.dists]
        labels: list[Path] = []
        for i, view in enumerate(self.child_views):
            labels.extend(("choice", str(i)) + label for label in view.vectorizer.labels)
        super().__init__(dist, labels)

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        n = len(data)
        blocks = [np.zeros((n, len(view.vectorizer.labels)), dtype=np.float64) for view in self.child_views]
        grouped: dict[int, list[tuple[int, Any]]] = {}
        for i, x in enumerate(data):
            grouped.setdefault(int(self.dist.choice_function(x)), []).append((i, x))
        for k, pairs in grouped.items():
            idx = np.asarray([i for i, _ in pairs], dtype=np.int64)
            vals = [x for _, x in pairs]
            blocks[k][idx] = self.child_views[k].expected_statistics_matrix(data=vals)
        return np.hstack(blocks) if blocks else np.zeros((n, 0), dtype=np.float64)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        xi, idx, enc_tuple = enc_data
        n = sum(len(u) for u in xi)
        blocks = [np.zeros((n, len(view.vectorizer.labels)), dtype=np.float64) for view in self.child_views]
        for g, k in enumerate(idx):
            rows = np.asarray(xi[g], dtype=np.int64)
            blocks[int(k)][rows] = self.child_views[int(k)].seq_expected_statistics(enc_tuple[g])
        return np.hstack(blocks) if blocks else np.zeros((n, 0), dtype=np.float64)


class HeterogeneousPCFGFisherView(FixedFisherView):
    """Inside-outside Fisher view for heterogeneous PCFGs.

    Coordinates are expected complete-data rule counts followed by terminal
    emission sufficient statistics gated by the posterior probability of the
    corresponding terminal rule at each token.  For finite enumerable PCFGs,
    the model Fisher is the exact observed Fisher covariance of these
    posterior-expected complete-data statistics under the model distribution.
    Recursive or infinite-support grammars should use observed_fisher_* on data.
    """

    _max_model_enum_terms = 100000
    _model_mass_tol = 1.0e-8

    def __init__(self, dist: Any) -> None:
        self.dist = dist
        self.child_views = [to_fisher(d) for d in dist.emissions]
        self._model_cache: tuple[np.ndarray, np.ndarray] | None = None
        super().__init__(dist, self._labels_from_children())

    def _labels_from_children(self) -> list[Path]:
        labels: list[Path] = [("terminal_rule", str(r)) for r in range(self.dist.num_terminal_rules)]
        labels.extend(("binary_rule", str(r)) for r in range(self.dist.num_binary_rules))
        for r, view in enumerate(self.child_views):
            labels.extend(("terminal_emission", str(r)) + label for label in view.vectorizer.labels)
        return labels

    def _refresh_labels(self) -> None:
        self.labels = self._labels_from_children()
        self.vectorizer = SufficientStatisticVectorizer(self.labels)
        self._model_cache = None

    def _matrix_from_values(self, values: Sequence[Any]) -> np.ndarray:
        if not values:
            return np.zeros((0, len(self.labels)), dtype=np.float64)

        terminal = np.vstack([np.asarray(v[0], dtype=np.float64).reshape(-1) for v in values])
        binary = np.vstack([np.asarray(v[1], dtype=np.float64).reshape(-1) for v in values])
        blocks = [terminal, binary]

        for r, view in enumerate(self.child_views):
            emission_values = [v[2][r] for v in values]
            blocks.append(_structured_values_matrix(view, emission_values))

        self._refresh_labels()
        return np.hstack(blocks)

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        model = self.dist if estimate is None else estimate
        enc = _seq_encode_model(model, list(data))
        return self._statistics_from_encoded(enc, estimate=model)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        model = self.dist if estimate is None else estimate
        lengths, enc_by_rule = enc_data
        lengths = np.asarray(lengths, dtype=np.int64)
        nobs = len(lengths)
        total = int(lengths.sum())

        terminal_ld = np.empty((total, model.num_terminal_rules), dtype=np.float64)
        child_stats: list[np.ndarray] = []
        for r, dist in enumerate(model.emissions):
            if total > 0:
                terminal_ld[:, r] = dist.seq_log_density(enc_by_rule[r])
                child_stats.append(self.child_views[r].seq_expected_statistics(enc_by_rule[r], estimate=dist))
            else:
                terminal_ld = np.empty((0, model.num_terminal_rules), dtype=np.float64)
                child_stats.append(np.zeros((0, len(self.child_views[r].vectorizer.labels)), dtype=np.float64))

        self._refresh_labels()
        terminal_counts = np.zeros((nobs, model.num_terminal_rules), dtype=np.float64)
        binary_counts = np.zeros((nobs, model.num_binary_rules), dtype=np.float64)
        emission_blocks = [np.zeros((nobs, stats.shape[1]), dtype=np.float64) for stats in child_stats]

        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        for i, n in enumerate(lengths):
            if n <= 0:
                continue
            start = int(offsets[i])
            stop = int(offsets[i + 1])
            _, terminal_post, binary_count, _ = model._inside_outside(terminal_ld[start:stop])
            terminal_counts[i] = terminal_post.sum(axis=0)
            binary_counts[i] = binary_count
            for r, stats in enumerate(child_stats):
                if stats.shape[1] > 0:
                    emission_blocks[r][i] = np.dot(terminal_post[:, r], stats[start:stop])

        blocks = [terminal_counts, binary_counts]
        blocks.extend(emission_blocks)
        return np.hstack(blocks) if blocks else np.zeros((nobs, 0), dtype=np.float64)

    def _enumerated_model_mean_cov(self) -> tuple[np.ndarray, np.ndarray]:
        if self._model_cache is not None:
            return self._model_cache

        values: list[Any] = []
        probs: list[float] = []
        try:
            iterator = iter(self.dist.enumerator())
            exhausted = False
            for _ in range(self._max_model_enum_terms):
                try:
                    value, log_prob = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                if np.isfinite(log_prob):
                    values.append(value)
                    probs.append(float(math.exp(log_prob)))
            if not exhausted:
                raise NotImplementedError(
                    "PCFG model Fisher requires finite enumerable support; use observed_fisher_information()."
                )
        except NotImplementedError:
            raise
        except Exception as exc:
            raise NotImplementedError(
                "PCFG model Fisher requires finite enumerable support; use observed_fisher_information()."
            ) from exc

        if not values:
            raise NotImplementedError("PCFG model Fisher requires non-empty finite support.")

        weights = np.asarray(probs, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total) or abs(total - 1.0) > self._model_mass_tol:
            raise NotImplementedError("PCFG finite support did not sum to one; use observed_fisher_information().")
        weights /= total

        stats = self.expected_statistics_matrix(data=values)
        mean = np.dot(weights, stats)
        second = np.dot((weights[:, None] * stats).T, stats)
        cov = second - np.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        diag = np.maximum(np.diag(cov), 0.0)
        cov[np.diag_indices_from(cov)] = diag
        self._model_cache = (mean, cov)
        return self._model_cache

    def _model_mean(self) -> np.ndarray:
        return self._enumerated_model_mean_cov()[0]

    def _model_fisher(self) -> np.ndarray:
        return self._enumerated_model_mean_cov()[1]

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        try:
            return FixedFisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge, **kwargs)
        except NotImplementedError:
            if stats is not None:
                return FisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge)
            raise

    def fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        try:
            return FixedFisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge, **kwargs
            )
        except NotImplementedError:
            if stats is None:
                raise
            return FisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge
            )


class HiddenMarkovFisherView(FixedFisherView):
    """Observed Fisher view for HMMs via forward-backward statistics.

    The per-observation vectors are posterior-expected complete-data
    sufficient statistics: initial-state counts, transition counts, per-state
    emission statistics, optional length statistics, and state occupancies
    when the model accumulator exposes them.  For finite enumerable HMMs, the
    full model Fisher is the exact observed covariance of these statistics
    under the model distribution.  For continuous or otherwise non-enumerable
    HMMs, diagonal model moments remain available; use
    observed_fisher_information() for empirical full covariance on data.
    """

    _max_model_enum_terms = 100000
    _model_mass_tol = 1.0e-8

    def __init__(self, dist: Any) -> None:
        self.dist = dist
        self.topic_views = [to_fisher(d) for d in dist.topics]
        self.len_view = None if _is_null_dist(getattr(dist, "len_dist", None)) else to_fisher(dist.len_dist)
        self._estimator = dist.estimator()
        self._has_state_counts = hasattr(dist, "n_states")
        self._model_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._diag_model_cache: tuple[np.ndarray, np.ndarray] | None = None
        super().__init__(dist, self._labels_from_children())

    def _num_states(self) -> int:
        if hasattr(self.dist, "n_states"):
            return int(self.dist.n_states)
        return int(self.dist.num_states)

    def _labels_from_children(self) -> list[Path]:
        k = self._num_states()
        labels: list[Path] = [("init", str(i)) for i in range(k)]
        if self._has_state_counts:
            labels.extend(("state", str(i)) for i in range(k))
        labels.extend(("transition", str(i), str(j)) for i in range(k) for j in range(k))
        for i, view in enumerate(self.topic_views):
            labels.extend(("emission", str(i)) + label for label in view.vectorizer.labels)
        if self.len_view is not None:
            labels.extend(("length",) + label for label in self.len_view.vectorizer.labels)
        return labels

    def _refresh_labels(self) -> None:
        self.labels = self._labels_from_children()
        self.vectorizer = SufficientStatisticVectorizer(self.labels)
        self._model_cache = None
        self._diag_model_cache = None

    def _accumulator_value_rows(self, enc_data: Any, model: Any | None = None) -> list[Any]:
        model = self.dist if model is None else model
        n = self._n_encoded(enc_data, model)
        values = []
        for i in range(n):
            weights = np.zeros(n, dtype=np.float64)
            weights[i] = 1.0
            acc = self._estimator.accumulator_factory().make()
            acc.seq_update(enc_data, weights, model)
            values.append(acc.value())
        return values

    def _matrix_from_values(self, values: Sequence[Any]) -> np.ndarray:
        if not values:
            return np.zeros((0, len(self.labels)), dtype=np.float64)

        init_idx, state_idx, trans_idx, topic_idx, len_idx = (
            (1, 2, 3, 4, 5) if self._has_state_counts else (0, None, 1, 2, 3)
        )
        init = np.vstack([np.asarray(v[init_idx], dtype=np.float64) for v in values])
        trans = np.vstack([np.asarray(v[trans_idx], dtype=np.float64).reshape(-1) for v in values])
        blocks = [init]
        if state_idx is not None:
            blocks.append(np.vstack([np.asarray(v[state_idx], dtype=np.float64) for v in values]))
        blocks.append(trans)

        for s, view in enumerate(self.topic_views):
            emission_values = [v[topic_idx][s] for v in values]
            blocks.append(_structured_values_matrix(view, emission_values))

        if self.len_view is not None:
            len_values = [v[len_idx] for v in values]
            blocks.append(_structured_values_matrix(self.len_view, len_values))

        self._refresh_labels()
        return np.hstack(blocks)

    @staticmethod
    def _sequence_forward_backward(
        log_b: np.ndarray, init: np.ndarray, transition: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, k = log_b.shape
        gamma = np.zeros((n, k), dtype=np.float64)
        trans = np.zeros((k, k), dtype=np.float64)
        init_row = np.zeros(k, dtype=np.float64)
        if n == 0:
            return init_row, gamma, trans

        row_max = np.max(log_b, axis=1, keepdims=True)
        safe_max = np.where(np.isfinite(row_max), row_max, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            obs = np.exp(log_b - safe_max)
        obs[~np.isfinite(obs)] = 0.0

        alpha = np.zeros((n, k), dtype=np.float64)
        scale = np.zeros(n, dtype=np.float64)
        alpha[0] = np.asarray(init, dtype=np.float64) * obs[0]
        scale[0] = alpha[0].sum()
        if scale[0] <= 0.0 or not np.isfinite(scale[0]):
            return init_row, gamma, trans
        alpha[0] /= scale[0]

        a_mat = np.asarray(transition, dtype=np.float64)
        for t in range(1, n):
            alpha[t] = np.dot(alpha[t - 1], a_mat) * obs[t]
            scale[t] = alpha[t].sum()
            if scale[t] <= 0.0 or not np.isfinite(scale[t]):
                return init_row, gamma, trans
            alpha[t] /= scale[t]

        gamma[-1] = alpha[-1]
        beta = np.ones(k, dtype=np.float64)
        for t in range(n - 2, -1, -1):
            bb = obs[t + 1] * beta
            denom = scale[t + 1] if scale[t + 1] > 0.0 else 1.0
            xi = alpha[t][:, None] * a_mat * bb[None, :] / denom
            xi_sum = xi.sum()
            if xi_sum > 0.0 and np.isfinite(xi_sum):
                xi /= xi_sum
                gamma[t] = xi.sum(axis=1)
                trans += xi
            beta = np.dot(a_mat, bb) / denom

        init_row = gamma[0].copy()
        return init_row, gamma, trans

    def _emission_log_matrix(self, enc_obs: Any, model: Any) -> np.ndarray:
        k = self._num_states()
        return np.asarray([model.topics[i].seq_log_density(enc_obs) for i in range(k)], dtype=np.float64).T

    def _hmm_rows_from_indexed_encoding(
        self,
        lengths: np.ndarray,
        enc_obs: Any,
        len_enc: Any,
        row_indices: Sequence[np.ndarray],
        flat_to_row: np.ndarray,
        model: Any,
    ) -> np.ndarray:
        lengths = np.asarray(lengths, dtype=np.int64)
        n = len(lengths)
        k = self._num_states()
        total = int(len(flat_to_row))

        init = np.zeros((n, k), dtype=np.float64)
        gamma = np.zeros((total, k), dtype=np.float64)
        trans = np.zeros((n, k, k), dtype=np.float64)

        if total > 0:
            log_b_all = self._emission_log_matrix(enc_obs, model)
            for i, rows in enumerate(row_indices):
                rows = np.asarray(rows, dtype=np.int64)
                if len(rows) == 0:
                    continue
                init_i, gamma_i, trans_i = self._sequence_forward_backward(log_b_all[rows], model.w, model.transitions)
                init[i] = init_i
                gamma[rows] = gamma_i
                trans[i] = trans_i

        blocks = [init]
        if self._has_state_counts:
            state = np.zeros((n, k), dtype=np.float64)
            if total > 0:
                np.add.at(state, np.asarray(flat_to_row, dtype=np.int64), gamma)
            blocks.append(state)
        blocks.append(trans.reshape((n, k * k)))

        for s, view in enumerate(self.topic_views):
            d = len(view.vectorizer.labels)
            emission = np.zeros((n, d), dtype=np.float64)
            if total > 0:
                flat_stats = view.seq_expected_statistics(enc_obs, estimate=model.topics[s])
                if flat_stats.shape[1] != d:
                    d = flat_stats.shape[1]
                    emission = np.zeros((n, d), dtype=np.float64)
                np.add.at(emission, np.asarray(flat_to_row, dtype=np.int64), gamma[:, [s]] * flat_stats)
            blocks.append(emission)

        if self.len_view is not None:
            blocks.append(self.len_view.seq_expected_statistics(len_enc, estimate=model.len_dist))

        self._refresh_labels()
        return np.hstack(blocks) if blocks else np.zeros((n, 0), dtype=np.float64)

    def _stats_hmm_rows_from_encoded(self, enc_data: Any, model: Any) -> np.ndarray:
        x0, x1 = enc_data
        if x1 is None:
            (tot_cnt, _, _, len_vec, idx_mat, idx_vec, enc_obs), _, len_enc = x0
            row_indices = [idx_mat[i, idx_mat[i] >= 0] for i in range(idx_mat.shape[0])]
            return self._hmm_rows_from_indexed_encoding(
                np.asarray(len_vec, dtype=np.int64),
                enc_obs,
                len_enc,
                row_indices,
                np.asarray(idx_vec, dtype=np.int64),
                model,
            )

        (idx, sz, enc_obs), len_enc = x1
        offsets = np.concatenate(([0], np.cumsum(np.asarray(sz, dtype=np.int64))))
        row_indices = [np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in range(len(sz))]
        return self._hmm_rows_from_indexed_encoding(
            np.asarray(sz, dtype=np.int64), enc_obs, len_enc, row_indices, np.asarray(idx, dtype=np.int64), model
        )

    def _bstats_hmm_rows_from_encoded(self, enc_data: Any, model: Any) -> np.ndarray:
        lengths, offsets, enc_obs, len_enc = enc_data
        lengths = np.asarray(lengths, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        row_indices = [np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in range(len(lengths))]
        flat_to_row = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        return self._hmm_rows_from_indexed_encoding(lengths, enc_obs, len_enc, row_indices, flat_to_row, model)

    def _fast_statistics_from_encoded(self, enc_data: Any, model: Any) -> np.ndarray:
        if isinstance(enc_data, tuple) and len(enc_data) == 2:
            return self._stats_hmm_rows_from_encoded(enc_data, model)
        if isinstance(enc_data, tuple) and len(enc_data) == 4:
            return self._bstats_hmm_rows_from_encoded(enc_data, model)
        raise NotImplementedError

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        enc = _seq_encode_model(self.dist if estimate is None else estimate, list(data))
        return self._statistics_from_encoded(enc, estimate=estimate)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        model = self.dist if estimate is None else estimate
        try:
            return self._fast_statistics_from_encoded(enc_data, model)
        except NotImplementedError:
            return self._matrix_from_values(self._accumulator_value_rows(enc_data, model))

    def structured_statistics(self, x: Any, estimate: Any | None = None, weight: float = 1.0) -> Any:
        model = self.dist if estimate is None else estimate
        enc = _seq_encode_model(model, [x])
        weights = np.asarray([weight], dtype=np.float64)
        acc = self._estimator.accumulator_factory().make()
        acc.seq_update(enc, weights, model)
        return acc.value()

    def _layout(self) -> tuple[int, list[int], int | None, int]:
        k = self._num_states()
        dims = [len(view.mean_statistics()) for view in self.topic_views]
        len_offset = k + (k if self._has_state_counts else 0) + k * k + sum(dims)
        total = len_offset + (0 if self.len_view is None else len(self.len_view.mean_statistics()))
        return k, dims, len_offset if self.len_view is not None else None, total

    def _inc_state(
        self,
        state: int,
        init: bool,
        prev_state: int | None,
        total: int,
        offsets: Sequence[int],
        emission_mu: Sequence[np.ndarray],
        emission_second: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        k = self._num_states()
        inc = np.zeros(total, dtype=np.float64)
        inc2 = np.zeros(total, dtype=np.float64)
        if init:
            inc[state] = 1.0
            inc2[state] = 1.0
        transition_offset = k
        if self._has_state_counts:
            inc[k + state] = 1.0
            inc2[k + state] = 1.0
            transition_offset += k
        if prev_state is not None:
            j = transition_offset + prev_state * k + state
            inc[j] = 1.0
            inc2[j] = 1.0
        s0 = offsets[state]
        s1 = s0 + len(emission_mu[state])
        inc[s0:s1] = emission_mu[state]
        inc2[s0:s1] = emission_second[state]
        return inc, inc2

    def _path_moments_for_length(
        self,
        n: int,
        total_no_len: int,
        offsets: Sequence[int],
        emission_mu: Sequence[np.ndarray],
        emission_second: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        k = self._num_states()
        if n <= 0:
            return np.zeros(total_no_len, dtype=np.float64), np.zeros(total_no_len, dtype=np.float64)

        pi = np.asarray(self.dist.w, dtype=np.float64)
        trans = np.asarray(self.dist.transitions, dtype=np.float64)
        p_state = pi.copy()
        first = np.zeros((k, total_no_len), dtype=np.float64)
        second = np.zeros((k, total_no_len), dtype=np.float64)
        for s in range(k):
            inc, inc2 = self._inc_state(s, True, None, total_no_len, offsets, emission_mu, emission_second)
            first[s] = pi[s] * inc
            second[s] = pi[s] * inc2

        for _ in range(1, n):
            next_p = np.zeros(k, dtype=np.float64)
            next_first = np.zeros_like(first)
            next_second = np.zeros_like(second)
            for prev in range(k):
                if p_state[prev] <= 0.0:
                    continue
                for s in range(k):
                    a = trans[prev, s]
                    if a <= 0.0:
                        continue
                    inc, inc2 = self._inc_state(s, False, prev, total_no_len, offsets, emission_mu, emission_second)
                    next_p[s] += p_state[prev] * a
                    next_first[s] += a * (first[prev] + p_state[prev] * inc)
                    next_second[s] += a * (second[prev] + 2.0 * inc * first[prev] + p_state[prev] * inc2)
            p_state = next_p
            first = next_first
            second = next_second

        return first.sum(axis=0), second.sum(axis=0)

    def _diagonal_model_moments(self) -> tuple[np.ndarray, np.ndarray]:
        if self._diag_model_cache is not None:
            return self._diag_model_cache

        support = _length_support(self.dist.len_dist)
        if support is None:
            raise NotImplementedError("HMM model Fisher requires a supported length distribution")
        lengths, probs = support
        k, dims, len_offset, total = self._layout()
        offsets = []
        pos = k + (k if self._has_state_counts else 0) + k * k
        for dim in dims:
            offsets.append(pos)
            pos += dim

        emission_mu = [np.asarray(view.mean_statistics(), dtype=np.float64) for view in self.topic_views]
        emission_second = [_second_diag_from_view(view) for view in self.topic_views]
        total_no_len = pos

        mean = np.zeros(total, dtype=np.float64)
        second = np.zeros(total, dtype=np.float64)
        len_mat = None
        if self.len_view is not None:
            len_mat = self.len_view.expected_statistics_matrix(data=[int(round(v)) for v in lengths])

        for r, (n_float, p) in enumerate(zip(lengths, probs)):
            n = max(int(round(n_float)), 0)
            m, q = self._path_moments_for_length(n, total_no_len, offsets, emission_mu, emission_second)
            row_mean = np.zeros(total, dtype=np.float64)
            row_second = np.zeros(total, dtype=np.float64)
            row_mean[:total_no_len] = m
            row_second[:total_no_len] = q
            if len_mat is not None and len_offset is not None:
                row_mean[len_offset:] = len_mat[r]
                row_second[len_offset:] = len_mat[r] * len_mat[r]
            mean += p * row_mean
            second += p * row_second

        self._diag_model_cache = (mean, np.maximum(second - mean * mean, 0.0))
        return self._diag_model_cache

    def _enumerated_model_mean_cov(self) -> tuple[np.ndarray, np.ndarray]:
        if self._model_cache is not None:
            return self._model_cache

        values: list[Any] = []
        probs: list[float] = []
        try:
            iterator = iter(self.dist.enumerator())
            exhausted = False
            for _ in range(self._max_model_enum_terms):
                try:
                    value, log_prob = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                if np.isfinite(log_prob):
                    values.append(value)
                    probs.append(float(math.exp(log_prob)))
            if not exhausted:
                raise NotImplementedError(
                    "HMM full model Fisher requires finite enumerable support; use observed_fisher_information()."
                )
        except NotImplementedError:
            raise
        except Exception as exc:
            raise NotImplementedError(
                "HMM full model Fisher requires finite enumerable support; use observed_fisher_information()."
            ) from exc

        if not values:
            raise NotImplementedError("HMM full model Fisher requires non-empty finite support.")

        weights = np.asarray(probs, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total) or abs(total - 1.0) > self._model_mass_tol:
            raise NotImplementedError("HMM finite support did not sum to one; use observed_fisher_information().")
        weights /= total

        stats = self.expected_statistics_matrix(data=values)
        mean = np.dot(weights, stats)
        second = np.dot((weights[:, None] * stats).T, stats)
        cov = second - np.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        diag = np.maximum(np.diag(cov), 0.0)
        cov[np.diag_indices_from(cov)] = diag
        self._model_cache = (mean, cov)
        return self._model_cache

    def _model_mean(self) -> np.ndarray:
        try:
            return self._enumerated_model_mean_cov()[0]
        except NotImplementedError:
            return self._diagonal_model_moments()[0]

    def _model_fisher(self) -> np.ndarray:
        try:
            return self._enumerated_model_mean_cov()[1]
        except NotImplementedError:
            return np.diag(self._diagonal_model_moments()[1])

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        if not diagonal:
            try:
                info = self._enumerated_model_mean_cov()[1]
                return info + np.eye(info.shape[0]) * ridge
            except NotImplementedError:
                if stats is not None:
                    return FisherView.fisher_information(self, stats=stats, diagonal=False, ridge=ridge)
                raise NotImplementedError(
                    "HMM full model Fisher requires finite enumerable support; "
                    "use diagonal=True or observed_fisher_information()."
                )
        try:
            return FixedFisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge, **kwargs)
        except NotImplementedError:
            return FisherView.fisher_information(self, stats=stats, diagonal=diagonal, ridge=ridge, **kwargs)

    def fisher_vectors(
        self,
        stats: np.ndarray | None = None,
        metric: str = "diagonal",
        center: np.ndarray | None = None,
        fisher: np.ndarray | None = None,
        ridge: float = 1.0e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        if metric == "full" and fisher is None:
            try:
                mean, info = self._enumerated_model_mean_cov()
            except NotImplementedError:
                if stats is not None:
                    raise NotImplementedError(
                        "HMM full model Fisher vectors require finite enumerable support; "
                        'use metric="diagonal" or observed_fisher_vectors().'
                    )
                raise
            if stats is None:
                stats = self.expected_statistics_matrix(**kwargs)
            return FisherView.fisher_vectors(
                self, stats=stats, metric="full", center=mean if center is None else center, fisher=info, ridge=ridge
            )
        try:
            return FixedFisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge, **kwargs
            )
        except NotImplementedError:
            if stats is None:
                stats = self.expected_statistics_matrix(**kwargs)
            return FisherView.fisher_vectors(
                self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge
            )


def _as_float_array(data: Any) -> np.ndarray:
    return np.asarray(data, dtype=np.float64)


# CountFisherView shared extractors (the per-family mean_var / encoded helpers now live in each
# count distribution's own module, which imports these and CountFisherView).
def _count_data(data: Any) -> np.ndarray:
    return _as_float_array(data)


def _identity_encoded(enc_data: Any) -> np.ndarray:
    return np.asarray(enc_data, dtype=np.float64)


def to_fisher(dist: Any, **kwargs: Any) -> FisherView:
    """Return a FisherView for ``dist`` via its own ``to_fisher`` hook.

    Fisher views are co-located with each distribution: a distribution owns its view by overriding
    ``ProbabilityDistribution.to_fisher``. This module keeps only the shared base machinery
    (FisherView/FixedFisherView/SufficientStatisticVectorizer and the reusable CountFisherView /
    EmpiricalMetricFixedFisherView helpers) plus :func:`_legacy_to_fisher`, the type-name dispatch for
    families not yet migrated to a per-file hook (and the generic fallback).
    """
    return dist.to_fisher(**kwargs)


def _legacy_to_fisher(dist: Any, **kwargs: Any) -> FisherView:
    tname = type(dist).__name__

    if tname == "SequenceDistribution" and hasattr(dist, "dist"):
        return SequenceFisherView(dist)

    if tname == "MultinomialDistribution" and hasattr(dist, "dist") and hasattr(dist, "len_dist"):
        return MultinomialFisherView(dist)

    if tname == "OptionalDistribution" and hasattr(dist, "dist"):
        return OptionalFisherView(dist)

    if tname == "WeightedDistribution" and hasattr(dist, "dist"):
        return WeightedFisherView(dist)

    if tname == "SelectDistribution" and hasattr(dist, "dists"):
        return SelectFisherView(dist)

    if (
        tname in ("HiddenMarkovModelDistribution", "QuantizedHiddenMarkovModelDistribution")
        and hasattr(dist, "topics")
        and hasattr(dist, "transitions")
    ):
        return HiddenMarkovFisherView(dist)

    if (
        tname == "HeterogeneousPCFGDistribution"
        and hasattr(dist, "terminal_rules")
        and hasattr(dist, "_inside_outside")
    ):
        return HeterogeneousPCFGFisherView(dist)

    if tname == "CompositeDistribution" and hasattr(dist, "dists"):
        return CompositeFisherView(dist)

    if tname == "MixtureDistribution" and hasattr(dist, "components") and hasattr(dist, "w"):
        return MixtureFisherView(dist)

    if tname == "HierarchicalMixtureDistribution" and hasattr(dist, "to_mixture"):
        return to_fisher(dist.to_mixture(), **kwargs)

    if tname == "JointMixtureDistribution" and hasattr(dist, "components1") and hasattr(dist, "components2"):
        return JointMixtureFisherView(dist)

    return FisherView(dist, **kwargs)
