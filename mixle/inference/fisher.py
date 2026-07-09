"""Generic Fisher-geometry views for mixle distributions.

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


# --- sufficient-statistic vectorizer (the Fisher accumulator core) ----------
class SufficientStatisticVectorizer:
    """Flatten nested sufficient-statistic structures into numeric vectors.

    Accumulators in mixle return tuples, arrays, dictionaries, and scalars.  This
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
        """Learn vector coordinates from structured sufficient-statistic values."""
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
        """Extend vector coordinates with labels found in additional values."""
        for value in values:
            for label, _ in self._items(value):
                if label not in self._index:
                    self._index[label] = len(self.labels)
                    self.labels.append(label)
        return self

    def transform(self, values: Sequence[Any], extend: bool = False) -> np.ndarray:
        """Vectorize structured sufficient-statistic values into a dense matrix."""
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
        """Learn coordinates and vectorize values in one pass."""
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
        """Return learned coordinate labels as dotted strings."""
        return [".".join(p) for p in self.labels]


# --- FisherView base ---------------------------------------------------------
class FisherView:
    """Accumulator-backed Fisher-geometry view of a distribution.

    Args:
        dist: mixle.stats or mixle.bstats distribution.
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
        """Alias for posterior-expected structured sufficient statistics."""
        return self.structured_statistics(x, estimate=estimate, weight=weight)

    def sufficient_statistics(
        self, x: Any, estimate: Any | None = None, vectorizer: SufficientStatisticVectorizer | None = None
    ) -> np.ndarray:
        """Return vectorized sufficient statistics for one observation."""
        ss = self.structured_statistics(x, estimate=estimate)
        vec = vectorizer if vectorizer is not None else SufficientStatisticVectorizer().fit([ss])
        return vec.transform([ss])[0]

    def expected_sufficient_statistics(
        self, x: Any, estimate: Any | None = None, vectorizer: SufficientStatisticVectorizer | None = None
    ) -> np.ndarray:
        """Alias for vectorized posterior-expected sufficient statistics."""
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
        """Alias for a matrix of posterior-expected sufficient statistics."""
        return self.statistics_matrix(data=data, enc_data=enc_data, estimate=estimate, vectorizer=vectorizer, fit=fit)

    def seq_expected_statistics(
        self,
        enc_data: Any,
        estimate: Any | None = None,
        vectorizer: SufficientStatisticVectorizer | None = None,
        fit: bool = True,
    ) -> np.ndarray:
        """Return posterior-expected statistics for encoded observations."""
        return self.expected_statistics_matrix(enc_data=enc_data, estimate=estimate, vectorizer=vectorizer, fit=fit)

    @staticmethod
    def _center(stats: np.ndarray, center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(stats, dtype=np.float64)
        mu = x.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
        return x - mu.reshape((1, -1)), mu

    def mean_statistics(self, stats: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        """Return empirical mean sufficient statistics."""
        if stats is None:
            stats = self.statistics_matrix(**kwargs)
        return np.asarray(stats, dtype=np.float64).mean(axis=0)

    def score_center(self, stats: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        """Return the center used for score or Fisher-vector whitening."""
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
        """Return centered statistics whitened by an observed Fisher metric."""
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
        """Return a single Fisher vector for one observation."""
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


# --- shared view bases (fixed-coordinate / count) ---------------------------
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
        """Return fixed-coordinate sufficient statistics for one observation."""
        return self._statistics_from_data([x], estimate=estimate)[0] * weight

    def sufficient_statistics(
        self, x: Any, estimate: Any | None = None, vectorizer: SufficientStatisticVectorizer | None = None
    ) -> np.ndarray:
        """Return projected sufficient statistics for one observation."""
        mat = self._statistics_from_data([x], estimate=estimate)
        return self._project_matrix(mat, vectorizer, fit=vectorizer is None)[0]

    def seq_structured_statistics(self, enc_data: Any, estimate: Any | None = None) -> list[Any]:
        """Return fixed-coordinate statistics for encoded observations."""
        return [row for row in self._statistics_from_encoded(enc_data, estimate=estimate)]

    def statistics_matrix(
        self,
        data: Sequence[Any] | None = None,
        enc_data: Any | None = None,
        estimate: Any | None = None,
        vectorizer: SufficientStatisticVectorizer | None = None,
        fit: bool = True,
    ) -> np.ndarray:
        """Return projected fixed-coordinate statistics for raw or encoded data."""
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
        """Return model-mean or empirical sufficient statistics."""
        if model or stats is None:
            return self._model_mean()
        return np.asarray(stats, dtype=np.float64).mean(axis=0)

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        """Return model Fisher information in fixed coordinates."""
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
        """Return centered or whitened Fisher vectors in fixed coordinates."""
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
    """Fixed Fisher view for scalar count-like families with analytic mean and variance."""

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


# --- view helpers (info extraction / encoding / finite-support enumeration) --
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
        # IntegerCategorical views (now defined in mixle.stats.univariate.discrete.integer_categorical) self-identify with a
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


# --- empirical-metric base view ---------------------------------------------
class EmpiricalMetricFixedFisherView(FixedFisherView):
    """Fixed-coordinate view whose whitening falls back to empirical Fisher."""

    def mean_statistics(self, stats: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        """Return empirical mean statistics for the supplied data."""
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        return np.asarray(stats, dtype=np.float64).mean(axis=0)

    def fisher_information(
        self, stats: np.ndarray | None = None, diagonal: bool = False, ridge: float = 1.0e-8, **kwargs: Any
    ) -> np.ndarray:
        """Return empirical Fisher information for this fixed-coordinate view."""
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
        """Return Fisher vectors using the empirical Fisher metric."""
        if stats is None:
            stats = self.expected_statistics_matrix(**kwargs)
        return FisherView.fisher_vectors(self, stats=stats, metric=metric, center=center, fisher=fisher, ridge=ridge)


# --- data coercion + the public to_fisher dispatch --------------------------
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
    # All families now own their Fisher view via ProbabilityDistribution.to_fisher; this remains the
    # generic accumulator-backed fallback for any distribution without a specialized view.
    return FisherView(dist, **kwargs)
