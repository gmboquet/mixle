"""Dirichlet distributions, estimators, samplers, accumulators, and encoders.

Observations are simplex vectors represented as ``list[float]`` or
``np.ndarray``. For a ``K``-dimensional Dirichlet, the log-density is
``-log(B(alpha)) + sum_k (alpha_k - 1) * log(x_k)`` when ``x`` lies on the
simplex and ``-inf`` otherwise.
"""

import operator
from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

from mixle.inference.fisher import FixedFisherView
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.special import *

_MIN_DIRICHLET_ALPHA = 1.0e-10
_MAX_DIRICHLET_ALPHA = 1.0e10
_MAX_DIRICHLET_ITERATIONS = 10000
_SIMPLEX_SUM_RTOL = 1.0e-10
_SIMPLEX_SUM_ATOL = 1.0e-12


class DirichletConvergenceError(RuntimeError):
    """Raised when concentration fitting cannot produce a verified finite solution."""


def _validated_dimension(value: Any, *, optional: bool) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Dirichlet dimension must be a positive integer.")
    try:
        dimension = operator.index(value)
    except TypeError as exc:
        raise TypeError("Dirichlet dimension must be a positive integer.") from exc
    if dimension <= 0:
        raise ValueError("Dirichlet dimension must be positive.")
    return int(dimension)


def _validated_weight(value: Any, name: str = "weight") -> float:
    if np.ndim(value) != 0:
        raise ValueError("Dirichlet %s must be a finite non-negative scalar." % name)
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet %s must be a finite non-negative scalar." % name) from exc
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("Dirichlet %s must be a finite non-negative scalar." % name)
    return weight


def _validated_weight_vector(value: Any, count: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet weights must be numeric.") from exc
    if weights.shape != (count,):
        raise ValueError("Dirichlet weights must have exact shape (%d,)." % count)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Dirichlet weights must be finite and non-negative.")
    return weights


def _simplex_row_mask(raw: np.ndarray) -> np.ndarray:
    return (
        np.all(np.isfinite(raw), axis=1)
        & np.all(raw >= 0.0, axis=1)
        & np.isclose(
            raw.sum(axis=1),
            1.0,
            rtol=_SIMPLEX_SUM_RTOL,
            atol=_SIMPLEX_SUM_ATOL,
        )
    )


def _encoded_raw_matrix(value: Any, dimension: int | None = None) -> np.ndarray:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("Dirichlet encoded data must be a three-item tuple.")
    try:
        raw = np.asarray(value[1], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet encoded observations must be numeric.") from exc
    if raw.ndim != 2:
        raise ValueError("Dirichlet encoded observations must have exact shape (n, d).")
    if raw.shape[1] == 0:
        raise ValueError("Dirichlet encoded observations cannot have zero dimension.")
    if dimension is not None and raw.shape[1] != dimension:
        raise ValueError("Dirichlet encoded observations must have %d columns." % dimension)
    for index, name in ((0, "log"), (2, "squared")):
        try:
            companion = np.asarray(value[index])
        except (TypeError, ValueError) as exc:
            raise ValueError("Dirichlet encoded %s payload must be array-like." % name) from exc
        if companion.shape != raw.shape:
            raise ValueError("Dirichlet encoded %s payload must match raw observation shape." % name)
    return raw


def _interior_simplex_matrix(value: Any, dimension: int | None = None) -> np.ndarray:
    raw = _encoded_raw_matrix(value, dimension)
    if not np.all(_simplex_row_mask(raw)):
        raise ValueError("Dirichlet fitting requires finite non-negative simplex observations.")
    if np.any(raw <= 0.0):
        raise ValueError(
            "Dirichlet fitting requires strictly positive simplex coordinates; exact boundary "
            "observations have infinite log sufficient statistics."
        )
    return raw


def _interior_simplex_vector(value: Any, dimension: int | None = None) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet fitting observations must be numeric vectors.") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("Dirichlet fitting observations must be non-empty vectors.")
    if dimension is not None and raw.shape != (dimension,):
        raise ValueError("Dirichlet fitting observation dimension changed.")
    if not _simplex_row_mask(raw[None, :])[0]:
        raise ValueError("Dirichlet fitting requires finite non-negative simplex observations.")
    if np.any(raw <= 0.0):
        raise ValueError(
            "Dirichlet fitting requires strictly positive simplex coordinates; exact boundary "
            "observations have infinite log sufficient statistics."
        )
    return raw


def _validated_dirichlet_statistics(
    value: Any,
    *,
    configured_dim: int | None,
    allow_unsized: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None, np.ndarray | None, int | None]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("Dirichlet sufficient statistics must be a four-item tuple.")
    count = _validated_weight(value[0], "sufficient-statistic count")
    payload = value[1:]
    if all(item is None for item in payload):
        if not allow_unsized or count != 0.0:
            raise ValueError("Only a zero-count Dirichlet statistic may be unsized.")
        return count, None, None, None, None
    if any(item is None for item in payload):
        raise ValueError("Dirichlet sufficient-statistic vectors must either all be present or all be None.")
    converted: list[np.ndarray] = []
    for name, item in zip(("sum_of_logs", "sum", "sum2"), payload):
        try:
            vector = np.asarray(item, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Dirichlet %s statistic must be a numeric vector." % name) from exc
        if vector.ndim != 1 or vector.size == 0 or np.any(~np.isfinite(vector)):
            raise ValueError("Dirichlet %s statistic must be a non-empty finite vector." % name)
        converted.append(vector.copy())
    dimension = len(converted[0])
    if any(len(vector) != dimension for vector in converted[1:]):
        raise ValueError("Dirichlet sufficient-statistic vectors must have identical widths.")
    if configured_dim is not None and dimension != configured_dim:
        raise ValueError(
            "Dirichlet sufficient-statistic width %d does not match configured dimension %d."
            % (dimension, configured_dim)
        )
    sum_of_logs, sum_v, sum_v2 = converted
    tolerance = 1.0e-9 * max(1.0, count)
    if np.any(sum_of_logs > tolerance):
        raise ValueError("Dirichlet sum-of-logs statistics cannot be positive.")
    if np.any(sum_v < -tolerance) or np.any(sum_v2 < -tolerance):
        raise ValueError("Dirichlet moment statistics cannot be negative.")
    if count == 0.0:
        if (
            np.any(np.abs(sum_of_logs) > tolerance)
            or np.any(np.abs(sum_v) > tolerance)
            or np.any(np.abs(sum_v2) > tolerance)
        ):
            raise ValueError("Zero-count Dirichlet sufficient statistics must contain zero vectors.")
    else:
        if not np.isclose(
            float(sum_v.sum()),
            count,
            rtol=_SIMPLEX_SUM_RTOL,
            atol=max(_SIMPLEX_SUM_ATOL, tolerance),
        ):
            raise ValueError("Dirichlet summed coordinates must total the observation weight.")
        if np.any(sum_v > count + tolerance) or np.any(sum_v2 > sum_v + tolerance):
            raise ValueError("Dirichlet moment statistics are inconsistent with simplex observations.")
    return count, sum_of_logs, sum_v, sum_v2, dimension


def _safe_simplex_mean(x: np.ndarray, dim: int) -> np.ndarray:
    rv = np.asarray(x, dtype=float).copy()
    if rv.size != dim:
        rv = np.ones(dim, dtype=float)
    rv[~np.isfinite(rv)] = 0.0
    rv = np.maximum(rv, 0.0)
    total = rv.sum()
    if total <= 0.0:
        rv = np.ones(dim, dtype=float) / float(dim)
    else:
        rv /= total
    rv = np.maximum(rv, _MIN_DIRICHLET_ALPHA)
    rv /= rv.sum()
    return rv


def _mean_from_mean_log(mean_log_p: np.ndarray, dim: int) -> np.ndarray:
    mlp = np.asarray(mean_log_p, dtype=float).copy()
    finite = np.isfinite(mlp)
    if not np.any(finite):
        return np.ones(dim, dtype=float) / float(dim)
    floor = np.min(mlp[finite])
    mlp[~finite] = floor
    mlp -= np.max(mlp)
    rv = np.exp(np.maximum(mlp, -745.0))
    return _safe_simplex_mean(rv, dim)


def _initial_dirichlet_alpha(
    mean_v: np.ndarray, mean_v2: np.ndarray | None = None, mean_log_p: np.ndarray | None = None
) -> np.ndarray:
    mean = _safe_simplex_mean(mean_v, len(mean_v))
    alpha0 = 1.0

    if mean_v2 is not None:
        second = np.asarray(mean_v2, dtype=float).copy()
        second[~np.isfinite(second)] = np.nan
        var = second - mean * mean
        good = (mean > _MIN_DIRICHLET_ALPHA) & (mean < 1.0 - _MIN_DIRICHLET_ALPHA) & np.isfinite(var) & (var > 0.0)
        if np.any(good):
            cand = mean[good] * (1.0 - mean[good]) / var[good] - 1.0
            cand = cand[np.isfinite(cand) & (cand > 0.0)]
            if cand.size > 0:
                alpha0 = float(np.median(cand))
        elif np.all(np.isfinite(second)) and np.all(np.abs(second - mean * mean) <= 1.0e-14):
            alpha0 = _MAX_DIRICHLET_ALPHA / float(len(mean))

    if (not np.isfinite(alpha0) or alpha0 <= 0.0) and mean_log_p is not None:
        alpha0 = float(len(mean))
    if not np.isfinite(alpha0) or alpha0 <= 0.0:
        alpha0 = 1.0

    alpha0 = min(_MAX_DIRICHLET_ALPHA, max(_MIN_DIRICHLET_ALPHA * len(mean), alpha0))
    alpha = mean * alpha0
    return np.clip(alpha, _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)


def _valid_alpha(alpha: np.ndarray, dim: int | None = None) -> bool:
    arr = np.asarray(alpha, dtype=float)
    return (dim is None or arr.size == dim) and np.all(np.isfinite(arr)) and np.all(arr > 0.0)


def dirichlet_param_solve(
    alpha: np.ndarray, mean_log_p: np.ndarray, delta: float, max_iter: int = _MAX_DIRICHLET_ITERATIONS
) -> tuple[np.ndarray, int]:
    """Solve for Dirichlet concentrations with safeguarded Newton updates.

    Args:
        alpha (np.ndarray): Numpy array of Dirichlet parameters (all entries should be non-negative).
        mean_log_p (np.ndarray): Sufficient statistic (1/N) sum_{i=1}^{N} log(x_{i,k}), where N is the number of
            observations.
        delta (float): Tolerance for convergence of Newton-Method.

    Returns:
        Tuple[np.ndarray, int] containing the alpha estimate and number of solver iterations.

    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 1 or alpha.size == 0 or not _valid_alpha(alpha):
        raise ValueError("Dirichlet solver initial alpha must be a non-empty finite positive vector.")
    dim = len(alpha)
    try:
        tolerance = float(delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet solver delta must be finite and positive.") from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Dirichlet solver delta must be finite and positive.")
    max_iterations = _validated_dimension(max_iter, optional=False)
    mlp = np.asarray(mean_log_p, dtype=np.float64)
    if mlp.shape != (dim,) or np.any(~np.isfinite(mlp)) or np.any(mlp > 0.0):
        raise ValueError("Dirichlet solver mean-log statistics must be a finite non-positive vector.")
    alpha = np.clip(alpha.copy(), _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)

    def objective(candidate: np.ndarray) -> float:
        return float(gammaln(candidate.sum()) - gammaln(candidate).sum() + np.dot(candidate - 1.0, mlp))

    for count in range(1, max_iterations + 1):
        old_alpha = alpha.copy()
        a_sum = float(alpha.sum())
        gradient = digamma(a_sum) - digamma(alpha) + mlp
        diagonal = -trigamma(alpha)
        common = trigamma(a_sum)
        denominator = (1.0 / common) + np.sum(1.0 / diagonal)
        if (
            np.any(~np.isfinite(gradient))
            or np.any(~np.isfinite(diagonal))
            or not np.isfinite(denominator)
            or denominator == 0.0
        ):
            raise DirichletConvergenceError("Dirichlet Newton solver produced invalid derivatives.")
        adjustment = np.sum(gradient / diagonal) / denominator
        step = (gradient - adjustment) / diagonal
        old_objective = objective(old_alpha)
        scale = 1.0
        accepted = False
        while scale >= 2.0**-40:
            candidate = old_alpha - scale * step
            if (
                np.all(np.isfinite(candidate))
                and np.all(candidate > _MIN_DIRICHLET_ALPHA)
                and np.all(candidate < _MAX_DIRICHLET_ALPHA)
            ):
                candidate_objective = objective(candidate)
                if np.isfinite(candidate_objective) and candidate_objective >= old_objective - 1.0e-12:
                    alpha = candidate
                    accepted = True
                    break
            scale *= 0.5
        if not accepted:
            if float(np.max(np.abs(gradient))) <= tolerance:
                return old_alpha, count
            raise DirichletConvergenceError("Dirichlet Newton solver could not find a finite improving step.")
        denom = max(_MIN_DIRICHLET_ALPHA, float(alpha.sum()))
        d_alpha = float(np.abs(alpha - old_alpha).sum() / denom)
        if d_alpha <= tolerance:
            return alpha, count

    raise DirichletConvergenceError("Dirichlet Newton solver did not converge within %d iterations." % max_iterations)


def mpe(
    x0: np.ndarray, f: Callable[[np.ndarray], np.ndarray], eps: float, max_iter: int = 1000
) -> tuple[np.ndarray, int]:
    """Minimal polynomial extrapolation for accelerating the fixed-point iteration x_{n+1} = f(x_n).

    Args:
        x0 (np.ndarray): Starting point for the fixed-point iteration.
        f (Callable[[np.ndarray], np.ndarray]): Fixed-point map being iterated.
        eps (float): Tolerance on the absolute change between extrapolated iterates.

    Returns:
        Tuple[np.ndarray, int] containing the extrapolated fixed point and the iteration count.

    """
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.ndim != 1 or x0.size == 0 or not _valid_alpha(x0):
        raise ValueError("Dirichlet MPE initial alpha must be a non-empty finite positive vector.")
    try:
        tolerance = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dirichlet MPE tolerance must be finite and positive.") from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Dirichlet MPE tolerance must be finite and positive.")
    max_iterations = _validated_dimension(max_iter, optional=False)
    x0 = np.clip(x0, _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)
    x1 = np.clip(f(x0), _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)
    x2 = np.clip(f(x1), _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)
    x3 = np.clip(f(x2), _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)
    X = np.asarray([x0, x1, x2, x3])
    s0 = x3
    s = s0
    res = np.abs(x3 - x2).sum()
    its_cnt = 2

    while res > tolerance and its_cnt < max_iterations:
        y = np.clip(f(X[-1, :]), _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)
        dy = y - X[-1, :]
        U = (X[1:, :] - X[:-1, :]).T
        X2 = X[1:, :].T
        c = np.dot(np.linalg.pinv(U), dy)
        c *= -1
        denom = c.sum() + 1
        if not np.isfinite(denom) or abs(denom) <= _MIN_DIRICHLET_ALPHA:
            s = y
        else:
            s = (np.dot(X2, c) + y) / denom
        if not _valid_alpha(s, len(x0)):
            s = y
        s = np.clip(s, _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)

        res = np.abs(s - s0).sum()
        s0 = s
        X = np.concatenate((X, np.reshape(y, (1, -1))), axis=0)
        its_cnt += 1

    if res > tolerance:
        raise DirichletConvergenceError("Dirichlet MPE solver did not converge within %d iterations." % max_iterations)
    return s, its_cnt


def alpha_seq_lambda(mean_log_p: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Returns the fixed-point map for the Dirichlet alpha given sufficient statistic mean_log_p.

    Args:
        mean_log_p (np.ndarray): Mean of the log of the observed proportions.

    Returns:
        Callable mapping the current alpha to the next alpha iterate.

    """

    mean_log = np.asarray(mean_log_p, dtype=np.float64)
    if mean_log.ndim != 1 or mean_log.size == 0 or np.any(~np.isfinite(mean_log)) or np.any(mean_log > 0.0):
        raise ValueError("Dirichlet mean-log statistics must be a finite non-positive vector.")

    def next_alpha(current_alpha):
        current = np.asarray(current_alpha, dtype=np.float64)
        if current.shape != mean_log.shape or not _valid_alpha(current, len(mean_log)):
            raise ValueError("Dirichlet fixed-point iterate has invalid concentration geometry.")
        result = np.asarray(digammainv(mean_log + digamma(current.sum())), dtype=np.float64)
        if not _valid_alpha(result, len(mean_log)):
            raise DirichletConvergenceError("Dirichlet fixed-point map produced invalid concentrations.")
        return result

    return next_alpha


def find_alpha(current_alpha, mlp, thresh) -> tuple[np.ndarray, int]:
    """Solve for the Dirichlet alpha with MPE-accelerated fixed-point iteration.

    Args:
        current_alpha (np.ndarray): Initial estimate of the Dirichlet parameters.
        mlp (np.ndarray): Mean of the log of the observed proportions (sufficient statistic).
        thresh (float): Convergence tolerance.

    Returns:
        Tuple[np.ndarray, int] containing the estimate of alpha and the iteration count.

    """
    f = alpha_seq_lambda(mlp)
    alpha, its = mpe(current_alpha, f, thresh)
    if not _valid_alpha(alpha, len(current_alpha)):
        raise DirichletConvergenceError("Dirichlet MPE solver returned invalid concentrations.")
    return alpha, its


class DirichletFisherView(FixedFisherView):
    """Fisher view over log-coordinate sufficient statistics for a Dirichlet."""

    def __init__(self, dist: Any) -> None:
        alpha = np.asarray(dist.alpha, dtype=np.float64).reshape(-1)
        self.alpha = alpha
        self.dim = len(alpha)
        labels = [("log", str(i)) for i in range(self.dim)]
        labels.append(("count",))
        super().__init__(dist, labels)

    def _matrix_from_values(self, values: Any, log_values: Any | None = None) -> np.ndarray:
        if log_values is None:
            x = np.asarray(values, dtype=np.float64).reshape((-1, self.dim))
            log_x = np.log(np.maximum(x, np.finfo(np.float64).tiny))
        else:
            log_x = np.asarray(log_values, dtype=np.float64).reshape((-1, self.dim))
        return np.hstack((log_x, np.ones((log_x.shape[0], 1), dtype=np.float64)))

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        return self._matrix_from_values(data)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        if isinstance(enc_data, tuple):
            return self._matrix_from_values(None, enc_data[0])
        return self._matrix_from_values(enc_data)

    def _model_mean(self) -> np.ndarray:
        a0 = float(np.sum(self.alpha))
        return np.concatenate((digamma(self.alpha) - digamma(a0), np.asarray([1.0])))

    def _model_fisher(self) -> np.ndarray:
        a0 = float(np.sum(self.alpha))
        cov = np.diag(trigamma(self.alpha)) - trigamma(a0)
        out = np.zeros((self.dim + 1, self.dim + 1), dtype=np.float64)
        out[: self.dim, : self.dim] = cov
        return out


class DirichletDistribution(SequenceEncodableProbabilityDistribution):
    """Dirichlet distribution over probability vectors, with concentration parameters alpha."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for Dirichlet generated kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="explicit_stacked")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the Dirichlet distribution."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="dirichlet",
            distribution_type=cls,
            parameters=(ParameterSpec("alpha", constraint="positive_vector"),),
            statistics=(
                StatisticSpec("count"),
                StatisticSpec("sum_of_logs", kind="vector_moment"),
                StatisticSpec("sum", kind="vector_moment"),
                StatisticSpec("sum2", kind="vector_moment"),
            ),
            support="simplex",
            differentiable=False,
        )

    def __init__(self, alpha: list[float] | np.ndarray, name: str | None = None, keys: str | None = None) -> None:
        """Create a Dirichlet distribution with concentration vector ``alpha``.

        Args:
            alpha: One-dimensional concentration vector.
            name: Optional distribution name.
            keys: Optional key for merging sufficient statistics.

        Attributes:
            dim: Number of simplex coordinates.
            alpha: Concentration parameters.
            alpha_ma: Boolean mask of positive concentration entries.
            log_const: Log normalizing constant ``log(B(alpha))``.
            has_invalid: True when any concentration is non-positive.
            name: Optional distribution name.
            keys: Optional merge key.
        """
        temp_alpha = np.asarray(alpha, dtype=float)
        if (
            temp_alpha.ndim != 1
            or temp_alpha.size == 0
            or not np.all(np.isfinite(temp_alpha))
            or not np.all(temp_alpha > 0.0)
        ):
            raise ValueError("DirichletDistribution requires a non-empty vector of positive finite alpha values.")
        temp_mask = temp_alpha <= 0

        self.dim = len(temp_alpha)
        self.alpha = temp_alpha.copy()
        self.alpha_ma = ~temp_mask
        self.log_const = sum(gammaln(self.alpha)) - gammaln(sum(self.alpha))
        self.has_invalid = np.any(temp_mask)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the Dirichlet distribution."""
        s1 = repr(list(self.alpha))
        s2 = repr(self.name)
        s3 = repr(self.keys)
        return "DirichletDistribution(%s, name=%s, keys=%s)" % (s1, s2, s3)

    def get_parameters(self) -> np.ndarray:
        """Return the concentration vector alpha.

        Lets a DirichletDistribution serve as a conjugate prior (on a Categorical/Mixture weight
        simplex) under the unified Bayesian estimation protocol.
        """
        return self.alpha.copy()

    def cross_entropy(self, dist: "SequenceEncodableProbabilityDistribution") -> float:
        """Cross entropy -E_self[log dist(x)] for a Dirichlet argument.

        Accepts another :class:`DirichletDistribution` (full concentration vector) or a
        :class:`~mixle.stats.bayes.symmetric_dirichlet.SymmetricDirichletDistribution` (scalar concentration
        broadcast to this distribution's dimension). Both arise as the conjugate prior/posterior
        over the same simplex during variational Bayes (e.g. the ELBO global term in DPM).
        """
        from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution

        a = self.alpha
        if isinstance(dist, DirichletDistribution):
            aa = dist.alpha
        elif isinstance(dist, SymmetricDirichletDistribution):
            aa = dist.alpha * np.ones(self.dim)
        else:
            raise NotImplementedError(
                "DirichletDistribution.cross_entropy is only implemented for Dirichlet arguments (got %s)."
                % type(dist).__name__
            )
        return float(-((gammaln(np.sum(aa)) - np.sum(gammaln(aa))) + np.dot(digamma(a) - digamma(np.sum(a)), aa - 1)))

    def entropy(self) -> float:
        """Returns the differential entropy in nats."""
        a = self.alpha
        a0 = np.sum(a)
        return float(-((gammaln(a0) - np.sum(gammaln(a))) + np.dot(digamma(a) - digamma(a0), a - 1)))

    def density(self, x: list[float] | np.ndarray) -> float:
        """Evaluate the density of a dirichlet observation.

        See log_density() for details.

        Args:
            x (Union[List[float], np.ndarray]): A single dirichlet observation.

        Returns:
            Density evaluated at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: list[float] | np.ndarray) -> float:
        """Evaluate the log-density of a dirichlet observation.

        The log-density of a Dirichlet with dim = K, is given by

            log(p_mat(x)) = -log(Const) + sum_{k=0}^{K-1} (alpha_k -1)*log(x_k), for sum_k x_k = 1.0,

        else -inf. In above

            log(Const) = sum_{k=0}^{K-1} log(Gamma(alpha_k)) - log(Gamma(sum_{k=0}^{K-1} alpha_k)).

        Args:
            x (Union[List[float], np.ndarray]): A single dirichlet observation.

        Returns:
            Log-density evaluated at x.

        """
        xx = np.asarray(x, dtype=float)
        if xx.shape != self.alpha.shape or not np.all(np.isfinite(xx)) or np.any(xx < 0.0):
            return -np.inf
        if not np.isclose(
            float(xx.sum()),
            1.0,
            rtol=_SIMPLEX_SUM_RTOL,
            atol=_SIMPLEX_SUM_ATOL,
        ):
            return -np.inf

        pos = xx > 0.0
        if not np.all(pos):
            zero_alpha = self.alpha[~pos]
            if np.any(zero_alpha < 1.0):
                return np.inf
            if np.any(zero_alpha > 1.0):
                return -np.inf
            # alpha == 1 contributes zero at the boundary.

        rv = np.dot(np.log(xx[pos]), self.alpha[pos] - 1.0)
        rv -= self.log_const
        return rv

    def seq_log_density(self, x: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Vectorized evaluation of the log-density at a sequence-encoded input x.

        Args:
            x (Tuple[np.ndarray, np.ndarray, np.ndarray]): Encoded data from
                DirichletDataEncoder.seq_encode(), a tuple of (log of observations, observations,
                squared observations).

        Returns:
            Numpy array containing the log-density of each encoded observation. Rows that are not on
            the simplex (a negative or non-finite entry, or a row that doesn't sum to one) score
            ``-inf``, mirroring ``log_density`` -- the encoder's clipped log representation alone
            can't distinguish a genuine near-zero coordinate from a negative one, so validation reads
            the accompanying raw observations instead.

        """
        raw_x = _encoded_raw_matrix(x, self.dim)
        good = _simplex_row_mask(raw_x)
        positive = raw_x > 0.0
        safe_logs = np.zeros_like(raw_x)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log(raw_x, out=safe_logs, where=positive)
        rv = np.dot(safe_logs, self.alpha - 1.0) - self.log_const
        zero = ~positive
        saw_pos_inf = np.any(zero & (self.alpha[None, :] < 1.0), axis=1)
        saw_neg_inf = np.any(zero & (self.alpha[None, :] > 1.0), axis=1)
        rv = np.where(saw_pos_inf, np.inf, np.where(saw_neg_inf, -np.inf, rv))
        return np.where(good, rv, -np.inf)

    @staticmethod
    def backend_log_density_from_params(log_x: Any, alpha: Any, log_const: Any, engine: Any) -> Any:
        """Engine-neutral Dirichlet log-density from encoded log observations."""
        return engine.sum(log_x * (alpha - engine.asarray(1.0)), axis=-1) - log_const

    @staticmethod
    def _backend_validated_log_density(
        raw_x: Any,
        alpha: Any,
        log_const: Any,
        engine: Any,
    ) -> Any:
        one = engine.asarray(1.0)
        zero_value = engine.asarray(0.0)
        raw = engine.asarray(raw_x)
        concentration = engine.asarray(alpha)
        positive = raw > zero_value
        safe_logs = engine.where(positive, engine.log(engine.where(positive, raw, one)), zero_value)
        result = engine.sum(safe_logs * (concentration - one), axis=-1) - log_const
        zero = raw == zero_value
        saw_pos_inf = engine.sum(zero & (concentration < one), axis=-1) > 0
        saw_neg_inf = engine.sum(zero & (concentration > one), axis=-1) > 0
        result = engine.where(
            saw_pos_inf,
            engine.asarray(np.inf),
            engine.where(saw_neg_inf, engine.asarray(-np.inf), result),
        )
        finite = ~(engine.isnan(raw) | engine.isinf(raw))
        valid = (
            (engine.sum(~finite, axis=-1) == 0)
            & (engine.sum(raw < zero_value, axis=-1) == 0)
            & (engine.abs(engine.sum(raw, axis=-1) - one) <= engine.asarray(_SIMPLEX_SUM_ATOL + _SIMPLEX_SUM_RTOL))
        )
        return engine.where(valid, result, engine.asarray(-np.inf))

    def backend_seq_log_density(self, x: tuple[Any, Any, Any], engine: Any) -> Any:
        """Engine-neutral vectorized log-density for encoded Dirichlet observations."""
        raw_x = _encoded_raw_matrix(x, self.dim)
        return self._backend_validated_log_density(
            raw_x,
            engine.asarray(self.alpha),
            engine.asarray(self.log_const),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["DirichletDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked parameters for equal-dimensional Dirichlet mixtures."""
        dim = int(dists[0].dim)
        if any(int(dist.dim) != dim for dist in dists):
            raise ValueError("Stacked DirichletDistribution components require equal dimension.")
        return {
            "alpha": engine.asarray([dist.alpha for dist in dists]),
            "log_const": engine.asarray([dist.log_const for dist in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, Any, Any], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of Dirichlet component log densities."""
        raw_x = _encoded_raw_matrix(x)
        return cls._backend_validated_log_density(
            engine.asarray(raw_x)[:, None, :],
            params["alpha"][None, :, :],
            params["log_const"][None, :],
            engine,
        )

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: tuple[Any, Any, Any], weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any, Any]:
        """Return component-stacked legacy Dirichlet sufficient statistics."""
        raw = _interior_simplex_matrix(x)
        weights_np = np.asarray(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            dtype=np.float64,
        )
        component_count = int(np.asarray(engine.to_numpy(params["alpha"])).shape[0])
        if weights_np.shape != (raw.shape[0], component_count):
            raise ValueError(
                "Stacked Dirichlet weights must have exact shape (%d, %d)." % (raw.shape[0], component_count)
            )
        if np.any(~np.isfinite(weights_np)) or np.any(weights_np < 0.0):
            raise ValueError("Stacked Dirichlet weights must be finite and non-negative.")
        ww = engine.asarray(weights_np)
        extra = (slice(None), slice(None), None)
        log_raw = engine.asarray(np.log(raw))
        raw_engine = engine.asarray(raw)
        squared = engine.asarray(raw * raw)
        return (
            engine.sum(ww, axis=0),
            engine.sum(ww[extra] * log_raw[:, None, :], axis=0),
            engine.sum(ww[extra] * raw_engine[:, None, :], axis=0),
            engine.sum(ww[extra] * squared[:, None, :], axis=0),
        )

    def to_fisher(self, **kwargs):
        """Return the Dirichlet's log-coordinate Fisher view (generic fallback for degenerate alpha)."""
        alpha = np.asarray(self.alpha, dtype=np.float64)
        if alpha.ndim > 0 and np.all(np.isfinite(alpha)) and np.all(alpha > 0.0):
            return DirichletFisherView(self)
        return super().to_fisher(**kwargs)

    def sampler(self, seed: int | None = None) -> "DirichletSampler":
        """Return a sampler for iid draws from this distribution.

        Args:
            seed: Optional random seed.
        """
        return DirichletSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "DirichletEstimator":
        """Return an estimator initialized from this distribution.

        When ``pseudo_count`` is provided, the normalized concentration vector
        regularizes the estimate.

        Args:
            pseudo_count: Optional weight for the current normalized
                concentration vector.
        """
        if pseudo_count is None:
            return DirichletEstimator(dim=self.dim, name=self.name)
        else:
            return DirichletEstimator(
                dim=self.dim,
                pseudo_count=pseudo_count,
                suff_stat=digamma(self.alpha) - digamma(sum(self.alpha)),
                name=self.name,
            )

    def dist_to_encoder(self) -> "DirichletDataEncoder":
        """Create the encoder for iid Dirichlet observations."""
        return DirichletDataEncoder(self.dim)


class DirichletSampler(DistributionSampler):
    """Sampler for iid draws from a Dirichlet distribution."""

    def __init__(self, dist: DirichletDistribution, seed: int | None = None) -> None:
        """Create a sampler for a Dirichlet distribution.

        Args:
            dist: Distribution to sample from.
            seed: Optional random seed.

        Attributes:
            dist: Distribution sampled by this object.
            rng: Random state used for reproducible draws.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray:
        """Draw iid samples from the Dirichlet distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw. If None, a single sample is drawn.

        Returns:
            Numpy array with shape (dim,) if size is None, else with shape (size, dim).

        """
        alpha = np.asarray(self.dist.alpha, dtype=np.float64)
        if not _valid_alpha(alpha) or alpha.ndim != 1:
            raise ValueError("DirichletSampler requires a finite positive concentration vector.")
        if size is not None:
            if isinstance(size, (bool, np.bool_)):
                raise TypeError("DirichletSampler size must be a non-negative integer.")
            try:
                size = operator.index(size)
            except TypeError as exc:
                raise TypeError("DirichletSampler size must be a non-negative integer.") from exc
            if size < 0:
                raise ValueError("DirichletSampler size must be non-negative.")
        return self.rng.dirichlet(alpha=alpha, size=size)


class DirichletAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for weighted Dirichlet sufficient statistics."""

    def __init__(self, dim: int | None = None, keys: str | None = None) -> None:
        """Create an accumulator for simplex observations.

        Args:
            dim: Dimension of the Dirichlet distribution. ``None`` defers allocation until
                the first observation reveals the dimension (lets ``DirichletEstimator()`` infer ``K`` from
                data).
            keys: Optional key for merging sufficient statistics.

        Attributes:
            dim: Dimension of the Dirichlet distribution, or ``None`` until
                the first observation sizes the accumulator.
            sum_of_logs (np.ndarray): Weighted sum of the log of observation vectors.
            sum (np.ndarray): Weighted sum of observation vectors.
            sum2 (np.ndarray): Weighted sum of squared observation vectors.
            counts (float): Sum of observation weights.
            key (Optional[str]): Key for merging sufficient statistics.

        """
        self.dim = _validated_dimension(dim, optional=True)
        self.sum_of_logs = None if self.dim is None else np.zeros(self.dim)
        self.sum = None if self.dim is None else np.zeros(self.dim)
        self.sum2 = None if self.dim is None else np.zeros(self.dim)
        self.counts = 0.0
        self.keys = keys

    def _ensure_dim(self, dim: int) -> None:
        """Allocate the moment accumulators once the data reveals the dimension."""
        dimension = _validated_dimension(dim, optional=False)
        if self.dim is None:
            self.dim = dimension
            self.sum_of_logs = np.zeros(self.dim)
            self.sum = np.zeros(self.dim)
            self.sum2 = np.zeros(self.dim)
        elif self.dim != dimension:
            raise ValueError("Dirichlet accumulator dimension changed from %d to %d." % (self.dim, dimension))

    def update(self, x: np.ndarray | list[float], weight: float, estimate: Optional["DirichletDistribution"]) -> None:
        """Update sufficient statistics with a single weighted observation.

        Boundary observations are rejected because an exact zero has an infinite
        log sufficient statistic and cannot support a finite concentration fit.

        Args:
            x (Union[np.ndarray, List[float]]): Length-dim probability vector observation.
            weight (float): Weight for the observation.
            estimate (Optional[DirichletDistribution]): Kept for consistency with
                SequenceEncodableStatisticAccumulator (not used).

        Returns:
            None.

        """
        xx = _interior_simplex_vector(x, self.dim)
        checked_weight = _validated_weight(weight)
        self._ensure_dim(xx.size)
        self.sum_of_logs += log(xx) * checked_weight
        self.sum += checked_weight * xx
        self.sum2 += checked_weight * xx * xx
        self.counts += checked_weight

    def initialize(self, x: np.ndarray | list[float], weight: float, estimate: RandomState | None) -> None:
        """Initialize the accumulator with a weighted observation. Calls update().

        Args:
            x (Union[np.ndarray, List[float]]): Length-dim probability vector observation.
            weight (float): Weight for the observation.
            estimate (Optional[RandomState]): Kept for consistency with
                SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def get_seq_lambda(self):
        """Returns a list containing the seq_update member function."""
        return [self.seq_update]

    def seq_update(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
        estimate: DirichletDistribution | None,
    ) -> None:
        """Vectorized update of sufficient statistics with an encoded sequence of observations.

        Args:
            x (Tuple[np.ndarray, np.ndarray, np.ndarray]): Encoded data from
                DirichletDataEncoder.seq_encode().
            weights (np.ndarray): Numpy array of observation weights.
            estimate (Optional[DirichletDistribution]): Kept for consistency (not used).

        Returns:
            None.

        """
        raw = _interior_simplex_matrix(x, self.dim)
        checked_weights = _validated_weight_vector(weights, raw.shape[0])
        self._ensure_dim(raw.shape[1])
        self.sum_of_logs += np.dot(checked_weights, np.log(raw))
        self.counts += float(checked_weights.sum())
        self.sum += np.dot(checked_weights, raw)
        self.sum2 += np.dot(checked_weights, raw * raw)

    def seq_update_engine(
        self,
        x: tuple[np.ndarray, np.ndarray, np.ndarray],
        weights: Any,
        estimate: DirichletDistribution | None,
        engine: Any,
    ) -> None:
        """Engine-resident accumulation of Dirichlet moment statistics (numpy or torch).

        The weighted vector moments (sum of logs, sum, sum of squares) are reduced via a
        weight-vector / observation-matrix product on the active engine. Matches seq_update.
        """
        raw = _interior_simplex_matrix(x, self.dim)
        weights_np = _validated_weight_vector(
            engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights,
            raw.shape[0],
        )
        self._ensure_dim(raw.shape[1])
        w = engine.asarray(weights_np)
        log_x = engine.asarray(np.log(raw))
        xv = engine.asarray(raw)
        xv2 = engine.asarray(raw * raw)

        self.sum_of_logs += np.asarray(engine.to_numpy(engine.matmul(w, log_x)))
        self.counts += float(engine.to_numpy(engine.sum(w)))
        self.sum += np.asarray(engine.to_numpy(engine.matmul(w, xv)))
        self.sum2 += np.asarray(engine.to_numpy(engine.matmul(w, xv2)))

    def seq_initialize(
        self, x: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState | None
    ) -> None:
        """Vectorized initialization of the accumulator. Calls seq_update().

        Args:
            x (Tuple[np.ndarray, np.ndarray, np.ndarray]): Encoded data from
                DirichletDataEncoder.seq_encode().
            weights (np.ndarray): Numpy array of observation weights.
            rng (Optional[RandomState]): Kept for consistency with SequenceEncodableStatisticAccumulator.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> "DirichletAccumulator":
        """Merge the sufficient statistics of suff_stat into this accumulator.

        Args:
            suff_stat (Tuple[int, np.ndarray, np.ndarray, np.ndarray]): Tuple of (counts, sum of logs,
                sum of observations, sum of squared observations).

        Returns:
            DirichletAccumulator object.

        """
        counts, sum_of_logs, sum_v, sum_v2, dimension = _validated_dirichlet_statistics(
            suff_stat,
            configured_dim=self.dim,
            allow_unsized=True,
        )
        if dimension is None:
            return self
        self._ensure_dim(dimension)
        self.sum_of_logs += sum_of_logs
        self.sum += sum_v
        self.sum2 += sum_v2
        self.counts += counts
        return self

    def value(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the sufficient statistics (counts, sum of logs, sum, sum of squares) of the accumulator."""
        return (
            self.counts,
            None if self.sum_of_logs is None else self.sum_of_logs.copy(),
            None if self.sum is None else self.sum.copy(),
            None if self.sum2 is None else self.sum2.copy(),
        )

    def from_value(self, x: tuple[int, np.ndarray, np.ndarray, np.ndarray]):
        """Set the sufficient statistics of the accumulator to x.

        Args:
            x (Tuple[int, np.ndarray, np.ndarray, np.ndarray]): Tuple of (counts, sum of logs,
                sum of observations, sum of squared observations).

        Returns:
            None.

        """
        counts, sum_of_logs, sum_v, sum_v2, dimension = _validated_dirichlet_statistics(
            x,
            configured_dim=self.dim,
            allow_unsized=True,
        )
        self.counts = counts
        self.sum_of_logs = sum_of_logs
        self.sum = sum_v
        self.sum2 = sum_v2
        if dimension is not None:
            self.dim = dimension
        return self

    def acc_to_encoder(self) -> "DirichletDataEncoder":
        """Create the encoder associated with this accumulator."""
        return DirichletDataEncoder(self.dim)


class DirichletAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for Dirichlet sufficient-statistic accumulators."""

    def __init__(self, dim: int | None = None, keys: str | None = None) -> None:
        """Create an accumulator factory with fixed or data-inferred dimension.

        Args:
            dim: Dimension of the Dirichlet distribution, or ``None`` to infer
                the dimension from data.
            keys: Optional key for merging sufficient statistics.

        Attributes:
            dim: Configured dimension.
            keys: Optional merge key.
        """
        self.dim = _validated_dimension(dim, optional=True)
        self.keys = keys

    def make(self) -> "DirichletAccumulator":
        """Return a new accumulator with the factory's dimension and keys."""
        return DirichletAccumulator(dim=self.dim, keys=self.keys)


class DirichletEstimator(ParameterEstimator):
    """Estimator for Dirichlet concentration parameters."""

    def __init__(
        self,
        dim: int | None = None,
        pseudo_count: float | None = None,
        suff_stat: np.ndarray | None = None,
        delta: float | None = 1.0e-8,
        keys: str | None = None,
        use_mpe: bool = False,
        name: str | None = None,
    ) -> None:
        """Create a Dirichlet estimator.

        Args:
            dim: Dimension of the Dirichlet distribution. ``None`` (the default) lets the
                accumulator discover the dimension from the first observation -- so ``DirichletEstimator()``
                fits any simplex data without being told ``K`` up front (the estimate's ``K`` is read off
                the data, exactly like the per-coordinate sufficient statistics). Pass ``dim`` only when
                using ``pseudo_count``/``suff_stat`` regularization, which needs the dimension to size the
                prior.
            pseudo_count: Weight assigned to the prior sufficient statistic.
            suff_stat: Mean-log-probability sufficient statistic used with
                pseudo_count to regularize the estimate.
            delta: Convergence tolerance for the concentration solver.
            keys: Optional merge key for sufficient statistics.
            use_mpe: If true, use MPE-accelerated fixed-point iteration.
            name: Optional name assigned to the estimated distribution.

        Attributes:
            dim: Configured or inferred dimension.
            pseudo_count: Weight assigned to prior sufficient statistics.
            delta: Solver convergence tolerance.
            suff_stat: Optional prior mean-log-probability statistic.
            keys: Optional merge key.
            use_mpe: Whether to use the MPE-accelerated solver.
            name: Optional fitted-distribution name.
        """
        configured_dim = _validated_dimension(dim, optional=True)
        if pseudo_count is None:
            checked_pseudo_count = None
        else:
            checked_pseudo_count = _validated_weight(
                pseudo_count,
                "pseudo_count",
            )
        if suff_stat is not None:
            if checked_pseudo_count is None:
                raise ValueError("DirichletEstimator suff_stat requires pseudo_count.")
            try:
                checked_suff_stat = np.asarray(suff_stat, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("DirichletEstimator suff_stat must be a numeric vector.") from exc
            if (
                checked_suff_stat.ndim != 1
                or checked_suff_stat.size == 0
                or np.any(~np.isfinite(checked_suff_stat))
                or np.any(checked_suff_stat > 0.0)
            ):
                raise ValueError("DirichletEstimator suff_stat must be a non-empty finite non-positive vector.")
            if configured_dim is None:
                configured_dim = len(checked_suff_stat)
            elif checked_suff_stat.shape != (configured_dim,):
                raise ValueError("DirichletEstimator suff_stat width must match dim.")
            checked_suff_stat = checked_suff_stat.copy()
        else:
            checked_suff_stat = None
        if delta is None or np.ndim(delta) != 0:
            raise ValueError("DirichletEstimator delta must be a finite positive scalar.")
        try:
            checked_delta = float(delta)
        except (TypeError, ValueError) as exc:
            raise ValueError("DirichletEstimator delta must be a finite positive scalar.") from exc
        if not np.isfinite(checked_delta) or checked_delta <= 0.0:
            raise ValueError("DirichletEstimator delta must be a finite positive scalar.")
        if not isinstance(use_mpe, (bool, np.bool_)):
            raise TypeError("DirichletEstimator use_mpe must be bool.")

        self.dim = configured_dim
        self.pseudo_count = checked_pseudo_count
        self.delta = checked_delta
        self.suff_stat = checked_suff_stat
        self.keys = keys
        self.use_mpe = bool(use_mpe)
        self.name = name

    def accumulator_factory(self) -> "DirichletAccumulatorFactory":
        """Create a Dirichlet accumulator factory from this estimator's settings."""
        return DirichletAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[int, np.ndarray, np.ndarray, np.ndarray]
    ) -> DirichletDistribution:
        """Estimate a Dirichlet distribution from aggregated sufficient statistics.

        ``suff_stat`` is ``(count, sum_log_x, sum_x, sum_x2)``. The
        concentration parameters are solved from the mean-log-probability
        statistic with a safeguarded Newton solver, or ``find_alpha`` when
        ``use_mpe`` is set.

        ``nobs`` is accepted for estimator API consistency but counts are taken
        from ``suff_stat[0]``.
        """
        nobs, sum_of_logs, sum_v, sum_v2, statistic_dim = _validated_dirichlet_statistics(
            suff_stat,
            configured_dim=self.dim,
            allow_unsized=True,
        )
        if statistic_dim is None:
            if self.dim is None:
                raise ValueError("DirichletEstimator cannot infer a dimension from an unsized empty accumulator.")
            dim = self.dim
            sum_of_logs = np.zeros(dim, dtype=np.float64)
            sum_v = np.zeros(dim, dtype=np.float64)
            sum_v2 = np.zeros(dim, dtype=np.float64)
        else:
            dim = statistic_dim

        if self.pseudo_count is not None and self.suff_stat is None:
            pc = self.pseudo_count
            c1 = digamma(one) - digamma(dim)
            c2 = sum_of_logs + c1 * pc
            total = nobs + pc
            if total == 0.0:
                mean_log_p = np.full(dim, c1, dtype=float)
            else:
                mean_log_p = c2 / total
            prior_mean = np.ones(dim, dtype=float) / float(dim)
            if nobs > 0.0:
                mean_v = _safe_simplex_mean((sum_v + pc * prior_mean) / total, dim)
                mean_v2 = (sum_v2 + pc * prior_mean * prior_mean) / total
            else:
                mean_v = prior_mean
                mean_v2 = None
            initial_estimate = _initial_dirichlet_alpha(mean_v, mean_v2, mean_log_p)

        elif self.pseudo_count is not None and self.suff_stat is not None:
            pc = self.pseudo_count
            prior_mlp = self.suff_stat
            if prior_mlp.shape != (dim,):
                raise ValueError("DirichletEstimator prior sufficient-statistic width changed.")
            c2 = sum_of_logs + prior_mlp * pc
            total = nobs + pc
            mean_log_p = prior_mlp if total == 0.0 else c2 / total
            prior_mean = _mean_from_mean_log(prior_mlp, dim)
            if nobs > 0.0:
                mean_v = _safe_simplex_mean((sum_v + pc * prior_mean) / total, dim)
                mean_v2 = (sum_v2 + pc * prior_mean * prior_mean) / total
            else:
                mean_v = prior_mean
                mean_v2 = None
            initial_estimate = _initial_dirichlet_alpha(mean_v, mean_v2, mean_log_p)

        else:
            if nobs <= 0.0:
                result = DirichletDistribution(np.ones(dim, dtype=float), name=self.name)
                result.fit_metadata = {
                    "solver": "none",
                    "converged": True,
                    "iterations": 0,
                    "repairs": (),
                    "reason": "no observations",
                }
                return result

            sum_v = sum_v / nobs
            sum_v2 = sum_v2 / nobs
            sum_v = _safe_simplex_mean(sum_v, dim)

            initial_estimate = _initial_dirichlet_alpha(sum_v, sum_v2)

            mean_log_p = sum_of_logs / nobs

        if mean_log_p.shape != (dim,) or np.any(~np.isfinite(mean_log_p)) or np.any(mean_log_p > 0.0):
            raise ValueError("DirichletEstimator produced invalid mean-log sufficient statistics.")

        if nobs <= 1.0 and self.pseudo_count is None:
            result = DirichletDistribution(initial_estimate, name=self.name)
            result.fit_metadata = {
                "solver": "moment_initialization",
                "converged": True,
                "iterations": 0,
                "repairs": (),
                "reason": "at most one weighted observation",
            }
            return result

        else:
            if self.use_mpe:
                alpha, its_cnt = find_alpha(np.asarray(initial_estimate), mean_log_p, self.delta)
                solver = "mpe"
            else:
                alpha, its_cnt = dirichlet_param_solve(np.asarray(initial_estimate), mean_log_p, self.delta)
                solver = "newton"

            result = DirichletDistribution(alpha, name=self.name)
            result.fit_metadata = {
                "solver": solver,
                "converged": True,
                "iterations": int(its_cnt),
                "repairs": (),
            }
            return result


class DirichletDataEncoder(DataSequenceEncoder):
    """Data encoder for iid Dirichlet simplex observations."""

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = _validated_dimension(dimension, optional=True)

    def __str__(self) -> str:
        """Return the Dirichlet encoder's display name."""
        return "DirichletDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is a Dirichlet data encoder."""
        return isinstance(other, DirichletDataEncoder) and self.dimension == other.dimension

    def seq_encode(self, x: Sequence[Sequence[float]]):
        """Encode a sequence of iid probability-vector observations for vectorized 'seq_' calls.

        Args:
            x (Sequence[Sequence[float]]): Sequence of length-dim probability vectors.

        Returns:
            Tuple of (exact logs, observations, squared observations). Zero
            coordinates remain ``-inf`` in the log payload so density and fitting
            paths can preserve their mathematical boundary semantics.

        """
        try:
            rv = np.asarray(x, dtype=np.float64).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError("DirichletDataEncoder observations must be numeric.") from exc
        if rv.ndim != 2:
            raise ValueError("DirichletDataEncoder observations must have exact shape (n, d).")
        if rv.shape[1] == 0:
            raise ValueError("DirichletDataEncoder observations cannot have zero dimension.")
        if self.dimension is not None and rv.shape[1] != self.dimension:
            raise ValueError("DirichletDataEncoder observations must have %d columns." % self.dimension)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rv = np.log(rv)
        return log_rv, rv, rv * rv
