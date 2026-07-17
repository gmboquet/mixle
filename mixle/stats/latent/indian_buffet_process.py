"""Indian buffet process finite truncation with variational Bayes estimation.

This module implements the finite Beta-Bernoulli truncation of the Indian buffet
process (IBP).  With K features and concentration alpha,

    pi_k ~ Beta(alpha / K, 1)
    z_nk | pi_k ~ Bernoulli(pi_k)

Rows z_n may be supplied either as dense binary vectors of length K or as sparse
lists/sets of active feature indices.  Estimation is variational Bayes with a
factorized posterior q(pi_k) = Beta(a_k, b_k).  The local variational factor for
an observed feature row is deterministic, so the accumulator gathers weighted
feature-use counts and the estimator performs the conjugate Beta update.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import betaln, digamma

from mixle.enumeration.algorithms import BufferedStream, ProductEnumerator
from mixle.inference.fisher import FixedFisherView
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

SS = tuple[np.ndarray, float, float | None]


def _check_data_format(data_format: str) -> str:
    if data_format not in ("auto", "dense", "sparse"):
        raise ValueError("data_format must be one of 'auto', 'dense', or 'sparse'")
    return data_format


def _validate_num_features(num_features: int) -> int:
    num_features = int(num_features)
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    return num_features


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be a finite positive value")
    return alpha


def _validate_probability_vector(
    p: Sequence[float] | np.ndarray, num_features: int, min_prob: float = 0.0
) -> np.ndarray:
    rv = np.asarray(p, dtype=np.float64)
    if rv.shape != (num_features,):
        raise ValueError("feature probability vector must have length num_features")
    if np.any(np.isnan(rv)) or np.any(rv < 0.0) or np.any(rv > 1.0):
        raise ValueError("feature probabilities must lie in [0, 1]")
    if min_prob > 0.0:
        rv = np.clip(rv, min_prob, 1.0 - min_prob)
    return rv


def _is_binary_vector(x: np.ndarray, num_features: int) -> bool:
    if x.ndim != 1 or x.size != num_features:
        return False
    return bool(np.all(np.logical_or(x == 0, x == 1)))


def _as_1d_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (set, frozenset)):
        return np.asarray(list(x))
    if isinstance(x, (str, bytes)):
        raise TypeError("IBP observations must be binary vectors or active-feature indices, not strings")
    return np.asarray(list(x))


def _to_binary_vector(x: Any, num_features: int, data_format: str) -> np.ndarray:
    data_format = _check_data_format(data_format)
    xx = _as_1d_array(x)

    if xx.ndim != 1:
        raise ValueError("IBP observations must be one-dimensional")

    if data_format == "dense" or (data_format == "auto" and _is_binary_vector(xx, num_features)):
        if xx.size != num_features:
            raise ValueError("dense IBP observations must have length num_features")
        if not _is_binary_vector(xx, num_features):
            raise ValueError("dense IBP observations must contain only 0/1 values")
        return xx.astype(bool, copy=False)

    idx = np.asarray(xx, dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= num_features):
        raise ValueError("sparse IBP feature index out of range")
    rv = np.zeros(num_features, dtype=bool)
    if idx.size:
        rv[np.unique(idx)] = True
    return rv


class IndianBuffetProcessFisherView(FixedFisherView):
    """Fixed Fisher view whose statistics are the binary feature indicators."""

    def __init__(self, dist: Any) -> None:
        self.num_features = int(dist.num_features)
        super().__init__(dist, [("feature", str(i)) for i in range(self.num_features)])

    def _statistics_from_data(self, data: Sequence[Any], estimate: Any | None = None) -> np.ndarray:
        enc = self.dist.dist_to_encoder().seq_encode(list(data))
        return self._statistics_from_encoded(enc, estimate=estimate)

    def _statistics_from_encoded(self, enc_data: Any, estimate: Any | None = None) -> np.ndarray:
        return np.asarray(enc_data, dtype=np.float64).reshape((-1, self.num_features))

    def _model_mean(self) -> np.ndarray:
        return np.asarray(self.dist.feature_probs, dtype=np.float64).copy()

    def _model_fisher(self) -> np.ndarray:
        p = np.asarray(self.dist.feature_probs, dtype=np.float64)
        return np.diag(p * (1.0 - p))


class IndianBuffetProcessDistribution(SequenceEncodableProbabilityDistribution):
    """Finite-truncated Indian buffet process over binary feature rows.

    Args:
        num_features: Truncation level K.
        alpha: IBP concentration parameter.
        beta_params: Optional (K, 2) variational Beta parameters for q(pi_k).
            If omitted, the prior Beta(alpha / K, 1) is used.
        feature_probs: Optional plug-in feature probabilities.  When supplied
            without beta_params, a lightweight Beta posterior with matching mean
            is created for expected-log-density calls.
        min_prob: Minimum plug-in probability used when feature_probs are given.
        name: Optional distribution name.
        keys: Optional key for tying sufficient statistics.
        data_format: 'dense', 'sparse', or 'auto' input interpretation.
    """

    @classmethod
    def compute_capabilities(cls):
        """Declare generic table-kernel capabilities for finite IBP rows."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_table")

    @classmethod
    def compute_declaration(cls):
        """Return the structured declaration for finite IBP parameters and statistics."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="indian_buffet_process",
            distribution_type=cls,
            parameters=(
                ParameterSpec("num_features", constraint="integer", differentiable=False),
                ParameterSpec("alpha", constraint="positive"),
                ParameterSpec("beta_params", constraint="positive_matrix"),
            ),
            statistics=(
                StatisticSpec("feature_counts", kind="count_vector"),
                StatisticSpec("total_count"),
                StatisticSpec("alpha", kind="metadata", additive=False, scales=False),
            ),
            support="binary_feature_matrix",
            differentiable=False,
        )

    def __init__(
        self,
        num_features: int,
        alpha: float = 1.0,
        beta_params: Sequence[Sequence[float]] | np.ndarray | None = None,
        feature_probs: Sequence[float] | np.ndarray | None = None,
        min_prob: float = 1.0e-128,
        name: str | None = None,
        keys: str | None = None,
        data_format: str = "auto",
    ) -> None:
        self.num_features = _validate_num_features(num_features)
        self.alpha = _validate_alpha(alpha)
        self.name = name
        self.keys = keys
        self.min_prob = float(min_prob)
        self.data_format = _check_data_format(data_format)

        if beta_params is not None:
            bp = np.asarray(beta_params, dtype=np.float64)
            if bp.shape != (self.num_features, 2):
                raise ValueError("beta_params must have shape (num_features, 2)")
            if np.any(bp <= 0.0) or np.any(~np.isfinite(bp)):
                raise ValueError("beta_params must be finite positive values")
            self.beta_params = bp.copy()
        elif feature_probs is not None:
            p = _validate_probability_vector(feature_probs, self.num_features, self.min_prob)
            self.beta_params = np.column_stack((p, 1.0 - p))
        else:
            self.beta_params = np.column_stack(
                (
                    np.full(self.num_features, self.alpha / float(self.num_features), dtype=np.float64),
                    np.ones(self.num_features, dtype=np.float64),
                )
            )

        beta_sum = self.beta_params.sum(axis=1)
        self.feature_probs = self.beta_params[:, 0] / beta_sum
        self.log_pvec = np.log(self.feature_probs)
        self.log_nvec = np.log1p(-self.feature_probs)
        self.log_dvec = self.log_pvec - self.log_nvec
        self.log_nsum = float(self.log_nvec.sum())

        self.expected_log_pvec = digamma(self.beta_params[:, 0]) - digamma(beta_sum)
        self.expected_log_nvec = digamma(self.beta_params[:, 1]) - digamma(beta_sum)
        self.expected_log_dvec = self.expected_log_pvec - self.expected_log_nvec
        self.expected_log_nsum = float(self.expected_log_nvec.sum())

    def __str__(self) -> str:
        return (
            "IndianBuffetProcessDistribution(%s, alpha=%s, beta_params=%s, min_prob=%s, "
            "name=%s, keys=%s, data_format=%s)"
        ) % (
            repr(self.num_features),
            repr(self.alpha),
            repr(self.beta_params.tolist()),
            repr(self.min_prob),
            repr(self.name),
            repr(self.keys),
            repr(self.data_format),
        )

    def density(self, x: Any) -> float:
        """Return the probability density or mass at a single observation."""
        return float(np.exp(self.log_density(x)))

    def density_semantics(self):
        """Return that plug-in densities use posterior mean feature probabilities."""
        from mixle.stats.compute.pdist import DensitySemantics

        return DensitySemantics.ESTIMATE  # plug-in using E_q[pi_k], not the exact marginal

    def log_density(self, x: Any) -> float:
        """Plug-in log-density of one feature row using E_q[pi_k]."""
        xx = _to_binary_vector(x, self.num_features, self.data_format)
        return float(self.log_nsum + np.dot(xx, self.log_dvec))

    def expected_log_density(self, x: Any) -> float:
        """VB expected log-density E_q[log p(z | pi)] for one observed row."""
        xx = _to_binary_vector(x, self.num_features, self.data_format)
        return float(self.expected_log_nsum + np.dot(xx, self.expected_log_dvec))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        xx = np.asarray(x, dtype=np.float64)
        return self.log_nsum + np.dot(xx, self.log_dvec)

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized plug-in log-density for encoded feature rows."""
        log_dvec = engine.asarray(self.log_dvec)
        xx = engine.asarray(x, dtype=getattr(log_dvec, "dtype", None))
        return float(self.log_nsum) + engine.matmul(xx, log_dvec)

    @classmethod
    def backend_stacked_params(cls, dists: Sequence["IndianBuffetProcessDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked finite-IBP parameters for equal-feature mixtures."""
        num_features = int(dists[0].num_features)
        if any(int(dist.num_features) != num_features for dist in dists):
            raise ValueError("Stacked IndianBuffetProcessDistribution components require equal feature dimension.")
        return {
            "__pysp_component_axis__": {"log_dvec": 1, "log_nsum": 0, "alpha": 0},
            "log_dvec": engine.asarray(np.stack([dist.log_dvec for dist in dists], axis=1)),
            "log_nsum": engine.asarray([dist.log_nsum for dist in dists]),
            "alpha": engine.asarray([dist.alpha for dist in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of finite-IBP component log densities."""
        xx = engine.asarray(x, dtype=getattr(params["log_dvec"], "dtype", None))
        return engine.matmul(xx, params["log_dvec"]) + params["log_nsum"][None, :]

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any, Any]:
        """Return component-stacked legacy ``(feature_counts, total_count, alpha)`` statistics."""
        ww = engine.asarray(weights)
        xx = engine.asarray(x, dtype=getattr(ww, "dtype", None))
        return engine.matmul(ww.T, xx), engine.sum(ww, axis=0), params["alpha"]

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized expected log-density values for encoded observations."""
        xx = np.asarray(x, dtype=np.float64)
        return self.expected_log_nsum + np.dot(xx, self.expected_log_dvec)

    def seq_local_elbo(self, x: np.ndarray) -> np.ndarray:
        """Per-row VB contribution; rows are observed, so there is no local entropy."""
        return self.seq_expected_log_density(x)

    def enumerator(self) -> DistributionEnumerator:
        """Enumerate feature rows in descending probability order.

        The plug-in row density factorizes over features -- ``log p(z) = sum_k [z_k log pi_k +
        (1-z_k) log(1-pi_k)]`` with ``pi_k = E_q[feature_probs]`` -- so the truncated IBP row is a
        product of independent Bernoulli features and enumerates by best-first over the per-feature
        supports (the same structure as the Erdos-Renyi graph). Rows are emitted in the configured
        ``data_format`` (a dense 0/1 list, or a sorted list of active feature indices when sparse),
        each carrying its exact ``log_density``.
        """
        return IndianBuffetProcessEnumerator(self)

    def to_fisher(self, **kwargs):
        """Return this distribution's own Fisher view."""
        return IndianBuffetProcessFisherView(self)

    def sampler(self, seed: int | None = None) -> "IndianBuffetProcessSampler":
        """Return a sampler for drawing observations from this distribution."""
        return IndianBuffetProcessSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IndianBuffetProcessEstimator":
        """Return an estimator for fitting this distribution from data."""
        suff_stat = self.feature_probs.copy() if pseudo_count is not None else None
        return IndianBuffetProcessEstimator(
            self.num_features,
            alpha=self.alpha,
            pseudo_count=pseudo_count,
            suff_stat=suff_stat,
            min_prob=self.min_prob,
            name=self.name,
            keys=self.keys,
            data_format=self.data_format,
        )

    def dist_to_encoder(self) -> "IndianBuffetProcessDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return IndianBuffetProcessDataEncoder(self.num_features, self.data_format)


class IndianBuffetProcessEnumerator(DistributionEnumerator):
    """Enumerate finite IBP feature rows in descending plug-in probability order."""

    def __init__(self, dist: IndianBuffetProcessDistribution) -> None:
        """Best-first enumeration of IBP feature rows over independent Bernoulli features.

        Args:
            dist (IndianBuffetProcessDistribution): Distribution whose rows are enumerated.
        """
        super().__init__(dist)
        sparse = dist.data_format == "sparse"
        streams = []
        for k in range(dist.num_features):
            opts = []
            if dist.log_pvec[k] > -np.inf:
                opts.append((1, float(dist.log_pvec[k])))
            if dist.log_nvec[k] > -np.inf:
                opts.append((0, float(dist.log_nvec[k])))
            opts.sort(key=lambda u: -u[1])
            streams.append(BufferedStream(iter(opts)))

        def combine(values: tuple[int, ...]) -> list[int]:
            if sparse:
                return [k for k, v in enumerate(values) if v]
            return [int(v) for v in values]

        self._product = ProductEnumerator(streams, combine=combine)

    def __next__(self) -> tuple[list[int], float]:
        return next(self._product)


class IndianBuffetProcessSampler(DistributionSampler):
    """Sampler for finite-truncated IBP feature rows."""

    def __init__(self, dist: IndianBuffetProcessDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _format(self, z: np.ndarray) -> list[int] | list[bool]:
        if self.dist.data_format == "sparse":
            return list(np.flatnonzero(z).astype(int))
        return z.astype(int).tolist()

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw one feature row or ``size`` iid feature rows."""
        if size is None:
            z = self.rng.rand(self.dist.num_features) <= self.dist.feature_probs
            return self._format(z)

        draws = self.rng.rand(size, self.dist.num_features) <= self.dist.feature_probs.reshape((1, -1))
        return [self._format(draws[i, :]) for i in range(size)]


class IndianBuffetProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulates weighted feature-use counts for the IBP VB update."""

    def __init__(
        self, num_features: int, alpha: float = 1.0, keys: str | None = None, data_format: str = "auto"
    ) -> None:
        self.num_features = _validate_num_features(num_features)
        self.alpha = _validate_alpha(alpha)
        self.keys = keys
        self.data_format = _check_data_format(data_format)
        self.feature_counts = np.zeros(self.num_features, dtype=np.float64)
        self.total_count = 0.0
        # Data log-likelihood accumulated as a byproduct of the E-step (the plug-in per-row
        # log_density), only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path
        # pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

    def update(self, x: Any, weight: float, estimate: IndianBuffetProcessDistribution | None) -> None:
        """Update weighted feature-use counts from one row."""
        if estimate is not None:
            self.alpha = estimate.alpha
        xx = _to_binary_vector(x, self.num_features, self.data_format)
        self.feature_counts += weight * xx
        self.total_count += weight

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize feature-use counts from one row."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: IndianBuffetProcessDistribution | None) -> None:
        """Update weighted feature-use counts from encoded rows."""
        if estimate is not None:
            self.alpha = estimate.alpha
        xx = np.asarray(x, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        self.feature_counts += np.dot(ww, xx)
        self.total_count += float(np.sum(ww))
        if self._track_ll:
            if estimate is None:
                # No model to score against (e.g. seq_initialize); signal the fused-EM caller to
                # fall back to a separate scoring pass.
                self._seq_ll = None
            elif self._seq_ll is not None:
                # Per-row plug-in log-density (== IndianBuffetProcessDistribution.seq_log_density).
                row_ll = estimate.log_nsum + np.dot(xx, estimate.log_dvec)
                self._seq_ll += float(np.dot(ww, row_ll))

    def seq_update_engine(
        self, x: np.ndarray, weights: Any, estimate: IndianBuffetProcessDistribution | None, engine: Any
    ) -> None:
        """Engine-resident accumulation of weighted feature-use counts (numpy or torch).

        The weighted feature counts are reduced via a weight-vector / binary-matrix product on the
        active engine; the alpha metadata is host bookkeeping. Matches seq_update.
        """
        if estimate is not None:
            self.alpha = estimate.alpha
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        w = engine.asarray(weights_np)
        xx = engine.asarray(np.asarray(x, dtype=np.float64))
        self.feature_counts += np.asarray(engine.to_numpy(engine.matmul(w, xx)), dtype=np.float64)
        self.total_count += float(engine.to_numpy(engine.sum(w)))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize feature-use counts from encoded rows."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: SS) -> "IndianBuffetProcessAccumulator":
        """Merge feature-use counts, total count, and alpha metadata."""
        self.feature_counts += suff_stat[0]
        self.total_count += suff_stat[1]
        if suff_stat[2] is not None:
            self.alpha = suff_stat[2]
        return self

    def value(self) -> SS:
        """Return feature-use counts, total count, and alpha metadata."""
        return self.feature_counts.copy(), self.total_count, self.alpha

    def from_value(self, x: SS) -> "IndianBuffetProcessAccumulator":
        """Restore feature-use counts, total count, and alpha metadata."""
        self.feature_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.total_count = float(x[1])
        if x[2] is not None:
            self.alpha = float(x[2])
        return self

    def scale(self, c: float) -> "IndianBuffetProcessAccumulator":
        """Scale additive counts while preserving the IBP concentration metadata."""
        self.feature_counts *= c
        self.total_count *= c
        return self

    def acc_to_encoder(self) -> "IndianBuffetProcessDataEncoder":
        """Return the encoder compatible with finite IBP sufficient statistics."""
        return IndianBuffetProcessDataEncoder(self.num_features, self.data_format)


class IndianBuffetProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for IBP accumulators."""

    def __init__(
        self, num_features: int, alpha: float = 1.0, keys: str | None = None, data_format: str = "auto"
    ) -> None:
        self.num_features = _validate_num_features(num_features)
        self.alpha = _validate_alpha(alpha)
        self.keys = keys
        self.data_format = _check_data_format(data_format)

    def make(self) -> IndianBuffetProcessAccumulator:
        """Create an empty finite IBP accumulator."""
        return IndianBuffetProcessAccumulator(self.num_features, self.alpha, self.keys, self.data_format)


class IndianBuffetProcessEstimator(ParameterEstimator):
    """Variational Bayes estimator for the finite-truncated IBP.

    pseudo_count follows the convention used by other mixle.stats Bernoulli
    estimators: if suff_stat is supplied, it is treated as a prior probability
    vector and re-weighted by pseudo_count; otherwise pseudo_count is centered at
    the IBP prior mean alpha / (alpha + K).
    """

    def __init__(
        self,
        num_features: int,
        alpha: float = 1.0,
        pseudo_count: float | None = None,
        suff_stat: Sequence[float] | np.ndarray | None = None,
        estimate_alpha: bool = True,
        min_alpha: float = 1.0e-12,
        max_alpha: float = 1.0e12,
        min_prob: float = 1.0e-128,
        name: str | None = None,
        keys: str | None = None,
        data_format: str = "auto",
    ) -> None:
        self.num_features = _validate_num_features(num_features)
        self.alpha = _validate_alpha(alpha)
        self.pseudo_count = pseudo_count
        self.suff_stat = None if suff_stat is None else _validate_probability_vector(suff_stat, self.num_features)
        self.estimate_alpha = estimate_alpha
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.min_prob = float(min_prob)
        self.name = name
        self.keys = keys
        self.data_format = _check_data_format(data_format)

    def accumulator_factory(self) -> IndianBuffetProcessAccumulatorFactory:
        """Return a factory for finite IBP sufficient-statistic accumulators."""
        return IndianBuffetProcessAccumulatorFactory(self.num_features, self.alpha, self.keys, self.data_format)

    def estimate(self, nobs: float | None, suff_stat: SS) -> IndianBuffetProcessDistribution:
        """Update the variational Beta posterior and optional concentration parameter."""
        feature_counts, total_count, prev_alpha = suff_stat
        alpha = self.alpha if prev_alpha is None else _validate_alpha(prev_alpha)
        feature_counts = np.asarray(feature_counts, dtype=np.float64)

        if feature_counts.shape != (self.num_features,):
            raise ValueError("IBP sufficient statistics have the wrong feature dimension")

        active_pseudo = np.zeros(self.num_features, dtype=np.float64)
        inactive_pseudo = np.zeros(self.num_features, dtype=np.float64)
        if self.pseudo_count is not None:
            pc = float(self.pseudo_count)
            if pc < 0.0:
                raise ValueError("pseudo_count must be non-negative")
            if self.suff_stat is not None:
                prior_probs = self.suff_stat
            else:
                prior_probs = np.full(self.num_features, alpha / (alpha + self.num_features), dtype=np.float64)
            active_pseudo = pc * prior_probs
            inactive_pseudo = pc * (1.0 - prior_probs)

        prior_a = alpha / float(self.num_features)
        post_a = prior_a + feature_counts + active_pseudo
        post_b = 1.0 + (float(total_count) - feature_counts) + inactive_pseudo
        post_a = np.maximum(post_a, np.finfo(np.float64).tiny)
        post_b = np.maximum(post_b, np.finfo(np.float64).tiny)

        new_alpha = alpha
        if self.estimate_alpha:
            elog_pi = digamma(post_a) - digamma(post_a + post_b)
            denom = float(np.sum(elog_pi))
            if denom < 0.0 and np.isfinite(denom):
                new_alpha = -float(self.num_features * self.num_features) / denom
                new_alpha = min(max(new_alpha, self.min_alpha), self.max_alpha)

        beta_params = np.column_stack((post_a, post_b))
        return IndianBuffetProcessDistribution(
            self.num_features,
            alpha=new_alpha,
            beta_params=beta_params,
            min_prob=self.min_prob,
            name=self.name,
            keys=self.keys,
            data_format=self.data_format,
        )

    def model_log_density(self, model: IndianBuffetProcessDistribution) -> float:
        """Global VB term E_q[log p(pi | alpha)] + H[q(pi)]."""
        bp = model.beta_params
        a = bp[:, 0]
        b = bp[:, 1]
        ab = a + b
        prior_a = model.alpha / float(model.num_features)
        elog_pi = digamma(a) - digamma(ab)
        entropy = betaln(a, b) - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b) + (ab - 2.0) * digamma(ab)
        return float(np.sum(np.log(prior_a) + (prior_a - 1.0) * elog_pi + entropy))


class IndianBuffetProcessDataEncoder(DataSequenceEncoder):
    """Encode dense or sparse IBP rows as a dense boolean matrix."""

    def __init__(self, num_features: int, data_format: str = "auto") -> None:
        self.num_features = _validate_num_features(num_features)
        self.data_format = _check_data_format(data_format)

    def __str__(self) -> str:
        return "IndianBuffetProcessDataEncoder(num_features=%s, data_format=%s)" % (
            repr(self.num_features),
            repr(self.data_format),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, IndianBuffetProcessDataEncoder)
            and self.num_features == other.num_features
            and self.data_format == other.data_format
        )

    def seq_encode(self, x: Sequence[Any] | np.ndarray) -> np.ndarray:
        """Validate and encode dense or sparse feature rows as a boolean matrix."""
        if isinstance(x, np.ndarray) and x.ndim == 2:
            if x.shape[1] != self.num_features:
                raise ValueError("dense IBP matrix must have num_features columns")
            if not np.all(np.logical_or(x == 0, x == 1)):
                raise ValueError("dense IBP matrix must contain only 0/1 values")
            return x.astype(bool, copy=False)

        rows = [_to_binary_vector(u, self.num_features, self.data_format) for u in x]
        if len(rows) == 0:
            return np.zeros((0, self.num_features), dtype=bool)
        return np.vstack(rows)
