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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import RandomState
from scipy.special import betaln, digamma

from pysp.stats.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


SS = Tuple[np.ndarray, float, Optional[float]]


def _check_data_format(data_format: str) -> str:
    if data_format not in ('auto', 'dense', 'sparse'):
        raise ValueError("data_format must be one of 'auto', 'dense', or 'sparse'")
    return data_format


def _validate_num_features(num_features: int) -> int:
    num_features = int(num_features)
    if num_features <= 0:
        raise ValueError('num_features must be positive')
    return num_features


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError('alpha must be a finite positive value')
    return alpha


def _validate_probability_vector(p: Union[Sequence[float], np.ndarray], num_features: int,
                                 min_prob: float = 0.0) -> np.ndarray:
    rv = np.asarray(p, dtype=np.float64)
    if rv.shape != (num_features,):
        raise ValueError('feature probability vector must have length num_features')
    if np.any(np.isnan(rv)) or np.any(rv < 0.0) or np.any(rv > 1.0):
        raise ValueError('feature probabilities must lie in [0, 1]')
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
        raise TypeError('IBP observations must be binary vectors or active-feature indices, not strings')
    return np.asarray(list(x))


def _to_binary_vector(x: Any, num_features: int, data_format: str) -> np.ndarray:
    data_format = _check_data_format(data_format)
    xx = _as_1d_array(x)

    if xx.ndim != 1:
        raise ValueError('IBP observations must be one-dimensional')

    if data_format == 'dense' or (data_format == 'auto' and _is_binary_vector(xx, num_features)):
        if xx.size != num_features:
            raise ValueError('dense IBP observations must have length num_features')
        if not _is_binary_vector(xx, num_features):
            raise ValueError('dense IBP observations must contain only 0/1 values')
        return xx.astype(bool, copy=False)

    idx = np.asarray(xx, dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= num_features):
        raise ValueError('sparse IBP feature index out of range')
    rv = np.zeros(num_features, dtype=bool)
    if idx.size:
        rv[np.unique(idx)] = True
    return rv


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

    def __init__(self, num_features: int, alpha: float = 1.0,
                 beta_params: Optional[Union[Sequence[Sequence[float]], np.ndarray]] = None,
                 feature_probs: Optional[Union[Sequence[float], np.ndarray]] = None,
                 min_prob: float = 1.0e-128, name: Optional[str] = None,
                 keys: Optional[str] = None, data_format: str = 'auto') -> None:
        self.num_features = _validate_num_features(num_features)
        self.alpha = _validate_alpha(alpha)
        self.name = name
        self.keys = keys
        self.min_prob = float(min_prob)
        self.data_format = _check_data_format(data_format)

        if beta_params is not None:
            bp = np.asarray(beta_params, dtype=np.float64)
            if bp.shape != (self.num_features, 2):
                raise ValueError('beta_params must have shape (num_features, 2)')
            if np.any(bp <= 0.0) or np.any(~np.isfinite(bp)):
                raise ValueError('beta_params must be finite positive values')
            self.beta_params = bp.copy()
        elif feature_probs is not None:
            p = _validate_probability_vector(feature_probs, self.num_features, self.min_prob)
            self.beta_params = np.column_stack((p, 1.0 - p))
        else:
            self.beta_params = np.column_stack((
                np.full(self.num_features, self.alpha / float(self.num_features), dtype=np.float64),
                np.ones(self.num_features, dtype=np.float64),
            ))

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
        return ('IndianBuffetProcessDistribution(%s, alpha=%s, beta_params=%s, min_prob=%s, '
                'name=%s, keys=%s, data_format=%s)') % (
                    repr(self.num_features), repr(self.alpha), repr(self.beta_params.tolist()),
                    repr(self.min_prob), repr(self.name), repr(self.keys), repr(self.data_format))

    def density(self, x: Any) -> float:
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Plug-in log-density of one feature row using E_q[pi_k]."""
        xx = _to_binary_vector(x, self.num_features, self.data_format)
        return float(self.log_nsum + np.dot(xx, self.log_dvec))

    def expected_log_density(self, x: Any) -> float:
        """VB expected log-density E_q[log p(z | pi)] for one observed row."""
        xx = _to_binary_vector(x, self.num_features, self.data_format)
        return float(self.expected_log_nsum + np.dot(xx, self.expected_log_dvec))

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64)
        return self.log_nsum + np.dot(xx, self.log_dvec)

    def seq_expected_log_density(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64)
        return self.expected_log_nsum + np.dot(xx, self.expected_log_dvec)

    def seq_local_elbo(self, x: np.ndarray) -> np.ndarray:
        """Per-row VB contribution; rows are observed, so there is no local entropy."""
        return self.seq_expected_log_density(x)

    def sampler(self, seed: Optional[int] = None) -> 'IndianBuffetProcessSampler':
        return IndianBuffetProcessSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IndianBuffetProcessEstimator':
        suff_stat = self.feature_probs.copy() if pseudo_count is not None else None
        return IndianBuffetProcessEstimator(
            self.num_features, alpha=self.alpha, pseudo_count=pseudo_count, suff_stat=suff_stat,
            min_prob=self.min_prob, name=self.name, keys=self.keys, data_format=self.data_format)

    def dist_to_encoder(self) -> 'IndianBuffetProcessDataEncoder':
        return IndianBuffetProcessDataEncoder(self.num_features, self.data_format)


class IndianBuffetProcessSampler(DistributionSampler):
    """Sampler for finite-truncated IBP feature rows."""

    def __init__(self, dist: IndianBuffetProcessDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _format(self, z: np.ndarray) -> Union[List[int], List[bool]]:
        if self.dist.data_format == 'sparse':
            return list(np.flatnonzero(z).astype(int))
        return z.astype(int).tolist()

    def sample(self, size: Optional[int] = None) -> Union[List[int], List[List[int]]]:
        if size is None:
            z = self.rng.rand(self.dist.num_features) <= self.dist.feature_probs
            return self._format(z)

        draws = self.rng.rand(size, self.dist.num_features) <= self.dist.feature_probs.reshape((1, -1))
        return [self._format(draws[i, :]) for i in range(size)]


class IndianBuffetProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulates weighted feature-use counts for the IBP VB update."""

    def __init__(self, num_features: int, alpha: float = 1.0, keys: Optional[str] = None,
                 data_format: str = 'auto') -> None:
        self.num_features = _validate_num_features(num_features)
        self.alpha = _validate_alpha(alpha)
        self.key = keys
        self.data_format = _check_data_format(data_format)
        self.feature_counts = np.zeros(self.num_features, dtype=np.float64)
        self.total_count = 0.0

    def update(self, x: Any, weight: float, estimate: Optional[IndianBuffetProcessDistribution]) -> None:
        if estimate is not None:
            self.alpha = estimate.alpha
        xx = _to_binary_vector(x, self.num_features, self.data_format)
        self.feature_counts += weight * xx
        self.total_count += weight

    def initialize(self, x: Any, weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray,
                   estimate: Optional[IndianBuffetProcessDistribution]) -> None:
        if estimate is not None:
            self.alpha = estimate.alpha
        xx = np.asarray(x, dtype=np.float64)
        self.feature_counts += np.dot(np.asarray(weights, dtype=np.float64), xx)
        self.total_count += float(np.sum(weights))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: SS) -> 'IndianBuffetProcessAccumulator':
        self.feature_counts += suff_stat[0]
        self.total_count += suff_stat[1]
        if suff_stat[2] is not None:
            self.alpha = suff_stat[2]
        return self

    def value(self) -> SS:
        return self.feature_counts.copy(), self.total_count, self.alpha

    def from_value(self, x: SS) -> 'IndianBuffetProcessAccumulator':
        self.feature_counts = np.asarray(x[0], dtype=np.float64).copy()
        self.total_count = float(x[1])
        if x[2] is not None:
            self.alpha = float(x[2])
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'IndianBuffetProcessDataEncoder':
        return IndianBuffetProcessDataEncoder(self.num_features, self.data_format)


class IndianBuffetProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for IBP accumulators."""

    def __init__(self, num_features: int, alpha: float = 1.0, keys: Optional[str] = None,
                 data_format: str = 'auto') -> None:
        self.num_features = _validate_num_features(num_features)
        self.alpha = _validate_alpha(alpha)
        self.keys = keys
        self.data_format = _check_data_format(data_format)

    def make(self) -> IndianBuffetProcessAccumulator:
        return IndianBuffetProcessAccumulator(self.num_features, self.alpha, self.keys, self.data_format)


class IndianBuffetProcessEstimator(ParameterEstimator):
    """Variational Bayes estimator for the finite-truncated IBP.

    pseudo_count follows the convention used by other pysp.stats Bernoulli
    estimators: if suff_stat is supplied, it is treated as a prior probability
    vector and re-weighted by pseudo_count; otherwise pseudo_count is centered at
    the IBP prior mean alpha / (alpha + K).
    """

    def __init__(self, num_features: int, alpha: float = 1.0,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Union[Sequence[float], np.ndarray]] = None,
                 estimate_alpha: bool = True,
                 min_alpha: float = 1.0e-12,
                 max_alpha: float = 1.0e12,
                 min_prob: float = 1.0e-128,
                 name: Optional[str] = None,
                 keys: Optional[str] = None,
                 data_format: str = 'auto') -> None:
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
        return IndianBuffetProcessAccumulatorFactory(self.num_features, self.alpha, self.keys, self.data_format)

    def estimate(self, nobs: Optional[float], suff_stat: SS) -> IndianBuffetProcessDistribution:
        feature_counts, total_count, prev_alpha = suff_stat
        alpha = self.alpha if prev_alpha is None else _validate_alpha(prev_alpha)
        feature_counts = np.asarray(feature_counts, dtype=np.float64)

        if feature_counts.shape != (self.num_features,):
            raise ValueError('IBP sufficient statistics have the wrong feature dimension')

        active_pseudo = np.zeros(self.num_features, dtype=np.float64)
        inactive_pseudo = np.zeros(self.num_features, dtype=np.float64)
        if self.pseudo_count is not None:
            pc = float(self.pseudo_count)
            if pc < 0.0:
                raise ValueError('pseudo_count must be non-negative')
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
            self.num_features, alpha=new_alpha, beta_params=beta_params, min_prob=self.min_prob,
            name=self.name, keys=self.keys, data_format=self.data_format)

    def model_log_density(self, model: IndianBuffetProcessDistribution) -> float:
        """Global VB term E_q[log p(pi | alpha)] + H[q(pi)]."""
        bp = model.beta_params
        a = bp[:, 0]
        b = bp[:, 1]
        ab = a + b
        prior_a = model.alpha / float(model.num_features)
        elog_pi = digamma(a) - digamma(ab)
        entropy = betaln(a, b) - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b) \
            + (ab - 2.0) * digamma(ab)
        return float(np.sum(np.log(prior_a) + (prior_a - 1.0) * elog_pi + entropy))


class IndianBuffetProcessDataEncoder(DataSequenceEncoder):
    """Encode dense or sparse IBP rows as a dense boolean matrix."""

    def __init__(self, num_features: int, data_format: str = 'auto') -> None:
        self.num_features = _validate_num_features(num_features)
        self.data_format = _check_data_format(data_format)

    def __str__(self) -> str:
        return 'IndianBuffetProcessDataEncoder(num_features=%s, data_format=%s)' % (
            repr(self.num_features), repr(self.data_format))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IndianBuffetProcessDataEncoder) \
            and self.num_features == other.num_features \
            and self.data_format == other.data_format

    def seq_encode(self, x: Union[Sequence[Any], np.ndarray]) -> np.ndarray:
        if isinstance(x, np.ndarray) and x.ndim == 2:
            if x.shape[1] != self.num_features:
                raise ValueError('dense IBP matrix must have num_features columns')
            if not np.all(np.logical_or(x == 0, x == 1)):
                raise ValueError('dense IBP matrix must contain only 0/1 values')
            return x.astype(bool, copy=False)

        rows = [_to_binary_vector(u, self.num_features, self.data_format) for u in x]
        if len(rows) == 0:
            return np.zeros((0, self.num_features), dtype=bool)
        return np.vstack(rows)
