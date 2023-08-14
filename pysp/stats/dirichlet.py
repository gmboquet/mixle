""""Create, estimate, and sample from a hierarchical mixture distribution with K components consisting of
sequence mixture distribution with L topics shared across all K components.

Defines the DirichletDistribution, DirichletSampler, DirichletAccumulatorFactory, DirichletAccumulator,
DirichletEstimator, and the DirichletDataEncoder classes for use with pysparkplug.

Data type: Union[List[float], np.ndarray[float]]

The log-density of a Dirichlet with dim = K, is given by
    log(p_mat(x)) = -log(Const) + sum_{k=0}^{K-1} (alpha_k -1)*log(x_k), for sum_k x_k = 1.0,
else 0. In above,
    log(Const) = sum_{k=0}^{K-1} log(Gamma(alpha_k)) - log(Gamma(sum_{k=0}^{K-1} alpha_k)).

"""
import numpy as np
import sys
from numpy.random import RandomState
from pysp.utils.special import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler,\
    SequenceEncodableStatisticAccumulator, DataSequenceEncoder, StatisticAccumulatorFactory

from typing import Union, List, Any, Optional, Dict, Sequence, Tuple, Callable


def dirichlet_param_solve(alpha: np.ndarray, mean_log_p: np.ndarray, delta: float) -> Tuple[np.ndarray, int]:
    """Iteratively solve for alpha of a Dirichlet distribution.

    Args:
        alpha (np.ndarray): Numpy array of Dirichlet parameters (all entries should be non-negative).
        mean_log_p (np.ndarray): Sufficient statistic (1/N) sum_{i=1}^{N} log(x_{i,k}), where N is the number of
            observations.
        delta (float): Tolerance for convergence of Newton-Method.

    Returns:
        Tuple[np.ndarray, int] containing the estimates of alpha and numer of iterations in solver.

    """
    dim = len(alpha)

    valid = np.bitwise_and(np.isfinite(alpha), alpha > 0)
    valid = np.bitwise_and(valid, np.isfinite(mean_log_p))

    alpha = alpha[valid]
    mlp = mean_log_p[valid]

    count = 0
    a_sum = alpha.sum()
    d_alpha = (2 * delta) + 1

    while d_alpha > delta:
        count += 1

        da_sum = digamma(a_sum)
        old_alpha = alpha
        adj_alpha = mlp + da_sum
        alpha = digammainv(adj_alpha)
        a_sum = np.sum(alpha)
        d_alpha = np.abs(alpha - old_alpha).sum()
        d_alpha /= a_sum

    if dim != alpha.size:
        rv = np.zeros(dim, dtype=float)
        rv[valid] = alpha
    else:
        rv = alpha

    return rv, count


def mpe(x0: np.ndarray, f: Callable[[np.ndarray], np.ndarray], eps: float) -> Tuple[np.ndarray, int]:
    x1 = f(x0)
    x2 = f(x1)
    x3 = f(x2)
    X = np.asarray([x0, x1, x2, x3])
    s0 = x3
    s = s0
    res = np.abs(x3 - x2).sum()
    its_cnt = 2

    while res > eps:
        y = f(X[-1, :])
        dy = y - X[-1, :]
        U = (X[1:, :] - X[:-1, :]).T
        X2 = X[1:, :].T
        c = np.dot(np.linalg.pinv(U), dy)
        c *= -1
        s = (np.dot(X2, c) + y) / (c.sum() + 1)

        res = np.abs(s - s0).sum()
        s0 = s
        X = np.concatenate((X, np.reshape(y, (1, -1))), axis=0)
        its_cnt += 1

    return s, its_cnt


def alpha_seq_lambda(mean_log_p: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def next_alpha(current_alpha):
        return digammainv(mean_log_p + digamma(current_alpha.sum()))

    return next_alpha


def find_alpha(current_alpha, mlp, thresh) -> Tuple[np.ndarray, int]:
    f = alpha_seq_lambda(mlp)
    return mpe(current_alpha, f, thresh)


class DirichletDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, alpha: Union[List[float], np.ndarray], name: Optional[str] = None, keys: Optional[str] = None) \
            -> None:
        """DirichletDistribution object defining Dirichlet distribution with parameter alpha.

        Args:
            alpha (Union[List[float], np.ndarray]): Array of alpha values. Determines size of Dirichlet distribution.
            name (Optional[str]): Set name for distribution.
            keys (Optional[str]): Set key for merging sufficient statistics with objects containing matching key.

        Attributes:
            dim (int): Number of categories in Dirichlet.
            alpha (np.ndarray): Concentration parameters of length dim.
            alpha_ma (ndarray): Numpy array of bools denoting positive alpha entries with True.
            log_const (float): Normalizing constant for distribution. Beta(alpha) on wiki.
            has_invalid (bool): True if any alpha are less than or equal to 0.
            name (Optional[str]): Optional name for object instance.
            key (Optional[str]): Optional key for merging sufficient statistics with objects containing matching key.

        """
        temp_alpha = np.asarray(alpha)
        temp_mask = temp_alpha <= 0

        self.dim = len(alpha)
        self.alpha = temp_alpha
        self.alpha_ma = ~temp_mask
        self.log_const = sum(gammaln(alpha)) - gammaln(sum(alpha))
        self.has_invalid = np.any(temp_mask)
        self.name = name
        self.key = keys

    def __str__(self) -> str:
        """Returns a string representation of object instance."""
        s1 = repr(list(self.alpha))
        s2 = repr(self.name)
        s3 = repr(self.key)
        return 'DirichletDistribution(%s, name=%s, keys=%s)' % (s1, s2, s3)

    def density(self, x: Union[List[float], np.ndarray]) -> float:
        """Evaluate the density of a dirichlet observation.

        See log_density() for details.

        Args:
            x (Union[List[float], np.ndarray]): A single dirichlet observation.

        Returns:
            Density evaluated at x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: Union[List[float], np.ndarray]) -> float:
        """Evaluate the log-density of a dirichlet observation.

        The log-density of a Dirichlet with dim = K, is given by

            log(p_mat(x)) = -log(Const) + sum_{k=0}^{K-1} (alpha_k -1)*log(x_k), for sum_k x_k = 1.0,

        else 0. In above

            log(Const) = sum_{k=0}^{K-1} log(Gamma(alpha_k)) - log(Gamma(sum_{k=0}^{K-1} alpha_k)).

        Args:
            x (Union[List[float], np.ndarray]): A single dirichlet observation.

        Returns:
            Log-density evaluated at x.

        """
        xx = np.asarray(x)
        zz = np.bitwise_or(xx > 0, self.alpha_ma)
        cnt = np.count_nonzero(zz)

        if cnt == self.dim:
            rv = np.dot(np.log(x), self.alpha - 1.0)
            rv -= self.log_const
        elif cnt == 0:
            rv = 0.0
        else:
            rv = np.dot(np.log(xx[zz]), self.alpha[zz] - 1.0)
            rv -= self.log_const

        return rv

    def seq_log_density(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        rv = np.dot(x[0], self.alpha - 1.0)
        rv -= self.log_const
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'DirichletSampler':
        return DirichletSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'DirichletEstimator':

        if pseudo_count is None:
            return DirichletEstimator(dim=self.dim, name=self.name)
        else:
            return DirichletEstimator(dim=self.dim, pseudo_count=pseudo_count,
                                      suff_stat=log(self.alpha / sum(self.alpha)), name=self.name)

    def dist_to_encoder(self) -> 'DirichletDataEncoder':
        """Create DirichletDataEncoder object for encoding sequences of iid Dirichlet observations."""
        return DirichletDataEncoder()

class DirichletSampler(DistributionSampler):

    def __init__(self, dist: DirichletDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> np.ndarray:

        alpha = self.dist.alpha
        has_invalid = self.dist.has_invalid
        alpha_ma = self.dist.alpha_ma

        if has_invalid:
            if size is None:
                rv = np.zeros(alpha.size)
                rv[alpha_ma] = self.rng.dirichlet(alpha=alpha[alpha_ma])
            else:
                rv = np.zeros((size, alpha.size))
                rv[:, alpha_ma] = self.rng.dirichlet(alpha=alpha[alpha_ma], size=size)

            return rv
        else:
            return self.rng.dirichlet(alpha=self.dist.alpha, size=size)


class DirichletAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, dim: int, keys: Optional[str] = None) -> None:
        self.dim = dim
        self.sum_of_logs = np.zeros(dim)
        self.sum = np.zeros(dim)
        self.sum2 = np.zeros(dim)
        self.counts = 0
        self.key = keys

    def update(self, x: Union[np.ndarray, List[float]], weight: float, estimate: Optional['DirichletDistribution'])\
            -> None:
        xx = np.asarray(x)
        z = xx > 0
        if np.all(z):
            self.sum_of_logs += log(xx) * weight
            self.sum += weight * xx
            self.sum2 += weight * xx * xx
            self.counts += weight
        else:
            self.sum_of_logs[z] += log(x[z]) * weight
            self.sum += weight * x
            self.sum2 += weight * x * x
            self.counts += weight

    def initialize(self, x: Union[np.ndarray, List[float]], weight: float, estimate: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def get_seq_lambda(self):
        return [self.seq_update]

    def seq_update(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray,
                   estimate: Optional[DirichletDistribution]) -> None:
        self.sum_of_logs += np.dot(weights, x[0])
        self.counts += weights.sum()
        self.sum += np.dot(weights, x[1])
        self.sum2 += np.dot(weights, x[2])

    def seq_initialize(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray,
                       rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> 'DirichletAccumulator':
        self.sum_of_logs += suff_stat[1]
        self.sum += suff_stat[2]
        self.sum2 += suff_stat[3]
        self.counts += suff_stat[0]
        return self

    def value(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        return self.counts, self.sum_of_logs, self.sum, self.sum2

    def from_value(self, x: Tuple[int, np.ndarray, np.ndarray, np.ndarray]):
        self.counts = x[0]
        self.sum_of_logs = x[1]
        self.sum = x[2]
        self.sum2 = x[3]

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'DirichletDataEncoder':
        """Create DirichletDataEncoder object for encoding sequences of iid Dirichlet observations."""
        return DirichletDataEncoder()

class DirichletAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, dim: int, keys: Optional[str] = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> 'DirichletAccumulator':
        return DirichletAccumulator(dim=self.dim, keys=self.keys)


class DirichletEstimator(ParameterEstimator):

    def __init__(self, dim: int, pseudo_count: Optional[float] = None, suff_stat: Optional[np.ndarray] = None,
                 delta: Optional[float] = 1.0e-8, keys: Optional[str] = None,
                 use_mpe: bool = False, name: Optional[str] = None) -> None:
        self.dim = dim
        self.pseudo_count = pseudo_count
        self.delta = delta
        self.suff_stat = suff_stat
        self.keys = keys
        self.use_mpe = use_mpe
        self.name = name

    def accumulator_factory(self) -> 'DirichletAccumulatorFactory':
        """Create DirichletAccumulator object from attributes variables."""
        return DirichletAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[int, np.ndarray, np.ndarray, np.ndarray])\
            -> DirichletDistribution:

        nobs, sum_of_logs, sum_v, sum_v2 = suff_stat
        dim = len(sum_of_logs)

        if self.pseudo_count is not None and self.suff_stat is None:
            c1 = digamma(one) - digamma(dim)
            c2 = sum_of_logs + c1 * self.pseudo_count
            initial_estimate = c2 * (dim / sum(c2))
            mean_log_p = c2 / (nobs + self.pseudo_count)

        elif self.pseudo_count is not None and self.suff_stat is not None:
            c2 = sum_of_logs + self.suff_stat * self.pseudo_count
            initial_estimate = c2 * (dim / sum(c2))
            mean_log_p = c2 / (nobs + self.pseudo_count)

        else:

            sum_v = sum_v / nobs
            sum_v2 = sum_v2 / nobs
            sum_v[-1] = 1.0 - sum_v[:-1].sum()

            '''
            #initialConst = (sum_v[0]-sum_v2[0])/(sum_v2[0]-sum_v[0]*sum_v[0])
            initialConst1 = (sum_v - sum_v2).mean()
            initialConst2 = (sum_v2 - sum_v*sum_v).mean()

            if initialConst2 > 0 and initialConst1 > 0:
                initial_estimate = (initialConst1/initialConst2)*sum_v
            else:
                initial_estimate = sum_of_logs * (dim / sum(sum_of_logs))

            #initial_estimate = sum_of_logs*(dim/sum(sum_of_logs))

            '''
            initial_estimate = sum_v

            mean_log_p = sum_of_logs / nobs

        if nobs == 1.0:
            return DirichletDistribution(initial_estimate, name=self.name)

        else:

            if self.use_mpe:
                alpha, its_cnt = find_alpha(np.asarray(initial_estimate), mean_log_p, self.delta)
            else:
                alpha, its_cnt = dirichlet_param_solve(np.asarray(initial_estimate), mean_log_p, self.delta)

            return DirichletDistribution(alpha, name=self.name)

class DirichletDataEncoder(DataSequenceEncoder):

    def __str__(self) -> str:
        return 'DirichletDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DirichletDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[float]]):
        rv = np.asarray(x).copy()

        # TODO: Add warning for invalid values

        rv2 = np.maximum(rv, sys.float_info.min)
        np.log(rv2, out=rv2)
        return rv2, rv, rv * rv

