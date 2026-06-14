"""Dirichlet distribution on probability vectors: observations are length-d
sequences/arrays of non-negative reals summing to one (points on the
(d-1)-simplex).

Two parameterizations are supported: a concentration vector alpha (dim = d),
or a single scalar alpha treated as a symmetric Dirichlet whose dimension is
inferred from each observation (dim == 0).

Estimation is maximum likelihood from the mean-log sufficient statistics,
solved with fixed-point iterations on the digamma inverse
(dirichlet_param_solve), optionally accelerated with minimal polynomial
extrapolation (use_mpe).
"""

import sys

import numpy as np
from numpy.random import RandomState
from scipy.special import gammaln

from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, SequenceEncodableAccumulator
from pysp.utils.special import *

_MIN_DIRICHLET_ALPHA = 1.0e-10
_MAX_DIRICHLET_ALPHA = 1.0e10
_MAX_DIRICHLET_ITERATIONS = 10000


def _safe_simplex_mean(x, dim):
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


def _mean_from_mean_log(meanLogP, dim):
    mlp = np.asarray(meanLogP, dtype=float).copy()
    finite = np.isfinite(mlp)
    if not np.any(finite):
        return np.ones(dim, dtype=float) / float(dim)
    floor = np.min(mlp[finite])
    mlp[~finite] = floor
    mlp -= np.max(mlp)
    rv = np.exp(np.maximum(mlp, -745.0))
    return _safe_simplex_mean(rv, dim)


def _initial_dirichlet_alpha(mean_v, mean_v2=None, meanLogP=None):
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

    if (not np.isfinite(alpha0) or alpha0 <= 0.0) and meanLogP is not None:
        alpha0 = float(len(mean))
    if not np.isfinite(alpha0) or alpha0 <= 0.0:
        alpha0 = 1.0

    alpha0 = min(_MAX_DIRICHLET_ALPHA, max(_MIN_DIRICHLET_ALPHA * len(mean), alpha0))
    alpha = mean * alpha0
    return np.clip(alpha, _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)


def _valid_alpha(alpha, dim=None):
    arr = np.asarray(alpha, dtype=float)
    return (dim is None or arr.size == dim) and np.all(np.isfinite(arr)) and np.all(arr > 0.0)


def dirichlet_param_solve(alpha, meanLogP, delta, max_iter=_MAX_DIRICHLET_ITERATIONS):
    """Fixed-point solve for the ML Dirichlet parameters.

    Iterates alpha <- digammainv(meanLogP + digamma(sum(alpha))) until the
    relative change drops below delta. Entries with non-finite or non-positive
    starting values (or non-finite mean-log statistics) are held at zero.

    Args:
            alpha (np.ndarray): Initial concentration parameter estimate.
            meanLogP (np.ndarray): Mean of the log observations per component.
            delta (float): Convergence threshold on the relative parameter change.

    Returns:
            Tuple[np.ndarray, int]: Estimated parameters and iteration count.
    """

    dim = len(alpha)
    delta = 1.0e-8 if delta is None else max(float(delta), 1.0e-12)
    mlp = np.asarray(meanLogP, dtype=float).copy()
    if mlp.size != dim:
        mlp = np.full(dim, digamma(1.0) - digamma(float(dim)), dtype=float)
    finite = np.isfinite(mlp)
    if not np.any(finite):
        mlp[:] = digamma(1.0) - digamma(float(dim))
    else:
        mlp[~finite] = np.min(mlp[finite])

    alpha = np.asarray(alpha, dtype=float).copy()
    if not _valid_alpha(alpha, dim):
        alpha = _initial_dirichlet_alpha(_mean_from_mean_log(mlp, dim), meanLogP=mlp)
    alpha = np.clip(alpha, _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)

    for count in range(1, max_iter + 1):
        old_alpha = alpha.copy()
        asum = float(alpha.sum())
        if not np.isfinite(asum) or asum <= 0.0:
            alpha = _initial_dirichlet_alpha(_mean_from_mean_log(mlp, dim), meanLogP=mlp)
            asum = float(alpha.sum())
        adj_alpha = mlp + digamma(asum)
        alpha = np.asarray(digammainv(adj_alpha), dtype=float)
        bad = ~np.isfinite(alpha) | (alpha <= 0.0)
        if np.any(bad):
            alpha[bad] = old_alpha[bad]
        alpha = np.clip(alpha, _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)
        denom = max(_MIN_DIRICHLET_ALPHA, float(alpha.sum()))
        dalpha = float(np.abs(alpha - old_alpha).sum() / denom)
        if dalpha <= delta:
            return alpha, count

    return alpha, max_iter


def mpe(x0, f, eps, max_iter=1000):
    """Minimal polynomial extrapolation of the fixed-point sequence x <- f(x).

    Args:
            x0 (np.ndarray): Starting point.
            f (Callable): Fixed-point map.
            eps (float): Convergence threshold on successive extrapolants.

    Returns:
            Tuple[np.ndarray, int]: Extrapolated fixed point and iteration count.
    """

    x0 = np.clip(np.asarray(x0, dtype=float), _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)
    x1 = np.clip(f(x0), _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)
    x2 = np.clip(f(x1), _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)
    x3 = np.clip(f(x2), _MIN_DIRICHLET_ALPHA, _MAX_DIRICHLET_ALPHA)
    X = np.asarray([x0, x1, x2, x3])
    s0 = x3
    s = s0
    res = np.abs(x3 - x2).sum()
    its_cnt = 2

    while res > eps and its_cnt < max_iter:
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

    return s, its_cnt


def alpha_seq_lambda(meanLogP):
    """Returns the Dirichlet ML fixed-point map for the given mean-log statistics."""

    def next_alpha(currentAlpha):
        return digammainv(meanLogP + digamma(currentAlpha.sum()))

    return next_alpha


def find_alpha(current_alpha, mlp, thresh):
    """Solves for the ML Dirichlet parameters with MPE-accelerated iterations.

    Args:
            current_alpha (np.ndarray): Initial concentration parameter estimate.
            mlp (np.ndarray): Mean of the log observations per component.
            thresh (float): Convergence threshold.

    Returns:
            Tuple[np.ndarray, int]: Estimated parameters and iteration count.
    """
    f = alpha_seq_lambda(mlp)
    alpha, its = mpe(current_alpha, f, thresh)
    if not _valid_alpha(alpha, len(current_alpha)):
        return dirichlet_param_solve(current_alpha, mlp, thresh)
    return alpha, its


class DirichletDistribution(ProbabilityDistribution):
    """Dirichlet distribution with concentration parameters alpha on probability
    vectors. A scalar alpha denotes a symmetric Dirichlet of unspecified
    dimension (dim == 0)."""

    def __init__(self, alpha):
        """DirichletDistribution object.

        Args:
                alpha: Concentration parameters. Either a length-d sequence of
                        positive reals or a single positive scalar (symmetric
                        Dirichlet, dimension inferred from each observation).
        """
        self.set_parameters(alpha)

    def __str__(self):
        return "DirichletDistribution(%s)" % (str(self.alpha))

    def get_parameters(self):
        """Returns the concentration parameters (np.ndarray, or scalar if dim == 0)."""
        return self.alpha

    def set_parameters(self, params):
        """Sets the concentration parameters and refreshes the normalizer.

        Args:
                params: Length-d sequence of positive reals, or a positive scalar
                        for a symmetric Dirichlet of unspecified dimension.
        """
        if isinstance(params, (float, int)):
            self.dim = 0
            self.alpha = params
        else:
            self.alpha = np.asarray(params, dtype=float)
            self.dim = len(self.alpha)
            self.log_const = sum(gammaln(self.alpha)) - gammaln(sum(self.alpha))

    def cross_entropy(self, dist):
        """Cross entropy -E_self[log dist(x)] for a Dirichlet argument.

        Args:
                dist (DirichletDistribution): Distribution evaluated under this one.

        Returns:
                float: Cross entropy value in nats.
        """
        if isinstance(dist, DirichletDistribution):
            if self.dim == 0 and dist.dim != 0:
                a = self.alpha * np.ones(dist.dim)
                aa = dist.alpha
            elif self.dim != 0 and dist.dim == 0:
                a = self.alpha
                aa = dist.alpha * np.ones(self.dim)
            else:
                a = self.alpha
                aa = dist.alpha

            return -((gammaln(np.sum(aa)) - np.sum(gammaln(aa))) + np.dot(digamma(a) - digamma(np.sum(a)), aa - 1))
        else:
            raise NotImplementedError(
                "DirichletDistribution.cross_entropy is only implemented for Dirichlet arguments (got %s)."
                % type(dist).__name__
            )

    def entropy(self):
        """Returns the differential entropy in nats."""
        a = self.alpha
        a0 = np.sum(a)
        return -((gammaln(a0) - np.sum(gammaln(a))) + np.dot(digamma(a) - digamma(a0), a - 1))

    def density(self, x):
        """Density of the Dirichlet distribution at the probability vector x.

        Args:
                x: Length-d sequence of non-negative reals summing to one.

        Returns:
                float: Density value exp(log_density(x)).
        """
        return exp(self.log_density(x))

    def log_density(self, x):
        """Log density of the Dirichlet distribution at the probability vector x.

        Args:
                x: Length-d sequence of non-negative reals summing to one.

        Returns:
                float: Log density value.
        """
        if self.dim == 0:
            a = self.alpha
            rv = np.log(x).sum() * (a - 1)
            cc = gammaln(a) * len(x) - gammaln(len(x) * a)
            return rv - cc
        else:
            rv = np.dot(np.log(x), self.alpha - 1)
            return rv - self.log_const

        return rv

    def seq_log_density(self, x):
        """Vectorized log density for sequence-encoded observations.

        Args:
                x: Encoding (log values, values, squared values) from seq_encode,
                        where each entry is an (m x d) array.

        Returns:
                np.ndarray: Log density for each of the m encoded observations.
        """
        log_x = x[0]

        if len(log_x) == 0:
            return np.zeros(0, dtype=float)

        a = self.alpha
        n = log_x.shape[1]
        m = log_x.shape[0]

        if self.dim == 0:
            cc = gammaln(a) * n - gammaln(n * a)
            rv = np.zeros(m) - cc
            if a != 1:
                rv += log_x.sum(axis=1) * (a - 1)
        else:
            g = a != 1
            rv = np.dot(log_x[:, g], a[g] - 1.0)
            rv -= self.log_const
        return rv

    def seq_encode(self, x):
        """Encodes an iterable of probability vectors for vectorized evaluation.

        Args:
                x: Iterable of length-d probability vectors.

        Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: (log values, values,
                squared values), each of shape (m, d). Logs of zero entries are
                clamped at log(float_min).
        """
        rv = np.asarray(x).copy()

        # TODO: Add warning for invalid values

        rv2 = np.maximum(rv, sys.float_info.min)
        np.log(rv2, out=rv2)
        return rv2, rv, rv * rv

    def sampler(self, seed=None):
        """Returns a DirichletSampler for this distribution.

        Args:
                seed (Optional[int]): Random seed.

        Returns:
                DirichletSampler object.
        """
        return DirichletSampler(self, seed)

    def estimator(self, pseudo_count=None):
        """Returns a DirichletEstimator matching this distribution's dimension.

        Args:
                pseudo_count (Optional[float]): If given, the estimator regularizes
                        towards this distribution's normalized mean-log statistics with
                        the given weight.

        Returns:
                DirichletEstimator object.
        """
        if pseudo_count is None:
            return DirichletEstimator(dim=self.dim)
        else:
            return DirichletEstimator(
                dim=self.dim, pseudo_count=pseudo_count, suff_stat=log(self.alpha / sum(self.alpha))
            )


class DirichletSampler:
    """Sampler for DirichletDistribution. Draws probability vectors; components
    with invalid concentration (non-finite or <= 0) are fixed at zero."""

    def __init__(self, dist, seed=None):
        """DirichletSampler object.

        Args:
                dist (DirichletDistribution): Distribution to sample from.
                seed (Optional[int]): Random seed.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw Dirichlet-distributed probability vectors.

        Args:
                size (Optional[int]): Number of samples to draw.

        Returns:
                np.ndarray of shape (d,) if size is None, else of shape (size, d).

        Raises:
                ValueError: If the distribution has scalar alpha (dim == 0), since
                        the sample dimension is then unspecified.
        """
        if self.dist.dim == 0:
            raise ValueError(
                "DirichletSampler cannot sample from a symmetric DirichletDistribution with unspecified dimension (scalar alpha)."
            )

        alpha = np.asarray(self.dist.alpha, dtype=float)
        alpha_ma = np.isfinite(alpha) & (alpha > 0)
        has_invalid = not np.all(alpha_ma)

        if has_invalid:
            if size is None:
                rv = np.zeros(alpha.size)
                rv[alpha_ma] = self.rng.dirichlet(alpha=alpha[alpha_ma])
            else:
                rv = np.zeros((size, alpha.size))
                rv[:, alpha_ma] = self.rng.dirichlet(alpha=alpha[alpha_ma], size=size)

            return rv
        else:
            return self.rng.dirichlet(alpha=alpha, size=size)


class DirichletAccumulator(SequenceEncodableAccumulator):
    """Accumulates the Dirichlet sufficient statistics: weighted count, sum of
    log observations, and first/second observation moments."""

    def __init__(self, dim, keys=None):
        """DirichletAccumulator object.

        Args:
                dim (int): Dimension of the probability vectors.
                keys (Optional[str]): Key for merging statistics across accumulators.
        """
        self.dim = dim
        self.sumOfLogs = np.zeros(dim)
        self.sum = np.zeros(dim)
        self.sum2 = np.zeros(dim)
        self.counts = 0
        self.key = keys

    def update(self, x, weight, estimate):
        """Adds a weighted probability-vector observation to the statistics.

        Zero entries contribute nothing to the sum-of-logs statistic.

        Args:
                x: Length-d probability vector.
                weight (float): Observation weight.
                estimate (Optional[DirichletDistribution]): Current estimate. Unused.
        """
        x = np.asarray(x)
        z = x > 0
        if np.all(z):
            self.sumOfLogs += log(x) * weight
            self.sum += weight * x
            self.sum2 += weight * x * x
            self.counts += weight
        else:
            self.sumOfLogs[z] += log(x[z]) * weight
            self.sum += weight * x
            self.sum2 += weight * x * x
            self.counts += weight

    def initialize(self, x, weight, rng):
        """Initializes the accumulator with a weighted observation.

        Args:
                x: Length-d probability vector.
                weight (float): Observation weight.
                rng: Random number generator. Unused.
        """
        self.update(x, weight, None)

    def get_seq_lambda(self):
        return [self.seq_update]

    def seq_update(self, x, weights, estimate):
        """Adds sequence-encoded weighted observations to the statistics.

        Args:
                x: Encoding (log values, values, squared values) from
                        DirichletDistribution.seq_encode.
                weights (np.ndarray): Observation weights.
                estimate (Optional[DirichletDistribution]): Current estimate. Unused.
        """
        self.sumOfLogs += np.dot(weights, x[0])
        self.counts += weights.sum()
        self.sum += np.dot(weights, x[1])
        self.sum2 += np.dot(weights, x[2])

    def seq_initialize(self, x, weights, rng):
        """Initializes the accumulator with sequence-encoded observations.

        Args:
                x: Encoding from DirichletDistribution.seq_encode.
                weights (np.ndarray): Observation weights.
                rng: Random number generator. Unused.
        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat):
        """Adds another accumulator's sufficient statistics to this one.

        Args:
                suff_stat: Tuple (count, sum of logs, sum, sum of squares).

        Returns:
                DirichletAccumulator: This accumulator.
        """
        self.sumOfLogs += suff_stat[1]
        self.sum += suff_stat[2]
        self.sum2 += suff_stat[3]
        self.counts += suff_stat[0]
        return self

    def value(self):
        """Returns the sufficient statistic tuple (count, sum of logs, sum, sum of squares)."""
        return self.counts, self.sumOfLogs, self.sum, self.sum2

    def from_value(self, x):
        """Sets the sufficient statistics from a value() tuple.

        Args:
                x: Tuple (count, sum of logs, sum, sum of squares).
        """
        self.counts = x[0]
        self.sumOfLogs = x[1]
        self.sum = x[2]
        self.sum2 = x[3]

    def key_merge(self, stats_dict):
        """Merges this accumulator into stats_dict under its key (if keyed)."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        """Replaces this accumulator's statistics from stats_dict (if keyed)."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class DirichletEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for DirichletDistribution, with optional
    pseudo-count regularization of the mean-log statistics."""

    def __init__(self, dim, pseudo_count=None, suff_stat=None, delta=1.0e-8, keys=None, use_mpe=False):
        """DirichletEstimator object.

        Args:
                dim (int): Dimension of the probability vectors.
                pseudo_count (Optional[float]): Weight of the regularizing
                        mean-log statistics blended into the data statistics.
                suff_stat (Optional[np.ndarray]): Regularizing mean-log statistics
                        used with pseudo_count (defaults to the symmetric value).
                delta (float): Convergence threshold for the fixed-point solver.
                keys (Optional[str]): Key for merging statistics across accumulators.
                use_mpe (bool): Use minimal polynomial extrapolation to accelerate
                        the fixed-point solve.
        """
        self.dim = dim
        self.pseudo_count = pseudo_count
        self.delta = delta
        self.suff_stat = suff_stat
        self.keys = keys
        self.use_mpe = use_mpe

    def accumulator_factory(self):
        """Returns a factory whose make() creates a DirichletAccumulator."""
        dim = self.dim
        keys = self.keys
        obj = type("", (object,), {"make": lambda self: DirichletAccumulator(dim, keys)})()
        return obj

    def accumulatorFactory(self):
        """Deprecated alias for accumulator_factory()."""
        return self.accumulator_factory()

    def estimate(self, suff_stat, legacy_suff_stat=None):
        """Estimates a DirichletDistribution from sufficient statistics.

        Args:
                suff_stat: Tuple (count, sum of logs, sum, sum of squares) as
                        produced by DirichletAccumulator.value().
                legacy_suff_stat: Deprecated. When given, the call is treated as
                        the legacy form estimate(nobs, suff_stat) and this argument is
                        used as the sufficient statistics.

        Returns:
                DirichletDistribution: Maximum-likelihood estimate, regularized by
                any configured pseudo-count.
        """
        if legacy_suff_stat is not None:
            suff_stat = legacy_suff_stat

        nobs, sum_of_logs, sum_v, sum_v2 = suff_stat
        nobs = float(nobs)
        dim = len(sum_of_logs)
        sum_of_logs = np.asarray(sum_of_logs, dtype=float)
        sum_v = np.asarray(sum_v, dtype=float)
        sum_v2 = np.asarray(sum_v2, dtype=float)

        if self.pseudo_count is not None and self.suff_stat is None:
            pc = max(0.0, float(self.pseudo_count))
            c1 = digamma(one) - digamma(dim)
            c2 = sum_of_logs + c1 * pc
            total = nobs + pc
            if total <= 0.0:
                meanLogP = np.full(dim, c1, dtype=float)
            else:
                meanLogP = c2 / total
            prior_mean = np.ones(dim, dtype=float) / float(dim)
            if nobs > 0.0:
                mean_v = _safe_simplex_mean((sum_v + pc * prior_mean) / total, dim)
                mean_v2 = (sum_v2 + pc * prior_mean * prior_mean) / total
            else:
                mean_v = prior_mean
                mean_v2 = None
            initialEstimate = _initial_dirichlet_alpha(mean_v, mean_v2, meanLogP)

        elif self.pseudo_count is not None and self.suff_stat is not None:
            pc = max(0.0, float(self.pseudo_count))
            prior_mlp = np.asarray(self.suff_stat, dtype=float)
            if prior_mlp.size != dim:
                prior_mlp = np.resize(prior_mlp, dim)
            prior_mlp[~np.isfinite(prior_mlp)] = digamma(one) - digamma(dim)
            c2 = sum_of_logs + prior_mlp * pc
            total = nobs + pc
            meanLogP = prior_mlp if total <= 0.0 else c2 / total
            prior_mean = _mean_from_mean_log(prior_mlp, dim)
            if nobs > 0.0:
                mean_v = _safe_simplex_mean((sum_v + pc * prior_mean) / total, dim)
                mean_v2 = (sum_v2 + pc * prior_mean * prior_mean) / total
            else:
                mean_v = prior_mean
                mean_v2 = None
            initialEstimate = _initial_dirichlet_alpha(mean_v, mean_v2, meanLogP)

        else:
            if nobs <= 0.0:
                return DirichletDistribution(np.ones(dim, dtype=float))

            sum_v = sum_v / nobs
            sum_v2 = sum_v2 / nobs
            sum_v = _safe_simplex_mean(sum_v, dim)

            """
			#initialConst = (sum_v[0]-sum_v2[0])/(sum_v2[0]-sum_v[0]*sum_v[0])
			initialConst1 = (sum_v - sum_v2).mean()
			initialConst2 = (sum_v2 - sum_v*sum_v).mean()

			if initialConst2 > 0 and initialConst1 > 0:
				initialEstimate = (initialConst1/initialConst2)*sum_v
			else:
				initialEstimate = sum_of_logs * (dim / sum(sum_of_logs))

			#initialEstimate = sum_of_logs*(dim/sum(sum_of_logs))

			"""
            initialEstimate = _initial_dirichlet_alpha(sum_v, sum_v2)

            meanLogP = sum_of_logs / nobs

        if not np.all(np.isfinite(meanLogP)):
            meanLogP = np.where(np.isfinite(meanLogP), meanLogP, digamma(one) - digamma(dim))

        if nobs <= 1.0 and self.pseudo_count is None:
            return DirichletDistribution(initialEstimate)

        else:
            if self.use_mpe:
                alpha, its_cnt = find_alpha(np.asarray(initialEstimate), meanLogP, self.delta)
            else:
                alpha, its_cnt = dirichlet_param_solve(np.asarray(initialEstimate), meanLogP, self.delta)

            return DirichletDistribution(alpha)
