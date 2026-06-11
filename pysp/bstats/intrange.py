"""Categorical distribution on a contiguous integer range with a Dirichlet
prior on the category probabilities.

Data type: (int): An IntegerCategoricalDistribution with probability vector
p = (p_0, ..., p_{n-1}) starting at min_index has log-density

    log f(x) = log p_{x - min_index} - log(1 + default_value)

for x in [min_index, min_index + n - 1], and log(default_value) -
log(1 + default_value) otherwise, where default_value is the unnormalized
probability assigned to out-of-range values. Defines the
IntegerCategoricalDistribution, IntegerCategoricalSampler,
IntegerCategoricalAccumulator, IntegerCategoricalAccumulatorFactory, and
IntegerCategoricalEstimator classes for use with pysp.bstats.

Conjugate prior: DirichletDistribution (or SymmetricDirichletDistribution) on
the probability vector. Estimation is MAP (counts + alpha - 1, clamped at the
simplex boundary, posterior mean when degenerate) and the posterior Dirichlet
is carried forward as the new prior. expected_log_density evaluates the
variational Bayes expectation E_q[log p(x)] via digamma terms.
"""
from typing import Optional, List, Union, Tuple
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.bstats.pdist import ProbabilityDistribution, SequenceEncodableAccumulator, ParameterEstimator
from pysp.bstats.symdirichlet import SymmetricDirichletDistribution
from pysp.bstats.dirichlet import DirichletDistribution
import numpy as np
from scipy.special import gammaln, digamma
import scipy.sparse as sp

default_prior = DirichletDistribution(1.0 + 1.0e-12)

class IntegerCategoricalDistribution(ProbabilityDistribution):
    """Categorical distribution on the integers [min_index, min_index + n - 1],
    optionally carrying a Dirichlet conjugate prior on the probabilities."""

    def __init__(self, prob_vec: Union[np.ndarray, List[float], sp.spmatrix, int] = None,
                 default_value: Union[float, np.ndarray, List[float]] = 0.0, min_index: int = 0,
                 name: Optional[str] = None, prior: ProbabilityDistribution = default_prior,
                 min_val: Optional[int] = None, p_vec=None):
        """Accepts both calling conventions:

            IntegerCategoricalDistribution(min_val, p_vec)   # pysp.stats order
            IntegerCategoricalDistribution(p_vec, min_index=...)

        plus the pysp.stats keyword names min_val/p_vec as aliases.

        Args:
            prob_vec: Probability vector over the supported range (or the
                integer min value in the pysp.stats calling order).
            default_value: Unnormalized probability assigned to out-of-range
                values (or the probability vector in the pysp.stats order).
            min_index (int): Smallest supported integer value.
            name (Optional[str]): Name of the distribution.
            prior (ProbabilityDistribution): Prior on the probability vector; a
                (symmetric) Dirichlet enables the conjugate machinery.
            min_val (Optional[int]): Alias for min_index (pysp.stats name).
            p_vec: Alias for prob_vec (pysp.stats name).
        """
        if p_vec is not None:
            prob_vec = p_vec
        if min_val is not None:
            min_index = int(min_val)
        if prob_vec is not None and np.ndim(prob_vec) == 0 and np.ndim(default_value) > 0:
            # pysp.stats argument order: (min_val, p_vec)
            prob_vec, min_index, default_value = default_value, int(prob_vec), 0.0

        self.min_index = min_index
        self.name = name
        self.set_parameters(prob_vec)#, default_value, min_index))
        self.set_prior(prior)


    def __str__(self):
        pstr = ','.join(map(str, self.prob_vec))
        astr = 'default_value=%s, min_index=%s, prior=%s'%(str(self.default_value), str(self.min_index), str(self.prior))
        return 'IntegerRangeDistribution([%s], %s)' % (pstr, astr)

    def get_parameters(self):
        """Returns the probability vector."""
        return self.prob_vec#, self.default_value, self.min_index

    def set_parameters(self, params):
        """Sets the probability vector and refreshes the cached log terms.

        Args:
            params: Probability vector over the supported range.
        """

        with np.errstate(divide='ignore'):

            self.prob_vec      = params
            #self.default_value = params[1]
            #self.min_index     = int(params[2])
            self.default_value = 0.0
            #self.min_index = 0
            self.max_index     = int(self.min_index + len(self.prob_vec) - 1)

            self.num_vals      = len(self.prob_vec)
            self.log_prob_vec  = np.log(self.prob_vec)
            self.log_default_value = np.log(self.default_value)
            self.log_const = np.log1p(self.default_value)

    def set_prior(self, prior):
        """Set the prior on the probability vector and cache digamma
        expectations when the prior is a (symmetric) Dirichlet.

        Args:
            prior (ProbabilityDistribution): New prior distribution.
        """
        self.prior = prior

        if isinstance(self.prior, DirichletDistribution):
            cpp = prior.get_parameters()
            if np.ndim(cpp) == 0:
                cpp = np.ones(self.num_vals)*cpp
            self.conj_prior_params = cpp
            self.expected_nparams = digamma(self.conj_prior_params) - digamma(np.sum(self.conj_prior_params))
        elif isinstance(self.prior, SymmetricDirichletDistribution):
            self.conj_prior_params = np.ones(self.num_vals)*prior.get_parameters()
            self.expected_nparams = digamma(self.conj_prior_params) - digamma(np.sum(self.conj_prior_params))
        else:
            self.conj_prior_params = None
            self.expected_nparams = None

    def get_prior(self):
        """Returns the prior on the probability vector."""
        return self.prior

    def get_data_type(self):
        """Returns the observation data type (int)."""
        return int

    def entropy(self):
        """Returns the entropy -sum_k p_k log p_k in nats."""
        p = self.prob_vec
        g = p > 0
        log_p = self.log_prob_vec

        return -np.dot(p[g], log_p[g])

    def cross_entropy(self, dist: ProbabilityDistribution):
        """Cross entropy -E_self[log dist(x)] over this distribution's support.

        Args:
            dist (ProbabilityDistribution): Distribution evaluated under this
                one.

        Returns:
            float: Cross entropy value in nats.
        """
        if isinstance(dist, IntegerCategoricalDistribution):
            p = self.prob_vec
            g = p > 0
            gg = np.flatnonzero(g) + (self.min_index - dist.min_index)
            log_p = dist.log_prob_vec
            return -np.dot(p[g], log_p[gg])
        else:
            rv = 0
            for x,p in enumerate(self.prob_vec):
                if p > 0:
                    rv += dist.log_density(x+self.min_index) * p
            return -rv

    def moment(self, p, o=0):
        """Returns the p-th moment E[(X - o)^p] over the supported range.

        Args:
            p: Moment order.
            o: Offset subtracted from the values (default 0).
        """
        return np.dot(np.power(np.arange(self.min_index, self.max_index+1)-o, p), self.prob_vec)

    def log_density(self, x: int) -> float:
        """Log-density at observation x.

        Args:
            x (int): Observed value.

        Returns:
            log p_{x - min_index} for in-range values, log(default_value)
            otherwise, both normalized by log(1 + default_value).
        """

        if (x < self.min_index) or (x > self.max_index):
            return self.log_default_value
        else:
            return self.log_prob_vec[x - self.min_index] - self.log_const

    def expected_log_density(self, x: int) -> float:
        """Prior-expected log-density E_q[log p(x)] at observation x via the
        cached digamma terms (requires a conjugate prior).

        Args:
            x (int): Observed value.

        Returns:
            Expected log-density (float) at x.
        """

        # E[ params ]*x
        #e_x = digamma(self.conj_prior_params[idx]) - digamma(np.sum(self.conj_prior_params))
        # E[ A(params) ] = 0
        # E[ ln(B(x)) ] = 0

        if (x < self.min_index) or (x > self.max_index):
            return self.log_default_value - self.log_const
        else:
            idx = int(x - self.min_index)
            return self.expected_nparams[idx] - self.log_const

    def seq_log_density(self, x):
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x: Encoded data from seq_encode().

        Returns:
            Numpy array of log-densities, one entry per observation.
        """

        v  = x - self.min_index
        u  = np.bitwise_and(v >= 0, v < self.num_vals)
        rv = np.zeros(len(x))
        rv.fill(self.log_default_value)
        rv[u] = self.log_prob_vec[v[u]]
        rv -= self.log_const

        return rv

    def seq_expected_log_density(self, x):
        """Vectorized expected log-density at sequence-encoded input x.

        Args:
            x: Encoded data from seq_encode().

        Returns:
            Numpy array of expected log-densities, one entry per observation.
        """
        idx = x - self.min_index
        in_range = np.bitwise_and(idx >= 0, idx < self.num_vals)
        rv = np.zeros(len(x))
        rv.fill(self.log_default_value)
        rv[in_range] = self.expected_nparams[idx[in_range]]
        rv -= self.log_const
        return rv

    def seq_encode(self, x):
        """Encode a sequence of integer observations for vectorized evaluation.

        Args:
            x: Iterable of observed integer values.

        Returns:
            Numpy integer array of the observations.
        """
        return np.asarray(x, dtype=int)

    def sampler(self, seed: Optional[int] = None):
        """Return an IntegerCategoricalSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.
        """
        return IntegerCategoricalSampler(self, seed)

    def estimator(self):
        """Return an IntegerCategoricalEstimator matching this distribution."""
        return IntegerCategoricalEstimator(name=self.name, prior=self.prior)



class IntegerCategoricalSampler(object):
    """Draws integer observations from an IntegerCategoricalDistribution."""

    def __init__(self, dist: IntegerCategoricalDistribution, seed: Optional[int] = None):
        """IntegerCategoricalSampler object.

        Args:
            dist (IntegerCategoricalDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.
        """
        self.rng  = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw size samples (or one sample when size is None).

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            A single integer when size is None, otherwise a list of integers.
        """

        if size is None:
            return self.rng.choice(range(self.dist.min_index, self.dist.max_index+1), p=self.dist.prob_vec)
        else:
            return self.rng.choice(range(self.dist.min_index, self.dist.max_index + 1), p=self.dist.prob_vec, size=size).tolist()


class IntegerCategoricalAccumulator(SequenceEncodableAccumulator):
    """Accumulates per-value counts on a (growable) integer range for
    categorical estimation."""

    def __init__(self, min_val: int, max_val: int, keys: Tuple[str,]):
        """IntegerCategoricalAccumulator object.

        Args:
            min_val (Optional[int]): Smallest supported value (grown on demand
                when None).
            max_val (Optional[int]): Largest supported value (grown on demand
                when None).
            keys: Tuple whose first entry keys this accumulator for statistic
                sharing.
        """

        self.minVal = min_val
        self.maxVal = max_val

        if min_val is not None and max_val is not None:
            self.countVec = vec.zeros(max_val-min_val+1)
        else:
            self.countVec = None

        self.key = keys[0]

    def update(self, x, weight, estimate):
        """Accumulate one weighted observation, growing the range as needed.

        Args:
            x (int): Observed value.
            weight (float): Observation weight.
            estimate: Unused (kept for protocol consistency).
        """

        if self.countVec is None:
            self.minVal   = x
            self.maxVal   = x
            self.countVec = vec.make([weight])

        elif self.maxVal < x:
            tempVec = self.countVec
            self.maxVal   = x
            self.countVec = vec.zeros(self.maxVal - self.minVal + 1)
            self.countVec[:len(tempVec)] = tempVec
            self.countVec[x-self.minVal] += weight
        elif self.minVal > x:
            tempVec  = self.countVec
            tempDiff = self.minVal - x
            self.minVal   = x
            self.countVec = vec.zeros(self.maxVal - self.minVal + 1)
            self.countVec[tempDiff:] = tempVec
            self.countVec[x-self.minVal] += weight
        else:
            self.countVec[x-self.minVal] += weight


    def seq_update(self, x, weights, estimate):
        """Vectorized update from sequence-encoded data.

        Args:
            x: Encoded data from IntegerCategoricalDistribution.seq_encode().
            weights (np.ndarray): Observation weights.
            estimate: Unused (kept for protocol consistency).
        """

        min_x = x.min()
        max_x = x.max()

        loc_cnt = np.bincount(x-min_x, weights=weights)

        if self.countVec is None:
            self.countVec = np.zeros(max_x-min_x+1)
            self.minVal = min_x
            self.maxVal = max_x

        if self.minVal > min_x or self.maxVal < max_x:
            prev_min    = self.minVal
            self.minVal = min(min_x, self.minVal)
            self.maxVal = max(max_x, self.maxVal)
            temp        = self.countVec
            prev_diff   = prev_min - self.minVal
            self.countVec = np.zeros(self.maxVal - self.minVal + 1)
            self.countVec[prev_diff:(prev_diff + len(temp))] = temp

        min_diff = min_x - self.minVal
        self.countVec[min_diff:(min_diff+len(loc_cnt))] += loc_cnt

    def seq_initialize(self, x, weights, rng):
        """Vectorized initialization (delegates to seq_update).

        Args:
            x: Encoded data from IntegerCategoricalDistribution.seq_encode().
            weights (np.ndarray): Observation weights.
            rng: Unused (kept for protocol consistency).
        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat):
        """Merge another accumulator's value() into this one.

        Args:
            suff_stat: Tuple (min value, count vector).

        Returns:
            This accumulator.
        """

        if self.countVec is None and suff_stat[1] is not None:
            self.minVal   = suff_stat[0]
            self.maxVal   = suff_stat[0] + len(suff_stat[1]) - 1
            self.countVec = suff_stat[1]

        elif self.countVec is not None and suff_stat[1] is not None:

            if self.minVal == suff_stat[0] and len(self.countVec) == len(suff_stat[1]):
                self.countVec += suff_stat[1]

            else:
                minVal = min(self.minVal, suff_stat[0])
                maxVal = max(self.maxVal, suff_stat[0] + len(suff_stat[1]) - 1)

                countVec = vec.zeros(maxVal-minVal+1)

                i0 = self.minVal - minVal
                i1 = self.maxVal - minVal + 1
                countVec[i0:i1] = self.countVec

                i0 = suff_stat[0] - minVal
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - minVal + 1
                countVec[i0:i1] += suff_stat[1]

                self.minVal   = minVal
                self.maxVal   = maxVal
                self.countVec = countVec

        return self

    def value(self):
        """Return (min value, count vector)."""
        return self.minVal, self.countVec

    def from_value(self, x):
        """Set this accumulator's state from a value() tuple.

        Args:
            x: Tuple (min value, count vector).
        """
        self.minVal   = x[0]
        self.maxVal   = x[0] + len(x[1]) - 1
        self.countVec = x[1]


    def key_merge(self, stats_dict):
        """Merge this accumulator into stats_dict under its key (if keyed)."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict):
        """Replace this accumulator's statistics from stats_dict (if keyed)."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())


class IntegerCategoricalAccumulatorFactory(object):
    """Factory for creating IntegerCategoricalAccumulator objects."""

    def __init__(self, min_val, max_val, keys: Tuple[Optional[str],] = (None,)):
        """IntegerCategoricalAccumulatorFactory object.

        Args:
            min_val (Optional[int]): Smallest supported value (grown on demand
                when None).
            max_val (Optional[int]): Largest supported value (grown on demand
                when None).
            keys: Tuple whose first entry keys the accumulators for statistic
                sharing.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.keys = keys

    def make(self):
        """Returns a new IntegerCategoricalAccumulator."""
        return IntegerCategoricalAccumulator(self.min_val, self.max_val, self.keys)


class IntegerCategoricalEstimator(object):
    """Estimates an IntegerCategoricalDistribution from accumulated counts,
    using Dirichlet MAP probabilities when a conjugate prior is set."""

    def __init__(self, min_index: Optional[int] = None, max_index: Optional[int] = None, default_value: float = 0.0, name: Optional[str] = None, prior: ProbabilityDistribution = default_prior, keys: Tuple[Optional[str], ] = (None,),
                 min_val: Optional[int] = None, max_val: Optional[int] = None):
        """IntegerCategoricalEstimator object.

        Args:
            min_index (Optional[int]): Smallest supported value (grown on
                demand when None).
            max_index (Optional[int]): Largest supported value (grown on
                demand when None).
            default_value (float): Unnormalized probability for out-of-range
                values in the estimated distribution.
            name (Optional[str]): Name of the estimated distribution.
            prior (ProbabilityDistribution): Prior on the probability vector; a
                (symmetric) Dirichlet enables conjugate MAP estimation.
            keys: Tuple whose first entry keys the accumulators for statistic
                sharing.
            min_val (Optional[int]): Alias for min_index (pysp.stats name).
            max_val (Optional[int]): Alias for max_index (pysp.stats name).
        """
        # min_val/max_val accepted as aliases to match pysp.stats.IntegerCategoricalEstimator
        if min_val is not None:
            min_index = min_val
        if max_val is not None:
            max_index = max_val

        self.minVal        = min_index
        self.maxVal        = max_index
        self.default_value = default_value
        self.keys          = keys
        self.name          = name

        self.set_prior(prior)

    def get_prior(self):
        """Returns the prior on the probability vector."""
        return self.prior

    def set_prior(self, prior):
        """Set the prior on the probability vector.

        Args:
            prior (ProbabilityDistribution): New prior distribution; a
                (symmetric) Dirichlet enables conjugate MAP estimation.
        """
        self.prior = prior

        if isinstance(self.prior, (DirichletDistribution, SymmetricDirichletDistribution)):
            self.has_conj_prior = True
        else:
            self.has_conj_prior = False


    def accumulator_factory(self):
        """Returns an IntegerCategoricalAccumulatorFactory for this estimator."""
        return IntegerCategoricalAccumulatorFactory(self.minVal, self.maxVal, self.keys)

    def accumulatorFactory(self):
        """Deprecated alias for accumulator_factory()."""
        return self.accumulator_factory()

    def estimate(self, suff_stat):
        """Estimate an IntegerCategoricalDistribution from sufficient statistics.

        With a (symmetric) Dirichlet prior this returns the MAP probabilities
        (counts + alpha - 1, clamped at the simplex boundary, posterior mean
        when the MAP is degenerate) carrying the posterior Dirichlet as the new
        prior; otherwise it returns the relative frequencies.

        Args:
            suff_stat: Tuple (min value, count vector) as returned by
                IntegerCategoricalAccumulator.value().

        Returns:
            IntegerCategoricalDistribution estimate.
        """

        if self.has_conj_prior:

            min_val, count_vec = suff_stat
            alpha0 = self.prior.get_parameters()
            if np.ndim(alpha0) == 0:
                alpha0 = np.ones(len(count_vec)) * alpha0

            posterior_params = count_vec + alpha0

            # Dirichlet MAP sits on the boundary when alpha_k + n_k < 1
            num = np.maximum(count_vec + (alpha0 - 1), 0.0)
            norm_const = np.sum(num)

            if norm_const > 0:
                prob_vec = num / norm_const
            else:
                # fall back to the posterior mean when the MAP is degenerate
                prob_vec = posterior_params / np.sum(posterior_params)

            hyper_posterior = DirichletDistribution(posterior_params)

            return IntegerCategoricalDistribution(prob_vec, min_index=min_val, default_value=self.default_value, name=self.name, prior=hyper_posterior)

        else:

            return IntegerCategoricalDistribution(suff_stat[0], suff_stat[1] / (suff_stat[1].sum()), name=self.name, prior=self.prior)
