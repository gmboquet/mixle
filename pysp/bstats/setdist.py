"""Bernoulli set distribution: independent Bernoulli inclusion of each item
from a fixed catalog, with a Beta (or Beta-mixture) prior on the inclusion
probabilities.

Data type: Set[Any] (any iterable of hashable items drawn from the keys of
the probability map). Each item k is included independently with probability
p_k, so the log-density of an observed set x is

        log f(x) = sum_{k in x} log p_k + sum_{k not in x} log(1 - p_k).

Probabilities close to one are stored as negative values v with p = 1 + v so
that log1p can be used for precision. Defines the BernoulliSetDistribution,
BernoulliSetSampler, BernoulliSetAccumulator, BernoulliSetAccumulatorFactory,
and BernoulliSetEstimator classes for use with pysparkplug.
"""

from collections import defaultdict

import numpy as np
from numpy.random import RandomState

from pysp.arithmetic import *
from pysp.bstats.beta import BetaDistribution
from pysp.bstats.mixture import MixtureDistribution
from pysp.bstats.pdist import ParameterEstimator, ProbabilityDistribution, SequenceEncodableAccumulator
from pysp.utils.special import gammaln

default_prior = BetaDistribution(1, 1)


class BernoulliSetDistribution(ProbabilityDistribution):
    """Distribution over subsets of a catalog with independent per-item
    inclusion probabilities."""

    def __init__(self, pmap, name=None, prior=None):
        """Create a Bernoulli set distribution.

        Args:
                pmap: Map from item to inclusion probability (values in
                        (-1, 0) encode p = 1 + v for precision near one).
                name (Optional[str]): Name of the distribution.
                prior (Optional[ProbabilityDistribution]): Prior on the
                        inclusion probabilities.
        """

        self.name = name
        self.prior = prior
        self.pmap = pmap
        self.log_pmap = {k: np.log1p(v) if v < 0 else np.log(v) for k, v in pmap.items()}
        self.log_nmap = {k: np.log(-v) if v < 0 else np.log1p(-v) for k, v in pmap.items()}
        self.nmap_sum = sum([u for u in self.log_nmap.values() if u != -np.inf])

    def __str__(self):
        return "BernoulliSetDistribution(%s, name=%s, prior=%s)" % (str(self.pmap), str(self.name), str(self.prior))

    def get_parameters(self):
        """Return the inclusion probability map."""
        return self.pmap

    def set_parameters(self, params):
        """Set the inclusion probability map.

        Args:
                params: Map from item to inclusion probability.
        """
        self.pmap = params

    def get_prior(self):
        """Return the prior on the inclusion probabilities."""
        return self.prior

    def set_prior(self, prior):
        """Set the prior on the inclusion probabilities.

        Args:
                prior (ProbabilityDistribution): New prior distribution.
        """
        self.prior = prior

    def density(self, x):
        """Density at the observed set x.

        Args:
                x: Iterable of items in the catalog.

        Returns:
                Density (float) at x.
        """
        return exp(self.log_density(x))

    def log_density(self, x):
        """Log-density of the observed set x.

        Args:
                x: Iterable of items in the catalog.

        Returns:
                Sum of log p_k over included items plus log(1 - p_k) over
                excluded items.
        """
        rv = self.nmap_sum
        for u in x:
            rv += self.log_pmap[u] - self.log_nmap[u]
        return rv

    def seq_log_density(self, x):
        """Vectorized log-density at sequence-encoded input x.

        Args:
                x: Encoded data from seq_encode().

        Returns:
                Numpy array of log-densities, one entry per observed set.
        """

        sz, idx, val_map_inv, xs = x

        log_prob_loc = np.asarray([self.log_pmap[u] - self.log_nmap[u] for u in val_map_inv])

        rv = np.bincount(idx, weights=log_prob_loc[xs], minlength=sz)
        rv += self.nmap_sum

        return rv

    def seq_encode(self, x):
        """Encode a sequence of observed sets for vectorized evaluation.

        Args:
                x: List of iterables of items.

        Returns:
                Tuple (number of sets, set index per item, unique item array,
                item index array).
        """

        idx = []
        xflat = []

        for i in range(len(x)):
            m = len(x[i])
            idx.extend([i] * m)
            xflat.extend(x[i])

        val_map_inv, xs = np.unique(xflat, return_inverse=True)
        idx = np.asarray(idx, dtype=int)

        return len(x), idx, val_map_inv, xs

    def sampler(self, seed=None):
        """Return a BernoulliSetSampler for this distribution.

        Args:
                seed (Optional[int]): Seed for the random number generator.
        """
        return BernoulliSetSampler(self, seed)

    def estimator(self):
        """Return a BernoulliSetEstimator."""
        return BernoulliSetEstimator()


class BernoulliSetSampler:
    """Draws subsets from a BernoulliSetDistribution."""

    def __init__(self, dist, seed=None):
        """Create a sampler for a BernoulliSetDistribution.

        Args:
                dist (BernoulliSetDistribution): Distribution to sample from.
                seed (Optional[int]): Seed for the random number generator.
        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size=None):
        """Draw size subsets (or one subset when size is None).

        Args:
                size (Optional[int]): Number of subsets to draw.

        Returns:
                A list of items when size is None, otherwise a list of such
                lists.
        """

        if size is not None:
            retval = [[] for i in range(size)]
            for k, v in self.dist.pmap.items():
                for i in np.flatnonzero(self.rng.rand(size) <= (v % 1)):
                    retval[i].append(k)
            return retval

        else:
            retval = []
            for k, v in self.dist.pmap.items():
                if self.rng.rand() <= (v % 1):
                    retval.append(k)
            return retval


class BernoulliSetAccumulator(SequenceEncodableAccumulator):
    """Accumulates per-item inclusion counts and the total weighted set
    count for Bernoulli set estimation."""

    def __init__(self):
        """Create a Bernoulli set accumulator."""
        self.pmap = defaultdict(float)
        self.tot_sum = 0.0

    def update(self, x, weight, estimate):
        """Accumulate one weighted set observation.

        Args:
                x: Iterable of included items.
                weight (float): Observation weight.
                estimate: Unused (kept for protocol consistency).
        """
        for u in x:
            self.pmap[u] += weight
        self.tot_sum += weight

    def initialize(self, x, weight, rng):
        """Initialize with one weighted observation (delegates to update)."""
        self.update(x, weight, None)

    def seq_update(self, x, weights, estimate):
        """Vectorized update from sequence-encoded data.

        Args:
                x: Encoded data from BernoulliSetDistribution.seq_encode().
                weights (np.ndarray): Per-set observation weights.
                estimate: Unused (kept for protocol consistency).
        """

        sz, idx, val_map_inv, xs = x
        agg_cnt = np.bincount(xs, weights[idx])

        for i, v in enumerate(agg_cnt):
            self.pmap[val_map_inv[i]] += v

        self.tot_sum += weights.sum()

    def combine(self, suff_stat):
        """Merge another accumulator's value() into this one.

        Args:
                suff_stat: Tuple (inclusion count map, total weight).

        Returns:
                This accumulator.
        """
        for k, v in suff_stat[0].items():
            self.pmap[k] += v
        self.tot_sum += suff_stat[1]
        return self

    def value(self):
        """Return (inclusion count map, total weight)."""
        return self.pmap, self.tot_sum

    def from_value(self, x):
        """Set this accumulator's state from a value() tuple.

        Args:
                x: Tuple (inclusion count map, total weight).

        Returns:
                This accumulator.
        """
        self.pmap = x[0]
        self.tot_sum = x[1]
        return self


class BernoulliSetAccumulatorFactory:
    """Factory for creating BernoulliSetAccumulator objects."""

    def __init__(self, keys=(None,)):
        """Create a Bernoulli set accumulator factory.

        Args:
                keys: Key tuple (kept for protocol consistency).
        """
        self.keys = keys

    def make(self):
        """Return a new BernoulliSetAccumulator."""
        return BernoulliSetAccumulator()


class BernoulliSetEstimator(ParameterEstimator):
    """Estimates a BernoulliSetDistribution from accumulated inclusion
    counts, using Beta (or Beta-mixture) posterior modes when a conjugate
    prior is set."""

    def __init__(self, name=None, prior=default_prior, keys=(None,)):
        """Create a Bernoulli set estimator.

        Args:
                name (Optional[str]): Name of the estimated distribution.
                prior: Prior on the inclusion probabilities (BetaDistribution
                        or a MixtureDistribution of BetaDistribution components).
                keys: Key tuple for sharing statistics.
        """
        self.name = name
        self.prior = prior
        self.keys = keys

    def accumulator_factory(self):
        """Return a BernoulliSetAccumulatorFactory for this estimator."""
        return BernoulliSetAccumulatorFactory(self.keys)

    def get_prior(self):
        """Return the prior on the inclusion probabilities."""
        return self.prior

    def set_prior(self, prior):
        """Set the prior on the inclusion probabilities.

        Args:
                prior (ProbabilityDistribution): New prior distribution.
        """
        self.prior = prior

    def estimate(self, suff_stat):
        """Estimate a BernoulliSetDistribution from sufficient statistics.

        Args:
                suff_stat: Tuple (inclusion count map, total weight) as
                        returned by BernoulliSetAccumulator.value().

        Returns:
                BernoulliSetDistribution with posterior-mode inclusion
                probabilities (under a Beta or Beta-mixture prior) or relative
                frequencies, carrying this estimator's name and prior.
        """

        if isinstance(self.prior, BetaDistribution):
            pmap = bernoulli_beta_posterior_mode(suff_stat[0], suff_stat[1], self.prior.get_parameters())

        elif isinstance(self.prior, MixtureDistribution) and all(
            [isinstance(u, BetaDistribution) for u in self.prior.components]
        ):
            beta_params = np.asarray([[u.a, u.b] for u in self.prior.components])
            pmap = bernoulli_betamix_posterior_mode(suff_stat[0], suff_stat[1], self.prior.w, beta_params)

        else:
            pmap = dict()
            for k, v in suff_stat[0].items():
                if v * 2 > suff_stat[1]:
                    pmap[k] = -(suff_stat[1] - v) / suff_stat[1]
                else:
                    pmap[k] = v / suff_stat[1]

        return BernoulliSetDistribution(pmap, name=self.name, prior=self.prior)


def bernoulli_beta_posterior_mode(obs_cnt, tot_cnt, beta_params):
    """Per-item Beta posterior-mode inclusion probabilities.

    Args:
            obs_cnt: Map from item to weighted inclusion count.
            tot_cnt (float): Total weighted set count.
            beta_params: Tuple (a, b) of Beta prior parameters.

    Returns:
            Map from item to posterior-mode probability (negative encoding for
            probabilities above one half).
    """

    pmap = dict()
    for k, v in obs_cnt.items():
        a = (beta_params[0] - 1) + v
        b = (beta_params[1] - 1) - v + tot_cnt

        if a > 0 and b > 0 and a > b:
            p = -b / (a + b)
        elif a > 0 and b > 0 and b > a:
            p = (a - 1) / (a + b - 2)
        elif a == 0 and b == 0:
            p = 0.5
        elif b > a:
            p = 0.0
        else:
            p = 1.0

        pmap[k] = p

    return pmap


def bernoulli_betamix_posterior_mode(obs_cnt, tot_cnt, w, beta_params):
    """Per-item posterior-mode inclusion probabilities under a Beta-mixture
    prior (the most likely mixture component is selected per item).

    Args:
            obs_cnt: Map from item to weighted inclusion count.
            tot_cnt (float): Total weighted set count.
            w: Mixture weights of the Beta components.
            beta_params (np.ndarray): Array of shape (K, 2) of Beta parameters.

    Returns:
            Map from item to posterior-mode probability (negative encoding for
            probabilities above one half).
    """

    dc = -gammaln(beta_params.sum(axis=1) + tot_cnt)
    lc = -gammaln(beta_params).sum(axis=1) + gammaln(beta_params.sum(axis=1)) + dc
    log_w = np.log(w)

    pmap = dict()
    for k, v in obs_cnt.items():
        ll = log_w + gammaln(beta_params[:, 0] + v) + gammaln(beta_params[:, 1] + (tot_cnt - v)) + lc
        bidx = ll.argmax()

        a = (beta_params[bidx, 0] - 1) + v
        b = (beta_params[bidx, 1] - 1) - v + tot_cnt

        if a > 0 and b > 0 and a > b:
            p = -b / (a + b)
        elif a > 0 and b > 0 and b > a:
            p = (a - 1) / (a + b - 2)
        elif a == 0 and b == 0:
            p = 0.5
        elif b > a:
            p = 0.0
        else:
            p = 1.0

        pmap[k] = p

    return pmap
