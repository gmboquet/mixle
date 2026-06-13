"""Finite mixture distribution with a Dirichlet prior on the mixture weights.

A mixture models data drawn from one of K component distributions chosen with
probabilities w = (w_1, ..., w_K). The data type accepted is whatever the
component distributions accept (all components must share a common data type),
and the density is

    f(x) = sum_k w_k * f_k(x).

Defines the MixtureDistribution, MixtureSampler, MixtureEstimatorAccumulator,
MixtureEstimatorAccumulatorFactory, and MixtureEstimator classes for use with
pysparkplug. With a (symmetric) Dirichlet prior on the weights, estimation is
MAP (counts + alpha - 1, clamped at the simplex boundary) and the posterior
Dirichlet is carried forward as the new prior. expected_log_density uses the
digamma expectations E[ln w_k] = psi(alpha_k) - psi(sum alpha).

Use ``mixture_prior(weight_prior, component_priors)`` when you want to pass a
single joint prior containing both the weight prior and one conjugate prior per
component.
"""
from typing import TypeVar, Optional, Generic, List, Mapping

from pysp.arithmetic import *
from pysp.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from numpy.random import RandomState
from pysp.bstats.composite import CompositeDistribution
from pysp.bstats.dirichlet import DirichletDistribution
from pysp.bstats.symdirichlet import SymmetricDirichletDistribution
import numpy as np
import pysp.utils.vector as vec
from scipy.special import gammaln, digamma


default_prior = SymmetricDirichletDistribution(1)


def mixture_prior(weight_prior, component_priors):
    """Build the joint prior form used by bstats mixtures.

    Args:
        weight_prior: Prior on the mixture weights, typically a
            DirichletDistribution or SymmetricDirichletDistribution.
        component_priors: Sequence of one conjugate prior per component.

    Returns:
        CompositeDistribution containing ``(weight_prior,
        CompositeDistribution(component_priors))``.
    """
    return CompositeDistribution((weight_prior, CompositeDistribution(tuple(component_priors))))


def _default_weight_prior(num_components: int):
    return DirichletDistribution(np.ones(num_components))


def _component_prior_tuple(component_priors, num_components: int):
    if component_priors is None:
        return None
    if isinstance(component_priors, CompositeDistribution):
        rv = tuple(component_priors.dists)
    elif isinstance(component_priors, (list, tuple)):
        rv = tuple(component_priors)
    elif num_components == 1:
        rv = (component_priors,)
    else:
        raise TypeError('mixture component priors must be a sequence or CompositeDistribution.')
    if len(rv) != num_components:
        raise ValueError('expected %d component priors, got %d.' % (num_components, len(rv)))
    return rv


def _split_mixture_prior(prior, num_components: int):
    if prior is None:
        return None, None
    if isinstance(prior, CompositeDistribution) and len(prior.dists) == 2 and \
            isinstance(prior.dists[1], CompositeDistribution):
        return prior.dists[0], _component_prior_tuple(prior.dists[1], num_components)
    if isinstance(prior, Mapping) and ('weights' in prior or 'weight_prior' in prior or
                                      'components' in prior or 'component_priors' in prior):
        weight_prior = prior.get('weights', prior.get('weight_prior'))
        component_priors = prior.get('components', prior.get('component_priors'))
        if weight_prior is None:
            weight_prior = _default_weight_prior(num_components)
        return weight_prior, _component_prior_tuple(component_priors, num_components)
    if isinstance(prior, (list, tuple)) and len(prior) == 2 and \
            isinstance(prior[1], (list, tuple, CompositeDistribution)):
        return prior[0], _component_prior_tuple(prior[1], num_components)
    return prior, None


def _dirichlet_expectations(prior, num_components: int):
    if isinstance(prior, DirichletDistribution):
        alpha = prior.get_parameters()
        return alpha, digamma(alpha) - digamma(np.sum(alpha))
    if isinstance(prior, SymmetricDirichletDistribution):
        alpha = np.ones(num_components) * prior.get_parameters()
        return alpha, digamma(alpha) - digamma(np.sum(alpha))
    return None, None

class MixtureDistribution(ProbabilityDistribution):
    """Finite mixture of component distributions with mixing weights w."""

    def __init__(self, components, w, name=None, prior: Optional[ProbabilityDistribution] = None):
        """Create a mixture distribution.

        Args:
            components: Sequence of component ProbabilityDistribution objects
                sharing a common data type.
            w: Mixture weights (length-K array summing to one).
            name (Optional[str]): Name of the distribution.
            prior (Optional[ProbabilityDistribution]): Prior on the mixture
                weights. Defaults to a flat Dirichlet.
        """
        self.set_name(name)

        self.components = components
        self.num_components = len(components)
        self.w = np.asarray(w)
        self.zw = (self.w == 0.0)
        self.log_w = np.log(self.w + self.zw)
        self.log_w[self.zw] = -np.inf
        self.prior = None

        #self.parents = []
        #for d in self.components:
        #    d.add_parent(self)

        self.set_prior(_default_weight_prior(self.num_components) if prior is None else prior)

    def __str__(self):
        return 'MixtureDistribution([%s], [%s], name=%s, prior=%s)' % (','.join([str(u) for u in self.components]), ','.join(map(str, self.w)), str(self.name), str(self.prior))

    def get_prior(self):
        """Return the joint prior as a CompositeDistribution of the weight
        prior and the component priors."""
        return CompositeDistribution((self.prior, CompositeDistribution([d.get_prior() for d in self.components])))

    def set_prior(self, prior):
        """Set a weight prior or a joint weight/component prior.

        Args:
            prior: Weight prior, ``mixture_prior(...)`` result,
                ``(weight_prior, component_priors)`` pair, or a mapping with
                ``weights``/``components`` entries.
        """
        weight_prior, component_priors = _split_mixture_prior(prior, self.num_components)
        self.prior = _default_weight_prior(self.num_components) if weight_prior is None else weight_prior
        if component_priors is not None:
            for d, p in zip(self.components, component_priors):
                d.set_prior(p)
        self.conj_prior_params, self.expected_nparams = _dirichlet_expectations(self.prior, self.num_components)

    def get_parameters(self):
        """Return (w, [component parameters])."""
        return self.w, [u.get_parameters() for u in self.components]

    def set_parameters(self, params):
        """Set parameters from (w, [component parameters]).

        Args:
            params: Tuple of mixture weights and a list of per-component
                parameter values.
        """
        self.w = params[0]
        for d,p in zip(self.components, params[1]):
            d.set_parameters(p)

    def density(self, x):
        """Density of the mixture at observation x.

        Args:
            x: Observation compatible with the component distributions.

        Returns:
            Density (float) at x.
        """
        return exp(self.log_density(x))

    def log_density(self, x):
        """Log-density log(sum_k w_k f_k(x)) at observation x.

        Args:
            x: Observation compatible with the component distributions.

        Returns:
            Log-density (float) at x.
        """
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def expected_log_density(self, x):
        """Prior-expected log-density at observation x.

        Uses E[ln w_k] under a (symmetric) Dirichlet weight prior, falling
        back to the current log weights when no conjugate prior is set.

        Args:
            x: Observation compatible with the component distributions.

        Returns:
            Expected log-density (float) at x.
        """
        cc = self.expected_nparams if self.expected_nparams is not None else self.log_w
        return vec.log_sum(np.asarray([u.expected_log_density(x) for u in self.components]) + cc)

    def seq_expected_log_density(self, x):
        """Vectorized expected log-density at sequence-encoded input x.

        Args:
            x: Encoded data from seq_encode().

        Returns:
            Numpy array of expected log-densities, one entry per observation.
        """
        cc = self.expected_nparams if self.expected_nparams is not None else self.log_w
        ll = np.asarray([u.seq_expected_log_density(x) for u in self.components]).T + cc
        ml = np.max(ll, axis=1, keepdims=True)
        return np.log(np.sum(np.exp(ll - ml), axis=1)) + ml.flatten()

    def posterior(self, x):
        """Posterior component-membership probabilities for observation x.

        Args:
            x: Observation compatible with the component distributions.

        Returns:
            Numpy array of length num_components summing to one.
        """

        comp_log_density = np.asarray([m.log_density(x) for m in self.components])
        comp_log_density += self.log_w
        comp_log_density[self.w == 0] = -np.inf

        max_val = np.max(comp_log_density)

        if max_val == -np.inf:
            return self.w.copy()
        else:
            comp_log_density -= max_val
            np.exp(comp_log_density, out=comp_log_density)
            comp_log_density /= comp_log_density.sum()
            return comp_log_density


    def seq_log_density(self, x):
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x: Encoded data from seq_encode().

        Returns:
            Numpy array of log-densities, one entry per observation.
        """

        ll_mat = np.asarray([u.seq_log_density(x) for u in self.components]).T + self.log_w
        ll_max  = ll_mat.max(axis=1, keepdims=True)

        good_rows = np.isfinite(ll_max.flatten())

        if np.all(good_rows):
            ll_mat -= ll_max

            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max

            return ll_sum.flatten()

        else:
            ll_mat = ll_mat[good_rows, :]
            ll_max = ll_max[good_rows]

            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max
            rv = np.zeros(good_rows.shape, dtype=float)
            rv[good_rows] = ll_sum.flatten()
            rv[~good_rows] = -np.inf

            return rv


    def seq_component_log_density(self, x):
        """Per-component log-densities at sequence-encoded input x.

        Args:
            x: Encoded data from seq_encode().

        Returns:
            Numpy array of shape (n, num_components).
        """
        ll_mat = np.asarray([u.seq_log_density(x) for u in self.components]).T
        return ll_mat

    def seq_posterior(self, x):
        """Posterior component-membership probabilities at encoded input x.

        Args:
            x: Encoded data from seq_encode().

        Returns:
            Numpy array of shape (n, num_components) with rows summing to one.
        """

        ll_mat = np.asarray([u.seq_log_density(x) for u in self.components]).T + self.log_w
        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())

        #if np.any(bad_rows):
        #	print('bad')

        ll_mat[bad_rows, :] = self.log_w
        ll_max[bad_rows]    = np.max(self.log_w)

        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)
        ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
        ll_mat /= ll_sum

        return ll_mat

    def seq_encode(self, x):
        """Encode a sequence of observations for vectorized evaluation.

        Args:
            x: Iterable of observations.

        Returns:
            Encoding produced by the first component (shared by all).
        """
        return self.components[0].seq_encode(x)

    def sampler(self, seed: Optional[int] = None):
        """Return a MixtureSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.
        """
        return MixtureSampler(self, seed)

    def estimator(self):
        """Return a MixtureEstimator matching this distribution."""
        return MixtureEstimator([u.estimator() for u in self.components], name=self.name, prior=self.prior)


class MixtureSampler(object):
    """Draws observations from a MixtureDistribution."""

    def __init__(self, dist: MixtureDistribution, seed: Optional[int] = None):
        """Create a sampler for a MixtureDistribution.

        Args:
            dist (MixtureDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.
        """

        rng_loc = RandomState(seed)

        self.rng = RandomState(rng_loc.randint(maxint))
        self.dist = dist
        self.compSamplers = [d.sampler(seed=rng_loc.randint(maxint)) for d in self.dist.components]

    def sample(self, size=None):
        """Draw size samples (or one sample when size is None).

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            A single observation when size is None, otherwise a list of
            observations.
        """

        compState = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
                return self.compSamplers[compState].sample()
        else:
                return [self.compSamplers[i].sample() for i in compState]


class MixtureEstimatorAccumulator(StatisticAccumulator):
    """Accumulates posterior-weighted component counts and component
    sufficient statistics for mixture estimation."""

    def __init__(self, accumulators, keys=(None, None)):
        """Create a mixture accumulator.

        Args:
            accumulators: List of per-component StatisticAccumulator objects.
            keys: Tuple (weight_key, comp_key) for sharing statistics across
                accumulators with matching keys.
        """
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]

    def update(self, x, weight, estimate):
        """Accumulate one observation, splitting weight by the posterior of
        the current estimate.

        Args:
            x: Observation.
            weight (float): Observation weight.
            estimate (MixtureDistribution): Current model estimate.
        """

        likelihood = np.asarray([estimate.components[i].log_density(x) for i in range(self.num_components)])
        likelihood += estimate.log_w
        max_likelihood = likelihood.max()
        likelihood -= max_likelihood

        np.exp(likelihood, out=likelihood)
        pp = likelihood.sum()
        likelihood /= pp

        self.comp_counts += likelihood * weight

        for i in range(self.num_components):
            self.accumulators[i].update(x, likelihood[i] * weight, estimate.components[i])

    def initialize(self, x, weight, rng):
        """Initialize component accumulators with randomly split weight.

        Args:
            x: Observation.
            weight (float): Observation weight.
            rng (RandomState): Random number generator.
        """

        if weight == 0:
            for i in range(self.num_components):
                self.accumulators[i].initialize(x, 0, rng)
        else:
            wc  = rng.dirichlet(np.ones(self.num_components))
            for i in range(self.num_components):
                w = weight*wc[i]
                self.accumulators[i].initialize(x, w, rng)
                self.comp_counts[i] += w

    def seq_update(self, x, weights, estimate):
        """Vectorized update from sequence-encoded data.

        Args:
            x: Encoded data from MixtureDistribution.seq_encode().
            weights (np.ndarray): Observation weights.
            estimate (MixtureDistribution): Current model estimate.
        """

        ll_mat = np.asarray([u.seq_log_density(x) for u in estimate.components]).T + estimate.log_w
        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())

        #if np.any(bad_rows):
        #	print('bad')

        ll_mat[bad_rows, :] = estimate.log_w
        ll_max[bad_rows]    = np.max(estimate.log_w)

        #ll_mat[bad_rows, :] = -np.log(self.num_components)
        #ll_max[bad_rows]    = -np.log(self.num_components)

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat)
        ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
        ll_mat /= ll_sum

        for i in range(self.num_components):
            w_loc = ll_mat[:, i]*weights
            self.comp_counts[i] += w_loc.sum()
            self.accumulators[i].seq_update(x, w_loc, estimate.components[i])




    def combine(self, suff_stat):
        """Merge another accumulator's value() into this one.

        Args:
            suff_stat: Tuple (comp_counts, component suff stats).

        Returns:
            This accumulator.
        """

        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self):
        """Return (comp_counts, tuple of component suff stats)."""
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x):
        """Set this accumulator's state from a value() tuple.

        Args:
            x: Tuple (comp_counts, component suff stats).

        Returns:
            This accumulator.
        """
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
        return self

    def key_merge(self, stats_dict):
        """Merge keyed statistics into stats_dict.

        Args:
            stats_dict: Mapping from key to shared statistics.
        """

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.comp_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict):
        """Replace this accumulator's statistics with keyed entries from
        stats_dict.

        Args:
            stats_dict: Mapping from key to shared statistics.
        """

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

class MixtureEstimatorAccumulatorFactory(object):
    """Factory for creating MixtureEstimatorAccumulator objects."""

    def __init__(self, factories, dim, keys):
        """Create a mixture accumulator factory.

        Args:
            factories: List of per-component accumulator factories.
            dim (int): Number of mixture components.
            keys: Tuple (weight_key, comp_key) passed to the accumulators.
        """
        self.factories = factories
        self.dim = dim
        self.keys = keys

    def make(self):
        """Return a new MixtureEstimatorAccumulator."""
        return MixtureEstimatorAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class MixtureEstimator(ParameterEstimator):
    """Estimates a MixtureDistribution from accumulated sufficient
    statistics, using Dirichlet MAP weights when a conjugate prior is set."""

    def __init__(self, estimators, fixed_w=None, name=None, prior=default_prior, keys=(None, None)):
        """Create a mixture estimator.

        Args:
            estimators: List of per-component ParameterEstimator objects.
            fixed_w: Optional fixed mixture weights (skips weight estimation).
            name (Optional[str]): Name of the estimated distribution.
            prior: Prior on the mixture weights, or a joint prior produced by
                ``mixture_prior(weight_prior, component_priors)``.
            keys: Tuple (weight_key, comp_key) for sharing statistics.
        """

        self.num_components = len(estimators)
        self.estimators = estimators
        self.prior = None
        self.keys = keys
        self.name = name
        self.fixed_w = None if fixed_w is None else np.copy(fixed_w)
        self.set_prior(prior)

    def accumulator_factory(self):
        """Return a MixtureEstimatorAccumulatorFactory for this estimator."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return MixtureEstimatorAccumulatorFactory(est_factories, self.num_components, self.keys)

    def get_prior(self):
        """Return the joint prior as a CompositeDistribution of the weight
        prior and the component estimators' priors."""
        return CompositeDistribution((self.prior, CompositeDistribution([d.get_prior() for d in self.estimators], name=self.keys[1])))

    def set_prior(self, prior):
        """Set a weight prior or a joint weight/component prior.

        Args:
            prior: Weight prior, ``mixture_prior(...)`` result,
                ``(weight_prior, component_priors)`` pair, or a mapping with
                ``weights``/``components`` entries.
        """
        weight_prior, component_priors = _split_mixture_prior(prior, self.num_components)
        self.prior = _default_weight_prior(self.num_components) if weight_prior is None else weight_prior
        if component_priors is not None:
            for d, p in zip(self.estimators, component_priors):
                d.set_prior(p)

    def model_log_density(self, model: MixtureDistribution) -> float:
        """Log density of the model parameters under this estimator's prior.

        Args:
            model (MixtureDistribution): Model to evaluate.

        Returns:
            Weight-prior log density at the estimated weights plus each
            component's prior term.
        """
        # weight prior at the estimated weights, plus each component's prior
        # term evaluated through the component estimator (which knows its own
        # prior parameterization)
        rv = 0.0 if self.prior is None else float(self.prior.log_density(model.w))

        for est, comp in zip(self.estimators, model.components):
            rv += est.model_log_density(comp)

        return rv

    def scale_suff_stat(self, suff_stat, c):
        """Scale mixture sufficient statistics, delegating component payloads.

        Component estimators may carry non-linear metadata such as integer
        support offsets, so the mixture must not structurally scale child
        payloads itself.
        """
        counts, comp_suff_stats = suff_stat
        return counts * c, tuple(
            est.scale_suff_stat(ss, c)
            for est, ss in zip(self.estimators, comp_suff_stats)
        )

    def estimate(self, suff_stat):
        """Estimate a MixtureDistribution from sufficient statistics.

        Args:
            suff_stat: Tuple (comp_counts, component suff stats) as returned
                by MixtureEstimatorAccumulator.value().

        Returns:
            MixtureDistribution with MAP weights (under a Dirichlet prior)
            and estimated components.
        """

        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [self.estimators[i].estimate(comp_suff_stats[i]) for i in range(num_components)]

        if self.fixed_w is not None:
            return MixtureDistribution(components, self.fixed_w, name=self.name, prior=self.prior)


        if isinstance(self.prior, (DirichletDistribution, SymmetricDirichletDistribution)):

            cpp = np.add(counts, self.prior.get_parameters())-1.0
            # MAP of a Dirichlet lies on the boundary when alpha_k + n_k < 1
            cpp = np.maximum(cpp, 0.0)

            if cpp.sum() == 0:
                w = np.ones(num_components)/float(num_components)
            else:
                w = cpp/(cpp.sum())

            return MixtureDistribution(components, w, name=self.name, prior=DirichletDistribution(cpp+1))

        else:

            nobs_loc = counts.sum()

            if nobs_loc == 0:
                w = np.ones(num_components)/float(num_components)
            else:
                w = counts / counts.sum()

            return MixtureDistribution(components, w, name=self.name, prior=self.prior)
