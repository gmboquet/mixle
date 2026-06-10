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

class MixtureDistribution(ProbabilityDistribution):

    def __init__(self, components, w, name=None, prior: Optional[ProbabilityDistribution] = None):
        self.set_name(name)

        if prior is None:
            prior = DirichletDistribution(np.ones(len(w)))

        self.components = components
        self.num_components = len(components)
        self.w = np.asarray(w)
        self.zw = (self.w == 0.0)
        self.log_w = np.log(w+self.zw)
        self.log_w[self.zw] = -np.inf
        self.prior = prior

        #self.parents = []
        #for d in self.components:
        #    d.add_parent(self)

        if isinstance(self.prior, DirichletDistribution):
            self.conj_prior_params = self.prior.get_parameters()
            self.expected_nparams  = digamma(self.conj_prior_params) - digamma(np.sum(self.conj_prior_params))
        elif isinstance(self.prior, SymmetricDirichletDistribution):
            self.conj_prior_params = np.ones(self.num_components)*self.prior.get_parameters()
            self.expected_nparams = digamma(self.conj_prior_params) - digamma(np.sum(self.conj_prior_params))
        else:
            self.conj_prior_params = None
            self.expected_nparams  = None

    def __str__(self):
        return 'MixtureDistribution([%s], [%s], name=%s, prior=%s)' % (','.join([str(u) for u in self.components]), ','.join(map(str, self.w)), str(self.name), str(self.prior))

    def get_prior(self):
        return CompositeDistribution((self.prior, CompositeDistribution([d.get_prior() for d in self.components])))

    def set_prior(self, prior):
        self.prior = prior.dists[0]
        for d,p in zip(self.components, prior.dists[1].dists):
            d.set_prior(p)

        if isinstance(self.prior, DirichletDistribution):
            self.conj_prior_params = self.prior.get_parameters()
            self.expected_nparams  =  digamma(self.conj_prior_params) - digamma(np.sum(self.conj_prior_params))
        elif isinstance(self.prior, SymmetricDirichletDistribution):
            self.conj_prior_params = np.ones(self.num_components)*self.prior.get_parameters()
            self.expected_nparams = digamma(self.conj_prior_params) - digamma(np.sum(self.conj_prior_params))
        else:
            self.conj_prior_params = None
            self.expected_nparams  = None

    def get_parameters(self):
        return self.w, [u.get_parameters() for u in self.components]

    def set_parameters(self, params):
        self.w = params[0]
        for d,p in zip(self.components, params[1]):
            d.set_parameters(p)

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x):
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def expected_log_density(self, x):
        cc = self.expected_nparams
        return vec.log_sum(np.asarray([u.expected_log_density(x) for u in self.components]) + cc)

    def seq_expected_log_density(self, x):
        cc = self.expected_nparams
        ll = np.asarray([u.seq_expected_log_density(x) for u in self.components]).T + cc
        ml = np.max(ll, axis=1)
        return np.log(np.sum(np.exp(ll - ml), axis=1)) + ml

    def posterior(self, x):

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
        ll_mat = np.asarray([u.seq_log_density(x) for u in self.components]).T
        return ll_mat

    def seq_posterior(self, x):

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
        return self.components[0].seq_encode(x)

    def sampler(self, seed: Optional[int] = None):
        return MixtureSampler(self, seed)

    def estimator(self):
        return MixtureEstimator([u.estimator() for u in self.components], name=self.name, prior=self.prior)


class MixtureSampler(object):

    def __init__(self, dist: MixtureDistribution, seed: Optional[int] = None):

        rng_loc = RandomState(seed)

        self.rng = RandomState(rng_loc.randint(maxint))
        self.dist = dist
        self.compSamplers = [d.sampler(seed=rng_loc.randint(maxint)) for d in self.dist.components]

    def sample(self, size=None):

        compState = self.rng.choice(range(0, self.dist.num_components), size=size, replace=True, p=self.dist.w)

        if size is None:
                return self.compSamplers[compState].sample()
        else:
                return [self.compSamplers[i].sample() for i in compState]


class MixtureEstimatorAccumulator(StatisticAccumulator):

    def __init__(self, accumulators, keys=(None, None)):
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.weight_key = keys[0]
        self.comp_key = keys[1]

    def update(self, x, weight, estimate):

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

        ttt = 1

        for i in range(self.num_components):
            w_loc = ll_mat[:, i]*weights
            self.comp_counts[i] += w_loc.sum()
            self.accumulators[i].seq_update(x, w_loc, estimate.components[i])




    def combine(self, suff_stat):

        self.comp_counts += suff_stat[0]
        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[1][i])

        return self

    def value(self):
        return self.comp_counts, tuple([u.value() for u in self.accumulators])

    def from_value(self, x):
        self.comp_counts = x[0]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[1][i])
        return self

    def key_merge(self, stats_dict):

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
    def __init__(self, factories, dim, keys):
        self.factories = factories
        self.dim = dim
        self.keys = keys

    def make(self):
        return MixtureEstimatorAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class MixtureEstimator(ParameterEstimator):

    def __init__(self, estimators, fixed_w=None, name=None, prior=default_prior, keys=(None, None)):

        self.num_components = len(estimators)
        self.estimators = estimators
        self.prior = prior
        self.keys = keys
        self.name = name
        self.fixed_w = None if fixed_w is None else np.copy(fixed_w)

    def accumulator_factory(self):
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return MixtureEstimatorAccumulatorFactory(est_factories, self.num_components, self.keys)

    def get_prior(self):
        return CompositeDistribution((self.prior, CompositeDistribution([d.get_prior() for d in self.estimators], name=self.keys[1])))

    def set_prior(self, prior):
        self.prior = prior.dists[0]
        for d,p in zip(self.estimators, prior.dists[1].dists):
            d.set_prior(p)

    def model_log_density(self, model: MixtureDistribution) -> float:
        # weight prior at the estimated weights, plus each component's prior
        # term evaluated through the component estimator (which knows its own
        # prior parameterization)
        rv = 0.0 if self.prior is None else float(self.prior.log_density(model.w))

        for est, comp in zip(self.estimators, model.components):
            rv += est.model_log_density(comp)

        return rv

    def estimate(self, suff_stat):

        num_components = self.num_components
        counts, comp_suff_stats = suff_stat

        components = [self.estimators[i].estimate(comp_suff_stats[i]) for i in range(num_components)]

        if self.fixed_w is not None:
            return MixtureDistribution(components, self.fixed_w)


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

            return MixtureDistribution(components, w, name=self.name)
