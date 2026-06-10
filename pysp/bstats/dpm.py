from pysp.arithmetic import *
from pysp.bstats.pdist import ProbabilityDistribution, StatisticAccumulator, ParameterEstimator
from pysp.utils.special import digamma, gammaln, betaln
from pysp.bstats.gamma import GammaDistribution
from pysp.bstats.beta import BetaDistribution
from pysp.bstats.sequence import SequenceDistribution
from pysp.bstats.composite import CompositeDistribution
from pysp.bstats.nulldist import null_dist
from numpy.random import RandomState
import numpy as np
import pysp.utils.vector as vec

def cbg(x,s1,s2):
    return np.log(s1) + s1*np.log(s2) - (s1+1)*np.log(s2-np.log1p(-x)) - np.log1p(-x)

default_prior = GammaDistribution(2,1)
#default_prior = null_dist

class DirichletProcessMixtureDistribution(ProbabilityDistribution):

    def __init__(self, components, w, a, g, component_priors, name=None, prior=default_prior):

        self.set_parameters((a, w, components))
        self.name = name
        self.prior = prior
        self.g = g
        self.component_priors = component_priors

    def __str__(self):
        return 'DirichletProcessMixtureDistribution([%s], [%s], %s, name=%s, prior=%s)' % (','.join([str(u) for u in self.components]), ','.join(map(str, self.v)), str(self.a), str(self.name), str(self.prior))

    def get_prior(self):
        vprior = SequenceDistribution(BetaDistribution(1, self.a))
        cprior = CompositeDistribution([u.get_prior() for u in self.components])
        return CompositeDistribution((self.prior, vprior, cprior))

    def set_prior(self, prior):
        self.prior = prior.dists[0]
        for u,p in zip(self.components, prior.dists[2]):
            u.set_prior(p)

    def get_parameters(self):
        return self.a, self.v, [u.get_parameters() for u in self.components]

    def set_parameters(self, params):
        a, w, components = params
        #w = np.zeros(len(v))
        #w[0]  = v[0]
        #w[1:] = np.exp(np.log(v[1:]) + np.cumsum(np.log1p(-v[:-1])))
        #w /= w.sum()

        self.components = components
        self.max_components = len(components)
        self.num_components = len(components)
        self.w = np.asarray(w)
        self.a = a
        self.log_w = np.log(w)
        self.expected_log_nw = self.log_w[-1]
        self.v = w

    def density(self, x):
        return exp(self.log_density(x))

    def log_density(self, x):
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def expected_log_density(self, x):
        return vec.log_sum(np.asarray([u.expected_log_density(x) for u in self.components]) + self.log_w)

    def seq_log_density(self, x):

        ll_mat = np.asarray([u.seq_log_density(x) for u in self.components]).T + self.log_w
        ll_max = ll_mat.max(axis=1, keepdims=True)

        good_rows = np.isfinite(ll_max.flatten())

        if np.all(good_rows):
            ll_mat -= ll_max
            np.exp(ll_mat, out=ll_mat)
            ll_sum = np.sum(ll_mat, axis=1, keepdims=True)
            np.log(ll_sum, out=ll_sum)
            ll_sum += ll_max
            return ll_sum.flatten()
        else:
            rv = np.zeros(len(good_rows), dtype=float)
            rv[~good_rows] = -np.inf
            if np.any(good_rows):
                ll_loc = ll_mat[good_rows, :] - ll_max[good_rows]
                rv[good_rows] = (np.log(np.sum(np.exp(ll_loc), axis=1, keepdims=True)) + ll_max[good_rows]).flatten()
            return rv

    def seq_local_elbo(self, x):
        """Per-observation local ELBO contributions.

        For each observation i this returns

            sum_k phi_ik * ( E_q[log p(z_i = k | v)] + E_q[log p(x_i | theta_k)] - log phi_ik )

        where phi_i is the optimal variational assignment for x_i. The global
        (data-independent) ELBO terms are returned by
        DirichletProcessMixtureEstimator.model_log_density.
        """

        exp_ll  = np.asarray([u.seq_expected_log_density(x) for u in self.components]).T + self.log_w
        max_ell = exp_ll.max(axis=1, keepdims=True)

        phi  = np.exp(exp_ll - max_ell)
        phi /= phi.sum(axis=1, keepdims=True)
        phi_g = 1 - np.cumsum(phi, axis=1)

        gam  = self.g
        gams = gam[:, 0] + gam[:, 1]

        # E_q[log p(z_i | v)] via stick-breaking expectations
        exp_v  = digamma(gam[:, 0]) - digamma(gams)
        exp_nv = digamma(gam[:, 1]) - digamma(gams)
        rv = np.dot(phi_g, exp_nv) + np.dot(phi, exp_v)

        # E_q[log p(x_i | theta_k)] under the variational assignments
        rv += np.sum(phi * (exp_ll - self.log_w), axis=1)

        # entropy of the local variational multinomials
        log_phi = np.log(phi, out=np.zeros_like(phi), where=phi > 0)
        rv -= np.sum(phi * log_phi, axis=1)

        return rv

    def seq_expected_log_density(self, x):
        ll = np.asarray([u.seq_expected_log_density(x) for u in self.components]).T + self.log_w
        ml = np.max(ll, axis=1, keepdims=True)
        return (np.log(np.sum(np.exp(ll - ml), axis=1, keepdims=True)) + ml).flatten()

    def seq_encode(self, x):
        return self.components[0].seq_encode(x)

    def sampler(self, seed=None):
        return DirichletProcessMixtureSampler(self, seed)

    def estimator(self, pseudo_count=None):
        if pseudo_count is not None:
            return DirichletProcessMixtureEstimator([u.estimator(pseudo_count=1.0/self.num_components) for u in self.components], pseudo_count=pseudo_count)
        else:
            return DirichletProcessMixtureEstimator([u.estimator() for u in self.components])


class DirichletProcessMixtureSampler(object):

    def __init__(self, dist, seed=None):

        rng_loc = RandomState(seed)

        self.rng = RandomState(rng_loc.randint(maxint))
        self.dist = dist
        self.compSamplers = [d.sampler(seed=rng_loc.randint(maxint)) for d in self.dist.components]

    def sample(self, size=None):

        compState = self.rng.choice(range(0, len(self.dist.w)), size=size, replace=True, p=self.dist.w)

        if size is None:
                return self.compSamplers[compState].sample()
        else:
                return [self.compSamplers[i].sample() for i in compState]


class DirichletProcessMixtureAccumulator(StatisticAccumulator):

    def __init__(self, accumulators, keys=(None, None)):
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.beta_counts = np.zeros((self.num_components, 2), dtype=float)
        self.prev_nw = np.log(0.5)*(self.num_components-1)
        self.a = 1.0
        self.weight_key = keys[0]
        self.comp_key = keys[1]

    def update(self, x, weight, estimate):

        exp_ll = np.asarray([estimate.components[i].expected_log_density(x) for i in range(self.num_components)])
        exp_ll += estimate.log_w
        exp_ll -= exp_ll.max()

        phi = np.exp(exp_ll)
        phi /= phi.sum()

        self.comp_counts += phi * weight
        self.beta_counts[:, 0] += phi * weight
        self.beta_counts[:, 1] += (1 - np.cumsum(phi)) * weight
        #self.prev_nw = estimate.expected_log_nw

        for i in range(self.num_components):
            self.accumulators[i].update(x, phi[i] * weight, estimate.components[i])

    def seq_update(self, x, weights, estimate):

        exp_ll = np.asarray([u.seq_expected_log_density(x) for u in estimate.components]).T
        exp_ll += estimate.log_w
        exp_ll -= exp_ll.max(axis=1, keepdims=True)

        phi = np.exp(exp_ll.T)
        phi /= phi.sum(axis=0, keepdims=True)

        cc_loc = np.dot(phi, weights)
        cs_loc = np.dot((1 - np.cumsum(phi, axis=0)), weights)

        self.comp_counts += cc_loc
        self.beta_counts[:, 0] += cc_loc
        self.beta_counts[:, 1] += cs_loc

        for i in range(self.num_components):
            self.accumulators[i].seq_update(x, phi[i,:] * weights, estimate.components[i])

    def initialize(self, x, weight, rng):

        #v = rng.beta(1,self.a,size=self.num_components)
        #lv = np.log(v)
        #lv[1:] += np.cumsum(np.log(1-v[:-1]))
        #lv -= np.max(lv)
        #p = np.exp(lv)
        #p /= p.sum()

        p = rng.dirichlet(np.ones(self.num_components))

        self.comp_counts += p * weight
        self.beta_counts[:, 0] += p * weight
        self.beta_counts[:, 1] += (1 - np.cumsum(p)) * weight

        for i in range(self.num_components):
            self.accumulators[i].initialize(x, p[i] * weight, rng)

    def combine(self, suff_stat):
        self.comp_counts += suff_stat[0]
        self.beta_counts += suff_stat[1]
        self.a = suff_stat[2]
        self.prev_nw = suff_stat[3]

        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[4][i])
        return self

    def value(self):
        return self.comp_counts, self.beta_counts, self.a, self.prev_nw, tuple([u.value() for u in self.accumulators])

    def from_value(self, x):
        self.comp_counts = x[0]
        self.beta_counts = x[1]
        self.a = x[2]
        self.prev_nw = x[3]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[4][i])
        return self

    def key_merge(self, stats_dict):

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.beta_counts
            else:
                stats_dict[self.weight_key] = self.beta_counts

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
                self.beta_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

class DirichletProcessMixtureAccumulatorFactory(object):
    def __init__(self, factories, dim, keys):
        self.factories = factories
        self.dim = dim

        self.keys = keys

    def make(self):
        return DirichletProcessMixtureAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class DirichletProcessMixtureEstimator(object):

    def __init__(self, estimators, name=None, prior=default_prior, keys=(None, None)):
        # self.estimator   = estimator
        # self.dim         = num_components
        self.name = name

        self.num_components = len(estimators)
        self.estimators = estimators
        self.keys = keys
        self.prior = prior

    def accumulator_factory(self):
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return DirichletProcessMixtureAccumulatorFactory(est_factories, self.num_components, self.keys)

    def get_prior(self):
        vprior = SequenceDistribution(BetaDistribution(1,(self.prior.k-1)*self.prior.theta), null_dist)
        cprior = CompositeDistribution([u.get_prior() for u in self.estimators])
        return CompositeDistribution((self.prior, vprior, cprior))

    def set_prior(self, prior):
        self.prior = prior.dists[0]
        for e,p in zip(self.estimators, prior.dists[2]):
            e.set_prior(p)

    def model_log_density(self, model):

        gam  = model.g
        gams = gam[:,0]+gam[:,1]
        a = model.a

        # cross entropy of beta and variational betas
        temp1 = np.sum(-betaln(1,a) + (digamma(gam[:,1])-digamma(gams))*(a-1))

        # cross entropy of component priors and variational priors
        temp2 = 0
        for i in range(model.max_components):
            temp2 += -model.components[i].get_prior().cross_entropy(model.component_priors[i])

        # entropy of the variational approximation
        # entropy of variational betas
        temp41 = -(betaln(gam[:,0],gam[:,1]).sum() - ((gam-1)*digamma(gam)).sum() + ((gams-2)*digamma(gams)).sum())
        # entropy of variational component priors
        temp42 = np.sum([-u.get_prior().entropy() for u in model.components])
        # entropy of sample variational multinomials
        #temp43 = np.sum(np.log(phi[phi > 0])*phi[phi > 0])
        temp4 = temp41 + temp42

        return temp1 + temp2 - temp4
        #return 0

    def estimate(self, suff_stat):

        num_components = self.num_components
        comp_counts, beta_counts, alpha, prev_nw, comp_suff_stats = suff_stat

        component_priors = [u.get_prior() for u in self.estimators]
        components = [self.estimators[i].estimate(comp_suff_stats[i]) for i in range(num_components)]

        #

        sidx = np.argsort(-comp_counts)
        comp_counts = comp_counts[sidx]
        beta_counts = beta_counts[sidx, :]
        components = [components[i] for i in sidx]

        #

        beta_counts[:, 1] = np.sum(beta_counts[:, 0]) - np.cumsum(beta_counts[:, 0])

        #

        #gammas = np.copy(beta_counts)
        #gammas[:,0] += 1
        #gammas[:,1] += alpha

        #

        dgsum_loc = digamma(beta_counts.sum(axis=1) + 1.0 + alpha)
        dg1_loc = digamma(beta_counts[:, 0] + 1.0)
        dg2_loc = digamma(beta_counts[:, 1] + alpha)

        expected_log_betas = np.vstack([dg1_loc - dgsum_loc, dg2_loc - dgsum_loc]).T

        #

        expected_log_w     = expected_log_betas[:,0]
        expected_log_nw    = np.cumsum(expected_log_betas[:, 1])
        expected_log_w[1:] += expected_log_nw[:-1]

        #

        w = np.exp(expected_log_w - np.max(expected_log_w))
        w /= w.sum()

        #


        if self.prior is None:

            s1 = 0
            s2 = 0
            hyper_posterior = None

        elif isinstance(self.prior, GammaDistribution):
            s1 = self.prior.k
            s2 = 1/self.prior.theta

            s1_new = s1 + num_components
            s2_new = s2 - expected_log_nw[-1]
            hyper_posterior = GammaDistribution(s1_new, 1/s2_new)

        else:
            s1 = 0
            s2 = 0
            hyper_posterior = None

        gw1 = s1 + num_components - 1.0
        gw2 = s2 - expected_log_nw[-2]
        new_alpha = gw1/gw2


        v = np.zeros(len(w))
        v[0] = w[0]
        for i in range(1,len(w)):
            v[i] = np.exp(np.log(w[i]) - np.sum(np.log1p(-v[:i])))
        v = np.minimum(np.maximum(v, 1.0e-9), 1-1.0e-9)


        gammas = np.copy(beta_counts)
        gammas[:,0] += 1
        gammas[:,1] += new_alpha


        return DirichletProcessMixtureDistribution(components, w, new_alpha, gammas, component_priors, prior=hyper_posterior)


