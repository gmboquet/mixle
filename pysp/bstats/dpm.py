"""Dirichlet process mixture (truncated stick-breaking) with variational
Bayes estimation.

A truncated stick-breaking representation with K components:

    alpha ~ Gamma(s1, 1/s2)                  (concentration hyper-prior)
    v_k | alpha ~ Beta(1, alpha)             (stick fractions, k < K)
    w_k = v_k * prod_{j<k} (1 - v_j)         (mixture weights)
    z_i | w ~ Categorical(w)
    x_i | z_i = k ~ components[k]

Data type: whatever the component distributions accept; a datum is a single
observation scored under the mixture log sum_k w_k p(x | theta_k).

Estimation is mean-field variational Bayes: accumulators collect the
optimal local assignments phi_ik (computed from the components'
expected_log_density, i.e. the VB E-step), and the estimator updates the
variational Beta posteriors gamma_k on the stick fractions, the Gamma
hyper-posterior on alpha, and each component's conjugate update. Components
are re-sorted by expected count each iteration, and each component's
posterior (carried as its prior) serves as the variational factor
q(theta_k). seq_local_elbo provides the per-observation data terms of the
ELBO; the data-independent terms live in
DirichletProcessMixtureEstimator.model_log_density.
"""
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
    """Log-density of a compound Beta-Gamma stick fraction: x = 1 - exp(-y)
    with y ~ Exponential(alpha) and alpha ~ Gamma(s1, 1/s2), marginalized
    over alpha.

    Args:
        x (float): Stick fraction in (0, 1).
        s1 (float): Gamma shape of the concentration hyper-prior.
        s2 (float): Gamma rate of the concentration hyper-prior.

    Returns:
        Log-density at x.

    """
    return np.log(s1) + s1*np.log(s2) - (s1+1)*np.log(s2-np.log1p(-x)) - np.log1p(-x)

default_prior = GammaDistribution(2,1)
#default_prior = null_dist

class DirichletProcessMixtureDistribution(ProbabilityDistribution):
    """Truncated Dirichlet process mixture with stick-breaking weights w over
    K component distributions, carrying the variational Beta posteriors."""

    def __init__(self, components, w, a, g, component_priors, name=None, prior=default_prior):
        """DirichletProcessMixtureDistribution object.

        Args:
            components: List of K component distributions (each carrying its
                own posterior as its prior).
            w: Length-K mixture weight vector.
            a (float): Concentration parameter alpha (point estimate).
            g (np.ndarray): (K, 2) array of variational Beta posterior
                parameters gamma_k on the stick fractions.
            component_priors: List of the K component priors used as the
                variational factors q(theta_k) in the ELBO.
            name (Optional[str]): Name of object.
            prior: Gamma hyper-prior (or hyper-posterior) on alpha.

        """
        self.set_parameters((a, w, components))
        self.name = name
        self.prior = prior
        self.g = g
        self.component_priors = component_priors

    def __str__(self):
        return 'DirichletProcessMixtureDistribution([%s], [%s], %s, name=%s, prior=%s)' % (','.join([str(u) for u in self.components]), ','.join(map(str, self.v)), str(self.a), str(self.name), str(self.prior))

    def get_prior(self):
        """Returns the composite prior (alpha hyper-prior, stick-fraction
        prior, component priors) in composable form."""
        vprior = SequenceDistribution(BetaDistribution(1, self.a))
        cprior = CompositeDistribution([u.get_prior() for u in self.components])
        return CompositeDistribution((self.prior, vprior, cprior))

    def set_prior(self, prior):
        """Set the priors from the composite form produced by get_prior().

        Args:
            prior (CompositeDistribution): Composite of (alpha hyper-prior,
                stick-fraction prior, component priors); the component
                priors are pushed down into the components.

        """
        self.prior = prior.dists[0]
        for u,p in zip(self.components, prior.dists[2]):
            u.set_prior(p)

    def get_parameters(self):
        """Returns the parameter tuple (alpha, weights, component parameters)."""
        return self.a, self.v, [u.get_parameters() for u in self.components]

    def set_parameters(self, params):
        """Set the parameters and refresh the cached log-weights.

        Args:
            params: Tuple (alpha, weights, components).

        """
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
        """Density of the mixture at observation x; see log_density().

        Args:
            x: Observation accepted by the component distributions.

        Returns:
            Density at observation x.

        """
        return exp(self.log_density(x))

    def log_density(self, x):
        """Mixture log-density log sum_k w_k p(x | theta_k) at observation x.

        Args:
            x: Observation accepted by the component distributions.

        Returns:
            Log-density at observation x.

        """
        return vec.log_sum(np.asarray([u.log_density(x) for u in self.components]) + self.log_w)

    def expected_log_density(self, x):
        """Mixture log-density with each component's plug-in log-density
        replaced by its variational expectation E_q[log p(x | theta_k)].

        Args:
            x: Observation accepted by the component distributions.

        Returns:
            Expected log-density at observation x.

        """
        return vec.log_sum(np.asarray([u.expected_log_density(x) for u in self.components]) + self.log_w)

    def seq_log_density(self, x):
        """Vectorized log-density at sequence-encoded input x.

        Args:
            x: Encoded observations from seq_encode().

        Returns:
            Numpy array of log-densities, one per observation.

        """
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
        """Vectorized expected_log_density() at sequence-encoded input x.

        Args:
            x: Encoded observations from seq_encode().

        Returns:
            Numpy array of expected log-densities, one per observation.

        """
        ll = np.asarray([u.seq_expected_log_density(x) for u in self.components]).T + self.log_w
        ml = np.max(ll, axis=1, keepdims=True)
        return (np.log(np.sum(np.exp(ll - ml), axis=1, keepdims=True)) + ml).flatten()

    def seq_encode(self, x):
        """Encode observations with the shared component encoding.

        Args:
            x: Iterable of observations.

        Returns:
            Encoded data for use with seq_ methods.

        """
        return self.components[0].seq_encode(x)

    def sampler(self, seed=None):
        """Create a DirichletProcessMixtureSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.

        Returns:
            DirichletProcessMixtureSampler object.

        """
        return DirichletProcessMixtureSampler(self, seed)

    def estimator(self, pseudo_count=None):
        """Create a DirichletProcessMixtureEstimator from this distribution's components.

        Args:
            pseudo_count (Optional[float]): Passed through to the component
                estimators when given.

        Returns:
            DirichletProcessMixtureEstimator object.

        """
        if pseudo_count is not None:
            return DirichletProcessMixtureEstimator([u.estimator(pseudo_count=1.0/self.num_components) for u in self.components], pseudo_count=pseudo_count)
        else:
            return DirichletProcessMixtureEstimator([u.estimator() for u in self.components])


class DirichletProcessMixtureSampler(object):
    """Draws samples from a DirichletProcessMixtureDistribution."""

    def __init__(self, dist, seed=None):
        """DirichletProcessMixtureSampler object.

        Args:
            dist (DirichletProcessMixtureDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.

        """
        rng_loc = RandomState(seed)

        self.rng = RandomState(rng_loc.randint(maxint))
        self.dist = dist
        self.compSamplers = [d.sampler(seed=rng_loc.randint(maxint)) for d in self.dist.components]

    def sample(self, size=None):
        """Draw size samples (a single observation when size is None).

        A component is chosen with probability w_k and an observation is
        drawn from that component.

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            A single observation if size is None, else a list of size observations.

        """
        compState = self.rng.choice(range(0, len(self.dist.w)), size=size, replace=True, p=self.dist.w)

        if size is None:
                return self.compSamplers[compState].sample()
        else:
                return [self.compSamplers[i].sample() for i in compState]


class DirichletProcessMixtureAccumulator(StatisticAccumulator):
    """Accumulates DPM sufficient statistics: expected component counts,
    Beta stick-fraction counts, and each component's weighted statistics."""

    def __init__(self, accumulators, keys=(None, None)):
        """DirichletProcessMixtureAccumulator object.

        Args:
            accumulators: List of K component accumulators.
            keys (Tuple[Optional[str], Optional[str]]): Keys for sharing the
                stick-fraction counts and the component accumulators.

        """
        self.accumulators = accumulators
        self.num_components = len(accumulators)
        self.comp_counts = np.zeros(self.num_components, dtype=float)
        self.beta_counts = np.zeros((self.num_components, 2), dtype=float)
        self.prev_nw = np.log(0.5)*(self.num_components-1)
        self.a = 1.0
        self.weight_key = keys[0]
        self.comp_key = keys[1]

    def update(self, x, weight, estimate):
        """Accumulate the VB E-step statistics for observation x.

        Computes the optimal variational assignment phi from the current
        estimate's expected log-densities and weights, then adds the
        weighted phi to the component and stick-fraction counts and pushes
        phi-weighted updates into the component accumulators.

        Args:
            x: Observation accepted by the component distributions.
            weight (float): Weight of the observation.
            estimate (DirichletProcessMixtureDistribution): Current estimate
                supplying expected log-densities and log-weights.

        """
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
        """Vectorized update() on sequence-encoded data.

        Args:
            x: Encoded observations from seq_encode().
            weights (np.ndarray): Weight per observation.
            estimate (DirichletProcessMixtureDistribution): Current estimate
                supplying expected log-densities and log-weights.

        """
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
        """Initialize with a random Dirichlet assignment of observation x.

        Args:
            x: Observation accepted by the component distributions.
            weight (float): Weight of the observation.
            rng (RandomState): Random number generator for the assignment.

        """
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
        """Add another accumulator's sufficient-statistic value into this one.

        Args:
            suff_stat: Tuple as returned by value().

        Returns:
            This accumulator.

        """
        self.comp_counts += suff_stat[0]
        self.beta_counts += suff_stat[1]
        self.a = suff_stat[2]
        self.prev_nw = suff_stat[3]

        for i in range(self.num_components):
            self.accumulators[i].combine(suff_stat[4][i])
        return self

    def value(self):
        """Returns (comp_counts, beta_counts, alpha, prev_nw, component values)."""
        return self.comp_counts, self.beta_counts, self.a, self.prev_nw, tuple([u.value() for u in self.accumulators])

    def from_value(self, x):
        """Set the sufficient statistics from a value() tuple.

        Args:
            x: Tuple as returned by value().

        Returns:
            This accumulator.

        """
        self.comp_counts = x[0]
        self.beta_counts = x[1]
        self.a = x[2]
        self.prev_nw = x[3]
        for i in range(self.num_components):
            self.accumulators[i].from_value(x[4][i])
        return self

    def key_merge(self, stats_dict):
        """Merge this accumulator's keyed statistics into a shared dict.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
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
        """Replace this accumulator's statistics with the pooled keyed values.

        Args:
            stats_dict (dict): Shared key-to-statistics dictionary.

        """
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
    """Factory that creates DirichletProcessMixtureAccumulator objects."""

    def __init__(self, factories, dim, keys):
        """DirichletProcessMixtureAccumulatorFactory object.

        Args:
            factories: List of K component accumulator factories.
            dim (int): Number of components K.
            keys: Keys passed to created accumulators.

        """
        self.factories = factories
        self.dim = dim

        self.keys = keys

    def make(self):
        """Returns a new DirichletProcessMixtureAccumulator."""
        return DirichletProcessMixtureAccumulator([self.factories[i].make() for i in range(self.dim)], self.keys)


class DirichletProcessMixtureEstimator(object):
    """Estimates a DirichletProcessMixtureDistribution by mean-field
    variational Bayes from accumulated assignment statistics."""

    def __init__(self, estimators, name=None, prior=default_prior, keys=(None, None)):
        """DirichletProcessMixtureEstimator object.

        Args:
            estimators: List of K component estimators.
            name (Optional[str]): Name of the estimated distribution.
            prior: Gamma hyper-prior on the concentration alpha.
            keys (Tuple[Optional[str], Optional[str]]): Keys for sharing the
                stick-fraction counts and the component accumulators.

        """
        # self.estimator   = estimator
        # self.dim         = num_components
        self.name = name

        self.num_components = len(estimators)
        self.estimators = estimators
        self.keys = keys
        self.prior = prior

    def accumulator_factory(self):
        """Returns a DirichletProcessMixtureAccumulatorFactory for this estimator."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        return DirichletProcessMixtureAccumulatorFactory(est_factories, self.num_components, self.keys)

    def get_prior(self):
        """Returns the composite prior (alpha hyper-prior, stick-fraction
        prior at the prior-mean alpha, component priors)."""
        vprior = SequenceDistribution(BetaDistribution(1,(self.prior.k-1)*self.prior.theta), null_dist)
        cprior = CompositeDistribution([u.get_prior() for u in self.estimators])
        return CompositeDistribution((self.prior, vprior, cprior))

    def set_prior(self, prior):
        """Set the priors from the composite form produced by get_prior().

        Args:
            prior (CompositeDistribution): Composite of (alpha hyper-prior,
                stick-fraction prior, component priors); the component
                priors are pushed down into the component estimators.

        """
        self.prior = prior.dists[0]
        for e,p in zip(self.estimators, prior.dists[2]):
            e.set_prior(p)

    def model_log_density(self, model):
        """Data-independent ELBO terms of the variational approximation.

        Combines the cross-entropies of the stick-fraction prior and the
        component priors against their variational posteriors with the
        entropies of those posteriors. Together with
        DirichletProcessMixtureDistribution.seq_local_elbo this forms the
        full ELBO maximized by bestimation.optimize.

        Args:
            model (DirichletProcessMixtureDistribution): Model to score.

        Returns:
            Sum of the global ELBO terms.

        """
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
        """Estimate a DirichletProcessMixtureDistribution by one VB M-step.

        Re-estimates each component (whose conjugate update carries its
        posterior forward as its prior), re-sorts components by expected
        count, updates the variational Beta posteriors gamma_k on the stick
        fractions, updates the Gamma hyper-posterior on the concentration
        alpha (carried as the returned distribution's prior), and converts
        the expected log stick fractions into the mixture weights w.

        Args:
            suff_stat: Tuple (comp_counts, beta_counts, alpha, prev_nw,
                component suff stats) as returned by
                DirichletProcessMixtureAccumulator.value().

        Returns:
            DirichletProcessMixtureDistribution object.

        """
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


        # invert weights to stick fractions: v_i = w_i / prod_{j<i}(1 - v_j).
        # tracked via the remaining mass directly (no logs), clipping each
        # fraction so late sticks with underflowed mass stay well-defined
        v = np.zeros(len(w))
        remaining = 1.0
        for i in range(len(w)):
            vi = w[i] / remaining if remaining > 1.0e-12 else 0.0
            v[i] = min(max(vi, 1.0e-9), 1.0 - 1.0e-9)
            remaining *= (1.0 - v[i])


        gammas = np.copy(beta_counts)
        gammas[:,0] += 1
        gammas[:,1] += new_alpha


        return DirichletProcessMixtureDistribution(components, w, new_alpha, gammas, component_priors, prior=hyper_posterior)


