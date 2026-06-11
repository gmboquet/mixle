"""Dirichlet distribution on dictionary-valued probability maps: observations
are dicts {value: probability} whose probabilities are non-negative and sum to
one (points on the simplex indexed by the dict keys).

Data type: (Dict[Any, float]): A DictDirichletDistribution with concentration
parameters alpha = {k: a_k} has log-density

    log f(x; alpha) = gammaln(sum_k a_k)
                      + sum_k [(a_k - 1)*log(x_k) - gammaln(a_k)].

A single scalar alpha is treated as a symmetric Dirichlet whose dimension is
inferred from each observation (is_unbounded). Defines the
DictDirichletDistribution and DictDirichletSampler classes for use with
pysp.bstats. This distribution is the conjugate prior used by
pysp.bstats.categorical, so its log_density feeds every
CategoricalEstimator.model_log_density evaluation.
"""
from typing import Optional, Union, List, Dict, Any

from pysp.bstats.pdist import ProbabilityDistribution
from pysp.utils.special import gammaln, digamma
import numpy as np



class DictDirichletDistribution(ProbabilityDistribution):
    """Dirichlet distribution over probability maps keyed by arbitrary values;
    a scalar alpha denotes a symmetric Dirichlet of unspecified dimension."""

    def __init__(self, alpha: Union[Dict[Any, float], float]):
        """DictDirichletDistribution object.

        Args:
            alpha: Concentration parameters. Either a dict {value: a_k} of
                positive reals or a single positive scalar (symmetric
                Dirichlet, dimension inferred from each observation).
        """
        self.set_parameters(alpha)

    def __str__(self):
        return 'DictDirichletDistribution(%s)'%(str(self.alpha))

    def get_parameters(self) -> Union[Dict, float]:
        """Returns the concentration parameters (dict, or scalar if unbounded)."""
        return self.alpha

    def set_parameters(self, params: Union[Dict[Any, float], float]) -> None:
        """Sets the concentration parameters.

        Args:
            params: Dict {value: a_k} of positive reals, or a positive scalar
                for a symmetric Dirichlet of unspecified dimension.
        """
        self.alpha = params
        self.is_unbounded = isinstance(params, float)

    def density(self, x: Dict[Any, float]) -> float:
        """Density at the probability map x (exp of log_density).

        Args:
            x (Dict[Any, float]): Map from value to probability, summing to one.

        Returns:
            Density value at x.
        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Dict[Any, float]) -> float:
        """Log-density of the Dirichlet at the probability map x.

        With scalar alpha the dimension is len(x); with dict alpha the
        observation is scored with the concentration entries matching its keys.

        Args:
            x (Dict[Any, float]): Map from value to probability, summing to one.

        Returns:
            Log-density value at x.
        """
        if self.is_unbounded:
            a = self.alpha
            n = len(x)
            c = (gammaln(a)*n - gammaln(a*n))
            if a == 1:
                return -c
            else:
                return np.sum(np.log(list(x.values())))*(a-1) - c
        else:
            rv = 0.0
            asum = 0.0
            for k,v in x.items():
                a   = self.alpha[k]
                rv += np.log(v)*(a-1) - gammaln(a)
                asum += a
            return rv + gammaln(asum)

    def cross_entropy(self, dist):
        """Cross entropy -E_self[log dist(x)] for a DictDirichlet argument.

        Args:
            dist (DictDirichletDistribution): Distribution evaluated under
                this one.

        Returns:
            float: Cross entropy value in nats.
        """
        if isinstance(dist, DictDirichletDistribution):
            if self.is_unbounded and not dist.is_unbounded:
                aa = np.asarray(list(dist.alpha.values()))
                a = self.alpha * np.ones(len(aa))
            elif not self.is_unbounded and dist.is_unbounded:
                a = np.asarray(list(self.alpha.values()))
                aa = dist.alpha * np.ones(len(a))
            else:
                keys = list(self.alpha.keys())
                a    = np.asarray([self.alpha.get(k) for k in keys])
                aa   = np.asarray([dist.alpha.get(k,0.0) for k in keys])

            return -((gammaln(np.sum(aa)) - np.sum(gammaln(aa))) + np.dot(digamma(a)-digamma(np.sum(a)), aa - 1))
        else:
            raise NotImplementedError('DictDirichletDistribution.cross_entropy is only implemented for DictDirichlet arguments (got %s).' % type(dist).__name__)

    def entropy(self):
        """Returns the differential entropy in nats (dict alpha only)."""
        a = np.asarray(list(self.alpha.values()))
        a0 = np.sum(a)
        return -((gammaln(a0) - np.sum(gammaln(a))) + np.dot(digamma(a) - digamma(a0), a - 1))



    def sampler(self, seed: Optional[int] = None):
        """Returns a DictDirichletSampler for this distribution.

        Args:
            seed (Optional[int]): Seed for the random number generator.
        """
        return DictDirichletSampler(self, seed)


class DictDirichletSampler(object):
    """Draws probability maps from a DictDirichletDistribution with dict-valued
    concentration parameters."""

    def __init__(self, dist: DictDirichletDistribution, seed: Optional[int] = None):
        """DictDirichletSampler object.

        Args:
            dist (DictDirichletDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for the random number generator.
        """
        self.dist = dist
        self.rng  = np.random.RandomState(seed)

    def sample(self, size: Optional[int] = None) -> Union[Dict, List[Dict]]:
        """Draw Dirichlet-distributed probability maps over the alpha keys.

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            A dict {value: probability} when size is None, otherwise a list of
            such dicts.

        Raises:
            ValueError: If the distribution has scalar alpha (is_unbounded),
                since the sample dimension is then unspecified.
        """
        if self.dist.is_unbounded:
            raise ValueError('DictDirichletSampler cannot sample from a DictDirichletDistribution with scalar alpha (unspecified dimension).')

        keys  = list(self.dist.alpha.keys())
        alpha = np.asarray([self.dist.alpha[k] for k in keys], dtype=float)

        if size is None:
            return dict(zip(keys, self.rng.dirichlet(alpha)))
        else:
            return [dict(zip(keys, p)) for p in self.rng.dirichlet(alpha, size=size)]
