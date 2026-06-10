"""Beta distribution on the unit interval with shape parameters a and b.

Data type: (float): The BetaDistribution with shape parameters a > 0 and
    b > 0 has log-density
    log(f(x; a, b)) = (a-1)*log(x) + (b-1)*log(1-x)
                      + gammaln(a+b) - gammaln(a) - gammaln(b), for x in (0, 1).

Defines the BetaDistribution and BetaSampler classes for use with pysp.bstats.
This distribution is primarily used as the conjugate prior on the success
probability of other bstats models (e.g. pysp.bstats.bernoulli,
pysp.bstats.geometric, pysp.bstats.binomial, pysp.bstats.setdist).
"""
from typing import Optional

from pysp.bstats.pdist import ProbabilityDistribution
from pysp.utils.special import gammaln, betaln, digamma, beta
import numpy as np
import scipy.integrate


class BetaDistribution(ProbabilityDistribution):
	"""Beta distribution with shape parameters a and b on the unit interval."""

	def __init__(self, a: float, b: float, name: Optional[str] = None, prior: Optional[ProbabilityDistribution] = None):
		"""BetaDistribution object.

		Args:
			a (float): Positive shape parameter (weight on log(x)).
			b (float): Positive shape parameter (weight on log(1-x)).
			name (Optional[str]): Name of object.
			prior (Optional[ProbabilityDistribution]): Prior on (a, b).
		"""

		self.set_parameters((a,b))
		self.name  = name
		self.prior = prior

	def __str__(self):
		return 'BetaDistribution(%f, %f, name=%s, prior=%s)' % (self.a, self.b, self.name, str(self.prior))

	def __repr__(self):
		return self.__str__()

	def get_parameters(self):
		"""Returns the shape parameters (a, b)."""
		return self.a, self.b

	def set_parameters(self, params):
		"""Sets the shape parameters and refreshes the normalizing constant.

		Args:
			params: Tuple (a, b) of positive shape parameters.
		"""
		a, b = params
		self.a = a
		self.b = b
		self.norm_const = gammaln(a+b) - gammaln(a) - gammaln(b)

	def cross_entropy(self, dist):
		"""Cross entropy -E_self[log dist(x)] over the unit interval.

		Closed form for a BetaDistribution argument; numerical quadrature
		otherwise.

		Args:
			dist (ProbabilityDistribution): Distribution evaluated under this
				one.

		Returns:
			float: Cross entropy value in nats.
		"""
		if isinstance(dist, BetaDistribution):
			a = self.a
			b = self.b
			aa = dist.a
			bb = dist.b
			return betaln(aa,bb) - (aa-1)*digamma(a) - (bb-1)*digamma(b) + (aa+bb-2)*digamma(a+b)
		else:
			return -scipy.integrate.quad(lambda x: dist.log_density(x) * self.density(x), 0, 1)[0]

	def entropy(self):
		"""Returns the differential entropy in nats."""
		a = self.a
		b = self.b
		return betaln(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b)

	def density(self, x):
		"""Density at observation x.

		Args:
			x (float): Observation in (0, 1).

		Returns:
			Density value at x.
		"""
		return np.power(x, self.a - 1) * np.power(1-x, self.b - 1)/beta(self.a, self.b)

	def log_density(self, x: float):
		"""Log-density at observation x.

		Args:
			x (float): Observation in (0, 1).

		Returns:
			Log-density value at x.
		"""
		a = self.a
		b = self.b

		return np.log(x)*(a-1) + np.log1p(-x)*(b-1) + self.norm_const

	def sampler(self, seed: int = None):
		"""Return a BetaSampler for this distribution.

		Args:
			seed (Optional[int]): Seed for the random number generator.
		"""
		return BetaSampler(self, seed)


class BetaSampler(object):
	"""Draws observations in (0, 1) from a BetaDistribution."""

	def __init__(self, dist, seed=None):
		"""BetaSampler object.

		Args:
			dist (BetaDistribution): Distribution to sample from.
			seed (Optional[int]): Seed for the random number generator.
		"""
		self.dist = dist
		self.seed = seed
		self.rng  = np.random.RandomState(seed)

	def sample(self, size=None):
		"""Draw size samples (or one sample when size is None).

		Args:
			size (Optional[int]): Number of samples to draw.

		Returns:
			A single float when size is None, otherwise a numpy array of
			length size.
		"""
		if size is None:
			return self.rng.beta(self.dist.a, self.dist.b)
		else:
			return self.rng.beta(self.dist.a, self.dist.b, size=size)
