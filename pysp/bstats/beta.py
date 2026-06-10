from typing import Optional

from pysp.stats.pdist import ProbabilityDistribution
from pysp.utils.special import gammaln, betaln, digamma, beta
import numpy as np
import scipy.integrate


class BetaDistribution(ProbabilityDistribution):

	def __init__(self, a: float, b: float, name: Optional[str] = None, prior: Optional[ProbabilityDistribution] = None):

		self.set_parameters((a,b))
		self.name  = name
		self.prior = prior

	def __str__(self):
		return 'BetaDistribution(%f, %f, name=%s, prior=%s)' % (self.a, self.b, self.name, str(self.prior))

	def get_parameters(self):
		return self.a, self.b

	def set_parameters(self, params):
		a, b = params
		self.a = a
		self.b = b
		self.norm_const = gammaln(a+b) - gammaln(a) - gammaln(b)

	def cross_entropy(self, dist):
		if isinstance(dist, BetaDistribution):
			a = self.a
			b = self.b
			aa = dist.a
			bb = dist.b
			return betaln(aa,bb) - (aa-1)*digamma(a) - (bb-1)*digamma(b) + (aa+bb-2)*digamma(a+b)
		else:
			return -scipy.integrate.quad(lambda x: dist.log_density(x) * self.density(x), 0, 1)[0]

	def entropy(self):
		a = self.a
		b = self.b
		return betaln(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b)

	def density(self, x):
		return np.power(x, self.a - 1) * np.power(1-x, self.b - 1)/beta(self.a, self.b)

	def log_density(self, x: float):
		a = self.a
		b = self.b

		return np.log(x)*(a-1) + np.log1p(-x)*(b-1) + self.norm_const

	def sampler(self, seed: int = None):
		return BetaSampler(self, seed)


class BetaSampler(object):

	def __init__(self, dist, seed=None):
		self.dist = dist
		self.seed = seed
		self.rng  = np.random.RandomState(seed)

	def sample(self, size=None):
		if size is None:
			return self.rng.beta(self.dist.a, self.dist.b)
		else:
			return self.rng.beta(self.dist.a, self.dist.b, size=size)

