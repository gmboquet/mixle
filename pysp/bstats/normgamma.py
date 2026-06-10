from typing import Optional, Union

from pysp.bstats.pdist import ProbabilityDistribution
from pysp.utils.special import gammaln, digamma
import numpy as np
import scipy.integrate


class NormalGammaDistribution(ProbabilityDistribution):

	def __init__(self, mu: float, lam: float, a: float, b: float, name: Optional[str] = None, prior: Optional[ProbabilityDistribution] = None):
		self.mu  = mu
		self.lam = lam
		self.a   = a
		self.b   = b
		self.parents = []
		self.name  = name
		self.prior = prior

	def __str__(self):
		return 'NormalGammaDistribution(%f, %f, %f, %f, name=%s, prior=%s)' % (self.mu, self.lam, self.a, self.b, self.name, str(self.prior))

	def get_parameters(self):
		return self.mu, self.lam, self.a, self.b

	def set_parameters(self, params):
		self.mu = params[0]
		self.lam = params[1]
		self.a = params[2]
		self.b = params[3]

	def cross_entropy(self, dist):
		if isinstance(dist, NormalGammaDistribution):
			a = self.a
			b = self.b
			m = self.mu
			l = self.lam

			aa = dist.a
			bb = dist.b
			mm = dist.mu
			ll = dist.lam

			c1 = np.log(bb)*aa + 0.5*np.log(ll) - gammaln(aa) - 0.5*np.log(2*np.pi)
			c2 = (aa - 0.5)*(digamma(a) - np.log(b)) - bb*(a/b)
			c3 = -0.5*ll*((1/l) + m*m*a/b - 2*mm*m*a/b + mm*mm*a/b)
			return -(c1 + c2 + c3)
		else:
			lf2 = lambda x, y: dist.log_density((x, y)) * self.density((x, y))
			lf1 = lambda x, y: dist.log_density((-x, y)) * self.density((-x, y))
			a1 = scipy.integrate.dblquad(lf1, 0, np.inf, lambda u: 0, lambda u: np.inf)
			a2 = scipy.integrate.dblquad(lf2, 0, np.inf, lambda u: 0, lambda u: np.inf)
			return -(a1[0] + a2[0])

	def entropy(self) -> float:
		a = self.a
		b = self.b
		lam = self.lam

		return -((a - 0.5)*(digamma(a) - np.log(b)) - a - 0.5 + np.log(b)*a + 0.5*np.log(lam) - gammaln(a) - 0.5*np.log(2*np.pi))

	def density(self, x: (float, float)) -> float:
		return np.exp(self.log_density(x))

	def log_density(self, x: (float, float)) -> float:
		a = self.a
		b = self.b
		mu = self.mu
		lam = self.lam

		c0 = np.log(b)*a + 0.5*np.log(lam/(2*np.pi)) - gammaln(a)
		c1 = np.log(x[1])*(a - 0.5) - b*x[1]
		c2 = -lam*x[1]*(x[0]-mu)*(x[0]-mu)/2
		return c0 + c1 + c2

	def sampler(self, seed: Optional[int] = None):
		return NormalGammaSampler(self, seed)


class NormalGammaSampler(object):

	def __init__(self, dist: NormalGammaDistribution, seed: Optional[int] = None):
		self.dist  = dist
		self.seed  = seed
		self.rng   = np.random.RandomState(seed)
		self.grng  = np.random.RandomState(self.rng.randint(0, 2**31 - 1))
		self.nrng  = np.random.RandomState(self.rng.randint(0, 2**31 - 1))

	def sample(self, size=None):
		if size is None:
			t = self.grng.gamma(self.dist.a, 1/self.dist.b)
			x = self.nrng.normal(self.dist.mu, np.sqrt(1/(self.dist.lam*t)))
			return x,t
		else:
			return [self.sample() for i in range(size)]

