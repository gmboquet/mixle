from typing import Optional, Union, List, Dict, Any

from pysp.bstats.pdist import ProbabilityDistribution
from pysp.utils.special import gammaln, digamma
import numpy as np



class DictDirichletDistribution(ProbabilityDistribution):

    def __init__(self, alpha: Union[Dict[Any, float], float]):
        self.set_parameters(alpha)

    def __str__(self):
        return 'DictDirichletDistribution(%s)'%(str(self.alpha))

    def get_parameters(self) -> Union[Dict, float]:
        return self.alpha

    def set_parameters(self, params: Union[Dict[Any, float], float]) -> None:
        self.alpha = params
        self.is_unbounded = isinstance(params, float)

    def density(self, x: Dict[Any, float]) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: Dict[Any, float]) -> float:
        if self.is_unbounded:
            a = self.alpha
            n = len(x)
            c = (gammaln(a)*n - gammaln(a*n))
            if a == 1:
                return c
            else:
                return np.sum(np.log(list(x.values())))*(a-1) - c
        else:
            rv = 0.0
            asum = 0.0
            for k,v in x.items():
                a   = self.alpha[k]
                rv += gammaln(a) + np.log(v)*(a-1)
                asum += a
            return rv - gammaln(asum)

    def cross_entropy(self, dist):
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
        a = np.asarray(list(self.alpha.values()))
        a0 = np.sum(a)
        return -((gammaln(a0) - np.sum(gammaln(a))) + np.dot(digamma(a) - digamma(a0), a - 1))



    def sampler(self, seed: Optional[int] = None):
        return DictDirichletSampler(self, seed)


class DictDirichletSampler(object):

    def __init__(self, dist: DictDirichletDistribution, seed: Optional[int] = None):
        self.dist = dist
        self.dir  = dist.dist.sampler(seed)

    #def sample(self, size: Optional[int] = None) -> Union[Dict, List[Dict]]:
    #	if size is None:
    #		return dict(zip(self.dist.alpha_levels, self.dir.sample()))
    #	else:
    #		return [self.sample() for u in range(size)]

