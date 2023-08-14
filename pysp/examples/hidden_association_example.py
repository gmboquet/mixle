import os

os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
from pysp.stats import *
from pysp.utils.estimation import optimize

if __name__ == '__main__':
    aa = 0.90
    bb = (1.0 - aa) / 2

    dist1 = CategoricalDistribution({'a': aa, 'b': bb, 'c': bb})
    dist2 = CategoricalDistribution({'a': bb, 'b': aa, 'c': bb})
    dist3 = CategoricalDistribution({'a': bb, 'b': bb, 'c': aa})
    cond_dist = ConditionalDistribution({'a': dist1, 'b': dist2, 'c': dist3})
    given_dist = MultinomialDistribution(CategoricalDistribution({'a': 0.3, 'b': 0.2, 'c': 0.5}),
                                         len_dist=CategoricalDistribution({5: 1.0}))
    len_dist = CategoricalDistribution({7: 1.0})
    dist = HiddenAssociationDistribution(cond_dist=cond_dist, given_dist=given_dist, len_dist=len_dist)

    data = dist.sampler(1).sample(1000)

    len_est = CategoricalEstimator()
    given_est = MultinomialEstimator(CategoricalEstimator(), len_estimator=CategoricalEstimator())
    cond_est = ConditionalDistributionEstimator({v: CategoricalEstimator() for v in ['a', 'b', 'c']})
    est = HiddenAssociationEstimator(cond_est, given_estimator=given_est, len_estimator=len_est)

    model = optimize(data, est, max_its=1000, rng=np.random.RandomState(1))

    print(str(model))
