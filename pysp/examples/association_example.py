import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'

import numpy as np
import time
from pysp.stats import *
from pysp.utils.estimation import best_of, partition_data, iterate, optimize


if __name__ == '__main__':

    rng = np.random.RandomState(2)
    init_prob_vec  = [0.2, 0.2, 0.3, 0.3]
    state_prob_mat = [[0.7, 0.1, 0.1, 0.1], [0.0, 0.3, 0.4, 0.3]]
    cond_weights   = [[0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5]]
    len_dist = CategoricalDistribution({3:1.0})
    init_dist = IntegerMultinomialDistribution(0, init_prob_vec, len_dist=len_dist)

    dist = IntegerHiddenAssociationDistribution(state_prob_mat, cond_weights, prev_dist=init_dist, len_dist=len_dist)

    data = dist.sampler(1).sample(500)

    len_est = CategoricalEstimator()
    prev_est = IntegerMultinomialEstimator(min_val=0, len_estimator=len_est)
    est = IntegerHiddenAssociationEstimator(4, 2, prev_estimator=prev_est, len_estimator=len_est, use_numba=False)

    model = optimize(data, est, max_its=1, rng=np.random.RandomState(1))
    t0 = time.time()
    model = optimize(data, est, max_its=500, print_iter=50, prev_estimate=model, init_p=1.0)
    print(time.time() - t0)
    print(str(model))