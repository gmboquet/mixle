"""Example use for IntegerBernoulliSetDistribution for random graph generation."""
import numpy as np
import os

os.environ['NUMBA_DISABLE_JIT'] = '1'

from pysp.stats import *
from pysp.utils.estimation import optimize

if __name__ == '__main__':
    p_mat = np.zeros((4, 4))
    p_mat[0, :] = [0.0, 0.8, 0.1, 0.8]
    p_mat[1, :] = [0.8, 0.0, 0.8, 0.1]
    p_mat[2, :] = [0.1, 0.8, 0.0, 0.8]
    p_mat[3, :] = [0.8, 0.1, 0.8, 0.0]

    p_mat = p_mat.flatten()
    log_pvec = np.zeros_like(p_mat)
    log_pvec.fill(-np.inf)
    log_pvec[p_mat != 0.0] = np.log(p_mat[p_mat != 0.0])

    dist1 = IntegerBernoulliSetDistribution(log_pvec)

    p_mat = np.zeros((4, 4))
    p_mat[0, :] = [0.0, 0.1, 0.8, 0.1]
    p_mat[1, :] = [0.1, 0.0, 0.1, 0.8]
    p_mat[2, :] = [0.8, 0.1, 0.0, 0.1]
    p_mat[3, :] = [0.1, 0.8, 0.1, 0.0]

    p_mat = p_mat.flatten()
    log_pvec = np.zeros_like(p_mat)
    log_pvec.fill(-np.inf)
    log_pvec[p_mat != 0.0] = np.log(p_mat[p_mat != 0.0])

    dist2 = IntegerBernoulliSetDistribution(log_pvec)
    dist = MixtureDistribution([dist1, dist2], [0.5, 0.5])

    data = dist.sampler(1).sample(1000)

    est = MixtureEstimator([IntegerBernoulliSetEstimator(16)] * 2)

    model = optimize(data, est, max_its=100, rng=np.random.RandomState(1))

    print(list(np.exp(model.components[0].log_pvec)))
    print(list(np.exp(model.components[1].log_pvec)))
