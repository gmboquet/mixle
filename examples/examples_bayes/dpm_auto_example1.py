"""Infer a Bayesian DPM mixture automatically for simple Gaussian-vector data."""

import numpy as np

from pysp.stats import DiagonalGaussianDistribution, MixtureDistribution
from pysp.utils.automatic import get_dpm_mixture

if __name__ == "__main__":
    d1 = DiagonalGaussianDistribution([-1, -1, -1], [5, 5, 5])
    d2 = DiagonalGaussianDistribution([0, 0, 0], [0.1, 0.1, 0.1])
    d3 = DiagonalGaussianDistribution([2, 2, 2], [1, 1, 1])
    d4 = DiagonalGaussianDistribution([4, 4, 4], [1, 1, 1])
    dist1 = MixtureDistribution([d1, d2, d3, d4], [0.3, 0.3, 0.2, 0.2])

    data = dist1.sampler(seed=1).sample(400)
    model = get_dpm_mixture(data, rng=np.random.RandomState(1))

    print(str(model))
